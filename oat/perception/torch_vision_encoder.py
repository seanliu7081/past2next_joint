import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as vision_models
import robomimic.models.base_nets as rmbn

from oat.common.pytorch_util import replace_submodules
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.crop_randomizer import CropRandomizer
from oat.model.common.normalizer import LinearNormalizer, _normalize


# Map backbone name -> (torchvision constructor, output channels after removing FC)
RESNET_REGISTRY = {
    'resnet18':  (vision_models.resnet18,  512),
    'resnet34':  (vision_models.resnet34,  512),
    'resnet50':  (vision_models.resnet50,  2048),
    'resnet101': (vision_models.resnet101, 2048),
    'resnet152': (vision_models.resnet152, 2048),
}


class ResNetConv(nn.Module):
    """
    Generic ResNet convolutional backbone that wraps any torchvision ResNet variant.
    Strips the final FC + avgpool layers, keeping only convolutional stages.
    """

    def __init__(
        self,
        backbone_name: str = 'resnet18',
        input_channel: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        if backbone_name not in RESNET_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from: {list(RESNET_REGISTRY.keys())}"
            )
        factory, self._out_channels = RESNET_REGISTRY[backbone_name]
        net = factory(pretrained=pretrained)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(
                input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove avgpool + fc, keep conv stages only
        self.nets = nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        return self.nets(x)

    def output_shape(self, input_shape):
        """input_shape: (C, H, W) without batch dim."""
        assert len(input_shape) == 3
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [self._out_channels, out_h, out_w]


class TorchVisionRgbEncoder(BaseObservationEncoder):
    """
    RGB encoder using torchvision ResNet backbones + SpatialSoftmax pooling.
    Drop-in replacement for RobomimicRgbEncoder with configurable backbone.

    Assumes rgb input: B, To, H, W, C
    """

    def __init__(
        self,
        shape_meta: dict,
        backbone_name: str = 'resnet18',
        pretrained: bool = False,
        feature_dimension: int = 64,
        num_kp: int = 32,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        use_group_norm: bool = True,
        eval_fixed_crop: bool = False,
        share_rgb_model: bool = False,
    ):
        super().__init__()

        rgb_ports = []
        port_shape = {}
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'rgb':
                rgb_ports.append(key)
                port_shape[key] = (shape[2], shape[0], shape[1])  # H,W,C -> C,H,W

        self.rgb_keys = rgb_ports
        self.backbone_name = backbone_name

        def make_crop_randomizer(shape, crop_shape):
            if crop_shape is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )

        def make_visual_net(shape, crop_shape):
            if crop_shape is not None:
                shape = (shape[0], crop_shape[0], crop_shape[1])

            backbone = ResNetConv(
                backbone_name=backbone_name,
                input_channel=shape[0],
                pretrained=pretrained,
            )
            backbone_out = backbone.output_shape(shape)

            pool = rmbn.SpatialSoftmax(
                input_shape=backbone_out,
                num_kp=num_kp,
                temperature=1.0,
                noise_std=0.0,
            )
            pool_out_dim = num_kp * 2

            fc = nn.Linear(pool_out_dim, feature_dimension)
            relu = nn.ReLU()

            net = nn.Sequential(backbone, pool, nn.Flatten(1, -1), fc, relu)
            return net

        # Build per-camera networks (or shared)
        self.nets = nn.ModuleDict()
        self.randomizers = nn.ModuleDict()

        if share_rgb_model:
            ref_port = rgb_ports[0]
            ref_shape = port_shape[ref_port]
            shared_net = make_visual_net(ref_shape, crop_shape)
            rand = make_crop_randomizer(ref_shape, crop_shape)
            for port in rgb_ports:
                assert port_shape[port] == ref_shape
                self.nets[port] = shared_net  # same nn.Module instance
                if rand is not None:
                    self.randomizers[port] = make_crop_randomizer(
                        port_shape[port], crop_shape
                    )
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                self.nets[port] = make_visual_net(shape, crop_shape)
                rand = make_crop_randomizer(shape, crop_shape)
                if rand is not None:
                    self.randomizers[port] = rand

        self.feature_dimension = feature_dimension

        if use_group_norm:
            replace_submodules(
                root_module=self,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                ),
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=self,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        self.normalizer = LinearNormalizer()

    def forward(self, obs_dict) -> torch.Tensor:
        nobs = self._normalize_obs_dict(obs_dict)

        sample = next(iter(nobs.values()))
        B, To, H, W, C = sample.shape

        feats = []
        for port in self.rgb_keys:
            x = nobs[port].reshape(B * To, H, W, C).permute(0, 3, 1, 2)  # [B*To, C, H, W]
            has_rand = port in self.randomizers
            if has_rand:
                x = self.randomizers[port].forward_in(x)
            feat = self.nets[port](x)  # [B*To, feature_dimension]
            if has_rand:
                feat = self.randomizers[port].forward_out(feat)
            feats.append(feat)

        image_feats = torch.cat(feats, dim=-1)  # [B*To, feature_dimension * N_cameras]
        image_feats = image_feats.reshape(B, To, -1)
        return image_feats

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        return self.feature_dimension * len(self.rgb_keys)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _normalize_obs_dict(self, obs_dict: Dict) -> Dict:
        nobs = {}
        for port, value in obs_dict.items():
            if port in self.rgb_keys:
                params = self.normalizer.params_dict.get(port, None)
                if params is None:
                    nobs[port] = value
                    print(f"no normalizer params for port {port}, skipping normalization.")
                else:
                    nobs[port] = _normalize(value, params, forward=True)
        return nobs

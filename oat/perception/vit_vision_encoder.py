from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
import robomimic.models.base_nets as rmbn

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.crop_randomizer import CropRandomizer
from oat.common.pytorch_util import replace_submodules
from oat.model.common.normalizer import LinearNormalizer, _normalize


class ViTRgbEncoder(BaseObservationEncoder):
    """
    RGB encoder using a torchvision Vision Transformer backbone.
    Drop-in replacement for TorchVisionRgbEncoder.

    Assumes rgb input: B, To, H, W, C
    """

    def __init__(
        self,
        shape_meta: dict,
        # ViT architecture
        patch_size: int = 4,
        num_layers: int = 8,
        num_heads: int = 8,
        hidden_dim: int = 256,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        # output
        feature_dimension: int = 64,
        # crop / augmentation
        crop_shape: Union[Tuple[int, int], None] = None,
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

            image_size = shape[1]  # assume square after crop
            assert shape[1] == shape[2], (
                f"ViT requires square images, got {shape[1]}x{shape[2]}"
            )
            assert image_size % patch_size == 0, (
                f"image_size {image_size} not divisible by patch_size {patch_size}"
            )

            vit = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                num_classes=feature_dimension,
            )
            net = nn.Sequential(vit, nn.ReLU())
            return net

        # Build per-camera networks (or shared)
        self.nets = nn.ModuleDict()
        self.randomizers = nn.ModuleDict()

        if share_rgb_model:
            ref_port = rgb_ports[0]
            ref_shape = port_shape[ref_port]
            shared_net = make_visual_net(ref_shape, crop_shape)
            for port in rgb_ports:
                assert port_shape[port] == ref_shape
                self.nets[port] = shared_net
                rand = make_crop_randomizer(port_shape[port], crop_shape)
                if rand is not None:
                    self.randomizers[port] = rand
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                self.nets[port] = make_visual_net(shape, crop_shape)
                rand = make_crop_randomizer(shape, crop_shape)
                if rand is not None:
                    self.randomizers[port] = rand

        self.feature_dimension = feature_dimension

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

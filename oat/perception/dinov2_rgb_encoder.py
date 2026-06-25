from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import robomimic.models.base_nets as rmbn

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.common.normalizer import LinearNormalizer, _normalize


class DINOv2RgbEncoder(BaseObservationEncoder):
    """
    Frozen DINOv2 ViT-S/14 visual encoder with SpatialSoftmax pooling.
    Produces the same output shape as RobomimicRgbEncoder (num_kp * 2 per camera).

    NOTE: torch.hub.load requires internet on first run to download weights.
    Pre-download with: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str = 'dinov2_vits14',
        freeze: bool = True,
        num_kp: int = 32,
        input_size: int = 224,
    ):
        super().__init__()

        # Parse rgb ports from shape_meta
        rgb_ports = []
        for key, attr in shape_meta['obs'].items():
            if attr.get('type', '') == 'rgb':
                rgb_ports.append(key)
        self.rgb_ports = rgb_ports

        # Load DINOv2 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.freeze = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.feat_dim = self.backbone.embed_dim        # 384 for ViT-S
        self.patch_size = self.backbone.patch_size      # 14
        self.input_size = input_size
        self.grid_size = input_size // self.patch_size  # 16 for 224/14
        self.num_kp = num_kp

        # SpatialSoftmax from robomimic (same as used in RobomimicRgbEncoder)
        self.spatial_softmax = rmbn.SpatialSoftmax(
            input_shape=[self.feat_dim, self.grid_size, self.grid_size],
            num_kp=num_kp,
            temperature=1.0,
            noise_std=0.0,
        )

        self.normalizer = LinearNormalizer()

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        feats = []
        for port in self.rgb_ports:
            x = obs_dict[port]  # [B, To, H, W, C]

            # Normalize
            params = self.normalizer.params_dict.get(port, None)
            if params is not None:
                x = _normalize(x, params, forward=True)

            B, To, H, W, C = x.shape
            x = x.reshape(B * To, H, W, C).permute(0, 3, 1, 2).float()  # [B*To, C, H, W]
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)

            # Extract patch tokens from frozen backbone
            with torch.set_grad_enabled(not self.freeze):
                out = self.backbone.forward_features(x)
                patch_tokens = out['x_norm_patchtokens']  # [B*To, N_patches, feat_dim]

            # Reshape to spatial grid and apply SpatialSoftmax
            spatial = patch_tokens.permute(0, 2, 1).reshape(
                -1, self.feat_dim, self.grid_size, self.grid_size
            )
            feat = self.spatial_softmax(spatial)  # [B*To, num_kp * 2]
            feat = feat.reshape(B, To, -1)        # [B, To, num_kp * 2]
            feats.append(feat)

        return torch.cat(feats, dim=-1)  # [B, To, num_kp * 2 * N_cameras]

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        return self.num_kp * 2 * len(self.rgb_ports)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def modalities(self) -> List[str]:
        return ['rgb']

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # always keep backbone in eval mode when frozen
        return self

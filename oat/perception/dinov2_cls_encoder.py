from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.common.normalizer import LinearNormalizer, _normalize


class DINOv2ClsEncoder(BaseObservationEncoder):
    """
    Frozen DINOv2 ViT-S/14 visual encoder using CLS token + MLP projection.

    NOTE: torch.hub.load requires internet on first run to download weights.
    Pre-download with: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    """

    def __init__(
        self,
        shape_meta: dict,
        model_name: str = 'dinov2_vits14',
        freeze: bool = True,
        proj_dim: int = 64,
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

        self.feat_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.input_size = input_size
        self.proj_dim = proj_dim

        # MLP projection: CLS token (384) → proj_dim (64) per camera
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, proj_dim),
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

            # Extract CLS token from frozen backbone
            with torch.set_grad_enabled(not self.freeze):
                out = self.backbone.forward_features(x)
                cls_token = out['x_norm_clstoken']  # [B*To, feat_dim]

            # Project CLS token
            feat = self.proj(cls_token)       # [B*To, proj_dim]
            feat = feat.reshape(B, To, -1)    # [B, To, proj_dim]
            feats.append(feat)

        return torch.cat(feats, dim=-1)  # [B, To, proj_dim * N_cameras]

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        return self.proj_dim * len(self.rgb_ports)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def modalities(self) -> List[str]:
        return ['rgb']

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # always keep backbone in eval mode when frozen
        return self

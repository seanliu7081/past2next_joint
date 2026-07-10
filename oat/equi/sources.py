"""EquiNoise source modules for the flow-matching policy (Workstream B).

Level 0 (implemented): ``BlockIsotropicSource`` -- a ``torch.randn`` drop-in
whose per-dim std is tied within each :class:`~oat.equi.blocks.BlockSpec`
block. Under group-compatible normalization (tied scale, zero offset within
every rotating block) plain per-block randn is exactly P1-correct; per-block
scales remain meaningful hyperparameters (different physical quantities).

``warp_correction='physical_so2'`` is the source-side fix of the per-dim
min-max covariance warp, kept as an ablation arm. It is computed from the
normalizer's SCALE vector (not raw ranges):

    std_i = block_scale * scale_i / geomean(scale over the block dims)

Under per-dim min-max (scale_i = 2 / R_i) this equals the diffusion-side
``EquiNoise`` correction geo(R)/R_i exactly; under group-compatible norm the
scales within a rho1 block are tied, so the correction degrades to identity
BY CONSTRUCTION (asserted at runtime by the policy).

Levels 1-3 (learned sources): scaffold only -- ``Level1ScaleHeadSource`` is a
stub proving the ``SourceModule`` slot works. No training wiring.
"""

from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from oat.equi.blocks import RHO1, BlockSpec, assert_blocks_cover, libero_action_blocks

WARP_NONE = "none"
WARP_PHYSICAL_SO2 = "physical_so2"
_WARPS = (WARP_NONE, WARP_PHYSICAL_SO2)


@runtime_checkable
class SourceModule(Protocol):
    """Drop-in for ``torch.randn(shape, device=..., dtype=...)``. Levels >= 1
    may additionally condition on (detached) observation features."""

    def sample(self, shape, device=None, dtype=None, cond_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        ...


class GaussianSource(nn.Module):
    """Exact ``torch.randn``: the bit-for-bit baseline source."""

    def sample(self, shape, device=None, dtype=None, cond_feat=None) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype)

    forward = sample


class BlockIsotropicSource(nn.Module):
    """Level-0 block-isotropic source: ``randn * std`` with std tied per block.

    Args:
        action_dim: last dim of sampled tensors (7 for LIBERO).
        blocks: the BlockSpec decomposition (single source of truth).
        scales: optional per-block base scales {block_name: float}; default 1.0.
        warp_correction: 'none' | 'physical_so2' (see module docstring).
        normalizer_scale: per-dim scale vector of the fitted action normalizer;
            required for 'physical_so2'.
    """

    def __init__(
        self,
        action_dim: int,
        blocks: Sequence[BlockSpec],
        scales: Optional[Dict[str, float]] = None,
        warp_correction: str = WARP_NONE,
        normalizer_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert warp_correction in _WARPS, warp_correction
        assert_blocks_cover(blocks, action_dim)
        self.action_dim = int(action_dim)
        self.blocks = list(blocks)
        self.warp_correction = warp_correction
        s = dict(scales or {})
        block_names = {b.name for b in self.blocks}
        unknown = sorted(set(s) - block_names)
        if unknown:
            # e.g. world_frame_rotation=True renames 'rot' -> 'rot_xy'/'rot_z';
            # a stale scales dict must fail loudly, not silently drop entries.
            raise ValueError(
                f"source scales {unknown} do not match any block in "
                f"{sorted(block_names)} for this block layout"
            )

        std = torch.ones(self.action_dim)
        correction = torch.ones(self.action_dim)
        for b in self.blocks:
            idx = torch.as_tensor(list(b.idx), dtype=torch.long)
            std[idx] = float(s.get(b.name, 1.0))
            if warp_correction == WARP_PHYSICAL_SO2 and b.rep == RHO1:
                assert normalizer_scale is not None, (
                    "physical_so2 needs the fitted action normalizer's scale vector"
                )
                sc = normalizer_scale.detach().float().cpu().flatten()[idx].abs().clamp_min(1e-12)
                geo = sc.log().mean().exp()
                correction[idx] = sc / geo
        # non-persistent: rebuilt from the normalizer, never checkpointed
        # (mirrors equi_noise.py; EMA/state_dict ignore these buffers)
        self.register_buffer("std", std * correction, persistent=False)
        self.register_buffer("std_correction", correction, persistent=False)

    @torch.no_grad()
    def sample(self, shape, device=None, dtype=None, cond_feat=None) -> torch.Tensor:
        assert shape[-1] == self.action_dim, (
            f"last dim {shape[-1]} != action_dim {self.action_dim}"
        )
        z = torch.randn(shape, device=device, dtype=dtype)
        return z * self.std.to(device=z.device, dtype=z.dtype)

    forward = sample

    def extra_repr(self) -> str:
        return (
            f"action_dim={self.action_dim}, warp={self.warp_correction}, "
            f"std={self.std.tolist()}"
        )


class Level1ScaleHeadSource(nn.Module):
    """SCAFFOLD (Level 1, not trained anywhere yet): per-block scales predicted
    from detached observation features, ``g_phi: sg[f(o)] -> R_{>0}^{n_blocks}``.

    The head is zero-initialized so it returns exactly the Level-0
    identity-scale source until trained. ``cond_feat`` is detached at the
    interface (attributability); expose ``detach=False`` only for the later
    ablation. Do NOT wire this into training in Phase 1.
    """

    def __init__(
        self,
        action_dim: int,
        blocks: Sequence[BlockSpec],
        cond_dim: int,
        hidden: int = 64,
        detach: bool = True,
    ):
        super().__init__()
        assert_blocks_cover(blocks, action_dim)
        self.action_dim = int(action_dim)
        self.blocks = list(blocks)
        self.detach = detach
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.blocks)),
        )
        # init == identity: log-scales start at exactly 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        idx_of_block = torch.zeros(self.action_dim, dtype=torch.long)
        for bi, b in enumerate(self.blocks):
            idx_of_block[torch.as_tensor(list(b.idx), dtype=torch.long)] = bi
        self.register_buffer("idx_of_block", idx_of_block, persistent=False)

    def sample(self, shape, device=None, dtype=None, cond_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert shape[-1] == self.action_dim
        z = torch.randn(shape, device=device, dtype=dtype)
        if cond_feat is None:
            return z
        if self.detach:
            cond_feat = cond_feat.detach()
        if cond_feat.dim() > 2:  # (B, To, d) -> (B, d)
            cond_feat = cond_feat.mean(dim=tuple(range(1, cond_feat.dim() - 1)))
        log_s = self.head(cond_feat.to(dtype=z.dtype))          # (B, n_blocks)
        per_dim = log_s[:, self.idx_of_block].exp()             # (B, action_dim)
        return z * per_dim.view(per_dim.shape[0], *([1] * (z.dim() - 2)), self.action_dim)

    forward = sample


def build_source(
    cfg: Optional[dict],
    action_dim: int,
    normalizer_scale: Optional[torch.Tensor] = None,
) -> Optional[nn.Module]:
    """Build the source from a plain-dict config. Returns None when disabled
    (callers fall back to ``torch.randn`` -- the default-off contract).

    cfg keys: enable(bool), kind('block_isotropic'|'gaussian'),
    warp_correction('none'|'physical_so2'), scales({name: float}),
    world_frame_rotation(bool).
    """
    cfg = dict(cfg or {})
    if not cfg.get("enable", False):
        return None
    kind = cfg.get("kind", "block_isotropic")
    if kind == "gaussian":
        return GaussianSource()
    if kind == "block_isotropic":
        blocks = libero_action_blocks(bool(cfg.get("world_frame_rotation", False)))
        return BlockIsotropicSource(
            action_dim=action_dim,
            blocks=blocks,
            scales=cfg.get("scales", None),
            warp_correction=cfg.get("warp_correction", WARP_NONE),
            normalizer_scale=normalizer_scale,
        )
    raise ValueError(f"unknown source kind '{kind}'")

"""Structured (block-wise isotropic) source distribution for diffusion policies.

Drop-in replacement for ``torch.randn(trajectory.shape)`` whose covariance is
block-diagonal: tied (isotropic) WITHIN each representation block, free ACROSS
blocks. This makes the diffusion SOURCE symmetry-compatible with the action
representation, per the irrep-noise principle:

    * within an SO(2) vector block the per-coordinate scale MUST be tied;
    * different blocks (translation / vertical / rotation / gripper) MAY carry
      different scales, because they are physically distinct irreps with
      different units.

Usage in a DDPM/DDIM policy (epsilon prediction)
------------------------------------------------
The source is used at BOTH ends of the diffusion process, and they MUST match:
  * training  -- the injected noise ``eps`` in ``add_noise`` (the regression
    target of the epsilon head), and
  * inference -- the terminal prior ``x_T`` the reverse process is seeded from.
Because ``sqrt(alpha_bar_T) ~ 0``, ``x_T ~ eps`` in distribution, so seeding the
sampler from the same block-isotropic distribution keeps the reverse dynamics
consistent with the forward process. When every block scale == 1 the source is
exactly ``torch.randn`` and the policy is bit-for-bit a standard DDPM/DDIM.

Two regimes
-----------
"normalized"
    Scales are applied directly in the (already per-dim-normalized) action space.
    Recovers ``torch.randn`` exactly when every scale == 1. Use this only to give
    translation / rotation / gripper *different* source magnitudes; it does NOT
    restore any geometric symmetry, because per-dim min_max normalization has
    already warped the planar block.

"physical_so2"
    Normalization-aware. Given the per-dim min_max ranges ``R = max - min`` used
    by the ``LinearNormalizer`` (mode="limits"), each SO(2) vector block is
    reweighted so the source is ISOTROPIC IN PHYSICAL UNITS rather than in
    normalized units. For a planar block ``[i, j]`` with ranges ``(R_i, R_j)`` the
    normalized-space stds are

        std_i = sqrt(R_j / R_i),   std_j = sqrt(R_i / R_j)      (geometric mean 1)

    which EXACTLY cancels the anisotropic warp introduced by independent per-dim
    min_max (``x_norm = 2 (x - min)/(max - min) - 1``). Reduces to isotropic
    [1, 1] when R_i == R_j, so it strictly generalizes ``torch.randn``.

The group acts identically on every timestep of an action chunk, so the block
structure is applied per action-vector and broadcast across the horizon H. No
cross-time covariance is introduced (that would be an orthogonal smoothness prior,
not a symmetry constraint).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn


@dataclass
class NoiseBlock:
    name: str
    idx: Sequence[int]                         # action-vector indices this block owns
    kind: str = "scalar"                       # "scalar" | "so2_vec" | "free"
    scale: float = 1.0                         # base scale for the block
    ranges: Optional[Sequence[float]] = None   # per-idx (max - min); for physical_so2 so2_vec


class EquiNoise(nn.Module):
    """Block-isotropic diffusion source. ``sample(shape)`` mirrors ``torch.randn``."""

    def __init__(self, action_dim: int, blocks: List[NoiseBlock], mode: str = "normalized"):
        super().__init__()
        assert mode in ("normalized", "physical_so2"), mode
        self.action_dim = int(action_dim)
        self.mode = mode
        self.blocks = blocks

        # Precompute a single per-dim std vector; sampling is then randn * std.
        std = torch.ones(self.action_dim)
        covered = torch.zeros(self.action_dim, dtype=torch.bool)

        for b in blocks:
            idx = torch.as_tensor(list(b.idx), dtype=torch.long)
            assert idx.numel() > 0, f"block '{b.name}' is empty"
            assert not covered[idx].any(), f"block '{b.name}' overlaps an earlier block"
            covered[idx] = True

            if self.mode == "physical_so2" and b.kind == "so2_vec":
                assert b.ranges is not None and len(b.ranges) == len(b.idx), (
                    f"so2_vec block '{b.name}' needs per-idx ranges in physical_so2 mode"
                )
                R = torch.as_tensor(list(b.ranges), dtype=torch.float32).abs().clamp_min(1e-8)
                geo = R.prod().pow(1.0 / R.numel())      # geometric mean of ranges
                block_std = (geo / R) * float(b.scale)   # cancels per-dim min_max warp
            else:
                # scalar / free / normalized-so2 -> tied isotropic scale
                block_std = torch.full((idx.numel(),), float(b.scale))

            std[idx] = block_std

        missing = (~covered).nonzero().flatten().tolist()
        assert not missing, f"blocks do not cover all {self.action_dim} dims; missing {missing}"
        self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def sample(self, shape, device=None, dtype=None) -> torch.Tensor:
        """shape = (..., action_dim). Returns source ~ N(0, diag(std^2))."""
        assert shape[-1] == self.action_dim, f"last dim {shape[-1]} != action_dim {self.action_dim}"
        z = torch.randn(shape, device=device, dtype=dtype)
        return z * self.std.to(device=z.device, dtype=z.dtype)

    # behave like torch.randn(trajectory.shape) when called positionally
    def forward(self, shape, device=None, dtype=None) -> torch.Tensor:
        return self.sample(shape, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return f"action_dim={self.action_dim}, mode={self.mode}, std={self.std.tolist()}"


def libero_7dof_blocks(
    mode: str = "normalized",
    ranges: Optional[Sequence[float]] = None,
    scales: Optional[dict] = None,
    world_frame_rotation: bool = False,
) -> List[NoiseBlock]:
    """Block spec for the 7-DoF LIBERO action.

    Layout (matches the LIBERO action space):  [x, y, z, r_x, r_y, r_z (axis-angle delta), gripper]

    Args:
        mode:    "normalized" or "physical_so2".
        ranges:  length-7 sequence of per-dim (max - min) from the normalizer's
                 min_max statistics. Required for mode="physical_so2".
        scales:  optional dict block_name -> base scale (defaults to 1.0).
        world_frame_rotation:
                 If True, split the axis-angle rotation into an SO(2) planar block
                 [r_x, r_y] + a yaw scalar r_z. Correct ONLY if rotation_delta is
                 expressed in the WORLD / base frame. If False (safe default), the
                 rotation is a single 3-dim block with one tied scale and NO SO(2)
                 structure imposed.
    """
    s = {"xy": 1.0, "z": 1.0, "rot": 1.0, "rot_xy": 1.0, "rot_z": 1.0, "grip": 1.0}
    if scales:
        s.update(scales)

    def rng(i):
        if ranges is None:
            return None
        return [float(ranges[k]) for k in i]

    blocks = [
        NoiseBlock("xy", [0, 1], kind="so2_vec", scale=s["xy"], ranges=rng([0, 1])),
        NoiseBlock("z", [2], kind="scalar", scale=s["z"]),
    ]
    if world_frame_rotation:
        blocks += [
            NoiseBlock("rot_xy", [3, 4], kind="so2_vec", scale=s["rot_xy"], ranges=rng([3, 4])),
            NoiseBlock("rot_z", [5], kind="scalar", scale=s["rot_z"]),
        ]
    else:
        blocks += [NoiseBlock("rot", [3, 4, 5], kind="free", scale=s["rot"])]
    blocks += [NoiseBlock("grip", [6], kind="scalar", scale=s["grip"])]
    return blocks


if __name__ == "__main__":
    # Smoke test + numerical verification of physical isotropy.
    torch.manual_seed(0)

    R = [0.06, 0.02, 0.04, 0.5, 0.5, 0.5, 2.0]  # (max - min) per dim, asymmetric x vs y
    blocks = libero_7dof_blocks(mode="physical_so2", ranges=R, world_frame_rotation=False)
    noise = EquiNoise(action_dim=7, blocks=blocks, mode="physical_so2")
    print(noise)

    a_norm = noise.sample((4096, 8, 7))
    print("sample shape:", tuple(a_norm.shape))

    # Map planar block back to physical units: a_phys = (R / 2) * a_norm.
    Rx, Ry = R[0], R[1]
    xphys = a_norm[..., 0].flatten() * (Rx / 2.0)
    yphys = a_norm[..., 1].flatten() * (Ry / 2.0)
    cov = torch.cov(torch.stack([xphys, yphys]))
    ratio = (cov[0, 0] / cov[1, 1]).item()
    print(f"var_x / var_y in PHYSICAL units = {ratio:.3f}  (must be ~1.0)")
    assert 0.8 < ratio < 1.25, "physical isotropy check FAILED"
    print("OK: physical_so2 source is isotropic in physical units.")

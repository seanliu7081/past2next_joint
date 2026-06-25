import torch
import torch.nn as nn


class PolarDecompose(nn.Module):
    """Deterministic decomposition of 7D actions into invariant and equivariant subspaces.

    No learnable parameters. Splits actions under SO(2) symmetry:
      - Invariant:   (r_trans, dz, r_rot, grip)   — unchanged under planar rotation
      - Equivariant: (theta_trans, theta_rot, dyaw) — shift under planar rotation

    Input:  (*, 7) actions = (dx, dy, dz, droll, dpitch, dyaw, grip)
    Output:
        invariant:   (*, 4)  = (r_trans, dz, r_rot, grip)
        equivariant: (*, 3)  = (theta_trans, theta_rot, dyaw)
        null_mask:   (*, 2)  = bool mask for (r_trans==0, r_rot==0)
    """

    def forward(self, actions: torch.Tensor):
        dx = actions[..., 0]
        dy = actions[..., 1]
        dz = actions[..., 2]
        droll = actions[..., 3]
        dpitch = actions[..., 4]
        dyaw = actions[..., 5]
        grip = actions[..., 6]

        # Polar decomposition
        r_trans = torch.sqrt(dx ** 2 + dy ** 2)
        theta_trans = torch.atan2(dy, dx)

        r_rot = torch.sqrt(droll ** 2 + dpitch ** 2)
        theta_rot = torch.atan2(dpitch, droll)

        # Null mask: exact zero check (data confirms near-zero samples are hard zeros)
        null_trans = r_trans == 0.0
        null_rot = r_rot == 0.0

        invariant = torch.stack([r_trans, dz, r_rot, grip], dim=-1)
        equivariant = torch.stack([theta_trans, theta_rot, dyaw], dim=-1)
        null_mask = torch.stack([null_trans, null_rot], dim=-1)

        return invariant, equivariant, null_mask

    def inverse(self, invariant: torch.Tensor, equivariant: torch.Tensor) -> torch.Tensor:
        """Reconstruct 7D Cartesian actions from polar representation.

        Used only for testing round-trip consistency. The decoder does NOT use this
        — it outputs Cartesian directly as an intentional relaxation.
        """
        r_trans = invariant[..., 0]
        dz = invariant[..., 1]
        r_rot = invariant[..., 2]
        grip = invariant[..., 3]

        theta_trans = equivariant[..., 0]
        theta_rot = equivariant[..., 1]
        dyaw = equivariant[..., 2]

        dx = r_trans * torch.cos(theta_trans)
        dy = r_trans * torch.sin(theta_trans)
        droll = r_rot * torch.cos(theta_rot)
        dpitch = r_rot * torch.sin(theta_rot)

        return torch.stack([dx, dy, dz, droll, dpitch, dyaw, grip], dim=-1)

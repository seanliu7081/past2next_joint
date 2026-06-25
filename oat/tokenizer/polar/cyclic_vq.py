import torch
import torch.nn as nn
import math
from typing import List, Tuple


class CyclicVQ(nn.Module):
    """Cyclic (circular topology) vector quantization for angular components.

    No learnable parameters — fixed uniform bins on [-pi, pi).
    Provides exact C_N equivariance: rotating input by k * (2*pi/N) shifts
    the output bin index by k (mod N).

    Each angular dimension is quantized independently with its own bin count.
    An extra NULL index (= n_bins) is used when the corresponding radius is zero,
    since theta is ill-defined at r=0.

    Args:
        n_bins: List of bin counts per angular dimension.
               e.g. [24, 12, 16] for (theta_trans, theta_rot, dyaw)
    """

    def __init__(self, n_bins: List[int]):
        super().__init__()
        self.n_bins = n_bins
        self.n_angles = len(n_bins)

        # Register bin centers as buffers (not parameters)
        for i, n in enumerate(n_bins):
            # Uniform bins on [-pi, pi): centers at -pi + (2*pi/n) * (j + 0.5)
            centers = -math.pi + (2 * math.pi / n) * (torch.arange(n, dtype=torch.float32) + 0.5)
            self.register_buffer(f"centers_{i}", centers, persistent=True)

    def _get_centers(self, i: int) -> torch.Tensor:
        return getattr(self, f"centers_{i}")

    @staticmethod
    def geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on S^1: min(|a-b|, 2*pi - |a-b|)."""
        diff = torch.abs(a - b)
        return torch.min(diff, 2 * math.pi - diff)

    def quantize_single(self, angles: torch.Tensor, dim_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a single angular dimension.

        Args:
            angles: (*, ) angle values in [-pi, pi]
            dim_idx: which angular dimension (index into n_bins)

        Returns:
            quantized: (*, ) bin center values (with STE gradient)
            indices:   (*, ) bin indices, dtype long
        """
        centers = self._get_centers(dim_idx)  # (N,)
        n = self.n_bins[dim_idx]

        # Compute geodesic distance to each bin center: (*, N)
        angles_expanded = angles.unsqueeze(-1)  # (*, 1)
        centers_expanded = centers.expand(*([1] * angles.dim()), n)  # broadcast-compatible
        dists = self.geodesic_distance(angles_expanded, centers_expanded)  # (*, N)

        # Nearest bin
        indices = dists.argmin(dim=-1)  # (*, )

        # Quantized value = bin center
        quantized = centers[indices]  # (*, )

        # Straight-through estimator: gradient flows through as if quantization didn't happen
        quantized = angles + (quantized - angles).detach()

        return quantized, indices

    def forward(
        self,
        angles: torch.Tensor,
        null_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize angular components with optional null masking.

        Args:
            angles: (*, 3) = (theta_trans, theta_rot, dyaw), values in [-pi, pi]
            null_mask: (*, 2) = bool mask for (null_trans, null_rot).
                       When True, the corresponding angle index becomes the NULL index (= n_bins).
                       dyaw (dim 2) is never null.

        Returns:
            quantized_angles: (*, 3) — bin center values (with STE gradient).
                              For null positions, value is 0.0.
            indices: (*, 3) — per-angle bin indices, dtype long.
                     For null positions: index = n_bins[i] (the extra NULL index).
        """
        assert angles.shape[-1] == self.n_angles, \
            f"Expected {self.n_angles} angles, got {angles.shape[-1]}"

        batch_shape = angles.shape[:-1]
        quantized_list = []
        indices_list = []

        for i in range(self.n_angles):
            q, idx = self.quantize_single(angles[..., i], i)
            quantized_list.append(q)
            indices_list.append(idx)

        quantized = torch.stack(quantized_list, dim=-1)  # (*, 3)
        indices = torch.stack(indices_list, dim=-1)       # (*, 3)

        # Apply null masking for theta_trans (dim 0) and theta_rot (dim 1)
        if null_mask is not None:
            assert null_mask.shape[-1] == 2
            # theta_trans null
            mask_trans = null_mask[..., 0]  # (*, )
            indices[..., 0] = torch.where(mask_trans, self.n_bins[0], indices[..., 0])
            quantized[..., 0] = torch.where(mask_trans, torch.zeros_like(quantized[..., 0]), quantized[..., 0])
            # theta_rot null
            mask_rot = null_mask[..., 1]  # (*, )
            indices[..., 1] = torch.where(mask_rot, self.n_bins[1], indices[..., 1])
            quantized[..., 1] = torch.where(mask_rot, torch.zeros_like(quantized[..., 1]), quantized[..., 1])

        return quantized, indices

    def indices_to_angles(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert bin indices back to angle values (bin centers).

        Args:
            indices: (*, 3) — per-angle bin indices. NULL indices (= n_bins[i])
                     map to 0.0.

        Returns:
            angles: (*, 3) — bin center values.
        """
        angles_list = []
        for i in range(self.n_angles):
            idx = indices[..., i]  # (*, )
            centers = self._get_centers(i)  # (N,)
            n = self.n_bins[i]

            # Clamp NULL indices to 0 for gathering, then overwrite with 0.0
            is_null = idx >= n
            safe_idx = torch.clamp(idx, 0, n - 1)
            angle = centers[safe_idx]
            angle = torch.where(is_null, torch.zeros_like(angle), angle)
            angles_list.append(angle)

        return torch.stack(angles_list, dim=-1)  # (*, 3)

    @property
    def vocab_sizes(self) -> List[int]:
        """Vocab size per angular dimension (including NULL token)."""
        # theta_trans and theta_rot get +1 for NULL; dyaw does not
        return [
            self.n_bins[0] + 1,  # theta_trans + NULL
            self.n_bins[1] + 1,  # theta_rot + NULL
            self.n_bins[2],      # dyaw (never null)
        ]

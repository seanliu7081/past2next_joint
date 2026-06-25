"""CyclicVQ with product index for PolarOATTok v2.

Per-dimension cyclic vector quantization that produces a single product index
per timestep for the AR policy. No NULL tokens — the RegisterEncoder compresses
temporal sequences, so per-timestep null handling is absorbed.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from torch.amp import autocast


class CyclicVQProduct(nn.Module):
    """Per-dimension cyclic VQ with product index output.

    Each dimension is independently quantized to N uniform bins on [-pi, pi).
    Distance metric: geodesic on S^1 = min(|a-b|, 2*pi-|a-b|).
    Fixed codebook (no learnable parameters).

    Produces a single product index per position:
        token = idx_0 * (N_1 * N_2) + idx_1 * N_2 + idx_2

    Args:
        n_bins: List of bin counts per dimension. e.g. [24, 12, 8]
    """

    def __init__(self, n_bins: List[int]):
        super().__init__()
        self.n_bins = n_bins
        self.n_dims = len(n_bins)
        self.codebook_size = 1
        for n in n_bins:
            self.codebook_size *= n

        # Fixed uniform bin centers on [-pi, pi) per dimension
        for i, n in enumerate(n_bins):
            centers = -math.pi + (2 * math.pi / n) * (torch.arange(n, dtype=torch.float32) + 0.5)
            self.register_buffer(f"centers_{i}", centers, persistent=True)

        # Basis for product index (big-endian: dim 0 is most significant)
        basis = []
        b = 1
        for n in reversed(n_bins):
            basis.append(b)
            b *= n
        basis.reverse()
        self.register_buffer("_basis", torch.tensor(basis, dtype=torch.long), persistent=True)

    def __repr__(self):
        return (
            f"CyclicVQProduct(\n"
            f"  n_bins={self.n_bins!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f")"
        )

    @staticmethod
    def geodesic_distance(a: Tensor, b: Tensor) -> Tensor:
        """Geodesic distance on S^1."""
        diff = torch.abs(a - b)
        return torch.min(diff, 2 * math.pi - diff)

    def _quantize_dim(self, values: Tensor, dim_idx: int) -> Tuple[Tensor, Tensor]:
        """Quantize one angular dimension."""
        centers = getattr(self, f"centers_{dim_idx}")
        dists = self.geodesic_distance(values.unsqueeze(-1), centers)
        indices = dists.argmin(dim=-1)
        quantized = centers[indices]
        # STE
        quantized = values + (quantized - values).detach()
        return quantized, indices

    @autocast(device_type="cuda", enabled=False)
    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize angular latents and produce product index.

        Args:
            latents: (..., n_dims) encoder output.

        Returns:
            quantized: (..., n_dims) with STE gradient.
            tokens: (...) LongTensor product index.
        """
        latents = latents.float()
        assert latents.shape[-1] == self.n_dims

        all_quantized = []
        all_indices = []
        for d in range(self.n_dims):
            q, idx = self._quantize_dim(latents[..., d], d)
            all_quantized.append(q)
            all_indices.append(idx)

        quantized = torch.stack(all_quantized, dim=-1)

        # Product index
        tokens = torch.zeros_like(all_indices[0], dtype=torch.long)
        for d in range(self.n_dims):
            tokens = tokens + all_indices[d] * self._basis[d]

        return quantized, tokens.long()

    def indices_to_embedding(self, tokens: Tensor) -> Tensor:
        """Inverse: product index -> per-dim bin centers.

        Args:
            tokens: (...) LongTensor product indices.

        Returns:
            embeddings: (..., n_dims) bin center values.
        """
        embeddings = []
        remaining = tokens.clone()
        for d in range(self.n_dims):
            dim_idx = remaining // self._basis[d]
            remaining = remaining % self._basis[d]
            centers = getattr(self, f"centers_{d}")
            dim_idx = dim_idx.clamp(0, len(centers) - 1)
            embeddings.append(centers[dim_idx])
        return torch.stack(embeddings, dim=-1)

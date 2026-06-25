"""
Spherical Finite Scalar Quantization (SphericalFSQ)
====================================================
Drop-in replacement for FSQ that quantizes in spherical coordinates:
  - Direction: nearest neighbor on a fixed uniform S^(D-1) point set
  - Magnitude: scalar FSQ-style quantization (tanh bound + round + STE)

Matches FSQ interface exactly:
  forward(latents) → (quantized, token_indices)
  indices_to_embedding(indices) → quantized
  .codebook_size (attribute, not property)
  .dim
  .implicit_codebook
  drop_quant_p, corrupt_tokens_p support

Usage in config:
  quantizer:
    _target_: oat.tokenizer.oat.quantizer.spherical_fsq.SphericalFSQ
    latent_dim: 4
    n_angular: 200
    n_radial: 5
    # codebook_size = 200 * 5 = 1000
"""

import random
from functools import partial
from einops import repeat
import torch
import torch.nn as nn
from torch import Tensor, int32
from torch.amp import autocast
from typing import List, Optional, Tuple

from oat.tokenizer.oat.util.packed_ops import packed_call


__all__ = ["SphericalFSQ"]


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with STE, randomly skip quantization per sample."""
    zhat = z.round()
    batch_size = z.shape[0]
    device = z.device
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    mask = mask.view(batch_size, *([1] * (z.ndim - 1)))
    output = z + ((1 - mask) * (zhat - z)).detach()
    return output


def generate_S_points(n_points: int, dim: int, seed: int = 42) -> Tensor:
    """
    Generate approximately uniform points on S^(dim-1).
    
    Gaussian sampling + normalization gives uniform distribution on
    unit hypersphere. Fixed seed ensures identical codebook across runs.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    points = torch.randn(n_points, dim, generator=gen)
    points = points / points.norm(dim=-1, keepdim=True)
    return points


def optimize_point_set(points: Tensor, n_iters: int = 500, lr: float = 0.01) -> Tensor:
    """
    Improve uniformity via electrostatic repulsion on S^(dim-1).
    Minimizes sum of 1/||p_i - p_j||. Run once at init time.
    """
    pts = points.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pts], lr=lr)
    
    for _ in range(n_iters):
        pts_normed = pts / pts.norm(dim=-1, keepdim=True)
        dists = torch.cdist(pts_normed, pts_normed)
        mask = ~torch.eye(len(pts), dtype=torch.bool, device=pts.device)
        energy = (1.0 / (dists[mask] + 1e-6)).sum()
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()
    
    with torch.no_grad():
        pts_final = pts / pts.norm(dim=-1, keepdim=True)
    return pts_final.detach()


class SphericalFSQ(nn.Module):
    """
    Spherical Finite Scalar Quantization.
    
    Decomposes latent into direction (S^(D-1)) + magnitude (scalar),
    quantizes each with fixed codebook + STE gradients.
    
    Args:
        latent_dim: Dimension of latent vectors.
        n_angular: Number of direction codebook points on S^(D-1).
        n_radial: Number of magnitude levels (including a small nonzero minimum).
        drop_quant_p: Probability of skipping quantization per sample (training only).
        corrupt_tokens_p: Probability of corrupting tokens (training only).
        min_corrupt_tokens_p: Minimum corruption rate when corruption is active.
        apply_corrupt_tokens_p: Probability of activating corruption per sample.
        optimize_points: Run electrostatic optimization at init.
        optimize_iters: Optimization steps for point set.
        seed: Random seed for reproducible codebook.
        packed_call: Use packed processing for list inputs.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        n_angular: int = 200,
        n_radial: int = 5,
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        optimize_points: bool = True,
        optimize_iters: int = 500,
        seed: int = 42,
        packed_call: bool = True,
    ):
        super().__init__()

        self.dim = latent_dim
        self.n_angular = n_angular
        self.n_radial = n_radial
        # Match FSQ: codebook_size is a plain attribute, not a property
        self.codebook_size = n_angular * n_radial

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p
        self.packed_call = packed_call

        # ── Fixed angular codebook on S^(D-1) ─────────────────────────
        angular_points = generate_S_points(n_angular, latent_dim, seed=seed)
        if optimize_points and n_angular > 10:
            print(f"SphericalFSQ: optimizing {n_angular} points on S^{latent_dim - 1}...")
            angular_points = optimize_point_set(
                angular_points, n_iters=optimize_iters
            )
        self.register_buffer("angular_codebook", angular_points, persistent=True)
        # shape: [n_angular, latent_dim]

        # ── Fixed radial levels ────────────────────────────────────────
        # Levels are [0, 1, 2, ..., n_radial-1].
        # After bounding+rounding, we normalize so that:
        #   level 0 → small nonzero magnitude (eps_mag) to preserve direction info
        #   level n_radial-1 → 1.0 (max magnitude)
        # This avoids the mag=0 issue where direction is lost.
        radial_levels = torch.arange(n_radial, dtype=torch.float32)
        self.register_buffer("radial_levels", radial_levels, persistent=True)
        self.eps_mag = 0.05  # minimum magnitude for level 0

        # ── Precompute full implicit codebook ──────────────────────────
        all_indices = torch.arange(self.codebook_size)
        implicit_codebook = self._indices_to_embedding_impl(all_indices)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        print(self)

    def __repr__(self):
        return (
            f"SphericalFSQ(\n"
            f"  latent_dim={self.dim},\n"
            f"  n_angular={self.n_angular} (points on S^{self.dim-1}),\n"
            f"  n_radial={self.n_radial},\n"
            f"  codebook_size={self.codebook_size},\n"
            f"  drop_quant_p={self.drop_quant_p},\n"
            f")"
        )

    # ── Magnitude bounding + quantization ──────────────────────────────

    def _normalize_mag_level(self, level: Tensor) -> Tensor:
        """
        Convert integer level [0, n_radial-1] to magnitude value.
        
        Maps: 0 → eps_mag, n_radial-1 → 1.0
        This ensures level 0 still has a small magnitude, preserving
        direction information (unlike mag=0 which kills the direction).
        """
        max_level = max(self.n_radial - 1, 1)
        # Linear interpolation from eps_mag to 1.0
        t = level / max_level  # [0, 1]
        return self.eps_mag + (1.0 - self.eps_mag) * t

    def bound_magnitude(self, mag: Tensor, eps: float = 1e-3) -> Tensor:
        """
        Bound raw magnitude to [0, n_radial - 1] using tanh.
        Analogous to FSQ's bound() function.
        """
        max_level = float(self.n_radial - 1)
        # Scale so moderate encoder magnitudes map to mid-range levels
        return torch.tanh(mag / (max_level * 0.5 + eps)) * max_level

    def quantize_magnitude(self, mag: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Bound, round (with STE + quant dropout), and normalize magnitude.
        
        Returns:
            mag_value: continuous magnitude in [eps_mag, 1.0]
            mag_idx: integer level in [0, n_radial-1]
        """
        bounded = self.bound_magnitude(mag)
        drop_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded.unsqueeze(-1), drop_p).squeeze(-1)
        quantized = quantized.clamp(0, self.n_radial - 1)
        mag_idx = quantized.detach().long()
        mag_value = self._normalize_mag_level(quantized)
        return mag_value, mag_idx

    # ── Direction quantization ─────────────────────────────────────────

    def quantize_direction(self, z_dir: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Find nearest angular codebook point via cosine similarity.
        
        Args:
            z_dir: [..., D] unit vectors
        Returns:
            dir_q: [..., D] quantized unit vector (STE gradient)
            ang_idx: [...] integer codebook index
        """
        orig_shape = z_dir.shape[:-1]
        z_flat = z_dir.reshape(-1, self.dim)

        # Cosine similarity (both unit vectors)
        similarities = z_flat @ self.angular_codebook.T
        ang_idx = similarities.argmax(dim=-1)
        dir_q = self.angular_codebook[ang_idx]

        # STE: forward uses quantized, backward passes through
        dir_q = z_flat + (dir_q - z_flat).detach()

        dir_q = dir_q.reshape(*orig_shape, self.dim)
        ang_idx = ang_idx.reshape(*orig_shape)
        return dir_q, ang_idx

    # ── Token corruption (matches FSQ) ─────────────────────────────────

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Randomly corrupt some entries of the quantized Tensor."""
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    # ── Forward ────────────────────────────────────────────────────────

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize a single tensor.
        
        Args:
            z: [..., D] latent vectors
        Returns:
            z_q: [..., D] quantized vectors
            tokens: [...] integer token indices
        """
        z = z.float()
        assert z.shape[-1] == self.dim, \
            f"Expected dim {self.dim}, got {z.shape[-1]}"

        # ── Decompose into direction + magnitude ──────────────────
        mag = z.norm(dim=-1)                    # [...]
        z_dir = z / (mag.unsqueeze(-1) + 1e-8)  # [..., D]

        # ── Quantize direction ────────────────────────────────────
        dir_q, ang_idx = self.quantize_direction(z_dir)

        # ── Quantize magnitude ────────────────────────────────────
        mag_value, mag_idx = self.quantize_magnitude(mag)

        # ── Combine ───────────────────────────────────────────────
        z_q = dir_q * mag_value.unsqueeze(-1)

        # ── Optional token corruption ─────────────────────────────
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            z_q = self.corrupt_quant(z_q)
            # Recompute tokens from corrupted z_q
            # (needed because corrupt_quant changes the values)
            mag_c = z_q.norm(dim=-1)
            dir_c = z_q / (mag_c.unsqueeze(-1) + 1e-8)
            sims = dir_c.reshape(-1, self.dim) @ self.angular_codebook.T
            ang_idx = sims.argmax(dim=-1).reshape(z_q.shape[:-1])
            # For mag_idx, find nearest level
            mag_levels_norm = self._normalize_mag_level(self.radial_levels)
            mag_idx = (mag_c.unsqueeze(-1) - mag_levels_norm).abs().argmin(dim=-1)

        # ── Token index ───────────────────────────────────────────
        tokens = (ang_idx * self.n_radial + mag_idx).to(int32)

        return z_q, tokens.long()

    @torch.compiler.disable
    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Main forward. Matches FSQ.forward interface."""
        if self.packed_call:
            quant, tokens = packed_call(partial(self.forward_z), latents)
        elif isinstance(latents, list):
            quant, tokens = [], []
            for z_i in latents:
                q_i, t_i = self.forward_z(z_i)
                quant.append(q_i)
                tokens.append(t_i)
        else:
            quant, tokens = self.forward_z(latents)
        return quant, tokens

    # ── Index ↔ Embedding ──────────────────────────────────────────────

    def _indices_to_embedding_impl(self, indices: Tensor) -> Tensor:
        """Convert token index to embedding vector."""
        ang_idx = (indices // self.n_radial).long()
        mag_idx = (indices % self.n_radial).long()

        dir_vectors = self.angular_codebook[ang_idx]
        mag_levels = self.radial_levels[mag_idx]
        mag_values = self._normalize_mag_level(mag_levels)

        return dir_vectors * mag_values.unsqueeze(-1)

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        """
        Convert token indices back to latent embeddings.
        Matches FSQ.indices_to_embedding interface.
        """
        return self._indices_to_embedding_impl(indices)
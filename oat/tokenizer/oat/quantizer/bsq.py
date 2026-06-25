# --------------------------------------------------------------------
# Binary Spherical Quantization (BSQ) for OATTok
# Drop-in replacement for FSQ.
#
# Reference:
#   Zhao et al., "Image and Video Tokenization with Binary Spherical
#   Quantization", ICLR 2025. arXiv:2406.07548
#
# Key idea:
#   Project latent to unit hypersphere, then binary quantize per axis.
#   Implicit codebook C_BSQ = {-1/√L, +1/√L}^L with size 2^L.
#   No learned parameters. Bounded quantization error.
# --------------------------------------------------------------------

import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32
from torch.amp import autocast
from einops import repeat
from typing import List, Optional, Tuple

from oat.tokenizer.oat.util.packed_ops import packed_call


__all__ = ["BSQ"]


def _bsq_sign(u: Tensor) -> Tensor:
    """Binary sign: maps positive and zero to +1, negative to -1.
    Unlike torch.sign which maps 0→0, this maps 0→+1 per BSQ paper convention,
    ensuring the output is always in {-1, +1}."""
    return torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))


def sign_ste(u: Tensor) -> Tensor:
    """Sign function with straight-through estimator.
    Forward: _bsq_sign(u).  Backward: gradient passes through as identity."""
    return u + (_bsq_sign(u) - u).detach()


def sign_ste_quant_dropout(u: Tensor, scale: float, drop_quant_p: float) -> Tensor:
    """Sign-quantize with per-sample dropout (skip quantization with probability drop_quant_p).
    When dropped, returns the continuous u (still on sphere) instead of quantized û."""
    u_hat = scale * _bsq_sign(u)
    # STE: forward uses sign, backward passes through
    u_hat_ste = u + (u_hat - u).detach()

    if drop_quant_p <= 0.0:
        return u_hat_ste

    batch_size = u.shape[0]
    device = u.device
    # mask=1 means skip quantization (keep continuous u)
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    mask = mask.view(batch_size, *([1] * (u.ndim - 1)))
    # When mask=1: output = u (continuous, on sphere)
    # When mask=0: output = u_hat_ste (quantized, on BSQ lattice)
    return mask * u + (1 - mask) * u_hat_ste


class BSQ(nn.Module):
    """Binary Spherical Quantization — drop-in replacement for FSQ.

    Projects latent vectors to the unit hypersphere S^{L-1}, then applies
    per-axis binary quantization.  The implicit codebook is
    C = {-1/√L, +1/√L}^L  with  |C| = 2^L.

    Interface matches FSQ exactly:
        forward(latents) → (quant, tokens)
        indices_to_embedding(indices) → embeddings

    Args:
        L: Dimensionality of the spherical latent / number of binary bits.
            codebook_size = 2^L.  L=10 → 1024, L=12 → 4096.
        drop_quant_p: During training, skip quantization per sample with this
            probability (pass continuous sphere point instead).
        corrupt_tokens_p: During training, corrupt this fraction of token
            positions by replacing with random codebook entries.
        min_corrupt_tokens_p: Minimum corruption fraction (actual is sampled
            uniformly in [min, max]).
        apply_corrupt_tokens_p: Probability of activating corruption per sample.
        packed_call: Use packed_call for list inputs (same as FSQ).
    """

    def __init__(
        self,
        L: int = 10,
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
    ):
        super().__init__()

        self.L = L
        self.dim = L
        self._codebook_size = 2 ** L
        self.scale = 1.0 / math.sqrt(L)

        # Basis for binary → integer conversion: [1, 2, 4, 8, ...]
        _basis = (2 ** torch.arange(L)).long()
        self.register_buffer("_basis", _basis, persistent=False)

        # Pre-compute the full implicit codebook for corrupt_quant & indices_to_embedding
        # Shape: [2^L, L], each row is a point on S^{L-1}
        # For L<=16 this is at most 65536 × 16 × 4 bytes ≈ 4MB, fine.
        if L <= 16:
            all_indices = torch.arange(self._codebook_size)
            implicit_codebook = self._indices_to_codes(all_indices)
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)
        else:
            # For very large L, don't store full codebook — compute on the fly
            self.implicit_codebook = None

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p
        self.packed_call = packed_call

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    def __repr__(self):
        return (
            f"BSQ(\n"
            f"  L={self.L},\n"
            f"  codebook_size={self._codebook_size},\n"
            f"  drop_quant_p={self.drop_quant_p},\n"
            f")"
        )

    # ─────────────────────────────────────────────────────
    #  Core quantization
    # ─────────────────────────────────────────────────────

    def quantize(self, z: Tensor) -> Tensor:
        """Quantize: project to sphere, then binary quantize with STE.

        Args:
            z: [..., L] raw encoder output (any magnitude).
        Returns:
            [..., L] quantized vector on S^{L-1} (values ±1/√L).
        """
        # Project to unit hypersphere
        u = F.normalize(z, dim=-1, eps=1e-8)

        # Binary quantize with STE and optional dropout
        drop_p = self.drop_quant_p if self.training else 0.0
        quant = sign_ste_quant_dropout(u, self.scale, drop_p)

        return quant

    def codes_to_indices(self, quant: Tensor) -> Tensor:
        """Convert quantized vectors to integer token indices.

        Args:
            quant: [..., L] quantized vectors (values ±1/√L).
        Returns:
            [...] integer indices in {0, ..., 2^L - 1}.
        """
        # bits: 1 where positive, 0 where negative
        bits = (quant > 0).long()
        return (bits * self._basis).sum(dim=-1).to(int32)

    def _indices_to_codes(self, indices: Tensor) -> Tensor:
        """Convert integer indices to quantized vectors (internal, no gradient).

        Args:
            indices: [...] integer indices.
        Returns:
            [..., L] quantized vectors (values ±1/√L).
        """
        # Expand indices for bit extraction
        idx = indices.unsqueeze(-1).long()  # [..., 1]
        # Extract each bit: (idx >> i) & 1
        bits = (idx >> torch.arange(self.L, device=indices.device)) & 1  # [..., L]
        # Map 0 → -1/√L, 1 → +1/√L
        return self.scale * (2.0 * bits.float() - 1.0)

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        """Convert token indices back to quantized embeddings.
        Used by tokenizer.detokenize() during policy inference.

        Args:
            indices: [...] integer indices in {0, ..., 2^L - 1}.
        Returns:
            [..., L] quantized vectors on S^{L-1}.
        """
        if self.implicit_codebook is not None:
            # Fast path: lookup from pre-computed codebook
            return self.implicit_codebook[indices.long()]
        else:
            # Compute on the fly for large L
            return self._indices_to_codes(indices)

    # ─────────────────────────────────────────────────────
    #  Token corruption (training regularization)
    # ─────────────────────────────────────────────────────

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Randomly corrupt some token positions with random codebook entries."""
        quant_shape, quant_device = quant.shape[:-1], quant.device
        # Sample random token indices
        random_indices = torch.randint(
            low=0, high=self._codebook_size, size=quant_shape, device=quant_device
        )
        # Look up corresponding embeddings
        random_quant = self.indices_to_embedding(random_indices)
        # Sample corruption rate
        sample_corrupt_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        # Create per-position mask
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=self.L)
        return torch.where(corruption_mask, random_quant, quant)

    # ─────────────────────────────────────────────────────
    #  Forward pass
    # ─────────────────────────────────────────────────────

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize a single tensor input.

        Args:
            z: [..., L] continuous latent vectors.
        Returns:
            quant: [..., L] quantized vectors on S^{L-1}.
            tokens: [...] integer token indices (LongTensor).
        """
        assert z.shape[-1] == self.dim, \
            f"expected dimension of {self.dim} but found {z.shape[-1]}"

        quant = self.quantize(z.float())

        # Optional token corruption during training
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            quant = self.corrupt_quant(quant)

        tokens = self.codes_to_indices(quant)
        return quant, tokens.long()

    @torch.compiler.disable
    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize latents — supports tensor, list, or packed input.
        Drop-in compatible with FSQ.forward().

        Args:
            latents: [..., L] tensor, or list of tensors.
        Returns:
            quant: same shape as latents, quantized.
            tokens: same shape minus last dim, integer indices.
        """
        if self.packed_call:
            bsq_fn = partial(self.forward_z)
            quant, tokens = packed_call(bsq_fn, latents)
        elif isinstance(latents, list):
            quant, tokens = [], []
            for z_i in latents:
                q_i, t_i = self.forward_z(z_i)
                quant.append(q_i)
                tokens.append(t_i)
        else:
            quant, tokens = self.forward_z(latents)

        return quant, tokens
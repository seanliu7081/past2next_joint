# --------------------------------------------------------------------
# Product Hilbert FSQ quantizer for ZHill tokenizer.
# Replaces lexicographic (mixed-radix) index encoding in FSQ with
# Product Hilbert curve ordering for better locality preservation.
# --------------------------------------------------------------------

import random
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor, int32
from torch.amp import autocast
from typing import List, Optional, Tuple

from oat.tokenizer.oat.util.packed_ops import packed_call
from oat.tokenizer.zhill.quantizer.hilbert import build_hilbert_lut


__all__ = ["ProductHilbertFSQ"]


# helper functions (copied from FSQ)

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def round_ste_quant_dropout(z: Tensor, drop_quant_p: float) -> Tensor:
    """Round with straight through gradients, randomly skip quantization per sample."""
    zhat = z.round()
    batch_size = z.shape[0]
    device = z.device
    mask = torch.bernoulli(torch.full((batch_size,), drop_quant_p, device=device))
    mask = mask.view(batch_size, *([1] * (z.ndim - 1)))
    output = z + ((1 - mask) * (zhat - z)).detach()
    return output


class ProductHilbertFSQ(nn.Module):
    """FSQ with Product Hilbert curve index encoding instead of lexicographic (mixed-radix).

    The quantization grid, encoder, and decoder are identical to FSQ. Only the mapping
    from grid coordinates to token indices (and back) is changed to use 2D Hilbert curves
    applied to pairs of dimensions, giving better locality for autoregressive prediction.

    Args:
        levels: List of FSQ levels per dimension.
        drop_quant_p: During training, pass the non-rounded values with this probability.
        corrupt_tokens_p: During training, corrupt this percentage of tokens to random indices.
        min_corrupt_tokens_p: Minimum corruption percentage (sampled uniformly with max).
        apply_corrupt_tokens_p: Probability of activating token corruption per sample.
        packed_call: Pack list of examples and quantize jointly.
    """

    def __init__(
        self,
        levels: List[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        self.dim = len(levels)
        self.codebook_size = _levels.prod().item()

        # Pair dimensions for Product Hilbert ordering
        self.num_pairs = self.dim // 2
        self.has_remainder = (self.dim % 2 == 1)

        # Build Hilbert LUTs for each pair and compute pair codebook sizes
        pair_sizes = []
        for i in range(self.num_pairs):
            L0, L1 = levels[2 * i], levels[2 * i + 1]
            g2h, h2g = build_hilbert_lut(L0, L1)
            self.register_buffer(f"_grid_to_hilbert_{i}", g2h, persistent=False)
            self.register_buffer(f"_hilbert_to_grid_{i}", h2g, persistent=False)
            pair_sizes.append(L0 * L1)

        if self.has_remainder:
            pair_sizes.append(levels[-1])

        # Cross-pair mixed-radix basis
        basis_list = [1]
        for s in pair_sizes[:-1]:
            basis_list.append(basis_list[-1] * s)
        _pair_basis = torch.tensor(basis_list, dtype=int32)
        self.register_buffer("_pair_basis", _pair_basis, persistent=False)

        _pair_sizes = torch.tensor(pair_sizes, dtype=int32)
        self.register_buffer("_pair_sizes", _pair_sizes, persistent=False)

        # Build implicit codebook
        implicit_codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p
        self.packed_call = packed_call

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  levels={self._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  drop_quant_p={self.drop_quant_p!r},\n"
            ")"
        )

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        bounded = self.bound(z)
        drop_quant_p = self.drop_quant_p if self.training else 0.0
        quantized = round_ste_quant_dropout(bounded, drop_quant_p)
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts quantized codes to token indices using Product Hilbert ordering."""
        assert zhat.shape[-1] == self.dim
        coords = self._scale_and_shift(zhat).long()

        # Compute pair indices via Hilbert LUT lookup
        pair_indices = []
        for i in range(self.num_pairs):
            x = coords[..., 2 * i]
            y = coords[..., 2 * i + 1]
            g2h = getattr(self, f"_grid_to_hilbert_{i}")
            pair_idx = g2h[x, y]
            pair_indices.append(pair_idx)

        if self.has_remainder:
            pair_indices.append(coords[..., -1])

        # Combine with mixed-radix encoding across pairs
        final_index = torch.zeros_like(pair_indices[0])
        for i, pidx in enumerate(pair_indices):
            final_index = final_index + pidx * self._pair_basis[i]

        return final_index.to(int32)

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        """Inverse of codes_to_indices: converts token indices back to quantized embeddings."""
        all_coords = []

        for i in range(self.num_pairs):
            pair_idx = (indices // self._pair_basis[i]) % self._pair_sizes[i]
            h2g = getattr(self, f"_hilbert_to_grid_{i}")
            coords_pair = h2g[pair_idx]  # (..., 2)
            all_coords.append(coords_pair)

        if self.has_remainder:
            rem_idx = (indices // self._pair_basis[self.num_pairs]) % self._pair_sizes[self.num_pairs]
            all_coords.append(rem_idx.unsqueeze(-1))

        codes_non_centered = torch.cat(all_coords, dim=-1).float()
        codes = self._scale_and_shift_inverse(codes_non_centered)
        return codes

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Randomly corrupt some entries of the quantized Tensor."""
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_tokens_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_tokens_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        quant = self.quantize(z.float())
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            quant = self.corrupt_quant(quant)
        tokens = self.codes_to_indices(quant)

        return quant, tokens.long()  # type: ignore

    @torch.compiler.disable
    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.packed_call:
            fsq_fn = partial(self.forward_z)
            quant, tokens = packed_call(fsq_fn, latents)
        elif isinstance(latents, list):
            quant, tokens = [], []
            for z_i in latents:
                quant_i, tokens_i = self.forward_z(z_i)
                quant.append(quant_i)
                tokens.append(tokens_i)
        else:
            quant, tokens = self.forward_z(latents)

        return quant, tokens

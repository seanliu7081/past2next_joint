"""PolarOATTok v2: Geometry-aware action tokenizer with temporal compression.

Architecture:
    (B,32,7) -> Normalize -> PolarDecompose (exact SO(2), no params)
    -> Invariant (B,32,4)  -> RegisterEncoder -> (B,8,4) -> FSQ   -> inv tokens (B,8)
    -> Equivariant (B,32,3) -> RegisterEncoder -> (B,8,3) -> CyclicVQ -> eq tokens (B,8)
    -> Concatenate quantized (B,8,7) -> SinglePassDecoder -> (B,32,7) Cartesian

Key properties:
    - latent_horizon = 8 (same as original OATTok)
    - tokenize() returns {'inv': (B,8), 'eq': (B,8)}
    - vocab_sizes: inv=1000, eq=2304 (product of CyclicVQ bins)
    - Soft equivariance regularization during training
    - Decoder outputs Cartesian directly (intentional relaxation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ
from oat.tokenizer.polar.polar_decompose import PolarDecompose
from oat.tokenizer.polar_v2.cyclic_vq_product import CyclicVQProduct


def pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    device, dtype = token_ids.device, token_ids.dtype
    pad_len = max_seq_len - token_ids.shape[1]
    pad_seq = torch.zeros((token_ids.shape[0], pad_len), device=device, dtype=dtype)
    return torch.cat([token_ids, pad_seq], dim=1)


class PolarOATTokV2(BaseTokenizer):
    """PolarOATTok v2 with temporal compression via RegisterEncoders.

    Args:
        polar_decompose: Deterministic SO(2) decomposition module.
        inv_encoder: RegisterEncoder for invariant subspace (sample_dim=4).
        eq_encoder: RegisterEncoder for equivariant subspace (sample_dim=3).
        inv_quantizer: FSQ for invariant latents.
        eq_quantizer: CyclicVQProduct for equivariant latents.
        decoder: SinglePassDecoder (latent_dim=7, concatenated).
        equiv_reg_weight: Lambda for soft equivariance regularization.
        n_equiv_samples: Number of random rotations per batch for equiv reg.
    """

    def __init__(
        self,
        polar_decompose: PolarDecompose,
        inv_encoder: RegisterEncoder,
        eq_encoder: RegisterEncoder,
        inv_quantizer: FSQ,
        eq_quantizer: CyclicVQProduct,
        decoder: SinglePassDecoder,
        equiv_reg_weight: float = 0.1,
        n_equiv_samples: int = 4,
    ):
        super().__init__()
        self.polar_decompose = polar_decompose
        self.inv_encoder = inv_encoder
        self.eq_encoder = eq_encoder
        self.inv_quantizer = inv_quantizer
        self.eq_quantizer = eq_quantizer
        self.decoder = decoder
        self.normalizer = LinearNormalizer()
        self.equiv_reg_weight = equiv_reg_weight
        self.n_equiv_samples = n_equiv_samples

        assert inv_encoder.num_registers == eq_encoder.num_registers
        self.latent_horizon = inv_encoder.num_registers  # = 8

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        return {
            'inv': self.inv_quantizer.codebook_size,
            'eq': self.eq_quantizer.codebook_size,
        }

    # ── Optimizer / Normalizer ───────────────────────────────────────────────

    def get_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ── Core encode / decode ─────────────────────────────────────────────────

    def _encode_internal(self, nactions: torch.Tensor):
        """Internal encode on normalized actions.

        Returns:
            inv_quantized: (B, 8, 4) with STE gradient
            inv_tokens: (B, 8)
            eq_quantized: (B, 8, 3) with STE gradient
            eq_tokens: (B, 8)
        """
        inv, eq, null_mask = self.polar_decompose(nactions)

        inv_latents = self.inv_encoder(inv)    # (B, 8, 4)
        eq_latents = self.eq_encoder(eq)       # (B, 8, 3)

        inv_quantized, inv_tokens = self.inv_quantizer(inv_latents)
        eq_quantized, eq_tokens = self.eq_quantizer(eq_latents)

        return inv_quantized, inv_tokens, eq_quantized, eq_tokens

    def _decode_internal(
        self,
        inv_quantized: torch.Tensor,
        eq_quantized: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Decode concatenated quantized latents to normalized actions."""
        if eval_keep_k is None:
            eval_keep_k = [inv_quantized.shape[1]] * inv_quantized.shape[0]

        combined = torch.cat([inv_quantized, eq_quantized], dim=-1)  # (B, 8, 7)
        return self.decoder(combined, eval_keep_k=eval_keep_k)

    # ── Training forward ─────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """Training forward pass. Returns recon_loss + lambda * equiv_reg_loss.

        Also stores decomposed losses as attributes for external logging:
            self.last_recon_loss, self.last_equiv_loss
        """
        actions = batch['action']  # (B, 32, 7)
        nactions = self.normalizer['action'].normalize(actions)

        inv_q, _, eq_q, _ = self._encode_internal(nactions)
        recons = self._decode_internal(inv_q, eq_q)

        loss_recon = F.mse_loss(recons, nactions)

        if self.equiv_reg_weight > 0 and self.training:
            loss_equiv = self._equiv_reg_loss(nactions)
            loss = loss_recon + self.equiv_reg_weight * loss_equiv
        else:
            loss_equiv = torch.tensor(0.0, device=actions.device)
            loss = loss_recon

        # Stash for logging (detached scalars)
        self.last_recon_loss = loss_recon.item()
        self.last_equiv_loss = loss_equiv.item()

        return loss

    def _equiv_reg_loss(self, nactions: torch.Tensor) -> torch.Tensor:
        """Soft equivariance regularization.

        For random SO(2) rotations:
          decoded(rotate(x)) ≈ rotate(decoded(x))
        """
        device = nactions.device
        loss = torch.tensor(0.0, device=device)

        for _ in range(self.n_equiv_samples):
            phi = torch.rand(1, device=device) * 2 * torch.pi
            cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)

            # Path 1: rotate then encode-decode
            rotated = self._rotate_actions(nactions, cos_phi, sin_phi)
            inv_q_r, _, eq_q_r, _ = self._encode_internal(rotated)
            decoded_rot = self._decode_internal(inv_q_r, eq_q_r)

            # Path 2: encode-decode then rotate
            inv_q, _, eq_q, _ = self._encode_internal(nactions)
            decoded_orig = self._decode_internal(inv_q, eq_q)
            decoded_orig_rot = self._rotate_actions(decoded_orig, cos_phi, sin_phi)

            loss = loss + F.mse_loss(decoded_rot, decoded_orig_rot)

        return loss / self.n_equiv_samples

    @staticmethod
    def _rotate_actions(actions: torch.Tensor, cos_phi, sin_phi) -> torch.Tensor:
        """Apply SO(2) rotation to (dx,dy) and (droll,dpitch) doublets."""
        rotated = actions.clone()
        rotated[..., 0] = cos_phi * actions[..., 0] - sin_phi * actions[..., 1]
        rotated[..., 1] = sin_phi * actions[..., 0] + cos_phi * actions[..., 1]
        rotated[..., 3] = cos_phi * actions[..., 3] - sin_phi * actions[..., 4]
        rotated[..., 4] = sin_phi * actions[..., 3] + cos_phi * actions[..., 4]
        return rotated

    # ── Public interface (BaseTokenizer) ─────────────────────────────────────

    def encode(self, samples: torch.Tensor):
        """Encode raw actions into quantized latents and tokens.

        Args:
            samples: (B, T, 7) raw actions.

        Returns:
            latents: dict {'inv': (B,8,4), 'eq': (B,8,3)} quantized latents.
            tokens: dict {'inv': (B,8), 'eq': (B,8)} token indices.
        """
        nsamples = self.normalizer['action'].normalize(samples)
        inv_q, inv_tok, eq_q, eq_tok = self._encode_internal(nsamples)
        latents = {'inv': inv_q, 'eq': eq_q}
        tokens = {'inv': inv_tok, 'eq': eq_tok}
        return latents, tokens

    def decode(
        self,
        latents: dict,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Decode quantized latents to raw actions.

        Args:
            latents: dict {'inv': (B,8,4), 'eq': (B,8,3)}.

        Returns:
            samples: (B, T, 7) unnormalized actions.
        """
        nsamples = self._decode_internal(
            latents['inv'], latents['eq'], eval_keep_k=eval_keep_k
        )
        return self.normalizer['action'].unnormalize(nsamples)

    def autoencode(
        self,
        samples: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        latents, _ = self.encode(samples)
        return self.decode(latents, eval_keep_k=eval_keep_k)

    def tokenize(self, samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Tokenize raw actions into factored token indices.

        Args:
            samples: (B, T, 7) raw actions.

        Returns:
            tokens: {'inv': (B, 8), 'eq': (B, 8)} LongTensor indices.
        """
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(
        self,
        tokens: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct raw actions from factored token indices.

        Args:
            tokens: {'inv': (B, 8), 'eq': (B, 8)} or shorter (will be padded).

        Returns:
            samples: (B, T, 7) unnormalized actions.
        """
        inv_tokens = tokens['inv']
        eq_tokens = tokens['eq']

        # Pad if needed
        if inv_tokens.shape[1] < self.latent_horizon:
            inv_tokens = pad_token_seq(inv_tokens, self.latent_horizon)
            eq_tokens = pad_token_seq(eq_tokens, self.latent_horizon)

        token_lens = [inv_tokens.shape[1]] * inv_tokens.shape[0]

        inv_q = self.inv_quantizer.indices_to_embedding(inv_tokens)
        eq_q = self.eq_quantizer.indices_to_embedding(eq_tokens)

        latents = {'inv': inv_q, 'eq': eq_q}
        return self.decode(latents, eval_keep_k=token_lens)

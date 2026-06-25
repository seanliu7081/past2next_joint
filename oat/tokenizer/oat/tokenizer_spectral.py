"""
OATTok with support for SpectralBasisEncoder auxiliary losses.

Changes from original tokenizer.py:
  1. Encoder type hint is nn.Module (accepts both RegisterEncoder and SpectralBasisEncoder)
  2. forward() adds ortho_loss when encoder supports it
  3. New forward_with_tcl() for time-contrastive training
  4. New compute_overlap_rate() for monitoring tokenizer quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ


def pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Pad the token sequence to the maximum length."""
    device, dtype = token_ids.device, token_ids.dtype
    pad_len = max_seq_len - token_ids.shape[1]
    pad_seq = torch.zeros((token_ids.shape[0], pad_len), device=device, dtype=dtype)
    return torch.cat([token_ids, pad_seq], dim=1)


class OATTok(BaseTokenizer):
    def __init__(
        self,
        encoder: nn.Module,      # RegisterEncoder or SpectralBasisEncoder
        decoder: SinglePassDecoder,
        quantizer: FSQ,
        # Auxiliary loss weights (only active if encoder supports them)
        tcl_weight: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon
        self.tcl_weight = tcl_weight

    @property
    def _encoder_has_ortho(self) -> bool:
        return hasattr(self.encoder, "ortho_loss") and callable(self.encoder.ortho_loss)

    @property
    def _encoder_has_tcl(self) -> bool:
        return hasattr(self.encoder, "tcl_loss") and callable(self.encoder.tcl_loss)

    def get_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch) -> torch.Tensor:
        """
        Standard training forward pass.

        Returns total loss = MSE_recon + ortho_reg (if encoder supports it).
        """
        samples = batch["action"]

        # normalize
        nsamples = self.normalizer["action"].normalize(samples)

        # encode & quantize
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)

        # decode
        recons = self.decoder(latents)
        recon_loss = F.mse_loss(recons, nsamples)

        # Add orthogonality regularization if encoder supports it
        total_loss = recon_loss
        if self._encoder_has_ortho:
            total_loss = total_loss + self.encoder.ortho_loss()

        return total_loss

    def forward_with_tcl(self, batch) -> torch.Tensor:
        """
        Training forward pass with Time Contrastive Loss.

        Expects batch to contain:
          - 'action':          [B, T, D]  current action chunks
          - 'action_adjacent': [B, T, D]  temporally adjacent chunks (t+1 or t-1)

        If 'action_adjacent' is not present, falls back to standard forward().
        """
        if "action_adjacent" not in batch or self.tcl_weight == 0.0 or not self._encoder_has_tcl:
            return self.forward(batch)

        samples = batch["action"]
        samples_adj = batch["action_adjacent"]

        nsamples = self.normalizer["action"].normalize(samples)
        nsamples_adj = self.normalizer["action"].normalize(samples_adj)

        # Encode both chunks (before quantization, for smoother contrastive signal)
        latents = self.encoder(nsamples)
        latents_adj = self.encoder(nsamples_adj)

        # TCL loss on pre-quantization latents
        z_anchor = self.encoder.pool_latents(latents)      # [B, latent_dim]
        z_positive = self.encoder.pool_latents(latents_adj) # [B, latent_dim]
        # Use other samples in batch as negatives
        z_negatives = z_positive.detach().roll(1, dims=0)   # simple: shifted batch
        # Gather more negatives from the batch
        B = z_anchor.shape[0]
        neg_indices = torch.arange(B, device=z_anchor.device)
        neg_indices = neg_indices.roll(1)
        # Stack multiple negative shifts
        neg_list = []
        for shift in range(1, min(B, 8)):
            neg_list.append(z_anchor.detach().roll(shift, dims=0))
        z_negatives = torch.stack(neg_list, dim=1)  # [B, num_neg, d]

        tcl_loss = self.encoder.tcl_loss(z_anchor, z_positive, z_negatives)

        # Quantize and decode (standard reconstruction path)
        latents_q, _ = self.quantizer(latents)
        recons = self.decoder(latents_q)
        recon_loss = F.mse_loss(recons, nsamples)

        # Combine losses
        total_loss = recon_loss + self.tcl_weight * tcl_loss
        if self._encoder_has_ortho:
            total_loss = total_loss + self.encoder.ortho_loss()

        return total_loss

    # -- Inference methods (unchanged from original) --------------------------

    def encode(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nsamples = self.normalizer["action"].normalize(samples)
        latents = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    def decode(
        self,
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]
        assert all(k <= self.latent_horizon for k in eval_keep_k), \
            f"All eval_keep_k must be <= {self.latent_horizon}"
        nsamples = self.decoder(latents, eval_keep_k=eval_keep_k)
        samples = self.normalizer["action"].unnormalize(nsamples)
        return samples

    def autoencode(
        self,
        samples: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        latents, _ = self.encode(samples)
        recons = self.decode(latents, eval_keep_k=eval_keep_k)
        return recons

    def tokenize(self, samples: torch.Tensor) -> torch.Tensor:
        _, tokens = self.encode(samples)
        return tokens

    def detokenize(
        self,
        tokens: Union[torch.Tensor, List[List[int]]],
    ) -> torch.Tensor:
        if isinstance(tokens, list):
            token_lens = [t.shape[1] for t in tokens]
            tokens = torch.cat(
                [pad_token_seq(t, self.latent_horizon) for t in tokens], dim=0
            )
        elif isinstance(tokens, torch.Tensor):
            token_lens = [tokens.shape[1]] * tokens.shape[0]
            if tokens.shape[-1] < self.latent_horizon:
                tokens = pad_token_seq(tokens, self.latent_horizon)
        else:
            raise ValueError(f"Unknown token type {type(tokens)}")

        latents = self.quantizer.indices_to_embedding(tokens)
        samples = self.decode(latents, eval_keep_k=token_lens)
        return samples

    # -- Evaluation utilities -------------------------------------------------

    @torch.no_grad()
    def compute_overlap_rate(
        self,
        actions_t: torch.Tensor,
        actions_t1: torch.Tensor,
    ) -> float:
        """
        Compute overlap rate (OR) between temporally adjacent action chunks.

        This is the key metric identified by ActionCodec: higher OR means
        more stable supervision signal for the AR policy.

        Args:
            actions_t:  Action chunks at time t   [B, T, D]
            actions_t1: Action chunks at time t+1 [B, T, D]

        Returns:
            Mean overlap rate in [0, 1].
        """
        tokens_t = self.tokenize(actions_t)    # [B, K]
        tokens_t1 = self.tokenize(actions_t1)  # [B, K]

        # Per-sample overlap: fraction of tokens that match
        matches = (tokens_t == tokens_t1).float()  # [B, K]
        per_sample_or = matches.mean(dim=-1)        # [B]
        return per_sample_or.mean().item()
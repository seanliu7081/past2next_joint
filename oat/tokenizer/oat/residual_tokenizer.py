"""
ResidualOATTok — ZeroOrder-residual drop-in for OATTok
=======================================================

Key idea (from Phase 1 analysis):
  baseline[h] = a_0  (last action before chunk, i.e. past_action[:, -1, :])
  residual[h] = action[:, h, :] - baseline      for h = 0 .. H-1

Phase 1 showed that ZeroOrder residuals have:
  - 94.7 % variance reduction at h=0
  - 37.1 % variance reduction over the full chunk (var_ratio_full = 0.629)
  - Beneficial compression for h=0..11 (out of H=16 latent tokens)
  - No divergence (unlike Linear/Taylor), making it safe for the full chunk

Compared to OATTok the ONLY changes are:
  1. forward / encode / decode / autoencode / tokenize / detokenize
     all accept an optional `baseline` tensor (B, action_dim).
  2. The normalizer is fit on residuals (done externally, same API as before).
  3. decode() adds the baseline back before returning raw actions.

Everything else (RegisterEncoder, SinglePassDecoder, FSQ, MaskedNestedDropout,
positional embeddings, ...) is completely unchanged.

Train/inference alignment
-------------------------
  Training:   baseline = batch["past_action"][:, -1, :]   # a_t, step before chunk
  Inference:  baseline = policy._past_buffer[:, -1, :]    # same step
  → zero train/inference distribution mismatch.

Time alignment (from zarr_dataset_with_past.py, To=2, Ta=32, past_n=7):
  obs:         [t,   t+1]
  past_action: [t-6, t-5, ..., t]     ← past_action[:, -1, :] = a_t
  action:      [t+1, t+2, ..., t+32]  ← residual = action - a_t

Usage — tokenizer training
--------------------------
  tok = ResidualOATTok(encoder, decoder, quantizer)
  res_norm = compute_residual_normalizer(train_dataloader)
  tok.set_normalizer(res_norm)

  loss = tok(batch)   # batch must have "action" + ("baseline" or "past_action")

Usage — policy (Phase 3), minimal diff from current policy
-----------------------------------------------------------
  # in forward() (training):
  baseline = batch["past_action"][:, -1, :]
  tokens   = self.action_tokenizer.tokenize(batch["action"], baseline=baseline)

  # in predict_action() (inference):
  baseline = self._past_buffer[:, -1, :]
  actions  = self.action_tokenizer.detokenize(tokens, baseline=baseline)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ


# ── helpers ────────────────────────────────────────────────────────────────

def _resolve_baseline(
    baseline: Optional[torch.Tensor],
    batch: Optional[dict],
) -> torch.Tensor:
    """
    Resolve baseline from explicit arg or batch dict.
    Priority: explicit baseline > batch["baseline"] > batch["past_action"][:, -1]
    """
    if baseline is not None:
        return baseline
    if batch is not None:
        if "baseline" in batch:
            return batch["baseline"]
        if "past_action" in batch:
            return batch["past_action"][:, -1, :]   # (B, action_dim)
    raise ValueError(
        "ResidualOATTok needs a baseline. "
        "Pass baseline= kwarg, or include 'baseline'/'past_action' in the batch."
    )


def _pad_token_seq(token_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    pad_len = max_seq_len - token_ids.shape[1]
    if pad_len <= 0:
        return token_ids
    pad = torch.zeros(
        (token_ids.shape[0], pad_len),
        device=token_ids.device,
        dtype=token_ids.dtype,
    )
    return torch.cat([token_ids, pad], dim=1)


# ── main class ─────────────────────────────────────────────────────────────

class ResidualOATTok(BaseTokenizer):
    """
    ZeroOrder-residual tokenizer.  Strict superset of OATTok:
    all methods accept an optional `baseline` kwarg (B, action_dim).
    When baseline=None the class operates identically to OATTok.
    """

    def __init__(
        self,
        encoder: RegisterEncoder,
        decoder: SinglePassDecoder,
        quantizer: FSQ,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.normalizer = LinearNormalizer()
        self.latent_horizon = self.decoder.latent_horizon

    # ── optimiser (unchanged) ──────────────────────────────────────────────

    def get_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        decay   = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() <  2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=learning_rate, betas=betas,
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        IMPORTANT: pass a normalizer fit on RESIDUALS, not raw actions.
        Use compute_residual_normalizer() to create it.
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ── residual helpers ───────────────────────────────────────────────────

    def _to_residual(self, samples: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """(B,H,D), (B,D) → (B,H,D) residual."""
        return samples - baseline.unsqueeze(1)

    def _from_residual(self, residuals: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """(B,H,D), (B,D) → (B,H,D) raw actions."""
        return residuals + baseline.unsqueeze(1)

    # ── training forward ───────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Training loss (MSE in normalised-residual space).

        batch must contain:
          "action"      : (B, H, action_dim)
          "baseline"    : (B, action_dim)         — OR —
          "past_action" : (B, past_n, action_dim)   baseline = past_action[:,-1]
        """
        samples  = batch["action"]
        baseline = _resolve_baseline(None, batch)

        # residual in raw action space
        residuals  = self._to_residual(samples, baseline)

        # normalise (stats fit on residuals, not raw actions)
        nresiduals = self.normalizer["action"].normalize(residuals)

        # encode → quantize → decode (architecture unchanged)
        latents          = self.encoder(nresiduals)
        latents, _tokens = self.quantizer(latents)
        recons           = self.decoder(latents)

        return F.mse_loss(recons, nresiduals)

    # ── encode ─────────────────────────────────────────────────────────────

    def encode(
        self,
        samples: torch.Tensor,                   # (B, H, action_dim) raw actions
        baseline: Optional[torch.Tensor] = None, # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (quantised latents, token ids)."""
        if baseline is not None:
            samples = self._to_residual(samples, baseline)
        nsamples        = self.normalizer["action"].normalize(samples)
        latents         = self.encoder(nsamples)
        latents, tokens = self.quantizer(latents)
        return latents, tokens

    # ── decode ─────────────────────────────────────────────────────────────

    def decode(
        self,
        latents: torch.Tensor,                   # (B, L, latent_dim)
        baseline: Optional[torch.Tensor] = None, # (B, action_dim)
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Returns raw actions (B, H, action_dim)."""
        if eval_keep_k is None:
            eval_keep_k = [latents.shape[1]] * latents.shape[0]

        nresiduals = self.decoder(latents, eval_keep_k=eval_keep_k)
        residuals  = self.normalizer["action"].unnormalize(nresiduals)

        if baseline is not None:
            return self._from_residual(residuals, baseline)
        return residuals  # no-baseline fallback (raw / OATTok-compatible mode)

    # ── convenience wrappers ───────────────────────────────────────────────

    def autoencode(
        self,
        samples: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        latents, _ = self.encode(samples, baseline=baseline)
        return self.decode(latents, baseline=baseline, eval_keep_k=eval_keep_k)

    def tokenize(
        self,
        samples: torch.Tensor,                   # (B, H, action_dim) raw actions
        baseline: Optional[torch.Tensor] = None, # (B, action_dim)
    ) -> torch.Tensor:
        """Raw actions → token ids (B, L)."""
        _, tokens = self.encode(samples, baseline=baseline)
        return tokens

    def detokenize(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
        baseline: Optional[torch.Tensor] = None, # (B, action_dim)
    ) -> torch.Tensor:
        """Token ids → raw actions (B, H, action_dim)."""
        if isinstance(tokens, list):
            token_lens = [t.shape[1] for t in tokens]
            tokens = torch.cat([
                _pad_token_seq(t, self.latent_horizon) for t in tokens
            ], dim=0)
        elif isinstance(tokens, torch.Tensor):
            token_lens = [tokens.shape[1]] * tokens.shape[0]
            if tokens.shape[1] < self.latent_horizon:
                tokens = _pad_token_seq(tokens, self.latent_horizon)
        else:
            raise ValueError(f"Unknown token type: {type(tokens)}")

        latents = self.quantizer.indices_to_embedding(tokens)
        return self.decode(latents, baseline=baseline, eval_keep_k=token_lens)

    # ── checkpoint helpers ─────────────────────────────────────────────────

    @classmethod
    def from_oattok_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "ResidualOATTok":
        """
        Wrap a pre-trained OATTok checkpoint as ResidualOATTok.
        Reuses encoder/decoder/quantizer weights.

        *** After loading, call set_normalizer(compute_residual_normalizer(...))
            before fine-tuning — the original normalizer was fit on raw actions. ***
        """
        from oat.tokenizer.oat.tokenizer import OATTok
        oattok = OATTok.from_checkpoint(checkpoint_path, device=device)
        return cls(
            encoder=oattok.encoder,
            decoder=oattok.decoder,
            quantizer=oattok.quantizer,
        )
        # normalizer intentionally NOT copied — must be re-fit on residuals


# ── normalizer utility ─────────────────────────────────────────────────────

def compute_residual_normalizer(
    dataloader,
    device: str = "cpu",
) -> LinearNormalizer:
    """
    One-pass fit of a LinearNormalizer on ZeroOrder residuals.

    Dataloader must yield batches with:
      "action"      : (B, H, action_dim)
      "past_action" : (B, past_n, action_dim)   OR   "baseline": (B, action_dim)

    Returns a LinearNormalizer ready for tok.set_normalizer().

    Example
    -------
        res_norm = compute_residual_normalizer(train_loader, device="cuda")
        tok.set_normalizer(res_norm)
    """
    all_residuals = []
    for batch in dataloader:
        actions  = batch["action"].to(device)
        baseline = _resolve_baseline(None, batch).to(device)
        residuals = actions - baseline.unsqueeze(1)         # (B, H, action_dim)
        all_residuals.append(residuals.reshape(-1, residuals.shape[-1]).cpu())

    all_residuals = torch.cat(all_residuals, dim=0)         # (N, action_dim)
    normalizer = LinearNormalizer()
    normalizer.fit({"action": all_residuals})
    return normalizer
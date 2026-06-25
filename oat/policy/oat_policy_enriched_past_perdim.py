"""
Path B: OATPolicy with Enriched Past + Per-Dimension Prediction + EMD Loss.

Architecture change: adds 4 independent heads predicting per FSQ dimension
over [8, 5, 5, 5] classes respectively, used alongside the AR model's
existing full-vocab head.

Training loss: per_dim_CE + lambda * per_dim_EMD
Inference: custom generate loop with per-dim sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from oat.policy.oat_policy_with_enriched_past import OATPolicyWithEnrichedPast
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.loss.ordinal_loss import (
    decompose_indices, compose_indices,
    per_dim_ce_loss, per_dim_emd_loss,
)


class OATPolicyEnrichedPastPerDim(OATPolicyWithEnrichedPast):
    """
    Per-dimension prediction variant. Inherits condition building,
    obs encoding, tokenizer, and buffer management from parent.

    Adds:
      - 4 per-dimension Linear heads (8, 5, 5, 5 classes)

    Overrides:
      - __init__: adds per-dim heads + FSQ buffers
      - forward: per-dim CE + EMD loss
      - predict_action: custom generate loop with per-dim sampling
      - get_optimizer: includes per-dim heads in optimizer
    """

    def __init__(
        self,
        # All parent args
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        past_n: int = 7,
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        topk: int = 10,
        # New: ordinal loss weight
        ordinal_lambda: float = 0.1,
    ):
        super().__init__(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            action_tokenizer=action_tokenizer,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            past_n=past_n,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            temperature=temperature,
            topk=topk,
        )

        self.ordinal_lambda = ordinal_lambda

        # Store FSQ structure
        self.register_buffer("_fsq_levels", action_tokenizer.quantizer._levels.clone())
        self.register_buffer("_fsq_basis", action_tokenizer.quantizer._basis.clone())
        self.fsq_levels_list = action_tokenizer.quantizer._levels.tolist()  # [8, 5, 5, 5]

        # Per-dimension prediction heads
        self.per_dim_heads = nn.ModuleList([
            nn.Linear(embed_dim, level) for level in self.fsq_levels_list
        ])

        # Initialize per-dim heads with xavier
        for head in self.per_dim_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        num_perdim_params = sum(
            sum(p.numel() for p in head.parameters())
            for head in self.per_dim_heads
        )
        print(f"  per_dim_heads: {num_perdim_params / 1e3:.1f}K params")
        print(f"  ordinal_lambda={ordinal_lambda}, loss=perDimCE + lambda*perDimEMD\n")

    def _get_hidden_states(self, tokens, cond):
        """
        Get pre-head hidden states from the AR model using a forward hook
        on the final normalization layer (ln_f).
        """
        hidden_container = {}

        def hook_fn(module, input, output):
            hidden_container['h'] = output

        handle = self.model.ln_f.register_forward_hook(hook_fn)
        try:
            _ = self.model(tokens, cond=cond)
        finally:
            handle.remove()

        return hidden_container['h']  # [B, seq_len, embed_dim]

    def forward(self, batch) -> torch.Tensor:
        # -- Tokenize (frozen) --
        with torch.no_grad():
            action_tokens = self.action_tokenizer.tokenize(batch["action"])

        B = batch["action"].shape[0]
        device = batch["action"].device

        # -- Encode obs + build condition --
        features = self.obs_encoder(batch["obs"])
        past_actions = batch["past_action"]
        cond = self._build_condition(features, past_actions)

        # -- Prepend BOS --
        tokens_with_bos = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device),
            action_tokens,
        ], dim=1)

        # -- Get hidden states before the head --
        hidden = self._get_hidden_states(tokens_with_bos[:, :-1], cond)
        # hidden: [B, 8, embed_dim]

        # -- Decompose targets into per-dim codes --
        targets = action_tokens  # [B, 8], values in {0..999}
        targets_flat = targets.reshape(-1)  # [B*8]
        targets_per_dim = decompose_indices(
            targets_flat, self._fsq_levels, self._fsq_basis
        )  # list of 4 tensors, each [B*8]

        # -- Per-dim logits --
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])  # [B*8, embed_dim]
        logits_per_dim = [head(hidden_flat) for head in self.per_dim_heads]
        # list of 4 tensors: [B*8, 8], [B*8, 5], [B*8, 5], [B*8, 5]

        # -- Losses --
        ce = per_dim_ce_loss(logits_per_dim, targets_per_dim)
        emd = per_dim_emd_loss(logits_per_dim, targets_per_dim)

        total_loss = ce + self.ordinal_lambda * emd
        return total_loss

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Custom inference with per-dimension sampling.

        At each AR step:
          1. Run the full AR model forward to get hidden states at last position
          2. Apply 4 per-dim heads -> 4 categorical distributions
          3. Sample from each -> 4 per-dim codes
          4. Compose back into a flat index
          5. Feed this index as the next input token

        Note: Does NOT use KV caching (re-runs full forward each step).
        Acceptable because sequence is only 9 tokens long.
        """
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # -- Encode observation --
        features = self.obs_encoder(obs_dict)
        B = features.shape[0]

        # -- Get or init past buffer --
        if (
            self._past_buffer is None
            or self._past_buffer.shape[0] != B
            or self._past_buffer.device != self.device
        ):
            self._past_buffer = torch.zeros(
                B, self.past_n, self.action_dim,
                device=self.device, dtype=features.dtype,
            )

        # -- Build condition --
        cond = self._build_condition(features, self._past_buffer)

        # -- Autoregressive generation with per-dim sampling --
        tokens = torch.full(
            (B, 1), self.bos_id, dtype=torch.long, device=self.device,
        )

        for step in range(use_k_tokens):
            # Get hidden states for all positions so far
            hidden = self._get_hidden_states(tokens, cond)  # [B, seq_len, d]
            h_last = hidden[:, -1, :]  # [B, d]

            # Per-dim sampling
            per_dim_codes = []
            for k, head in enumerate(self.per_dim_heads):
                logits_k = head(h_last) / temperature  # [B, L_k]

                # Optional top-k per dimension
                L_k = logits_k.shape[-1]
                effective_topk = min(topk, L_k)
                if effective_topk < L_k:
                    topk_vals, _ = logits_k.topk(effective_topk, dim=-1)
                    threshold = topk_vals[:, -1:]
                    logits_k = logits_k.where(
                        logits_k >= threshold,
                        torch.full_like(logits_k, float('-inf'))
                    )

                probs_k = F.softmax(logits_k, dim=-1)
                code_k = torch.multinomial(probs_k, 1).squeeze(-1)  # [B]
                per_dim_codes.append(code_k)

            # Compose flat index from per-dim codes
            flat_idx = compose_indices(per_dim_codes, self._fsq_basis)  # [B]
            tokens = torch.cat([tokens, flat_idx.unsqueeze(1)], dim=1)

        # Drop BOS, clamp to valid range
        action_tokens = tokens[:, 1:]
        action_tokens = action_tokens.clamp(0, self.bos_id - 1)

        # -- Decode tokens -> continuous actions --
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(tokens=action_tokens)

        action = action_pred[:, : self.n_action_steps]

        # -- Update past buffer --
        n_exec = self.n_action_steps
        past_n = self.past_n
        if n_exec >= past_n:
            self._past_buffer = action_pred[:, n_exec - past_n: n_exec].detach().clone()
        else:
            self._past_buffer = torch.cat([
                self._past_buffer[:, n_exec:],
                action_pred[:, :n_exec].detach().clone(),
            ], dim=1)

        return {"action": action, "action_pred": action_pred}

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Override to include per_dim_heads in the policy optimizer group."""
        encoder_decay, encoder_nodecay = [], []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_decay if param.dim() >= 2 else encoder_nodecay).append(param)

        policy_decay, policy_nodecay = [], []
        policy_modules = [
            self.model, self.acc_proj, self.jerk_proj, self.raw_proj,
            self.per_dim_heads,
        ]
        for module in policy_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (policy_decay if param.dim() >= 2 else policy_nodecay).append(param)

        optim_groups = [
            {"params": policy_decay,    "lr": policy_lr,  "weight_decay": weight_decay},
            {"params": policy_nodecay,  "lr": policy_lr,  "weight_decay": 0.0},
            {"params": encoder_decay,   "lr": obs_enc_lr, "weight_decay": weight_decay},
            {"params": encoder_nodecay, "lr": obs_enc_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, betas=betas)

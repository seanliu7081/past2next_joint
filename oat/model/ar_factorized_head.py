"""Factorized autoregressive head for the predictive-tokenizer policy.

This module adds a *factorized* output/input mode on top of the existing
``AutoregressiveModel`` (``oat/model/autoregressive/transformer_cache.py``).
The transformer body, depth, heads, ``embed_dim``, cross-attention to the
Past2Next condition, and positional embeddings are all reused unchanged. Only
the input embedding and the output projection are replaced:

  * Output: instead of one flat ``5001``-way logit per register position, emit
    ``len(levels)`` heads of sizes ``levels`` (e.g. ``[8,5,5,5,5]``), each a
    per-dimension categorical over that FSQ axis's grid points. These align
    index-for-index with the FSQ grid (no geometric mismatch).
  * Input: instead of a flat ``5000``-vocab embedding table, use per-dimension
    embedding tables ``dim_embed[d]: Emb(L_d -> n_emb)`` summed across dims. The
    teacher-forcing inputs are built from the (detached) soft posterior or from
    hard argmax ids; a learned ``<BOS>`` embedding starts the sequence.

The legacy flat path (``forward``/``generate`` inherited from the parent) is
left intact but its ``tok_emb``/``head`` parameters are *frozen* so they do not
trip DDP ``find_unused_parameters=False`` in factorized mode.

NOTE on correctness: ``forward_factorized`` returns **log-probabilities** (one
``[B, L, L_d]`` tensor per FSQ dim). The rate term in the policy is the
cross-entropy ``H(q, p) = -sum_k q*log p`` computed directly from these
log-probs — never a symmetric KL, never a ``+ q*log q`` term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class FactorizedAutoregressiveModel(AutoregressiveModel):
    """``AutoregressiveModel`` with an optional per-dimension factorized head.

    Args:
        levels: FSQ levels, e.g. ``[8, 5, 5, 5, 5]``. Defines the per-dim
            categorical sizes for both the input embeddings and output heads.
        ar_head_mode: ``"factorized"`` (default for this policy) or ``"flat"``.
            ``"flat"`` keeps the parent behaviour entirely; ``"factorized"``
            additionally builds the per-dim embeddings/heads and freezes the
            legacy flat ``tok_emb``/``head`` so they stay out of the autograd
            graph (no DDP unused-parameter error).
        All other args are forwarded verbatim to ``AutoregressiveModel``.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_cond_len: int,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        levels: Optional[List[int]] = None,
        ar_head_mode: str = "factorized",
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            max_cond_len=max_cond_len,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
        )

        assert ar_head_mode in ("flat", "factorized"), ar_head_mode
        self.ar_head_mode = ar_head_mode

        if levels is None:
            levels = []
        self.levels = list(levels)
        self.n_dims = len(self.levels)

        if ar_head_mode == "factorized":
            assert self.n_dims > 0, "factorized mode requires non-empty `levels`"

            # per-dim input embedding tables (replace the flat tok_emb for inputs)
            self.dim_embed = nn.ModuleList(
                [nn.Embedding(Ld, n_emb) for Ld in self.levels]
            )
            # per-dim output heads (replace the flat head)
            self.dim_head = nn.ModuleList(
                [nn.Linear(n_emb, Ld, bias=True) for Ld in self.levels]
            )
            # learned <BOS> input embedding (BOS is input-only, never a target)
            self.bos_embed = nn.Parameter(torch.zeros(n_emb))

            self._init_factorized_weights()

            # Freeze the inherited flat input/output projection. They are tied
            # (head.weight is tok_emb.weight), so freezing one freezes both.
            # In factorized mode they are never used; freezing keeps them out of
            # the optimizer (get_optimizer filters requires_grad) and out of the
            # DDP reducer (find_unused_parameters=False stays happy).
            self.tok_emb.weight.requires_grad_(False)
            self.head.weight.requires_grad_(False)

    def _init_factorized_weights(self):
        for emb in self.dim_embed:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for lin in self.dim_head:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.constant_(lin.bias, 0.0)
        nn.init.normal_(self.bos_embed, mean=0.0, std=0.02)

    # ── Input-embedding builders (teacher forcing) ──────────────────────────

    def _shift_with_bos(self, emb: torch.Tensor) -> torch.Tensor:
        """Prepend the BOS embedding and drop the last position.

        ``emb`` holds the per-position embeddings of codes ``[z_0, ..., z_{L-1}]``.
        The AR consumes ``[BOS, z_0, ..., z_{L-2}]`` (length L) to predict
        ``[z_0, ..., z_{L-1}]`` (length L) — matching the flat-head off-by-one
        convention (``action_tokens[:, :-1]`` predicts ``action_tokens[:, 1:]``).
        """
        B, L, Dm = emb.shape
        bos = self.bos_embed.view(1, 1, Dm).expand(B, 1, Dm)
        return torch.cat([bos, emb[:, :-1, :]], dim=1)

    def build_soft_inputs(self, q_list_detached: List[torch.Tensor]) -> torch.Tensor:
        """Expected (soft) code embeddings from the DETACHED posterior.

        ``q_list_detached``: list over d of ``[B, L, L_d]`` (already detached).
        Returns input embeddings ``[B, L, n_emb]`` for ``[BOS, z_0..z_{L-2}]``.

        The detachment is load-bearing: ``q -> p`` must couple through the rate
        loss only, never through the teacher-forcing input path.
        """
        emb = None
        for d, qd in enumerate(q_list_detached):
            # qd: [B, L, L_d] @ [L_d, n_emb] -> [B, L, n_emb]
            contrib = qd @ self.dim_embed[d].weight
            emb = contrib if emb is None else emb + contrib
        return self._shift_with_bos(emb)

    def build_hard_inputs(self, ids: torch.Tensor) -> torch.Tensor:
        """Hard (argmax) code embeddings for the final hard-token finetune.

        ``ids``: ``[B, L, D]`` long level indices (detached). Returns input
        embeddings ``[B, L, n_emb]`` for ``[BOS, z_0..z_{L-2}]`` using the
        embedding of the single chosen code per dim — matching inference.
        """
        emb = None
        for d in range(self.n_dims):
            contrib = self.dim_embed[d](ids[..., d])
            emb = contrib if emb is None else emb + contrib
        return self._shift_with_bos(emb)

    # ── Factorized forward ──────────────────────────────────────────────────

    def forward_factorized(
        self,
        input_embeds: torch.Tensor,
        cond: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the transformer body on pre-built input embeddings.

        Args:
            input_embeds: ``[B, L, n_emb]`` already includes the BOS shift.
            cond: ``[B, T_cond, cond_dim]`` Past2Next condition.

        Returns:
            ``logp_list``: list over d of ``[B, L, L_d]`` **log-probabilities**
            (``log_softmax`` applied per dim).
        """
        T_tok = input_embeds.shape[1]
        T_cond = cond.shape[1]

        # Token-stream positional embedding (no flat tok_emb lookup here).
        pos_emb = self.tok_pos_emb[:, :T_tok, :]
        x = self.drop(input_embeds + pos_emb)

        # Condition processing -> cross-attention memory (identical to parent).
        cond_emb = self.cond_emb(cond)
        cond_pos_emb = self.cond_pos_emb[:, :T_cond, :]
        memory = self.drop(cond_emb + cond_pos_emb)
        memory = self.encoder(memory)

        # Causal decoding (CausalSelfAttention auto-applies causal mask when
        # layer_past is None and T > 1).
        for block in self.blocks:
            x, _ = block(x, memory)

        x = self.ln_f(x)

        logp_list = []
        for d in range(self.n_dims):
            logits_d = self.dim_head[d](x)              # [B, L, L_d]
            logp_list.append(F.log_softmax(logits_d.float(), dim=-1))
        return logp_list

    # ── Factorized generation (inference) ───────────────────────────────────

    @torch.no_grad()
    def generate_factorized(
        self,
        cond: torch.Tensor,
        n_positions: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """Autoregressively sample per-dim level ids for L register positions.

        Re-runs the (causal) body each step over the growing prefix — L is small
        (8), so this is cheap and avoids KV-cache bookkeeping for the factorized
        head. Matches the teacher-forcing alignment: start from ``[BOS]``, predict
        ``z_0`` at the last position, append ``emb(z_0)``, predict ``z_1``, etc.

        Returns:
            ``ids``: ``[B, n_positions, D]`` long level indices.
        """
        B = cond.shape[0]
        device = cond.device
        n_emb = self.bos_embed.shape[0]

        emb_seq = self.bos_embed.view(1, 1, n_emb).expand(B, 1, n_emb).clone()
        all_ids = []
        for _ in range(n_positions):
            logp_list = self.forward_factorized(emb_seq, cond)  # each [B, t, L_d]
            step_ids = []
            step_emb = None
            for d in range(self.n_dims):
                logp_d = logp_list[d][:, -1, :]                 # [B, L_d]
                if hard or temperature <= 0:
                    idx = logp_d.argmax(dim=-1)                 # [B]
                else:
                    logits_d = logp_d / temperature
                    if top_k is not None:
                        k = min(top_k, logits_d.size(-1))
                        v, _ = torch.topk(logits_d, k)
                        logits_d = logits_d.masked_fill(
                            logits_d < v[:, [-1]], float("-inf")
                        )
                    probs = F.softmax(logits_d, dim=-1)
                    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
                step_ids.append(idx)
                contrib = self.dim_embed[d](idx)                # [B, n_emb]
                step_emb = contrib if step_emb is None else step_emb + contrib
            all_ids.append(torch.stack(step_ids, dim=-1))       # [B, D]
            emb_seq = torch.cat([emb_seq, step_emb.unsqueeze(1)], dim=1)

        return torch.stack(all_ids, dim=1)                      # [B, n_positions, D]

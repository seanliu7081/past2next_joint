"""
Path A: OATPolicy with Enriched Past + Distance-Weighted CE Loss.

Architecture is identical to OATPolicyWithEnrichedPast.
Only the training loss changes:
    total_loss = CE_loss + lambda * distance_weighted_CE_loss

No architecture changes. No inference changes.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.oat_policy_with_enriched_past import OATPolicyWithEnrichedPast
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.loss.ordinal_loss import distance_weighted_ce_loss


class OATPolicyEnrichedPastDistCE(OATPolicyWithEnrichedPast):
    """
    Same architecture as OATPolicyWithEnrichedPast but adds a
    distance-weighted CE loss that penalizes far-wrong predictions more.

    total_loss = CE + ordinal_lambda * dist_weighted_CE
    """

    def __init__(
        self,
        # All parent args (pass through unchanged)
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

        # Precompute codebook embeddings -> registered as buffer (no grad)
        codebook_size = action_tokenizer.quantizer.codebook_size
        codebook_emb = action_tokenizer.quantizer.indices_to_embedding(
            torch.arange(codebook_size)
        )  # [1000, 4]
        self.register_buffer("codebook_embeddings", codebook_emb)

        print(f"  ordinal_lambda={ordinal_lambda}, loss=CE + lambda*DistCE\n")

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
        action_tokens = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device),
            action_tokens,
        ], dim=1)

        # -- Forward AR model --
        logits = self.model(action_tokens[:, :-1], cond=cond)
        vocab_size = logits.size(-1)  # 1001

        targets = action_tokens[:, 1:]  # [B, 8]

        # -- Standard CE loss --
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        # -- Distance-weighted CE loss --
        # Slice off BOS logit column -> only first codebook_size logits
        codebook_size = self.codebook_embeddings.shape[0]  # 1000
        logits_no_bos = logits.reshape(-1, vocab_size)[:, :codebook_size]  # [B*8, 1000]
        targets_flat = targets.reshape(-1)  # [B*8]

        dist_loss = distance_weighted_ce_loss(
            logits_no_bos, targets_flat, self.codebook_embeddings
        )

        total_loss = ce_loss + self.ordinal_lambda * dist_loss
        return total_loss

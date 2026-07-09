"""Predictive-tokenizer OAT policy (rate-distortion coupled AR policy).

Turns the OAT joint + enriched-past policy into a *predictive tokenizer*: the
action tokenizer is trained not only to reconstruct future action chunks, but
also so that its codes are *easy for the AR backbone to predict* from the
Past2Next condition. The trainable formulation is learned lossy compression with
an autoregressive conditional entropy model:

    L = w_rec * D  +  beta * R  +  w_use * U

    D (distortion) = MSE(Dec(z_st), nsamp)                         # straight-through hard decode
    R (rate)       = (1/L)(1/D) sum_l sum_d  H(q^d_l, p^d_l)       # cross-entropy, FULLY COUPLED
    U (usage)      = (1/D) sum_d  sum_k qbar^d(k) log qbar^d(k)    # = -sum_d H(qbar^d), light insurance

The single most important correctness rule: the rate term R MUST be the
cross-entropy ``H(q, p) = -sum q*log p`` with gradients flowing to BOTH q
(tokenizer) and p (AR). It is NOT a symmetric ``KL(q||p)``; there is never a
``+ q*log q`` term in the coupled loss. Reconstruction (D) is the information
anchor that prevents the rate term from collapsing the posterior.

Everything about conditioning / inference structure (``_build_condition``, the
Past2Next condition shape, normalizer sharing, the obs encoder, acc/jerk and
raw-past projections) is inherited unchanged from the joint/enriched-past
policy. Only the quantization (factorized soft-FSQ), the AR head (factorized),
the teacher forcing (soft, detached), and the loss are new.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from oat.policy.oat_policy_with_enriched_past import OATPolicyWithEnrichedPast
from oat.policy.oat_policy_with_enriched_past_joint import OATPolicyWithEnrichedPastJoint
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.ar_factorized_head import FactorizedAutoregressiveModel


# ── Factorized soft-FSQ ─────────────────────────────────────────────────────


def make_grids(levels: List[int], device) -> List[torch.Tensor]:
    """Centered integer grids, one per FSQ dim: v_d = arange(L_d) - (L_d-1)/2."""
    return [
        torch.arange(Ld, device=device, dtype=torch.float32) - (Ld - 1) / 2.0
        for Ld in levels
    ]


def soft_fsq(
    h: torch.Tensor,
    levels: List[int],
    grids: List[torch.Tensor],
    tau: float,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Per-dimension categorical soft-FSQ with a straight-through hard decode.

    Args:
        h:      ``[B, L, D]`` pre-quant scalar projections (one per FSQ dim).
        levels: e.g. ``[8, 5, 5, 5, 5]``.
        grids:  list of ``[L_d]`` centered grid values (see ``make_grids``).
        tau:    temperature (scheduled).

    Returns:
        q_list: list over d of ``[B, L, L_d]`` soft categorical posterior q^d.
        z_st:   ``[B, L, D]`` straight-through latent (hard forward, soft backward).
        ids:    ``[B, L, D]`` long hard level indices (logging / inference parity).
    """
    h = h.float()
    B, L, D = h.shape
    q_list: List[torch.Tensor] = []
    z_soft_dims: List[torch.Tensor] = []
    z_hard_dims: List[torch.Tensor] = []
    id_dims: List[torch.Tensor] = []
    for d in range(D):
        Ld = levels[d]
        v = grids[d]                                            # [L_d]
        half = (Ld - 1) / 2.0
        # smoothly bound the pre-quant value into the grid's range
        u = half * torch.tanh(h[..., d])                       # [B, L] in [-half, +half]
        # soft assignment over the L_d grid points
        logits = -(u.unsqueeze(-1) - v).pow(2) / tau           # [B, L, L_d]
        qd = logits.softmax(dim=-1)                            # [B, L, L_d]
        u_soft = (qd * v).sum(-1)                              # [B, L] expected grid value
        idx = (u + half).round().clamp(0, Ld - 1).long()       # [B, L] nearest grid index
        u_hard = v[idx]                                        # [B, L] nearest grid value
        q_list.append(qd)
        z_soft_dims.append(u_soft)
        z_hard_dims.append(u_hard)
        id_dims.append(idx)
    z_soft = torch.stack(z_soft_dims, dim=-1)                  # [B, L, D]
    z_hard = torch.stack(z_hard_dims, dim=-1)                  # [B, L, D]
    z_st = (z_hard - z_soft).detach() + z_soft                 # ST: forward=hard, backward=soft
    ids = torch.stack(id_dims, dim=-1)                         # [B, L, D] long
    return q_list, z_st, ids


def ids_to_latent(ids: torch.Tensor, grids: List[torch.Tensor]) -> torch.Tensor:
    """Map per-dim level indices ``[B, L, D]`` to centered-grid latents ``[B, L, D]``."""
    dims = [grids[d][ids[..., d]] for d in range(ids.shape[-1])]
    return torch.stack(dims, dim=-1)


# ── Policy ──────────────────────────────────────────────────────────────────


class OATPolicyPredictiveTokenizer(OATPolicyWithEnrichedPastJoint):
    """Rate-distortion coupled predictive-tokenizer policy.

    Reuses the joint/enriched-past policy's obs encoder, acc/jerk + raw-past
    projections, normalizer sharing, condition construction, and optimizer
    grouping. Replaces the flat AR head with a factorized one, the FSQ forward
    with ``soft_fsq``, and the recon+CE loss with the coupled rate-distortion
    objective above.
    """

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        past_n: int = 7,
        # policy model params
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        # rate-distortion loss params
        w_rec: float = 1.0,
        w_use: float = 1e-2,
        beta: float = 0.0,
        free_bits: float = 0.05,
        tau_start: float = 2.0,
        tau_end: float = 0.25,
        ar_head_mode: str = "factorized",
    ):
        # Build the joint policy normally (creates the obs encoder, the trainable
        # tokenizer, the acc/jerk/raw projections, normalizers, bos_id, and a flat
        # AR model). recon_loss_weight is set to w_rec; ce_loss_weight is unused
        # here (we override forward), pass 0.0.
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
            recon_loss_weight=w_rec,
            ce_loss_weight=0.0,
        )

        # FSQ levels define the per-dim categoricals (codebook geometry).
        self.levels: List[int] = action_tokenizer.quantizer._levels.tolist()
        assert action_tokenizer.encoder.head.dim_out == len(self.levels), (
            "encoder latent_dim must equal the number of FSQ levels: "
            f"{action_tokenizer.encoder.head.dim_out} vs {len(self.levels)}"
        )

        # Replace the flat AR model with a factorized one. The flat model built
        # by the parent is discarded (GC'd); we reuse the exact same hyperparams.
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        max_cond_len = n_obs_steps + self.N_EXPLICIT_FEATURES + past_n
        self.model = FactorizedAutoregressiveModel(
            vocab_size=codebook_size + 1,
            max_seq_len=latent_horizon + 1,
            max_cond_len=max_cond_len,
            cond_dim=self.obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
            levels=self.levels,
            ar_head_mode=ar_head_mode,
        )
        self.ar_head_mode = ar_head_mode

        # ── Rate-distortion knobs (static) ──────────────────────────────────
        self.w_rec = float(w_rec)
        self.w_use = float(w_use)
        self.free_bits = float(free_bits)

        # ── Schedule state (driven each epoch by the workspace) ─────────────
        self.beta = float(beta)
        self.tau = float(tau_start)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.hard_finetune = False

        print(
            f"{self.get_policy_name()} (predictive tokenizer) configured with\n"
            f"  levels       : {self.levels} (codebook {codebook_size})\n"
            f"  ar_head_mode : {ar_head_mode}\n"
            f"  w_rec={self.w_rec}  w_use={self.w_use}  free_bits={self.free_bits}\n"
            f"  beta(init)={self.beta}  tau_start={self.tau_start}  tau_end={self.tau_end}\n"
        )

    def get_policy_name(self):
        base_name = "oat_predtok_enriched_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    # ── Schedule hooks (called per-epoch by the workspace) ──────────────────

    def set_beta(self, beta: float):
        self.beta = float(beta)

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def set_hard_finetune(self, flag: bool):
        self.hard_finetune = bool(flag)

    # ── Tokenizer-only reconstruction diagnostic ────────────────────────────

    @torch.no_grad()
    def reconstruct_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encoder -> soft_fsq (hard ids) -> decoder -> unnormalize.

        The centered-grid decode path the tokenizer is actually trained on (NOT
        the FSQ quantizer's normalized codes, which this decoder never sees).
        """
        tok = self.action_tokenizer
        nsamp = tok.normalizer["action"].normalize(actions)
        h = tok.encoder(nsamp)
        grids = make_grids(self.levels, h.device)
        _, _, ids = soft_fsq(h, self.levels, grids, self.tau)
        z = ids_to_latent(ids, grids)
        nrecon = tok.decoder(z)
        return tok.normalizer["action"].unnormalize(nrecon)

    # ── Training forward (rate-distortion coupled) ──────────────────────────

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        actions = batch["action"]                       # [B, 16, 7]
        past = batch["past_action"]                     # [B, 7, 7]
        device = actions.device
        tok = self.action_tokenizer

        nsamp = tok.normalizer["action"].normalize(actions)
        h = tok.encoder(nsamp)                          # [B, L=8, D=5]

        grids = make_grids(self.levels, device)
        q_list, z_st, ids = soft_fsq(h, self.levels, grids, self.tau)

        # ── distortion (straight-through hard decode) ───────────────────────
        recon = tok.decoder(z_st)                       # nested_dropout active (train mode)
        D = F.mse_loss(recon, nsamp)

        # ── Past2Next condition (unchanged) ─────────────────────────────────
        features = self.obs_encoder(batch["obs"])       # [B, To, d]
        cond = self._build_condition(features, past)    # [B, 11, d]

        # ── soft (or hard, in finetune), DETACHED teacher forcing ───────────
        if self.hard_finetune:
            in_emb = self.model.build_hard_inputs(ids.detach())
        else:
            in_emb = self.model.build_soft_inputs([q.detach() for q in q_list])

        logp_list = self.model.forward_factorized(in_emb, cond)  # list of [B, L, L_d] log-probs

        # ── rate = cross-entropy H(q, p), fully coupled, per-dim free-bits ──
        # H(q^d, p^d) couples grads to BOTH q^d (tokenizer encoder) and p^d (AR).
        # No `+ q*log q`; this is cross-entropy, not symmetric KL.
        R_dims = []
        ce_raw = []
        for qd, logp in zip(q_list, logp_list):
            ce_d = -(qd * logp).sum(-1).mean()          # H(q^d, p^d)
            ce_raw.append(ce_d.detach())
            R_dims.append(torch.clamp(ce_d, min=self.free_bits))  # per-dim free-bits floor
        R = torch.stack(R_dims).mean()

        # ── light marginal usage (per-dim collapse insurance) ───────────────
        # U = sum_d (qbar^d . log qbar^d) = -sum_d H(qbar^d); minimizing it
        # maximizes marginal entropy so no axis goes unused. No `+ q*log q` on
        # the *conditional* q here — this is the marginal over (B, L).
        U = h.new_zeros(())
        for qd in q_list:
            qbar = qd.float().mean(dim=(0, 1))          # [L_d]
            U = U + (qbar * qbar.clamp_min(1e-8).log()).sum()
        U = U / len(q_list)

        loss = self.w_rec * D + self.beta * R + self.w_use * U

        # ── cheap diagnostics (no_grad) ─────────────────────────────────────
        with torch.no_grad():
            rate_per_dim = torch.stack(ce_raw)          # [D] pre-clamp H(q^d, p^d)
            # hard-token CE: AR NLL of the argmax ids (parity with soft rate)
            hard_ce = h.new_zeros(())
            acc = h.new_zeros(())
            for d, logp in enumerate(logp_list):
                tgt = ids[..., d]                       # [B, L]
                gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, L]
                hard_ce = hard_ce - gathered.mean()
                acc = acc + (logp.argmax(-1) == tgt).float().mean()
            hard_ce = hard_ce / len(logp_list)
            ar_acc = acc / len(logp_list)
            # per-dim marginal perplexity (collapse detector)
            ppl = []
            for qd in q_list:
                qbar = qd.float().mean(dim=(0, 1)).clamp_min(1e-8)
                ent = -(qbar * qbar.log()).sum()
                ppl.append(torch.exp(ent))
            usage_ppl = torch.stack(ppl).mean()

        return {
            "loss": loss,
            "recon": D.detach(),
            "rate": R.detach(),
            "rate_per_dim": rate_per_dim,
            "usage": U.detach(),
            "hard_ce": hard_ce,
            "ar_acc": ar_acc,
            "usage_ppl": usage_ppl,
            "tau": torch.as_tensor(float(self.tau), device=device),
            "beta": torch.as_tensor(float(self.beta), device=device),
            # back-compat aliases so generic joint-workspace logging still works
            "recon_loss": D.detach(),
            "ce_loss": R.detach(),
        }

    # ── Inference (factorized receding-horizon control) ─────────────────────

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_k_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_k_tokens is None:
            use_k_tokens = self.max_seq_len
        else:
            use_k_tokens = min(use_k_tokens, self.max_seq_len)
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        features = self.obs_encoder(obs_dict)           # (B, To, d)
        B = features.shape[0]
        device = self.device

        # ── get or initialise past buffer ───────────────────────────────────
        if (
            self._past_buffer is None
            or self._past_buffer.shape[0] != B
            or self._past_buffer.device != device
        ):
            self._past_buffer = torch.zeros(
                B, self.past_n, self.action_dim,
                device=device, dtype=features.dtype,
            )

        cond = self._build_condition(features, self._past_buffer)

        # ── factorized autoregressive generation ────────────────────────────
        # At inference use argmax iff we trained to the hard-finetune regime;
        # otherwise sample with the configured temperature / top-k.
        grids = make_grids(self.levels, device)
        ids = self.model.generate_factorized(
            cond,
            n_positions=use_k_tokens,
            temperature=temperature,
            top_k=topk,
            hard=self.hard_finetune,
        )                                                # [B, L, D]

        # decode the centered-grid latents -> continuous actions
        with torch.no_grad():
            z = ids_to_latent(ids, grids)               # [B, L, D]
            nrecon = self.action_tokenizer.decoder(z)   # eval: nested_dropout inert
            action_pred = self.action_tokenizer.normalizer["action"].unnormalize(nrecon)

        action = action_pred[:, : self.n_action_steps]

        # ── update past buffer (identical to enriched-past parent) ──────────
        n_exec = self.n_action_steps
        past_n = self.past_n
        if n_exec >= past_n:
            self._past_buffer = action_pred[:, n_exec - past_n: n_exec].detach().clone()
        else:
            self._past_buffer = torch.cat([
                self._past_buffer[:, n_exec:],
                action_pred[:, :n_exec].detach().clone(),
            ], dim=1)

        return {
            "action": action,
            "action_pred": action_pred,
        }

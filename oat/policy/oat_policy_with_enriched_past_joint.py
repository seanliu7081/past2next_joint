import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.policy.oat_policy_with_enriched_past import OATPolicyWithEnrichedPast
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.common.normalizer import LinearNormalizer


class OATPolicyWithEnrichedPastJoint(OATPolicyWithEnrichedPast):
    """End-to-end co-training of the action tokenizer + the enriched-past AR backbone.

    This is the joint-training counterpart of `OATPolicyWithEnrichedPast`, exactly
    as `OATPolicyJoint` is the joint counterpart of `OATPolicy`.  Unlike the parent
    (which loads a *frozen* tokenizer from a checkpoint), this policy owns a
    *trainable* `OATTok` built from scratch and optimizes it jointly with the
    backbone + obs encoder + enriched-past projections.  The combined loss is

        loss = recon_loss_weight * mse(decode(quant(encode(a))), a)      # trains tokenizer
             + current_ce_weight  * CE(backbone(tokens | cond), tokens)  # trains backbone/obs/proj

    where `cond` is the enriched condition sequence [obs, acc, jerk, raw_past].

    As in `OATPolicyJoint`, the cross-entropy targets are the *detached* discrete FSQ
    token indices, so the CE loss never backprops into the tokenizer (FSQ indices are
    non-differentiable).  The tokenizer learns solely from the reconstruction loss
    (STE through `quant`).

    `current_ce_weight` is a runtime-toggleable copy of `ce_loss_weight` used by the
    workspace (`TrainOATJointWorkspace`) to implement a tokenizer-only warmup phase
    (set it to 0.0 during warmup).

    Everything related to conditioning / inference (`_build_condition`,
    `predict_action`, `reset`, the inference-time past buffer, the explicit acc/jerk
    and raw-past projections) is inherited unchanged from `OATPolicyWithEnrichedPast`.
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
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        # joint training params
        recon_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
    ):
        # Skip OATPolicyWithEnrichedPast.__init__ (it freezes the tokenizer); init the
        # nn.Module base directly so the tokenizer stays trainable for joint training.
        super(OATPolicyWithEnrichedPast, self).__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta["obs"].items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            _type = attr["type"]
            if _type in modalities:
                obs_ports.append(key)


        # ── Explicit feature projections (independent, different scales) ──
        acc_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )
        jerk_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Raw past action projection (shared across all past_n steps) ──
        raw_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Action normalizer (for the enriched-past conditioning path) ──
        action_normalizer = LinearNormalizer()

        # ── AR model ─────────────────────────────────────────────────────
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        max_cond_len = n_obs_steps + self.N_EXPLICIT_FEATURES + past_n

        model = AutoregressiveModel(
            vocab_size=codebook_size + 1,       # +1 for <BOS>
            max_seq_len=latent_horizon + 1,
            max_cond_len=max_cond_len,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )
        bos_id = codebook_size

        # ── Store everything ─────────────────────────────────────────────
        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.acc_proj = acc_proj
        self.jerk_proj = jerk_proj
        self.raw_proj = raw_proj
        self.action_normalizer = action_normalizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.past_n = past_n
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        # joint training weights
        self.recon_loss_weight = recon_loss_weight
        self.ce_loss_weight = ce_loss_weight
        # runtime-toggleable CE weight (set to 0.0 during tokenizer-only warmup)
        self.current_ce_weight = ce_loss_weight

        # Inference-time past buffer
        self._past_buffer: Optional[torch.Tensor] = None

        # ── Report ───────────────────────────────────────────────────────
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs = sum(
            p.numel() for p in obs_encoder.parameters() if p.requires_grad
        )
        obs_trainable_ratio = num_trainable_obs / num_obs_params
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok = sum(
            p.numel() for p in action_tokenizer.parameters() if p.requires_grad
        )
        tok_trainable_ratio = num_trainable_tok / num_tok_params
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        model_trainable_ratio = num_trainable_model / num_model_params
        num_explicit_params = (
            sum(p.numel() for p in acc_proj.parameters())
            + sum(p.numel() for p in jerk_proj.parameters())
        )
        num_raw_params = sum(p.numel() for p in raw_proj.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc      : {num_obs_params / 1e6:.1f}M "
            f"({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok      : {num_tok_params / 1e6:.1f}M "
            f"({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy       : {num_model_params / 1e6:.1f}M "
            f"({model_trainable_ratio:.5%} trainable)\n"
            f"  explicit proj: {num_explicit_params / 1e3:.1f}K (acc + jerk)\n"
            f"  raw proj     : {num_raw_params / 1e3:.1f}K (shared)\n"
            f"  cond_len={n_obs_steps}+{self.N_EXPLICIT_FEATURES}+{past_n}"
            f"={max_cond_len}\n"
        )

    # ── BasePolicy interface overrides ──────────────────────────────────────

    def get_policy_name(self):
        base_name = "oat_joint_enriched_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    def set_normalizer(self, normalizer):
        # Obs encoder, the (trainable) tokenizer, and the enriched-past conditioning
        # path all share the dataset normalizer so the recon path (normalize) and the
        # detokenize path (unnormalize) stay consistent.
        self.obs_encoder.set_normalizer(normalizer)
        self.action_tokenizer.set_normalizer(normalizer)
        self.action_normalizer.load_state_dict(normalizer.state_dict())

    def set_ce_weight(self, weight: float):
        """Runtime toggle for the warmup phase (0.0 = tokenizer-only)."""
        self.current_ce_weight = weight

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        tokenizer_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay for 2D params only, across backbone (+ enriched-past
        projections) / obs enc / tokenizer."""
        encoder_decay, encoder_nodecay = [], []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_decay if param.dim() >= 2 else encoder_nodecay).append(param)

        policy_decay, policy_nodecay = [], []
        policy_modules = [self.model, self.acc_proj, self.jerk_proj, self.raw_proj]
        for module in policy_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (policy_decay if param.dim() >= 2 else policy_nodecay).append(param)

        tokenizer_decay, tokenizer_nodecay = [], []
        for name, param in self.action_tokenizer.named_parameters():
            if not param.requires_grad:
                continue
            (tokenizer_decay if param.dim() >= 2 else tokenizer_nodecay).append(param)

        optim_groups = [
            {"params": policy_decay,     "lr": policy_lr,    "weight_decay": weight_decay},
            {"params": policy_nodecay,   "lr": policy_lr,    "weight_decay": 0.0},
            {"params": encoder_decay,    "lr": obs_enc_lr,   "weight_decay": weight_decay},
            {"params": encoder_nodecay,  "lr": obs_enc_lr,   "weight_decay": 0.0},
            {"params": tokenizer_decay,  "lr": tokenizer_lr, "weight_decay": weight_decay},
            {"params": tokenizer_nodecay, "lr": tokenizer_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, betas=betas)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        action = batch["action"]
        B = action.shape[0]
        device = action.device

        tok = self.action_tokenizer

        # --- tokenizer reconstruction path (trainable: needs grad, NO inference_mode) ---
        nsamples = tok.normalizer["action"].normalize(action)
        latents = tok.encoder(nsamples)
        quant, tokens = tok.quantizer(latents)   # quant: STE-differentiable, tokens: long indices
        recons = tok.decoder(quant)              # nested_dropout active in train mode (desired)
        recon_loss = F.mse_loss(recons, nsamples)

        # --- CE targets: detached discrete tokens (no backprop into tokenizer) ---
        action_tokens = tokens.detach().long()   # [B, latent_horizon]

        # --- obs + enriched-past conditioning ---
        features = self.obs_encoder(batch["obs"])        # [B, To, d]
        past_actions = batch["past_action"]              # [B, past_n, action_dim]
        cond = self._build_condition(features, past_actions)  # [B, To + 2 + past_n, d]

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id,
                dtype=torch.long, device=device,
            ),
            action_tokens,
        ], dim=1)                                        # [B, latent_horizon + 1]

        logits = self.model(action_tokens[:, :-1], cond=cond)
        vocab_size = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),              # (B*T, vocab_size)
            action_tokens[:, 1:].reshape(-1),            # (B*T,)
        )

        total = self.recon_loss_weight * recon_loss + self.current_ce_weight * ce_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "ce_loss": ce_loss,
        }

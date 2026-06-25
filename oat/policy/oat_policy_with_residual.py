import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
# from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.residual_tokenizer import ResidualOATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.common.normalizer import LinearNormalizer


class OATPolicyWithResidual(BasePolicy):
    """
    OATPolicy variant that conditions on:
      1. Full 7-step raw past actions (preserves temporal structure)
      2. Explicit acceleration and jerk features (provides information
         that observations lack and that requires cross-timestep computation)

    Design rationale (from theoretical analysis v2):
      - Observations provide: position (eef_pos) + coarse velocity (2-frame diff)
      - Raw past actions provide: exact velocity, temporal patterns,
        task progress, command inertia
      - Explicit acc/jerk provide: higher-order derivative info that
        obs cannot access and that the model would otherwise need to
        learn to extract via cross-timestep differencing
    """

    N_EXPLICIT_FEATURES = 2  # acc, jerk

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: ResidualOATTok,
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
    ):
        super().__init__()

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

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

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

        # ── Raw past action projection (shared across all 7 steps) ───────
        raw_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Action normalizer ────────────────────────────────────────────
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

        # Inference-time past buffer
        self._past_buffer: Optional[torch.Tensor] = None

        # ── Report ───────────────────────────────────────────────────────
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs = sum(
            p.numel() for p in obs_encoder.parameters() if p.requires_grad
        )
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        num_explicit_params = (
            sum(p.numel() for p in acc_proj.parameters())
            + sum(p.numel() for p in jerk_proj.parameters())
        )
        num_raw_params = sum(p.numel() for p in raw_proj.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc     : {num_obs_params / 1e6:.1f}M "
            f"({num_trainable_obs / num_obs_params:.5%} trainable)\n"
            f"  act tok     : {num_tok_params / 1e6:.1f}M (frozen)\n"
            f"  policy      : {num_model_params / 1e6:.1f}M "
            f"({num_trainable_model / num_model_params:.5%} trainable)\n"
            f"  explicit proj: {num_explicit_params / 1e3:.1f}K "
            f"(acc + jerk)\n"
            f"  raw proj    : {num_raw_params / 1e3:.1f}K (shared)\n"
            f"  cond_len={n_obs_steps}+{self.N_EXPLICIT_FEATURES}+{past_n}"
            f"={max_cond_len}\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = "oatpolicy_enriched_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    def create_dummy_observation(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device,
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        self.action_normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        encoder_decay, encoder_nodecay = [], []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_decay if param.dim() >= 2 else encoder_nodecay).append(param)

        policy_decay, policy_nodecay = [], []
        policy_modules = [
            self.model, self.acc_proj, self.jerk_proj, self.raw_proj
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

    def reset(self):
        """Called by env_runner at the start of each rollout episode."""
        self._past_buffer = None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_condition(
        self,
        obs_features: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the full condition sequence from obs features and past actions.

        Args:
            obs_features: (B, To, obs_feature_dim)
            past_actions: (B, past_n, action_dim) raw (unnormalized) actions
                          ordered as [a_{t-7}, a_{t-6}, ..., a_{t-1}]

        Returns:
            (B, To + 2 + past_n, obs_feature_dim)
        """
        # Normalize all past actions
        norm_past = self.action_normalizer["action"].normalize(past_actions)

        # ── Explicit features from near 3 steps ──────────────────────────
        a_t1 = norm_past[:, -1]   # a_{t-1}
        a_t2 = norm_past[:, -2]   # a_{t-2}
        a_t3 = norm_past[:, -3]   # a_{t-3}

        acc  = a_t1 - a_t2                       # [B, 7] acceleration-level
        jerk = a_t1 - 2.0 * a_t2 + a_t3         # [B, 7] jerk-level

        acc_feat  = self.acc_proj(acc)            # [B, d]
        jerk_feat = self.jerk_proj(jerk)          # [B, d]

        explicit = torch.stack([acc_feat, jerk_feat], dim=1)  # [B, 2, d]

        # ── Raw history (all 7 steps, shared projection) ─────────────────
        raw_feat = self.raw_proj(norm_past)       # [B, 7, d]

        # ── Concatenate: [obs, explicit, raw] ────────────────────────────
        return torch.cat([obs_features, explicit, raw_feat], dim=1)

    # ── Inference ───────────────────────────────────────────────────────────

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

        # encode observation
        features = self.obs_encoder(obs_dict)       # (B, To, d)
        B = features.shape[0]

        # ── get or initialise past buffer ─────────────────────────────────
        if (
            self._past_buffer is None
            or self._past_buffer.shape[0] != B
            or self._past_buffer.device != self.device
        ):
            self._past_buffer = torch.zeros(
                B, self.past_n, self.action_dim,
                device=self.device, dtype=features.dtype,
            )

        # ── build extended condition ──────────────────────────────────────
        cond = self._build_condition(features, self._past_buffer)

        # ── autoregressive generation ─────────────────────────────────────
        action_tokens = torch.full(
            (B, 1), self.bos_id,
            dtype=torch.long, device=self.device,
        )
        action_tokens = self.model.generate(
            action_tokens,
            cond=cond,
            max_new_tokens=use_k_tokens,
            temperature=temperature,
            top_k=topk,
        )[:, 1:]    # drop <BOS>
        action_tokens = action_tokens.clamp(0, self.bos_id - 1)

        # decode tokens -> continuous actions
        with torch.inference_mode():
            baseline = self._past_buffer[:, -1, :]      # (B, action_dim) = a_t
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
                baseline=baseline,
            )

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        # ── update past buffer ────────────────────────────────────────────
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

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        # tokenize ground-truth actions (frozen tokenizer)
        with torch.no_grad():
            baseline = batch["past_action"][:, -1, :]   # (B, action_dim) = a_t
            action_tokens = self.action_tokenizer.tokenize(batch["action"], baseline)

        B = batch["action"].shape[0]
        device = batch["action"].device

        # encode observation
        features = self.obs_encoder(batch["obs"])       # (B, To, d)

        # ── build extended condition ──────────────────────────────────────
        past_actions = batch["past_action"]              # (B, past_n, action_dim)
        cond = self._build_condition(features, past_actions)

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id,
                dtype=torch.long, device=device,
            ),
            action_tokens,
        ], dim=1)

        # forward model
        logits = self.model(action_tokens[:, :-1], cond=cond)

        # compute loss
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            action_tokens[:, 1:].reshape(-1),
        )
        return loss
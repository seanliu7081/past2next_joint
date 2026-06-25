import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.common.normalizer import LinearNormalizer


class OATPolicyWithKinematics(BasePolicy):
    """
    OATPolicy variant that conditions on explicit kinematic features
    computed from past actions: position, velocity, acceleration.

    Instead of feeding raw past actions into a shared MLP (as in
    OATPolicyWithPast), we:
      1. Compute finite differences: pos, vel, acc from 3 past steps
      2. Project each through a SEPARATE MLP (to respect scale differences)
      3. Concatenate with obs features as condition for the AR model

    Condition sequence:
        cond = [obs_1, obs_2, pos_feat, vel_feat, acc_feat]
               |-- To=2 --|------------ 3 --------------|
               → [B, 5, obs_feature_dim]
    """

    # Number of past steps needed for kinematics (pos + vel + acc)
    KINEMATICS_PAST_N = 3
    # Number of kinematic features (pos, vel, acc)
    N_KIN_FEATURES = 3

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
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

        # ── Kinematic feature projections (separate per order) ────────────
        # Each projects action_dim → obs_feature_dim independently,
        # respecting the different scales of pos, vel, acc.
        pos_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )
        vel_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )
        acc_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Action normalizer ────────────────────────────────────────────
        action_normalizer = LinearNormalizer()

        # ── AR model (cond_len = n_obs_steps + 3 kinematic features) ─────
        codebook_size = action_tokenizer.quantizer.codebook_size
        latent_horizon = action_tokenizer.latent_horizon
        max_cond_len = n_obs_steps + self.N_KIN_FEATURES

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
        self.pos_proj = pos_proj
        self.vel_proj = vel_proj
        self.acc_proj = acc_proj
        self.action_normalizer = action_normalizer
        self.model = model
        self.max_seq_len = latent_horizon
        self.bos_id = bos_id
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
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
        num_kin_params = (
            sum(p.numel() for p in pos_proj.parameters())
            + sum(p.numel() for p in vel_proj.parameters())
            + sum(p.numel() for p in acc_proj.parameters())
        )
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc : {num_obs_params / 1e6:.1f}M "
            f"({num_trainable_obs / num_obs_params:.5%} trainable)\n"
            f"  act tok : {num_tok_params / 1e6:.1f}M (frozen)\n"
            f"  policy  : {num_model_params / 1e6:.1f}M "
            f"({num_trainable_model / num_model_params:.5%} trainable)\n"
            f"  kin proj: {num_kin_params / 1e3:.1f}K "
            f"(pos={sum(p.numel() for p in pos_proj.parameters())}, "
            f"vel={sum(p.numel() for p in vel_proj.parameters())}, "
            f"acc={sum(p.numel() for p in acc_proj.parameters())})\n"
            f"  cond_len={n_obs_steps}+{self.N_KIN_FEATURES}={max_cond_len}\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = "oatpolicy_kin_"
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
        policy_modules = [self.model, self.pos_proj, self.vel_proj, self.acc_proj]
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

    def _compute_and_encode_kinematics(
        self, past_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute finite-difference kinematic features from 3 past actions,
        then project each through its own MLP.

        Args:
            past_actions: (B, 3, action_dim) raw (unnormalized) actions
                          ordered as [a_{t-3}, a_{t-2}, a_{t-1}]

        Returns:
            (B, 3, obs_feature_dim) — [pos_feat, vel_feat, acc_feat]
        """
        # Normalize
        norm_past = self.action_normalizer["action"].normalize(past_actions)

        # Extract individual timesteps
        a_t3 = norm_past[:, 0]  # a_{t-3}   [B, action_dim]
        a_t2 = norm_past[:, 1]  # a_{t-2}   [B, action_dim]
        a_t1 = norm_past[:, 2]  # a_{t-1}   [B, action_dim]

        # Finite differences
        pos = a_t1                                # [B, 7]  position
        vel = a_t1 - a_t2                         # [B, 7]  velocity (1st order)
        acc = a_t1 - 2.0 * a_t2 + a_t3           # [B, 7]  acceleration (2nd order)

        # Separate projections
        pos_feat = self.pos_proj(pos)             # [B, d]
        vel_feat = self.vel_proj(vel)             # [B, d]
        acc_feat = self.acc_proj(acc)             # [B, d]

        # Stack into condition tokens
        kin_features = torch.stack([pos_feat, vel_feat, acc_feat], dim=1)  # [B, 3, d]
        return kin_features

    def _build_condition(
        self,
        obs_features: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate obs features and kinematic features.

        Args:
            obs_features: (B, To, obs_feature_dim)
            past_actions: (B, 3, action_dim) raw actions

        Returns:
            (B, To + 3, obs_feature_dim)
        """
        kin_features = self._compute_and_encode_kinematics(past_actions)
        return torch.cat([obs_features, kin_features], dim=1)

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
                B, self.KINEMATICS_PAST_N, self.action_dim,
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

        # decode tokens → continuous actions
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
            )

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        # ── update past buffer ────────────────────────────────────────────
        # We need the last 3 executed steps for next call's kinematics
        n_exec = self.n_action_steps
        past_n = self.KINEMATICS_PAST_N

        if n_exec >= past_n:
            # Take the last 3 steps of the executed chunk
            self._past_buffer = action_pred[:, n_exec - past_n: n_exec].detach().clone()
        else:
            # Slide: keep old tail + append new executed steps
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
            action_tokens = self.action_tokenizer.tokenize(batch["action"])

        B = batch["action"].shape[0]
        device = batch["action"].device

        # encode observation
        features = self.obs_encoder(batch["obs"])       # (B, To, d)

        # ── build extended condition with kinematic features ──────────────
        past_actions = batch["past_action"]              # (B, 3, action_dim)
        cond = self._build_condition(features, past_actions)  # (B, To+3, d)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.autoregressive.transformer_cache import AutoregressiveModel
from oat.model.common.normalizer import LinearNormalizer


class OATPolicyWithRoundPast(BasePolicy):
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

    Plan C: Token Round-Trip Augmentation
    ----------------------------------------
    At training time, past_action comes from the dataset (ground truth).
    At inference time, past_action comes from the policy's own predictions
    (which have quantization error baked in).

    To close this gap, during training we randomly replace ground-truth
    past_actions with their tokenizer round-trip reconstructions:
        past_action → pad to horizon → encode → FSQ → decode → crop to past_n
    This exposes the model to quantization-level noise that matches the
    actual error distribution at inference time.

    The round-trip uses the frozen tokenizer, so no extra parameters are
    introduced and the operation is done under torch.no_grad().
    """

    N_EXPLICIT_FEATURES = 2  # acc, jerk

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
        # ── Plan C augmentation params ──────────────────────────────────
        roundtrip_p: float = 0.3,
        # Probability of replacing past_action with its round-trip reconstruction.
        # 0.0 = disabled (original behaviour), 0.3 = recommended starting point.
        roundtrip_warmup_steps: int = 500,
        # Number of training steps before round-trip augmentation is activated.
        # Allows the model to first learn from clean data before seeing noisy past.
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

        # ── Tokenizer horizon (needed for round-trip padding) ────────────
        # RegisterEncoder.pos_emb is built for sample_horizon timesteps.
        # past_n is typically smaller, so we must pad before autoencoding.
        tok_horizon = action_tokenizer.decoder.sample_horizon  # e.g. 32

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

        # ── Per-step positional embedding for raw past ───────────────────
        # Fix: shared raw_proj has no notion of which timestep each action
        # came from. A learned embedding gives the model temporal ordering.
        past_pos_emb = nn.Embedding(past_n, obs_feature_dim)

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
        self.tok_horizon = tok_horizon
        self.acc_proj = acc_proj
        self.jerk_proj = jerk_proj
        self.raw_proj = raw_proj
        self.past_pos_emb = past_pos_emb
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

        # Plan C params
        self.roundtrip_p = roundtrip_p
        self.roundtrip_warmup_steps = roundtrip_warmup_steps
        self._train_step = 0  # incremented in forward()

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
        num_raw_params = (
            sum(p.numel() for p in raw_proj.parameters())
            + sum(p.numel() for p in past_pos_emb.parameters())
        )
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc      : {num_obs_params / 1e6:.1f}M "
            f"({num_trainable_obs / num_obs_params:.5%} trainable)\n"
            f"  act tok      : {num_tok_params / 1e6:.1f}M (frozen)\n"
            f"  policy       : {num_model_params / 1e6:.1f}M "
            f"({num_trainable_model / num_model_params:.5%} trainable)\n"
            f"  explicit proj: {num_explicit_params / 1e3:.1f}K (acc + jerk)\n"
            f"  raw proj     : {num_raw_params / 1e3:.1f}K (shared + pos_emb)\n"
            f"  cond_len={n_obs_steps}+{self.N_EXPLICIT_FEATURES}+{past_n}"
            f"={max_cond_len}\n"
            f"  roundtrip_p={roundtrip_p}, warmup={roundtrip_warmup_steps} steps\n"
            f"  tok_horizon={tok_horizon} (pad target for round-trip)\n"
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
            self.model, self.acc_proj, self.jerk_proj,
            self.raw_proj, self.past_pos_emb,
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

    # ── Plan C: Token Round-Trip Augmentation ───────────────────────────────

    @torch.no_grad()
    def _roundtrip_past_actions(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Simulate the quantization noise that past_actions would have at
        inference time by running them through the frozen tokenizer.

        Problem: tokenizer expects (B, tok_horizon=32, action_dim),
                 but past_actions is (B, past_n=7, action_dim).

        Solution:
          1. Zero-pad past_actions on the RIGHT to tok_horizon
             (zeros = stationary, a natural continuation)
          2. Run autoencode on the padded sequence
          3. Crop back to the first past_n steps

        The quantization artifacts that appear in the first past_n steps of
        the reconstruction are exactly the kind of noise the model will see
        at inference time when its own past predictions are fed back in.

        Args:
            past_actions: (B, past_n, action_dim) unnormalized

        Returns:
            (B, past_n, action_dim) unnormalized, with quantization noise
        """
        B, past_n, action_dim = past_actions.shape
        tok_horizon = self.tok_horizon

        # ── 1. Pad to tok_horizon ────────────────────────────────────────
        # Zero-padding on the right: shape (B, tok_horizon, action_dim)
        if past_n < tok_horizon:
            pad = torch.zeros(
                B, tok_horizon - past_n, action_dim,
                device=past_actions.device,
                dtype=past_actions.dtype,
            )
            padded = torch.cat([past_actions, pad], dim=1)
        else:
            # past_n >= tok_horizon: just truncate (shouldn't happen normally)
            padded = past_actions[:, :tok_horizon]

        # ── 2. Autoencode (tokenizer normalizes internally) ──────────────
        # autoencode: normalize → encode → FSQ → decode → unnormalize
        recons_padded = self.action_tokenizer.autoencode(padded)
        # shape: (B, tok_horizon, action_dim)

        # ── 3. Crop back to past_n ───────────────────────────────────────
        recons = recons_padded[:, :past_n]
        # shape: (B, past_n, action_dim)

        return recons

    def _maybe_apply_roundtrip(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Apply round-trip augmentation to past_actions with probability
        roundtrip_p, but only after warmup_steps have passed.

        Per-sample application: each sample in the batch independently
        decides whether to use round-trip or ground truth.

        Args:
            past_actions: (B, past_n, action_dim) ground truth

        Returns:
            (B, past_n, action_dim) augmented (some samples round-tripped)
        """
        if self.roundtrip_p <= 0.0:
            return past_actions

        if self._train_step < self.roundtrip_warmup_steps:
            return past_actions

        B = past_actions.shape[0]

        # Compute round-trip for the full batch (one forward pass)
        recons = self._roundtrip_past_actions(past_actions)

        # Per-sample mask: shape (B, 1, 1) for broadcasting
        # Each sample independently uses round-trip with probability roundtrip_p
        use_roundtrip = (
            torch.rand(B, 1, 1, device=past_actions.device) < self.roundtrip_p
        )

        augmented = torch.where(use_roundtrip, recons, past_actions)
        return augmented

    # ── Condition builder ───────────────────────────────────────────────────

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
                          ordered as [a_{t-6}, ..., a_{t-1}, a_t]
                          (past_action[-1] = a_t per dataset alignment)

        Returns:
            (B, To + 2 + past_n, obs_feature_dim)
        """
        # Normalize all past actions
        norm_past = self.action_normalizer["action"].normalize(past_actions)

        # ── Explicit features from near 3 steps ──────────────────────────
        # delta action space: 1st diff = acc, 2nd diff = jerk (correct)
        a_t1 = norm_past[:, -1]    # a_t
        a_t2 = norm_past[:, -2]    # a_{t-1}
        a_t3 = norm_past[:, -3]    # a_{t-2}

        acc  = a_t1 - a_t2                       # [B, action_dim]
        jerk = a_t1 - 2.0 * a_t2 + a_t3         # [B, action_dim]

        acc_feat  = self.acc_proj(acc)            # [B, d]
        jerk_feat = self.jerk_proj(jerk)          # [B, d]

        explicit = torch.stack([acc_feat, jerk_feat], dim=1)  # [B, 2, d]

        # ── Raw history (all past_n steps, shared projection + pos emb) ──
        raw_feat = self.raw_proj(norm_past)       # [B, past_n, d]

        # Add per-step positional embedding so the model knows
        # which timestep each past action came from
        past_positions = torch.arange(
            self.past_n, device=raw_feat.device
        )                                         # [past_n]
        pos_emb = self.past_pos_emb(past_positions)  # [past_n, d]
        raw_feat = raw_feat + pos_emb.unsqueeze(0)   # [B, past_n, d]

        # ── Concatenate: [obs | acc, jerk | raw_past] ────────────────────
        return torch.cat([obs_features, explicit, raw_feat], dim=1)
        # shape: (B, To + 2 + past_n, d)

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
        # NOTE: at inference, _past_buffer holds predicted actions.
        # The model has been trained (via Plan C) to handle this noise.
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
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
            )

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        # ── update past buffer ────────────────────────────────────────────
        # Uses predicted action_pred (inference reality).
        # Plan C training ensures the model is robust to this.
        n_exec = self.n_action_steps
        past_n = self.past_n

        if n_exec >= past_n:
            self._past_buffer = action_pred[
                :, n_exec - past_n: n_exec
            ].detach().clone()
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
            action_tokens = self.action_tokenizer.tokenize(batch["action"])

        B = batch["action"].shape[0]
        device = batch["action"].device

        # encode observation
        features = self.obs_encoder(batch["obs"])       # (B, To, d)

        # ── Plan C: Token Round-Trip Augmentation ─────────────────────────
        # Ground-truth past_action from dataset.
        # With probability roundtrip_p (after warmup), replace with
        # tokenizer round-trip reconstruction to simulate inference noise.
        past_actions = batch["past_action"]              # (B, past_n, action_dim)
        past_actions = self._maybe_apply_roundtrip(past_actions)

        # ── build extended condition ──────────────────────────────────────
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

        # increment step counter (used by _maybe_apply_roundtrip)
        self._train_step += 1

        return loss
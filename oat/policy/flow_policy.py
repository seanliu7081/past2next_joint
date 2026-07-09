import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from oat.model.common.normalizer import LinearNormalizer


class FlowPolicy(BasePolicy):
    """
    Baseline rectified flow-matching policy conditioned only on observations.

    This is the plain flow-matching baseline for
    ``FlowPolicyWithEnrichedPast``: no enriched-past conditioning, no warm-start
    prior, no past-action buffer. The action chunk is generated from a standard
    Gaussian source and the velocity field is conditioned on the observation
    tokens only:
      [ obs (To) ]   -> (B, To, d)

    Training (rectified flow):
      x1   = normalized GT action chunk          (B, H, A)
      x0   = sigma * noise                       (pure Gaussian source)
      t    ~ U(0, 1)
      xt   = (1 - t) * x0 + t * x1
      loss = MSE( model(xt, t, cond), x1 - x0 )

    Sampling (Euler, N steps):
      x = sigma * noise; for i in 0..N-1: x += (1/N) * model(x, i/N, cond)
    """

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # backbone model params
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        # flow-matching params
        num_inference_steps: int = 10,
        prior_noise_scale: float = 1.0,
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

        # ── Velocity field backbone ──────────────────────────────────────
        # The conditioning sequence is just the To observation tokens.
        # causal_attn=False -> no causal/memory mask: the whole action chunk is
        # denoised jointly with full cross-attention to all conditioning tokens.
        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.num_inference_steps = num_inference_steps
        self.prior_noise_scale = prior_noise_scale
        self._t_scale = 1000.0  # map flow time [0,1] into SinusoidalPosEmb range

        # ── Report ───────────────────────────────────────────────────────
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs = sum(
            p.numel() for p in obs_encoder.parameters() if p.requires_grad
        )
        num_model_params = sum(p.numel() for p in model.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc : {num_obs_params / 1e6:.1f}M "
            f"({num_trainable_obs / max(num_obs_params, 1):.5%} trainable)\n"
            f"  policy  : {num_model_params / 1e6:.1f}M\n"
            f"  cond_len={n_obs_steps}, flow_steps={num_inference_steps}, "
            f"sigma={prior_noise_scale}\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = "flowpolicy_"
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
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

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
        policy_modules = [self.model]
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

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _scale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Map flow time t in [0,1] into the SinusoidalPosEmb frequency range."""
        return t * self._t_scale

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        cond = self.obs_encoder(obs_dict)           # (B, To, d)
        B = cond.shape[0]

        # ── pure Gaussian source ──────────────────────────────────────────
        x = self.prior_noise_scale * torch.randn(
            B, self.horizon, self.action_dim,
            device=self.device, dtype=cond.dtype,
        )

        # ── Euler integration of the velocity field ───────────────────────
        N = self.num_inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((B,), i * dt, device=cond.device, dtype=cond.dtype)
            x = x + dt * self.model(x, self._scale_t(t), cond)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(x)

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        # normalize target action chunk
        x1 = self.normalizer["action"].normalize(batch["action"])   # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        # encode observation
        cond = self.obs_encoder(batch["obs"])                       # (B, To, d)

        noise = torch.randn_like(x1)
        x0 = self.prior_noise_scale * noise                          # pure Gaussian source

        # ── rectified-flow interpolation ──────────────────────────────────
        t = torch.rand(B, device=device, dtype=x1.dtype)             # (B,)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self.model(xt, self._scale_t(t), cond)
        loss = F.mse_loss(v_pred, v_target)
        return loss

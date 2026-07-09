"""Diffusion policies with a structured ("equi-noise") source distribution.

SOURCE-ONLY probe: identical to ``DiffusionTransformerPolicy`` /
``DiffusionUnetPolicy`` except that the DDPM/DDIM Gaussian source is drawn from a
block-isotropic, normalization-aware distribution (:class:`EquiNoise`) instead of
plain ``torch.randn``. Everything else -- the obs encoder, the diffusion
backbone, the noise scheduler, the epsilon loss, the normalizer -- is unchanged.

Default-off contract
--------------------
When ``policy.equi_noise`` is absent or ``enable: false`` the source falls back to
``torch.randn`` and the policy is bit-for-bit equivalent to its parent (same train
+ eval behavior). The SAME structured source is used at BOTH training time
(``forward``: the injected epsilon target) and inference time (``predict_action``:
the terminal prior ``x_T``) -- they MUST match.

physical_so2 ranges
-------------------
For ``mode: physical_so2`` the per-dim min_max ranges ``R = max - min`` are read
automatically from the fitted action ``LinearNormalizer`` (in ``set_normalizer``),
so nothing has to be hard-coded. An explicit ``ranges`` list in the config
overrides the auto-derived values. The source is rebuilt whenever the normalizer
is (re)set, and lazily on first use so eval-from-checkpoint also works.

PROVENANCE (vendored methods)
  ``forward`` and ``predict_action`` are copied VERBATIM from the parent classes
  in ``oat/policy/diffpolicy.py``; ONLY the ``torch.randn(...)`` source lines are
  replaced with ``self._source(...)``. If the parent bodies change, re-vendor and
  re-diff these copies.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from oat.policy.diffpolicy import DiffusionTransformerPolicy, DiffusionUnetPolicy
from oat.model.diffusion.equi_noise import EquiNoise, libero_7dof_blocks


def _to_container(x):
    """Convert an OmegaConf node to a plain python container; pass through dict/None."""
    if x is None:
        return None
    try:
        from omegaconf import OmegaConf
        from omegaconf.basecontainer import BaseContainer
        if isinstance(x, BaseContainer):
            return OmegaConf.to_container(x, resolve=True)
    except Exception:
        pass
    return x


class _EquiNoiseMixin:
    """Adds a block-isotropic :class:`EquiNoise` source to a diffusion policy.

    Provides ``_source(...)`` (the ``torch.randn`` drop-in) plus lazy/normalizer-
    driven construction of the source module. Default-off unless ``enable: True``.
    """

    def _init_equi_noise(self, equi_noise):
        cfg = _to_container(equi_noise) or {}
        self._equi_enable = bool(cfg.get("enable", False))
        self._equi_mode = cfg.get("mode", "normalized")
        self._equi_scales = cfg.get("scales", None)
        self._equi_ranges = cfg.get("ranges", None)
        self._equi_world_frame_rotation = bool(cfg.get("world_frame_rotation", False))
        # Built lazily once the (per-dim) ranges are known -- either from config or
        # derived from the fitted action normalizer in ``set_normalizer``.
        self.equi_noise: Optional[EquiNoise] = None
        if self._equi_enable:
            print(
                f"{self.get_policy_name()} equi-noise ENABLED "
                f"(mode={self._equi_mode}, world_frame_rotation={self._equi_world_frame_rotation})"
            )

    def _resolve_ranges(self):
        if self._equi_ranges is not None:
            return [float(r) for r in self._equi_ranges]
        if self._equi_mode == "physical_so2":
            # per-dim (max - min) from the fitted action normalizer (mode="limits").
            stats = self.normalizer["action"].get_input_stats()
            R = (stats["max"] - stats["min"]).detach().float().cpu().flatten().tolist()
            return [float(r) for r in R]
        return None

    def _build_equi_noise(self):
        if self.action_dim != 7:
            raise ValueError(
                f"equi-noise block spec (libero_7dof_blocks) expects action_dim=7, "
                f"got action_dim={self.action_dim}. Provide a matching block spec."
            )
        blocks = libero_7dof_blocks(
            mode=self._equi_mode,
            ranges=self._resolve_ranges(),
            scales=self._equi_scales,
            world_frame_rotation=self._equi_world_frame_rotation,
        )
        self.equi_noise = EquiNoise(action_dim=self.action_dim, blocks=blocks, mode=self._equi_mode)

    def _source(self, shape, device, dtype):
        """Structured drop-in for ``torch.randn(shape, device=..., dtype=...)``."""
        if not getattr(self, "_equi_enable", False):
            return torch.randn(shape, device=device, dtype=dtype)
        if self.equi_noise is None:
            self._build_equi_noise()
        return self.equi_noise.sample(shape, device=device, dtype=dtype)

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        # Rebuild the source with fresh per-dim ranges from this normalizer.
        if getattr(self, "_equi_enable", False):
            self._build_equi_noise()


class EquiDiffusionTransformerPolicy(_EquiNoiseMixin, DiffusionTransformerPolicy):
    def __init__(self, *args, equi_noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_equi_noise(equi_noise)

    # ------------------------------------------------------------------
    # Vendored from DiffusionTransformerPolicy.predict_action.
    # ONLY change: the `trajectory = torch.randn(...)` seed -> `self._source(...)`.
    # ------------------------------------------------------------------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]

        # diffusion sampling
        scheduler = self.noise_scheduler
        model = self.model
        trajectory = self._source(
            (len(features), self.horizon, self.action_dim),
            features.device, features.dtype,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, features),
                t, trajectory,
            ).prev_sample

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ------------------------------------------------------------------
    # Vendored from DiffusionTransformerPolicy.forward.
    # ONLY change: the `noise = torch.randn(...)` line -> `self._source(...)`.
    # ------------------------------------------------------------------
    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)

        noise = self._source(trajectory.shape, trajectory.device, trajectory.dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # predict noise residual
        pred = self.model(noisy_trajectory, timesteps, features)
        loss = F.mse_loss(pred, noise)
        return loss


class EquiDiffusionUnetPolicy(_EquiNoiseMixin, DiffusionUnetPolicy):
    def __init__(self, *args, equi_noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_equi_noise(equi_noise)

    # ------------------------------------------------------------------
    # Vendored from DiffusionUnetPolicy.predict_action.
    # ONLY change: the `trajectory = torch.randn(...)` seed -> `self._source(...)`.
    # ------------------------------------------------------------------
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        features = features.reshape(features.shape[0], -1)  # [B, To*d]

        # diffusion sampling
        scheduler = self.noise_scheduler
        model = self.model
        trajectory = self._source(
            (len(features), self.horizon, self.action_dim),
            features.device, features.dtype,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, global_cond=features),
                t, trajectory,
            ).prev_sample

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ------------------------------------------------------------------
    # Vendored from DiffusionUnetPolicy.forward.
    # ONLY change: the `noise = torch.randn(...)` line -> `self._source(...)`.
    # ------------------------------------------------------------------
    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])  # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)
        features = features.reshape(batch_size, -1)  # [B, To*d]

        noise = self._source(trajectory.shape, trajectory.device, trajectory.dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # predict noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=features)
        loss = F.mse_loss(pred, noise)
        return loss

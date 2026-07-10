"""Flow-matching policy with a structured ("equi-noise") source distribution.

SOURCE-ONLY change relative to :class:`~oat.policy.flow_policy.FlowPolicy`:
the Gaussian source is drawn from a block-isotropic
:class:`~oat.equi.sources.BlockIsotropicSource` instead of plain
``torch.randn``. Everything else -- obs encoder, transformer backbone,
rectified-flow loss, normalizer -- is unchanged.

Default-off contract
--------------------
When ``policy.source`` is absent or ``enable: false`` the source falls back to
``torch.randn`` and the policy is bit-for-bit equivalent to ``FlowPolicy``
(same train + eval behavior under the same seed). The SAME source is used at
BOTH ends of the flow: training (``forward``: x0 feeds the interpolation
``xt = (1-t) x0 + t x1`` AND the regression target ``v = x1 - x0``) and
inference (``predict_action``: the Euler-integration seed) -- they MUST match.

Freezing / spec guards
----------------------
``freeze_obs_encoder=True`` turns off gradients for the whole observation
encoder (the parent ``get_optimizer`` already skips non-trainable params) and
keeps the encoder in eval mode even when the policy trains.
``norm_spec_path`` arms the stats-frozen guard: ``set_normalizer`` asserts the
incoming normalizer equals the persisted NormalizationSpec, so every
experimental arm provably trains under identical stats. When the spec mode is
``group_compatible`` and the source requests ``physical_so2``, the warp
correction must degrade to identity -- asserted here.

PROVENANCE (vendored methods)
  ``forward`` and ``predict_action`` are copied VERBATIM from ``FlowPolicy``
  (oat/policy/flow_policy.py); ONLY the ``torch.randn(...)`` source lines are
  replaced with ``self._source(...)``. If the parent bodies change, re-vendor
  and re-diff these copies.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from oat.policy.flow_policy import FlowPolicy
from oat.equi.sources import WARP_PHYSICAL_SO2, build_source
from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    assert_spec_matches,
    load_spec,
)


def _to_container(x):
    """OmegaConf node -> plain python container; pass through dict/None."""
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


class EquiFlowPolicy(FlowPolicy):
    def __init__(
        self,
        *args,
        source=None,
        freeze_obs_encoder: bool = False,
        norm_spec_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._source_cfg = _to_container(source) or {}
        self._source_enable = bool(self._source_cfg.get("enable", False))
        self._norm_spec_path = norm_spec_path
        # Built lazily once the fitted normalizer is available -- either via
        # set_normalizer (training) or the checkpoint state dict (eval).
        self.source_module = None
        self.freeze_obs_encoder = bool(freeze_obs_encoder)
        if self.freeze_obs_encoder:
            self.obs_encoder.requires_grad_(False)
            self.obs_encoder.eval()
        if self._source_enable:
            print(
                f"{self.get_policy_name()} equi source ENABLED "
                f"(kind={self._source_cfg.get('kind', 'block_isotropic')}, "
                f"warp={self._source_cfg.get('warp_correction', 'none')}, "
                f"world_frame_rotation={self._source_cfg.get('world_frame_rotation', False)})"
            )

    # ── source plumbing ─────────────────────────────────────────────────────

    def _build_source(self):
        normalizer_scale = None
        if len(self.normalizer.params_dict) > 0 and "action" in self.normalizer.params_dict:
            normalizer_scale = self.normalizer.params_dict["action"]["scale"]
        self.source_module = build_source(
            self._source_cfg, self.action_dim, normalizer_scale=normalizer_scale
        )
        # Degrade-to-identity assert: under group-compatible normalization the
        # physical_so2 warp correction MUST be exactly ones (tied rho1 scales).
        if (
            self.source_module is not None
            and self._source_cfg.get("warp_correction") == WARP_PHYSICAL_SO2
            and self._norm_spec_path is not None
        ):
            spec = load_spec(self._norm_spec_path)
            if spec["mode"] == MODE_GROUP_COMPATIBLE:
                corr = self.source_module.std_correction
                assert torch.allclose(corr, torch.ones_like(corr), atol=1e-6), (
                    "physical_so2 warp correction must be identity under "
                    f"group_compatible normalization, got {corr.tolist()}"
                )

    def _source(self, shape, device, dtype) -> torch.Tensor:
        """Structured drop-in for ``torch.randn(shape, device=..., dtype=...)``."""
        if not self._source_enable:
            return torch.randn(shape, device=device, dtype=dtype)
        if self.source_module is None:
            self._build_source()
        return self.source_module.sample(shape, device=device, dtype=dtype)

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        # Stats-frozen guard: all arms must train under the persisted spec.
        if self._norm_spec_path is not None:
            assert_spec_matches(self.normalizer, load_spec(self._norm_spec_path))
        # Rebuild the source with fresh scales from this normalizer (also runs
        # the degrade-to-identity assert eagerly at train start).
        if self._source_enable:
            self._build_source()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_obs_encoder:
            self.obs_encoder.eval()
        return self

    # ── Inference ───────────────────────────────────────────────────────────
    # Vendored from FlowPolicy.predict_action (flow_policy.py).
    # ONLY change: the `torch.randn(...)` prior seed -> `self._source(...)`.
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        cond = self.obs_encoder(obs_dict)           # (B, To, d)
        B = cond.shape[0]

        # ── structured source ─────────────────────────────────────────────
        x = self.prior_noise_scale * self._source(
            (B, self.horizon, self.action_dim),
            self.device, cond.dtype,
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
    # Vendored from FlowPolicy.forward (flow_policy.py).
    # ONLY change: the `torch.randn_like(x1)` noise -> `self._source(...)`.
    # x0 feeds BOTH the interpolation xt and the target v = x1 - x0.
    def forward(self, batch) -> torch.Tensor:
        # normalize target action chunk
        x1 = self.normalizer["action"].normalize(batch["action"])   # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        # encode observation
        cond = self.obs_encoder(batch["obs"])                       # (B, To, d)

        noise = self._source(x1.shape, x1.device, x1.dtype)
        x0 = self.prior_noise_scale * noise                          # structured source

        # ── rectified-flow interpolation ──────────────────────────────────
        t = torch.rand(B, device=device, dtype=x1.dtype)             # (B,)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self.model(xt, self._scale_t(t), cond)
        loss = F.mse_loss(v_pred, v_target)
        return loss

"""Flow-matching policy for the AUG-ONLY arm: SE(2) data augmentation WITHOUT
the EquiNoise structured source.

The SE(2) augmentation lives entirely in the data pipeline
(``oat/dataset/se2_aug_zarr_dataset.py``, driven by the ``aug.*`` config
group) -- the policy itself never sees it. This class is therefore a plain
:class:`~oat.policy.flow_policy.FlowPolicy` with a pure ``torch.randn`` source
at BOTH ends of the flow (training interpolation and inference Euler seed):
``forward`` and ``predict_action`` are inherited untouched, so under the same
seed this arm is bit-for-bit the baseline policy.

Relative to :class:`~oat.policy.equi_flow_policy.EquiFlowPolicy` this class
REMOVES all source plumbing and keeps only the experiment-control features
needed for comparable arms:

- ``freeze_obs_encoder`` -- gradients off for the whole observation encoder
  (the parent ``get_optimizer`` already skips non-trainable params) and the
  encoder pinned to eval mode even while the policy trains.
- ``norm_spec_path`` -- stats-frozen guard: ``set_normalizer`` asserts the
  incoming normalizer equals the persisted NormalizationSpec, so every
  experimental arm provably trains under identical stats.

A ``source`` kwarg is accepted ONLY so configs derived from
``train_equi_flowpolicy.yaml`` can pass ``policy.source: null``; an ENABLED
source config is an error here -- use ``EquiFlowPolicy`` for that arm.
"""

from typing import Optional

from oat.policy.flow_policy import FlowPolicy
from oat.equi.normalization import assert_spec_matches, load_spec


class AugFlowPolicy(FlowPolicy):
    def __init__(
        self,
        *args,
        source=None,
        freeze_obs_encoder: bool = False,
        norm_spec_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if source is not None and bool(source.get("enable", False)):
            raise ValueError(
                "AugFlowPolicy is the no-EquiNoise arm but received an ENABLED "
                "source config; use oat.policy.equi_flow_policy.EquiFlowPolicy "
                "instead, or set policy.source: null."
            )
        self._norm_spec_path = norm_spec_path
        self.freeze_obs_encoder = bool(freeze_obs_encoder)
        if self.freeze_obs_encoder:
            self.obs_encoder.requires_grad_(False)
            self.obs_encoder.eval()

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        # Stats-frozen guard: all arms must train under the persisted spec.
        if self._norm_spec_path is not None:
            assert_spec_matches(self.normalizer, load_spec(self._norm_spec_path))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_obs_encoder:
            self.obs_encoder.eval()
        return self

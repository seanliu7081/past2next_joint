"""Tests for oat.policy.equi_flow_policy.EquiFlowPolicy against a minimal
FlowPolicy twin: the default-off bit-for-bit contract, the norm-spec freeze
guard, encoder freezing, and physical_so2 degrade-to-identity."""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    MODE_PER_DIM_MINMAX,
    build_group_compatible_normalizer,
    save_spec,
    spec_from_normalizer,
)
from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.policy.equi_flow_policy import EquiFlowPolicy
from oat.policy.flow_policy import FlowPolicy

OBS_KEY = "agent_state"
OBS_DIM = 5
FEAT_DIM = 32
HORIZON = 16
N_OBS_STEPS = 2

SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {OBS_KEY: {"shape": [OBS_DIM], "type": "low_dim"}},
}

FAKE_FINGERPRINT = {
    "zarr_path": "data/fake.zarr",
    "n_steps": 4096,
    "n_episodes": 8,
    "action_sha1": "0" * 40,
}


class StubEncoder(BaseObservationEncoder):
    """Minimal low-dim observation encoder: normalize + linear projection."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(OBS_DIM, FEAT_DIM)
        self.normalizer = LinearNormalizer()

    def modalities(self):
        return ["low_dim"]

    def output_feature_dim(self):
        return FEAT_DIM

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, obs_dict):
        x = obs_dict[OBS_KEY]
        if OBS_KEY in self.normalizer.params_dict:
            x = self.normalizer[OBS_KEY].normalize(x)
        return self.proj(x)  # (B, To, FEAT_DIM)


def _policy_kwargs():
    return dict(
        shape_meta=SHAPE_META,
        horizon=HORIZON,
        n_action_steps=8,
        n_obs_steps=N_OBS_STEPS,
        embed_dim=64,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        num_inference_steps=3,
        prior_noise_scale=1.0,
    )


def _make_flow_policy(seed=0):
    torch.manual_seed(seed)
    return FlowPolicy(obs_encoder=StubEncoder(), **_policy_kwargs())


def _make_equi_policy(seed=0, **equi_kwargs):
    torch.manual_seed(seed)
    return EquiFlowPolicy(obs_encoder=StubEncoder(), **_policy_kwargs(), **equi_kwargs)


def _batch(seed=10, batch_size=4):
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": {OBS_KEY: torch.rand((batch_size, N_OBS_STEPS, OBS_DIM), generator=g)},
        "action": torch.rand((batch_size, HORIZON, 7), generator=g) - 0.5,
    }


def _assert_bit_identical(pol_a, pol_b, fitted_normalizer):
    pol_a.set_normalizer(fitted_normalizer)
    pol_b.set_normalizer(fitted_normalizer)
    pol_a.eval()
    pol_b.eval()
    batch = _batch()

    torch.manual_seed(1)
    loss_a = pol_a(batch)
    torch.manual_seed(1)
    loss_b = pol_b(batch)
    assert torch.equal(loss_a, loss_b)

    torch.manual_seed(2)
    out_a = pol_a.predict_action(batch["obs"])
    torch.manual_seed(2)
    out_b = pol_b.predict_action(batch["obs"])
    assert torch.equal(out_a["action"], out_b["action"])
    assert torch.equal(out_a["action_pred"], out_b["action_pred"])


# ── (a) default-off: bit-for-bit == FlowPolicy ───────────────────────────────

def test_disabled_source_bit_identical_to_flow_policy(fitted_normalizer):
    pol_a = _make_flow_policy(seed=0)
    pol_b = _make_equi_policy(seed=0)  # no source cfg at all
    assert pol_b.source_module is None
    _assert_bit_identical(pol_a, pol_b, fitted_normalizer)
    assert pol_b.source_module is None  # never built when disabled


def test_enable_false_source_bit_identical(fitted_normalizer):
    pol_a = _make_flow_policy(seed=0)
    pol_b = _make_equi_policy(seed=0, source={"enable": False, "kind": "block_isotropic"})
    _assert_bit_identical(pol_a, pol_b, fitted_normalizer)


# ── (b) enabled with unit scales + warp none: still bit-identical ────────────

def test_unit_scale_source_bit_identical(fitted_normalizer):
    source_cfg = {
        "enable": True,
        "kind": "block_isotropic",
        "warp_correction": "none",
        "scales": {"xy": 1.0, "z": 1.0, "rot": 1.0, "grip": 1.0},
    }
    pol_a = _make_flow_policy(seed=0)
    pol_b = _make_equi_policy(seed=0, source=source_cfg)
    _assert_bit_identical(pol_a, pol_b, fitted_normalizer)
    assert pol_b.source_module is not None
    assert torch.equal(pol_b.source_module.std, torch.ones(7))


# ── (c) norm_spec_path stats-frozen guard ────────────────────────────────────

def test_norm_spec_guard(tmp_path, fitted_normalizer):
    spec = spec_from_normalizer(fitted_normalizer, MODE_PER_DIM_MINMAX, FAKE_FINGERPRINT)
    path = str(tmp_path / "spec.json")
    save_spec(spec, path)

    pol = _make_equi_policy(seed=0, norm_spec_path=path)
    pol.set_normalizer(fitted_normalizer)  # matches the spec: must not raise

    # perturb the spec ON DISK: the same normalizer must now be rejected
    with open(path) as f:
        raw = json.load(f)
    raw["keys"]["action"]["scale"][0] *= 1.01
    with open(path, "w") as f:
        json.dump(raw, f)

    pol_bad = _make_equi_policy(seed=0, norm_spec_path=path)
    with pytest.raises(RuntimeError, match="deviates from the frozen"):
        pol_bad.set_normalizer(fitted_normalizer)


# ── (d) freeze_obs_encoder ───────────────────────────────────────────────────

def test_freeze_obs_encoder():
    pol_free = _make_equi_policy(seed=0)
    assert sum(p.numel() for p in pol_free.obs_encoder.parameters() if p.requires_grad) > 0

    pol = _make_equi_policy(seed=0, freeze_obs_encoder=True)
    assert sum(p.numel() for p in pol.obs_encoder.parameters() if p.requires_grad) == 0

    pol.train()
    assert pol.training and pol.model.training
    assert not pol.obs_encoder.training  # encoder pinned to eval while training

    # parent get_optimizer skips frozen params: encoder groups are empty
    opt = pol.get_optimizer(1e-4, 1e-4, 1e-3, (0.9, 0.95))
    assert all(len(g["params"]) == 0 for g in opt.param_groups[2:])


# ── (e) physical_so2 degrade-to-identity by normalization mode ───────────────

SOURCE_SO2 = {
    "enable": True,
    "kind": "block_isotropic",
    "warp_correction": "physical_so2",
}


def test_physical_so2_identity_under_group_compatible_spec(tmp_path, fitted_normalizer):
    gc = build_group_compatible_normalizer(fitted_normalizer, rgb_keys=[])
    path = str(tmp_path / "spec_gc.json")
    save_spec(spec_from_normalizer(gc, MODE_GROUP_COMPATIBLE, FAKE_FINGERPRINT), path)

    pol = _make_equi_policy(seed=0, source=dict(SOURCE_SO2), norm_spec_path=path)
    pol.set_normalizer(gc)  # runs the degrade-to-identity assert eagerly
    corr = pol.source_module.std_correction
    assert torch.allclose(corr, torch.ones_like(corr), atol=1e-6)


def test_physical_so2_corrects_under_per_dim_spec(tmp_path, fitted_normalizer):
    path = str(tmp_path / "spec_pd.json")
    save_spec(
        spec_from_normalizer(fitted_normalizer, MODE_PER_DIM_MINMAX, FAKE_FINGERPRINT),
        path,
    )

    pol = _make_equi_policy(seed=0, source=dict(SOURCE_SO2), norm_spec_path=path)
    pol.set_normalizer(fitted_normalizer)
    corr = pol.source_module.std_correction
    # asymmetric xy ranges -> untied per-dim scales -> real correction on (0, 1)
    assert not torch.allclose(corr, torch.ones_like(corr), atol=1e-3)
    np.testing.assert_allclose(
        (corr[0] * corr[1]).item(), 1.0, atol=1e-5
    )  # geometric mean 1 within the pair
    assert torch.allclose(corr[2:], torch.ones(5))

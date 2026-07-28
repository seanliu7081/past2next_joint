"""Tests for oat.policy.aug_flow_policy.AugFlowPolicy: bit-for-bit equivalence
to FlowPolicy (no source plumbing at all), rejection of an enabled source
config, and the retained freeze / norm-spec-guard behavior."""

import json

import pytest
import torch

from oat.equi.normalization import (
    MODE_PER_DIM_MINMAX,
    save_spec,
    spec_from_normalizer,
)
from oat.policy.aug_flow_policy import AugFlowPolicy
from tests.test_equi_flow_policy import (
    FAKE_FINGERPRINT,
    StubEncoder,
    _assert_bit_identical,
    _make_flow_policy,
    _policy_kwargs,
)


def _make_aug_policy(seed=0, **aug_kwargs):
    torch.manual_seed(seed)
    return AugFlowPolicy(obs_encoder=StubEncoder(), **_policy_kwargs(), **aug_kwargs)


# ── (a) bit-for-bit == FlowPolicy ────────────────────────────────────────────

def test_bit_identical_to_flow_policy(fitted_normalizer):
    pol_a = _make_flow_policy(seed=0)
    pol_b = _make_aug_policy(seed=0)
    _assert_bit_identical(pol_a, pol_b, fitted_normalizer)


def test_disabled_source_cfg_accepted_and_bit_identical(fitted_normalizer):
    pol_a = _make_flow_policy(seed=0)
    pol_b = _make_aug_policy(seed=0, source={"enable": False, "kind": "block_isotropic"})
    _assert_bit_identical(pol_a, pol_b, fitted_normalizer)


# ── (b) enabled source config is refused ─────────────────────────────────────

def test_enabled_source_cfg_raises():
    with pytest.raises(ValueError, match="no-EquiNoise arm"):
        _make_aug_policy(seed=0, source={"enable": True, "kind": "block_isotropic"})


# ── (c) norm_spec_path stats-frozen guard ────────────────────────────────────

def test_norm_spec_guard(tmp_path, fitted_normalizer):
    spec = spec_from_normalizer(fitted_normalizer, MODE_PER_DIM_MINMAX, FAKE_FINGERPRINT)
    path = str(tmp_path / "spec.json")
    save_spec(spec, path)

    pol = _make_aug_policy(seed=0, norm_spec_path=path)
    pol.set_normalizer(fitted_normalizer)  # matches the spec: must not raise

    with open(path) as f:
        raw = json.load(f)
    raw["keys"]["action"]["scale"][0] *= 1.01
    with open(path, "w") as f:
        json.dump(raw, f)

    pol_bad = _make_aug_policy(seed=0, norm_spec_path=path)
    with pytest.raises(RuntimeError, match="deviates from the frozen"):
        pol_bad.set_normalizer(fitted_normalizer)


# ── (d) freeze_obs_encoder ───────────────────────────────────────────────────

def test_freeze_obs_encoder():
    pol = _make_aug_policy(seed=0, freeze_obs_encoder=True)
    assert sum(p.numel() for p in pol.obs_encoder.parameters() if p.requires_grad) == 0

    pol.train()
    assert pol.training and pol.model.training
    assert not pol.obs_encoder.training  # encoder pinned to eval while training

    opt = pol.get_optimizer(1e-4, 1e-4, 1e-3, (0.9, 0.95))
    assert all(len(g["params"]) == 0 for g in opt.param_groups[2:])

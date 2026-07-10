"""Tests for oat.equi.normalization: per-block scale rules, the
rotate<->normalize commutation that motivates the whole workstream, spec
round-trip, and the stats-frozen / fingerprint guards."""

import copy

import numpy as np
import pytest
import torch

from oat.common.replay_buffer import ReplayBuffer
from oat.equi.blocks import (
    action_rho_matrix,
    libero_action_blocks,
)
from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    MODE_PER_DIM_MINMAX,
    assert_spec_matches,
    build_group_compatible_normalizer,
    fingerprint_replay_buffer,
    group_compatible_scale_offset,
    load_spec,
    normalizer_from_spec,
    save_spec,
    spec_from_normalizer,
    verify_fingerprint,
)
from oat.model.common.normalizer import LinearNormalizer

THETAS = np.deg2rad([-120.0, -30.0, 10.0, 45.0, 90.0])

FAKE_FINGERPRINT = {
    "zarr_path": "data/fake.zarr",
    "n_steps": 4096,
    "n_episodes": 8,
    "action_sha1": "0" * 40,
}


# ── per-block scale rules ────────────────────────────────────────────────────

def test_rho1_tied_scale_and_zero_offset(synthetic_stats):
    stats = synthetic_stats["action"]
    scale, offset = group_compatible_scale_offset(stats, libero_action_blocks())
    # xy pair: one shared scale = 1 / max |min|,|max| over BOTH dims
    s_xy = max(abs(stats["min"][0]), abs(stats["max"][0]),
               abs(stats["min"][1]), abs(stats["max"][1]))
    assert scale[0] == scale[1] == pytest.approx(1.0 / s_xy, abs=0.0)
    np.testing.assert_array_equal(offset[[0, 1]], 0.0)


def test_free_iso_shared_scale(synthetic_stats):
    stats = synthetic_stats["action"]
    scale, offset = group_compatible_scale_offset(stats, libero_action_blocks())
    # rot block (3,4,5): one shared scale over the whole block
    s_rot = max(np.max(np.abs(stats["min"][3:6])), np.max(np.abs(stats["max"][3:6])))
    assert scale[3] == scale[4] == scale[5] == pytest.approx(1.0 / s_rot, abs=0.0)
    np.testing.assert_array_equal(offset[3:6], 0.0)


def test_rho0_dims_match_fitted_limits_normalizer(action_data, fitted_normalizer):
    """rho0 dims must reproduce LinearNormalizer().fit(mode='limits') exactly
    (the formula is vendored from normalizer._fit)."""
    params = fitted_normalizer.params_dict["action"]
    stats = {
        k: params["input_stats"][k].numpy().astype(np.float64)
        for k in ("min", "max", "mean", "std")
    }
    scale, offset = group_compatible_scale_offset(stats, libero_action_blocks())
    fit_scale = params["scale"].numpy().astype(np.float64)
    fit_offset = params["offset"].numpy().astype(np.float64)
    for i in (2, 6):  # z, grip
        assert scale[i] == pytest.approx(fit_scale[i], rel=1e-6, abs=1e-9)
        assert offset[i] == pytest.approx(fit_offset[i], rel=1e-6, abs=1e-9)


def test_identity_and_degenerate_blocks():
    from oat.equi.blocks import BlockSpec, IDENTITY, RHO1

    stats = {
        "min": np.array([0.0, 0.0, -1.0, -1.0]),
        "max": np.array([0.0, 0.0, 1.0, 1.0]),
        "mean": np.zeros(4),
        "std": np.zeros(4),
    }
    # constant-zero rho1 block: guarded, left unscaled
    scale, offset = group_compatible_scale_offset(
        stats, [BlockSpec("xy", (0, 1), RHO1), BlockSpec("q", (2, 3), IDENTITY)]
    )
    np.testing.assert_array_equal(scale, 1.0)
    np.testing.assert_array_equal(offset, 0.0)


# ── rotate <-> normalize commutation (D2) ────────────────────────────────────

def _commutation_max_err(normalizer_field, actions, theta, rotate_rotation):
    """max |normalize(rho(theta) a_raw) - rho(theta) normalize(a_raw)|"""
    from oat.equi.se2_transforms import rotate_action_chunk

    rho_T = action_rho_matrix(theta, rotate_rotation).T
    lhs = normalizer_field.normalize(
        torch.from_numpy(rotate_action_chunk(actions, theta, rotate_rotation))
    ).numpy().astype(np.float64)
    rhs = normalizer_field.normalize(torch.from_numpy(actions)).numpy().astype(np.float64) @ rho_T
    return float(np.max(np.abs(lhs - rhs)))


@pytest.mark.parametrize("rotate_rotation", [False, True])
def test_commutation_holds_under_group_compatible(
    action_data, fitted_normalizer, rotate_rotation
):
    gc = build_group_compatible_normalizer(fitted_normalizer, rgb_keys=[])
    actions = action_data[:512].reshape(32, 16, 7)
    for theta in THETAS:
        err = _commutation_max_err(gc["action"], actions, theta, rotate_rotation)
        assert err < 1e-5, f"commutation broken at theta={theta}: {err}"


def test_commutation_fails_under_per_dim_minmax(action_data, fitted_normalizer):
    """The same identity must FAIL under the baseline per-dim fitted normalizer
    (asymmetric xy ranges -> untied scales + nonzero offsets), proving the
    group-compatible test above has teeth."""
    actions = action_data[:512].reshape(32, 16, 7)
    err = _commutation_max_err(
        fitted_normalizer["action"], actions, np.deg2rad(30.0), rotate_rotation=True
    )
    assert err > 1e-3, f"per-dim normalizer unexpectedly commutes (err={err})"


def test_group_compatible_world_frame_variant_commutes(action_data, fitted_normalizer):
    gc = build_group_compatible_normalizer(
        fitted_normalizer, rgb_keys=[], world_frame_rotation=True
    )
    actions = action_data[:512].reshape(32, 16, 7)
    err = _commutation_max_err(gc["action"], actions, np.deg2rad(30.0), rotate_rotation=True)
    assert err < 1e-5


def test_group_compatible_preserves_input_stats(fitted_normalizer):
    gc = build_group_compatible_normalizer(fitted_normalizer, rgb_keys=[])
    for key in ("action", "agent_state"):
        for stat in ("min", "max", "mean", "std"):
            np.testing.assert_array_equal(
                gc.params_dict[key]["input_stats"][stat].numpy(),
                fitted_normalizer.params_dict[key]["input_stats"][stat].numpy(),
            )


# ── spec round-trip + freeze guards ──────────────────────────────────────────

@pytest.mark.parametrize(
    "mode", [MODE_GROUP_COMPATIBLE, MODE_PER_DIM_MINMAX]
)
def test_spec_roundtrip_exact(tmp_path, fitted_normalizer, mode):
    if mode == MODE_GROUP_COMPATIBLE:
        normalizer = build_group_compatible_normalizer(fitted_normalizer, rgb_keys=[])
    else:
        normalizer = fitted_normalizer
    spec = spec_from_normalizer(normalizer, mode, FAKE_FINGERPRINT)
    path = str(tmp_path / "spec.json")
    save_spec(spec, path)
    loaded = load_spec(path)
    assert loaded == spec

    rebuilt = normalizer_from_spec(loaded)
    for key in normalizer.params_dict.keys():
        for name in ("scale", "offset"):
            np.testing.assert_array_equal(
                rebuilt.params_dict[key][name].numpy(),
                normalizer.params_dict[key][name].numpy(),
            )
            assert rebuilt.params_dict[key][name].dtype == torch.float32


def test_load_spec_missing_file_points_at_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_normalization_spec"):
        load_spec(str(tmp_path / "nope.json"))


def test_assert_spec_matches_passes_then_raises_on_perturbation(fitted_normalizer):
    spec = spec_from_normalizer(fitted_normalizer, MODE_PER_DIM_MINMAX, FAKE_FINGERPRINT)
    assert_spec_matches(fitted_normalizer, spec)

    bad = copy.deepcopy(spec)
    bad["keys"]["action"]["scale"][0] *= 1.01
    with pytest.raises(RuntimeError, match="deviates from the frozen"):
        assert_spec_matches(fitted_normalizer, bad)


def _tiny_replay_buffer(rng, n_episodes=3, episode_len=20):
    rb = ReplayBuffer.create_empty_numpy()
    for _ in range(n_episodes):
        rb.add_episode(
            {"action": rng.normal(size=(episode_len, 7)).astype(np.float32)}
        )
    return rb


def test_verify_fingerprint_roundtrip_and_tamper():
    rng = np.random.default_rng(7)
    rb = _tiny_replay_buffer(rng)
    fp = fingerprint_replay_buffer(rb, "data/tiny.zarr")
    assert fp["n_steps"] == 60 and fp["n_episodes"] == 3

    spec = {"fingerprint": fp}
    verify_fingerprint(spec, rb, "data/tiny.zarr")  # must not raise

    tampered = copy.deepcopy(spec)
    tampered["fingerprint"]["action_sha1"] = "deadbeef" * 5
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_fingerprint(tampered, rb, "data/tiny.zarr")


def test_verify_fingerprint_catches_different_data():
    rng = np.random.default_rng(8)
    rb = _tiny_replay_buffer(rng)
    spec = {"fingerprint": fingerprint_replay_buffer(rb, "data/tiny.zarr")}
    other = _tiny_replay_buffer(np.random.default_rng(9))
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_fingerprint(spec, other, "data/tiny.zarr")

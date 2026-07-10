"""Tests for oat.equi.blocks: block coverage/disjointness guards, the canonical
LIBERO layouts, and the exact group action rho_A(theta)."""

import numpy as np
import pytest

from oat.equi.blocks import (
    ACTION_DIM,
    FREE_ISO,
    RHO0,
    RHO1,
    BlockSpec,
    action_rho_matrix,
    assert_blocks_cover,
    libero_action_blocks,
    libero_obs_blocks,
)

THETAS = np.deg2rad([-170.0, -60.0, -10.0, 0.0, 10.0, 30.0, 90.0, 179.0])


# ── BlockSpec / coverage guards ──────────────────────────────────────────────

def test_blockspec_rejects_bad_specs():
    with pytest.raises(AssertionError):
        BlockSpec("bad_rep", (0,), "rho2")
    with pytest.raises(AssertionError):
        BlockSpec("empty", (), RHO0)
    with pytest.raises(AssertionError):
        BlockSpec("rho1_not_pair", (0, 1, 2), RHO1)


def test_assert_blocks_cover_fires_on_overlap():
    blocks = [BlockSpec("a", (0, 1), RHO1), BlockSpec("b", (1, 2), RHO1)]
    with pytest.raises(AssertionError):
        assert_blocks_cover(blocks, 3)


def test_assert_blocks_cover_fires_on_gap():
    blocks = [BlockSpec("a", (0, 1), RHO1), BlockSpec("b", (3,), RHO0)]
    with pytest.raises(AssertionError):
        assert_blocks_cover(blocks, 4)


def test_assert_blocks_cover_fires_on_out_of_range():
    blocks = [BlockSpec("a", (0, 7), RHO1)]
    with pytest.raises(AssertionError):
        assert_blocks_cover(blocks, 7)


def test_assert_blocks_cover_accepts_exact_cover():
    assert_blocks_cover(
        [BlockSpec("a", (2, 0), RHO1), BlockSpec("b", (1,), RHO0)], 3
    )


# ── canonical LIBERO layouts ─────────────────────────────────────────────────

@pytest.mark.parametrize("world_frame_rotation", [False, True])
def test_libero_action_blocks_cover_7_dims(world_frame_rotation):
    blocks = libero_action_blocks(world_frame_rotation)
    assert_blocks_cover(blocks, ACTION_DIM)
    covered = sorted(i for b in blocks for i in b.idx)
    assert covered == list(range(ACTION_DIM))
    reps = {b.name: b.rep for b in blocks}
    assert reps["xy"] == RHO1
    assert reps["z"] == RHO0 and reps["grip"] == RHO0
    if world_frame_rotation:
        assert reps["rot_xy"] == RHO1 and reps["rot_z"] == RHO0
    else:
        assert reps["rot"] == FREE_ISO


def test_libero_obs_blocks_cover_their_keys():
    dims = {
        "robot0_eef_pos": 3,
        "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2,
        "task_uid": 1,
        "robot0_joint_pos": 7,
    }
    obs_blocks = libero_obs_blocks()
    assert set(obs_blocks.keys()) == set(dims.keys())
    for key, blocks in obs_blocks.items():
        assert_blocks_cover(blocks, dims[key])


# ── rho_A(theta) ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rotate_rotation", [False, True])
@pytest.mark.parametrize("theta", THETAS)
def test_action_rho_matrix_orthogonal(theta, rotate_rotation):
    rho = action_rho_matrix(theta, rotate_rotation)
    assert rho.shape == (ACTION_DIM, ACTION_DIM)
    np.testing.assert_allclose(rho @ rho.T, np.eye(ACTION_DIM), atol=1e-12)


@pytest.mark.parametrize("world_frame_rotation", [False, True])
@pytest.mark.parametrize("rotate_rotation", [False, True])
@pytest.mark.parametrize("theta", THETAS)
def test_action_rho_matrix_block_diagonal(theta, rotate_rotation, world_frame_rotation):
    """rho_A(theta) must be block-diagonal w.r.t. BOTH block decompositions
    (the free_iso rot block strictly contains the world-frame (rx, ry) pair)."""
    rho = action_rho_matrix(theta, rotate_rotation)
    for b in libero_action_blocks(world_frame_rotation):
        inside = np.zeros(ACTION_DIM, dtype=bool)
        inside[list(b.idx)] = True
        off_block = rho[np.ix_(inside, ~inside)]
        np.testing.assert_allclose(off_block, 0.0, atol=0.0)


@pytest.mark.parametrize("rotate_rotation", [False, True])
def test_action_rho_matrix_is_a_group_homomorphism(rotate_rotation):
    rng = np.random.default_rng(2)
    for a, b in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        lhs = action_rho_matrix(a, rotate_rotation) @ action_rho_matrix(b, rotate_rotation)
        rhs = action_rho_matrix(a + b, rotate_rotation)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_action_rho_matrix_identity_at_zero():
    np.testing.assert_array_equal(action_rho_matrix(0.0, True), np.eye(ACTION_DIM))
    np.testing.assert_array_equal(action_rho_matrix(0.0, False), np.eye(ACTION_DIM))

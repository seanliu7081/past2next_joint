"""Tests for oat.equi.se2_transforms: quaternion helpers against scipy, the
raw-space action/proprio group action, and the axis-angle conjugation identity
that justifies rotating (rx, ry)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from oat.equi.blocks import action_rho_matrix
from oat.equi.se2_transforms import (
    quat_geodesic_angle_wxyz,
    quat_mul_wxyz,
    quat_mul_xyzw,
    quat_z_wxyz,
    quat_z_xyzw,
    rot2d,
    rotate_action_chunk,
    rotate_proprio,
    rotate_xy,
)

THETAS = np.deg2rad([-135.0, -30.0, 0.0, 10.0, 45.0, 120.0])


def _random_unit_quats_xyzw(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _assert_quat_close_signfree(q, q_ref, atol=1e-10):
    """Quaternions double-cover rotations: q and -q are the same rotation."""
    d = np.minimum(
        np.max(np.abs(q - q_ref), axis=-1), np.max(np.abs(q + q_ref), axis=-1)
    )
    np.testing.assert_allclose(d, 0.0, atol=atol)


# ── quaternion helpers vs scipy ──────────────────────────────────────────────

def test_quat_mul_xyzw_matches_scipy_composition():
    rng = np.random.default_rng(0)
    q1 = _random_unit_quats_xyzw(rng, 50)
    q2 = _random_unit_quats_xyzw(rng, 50)
    ours = quat_mul_xyzw(q1, q2)
    ref = (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()  # xyzw
    _assert_quat_close_signfree(ours, ref)


def test_quat_mul_wxyz_matches_scipy_composition():
    rng = np.random.default_rng(1)
    q1 = _random_unit_quats_xyzw(rng, 50)
    q2 = _random_unit_quats_xyzw(rng, 50)
    to_wxyz = [3, 0, 1, 2]
    ours_wxyz = quat_mul_wxyz(q1[:, to_wxyz], q2[:, to_wxyz])
    ref = (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()  # xyzw
    _assert_quat_close_signfree(ours_wxyz[:, [1, 2, 3, 0]], ref)


@pytest.mark.parametrize("theta", THETAS)
def test_quat_z_matches_scipy(theta):
    ref = Rotation.from_euler("z", theta).as_quat()  # xyzw
    _assert_quat_close_signfree(quat_z_xyzw(theta), ref)
    _assert_quat_close_signfree(quat_z_wxyz(theta)[[1, 2, 3, 0]], ref)


def test_quat_geodesic_angle_wxyz():
    theta = 0.4
    q0 = quat_z_wxyz(0.0)
    q1 = quat_z_wxyz(theta)
    np.testing.assert_allclose(quat_geodesic_angle_wxyz(q0, q1), theta, atol=1e-12)
    # sign-insensitive: -q is the same rotation
    np.testing.assert_allclose(quat_geodesic_angle_wxyz(q0, -q1), theta, atol=1e-12)
    np.testing.assert_allclose(quat_geodesic_angle_wxyz(q1, q1), 0.0, atol=1e-6)


# ── rotate_xy ────────────────────────────────────────────────────────────────

def test_rotate_xy_about_origin_and_center():
    pts = np.array([[2.0, 0.0, 0.5], [1.0, 1.0, -0.3]])
    theta = np.pi / 2.0

    out = rotate_xy(pts, theta)
    np.testing.assert_allclose(out[0], [0.0, 2.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(out[1], [-1.0, 1.0, -0.3], atol=1e-12)

    center = np.array([1.0, 0.0])
    out_c = rotate_xy(pts, theta, center_xy=center)
    np.testing.assert_allclose(out_c[0], [1.0, 1.0, 0.5], atol=1e-12)
    # input must not be mutated; trailing dims untouched
    np.testing.assert_array_equal(pts[0], [2.0, 0.0, 0.5])
    np.testing.assert_array_equal(out_c[:, 2], pts[:, 2])


# ── rotate_action_chunk ──────────────────────────────────────────────────────

@pytest.mark.parametrize("rotate_rotation", [False, True])
@pytest.mark.parametrize("theta", THETAS)
def test_rotate_action_chunk_roundtrip(theta, rotate_rotation):
    rng = np.random.default_rng(2)
    a = rng.uniform(-1.0, 1.0, size=(4, 16, 7)).astype(np.float32)
    back = rotate_action_chunk(
        rotate_action_chunk(a, theta, rotate_rotation), -theta, rotate_rotation
    )
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, a, atol=1e-6)


@pytest.mark.parametrize("rotate_rotation", [False, True])
@pytest.mark.parametrize("theta", THETAS)
def test_rotate_action_chunk_equals_rho_matrix(theta, rotate_rotation):
    rng = np.random.default_rng(3)
    a = rng.uniform(-1.0, 1.0, size=(5, 16, 7))
    ours = rotate_action_chunk(a, theta, rotate_rotation)
    ref = a @ action_rho_matrix(theta, rotate_rotation).T
    np.testing.assert_allclose(ours, ref, atol=1e-6)
    # invariant dims are bit-untouched (only cast to float32)
    np.testing.assert_array_equal(ours[..., [2, 5, 6]], a[..., [2, 5, 6]].astype(np.float32))
    if not rotate_rotation:
        np.testing.assert_array_equal(ours[..., 3:5], a[..., 3:5].astype(np.float32))


def test_rotate_action_chunk_rejects_wrong_last_dim():
    with pytest.raises(AssertionError):
        rotate_action_chunk(np.zeros((3, 6)), 0.1, True)


@pytest.mark.parametrize("theta", np.deg2rad([-75.0, 15.0, 100.0]))
def test_axis_angle_conjugation_identity(theta):
    """Rz exp(r^) Rz^-1 == exp((Rz r)^): rotating the (rx, ry) components of an
    axis-angle vector by R(theta) is exactly the world-frame conjugation of the
    rotation it encodes. This is the geometric fact behind rotate_rotation."""
    rng = np.random.default_rng(4)
    Rz = Rotation.from_euler("z", theta)
    for r in rng.uniform(-1.5, 1.5, size=(20, 3)):
        lhs = (Rz * Rotation.from_rotvec(r) * Rz.inv()).as_matrix()
        rhs = Rotation.from_rotvec(Rz.apply(r)).as_matrix()
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
        # and Rz.apply(r) is exactly [R(theta) @ (rx, ry), rz]
        r_rot = np.concatenate([rot2d(theta) @ r[:2], r[2:]])
        np.testing.assert_allclose(Rz.apply(r), r_rot, atol=1e-12)


# ── rotate_proprio ───────────────────────────────────────────────────────────

def test_rotate_proprio_eef_pos_about_base_hand_computed():
    theta = np.pi / 2.0
    p_base = np.array([1.0, 0.0])
    obs = {"robot0_eef_pos": np.array([[2.0, 0.0, 0.5]], dtype=np.float32)}
    out = rotate_proprio(obs, theta, p_base)
    # (2,0) - (1,0) = (1,0) --R(90deg)--> (0,1); + (1,0) = (1,1); z untouched
    np.testing.assert_allclose(out["robot0_eef_pos"], [[1.0, 1.0, 0.5]], atol=1e-6)
    assert out["robot0_eef_pos"].dtype == np.float32


@pytest.mark.parametrize("theta", THETAS)
def test_rotate_proprio_eef_quat_matches_scipy(theta):
    rng = np.random.default_rng(5)
    q = _random_unit_quats_xyzw(rng, 30).astype(np.float32)
    out = rotate_proprio({"robot0_eef_quat": q}, theta, np.zeros(2))
    ref = (Rotation.from_euler("z", theta) * Rotation.from_quat(q)).as_quat()
    _assert_quat_close_signfree(out["robot0_eef_quat"], ref, atol=1e-6)
    assert out["robot0_eef_quat"].dtype == np.float32


def test_rotate_proprio_joint_pos_shift_and_passthrough():
    theta = 0.3
    jp = np.arange(14, dtype=np.float32).reshape(2, 7)
    grip = np.array([[0.02, -0.02]], dtype=np.float32)
    uid = np.array([[3.0]], dtype=np.float32)
    obs = {
        "robot0_joint_pos": jp,
        "robot0_gripper_qpos": grip,
        "task_uid": uid,
    }
    out = rotate_proprio(obs, theta, np.array([0.1, -0.2]))
    np.testing.assert_allclose(out["robot0_joint_pos"][:, 0], jp[:, 0] + theta, atol=1e-6)
    np.testing.assert_array_equal(out["robot0_joint_pos"][:, 1:], jp[:, 1:])
    # untouched keys pass through as the same objects; input jp not mutated
    assert out["robot0_gripper_qpos"] is grip
    assert out["task_uid"] is uid
    np.testing.assert_array_equal(jp[:, 0], np.arange(14, dtype=np.float32).reshape(2, 7)[:, 0])


def test_rotate_proprio_theta_zero_is_identity():
    rng = np.random.default_rng(6)
    obs = {
        "robot0_eef_pos": rng.normal(size=(3, 3)).astype(np.float32),
        "robot0_eef_quat": _random_unit_quats_xyzw(rng, 3).astype(np.float32),
        "robot0_joint_pos": rng.normal(size=(3, 7)).astype(np.float32),
    }
    out = rotate_proprio(obs, 0.0, np.array([0.5, 0.5]))
    for k, v in obs.items():
        np.testing.assert_allclose(out[k], v, atol=1e-7)

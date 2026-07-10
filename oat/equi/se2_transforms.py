"""Pure-numpy SE(2) transforms for LIBERO action chunks, proprio states, and
MuJoCo quaternions.

This module is imported inside dataloader workers and in the offline
pre-render pipeline. It must stay free of robosuite / mujoco / torch imports.

Conventions (footgun alert -- two quaternion layouts coexist in this repo):
  * MuJoCo qpos / body_xquat quaternions are **wxyz** (scalar first).
  * robosuite ``transform_utils`` and the dataset's ``robot0_eef_quat`` (built
    by ``dataset_conversion.axisangle2quat``) are **xyzw** (scalar last).

All rotations here are the world-frame yaw R_z(theta) about the robot base
axis; theta is in **radians**.
"""

from typing import Dict, Optional

import numpy as np


def rot2d(theta: float) -> np.ndarray:
    """2x2 rotation matrix R(theta); acts on column vectors v' = R @ v."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_xy(points: np.ndarray, theta: float, center_xy: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate the leading two components of ``points (..., >=2)`` about
    ``center_xy`` (default: origin). Only dims 0 and 1 are touched."""
    out = np.array(points, dtype=np.float64, copy=True)
    xy = out[..., :2]
    if center_xy is not None:
        xy = xy - np.asarray(center_xy, dtype=np.float64)
    xy = xy @ rot2d(theta).T
    if center_xy is not None:
        xy = xy + np.asarray(center_xy, dtype=np.float64)
    out[..., :2] = xy
    return out


# ── quaternions ──────────────────────────────────────────────────────────────

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both ``(..., 4)`` in wxyz (MuJoCo) order."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both ``(..., 4)`` in xyzw (robosuite/dataset)
    order."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w = quat_mul_wxyz(q1[..., [3, 0, 1, 2]], q2[..., [3, 0, 1, 2]])
    return w[..., [1, 2, 3, 0]]


def quat_z_wxyz(theta: float) -> np.ndarray:
    """Quaternion of R_z(theta), wxyz order."""
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float64)


def quat_z_xyzw(theta: float) -> np.ndarray:
    """Quaternion of R_z(theta), xyzw order."""
    return np.array([0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)], dtype=np.float64)


def quat_geodesic_angle_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Sign-insensitive geodesic angle (radians) between two wxyz quaternions."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


# ── action / proprio group action ────────────────────────────────────────────

def rotate_action_chunk(actions: np.ndarray, theta: float, rotate_rotation: bool) -> np.ndarray:
    """Apply rho_A(theta) to a RAW action chunk ``(..., 7)``:
    ``[dx, dy, dz, rx, ry, rz, grip]``.

    (dx, dy) always rotates (world-frame position delta under both controller
    hypotheses). The axis-angle delta transforms by conjugation:
    ``Rz exp(r^) Rz^T = exp((Rz r)^)``, i.e. (rx, ry) rotates and rz is
    invariant -- but ONLY if the controller interprets the delta in the world
    frame (``rotate_rotation=True``, gated by the M2 probe). dz and grip are
    invariant.

    Rotating RAW labels commutes exactly with group-compatible normalization
    (tied scale, zero offset within each rotating block), so the policy-side
    ``normalizer['action'].normalize`` stays untouched.
    """
    a = np.array(actions, dtype=np.float64, copy=True)
    assert a.shape[-1] == 7, f"expected (..., 7) actions, got {a.shape}"
    R = rot2d(theta)
    a[..., 0:2] = a[..., 0:2] @ R.T
    if rotate_rotation:
        a[..., 3:5] = a[..., 3:5] @ R.T
    return a.astype(np.float32)


def rotate_proprio(
    obs: Dict[str, np.ndarray],
    theta: float,
    p_base_xy: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Apply the world rotation R_z(theta) about ``p_base_xy`` to the RAW
    proprio observation dict (dataset conventions). Returns a new dict; keys
    not listed below pass through unchanged (same objects).

      * ``robot0_eef_pos (..., 3)``:  p_xy <- R(theta)(p_xy - p_base) + p_base
      * ``robot0_eef_quat (..., 4)`` **xyzw**:  q <- q_Rz(theta) ⊗ q
      * ``robot0_joint_pos (..., 7)``: joint 1 (index 0) += theta
      * ``robot0_gripper_qpos``, ``task_uid``: invariant
    """
    out = dict(obs)
    if "robot0_eef_pos" in out:
        pos = rotate_xy(out["robot0_eef_pos"], theta, center_xy=p_base_xy)
        out["robot0_eef_pos"] = pos.astype(np.float32)
    if "robot0_eef_quat" in out:
        q = np.asarray(out["robot0_eef_quat"], dtype=np.float64)
        qz = np.broadcast_to(quat_z_xyzw(theta), q.shape)
        out["robot0_eef_quat"] = quat_mul_xyzw(qz, q).astype(np.float32)
    if "robot0_joint_pos" in out:
        jp = np.array(out["robot0_joint_pos"], dtype=np.float64, copy=True)
        jp[..., 0] += theta
        out["robot0_joint_pos"] = jp.astype(np.float32)
    return out

"""Camera math for the GS pipeline: fovy -> K, MuJoCo camera -> OpenCV w2c.

Conventions (G7: measured by scripts/gsaug/probe_render_facts.py, asserted here):

* MuJoCo cameras look along **-z** with **+y up** (OpenGL convention);
  ``data.cam_xmat[i]`` columns are the camera axes expressed in world frame and
  ``data.cam_xpos[i]`` is the camera center in world.
* gsplat / OpenCV cameras look along **+z** with **+y down**; ``viewmats`` passed
  to ``gsplat.rasterization`` are **world-to-camera** in that convention.
* The GL->CV conversion is a fixed axis flip ``F`` (candidate set below); the
  *winning* flip is a measured fact (F1) recorded in ``gs_render_facts.json``,
  never assumed at runtime (facts-loading helpers at the bottom).
* ``fovy`` is the **vertical** field of view in degrees; MuJoCo pixels are
  square, so ``fx == fy``. Principal point at ``(W/2, H/2)`` — F1's <=0.5 px
  gate is what validates this choice of pixel-center convention.

This module is sim-free (numpy only) so it can be imported everywhere.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Candidate GL->CV axis flips for the F1 sweep, keyed by a stable name that the
# facts file records. ``gl_to_cv`` (y and z negated) is the textbook answer.
FLIP_CANDIDATES: Dict[str, np.ndarray] = {
    "gl_to_cv": np.diag([1.0, -1.0, -1.0]),
    "identity": np.diag([1.0, 1.0, 1.0]),
    "flip_xz": np.diag([-1.0, 1.0, -1.0]),
    "flip_xy": np.diag([-1.0, -1.0, 1.0]),
}


def fovy_to_K(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics (3,3) float64 from a vertical FOV in degrees."""
    fy = height / (2.0 * np.tan(np.deg2rad(float(fovy_deg)) / 2.0))
    fx = fy  # square pixels (MuJoCo)
    return np.array(
        [[fx, 0.0, width / 2.0],
         [0.0, fy, height / 2.0],
         [0.0, 0.0, 1.0]], dtype=np.float64)


def mujoco_cam_to_w2c(cam_xpos: np.ndarray, cam_xmat: np.ndarray,
                      flip: np.ndarray) -> np.ndarray:
    """World-to-camera (4,4) float64 in OpenCV convention.

    ``cam_xmat`` is the (3,3) (or flat (9,)) MuJoCo camera rotation whose
    COLUMNS are the GL camera axes in world frame; ``flip`` is the GL->CV axis
    flip (one of FLIP_CANDIDATES, per facts F1). Camera-to-world is
    ``R_c2w = R_gl @ flip``; this returns its inverse packed with the
    translation.
    """
    R_gl = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    p = np.asarray(cam_xpos, dtype=np.float64).reshape(3)
    R_c2w = R_gl @ np.asarray(flip, dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_c2w.T
    w2c[:3, 3] = -R_c2w.T @ p
    return w2c


def c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """Invert a rigid (4,4) transform."""
    c2w = np.eye(4, dtype=np.float64)
    R = w2c[:3, :3]
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ w2c[:3, 3]
    return c2w


def project(K: np.ndarray, w2c: np.ndarray, points_w: np.ndarray) -> np.ndarray:
    """Project world points (...,3) to pixel coords (...,2) (OpenCV: u right, v down).

    Points behind the camera get NaN coordinates rather than a bogus mirror
    projection.
    """
    pts = np.asarray(points_w, dtype=np.float64)
    ph = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = ph[..., 2:3]
    uv = ph[..., :2] / np.where(z > 1e-9, z, np.nan)
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def lookat_c2w(eye: np.ndarray, target: np.ndarray,
               up: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV camera-to-world (4,4) looking from ``eye`` at ``target`` (z fwd, y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd = fwd / np.linalg.norm(fwd)
    upv = np.asarray(up, dtype=np.float64)
    right = np.cross(fwd, upv)
    n = np.linalg.norm(right)
    if n < 1e-8:  # looking straight along up: pick an arbitrary right
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(right)
    right = right / n
    down = np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = fwd
    c2w[:3, 3] = eye
    return c2w


def c2w_to_mjv_camera(c2w: np.ndarray):
    """Free-camera parameters (lookat, distance, azimuth, elevation) hitting a
    given OpenCV c2w pose is NOT generally representable; capture instead drives
    ``mujoco.MjvCamera`` in ``mjCAMERA_FREE`` mode via lookat/azimuth/elevation
    and *records* the resulting exact pose from ``data.cam_*``-equivalent
    scene state. Kept here as the single documented statement of that fact so
    nobody re-attempts the inverse mapping."""
    raise NotImplementedError(
        "drive MjvCamera by (lookat, distance, azimuth, elevation) and record "
        "the achieved pose; see oat/gsaug/capture.py")


# ── facts file (G7) ─────────────────────────────────────────────────────────

DEFAULT_FACTS_PATH = "data/libero/gs_render_facts.json"


def load_render_facts(path: str = DEFAULT_FACTS_PATH,
                      require_pass: bool = True) -> dict:
    """Load gs_render_facts.json; by default refuse a non-PASSing file (G7)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"renderer facts not found at '{path}': run "
            f"scripts/gsaug/probe_render_facts.py first (M1 gate).")
    with open(path) as f:
        facts = json.load(f)
    if require_pass and facts.get("pass") is not True:
        raise RuntimeError(
            f"'{path}' has pass={facts.get('pass')!r}: the renderer-facts probe "
            f"did not PASS; refusing to build GS components on unverified "
            f"conventions (G7). Re-run scripts/gsaug/probe_render_facts.py.")
    return facts


def facts_flip(facts: dict) -> np.ndarray:
    """The measured GL->CV flip matrix (F1) as a (3,3) float64 array."""
    name = facts["F1"]["flip_name"]
    if name not in FLIP_CANDIDATES:
        raise RuntimeError(
            f"facts file records unknown flip '{name}'; known: "
            f"{sorted(FLIP_CANDIDATES)} — facts file and code disagree (G7).")
    return FLIP_CANDIDATES[name].copy()


def facts_orientation_flip_ud(facts: dict) -> bool:
    """Whether gsplat output must be up/down-flipped into dataset orientation (F2)."""
    return bool(facts["F2"]["gsplat_flip_ud"])

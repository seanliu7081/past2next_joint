"""Sim-facing capture backend for GS asset building (plan §5.1, M2).

Everything the capture script needs to turn a live LIBERO ``ControlEnv`` into
per-component orbit captures (RGB + metric depth + geom-id seg per view, G8:
512², intrinsics/extrinsics in OpenCV convention), plus the ``CaptureBundle``
loader the asset trainer consumes.

Free-camera geometry (the one convention decision made here, validated at
runtime, G7): orbit views are driven through ``mujoco.MjvCamera`` in
``mjCAMERA_FREE`` mode via (lookat, distance, azimuth, elevation). MuJoCo's
free camera has zero roll about the view axis relative to world z, and its
GL forward vector is::

    forward = [cos(el) cos(az), cos(el) sin(az), sin(el)]   (az, el in MJV degrees)
    eye     = lookat - distance * forward

with MJV elevation NEGATIVE for a camera ABOVE the lookat. ``OrbitPose``
stores elevation in the plan's convention (positive above the horizon) and
negates it for MuJoCo. The recorded OpenCV c2w is computed analytically from
(eye, lookat) via ``cameras.lookat_c2w`` — record what we compute — and
validated two ways:

* every view: :func:`assert_free_camera_pose` compares the analytic c2w
  against the GL camera MuJoCo actually placed in the scene (exact,
  extrinsics-only);
* once per capture (F1 pattern): :class:`PoseValidator` checks the FULL
  (K, c2w) chain against rendered pixels — a seg-projected landmark (small
  centered-primitive geom, ≤ 1 px) where one exists, else backprojected
  depth planarity (F5 pattern, median ≤ 5 mm) against the table-top box or,
  when only a mesh table is visible, the 'floor' plane geom.

Renderer facts (G7): the raw ``mujoco.Renderer`` needs the measured F2b
visualization flags (:func:`scene_option_from_facts`) for parity with the
robosuite obs pipeline; depth is ALREADY METRIC (M0); seg renders return
(H, W, 2) int32 with geom id in channel 0 (−1 = none).

Capture layout on disk (consumed by the trainer via ``CaptureBundle.load``)::

    <captures>/<component>/view_%04d.png            uint8 RGB
                           view_%04d_depth.npy      float32 meters
                           view_%04d_seg.png        uint16 geom-id+1 (0 = none)
                           transforms.json          K, per-view c2w, poses, sha1
                           masks/view_%04d_mask.png (robot_hide == 'masked' only)

Import discipline: module scope is numpy/stdlib + ``oat.gsaug.cameras`` only;
``mujoco`` and image IO are imported lazily so the (sim-free) trainer can
import ``CaptureBundle`` without a GL stack.
"""

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from oat.gsaug.cameras import fovy_to_K, lookat_c2w, project

# Bodies/geoms with these name prefixes form the robot stack (arm, gripper,
# mount plate) for hiding, masking, and link-pose recording.
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")

# xy offset applied to graveyarded movables' free joints (facts F4-a).
GRAVEYARD_XY_OFFSET = 50.0

TRANSFORMS_NAME = "transforms.json"
VIEW_FILE_FMT = "view_{:04d}"

# mujoco enum values (stable across versions; asserted lazily against the
# live enums where mujoco is imported).
_MJ_OBJ_GEOM = 5          # mujoco.mjtObj.mjOBJ_GEOM
_MJ_GEOM_PLANE = 0        # mujoco.mjtGeom.mjGEOM_PLANE
_MJ_GEOM_BOX = 6          # mujoco.mjtGeom.mjGEOM_BOX
# geom types whose frame origin is the visual center — usable as landmarks
_CENTERED_GEOM_TYPES = (2, 3, 4, 5, 6)  # sphere, capsule, ellipsoid, cylinder, box

# per-view GL-camera cross-check tolerance (mjvGLCamera stores float32)
_GL_POSE_TOL = 2e-3
# once-per-capture pixel-chain validation gates
_LANDMARK_MAX_PX = 1.0
_TABLE_DEPTH_MAX_MED_M = 5e-3


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    assert n > 1e-9, f"zero-length vector {v}"
    return v / n


def _raw_model(model):
    return getattr(model, "_model", model)


def _raw_data(data):
    return getattr(data, "_data", data)


# ── orbit geometry (plan §5.1) ───────────────────────────────────────────────

@dataclass(frozen=True)
class OrbitPose:
    """One free-camera orbit view. ``elevation_deg`` is in the PLAN convention:
    positive = camera above the lookat horizon (MuJoCo's MjvCamera elevation is
    the negation)."""

    lookat: Tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float

    @property
    def mjv_elevation_deg(self) -> float:
        return -self.elevation_deg

    def eye(self) -> np.ndarray:
        """Camera center in world: MuJoCo free-camera placement, analytically."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.mjv_elevation_deg)
        forward = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return np.asarray(self.lookat, dtype=np.float64) - self.distance * forward

    def c2w_opencv(self) -> np.ndarray:
        """Analytic OpenCV camera-to-world (4,4). MuJoCo's free camera has zero
        roll about the view axis w.r.t. world z, so ``lookat_c2w`` with world-z
        up reproduces its orientation exactly (validated per view against the
        scene's GL camera and once per capture against pixels, G7)."""
        return lookat_c2w(self.eye(), np.asarray(self.lookat, dtype=np.float64))

    def mjv_camera(self):
        """A ``mujoco.MjvCamera`` (mjCAMERA_FREE) realizing this pose."""
        import mujoco
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat[:] = np.asarray(self.lookat, dtype=np.float64)
        cam.distance = float(self.distance)
        cam.azimuth = float(self.azimuth_deg)
        cam.elevation = float(self.mjv_elevation_deg)
        return cam

    def cam_params(self) -> dict:
        return {
            "lookat": [float(x) for x in self.lookat],
            "distance": float(self.distance),
            "azimuth_deg": float(self.azimuth_deg),
            "elevation_deg": float(self.elevation_deg),
            "mjv_elevation_deg": float(self.mjv_elevation_deg),
        }


def orbit_poses(lookat: Sequence[float], radius: float,
                ring_elevations: Sequence[float], n_azimuths: int,
                az_offset_deg: float = 0.0,
                stagger_rings: bool = True) -> List[OrbitPose]:
    """Rings of ``n_azimuths`` evenly spaced views at each elevation.

    Rings are azimuth-staggered against each other (i-th ring shifted by
    ``i * spacing / n_rings``) so stacked rings do not share view directions.
    """
    assert n_azimuths >= 1 and radius > 0, (n_azimuths, radius)
    lookat_t = tuple(float(x) for x in np.asarray(lookat, dtype=np.float64).reshape(3))
    spacing = 360.0 / n_azimuths
    poses: List[OrbitPose] = []
    for i, el in enumerate(ring_elevations):
        ring_off = az_offset_deg + (
            i * spacing / max(len(ring_elevations), 1) if stagger_rings else 0.0)
        for j in range(n_azimuths):
            poses.append(OrbitPose(lookat_t, float(radius),
                                   (ring_off + j * spacing) % 360.0, float(el)))
    return poses


def background_orbit_poses(lookat: Sequence[float], radius: float,
                           radius_for_azimuth: Optional[Callable[[float], float]]
                           = None) -> List[OrbitPose]:
    """Plan §5.1 background orbit: rings at 25°/50° × 24 azimuths + 8 top-down
    (80°) = 56 views.

    ``radius_for_azimuth`` (az_deg → radius m), when given, replaces each
    pose's distance with a per-azimuth value — the interior-safe wall clamp
    (:func:`wall_distance_2d` + :func:`interior_orbit_radius`): LIBERO rooms
    can be smaller than the requested orbit, leaving cameras outside staring
    at wall backfaces."""
    poses = orbit_poses(lookat, radius, (25.0, 50.0), 24)
    poses += orbit_poses(lookat, radius, (80.0,), 8, az_offset_deg=10.0)
    assert len(poses) == 56
    if radius_for_azimuth is not None:
        poses = [replace(p, distance=float(radius_for_azimuth(p.azimuth_deg)))
                 for p in poses]
        assert all(p.distance > 0 for p in poses), "non-positive clamped radius"
    return poses


def wall_distance_2d(model, data, lookat_xy: Sequence[float],
                     az_deg: float) -> float:
    """Distance (m) from the lookat, measured in the xy plane toward the
    CAMERA position of an orbit view at azimuth ``az_deg``, to the nearest
    wall geom's 2D AABB.

    The free camera sits at ``eye = lookat − distance · forward`` with
    ``forward_xy ∝ [cos az, sin az]``, so the camera direction from the lookat
    is ``−forward``: the ray marched here is ``lookat_xy + t · (−[cos az,
    sin az])``, t ≥ 0. Wall geoms are box geoms whose geom OR body name
    contains 'wall'; each contributes the axis-aligned xy box
    ``data.geom_xpos ± |R| @ geom_size``. Returns +inf when no wall geom
    intersects the ray (callers min() against a requested radius)."""
    o = np.asarray(lookat_xy, dtype=np.float64).reshape(-1)[:2]
    az = math.radians(float(az_deg))
    d = -np.array([math.cos(az), math.sin(az)])  # toward the camera, xy unit
    geom_type = np.asarray(model.geom_type)
    geom_bodyid = np.asarray(model.geom_bodyid)
    best = math.inf
    for gid in range(int(model.ngeom)):
        if int(geom_type[gid]) != _MJ_GEOM_BOX:
            continue
        geom_name = (model.geom_id2name(int(gid)) or "").lower()
        body_name = (model.body_id2name(int(geom_bodyid[gid])) or "").lower()
        if "wall" not in geom_name and "wall" not in body_name:
            continue
        center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        ext = (np.abs(R) @ np.asarray(model.geom_size[gid], dtype=np.float64))[:2]
        mn, mx = center[:2] - ext, center[:2] + ext
        # 2D slab ray/AABB intersection along o + t*d
        t_enter, t_exit, hit = -math.inf, math.inf, True
        for ax in range(2):
            if abs(d[ax]) < 1e-12:
                if not (mn[ax] <= o[ax] <= mx[ax]):
                    hit = False
                    break
                continue
            t1 = (mn[ax] - o[ax]) / d[ax]
            t2 = (mx[ax] - o[ax]) / d[ax]
            t_enter = max(t_enter, min(t1, t2))
            t_exit = min(t_exit, max(t1, t2))
        if not hit or t_enter > t_exit or t_exit < 0:
            continue
        best = min(best, t_enter if t_enter >= 0 else t_exit)
    return best


def interior_orbit_radius(requested_r: float, wall_dist_m: float,
                          table_half_diag_m: float,
                          wall_margin_m: float = 0.30,
                          floor_scale: float = 1.05,
                          floor_pad_m: float = 0.10) -> float:
    """Interior-safe per-azimuth background orbit radius:
    ``min(requested_r, wall_dist − wall_margin)``, floored at
    ``floor_scale · (table_half_diag + floor_pad)`` so the camera never sits
    over the table itself; when the floor exceeds the wall clamp the wall
    clamp wins (camera close to the wall but still inside the room)."""
    r_wall = float(wall_dist_m) - float(wall_margin_m)
    floor = float(floor_scale) * (float(table_half_diag_m) + float(floor_pad_m))
    r = min(max(min(float(requested_r), r_wall), floor), r_wall)
    assert r > 0, (
        f"interior orbit radius {r:.3f} m <= 0 (wall {wall_dist_m:.3f} m, "
        f"margin {wall_margin_m} m) — lookat essentially on a wall")
    return r


def object_orbit_poses(lookat: Sequence[float], radius: float) -> List[OrbitPose]:
    """Plan §5.1 object orbit: rings at −20°/20°/55° × 16 azimuths = 48 views
    (the −20° ring sees the underside of the floated object)."""
    poses = orbit_poses(lookat, radius, (-20.0, 20.0, 55.0), 16)
    assert len(poses) == 48
    return poses


def robot_orbit_poses(lookat: Sequence[float], radius: float,
                      az_offset_deg: float = 0.0) -> List[OrbitPose]:
    """Robot per-config orbit: rings at 20°/50° × 8 azimuths = 16 views; the
    per-config ``az_offset_deg`` spreads directions across configs."""
    poses = orbit_poses(lookat, radius, (20.0, 50.0), 8, az_offset_deg=az_offset_deg)
    assert len(poses) == 16
    return poses


# ── renderer (M0 recipe + facts F2b) ─────────────────────────────────────────

def make_renderer(env, image_size: int):
    """``mujoco.Renderer`` over the raw model handle (M0 recipe, G8: capture
    resolution is square ``image_size``)."""
    import mujoco
    return mujoco.Renderer(_raw_model(env.sim.model),
                           height=int(image_size), width=int(image_size))


@dataclass(frozen=True)
class SceneOption:
    """The measured F2b visualization state (G7): an ``mujoco.MjvOption``
    carrying ONLY the geom/site group toggles, plus the ``mjtRndFlag`` values
    to clear on ``renderer.scene.flags`` after ``update_scene`` — the facts
    producer (scripts/gsaug/probe_render_facts.py RawRenderer, F2B_RND_FLAGS)
    applies ``flags_off`` to the SCENE flags, never to ``MjvOption.flags``."""

    mjv_option: object                  # mujoco.MjvOption
    rnd_flags_off: Tuple[int, ...]      # mujoco.mjtRndFlag values


def scene_option_from_facts(facts: dict) -> SceneOption:
    """:class:`SceneOption` realizing the measured F2b visualization flags.

    ``MjvOption`` gets only the geom/site group toggles; ``flags_off`` entries
    are ``mjRND_*`` scene-flag names (the facts producer's sweep space) that
    :func:`render_view` clears on ``renderer.scene.flags`` after
    ``update_scene``, mirroring how the producer applied them. Raw-Renderer
    output only matches robosuite obs rendering under these settings (G7)."""
    import mujoco
    flags = facts["F2b"]["flags"]
    opt = mujoco.MjvOption()
    for attr in ("geomgroup", "sitegroup"):
        values = flags.get(attr)
        assert values is not None and len(values) == len(getattr(opt, attr)), (
            f"facts F2b flags[{attr!r}] = {values!r}: expected "
            f"{len(getattr(opt, attr))} 0/1 entries (G7)")
        getattr(opt, attr)[:] = np.asarray(values, dtype=np.uint8)
    rnd_off = []
    for name in flags.get("flags_off", []):
        flag = getattr(mujoco.mjtRndFlag, name, None)
        if flag is None:
            raise RuntimeError(
                f"facts F2b flags_off entry {name!r} is not a "
                f"mujoco.mjtRndFlag scene flag — facts file and mujoco "
                f"version disagree (G7)")
        rnd_off.append(int(flag))
    return SceneOption(mjv_option=opt, rnd_flags_off=tuple(rnd_off))


def free_camera_fovy(env) -> float:
    """Vertical FOV (deg) MuJoCo uses for FREE cameras: ``model.vis.global_.fovy``."""
    return float(_raw_model(env.sim.model).vis.global_.fovy)


def free_camera_K(env, image_size: int) -> np.ndarray:
    """Pinhole intrinsics for orbit captures at ``image_size``² (G8)."""
    return fovy_to_K(free_camera_fovy(env), int(image_size), int(image_size))


def render_view(renderer, data, cam, scene_option=None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One view → (rgb uint8 (H,W,3), depth float32 meters (H,W), seg int32
    geom ids (H,W), −1 = none).

    Depth from ``mujoco.Renderer`` is already metric (M0). The seg render
    returns (H, W, 2) int32 — channel 0 is the object id, channel 1 the object
    type; under the F2b flags every visible object must be a geom, which is
    asserted (a leaked site/other type would corrupt geom-id maps, F6).

    ``scene_option`` may be a :class:`SceneOption` (the F2b contract: its
    MjvOption drives ``update_scene`` and its mjRND_* entries are cleared on
    ``renderer.scene.flags`` afterwards — the same scene the three renders
    below share) or a bare ``mujoco.MjvOption``.
    """
    import mujoco
    raw = _raw_data(data)
    opt, rnd_flags_off = scene_option, ()
    if isinstance(scene_option, SceneOption):
        opt = scene_option.mjv_option
        rnd_flags_off = scene_option.rnd_flags_off
    renderer.update_scene(raw, camera=cam, scene_option=opt)
    for flag in rnd_flags_off:
        renderer.scene.flags[flag] = 0

    rgb = renderer.render()
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, (
        f"unexpected rgb render {rgb.shape} {rgb.dtype}")

    renderer.enable_depth_rendering()
    depth = np.asarray(renderer.render(), dtype=np.float32)
    renderer.disable_depth_rendering()
    assert depth.shape == rgb.shape[:2], f"depth shape {depth.shape}"

    renderer.enable_segmentation_rendering()
    seg2 = renderer.render()
    renderer.disable_segmentation_rendering()
    assert seg2.ndim == 3 and seg2.shape[2] == 2 and seg2.shape[:2] == rgb.shape[:2], (
        f"unexpected seg render shape {seg2.shape}; expected (H, W, 2) int32 (M0)")
    seg = np.ascontiguousarray(seg2[..., 0]).astype(np.int32)
    obj_types = np.unique(seg2[..., 1])
    bad = [int(t) for t in obj_types
           if int(t) not in (-1, int(mujoco.mjtObj.mjOBJ_GEOM))]
    assert not bad, (
        f"seg render contains non-geom object types {bad} (mjtObj values) — "
        f"the F2b flags should hide sites/decor; re-run probe_render_facts (G7)")
    return rgb, depth, seg


def achieved_gl_c2w(scene) -> np.ndarray:
    """OpenCV c2w of the camera MuJoCo actually placed in ``renderer.scene``
    (average of the two stereo mjvGLCamera halves — mono rendering uses their
    average, which cancels the ipd offset)."""
    cams = scene.camera
    pos = (np.asarray(cams[0].pos, dtype=np.float64)
           + np.asarray(cams[1].pos, dtype=np.float64)) / 2.0
    fwd = _unit(cams[0].forward)
    up = _unit(cams[0].up)
    right = _unit(np.cross(fwd, up))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up   # OpenCV y is down
    c2w[:3, 2] = fwd
    c2w[:3, 3] = pos
    return c2w


def assert_free_camera_pose(scene, c2w_expected: np.ndarray, tag: str,
                            tol: float = _GL_POSE_TOL) -> None:
    """Per-view extrinsics check: the analytic pose we RECORD must equal the GL
    camera MuJoCo RENDERED with (float32 storage → loose-ish tol). Catches any
    azimuth/elevation/forward-formula drift immediately (G7)."""
    achieved = achieved_gl_c2w(scene)
    err = float(np.abs(achieved - np.asarray(c2w_expected, dtype=np.float64)).max())
    if err > tol:
        raise AssertionError(
            f"{tag}: analytic free-camera c2w disagrees with the scene GL "
            f"camera (max|Δ|={err:.3e} > {tol}); recorded transforms.json "
            f"extrinsics would be wrong — free-camera convention drifted (G7).\n"
            f"analytic:\n{np.round(c2w_expected, 5)}\nachieved:\n"
            f"{np.round(achieved, 5)}")


# ── once-per-capture pixel-chain validation (F1/F5 pattern) ─────────────────

def seg_landmark_error_px(model, data, K: np.ndarray, c2w: np.ndarray,
                          seg: np.ndarray, depth: np.ndarray,
                          min_px: int = 20, max_px_count: int = 5000,
                          max_candidates: int = 8, max_rbound_m: float = 0.2,
                          min_fill_frac: float = 0.15) -> float:
    """Project centered-primitive geom centers through (K, w2c) and compare
    against their seg-mask centroids (F1 pattern); returns the best (minimum)
    error in pixels over up to ``max_candidates`` fully-visible small geoms.

    Mask-centroid ≈ projected-center only holds for PHYSICALLY small, mostly
    visible geoms, so candidates must additionally pass:

    * ``geom_rbound <= max_rbound_m`` — room-scale boxes (walls, table tops)
      can leave a small in-frame sliver that passes the pixel-count and
      frame-crop filters while its centroid has nothing to do with the
      projected geom center (measured 695 px on 'wall_rightcorner_visual');
    * a projected-size consistency guard against occlusion-truncated slivers:
      with r_px = fy · rbound / depth(centroid), the mask must cover at least
      ``min_fill_frac · π · r_px²`` pixels (loose lower bound on the visible
      fraction of the geom's bounding disk).

    Raises ``LookupError`` when the view contains no usable landmark (all-mesh
    scenes) — callers fall back to :func:`table_depth_error_m`.
    """
    H, W = seg.shape
    depth = np.asarray(depth)
    assert depth.shape == seg.shape, (depth.shape, seg.shape)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    geom_type = np.asarray(model.geom_type)
    geom_rbound = np.asarray(model.geom_rbound)
    fy = float(np.asarray(K, dtype=np.float64)[1, 1])
    ids, counts = np.unique(seg[seg >= 0], return_counts=True)
    order = np.argsort(counts)  # smallest geoms first: closest to point landmarks
    best: Optional[float] = None
    n_cand = 0
    for idx in order:
        gid, cnt = int(ids[idx]), int(counts[idx])
        if cnt < min_px or cnt > max_px_count:
            continue
        if int(geom_type[gid]) not in _CENTERED_GEOM_TYPES:
            continue  # mesh/plane origins are not visual centers
        if float(geom_rbound[gid]) > max_rbound_m:
            continue  # room/table-scale primitive: sliver centroid ≠ center
        ys, xs = np.nonzero(seg == gid)
        if xs.min() < 2 or ys.min() < 2 or xs.max() >= W - 2 or ys.max() >= H - 2:
            continue  # cropped by the frame → centroid biased
        centroid = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
        d = float(depth[int(np.clip(round(ys.mean()), 0, H - 1)),
                        int(np.clip(round(xs.mean()), 0, W - 1))])
        if not (np.isfinite(d) and d > 1e-6):
            continue
        r_px = fy * float(geom_rbound[gid]) / d
        if cnt < min_fill_frac * math.pi * r_px * r_px:
            continue  # mostly occluded → centroid biased
        uv = project(K, w2c, np.asarray(data.geom_xpos[gid], dtype=np.float64))
        if not np.all(np.isfinite(uv)):
            continue
        err = float(np.linalg.norm(uv - centroid))
        best = err if best is None else min(best, err)
        n_cand += 1
        if n_cand >= max_candidates:
            break
    if best is None:
        raise LookupError("no centered-primitive landmark geom in this view")
    return best


def _median_plane_dz_m(K: np.ndarray, c2w: np.ndarray, seg: np.ndarray,
                       depth: np.ndarray, gid: int, plane_z: float,
                       min_px: int, max_samples: int) -> Optional[float]:
    """Backproject ``seg == gid`` pixels through (K, c2w) with metric z-depth
    (M0) and return the median |world z − plane_z| in meters (None if too few
    finite-depth pixels)."""
    ys, xs = np.nonzero(seg == gid)
    sel = np.linspace(0, len(xs) - 1, min(max_samples, len(xs))).astype(int)
    ys, xs = ys[sel], xs[sel]
    d = depth[ys, xs].astype(np.float64)
    finite = np.isfinite(d)
    if finite.sum() < min_px // 2:
        return None
    ys, xs, d = ys[finite], xs[finite], d[finite]
    uv1 = np.stack([xs + 0.5, ys + 0.5, np.ones_like(xs, dtype=np.float64)], axis=1)
    rays = uv1 @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T  # z-component 1
    pts_c = rays * d[:, None]  # metric z-depth (M0)
    c2w = np.asarray(c2w, dtype=np.float64)
    pts_w = pts_c @ c2w[:3, :3].T + c2w[:3, 3]
    return float(np.median(np.abs(pts_w[:, 2] - float(plane_z))))


def table_depth_error_m(model, data, K: np.ndarray, c2w: np.ndarray,
                        seg: np.ndarray, depth: np.ndarray,
                        min_px: int = 300, max_samples: int = 2000
                        ) -> Optional[float]:
    """F5-pattern fallback: backproject the largest visible 'table' box geom's
    pixels through (K, c2w) and compare world z against that box's analytic top
    plane; returns the median |error| in meters, or None if no table box is
    visible. Only meaningful for downward-looking views (callers gate on
    elevation) where the visible face is the top face; the median is robust to
    edge pixels showing side faces."""
    geom_type = np.asarray(model.geom_type)
    geom_bodyid = np.asarray(model.geom_bodyid)
    ids, counts = np.unique(seg[seg >= 0], return_counts=True)
    cands = []
    for gid, cnt in zip(ids.tolist(), counts.tolist()):
        if cnt < min_px or int(geom_type[gid]) != _MJ_GEOM_BOX:
            continue
        body_name = (model.body_id2name(int(geom_bodyid[gid])) or "").lower()
        geom_name = (model.geom_id2name(int(gid)) or "").lower()
        if "table" not in body_name and "table" not in geom_name:
            continue
        cands.append((int(cnt), int(gid)))
    if not cands:
        return None
    _, gid = max(cands)
    center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    top_z = float(center[2] + (np.abs(R) @ size)[2])
    return _median_plane_dz_m(K, c2w, seg, depth, gid, top_z, min_px, max_samples)


def floor_depth_error_m(model, data, K: np.ndarray, c2w: np.ndarray,
                        seg: np.ndarray, depth: np.ndarray,
                        min_px: int = 300, max_samples: int = 2000
                        ) -> Optional[float]:
    """F5-pattern fallback for scenes with NO visible 'table' box geom (LIBERO
    living rooms: the visible table is a MESH whose top sits ~11 mm off the
    hidden group-0 collision boxes — measured, so the box plane cannot gate a
    mesh surface): backproject the largest visible world-horizontal 'floor'
    PLANE geom's pixels and compare world z against the plane's analytic
    height. An exact analytic plane with no side faces or legs — measured
    0.2–1.3 mm median on a correct chain. Returns None when no such floor
    plane is visible."""
    geom_type = np.asarray(model.geom_type)
    geom_bodyid = np.asarray(model.geom_bodyid)
    ids, counts = np.unique(seg[seg >= 0], return_counts=True)
    cands = []
    for gid, cnt in zip(ids.tolist(), counts.tolist()):
        if cnt < min_px or int(geom_type[gid]) != _MJ_GEOM_PLANE:
            continue
        body_name = (model.body_id2name(int(geom_bodyid[gid])) or "").lower()
        geom_name = (model.geom_id2name(int(gid)) or "").lower()
        if "floor" not in body_name and "floor" not in geom_name:
            continue
        R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        if R[2, 2] < 0.999:
            continue  # not a world-horizontal plane: no analytic z height
        cands.append((int(cnt), int(gid)))
    if not cands:
        return None
    _, gid = max(cands)
    plane_z = float(np.asarray(data.geom_xpos[gid], dtype=np.float64)[2])
    return _median_plane_dz_m(K, c2w, seg, depth, gid, plane_z, min_px, max_samples)


class PoseValidator:
    """Once-per-capture validation of the RECORDED (K, c2w) chain against
    rendered pixels (plan §5.1 / F1 pattern, G7).

    Feed every view via :meth:`try_view` while its sim state is still current;
    the first view with a seg landmark decides (hard ≤ ``landmark_max_px``).
    If NO view offers a landmark (all-mesh close-ups), the best depth-planarity
    result over downward views (elevation ≥ 30°) gates at ``table_max_med_m``
    in :meth:`finalize` — table-top where a 'table' box geom is visible, else
    the 'floor' plane geom (living-room scenes render only a MESH table)."""

    def __init__(self, model, K: np.ndarray,
                 landmark_max_px: float = _LANDMARK_MAX_PX,
                 table_max_med_m: float = _TABLE_DEPTH_MAX_MED_M,
                 max_table_evals: int = 6):
        self.model = model
        self.K = np.asarray(K, dtype=np.float64)
        self.landmark_max_px = float(landmark_max_px)
        self.table_max_med_m = float(table_max_med_m)
        self.result: Optional[dict] = None
        # best depth-planarity fallback so far: (med_m, view, method)
        self._best_table: Optional[Tuple[float, int, str]] = None
        self._table_evals_left = int(max_table_evals)

    @property
    def done(self) -> bool:
        return self.result is not None

    def try_view(self, data, c2w: np.ndarray, seg: np.ndarray, depth: np.ndarray,
                 view_idx: int, elevation_deg: float, tag: str) -> None:
        if self.done:
            return
        try:
            err = seg_landmark_error_px(self.model, data, self.K, c2w, seg, depth)
        except LookupError:
            err = None
        if err is not None:
            if err > self.landmark_max_px:
                raise AssertionError(
                    f"{tag}: seg-landmark projection error {err:.2f} px > "
                    f"{self.landmark_max_px} px at view {view_idx} — the recorded "
                    f"(K, c2w) chain does not match rendered pixels (wrong "
                    f"free-camera fovy/flip/convention, G7). Fix before training "
                    f"assets on these captures.")
            self.result = {"method": "seg_landmark", "err_px": float(err),
                           "view": int(view_idx)}
            return
        if elevation_deg >= 30.0 and self._table_evals_left > 0:
            self._table_evals_left -= 1
            med = table_depth_error_m(self.model, data, self.K, c2w, seg, depth)
            method = "table_depth"
            if med is None:
                med = floor_depth_error_m(self.model, data, self.K, c2w, seg,
                                          depth)
                method = "floor_depth"
            if med is not None and (self._best_table is None
                                    or med < self._best_table[0]):
                self._best_table = (float(med), int(view_idx), method)

    def finalize(self, tag: str) -> dict:
        if self.result is not None:
            return self.result
        if self._best_table is not None:
            med, view, method = self._best_table
            if med > self.table_max_med_m:
                raise AssertionError(
                    f"{tag}: no seg landmark available and {method} "
                    f"backprojection off-plane by median {med * 1e3:.1f} mm > "
                    f"{self.table_max_med_m * 1e3:.0f} mm (view {view}) — the "
                    f"recorded (K, c2w) chain does not match rendered pixels (G7).")
            self.result = {"method": method, "med_err_m": med, "view": view}
            return self.result
        raise AssertionError(
            f"{tag}: could not validate the recorded camera chain against pixels "
            f"— no centered-primitive landmark in any view AND no visible "
            f"'table' box geom or 'floor' plane geom in any downward view. "
            f"Refusing unvalidated captures (G7); widen the orbit or add a "
            f"validation view.")


# ── state edits & hide mechanisms (facts F4) ────────────────────────────────

def get_flat_state(env) -> np.ndarray:
    """Flattened ``[time, qpos, qvel]`` sim state as an owned float64 copy."""
    return np.array(env.sim.get_state().flatten(), dtype=np.float64)


def set_flat_state(env, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64).copy())
    env.sim.forward()


def graveyard_movables(env, addr, exclude: Iterable[str] = (),
                       offset_xy: float = GRAVEYARD_XY_OFFSET) -> np.ndarray:
    """Teleport movable free-joint objects (except ``exclude``) far away
    (qpos xy += ``offset_xy``, facts F4-a), apply via set_state + forward, and
    return the applied flattened state."""
    exclude = set(exclude)
    unknown = exclude - set(addr.obj_qpos_slices)
    assert not unknown, (
        f"graveyard exclude names {sorted(unknown)} are not movable free "
        f"joints; known: {list(addr.obj_qpos_slices)}")
    state = get_flat_state(env)
    for name, (start, _end) in addr.obj_qpos_slices.items():
        if name in exclude:
            continue
        state[1 + start:1 + start + 2] += offset_xy  # +1: time slot
    set_flat_state(env, state)
    return state


def float_object_state(state: np.ndarray, addr, joint_name: str,
                       dz: float) -> np.ndarray:
    """Pure edit: raise one movable's free-joint z by ``dz`` (plan §5.1 float —
    the underside becomes visible to the −20° ring). Returns a new state."""
    assert joint_name in addr.obj_qpos_slices, (
        f"{joint_name!r} is not a movable free joint; known: "
        f"{list(addr.obj_qpos_slices)}")
    out = np.asarray(state, dtype=np.float64).copy()
    start, _end = addr.obj_qpos_slices[joint_name]
    out[1 + start + 2] += float(dz)
    return out


class RobotHide:
    """Restore handle for :func:`hide_robot_alpha0` (also a context manager)."""

    def __init__(self, model, geom_ids: np.ndarray, saved_alpha: np.ndarray):
        self._model = model
        self._geom_ids = geom_ids
        self._saved = saved_alpha
        self._restored = False

    def restore(self) -> None:
        if not self._restored:
            self._model.geom_rgba[self._geom_ids, 3] = self._saved
            self._restored = True

    def __enter__(self) -> "RobotHide":
        return self

    def __exit__(self, *exc) -> None:
        self.restore()


def hide_robot_alpha0(env) -> RobotHide:
    """Hide the robot stack by zeroing geom alpha (facts F4-b: alpha-0 geoms are
    dropped from the scene, so seg ids AND cast shadows vanish). Returns a
    restore handle; model edits are in-place on the live model the renderer
    shares."""
    model = env.sim.model
    gids = np.asarray(robot_geom_ids(model), dtype=np.int64)
    assert gids.size > 0, "no robot geoms found to hide (prefixes " \
        f"{ROBOT_BODY_PREFIXES}) — unexpected for a LIBERO scene"
    saved = model.geom_rgba[gids, 3].copy()
    model.geom_rgba[gids, 3] = 0.0
    return RobotHide(model, gids, saved)


# ── id maps & bounding volumes (facts F6 companions) ────────────────────────

def body_subtree_ids(model, root_bid: int) -> List[int]:
    """All body ids in the kinematic subtree rooted at ``root_bid``."""
    parent = np.asarray(model.body_parentid)
    out = []
    for b in range(int(model.nbody)):
        x = b
        while x != 0 and x != root_bid:
            x = int(parent[x])
        if x == root_bid:
            out.append(b)
    assert out, f"body {root_bid} has an empty subtree"
    return out


def geom_ids_of_bodies(model, body_ids: Iterable[int]) -> List[int]:
    body_set = set(int(b) for b in body_ids)
    geom_bodyid = np.asarray(model.geom_bodyid)
    return [g for g in range(int(model.ngeom)) if int(geom_bodyid[g]) in body_set]


def object_body_id(model, joint_name: str) -> int:
    """Body carrying a movable object's free joint (compose contract: joint
    name → body via ``model.jnt_bodyid``)."""
    jid = model.joint_name2id(joint_name)
    return int(np.asarray(model.jnt_bodyid)[jid])


def object_geom_ids(model, joint_name: str) -> List[int]:
    """All geoms in the object's body subtree (an object may have child bodies)."""
    gids = geom_ids_of_bodies(model, body_subtree_ids(model, object_body_id(model, joint_name)))
    assert gids, f"movable object {joint_name!r} has no geoms"
    return gids


def movable_geom_ids(model, addr) -> Dict[str, List[int]]:
    """Free-joint name → geom ids, for every movable in ``addr`` (F6)."""
    return {name: object_geom_ids(model, name) for name in addr.obj_qpos_slices}


def robot_body_ids(model, prefixes: Tuple[str, ...] = ROBOT_BODY_PREFIXES,
                   with_geoms_only: bool = False) -> List[int]:
    body_geomnum = np.asarray(model.body_geomnum)
    out = []
    for b in range(int(model.nbody)):
        name = model.body_id2name(b) or ""
        if not name.startswith(prefixes):
            continue
        if with_geoms_only and int(body_geomnum[b]) == 0:
            continue
        out.append(b)
    return out


def robot_geom_ids(model) -> List[int]:
    """Geoms of every robot-stack body (robot0_/gripper0_/mount0_)."""
    return geom_ids_of_bodies(model, robot_body_ids(model))


def geoms_world_aabb(model, data, geom_ids: Sequence[int]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative world AABB over geoms via bounding spheres
    (``geom_xpos ± geom_rbound`` — exact enough for orbit-radius fitting and
    works for meshes whose ``geom_size`` is empty)."""
    gids = np.asarray(list(geom_ids), dtype=np.int64)
    assert gids.size > 0, "empty geom set"
    centers = np.asarray(data.geom_xpos)[gids].astype(np.float64)
    r = np.asarray(model.geom_rbound)[gids].astype(np.float64)[:, None]
    return (centers - r).min(axis=0), (centers + r).max(axis=0)


def table_top_z(model, data, addr, z_below: float = 0.30,
                z_above: float = 0.02) -> float:
    """World z of the table-top surface: highest top face among 'table' box
    geoms within the same z window ``table_top_xy_aabb`` uses (anchored at the
    lowest movable resting z — call with movables at their reset placement)."""
    geom_bodyid = np.asarray(model.geom_bodyid)
    geom_type = np.asarray(model.geom_type)
    qpos = np.asarray(data.qpos)
    obj_z = [float(qpos[start + 2]) for start, _end in addr.obj_qpos_slices.values()]
    assert obj_z, "no movable objects — cannot anchor the table-top z window"
    obj_min_z = min(obj_z)
    tops = []
    for gid in range(int(model.ngeom)):
        if int(geom_type[gid]) != _MJ_GEOM_BOX:
            continue
        body_name = (model.body_id2name(int(geom_bodyid[gid])) or "").lower()
        geom_name = (model.geom_id2name(gid) or "").lower()
        if "table" not in body_name and "table" not in geom_name:
            continue
        center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        size = np.asarray(model.geom_size[gid], dtype=np.float64)
        top = float(center[2] + (np.abs(R) @ size)[2])
        if obj_min_z - z_below <= top <= obj_min_z + z_above:
            tops.append(top)
    assert tops, (
        f"no 'table' box geom top surface within [{obj_min_z - z_below:.3f}, "
        f"{obj_min_z + z_above:.3f}] — table detection failed (see "
        f"table_top_xy_aabb for the same window)")
    return max(tops)


def canonical_model_xml(env) -> str:
    """The compiled model XML with render-buffer attributes stripped.

    G9 binds assets to the task SCENE, not to the offscreen framebuffer size:
    robosuite serializes ``<global offwidth/offheight>`` only when non-default,
    so a 512² capture env and a 128² composite-render env differ by exactly
    that one line (measured on LIVING_ROOM_SCENE2). Hashing the raw XML made
    the G9 gate reject its own assets at render time."""
    import re
    xml = env.sim.model.get_xml()
    xml = re.sub(r'\s+off(width|height)="\d+"', "", xml)
    xml = re.sub(r'\n\s*<global\s*/>', "", xml)
    return xml


def model_xml_sha1(env) -> str:
    """THE model-hash recipe (G9) — pinned; every consumer compares this exact
    quantity: sha1 over the CANONICALIZED model XML (see canonical_model_xml)."""
    import hashlib
    return hashlib.sha1(canonical_model_xml(env).encode()).hexdigest()


# ── capture file IO ──────────────────────────────────────────────────────────

def _imwrite(path: str, arr: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, arr)
    except ImportError:
        from PIL import Image
        Image.fromarray(arr).save(path)


def _imread(path: str) -> np.ndarray:
    assert os.path.exists(path), f"capture file missing: {path}"
    try:
        import imageio.v2 as imageio
        return np.asarray(imageio.imread(path))
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path))


def write_view_files(out_dir: str, view_idx: int, rgb: np.ndarray,
                     depth: np.ndarray, seg: np.ndarray) -> str:
    """Write one view's rgb/depth/seg per the pinned layout; returns the
    relative file prefix recorded in transforms.json. Seg is stored uint16 as
    geom-id + 1 (0 = none)."""
    prefix = VIEW_FILE_FMT.format(view_idx)
    base = os.path.join(out_dir, prefix)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, rgb.shape
    _imwrite(base + ".png", rgb)
    np.save(base + "_depth.npy", np.asarray(depth, dtype=np.float32))
    seg = np.asarray(seg)
    assert seg.min() >= -1, f"seg ids below -1 ({seg.min()})"
    assert seg.max() + 1 < 2 ** 16, (
        f"geom id {seg.max()} does not fit uint16 geom-id+1 storage")
    _imwrite(base + "_seg.png", (seg.astype(np.int64) + 1).astype(np.uint16))
    return prefix


def write_mask_file(masks_dir: str, view_idx: int, mask: np.ndarray) -> str:
    """Binary uint8 mask PNG (255 = masked pixel), robot_hide=='masked' mode."""
    name = VIEW_FILE_FMT.format(view_idx) + "_mask.png"
    _imwrite(os.path.join(masks_dir, name),
             (np.asarray(mask, dtype=bool).astype(np.uint8) * 255))
    return name


def write_transforms(out_dir: str, transforms: dict) -> str:
    path = os.path.join(out_dir, TRANSFORMS_NAME)
    with open(path, "w") as f:
        json.dump(transforms, f, indent=2, sort_keys=True)
    return path


# ── the trainer-facing loader ────────────────────────────────────────────────

@dataclass
class CaptureView:
    """One loaded capture view (dataset orientation of the RAW renderer — i.e.
    NOT vertically flipped; the F2 flip chain applies to composite renders,
    not to captures, which are self-consistent with their recorded c2w)."""

    file_prefix: str
    rgb: np.ndarray            # uint8 (H, W, 3)
    depth: np.ndarray          # float32 (H, W), meters
    seg: np.ndarray            # int32 (H, W), geom ids, -1 = none
    K: np.ndarray              # (3, 3) float64
    c2w: np.ndarray            # (4, 4) float64, OpenCV camera-to-world
    cam_params: dict = field(default_factory=dict)
    mask: Optional[np.ndarray] = None  # bool (H, W); robot mask in 'masked' mode


class CaptureBundle:
    """A loaded per-component capture directory (the pinned trainer contract):
    ``bundle.views`` (each with .rgb/.depth/.seg/.K/.c2w) and
    ``bundle.transforms`` (the raw transforms.json dict)."""

    def __init__(self, directory: str, transforms: dict, views: List[CaptureView]):
        self.directory = directory
        self.transforms = transforms
        self.views = views

    # convenience accessors over the pinned schema
    @property
    def component(self) -> str:
        return self.transforms["component"]

    @property
    def task(self) -> str:
        return self.transforms["task"]

    @property
    def image_size(self) -> int:
        return int(self.transforms["image_size"])

    @property
    def model_xml_sha1(self) -> str:
        return self.transforms["model_xml_sha1"]

    @property
    def configs(self) -> Optional[list]:
        """Robot captures only: [{'qpos', 'view_ids', 'link_poses'}, ...]."""
        return self.transforms.get("configs")

    @property
    def body_pose(self) -> Optional[dict]:
        """Object captures only: {'p': [3], 'q_wxyz': [4]} at capture."""
        return self.transforms.get("body_pose")

    def __len__(self) -> int:
        return len(self.views)

    @classmethod
    def load(cls, directory: str) -> "CaptureBundle":
        tf_path = os.path.join(directory, TRANSFORMS_NAME)
        if not os.path.exists(tf_path):
            raise FileNotFoundError(
                f"no {TRANSFORMS_NAME} in {directory!r} — not a capture "
                f"directory (run scripts/gsaug/capture_assets.py first)")
        with open(tf_path) as f:
            transforms = json.load(f)
        for key in ("image_size", "views", "component", "task", "model_xml_sha1"):
            assert key in transforms, (
                f"{tf_path}: missing required key {key!r} (capture schema)")
        size = int(transforms["image_size"])
        K_shared = (np.asarray(transforms["K"], dtype=np.float64)
                    if "K" in transforms else None)
        masks_dir = transforms.get("masks_dir")

        views: List[CaptureView] = []
        for v in transforms["views"]:
            prefix = v["file_prefix"]
            base = os.path.join(directory, prefix)
            rgb = _imread(base + ".png")
            assert rgb.dtype == np.uint8 and rgb.shape == (size, size, 3), (
                f"{base}.png: expected uint8 ({size},{size},3), got "
                f"{rgb.dtype} {rgb.shape}")
            depth = np.load(base + "_depth.npy")
            assert depth.shape == (size, size), f"{base}_depth.npy: {depth.shape}"
            seg_raw = _imread(base + "_seg.png")
            assert seg_raw.ndim == 2 and seg_raw.shape == (size, size), (
                f"{base}_seg.png: expected ({size},{size}) uint16, got "
                f"{seg_raw.dtype} {seg_raw.shape}")
            seg = seg_raw.astype(np.int32) - 1  # 0 = none → -1
            K = np.asarray(v["K"], dtype=np.float64) if "K" in v else K_shared
            assert K is not None and K.shape == (3, 3), (
                f"{tf_path}: no K for view {prefix} (neither shared nor per-view)")
            c2w = np.asarray(v["c2w_opencv"], dtype=np.float64)
            assert c2w.shape == (4, 4), f"view {prefix}: c2w shape {c2w.shape}"
            mask = None
            if masks_dir:
                mpath = os.path.join(directory, masks_dir, prefix + "_mask.png")
                mask = _imread(mpath) > 0
                assert mask.shape == (size, size), f"{mpath}: {mask.shape}"
            views.append(CaptureView(
                file_prefix=prefix, rgb=rgb, depth=np.asarray(depth, np.float32),
                seg=seg, K=K, c2w=c2w, cam_params=v.get("cam_params", {}),
                mask=mask))
        assert views, f"{tf_path}: empty views list"
        return cls(directory, transforms, views)

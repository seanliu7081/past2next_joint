"""M1 GATING probe (Stage 0): measure renderer conventions -> gs_render_facts.json.

Conventions are measured, then asserted -- never assumed (G7). Every downstream
GS constructor (``GSCompositeRenderer``, capture, both GS probes, the GS
pre-render path) loads the facts file written here via
``oat.gsaug.cameras.load_render_facts`` and refuses to run unless ``pass`` is
true. Facts (plan §4):

  F1  camera convention -- which GL->CV axis flip in ``cameras.FLIP_CANDIDATES``
      makes ``fovy_to_K`` + ``mujoco_cam_to_w2c`` agree with the live renderer.
      Ground truth is render-derived, not assumed: (a) depth-backprojected
      pixels of each geom must land inside that geom's world bounding sphere
      (wrong flips rigidly rotate the cloud about the camera center -- near-zero
      hit rate); (b) the projection of a visible robot body's origin must land
      INSIDE its subtree seg mask (containment, robust to mask size, unlike
      centroids); (c) depth-backprojected table-top pixels must fit a
      horizontal plane at the table height. Note robosuite's Panda hangs the
      hand geoms off ``gripper0_right_gripper`` (``robot0_right_hand`` is a
      massless attach frame) and the wrist camera sits in the hand's own body
      plane, so the containment anchor is picked per camera: agentview ->
      robot0_right_hand origin vs its subtree mask; eye-in-hand -> a fingertip
      pad body (``*_tip``) vs the full visible robot silhouette (the only
      robot origins actually in front of and inside the wrist frame).
      A pass at both 128 (dataset-native) and 512 (capture) also pins that raw
      renderer row 0 is the OpenCV image TOP: a row inversion is det=-1 and no
      det=+1 flip candidate could compensate it.
  F2  image orientation -- raw-MuJoCo -> stored-zarr is np.flip(axis=0) (M0);
      gsplat -> stored-zarr is measured by rasterizing one Gaussian at the
      depth-backprojection of a known off-center raw pixel and checking which
      image half it lands in.
  F2b renderer vis parity -- sweep MjvOption geom/site groups x mjRND flags
      (<=64 combos) to match ``mujoco.Renderer`` output to the robosuite obs
      pipeline (M0 measured MAD ~15-30 at stock flags). All capture and
      oracle-comparison renders must use the winning flags.
  F3  headlight contribution (record-only, informs capture lighting policy).
  F4  hide mechanisms: (a) movables teleported to a graveyard (xy += 50) must
      vanish from seg with the rgb diff confined near their former silhouette;
      (b) robot geoms at rgba alpha=0 must vanish from seg with no far-field
      rgb residue (>40 px from the robot mask, scaled with resolution) --
      far-field residue (e.g. reflections) would mean alpha-0 capture is
      unsound and flips ``robot_hide`` to 'masked' (plan §4 F4).
  F5  depth sanity: depth is already metric (M0); backproject the largest
      'table' collision-box seg mask, SVD plane fit, planarity <= 2 mm RMS and
      height within 3 cm of the box top (loose bound documented in plan §4).
  F6  geom -> body -> component map: every geom classifies as exactly one of
      movable object (free-joint subtree from ``resolve_addresses``), robot
      (root body prefixed robot0/gripper0/mount0), or background. No orphans.
  F7  gsplat FPS at 128/512 with a 300k-Gaussian sh3 scene (record-only).

pass = AND(F1, F2, F2b, F4, F5, F6); F3/F7 are record-only.

Demo states: ``--n_states`` fresh ``env.reset()`` states passed through
``rewrite_state(theta=0)`` (an exact no-op that still exercises the aug state
path) -- no HDF5 dependency, the conventions under test are state-independent.

Usage:
    export PATH=/home/haotian/miniforge3/envs/oat/bin:/usr/local/cuda/bin:$PATH
    MUJOCO_GL=egl python scripts/gsaug/probe_render_facts.py \
        --out data/libero/gs_render_facts.json
"""

import os

# Must be set before robosuite / mujoco / libero are imported (one GL context
# per process).
os.environ.setdefault("MUJOCO_GL", "egl")

if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import datetime
import itertools
import json
import pathlib
import sys
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np
from scipy import ndimage

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.factory import get_subtasks, is_multitask
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.gsaug.cameras import (
    FLIP_CANDIDATES,
    c2w_from_w2c,
    facts_flip,
    facts_orientation_flip_ud,
    fovy_to_K,
    load_render_facts,
    lookat_c2w,
    mujoco_cam_to_w2c,
    project,
)

CAMS = ("agentview", "robot0_eye_in_hand")
ROBOT_BODY_PREFIXES = ("robot0", "gripper0", "mount0")
GEOM_T = int(mujoco.mjtObj.mjOBJ_GEOM)

# F1 gates. Containment tolerance scales with resolution (a body origin can sit
# a couple of pixels off its own visible silhouette); wrong flips miss by tens
# of pixels (180-degree image rotation) or project behind the camera (inf).
CONT_TOL_PX_AT_128 = 4.0
CONT_MIN_MASK_PX = 30
BS_FRAC_MIN = 0.90          # bounding-sphere hit rate for the winning flip
BS_PER_GEOM = 5
BS_MAX_PX = 240
PLANE_MIN_MASK_PX = 200     # skip the plane signal for a camera that barely sees the table
PLANE_NORMAL_MIN_Z = float(np.cos(np.deg2rad(3.0)))
PLANE_Z_TOL_M = 0.03

# F2b gate (plan §4) and sweep space (<=64 combos).
F2B_MAD_MAX = 2.0
F2B_RND_FLAGS = ("mjRND_SKYBOX", "mjRND_SHADOW", "mjRND_REFLECTION")

# F4 thresholds (plan §4 / pinned schema).
DIFF_THRESH = 3             # |uint8 diff| above this counts as a changed pixel
MOVABLE_DILATE_PX = 8
MOVABLE_OUTSIDE_FRAC_MAX = 0.01
ROBOT_FAR_PX_AT_128 = 40

# F5 gates.
PLANE_RMS_MAX_M = 0.002
F5_MIN_MASK_PX = 300

# F7 scene.
F7_N_GAUSSIANS = 300_000
F7_ITERS = 50
F7_WARMUP = 5


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def build_control_env(task_name: str, image_size: int, seed: int) -> ControlEnv:
    """Build a ControlEnv directly (mirrors scripts/probes/probe_controller_frame.py):
    LiberoEnv is deliberately not used -- it hides the state setters."""
    libero_suite, task_suite_id, _ = task_name_to_suite_and_ids[task_name]
    task = benchmark.get_benchmark_dict()[libero_suite]().get_task(task_suite_id)
    env = ControlEnv(
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_names=list(CAMS),
        camera_heights=image_size,
        camera_widths=image_size,
        has_renderer=False,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    env.seed(seed)
    return env


# ── raw mujoco.Renderer wrapper ─────────────────────────────────────────────

def make_scene_option(geomgroup: Sequence[int], sitegroup: Sequence[int]) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = np.asarray(geomgroup, dtype=np.uint8)
    opt.sitegroup[:] = np.asarray(sitegroup, dtype=np.uint8)
    return opt


class RawRenderer:
    """``mujoco.Renderer`` with explicit MjvOption + mjRND flag control (F2b).

    Scene flags are reset to the library defaults before every render and the
    requested ``flags_off`` are then cleared, so a seg/depth render can never
    inherit stale flags from a previous rgb render.

    MSAA footgun (measured, G7): segmentation renders on a multisampled
    offscreen buffer BLEND the per-geom ID colors at silhouette edges; the
    blended color decodes to an unrelated -- possibly out-of-range -- geom id
    (observed live: table-depth edge pixels labeled as tiny robot meshes, and
    an IndexError inside ``mujoco.Renderer.render``). Segmentation and depth
    must therefore come from a context created with
    ``model.vis.quality.offsamples = 0`` (``ProbeRenderers.geo``), while rgb
    parity renders keep the stock MSAA context that matches the robosuite obs
    pipeline bit-exactly (``ProbeRenderers.rgb``).
    """

    def __init__(self, model: mujoco.MjModel, size: int):
        self.r = mujoco.Renderer(model, height=size, width=size)
        self.size = size
        self._default_flags = np.array(self.r.scene.flags, dtype=np.uint8).copy()

    def _update(self, data, cam: str, opt: mujoco.MjvOption, flags_off: Sequence[str]):
        self.r.update_scene(data, camera=cam, scene_option=opt)
        self.r.scene.flags[:] = self._default_flags
        for name in flags_off:
            self.r.scene.flags[getattr(mujoco.mjtRndFlag, name)] = 0

    def rgb(self, data, cam, opt, flags_off=()) -> np.ndarray:
        self._update(data, cam, opt, flags_off)
        return self.r.render().copy()

    def depth(self, data, cam, opt, flags_off=()) -> np.ndarray:
        self.r.enable_depth_rendering()
        try:
            self._update(data, cam, opt, flags_off)
            return self.r.render().copy()  # float32 (H,W), METRIC meters (M0)
        finally:
            self.r.disable_depth_rendering()

    def seg(self, data, cam, opt, flags_off=()) -> np.ndarray:
        self.r.enable_segmentation_rendering()
        try:
            self._update(data, cam, opt, flags_off)
            return self.r.render().copy()  # int32 (H,W,2): [obj id, mjtObj type]
        finally:
            self.r.disable_segmentation_rendering()

    def close(self):
        self.r.close()


class ProbeRenderers:
    """The rgb (MSAA, robosuite-parity) / geo (offsamples=0, exact seg+depth)
    renderer pair for one image size -- see the RawRenderer docstring."""

    def __init__(self, rgb: RawRenderer, geo: RawRenderer):
        self.rgb = rgb
        self.geo = geo
        self.size = rgb.size

    def close(self):
        self.rgb.close()
        self.geo.close()

    @staticmethod
    def create(model, sizes: Sequence[int]) -> Dict[int, "ProbeRenderers"]:
        rgb = {s: RawRenderer(model, s) for s in sizes}
        saved = int(model.vis.quality.offsamples)
        model.vis.quality.offsamples = 0  # read at MjrContext creation only
        try:
            geo = {s: RawRenderer(model, s) for s in sizes}
        finally:
            model.vis.quality.offsamples = saved
        return {s: ProbeRenderers(rgb[s], geo[s]) for s in sizes}


# ── shared geometry helpers ─────────────────────────────────────────────────

def camera_geometry(model, data, cam_name: str, size: int):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    assert cid >= 0, f"camera '{cam_name}' not found in the model"
    K = fovy_to_K(float(model.cam_fovy[cid]), size, size)
    cam_p = np.array(data.cam_xpos[cid], dtype=np.float64)
    cam_R = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
    return cid, K, cam_p, cam_R


def geom_mask_of(seg: np.ndarray, gids: np.ndarray) -> np.ndarray:
    return (seg[..., 1] == GEOM_T) & np.isin(seg[..., 0], gids)


def seg_geom_ids(seg: np.ndarray) -> np.ndarray:
    m = seg[..., 1] == GEOM_T
    ids = np.unique(seg[..., 0][m])
    return ids[ids >= 0]


def backproject_pixels(K, c2w, rows, cols, depth) -> np.ndarray:
    """Pixel centers (row+0.5, col+0.5) at metric depth -> world points (N,3).

    MuJoCo depth is planar z-depth along the camera axis (M0: already metric),
    which is exactly the OpenCV backprojection model used here.
    """
    d = depth[rows, cols].astype(np.float64)
    u = cols.astype(np.float64) + 0.5
    v = rows.astype(np.float64) + 0.5
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], axis=-1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def fit_plane(pts: np.ndarray):
    """SVD plane fit -> (unit normal with n_z>=0, centroid, rms residual m)."""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    rms = float(np.sqrt(np.mean((q @ n) ** 2)))
    return n, c, rms


def body_subtree_geoms(model) -> Dict[int, np.ndarray]:
    """body id -> geom ids of the body's kinematic subtree (mujoco guarantees
    parent index < child index, so one forward pass suffices)."""
    parent = np.asarray(model.body_parentid)
    geom_body = np.asarray(model.geom_bodyid)
    out: Dict[int, np.ndarray] = {}
    for root in range(model.nbody):
        members = {root}
        for b in range(root + 1, model.nbody):
            if int(parent[b]) in members:
                members.add(b)
        out[root] = np.nonzero(np.isin(geom_body, list(members)))[0]
    return out


def body_name(model, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""


def geom_label(model, gid: int) -> str:
    b = body_name(model, int(model.geom_bodyid[gid]))
    g = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return f"{b}:{g if g else f'geom{gid}'}"


def table_box_candidates(model, data, addr, z_below=0.30, z_above=0.02) -> List[int]:
    """Collision/visual BOX geoms named 'table' whose top surface sits within
    the object resting-z window (same detection window as ``table_top_xy_aabb``)."""
    obj_min_z = min(float(data.qpos[s + 2]) for s, _e in addr.obj_qpos_slices.values())
    out = []
    for gid in range(int(model.ngeom)):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        label = geom_label(model, gid).lower()
        if "table" not in label:
            continue
        center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        top_z = float(center[2] + (np.abs(R) @ np.asarray(model.geom_size[gid]))[2])
        if obj_min_z - z_below <= top_z <= obj_min_z + z_above:
            out.append(gid)
    assert out, "no 'table' box geom near the object resting z -- F1/F5 need one"
    return out


def geom_top_z(model, data, gid: int) -> float:
    center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    return float(center[2] + (np.abs(R) @ np.asarray(model.geom_size[gid]))[2])


def _import_gsplat():
    try:
        import gsplat  # noqa: F401 -- JIT compile cached after M0
        return gsplat
    except Exception as e:  # pragma: no cover - environment failure path
        raise RuntimeError(
            "gsplat import failed. Run with the oat env python and\n"
            "  export PATH=/home/haotian/miniforge3/envs/oat/bin:/usr/local/cuda/bin:$PATH\n"
            "(gsplat JIT needs ninja + nvcc on first import; see M0_NOTES.md).") from e


# ── F2b: raw-renderer visualization parity ──────────────────────────────────

def measure_f2b(rr: RawRenderer, model, data, obs) -> Tuple[dict, mujoco.MjvOption, Tuple[str, ...]]:
    """Grid-search MjvOption geom/site groups x mjRND flags (64 combos) for the
    minimum MAD between the raw ``mujoco.Renderer`` and the robosuite obs
    pipeline at the same camera/state. robosuite obs are stored flipped
    (dataset orientation), so the reference is np.flip(obs, axis=0)."""
    ref = np.flip(np.asarray(obs["agentview_image"]), axis=0).astype(np.int16)
    ref_eye = np.flip(np.asarray(obs["robot0_eye_in_hand_image"]), axis=0).astype(np.int16)

    stock_opt = mujoco.MjvOption()
    geomgroups = [
        ("stock", list(np.asarray(stock_opt.geomgroup, dtype=int))),
        ("all_on", [1, 1, 1, 1, 1, 1]),
        ("g1_only", [0, 1, 0, 0, 0, 0]),
        ("g0_g1", [1, 1, 0, 0, 0, 0]),
    ]
    sitegroups = [
        ("sites_off", [0, 0, 0, 0, 0, 0]),
        ("sites_stock", list(np.asarray(stock_opt.sitegroup, dtype=int))),
    ]
    flag_subsets = [tuple(sorted(c)) for r in range(len(F2B_RND_FLAGS) + 1)
                    for c in itertools.combinations(F2B_RND_FLAGS, r)]

    rows = []
    for (gg_name, gg), (sg_name, sg), off in itertools.product(geomgroups, sitegroups, flag_subsets):
        opt = make_scene_option(gg, sg)
        raw = rr.rgb(data, "agentview", opt, off).astype(np.int16)
        mad = float(np.abs(raw - ref).mean())
        # deterministic tie-break: fewest flags off, fewest groups on, sites off
        rows.append((mad, len(off), sum(gg), sum(sg), gg_name, sg_name, off, gg, sg))
    rows.sort(key=lambda r: r[:6])
    mad, _, _, _, gg_name, sg_name, off, gg, sg = rows[0]
    stock_mad = next(r[0] for r in rows if r[4] == "stock" and r[5] == "sites_stock" and r[6] == ())

    win_opt = make_scene_option(gg, sg)
    mad_eye = float(np.abs(
        rr.rgb(data, "robot0_eye_in_hand", win_opt, off).astype(np.int16) - ref_eye).mean())

    rec = {
        "flags": {"geomgroup": list(gg), "sitegroup": list(sg), "flags_off": list(off)},
        "mad": mad,
        "pass": bool(mad <= F2B_MAD_MAX),
        "mad_stock": stock_mad,
        "mad_eye_in_hand": mad_eye,
        "n_combos": len(rows),
        "winner_labels": {"geomgroup": gg_name, "sitegroup": sg_name},
    }
    print(f"[probe_render_facts] F2b: MAD {mad:.3f} (stock {stock_mad:.2f}) with "
          f"geomgroup={gg} sitegroup={sg_name} flags_off={list(off)} "
          f"eye_in_hand MAD {mad_eye:.3f} -> {'PASS' if rec['pass'] else 'FAIL'}")
    return rec, win_opt, off


# ── F6: geom -> body -> component map ───────────────────────────────────────

def measure_f6(model, addr) -> Tuple[dict, dict]:
    """Classify every geom as movable object / robot / background by kinematic
    root (body_rootid). Returns (record, id_maps) where id_maps carries the
    arrays other facts reuse (robot/movable geom ids)."""
    body_rootid = np.asarray(model.body_rootid)
    jnt_bodyid = np.asarray(model.jnt_bodyid)
    geom_bodyid = np.asarray(model.geom_bodyid)

    movable_root: Dict[str, int] = {}
    for jname in addr.obj_qpos_slices:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"free joint '{jname}' vanished from the model"
        movable_root[jname] = int(body_rootid[jnt_bodyid[jid]])
    movable_roots = np.array(sorted(set(movable_root.values())), dtype=int)

    robot_bodies = [b for b in range(model.nbody)
                    if body_name(model, b).startswith(ROBOT_BODY_PREFIXES)]
    assert robot_bodies, "no robot0/gripper0/mount0 bodies in the model"
    robot_roots = np.array(sorted({int(body_rootid[b]) for b in robot_bodies}), dtype=int)
    overlap = set(robot_roots.tolist()) & set(movable_roots.tolist())
    assert not overlap, f"robot and movable subtrees share roots: {overlap}"

    geom_roots = body_rootid[geom_bodyid]
    is_object = np.isin(geom_roots, movable_roots)
    is_robot = np.isin(geom_roots, robot_roots)
    orphans = []  # a robot-prefixed body outside the robot root partition
    for gid in range(int(model.ngeom)):
        bn = body_name(model, int(geom_bodyid[gid]))
        if bn.startswith(ROBOT_BODY_PREFIXES) and not is_robot[gid]:
            orphans.append(geom_label(model, gid))

    per_joint_geoms = {j: int(np.sum(geom_roots == r)) for j, r in movable_root.items()}
    empty_joints = [j for j, n in per_joint_geoms.items() if n == 0]
    robot_link_bodies = sorted(
        {int(b) for b in geom_bodyid[is_robot]}, key=lambda b: body_name(model, b))
    unnamed = [b for b in robot_link_bodies if not body_name(model, b)]

    ok = not orphans and not empty_joints and not unnamed and len(robot_link_bodies) >= 8
    rec = {
        "n_objects": len(addr.obj_qpos_slices),
        "n_robot_links": len(robot_link_bodies),
        "orphan_geoms": orphans,
        "pass": bool(ok),
        "n_geoms": {"object": int(is_object.sum()), "robot": int(is_robot.sum()),
                    "background": int((~is_object & ~is_robot).sum())},
        "per_joint_geoms": per_joint_geoms,
        "robot_link_bodies": [body_name(model, b) for b in robot_link_bodies],
        "empty_joints": empty_joints,
    }
    id_maps = {
        "robot_gids": np.nonzero(is_robot)[0],
        "movable_gids": np.nonzero(is_object)[0],
        "robot_body_ids": robot_bodies,
    }
    print(f"[probe_render_facts] F6: {rec['n_objects']} objects, "
          f"{rec['n_robot_links']} robot link bodies with geoms, geoms "
          f"{rec['n_geoms']} -> {'PASS' if ok else 'FAIL'}")
    return rec, id_maps


# ── F1: camera convention ───────────────────────────────────────────────────

def pick_containment_target(model, data, seg, cam_p, cam_R, robot_body_ids,
                            subtree_geoms) -> Tuple[int, str, np.ndarray]:
    """Per-camera containment anchor -> (body id, name, containment mask).

    Preference order (all gated on the body origin sitting >=3 cm in FRONT of
    the camera -- GL forward = -z column, a test that presupposes no flip
    candidate):
      1. robot0_right_hand vs its subtree mask (plan §4; a massless attach
         frame whose subtree carries the hand geoms) -- the agentview case.
      2. fingertip pad bodies (``*_tip``) vs the FULL visible robot silhouette
         -- the wrist-camera case, where the hand/finger-knuckle origins sit in
         the camera plane or project off-frame; the tip pads are the only
         robot origins in view (measured: inside the mask for the true flip,
         ~70 px away under the 180-degree-rotated alternative, so the
         discrimination survives the coarser mask).
      3. any robot body with >=30 own visible pixels vs its subtree mask --
         ranking by OWN pixels, since subtree ranking would promote ancestors
         like robot0_base (subtree = whole arm) whose origin projects nowhere
         near the visible mask.
    """
    gmask = seg[..., 1] == GEOM_T
    geom_body = np.asarray(model.geom_bodyid)
    fwd = -cam_R[:, 2]
    robot_geoms = np.nonzero(np.isin(geom_body, robot_body_ids))[0]
    robot_mask = np.isin(seg[..., 0], robot_geoms) & gmask

    def own_px(bid: int) -> int:
        own = np.nonzero(geom_body == bid)[0]
        return int((np.isin(seg[..., 0], own) & gmask).sum())

    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    tips = [b for b in robot_body_ids if body_name(model, b).endswith("_tip")]
    candidates: List[Tuple[int, str]] = ([(hand, "subtree")] if hand >= 0 else [])
    candidates += [(b, "robot") for b in tips]
    candidates += [(b, "subtree") for b in
                   sorted(robot_body_ids, key=own_px, reverse=True)
                   if b != hand and b not in tips]
    for bid, mode in candidates:
        if float(fwd @ (np.asarray(data.xpos[bid]) - cam_p)) < 0.03:
            continue
        if mode == "subtree":
            if bid != hand and own_px(bid) < CONT_MIN_MASK_PX:
                continue
            mask = np.isin(seg[..., 0], subtree_geoms[bid]) & gmask
        else:
            mask = robot_mask
        if int(mask.sum()) < CONT_MIN_MASK_PX:
            continue
        return bid, body_name(model, bid), mask
    raise RuntimeError(
        "no robot body with a >=30 px visible mask in front of the camera -- "
        "cannot run the F1 containment test")


def containment_px(K, w2c, p_world, dist_to_mask) -> float:
    """Pixel distance from the projected point to the nearest mask pixel
    (0 = inside). inf for behind-camera / out-of-image projections."""
    uv = project(K, w2c, np.asarray(p_world, dtype=np.float64)[None])[0]
    if not np.all(np.isfinite(uv)):
        return float("inf")
    H, W = dist_to_mask.shape
    r, c = int(round(uv[1] - 0.5)), int(round(uv[0] - 0.5))
    if not (0 <= r < H and 0 <= c < W):
        return float("inf")
    return float(dist_to_mask[r, c])


def sample_bs_pixels(seg, depth, model, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Up to BS_PER_GEOM pixels per visible geom (planes / rbound<=0 excluded,
    depth capped at 15 m), BS_MAX_PX total -- stratified so a wall or table
    mesh cannot dominate the bounding-sphere statistic."""
    rows_out, cols_out = [], []
    for g in seg_geom_ids(seg):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if float(model.geom_rbound[g]) <= 0.0:
            continue
        rr_, cc_ = np.nonzero((seg[..., 0] == g) & (seg[..., 1] == GEOM_T) & (depth < 15.0))
        if rr_.size == 0:
            continue
        take = min(BS_PER_GEOM, rr_.size)
        idx = rng.choice(rr_.size, size=take, replace=False)
        rows_out.append(rr_[idx])
        cols_out.append(cc_[idx])
    assert rows_out, "no geom pixels to sample for the bounding-sphere test"
    rows = np.concatenate(rows_out)[:BS_MAX_PX]
    cols = np.concatenate(cols_out)[:BS_MAX_PX]
    return rows, cols


def bounding_sphere_frac(K, c2w, depth, seg, rows, cols, model, data) -> float:
    pw = backproject_pixels(K, c2w, rows, cols, depth)
    g = seg[rows, cols, 0]
    centers = np.asarray(data.geom_xpos)[g]
    rb = np.asarray(model.geom_rbound)[g]
    ok = np.linalg.norm(pw - centers, axis=1) <= rb * 1.05 + 0.02
    return float(ok.mean())


def measure_f1(env, model, data, renderers: Dict[int, ProbeRenderers], states,
               addr, opt, flags_off, image_size: int, rng) -> dict:
    subtree = body_subtree_geoms(model)
    robot_body_ids = [b for b in range(model.nbody)
                      if body_name(model, b).startswith(ROBOT_BODY_PREFIXES)]
    table_gids = np.array(table_box_candidates(model, data, addr), dtype=int)
    # dedicated collision-group render for the table boxes: LIBERO table tops
    # are group-0 collision boxes hidden by the F2b parity flags (group 1 only)
    opt_col = make_scene_option([1, 0, 0, 0, 0, 0], [0] * 6)
    obj_min_z = min(float(states[0][1 + s + 2]) for s, _e in addr.obj_qpos_slices.values())

    sizes = [image_size] + ([512] if image_size != 512 else [])
    per_size: Dict[int, dict] = {}
    targets: Dict[str, set] = {c: set() for c in CAMS}

    for size in sizes:
        rr = renderers[size].geo  # seg+depth: aliased context (see RawRenderer)
        tol = CONT_TOL_PX_AT_128 * size / 128.0
        agg = {name: {"ok": True, "max_containment_px": {c: 0.0 for c in CAMS},
                      "min_bs_frac": 1.0, "n_checks": 0, "n_plane_checks": 0}
               for name in FLIP_CANDIDATES}
        state_list = states if size == image_size else states[:1]
        for st in state_list:
            env.sim.set_state_from_flattened(st)
            env.sim.forward()
            for cam in CAMS:
                _, K, cam_p, cam_R = camera_geometry(model, data, cam, size)
                depth = rr.depth(data, cam, opt, flags_off)
                seg = rr.seg(data, cam, opt, flags_off)
                seg_col = rr.seg(data, cam, opt_col, flags_off)
                depth_col = rr.depth(data, cam, opt_col, flags_off)

                bid, bname, tmask = pick_containment_target(
                    model, data, seg, cam_p, cam_R, robot_body_ids, subtree)
                targets[cam].add(bname)
                dist_to_mask = ndimage.distance_transform_edt(~tmask)
                p_body = np.asarray(data.xpos[bid], dtype=np.float64)

                bs_rows, bs_cols = sample_bs_pixels(seg, depth, model, rng)

                # largest visible table box (flip-independent selection)
                counts = [(int(geom_mask_of(seg_col, np.array([g])).sum()), int(g))
                          for g in table_gids]
                tb_px, tb_gid = max(counts)
                plane_rows = plane_cols = None
                if tb_px >= PLANE_MIN_MASK_PX:
                    pmask = ndimage.binary_erosion(
                        geom_mask_of(seg_col, np.array([tb_gid])), iterations=2)
                    pr, pc = np.nonzero(pmask & (depth_col < 15.0))
                    if pr.size >= PLANE_MIN_MASK_PX // 2:
                        take = min(pr.size, 4000)
                        idx = rng.choice(pr.size, size=take, replace=False)
                        plane_rows, plane_cols = pr[idx], pc[idx]

                for fname, flip in FLIP_CANDIDATES.items():
                    w2c = mujoco_cam_to_w2c(cam_p, cam_R, flip)
                    c2w = c2w_from_w2c(w2c)
                    cont = containment_px(K, w2c, p_body, dist_to_mask)
                    bsf = bounding_sphere_frac(K, c2w, depth, seg, bs_rows, bs_cols,
                                               model, data)
                    plane_ok = True
                    if plane_rows is not None:
                        pts = backproject_pixels(K, c2w, plane_rows, plane_cols, depth_col)
                        n, cen, _rms = fit_plane(pts)
                        plane_ok = (float(n[2]) >= PLANE_NORMAL_MIN_Z
                                    and abs(float(cen[2]) - obj_min_z) <= PLANE_Z_TOL_M)
                        agg[fname]["n_plane_checks"] += 1
                    a = agg[fname]
                    a["n_checks"] += 1
                    a["max_containment_px"][cam] = max(a["max_containment_px"][cam], cont)
                    a["min_bs_frac"] = min(a["min_bs_frac"], bsf)
                    a["ok"] = a["ok"] and (cont <= tol) and (bsf >= BS_FRAC_MIN) and plane_ok
        per_size[size] = {"tol_px": tol, "per_flip": agg,
                          "winners": sorted(n for n, a in agg.items() if a["ok"])}

    winners_main = per_size[image_size]["winners"]
    winners_512 = per_size[sizes[-1]]["winners"]
    unique = len(winners_main) == 1 and winners_main == winners_512
    flip_name = winners_main[0] if winners_main else "NONE"
    px_err = {c: (per_size[image_size]["per_flip"][flip_name]["max_containment_px"][c]
                  if winners_main else float("inf")) for c in CAMS}

    rec = {
        "flip_name": flip_name,
        "px_err": px_err,
        "pass": bool(unique),
        "px_err_512": ({c: per_size[sizes[-1]]["per_flip"][flip_name]["max_containment_px"][c]
                        for c in CAMS} if winners_main else None),
        "containment_targets": {c: sorted(targets[c]) for c in CAMS},
        "containment_tol_px": {str(s): per_size[s]["tol_px"] for s in sizes},
        "per_flip": {n: {"ok": a["ok"], "min_bs_frac": a["min_bs_frac"],
                         "max_containment_px": a["max_containment_px"],
                         "n_checks": a["n_checks"],
                         "n_plane_checks": a["n_plane_checks"]}
                     for n, a in per_size[image_size]["per_flip"].items()},
        "winners_by_size": {str(s): per_size[s]["winners"] for s in sizes},
    }
    print(f"[probe_render_facts] F1: winning flip='{flip_name}' px_err={ {c: round(v, 2) for c, v in px_err.items()} } "
          f"targets={rec['containment_targets']} -> {'PASS' if unique else 'FAIL'}")
    return rec


# ── F2: image orientation chain ─────────────────────────────────────────────

def measure_f2(model, data, rr: RawRenderer, opt, flags_off, flip,
               image_size: int) -> dict:
    """gsplat -> stored-zarr orientation, measured: rasterize one Gaussian at
    the depth-backprojection of an off-center raw pixel (row r < H/2) and check
    whether it lands at row ~r (raw orientation, top-origin OpenCV) or row
    ~H-1-r (flipped). Orientation chain (measured, not assumed): robosuite obs
    = np.flip(raw_render, axis=0) (raw_flip_ud, M0) AND stored zarr frames =
    np.flip(obs, axis=0) (dataset_conversion), so RAW orientation == STORED
    orientation — F2b's bit-exact MAD 0.0 against flip(obs) is the proof.
    ``gsplat_flip_ud`` is the flip needed on gsplat output to reach the
    STORED-zarr orientation: False when gsplat matches raw (== stored)."""
    gsplat = _import_gsplat()
    import torch

    assert torch.cuda.is_available(), "F2/F7 need CUDA for gsplat"
    dev = "cuda:0"
    H = W = image_size
    per_cam = {}
    matches_raw_flags = []
    ok = True
    for cam in CAMS:
        _, K, cam_p, cam_R = camera_geometry(model, data, cam, image_size)
        depth = rr.depth(data, cam, opt, flags_off)
        seg = rr.seg(data, cam, opt, flags_off)
        valid = ((seg[..., 1] == GEOM_T) & (depth > 0.15) & (depth < 10.0))
        valid[H // 2 - 8:, :] = False  # keep the probe row unambiguous vs its flip
        if not valid.any():
            valid = (seg[..., 1] == GEOM_T) & (depth > 0.05) & (depth < 10.0)
            valid[H // 2 - 8:, :] = False
        assert valid.any(), f"F2: no usable off-center pixel in camera '{cam}'"
        vr, vc = np.nonzero(valid)
        i = np.argmin(np.abs(vr - H // 4) * 4 + np.abs(vc - W // 3))
        r, c = int(vr[i]), int(vc[i])

        w2c = mujoco_cam_to_w2c(cam_p, cam_R, flip)
        c2w = c2w_from_w2c(w2c)
        X = backproject_pixels(K, c2w, np.array([r]), np.array([c]), depth)[0]
        uv = project(K, w2c, X[None])[0]
        assert np.allclose(uv, [c + 0.5, r + 0.5], atol=1e-3), (
            f"backproject/project round-trip broke: {uv} vs {(c + 0.5, r + 0.5)}")

        d = float(depth[r, c])
        s = max(0.004, 0.015 * d)
        img, alpha, _ = gsplat.rasterization(
            torch.tensor(X, dtype=torch.float32, device=dev)[None],
            torch.tensor([[1.0, 0, 0, 0]], device=dev),
            torch.full((1, 3), s, device=dev),
            torch.tensor([0.995], device=dev),
            torch.tensor([[1.0, 0.3, 0.1]], device=dev),
            torch.tensor(w2c, dtype=torch.float32, device=dev)[None],
            torch.tensor(K, dtype=torch.float32, device=dev)[None],
            W, H)
        a = alpha[0, ..., 0].detach().cpu().numpy()
        assert a.max() > 0.05, f"F2: probe Gaussian invisible in '{cam}' (max alpha {a.max():.3f})"
        rg, cg = np.unravel_index(int(np.argmax(a)), a.shape)

        col_ok = abs(cg - c) <= 3
        if abs(rg - r) <= 3:
            matches_raw = True
        elif abs(rg - (H - 1 - r)) <= 3:
            matches_raw = False
        else:
            matches_raw = None
        cam_ok = col_ok and matches_raw is not None
        ok = ok and cam_ok
        matches_raw_flags.append(matches_raw)
        per_cam[cam] = {"raw_pixel_rc": [r, c], "gsplat_argmax_rc": [int(rg), int(cg)],
                        "alpha_max": float(a.max()), "matches_raw": matches_raw,
                        "ok": bool(cam_ok)}

    consistent = ok and len(set(matches_raw_flags)) == 1
    # RAW orientation == STORED-zarr orientation (raw == flip(obs) bit-exact
    # per F2b, and the zarr stores flip(obs)), so gsplat needs a flip into
    # dataset orientation only if it does NOT match raw. matches_raw=True ->
    # gsplat_flip_ud=False. (A previous derivation inverted this — the
    # double-flip would have written upside-down GS zarr frames.)
    gsplat_flip_ud = (not matches_raw_flags[0]) if consistent else None
    rec = {
        "raw_flip_ud": True,
        "gsplat_flip_ud": gsplat_flip_ud,
        "pass": bool(consistent and gsplat_flip_ud is not None),
        "per_cam": per_cam,
    }
    print(f"[probe_render_facts] F2: gsplat_flip_ud={gsplat_flip_ud} "
          f"(gsplat matches raw orientation: {matches_raw_flags}) -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}")
    return rec


# ── F3: headlight (record-only) ─────────────────────────────────────────────

def measure_f3(model, data, rr: RawRenderer, opt, flags_off) -> dict:
    stock = rr.rgb(data, "agentview", opt, flags_off).astype(np.int16)
    saved = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy(),
             model.vis.headlight.specular.copy())
    try:
        model.vis.headlight.ambient[:] = 0.0
        model.vis.headlight.diffuse[:] = 0.0
        model.vis.headlight.specular[:] = 0.0
        dark = rr.rgb(data, "agentview", opt, flags_off).astype(np.int16)
    finally:
        model.vis.headlight.ambient[:] = saved[0]
        model.vis.headlight.diffuse[:] = saved[1]
        model.vis.headlight.specular[:] = saved[2]
    mad = float(np.abs(dark - stock).mean())
    rec = {"mad_headlight_off": mad, "keep_stock_lighting": bool(mad > 1.0)}
    print(f"[probe_render_facts] F3: headlight-off MAD {mad:.2f} -> "
          f"keep_stock_lighting={rec['keep_stock_lighting']} (record-only)")
    return rec


# ── F4: hide mechanisms ─────────────────────────────────────────────────────

def measure_f4(env, model, data, pr: ProbeRenderers, opt, flags_off, state0,
               addr, id_maps, image_size: int) -> dict:
    def set_state(st):
        env.sim.set_state_from_flattened(st)
        env.sim.forward()

    set_state(state0)
    rgb_base = pr.rgb.rgb(data, "agentview", opt, flags_off).astype(np.int16)
    seg_base = pr.geo.seg(data, "agentview", opt, flags_off)
    movable_gids = id_maps["movable_gids"]
    robot_gids = id_maps["robot_gids"]

    # (a) movables -> graveyard (xy += 50): seg ids absent, rgb diff confined
    # to the dilated former-silhouette region (< 1% of the image outside it,
    # which absorbs former cast shadows -- plan §4 F4a).
    st_g = np.array(state0, dtype=np.float64)
    for s, _e in addr.obj_qpos_slices.values():
        st_g[1 + s:1 + s + 2] += 50.0
    set_state(st_g)
    seg_g = pr.geo.seg(data, "agentview", opt, flags_off)
    rgb_g = pr.rgb.rgb(data, "agentview", opt, flags_off).astype(np.int16)
    leak_mov = sorted(set(seg_geom_ids(seg_g).tolist()) & set(movable_gids.tolist()))
    changed = (np.abs(rgb_g - rgb_base) > DIFF_THRESH).any(axis=-1)
    mov_mask = geom_mask_of(seg_base, movable_gids)
    dil = ndimage.binary_dilation(mov_mask, iterations=MOVABLE_DILATE_PX)
    frac_outside = float((changed & ~dil).sum()) / changed.size
    movable_hide_ok = (not leak_mov) and frac_outside < MOVABLE_OUTSIDE_FRAC_MAX
    set_state(state0)

    # (b) robot alpha=0: seg ids absent AND removal is LOCAL -- no changed
    # pixel farther than 40 px (at 128; scaled) from the baseline robot mask.
    # Far-field residue (e.g. reflections of the robot) would make alpha-0
    # background captures unsound -> robot_hide='masked' (plan §4 F4b).
    far_px = ROBOT_FAR_PX_AT_128 * image_size / 128.0
    rgba_saved = model.geom_rgba.copy()
    try:
        model.geom_rgba[robot_gids, 3] = 0.0
        seg_a = pr.geo.seg(data, "agentview", opt, flags_off)
        rgb_a = pr.rgb.rgb(data, "agentview", opt, flags_off).astype(np.int16)
    finally:
        model.geom_rgba[:] = rgba_saved
    leak_rob = sorted(set(seg_geom_ids(seg_a).tolist()) & set(robot_gids.tolist()))
    changed_a = (np.abs(rgb_a - rgb_base) > DIFF_THRESH).any(axis=-1)
    rob_mask = geom_mask_of(seg_base, robot_gids)
    dist = ndimage.distance_transform_edt(~rob_mask)
    n_far = int((changed_a & (dist > far_px)).sum())
    alpha0_ok = (not leak_rob) and n_far == 0

    rec = {
        "movable_hide_ok": bool(movable_hide_ok),
        "robot_hide": "alpha0" if alpha0_ok else "masked",
        "pass": bool(movable_hide_ok),
        "movable": {"seg_leak_gids": leak_mov, "frac_changed_outside": frac_outside,
                    "dilate_px": MOVABLE_DILATE_PX},
        "robot": {"seg_leak_gids": leak_rob, "n_changed_far_px": n_far,
                  "far_px_thresh": far_px,
                  "n_changed_total": int(changed_a.sum())},
    }
    print(f"[probe_render_facts] F4: movable_hide_ok={movable_hide_ok} "
          f"(leak={leak_mov}, outside_frac={frac_outside:.4f}) "
          f"robot_hide={rec['robot_hide']} (leak={leak_rob}, n_far={n_far}) -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}")
    return rec


# ── F5: depth sanity / table plane ──────────────────────────────────────────

def measure_f5(model, data, rr: RawRenderer, opt, flags_off, flip, addr, rng) -> dict:
    """Backproject the largest 'table' box seg mask (dedicated group-0 render;
    the tops are collision boxes) through the winning flip; SVD plane fit.
    Planarity <= 2 mm RMS; height within 3 cm of the box top (and of the
    resting-object min z, recorded -- object origins sit a little above the
    surface, hence the loose bound; plan §4)."""
    opt_col = make_scene_option([1, 0, 0, 0, 0, 0], [0] * 6)
    _, K, cam_p, cam_R = camera_geometry(model, data, "agentview", rr.size)
    w2c = mujoco_cam_to_w2c(cam_p, cam_R, flip)
    c2w = c2w_from_w2c(w2c)
    obj_min_z = min(float(data.qpos[s + 2]) for s, _e in addr.obj_qpos_slices.values())

    seg_col = rr.seg(data, "agentview", opt_col, flags_off)
    depth_col = rr.depth(data, "agentview", opt_col, flags_off)
    cands = table_box_candidates(model, data, addr)
    counts = [(int(geom_mask_of(seg_col, np.array([g])).sum()), int(g)) for g in cands]
    tb_px, tb_gid = max(counts)

    if tb_px >= F5_MIN_MASK_PX:
        method = "collision_box"
        mask = geom_mask_of(seg_col, np.array([tb_gid]))
        depth_use = depth_col
        z_ref = geom_top_z(model, data, tb_gid)
    else:
        # fallback: visual 'table' geoms filtered to a slab around the object
        # resting z (documented in plan §4 -- only used when the collision
        # boxes are not renderable in this scene)
        method = "visual_slab"
        seg_vis = rr.seg(data, "agentview", opt, flags_off)
        depth_use = rr.depth(data, "agentview", opt, flags_off)
        table_vis = np.array([g for g in seg_geom_ids(seg_vis)
                              if "table" in geom_label(model, int(g)).lower()], dtype=int)
        assert table_vis.size, "F5 fallback: no visual 'table' geom visible"
        mask = geom_mask_of(seg_vis, table_vis)
        rrows, rcols = np.nonzero(mask)
        z = backproject_pixels(K, c2w, rrows, rcols, depth_use)[:, 2]
        keep = np.abs(z - obj_min_z) <= 0.02
        mask = np.zeros_like(mask)
        mask[rrows[keep], rcols[keep]] = True
        z_ref = obj_min_z

    mask = ndimage.binary_erosion(mask, iterations=2) & (depth_use < 15.0)
    rows, cols = np.nonzero(mask)
    assert rows.size >= 50, f"F5: only {rows.size} table-top pixels ({method})"
    if rows.size > 8000:
        idx = rng.choice(rows.size, size=8000, replace=False)
        rows, cols = rows[idx], cols[idx]
    pts = backproject_pixels(K, c2w, rows, cols, depth_use)
    n, cen, rms = fit_plane(pts)
    plane_z = float(cen[2])
    z_err = abs(plane_z - z_ref)
    ok = (rms <= PLANE_RMS_MAX_M and z_err <= PLANE_Z_TOL_M
          and float(n[2]) >= PLANE_NORMAL_MIN_Z)
    rec = {
        "plane_rms_m": rms,
        "z_err_m": z_err,
        "pass": bool(ok),
        "method": method,
        "n_px": int(rows.size),
        "normal_z": float(n[2]),
        "plane_z": plane_z,
        "z_ref": float(z_ref),
        "z_err_vs_obj_min_z": abs(plane_z - obj_min_z),
        "table_geom": geom_label(model, tb_gid),
    }
    print(f"[probe_render_facts] F5: rms {rms * 1000:.2f} mm, z_err {z_err * 1000:.1f} mm, "
          f"normal_z {n[2]:.5f}, {rows.size} px ({method}) -> {'PASS' if ok else 'FAIL'}")
    return rec


# ── F7: gsplat perf (record-only) ───────────────────────────────────────────

def measure_f7(seed: int) -> dict:
    gsplat = _import_gsplat()
    import torch

    torch.manual_seed(seed)
    dev = "cuda:0"
    N = F7_N_GAUSSIANS
    means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    scales = 0.005 + 0.01 * torch.rand(N, 3, device=dev)
    opac = 0.1 + 0.8 * torch.rand(N, device=dev)
    sh = torch.randn(N, 16, 3, device=dev) * 0.2
    c2w = lookat_c2w(np.array([2.5, 0.4, 1.0]), np.zeros(3))
    w2c = torch.tensor(c2w_from_w2c(c2w), dtype=torch.float32, device=dev)[None]

    fps = {}
    for size in (128, 512):
        K = torch.tensor(fovy_to_K(45.0, size, size), dtype=torch.float32, device=dev)[None]

        def run():
            img, alpha, _ = gsplat.rasterization(
                means, quats, scales, opac, sh, w2c, K, size, size, sh_degree=3)
            return img

        for _ in range(F7_WARMUP):
            run()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(F7_ITERS):
            run()
        torch.cuda.synchronize()
        fps[size] = F7_ITERS / (time.perf_counter() - t0)

    rec = {"fps_128": float(fps[128]), "fps_512": float(fps[512]),
           "n_gaussians": N, "gsplat_version": getattr(gsplat, "__version__", "?"),
           "torch_version": __import__("torch").__version__}
    print(f"[probe_render_facts] F7: {N} gaussians sh3 -> "
          f"{fps[128]:.0f} FPS @128, {fps[512]:.0f} FPS @512 (record-only)")
    return rec


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_name", default=None,
                        help="LIBERO task name (default: first libero10 subtask); "
                             "a suite name also resolves to its first subtask")
    parser.add_argument("--out", default="data/libero/gs_render_facts.json")
    parser.add_argument("--image_size", type=int, default=128,
                        help="dataset-native size for F1/F2/F2b..F5; F1 is "
                             "additionally verified at 512 (capture size)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_states", type=int, default=3,
                        help="fresh env.reset() states (theta=0 rewrite) for F1; "
                             "no HDF5 demos needed -- conventions are state-independent")
    args = parser.parse_args()

    task_name = args.task_name or get_subtasks("libero10")[0]
    if is_multitask(task_name):
        task_name = get_subtasks(task_name)[0]
    rng = np.random.default_rng(args.seed)

    print(f"[probe_render_facts] task={task_name} image_size={args.image_size} "
          f"seed={args.seed} n_states={args.n_states}")
    env = build_control_env(task_name, args.image_size, args.seed)

    # ALL resets first: robosuite hard-resets destroy and rebuild the sim, so
    # any model/data/renderer handle taken before the last reset would be
    # stale (renders a frozen scene; a Renderer surviving the teardown can
    # even read back garbage seg ids). The flattened qpos/qvel layout is
    # name-stable across rebuilds, so earlier reset states replay fine.
    raw_states = []
    for _ in range(args.n_states):
        env.reset()
        raw_states.append(env.sim.get_state().flatten())

    addr = resolve_addresses(env)
    model = env.sim.model._model
    data = env.sim.data._data
    # demo states: reset states through the theta=0 aug rewrite (exact no-op;
    # exercises the same state path the pre-render uses); no HDF5 needed
    states = [rewrite_state(s, 0.0, addr) for s in raw_states]

    # robosuite reference obs BEFORE any extra renderer context exists:
    # robosuite's offscreen context becomes unreliable once additional
    # mujoco.Renderer EGL contexts are created (observed live: a later
    # regenerate_obs_from_state returned a different viewpoint entirely).
    # The probe needs obs exactly once (F2b reference); grab and copy it now.
    obs_raw = env.regenerate_obs_from_state(states[0])
    obs0 = {k: np.array(v, copy=True) for k, v in obs_raw.items()
            if k.endswith("_image")}
    got = obs0["agentview_image"].shape
    assert got == (args.image_size, args.image_size, 3), (
        f"obs image shape {got} != requested {args.image_size} -- env/camera "
        f"config mismatch")

    # the model's offscreen framebuffer must fit the largest probe render
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 512)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 512)
    sizes = sorted({args.image_size, 512})
    renderers = ProbeRenderers.create(model, sizes)

    try:

        f2b, opt, flags_off = measure_f2b(renderers[args.image_size].rgb, model,
                                          data, obs0)
        f6, id_maps = measure_f6(model, addr)
        f1 = measure_f1(env, model, data, renderers, states, addr, opt, flags_off,
                        args.image_size, rng)

        # back to state 0 for the single-state facts
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        pr = renderers[args.image_size]

        if f1["pass"]:
            flip = FLIP_CANDIDATES[f1["flip_name"]]
            f2 = measure_f2(model, data, pr.geo, opt, flags_off, flip, args.image_size)
            f5 = measure_f5(model, data, pr.geo, opt, flags_off, flip, addr, rng)
        else:  # cannot measure orientation/depth facts without a camera convention
            f2 = {"raw_flip_ud": True, "gsplat_flip_ud": None, "pass": False,
                  "skipped": "F1 found no unique flip"}
            f5 = {"plane_rms_m": float("inf"), "z_err_m": float("inf"), "pass": False,
                  "skipped": "F1 found no unique flip"}
        f3 = measure_f3(model, data, pr.rgb, opt, flags_off)
        f4 = measure_f4(env, model, data, pr, opt, flags_off, states[0], addr,
                        id_maps, args.image_size)
    finally:
        for r in renderers.values():
            r.close()
        env.close()

    f7 = measure_f7(args.seed)

    overall = bool(f1["pass"] and f2["pass"] and f2b["pass"] and f4["pass"]
                   and f5["pass"] and f6["pass"])
    facts = {
        "probe": "gs_render_facts",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "task_name": task_name,
        "image_size": args.image_size,
        "seed": args.seed,
        "n_states": args.n_states,
        "F1": f1, "F2": f2, "F2b": f2b, "F3": f3, "F4": f4, "F5": f5,
        "F6": f6, "F7": f7,
        "pass": overall,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(facts, fh, indent=2, default=_json_default)
    print(f"[probe_render_facts] wrote {out_path}")

    if overall:
        # G7 self-check: the facts loaders every downstream ctor uses must
        # round-trip this exact file
        loaded = load_render_facts(out_path, require_pass=True)
        assert np.array_equal(facts_flip(loaded), FLIP_CANDIDATES[f1["flip_name"]])
        assert facts_orientation_flip_ud(loaded) == f2["gsplat_flip_ud"]

    print(f"[probe_render_facts] {'PASS' if overall else 'FAIL'}: "
          f"F1={f1['pass']} F2={f2['pass']} F2b={f2b['pass']} F4={f4['pass']} "
          f"F5={f5['pass']} F6={f6['pass']} (F3/F7 record-only)")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

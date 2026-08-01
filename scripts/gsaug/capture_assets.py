"""Stage-1 GS asset capture (plan §5.1, M2): per-task orbit captures of the
background, each movable object, and the robot from a live LIBERO ControlEnv.

Per task (G1, asset-based reconstruction):
  * background — movables graveyarded (F4-a), robot hidden per facts F4
    (alpha-0, or present-at-stow + per-pixel masks if robot_hide=='masked');
    up to 56 views (rings 25°/50° × 24 azimuths + 8 top-down), requested
    radius 1.6× the table-AABB diagonal clamped per-azimuth inside the room
    walls (LIBERO rooms can be smaller than that orbit), lookat = table
    center; point-blank wall views are skipped, >= 32 usable views required.
  * objects — one capture per movable free joint (names from
    resolve_addresses): solo on the table (others graveyarded, robot hidden),
    floated +z so the −20° ring sees the underside; 48 close-radius views.
  * robot — movables graveyarded, robot visible; ~60 qpos configs (36
    farthest-point-sampled from the task's demo HDF5 frames + joint1-shifted
    copies of 12 at ±20°/±30° so the θ-grid tails are in-distribution, plus a
    gripper-range span assertion); 16 views per config; per-config per-link
    body poses recorded for the articulated trainer.

Every capture directory gets RGB + metric depth + geom-id seg per view and a
transforms.json with OpenCV c2w extrinsics (validated against pixels, G7),
shared K, and the task model-XML sha1 (G9). Fail-fast checks: view counts,
seg purity, >=99% finite depth, camera-chain validation.

Usage:
    MUJOCO_GL=egl python scripts/gsaug/capture_assets.py \
        --task LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket \
        --component all
"""

import os

# Must be set before robosuite / mujoco / libero are imported (one GL context
# per process).
os.environ.setdefault("MUJOCO_GL", "egl")

if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import datetime
import glob
import math
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.se2_state_rewrite import (
    PANDA_JOINT1_LIMIT,
    resolve_addresses,
    table_top_xy_aabb,
)
from oat.gsaug import capture as cap
from oat.gsaug.cameras import load_render_facts

COMPONENTS = ("background", "objects", "robot")

N_BASE_ROBOT_CONFIGS = 36     # farthest-point-deduplicated demo configs
N_SHIFTED_ROBOT_CONFIGS = 12  # of which get joint1-shifted copies (plan §5.1)
JOINT1_SHIFT_PAIRS = ((20.0, -20.0), (30.0, -30.0))  # alternated per config
JOINT1_SHIFT_MARGIN = 0.05    # rad, matches the validity check margin
MIN_OBJECT_PIXELS = 30        # object visible in EVERY object-capture view
MIN_ROBOT_PIXELS = 100        # robot visible in EVERY robot-capture view
BG_POINT_BLANK_DEPTH_M = 0.5  # background view skip rule: pixels closer than
BG_POINT_BLANK_FRAC = 0.60    # ...this over this fraction = wall/backface view
MIN_BACKGROUND_VIEWS = 32     # usable background views after the skip rule
ROBOT_RADIUS_SHRINK = 0.8     # robot-view retry: pull the camera in by this
ROBOT_MIN_RADIUS = 1.0        # ...per step, down to the orbit's radius floor
MIN_FINITE_DEPTH_FRAC = 0.99  # plan §5.1 fail-fast
MAX_ROBOT_POOL = 8000         # FPS pool cap (subsampled with --seed above this)
GRIPPER_SPAN_TOL = 0.05       # fraction of the demo gripper range per joint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", type=str, default="all",
                        help="task name or 'all' (tasks discovered from --hdf5_dir)")
    parser.add_argument("--component", type=str, default="all",
                        choices=list(COMPONENTS) + ["all"])
    parser.add_argument("--out_root", type=str, default="data/libero/gs_assets")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--facts", type=str,
                        default="data/libero/gs_render_facts.json",
                        help="gs_render_facts.json from probe_render_facts.py (M1 gate)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the robot-config pool subsample (only used "
                             "when a task has more demo frames than the FPS pool cap)")
    parser.add_argument("--hdf5_dir", type=str,
                        default="third_party/LIBERO/libero/datasets/libero_10",
                        help="LIBERO *_demo.hdf5 dir (robot qpos configs source)")
    parser.add_argument("--float_dz", type=float, default=0.15,
                        help="minimum object float height (plan §5.1); raised "
                             "automatically when the -20 deg ring would dip "
                             "below the table top")
    return parser.parse_args()


def task_name_from_hdf5(hdf5_path: str) -> str:
    base = os.path.basename(hdf5_path)
    suffix = "_demo.hdf5"
    assert base.endswith(suffix), f"unexpected demo file name: {base}"
    task_name = base[: -len(suffix)]
    assert task_name in task_name_to_suite_and_ids, \
        f"hdf5 file name {base!r} does not map to a known LIBERO task"
    return task_name


def build_control_env(task_name: str, image_size: int) -> ControlEnv:
    """Build a ControlEnv exactly as LiberoEnv does (oat/env/libero/env.py:45-59)."""
    libero_suite, task_suite_id, _ = task_name_to_suite_and_ids[task_name]
    task = benchmark.get_benchmark_dict()[libero_suite]().get_task(task_suite_id)
    env = ControlEnv(
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        ),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=image_size,
        camera_widths=image_size,
        has_renderer=False,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    env.seed(0)
    return env


def object_component_name(joint_name: str) -> str:
    """Capture dir name under objects/: joint name with a trailing _joint<N>
    suffix stripped (layout contract); the exact joint name is recorded in
    transforms.json['joint_name'] so the asset keeps the joint-name identity."""
    import re
    return re.sub(r"_joint\d+$", "", joint_name)


# ── shared per-task capture context ─────────────────────────────────────────

class TaskContext:
    """Everything computed once per task on the freshly reset env."""

    def __init__(self, task_name: str, env: ControlEnv, args: argparse.Namespace,
                 facts: dict):
        self.task_name = task_name
        self.env = env
        self.args = args
        self.facts = facts
        self.robot_hide = facts["F4"]["robot_hide"]
        assert self.robot_hide in ("alpha0", "masked"), (
            f"facts F4 robot_hide={self.robot_hide!r} — unknown hide mode (G7)")

        env.reset()  # once per task; all captures restore base_state from here
        self.addr = resolve_addresses(env)
        self.model_xml_sha1 = cap.model_xml_sha1(env)  # G9
        self.base_state = cap.get_flat_state(env)

        # table geometry needs movables at their reset placement (z anchor)
        tmin, tmax, tgeoms = table_top_xy_aabb(env, self.addr)
        top_z = cap.table_top_z(env.sim.model, env.sim.data, self.addr)
        self.table_lookat = np.array(
            [(tmin[0] + tmax[0]) / 2.0, (tmin[1] + tmax[1]) / 2.0, top_z])
        self.table_diag = float(np.linalg.norm(tmax - tmin))
        self.table_top_z = float(top_z)
        self.table_geoms = tgeoms

        self.renderer = cap.make_renderer(env, args.image_size)
        self.scn_opt = cap.scene_option_from_facts(facts)  # F2b parity
        self.K = cap.free_camera_K(env, args.image_size)
        self.fovy_deg = cap.free_camera_fovy(env)

        model = env.sim.model
        self.movable_gids = cap.movable_geom_ids(model, self.addr)  # F6
        self.robot_gids = np.asarray(cap.robot_geom_ids(model), dtype=np.int64)

    def close(self) -> None:
        self.renderer.close()
        self.env.close()

    # ── shared helpers ──────────────────────────────────────────────────────

    def out_dir(self, component: str) -> str:
        d = os.path.join(self.args.out_root, self.task_name, "captures", component)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, cap.TRANSFORMS_NAME)):
            print(f"[capture] note: overwriting existing capture in {d}")
        return d

    def capture_args(self, **extra) -> dict:
        base = {
            "date": datetime.datetime.now().isoformat(),
            "image_size": self.args.image_size,
            "seed": self.args.seed,
            "fovy_deg": self.fovy_deg,
            "robot_hide": self.robot_hide,
            "facts_path": self.args.facts,
        }
        base.update(extra)
        return base

    def geom_names(self, gids: Sequence[int]) -> List[str]:
        model = self.env.sim.model
        out = []
        for g in list(gids)[:12]:
            body = model.body_id2name(int(np.asarray(model.geom_bodyid)[g])) or "?"
            out.append(f"{body}:{model.geom_id2name(int(g)) or f'geom{g}'}")
        return out

    def check_view(self, tag: str, depth: np.ndarray, seg: np.ndarray,
                   forbidden_gids: np.ndarray, forbidden_what: str) -> None:
        """Per-view fail-fast: finite depth + seg purity (plan §5.1)."""
        finite = float(np.isfinite(depth).mean())
        assert finite >= MIN_FINITE_DEPTH_FRAC, (
            f"{tag}: only {finite:.1%} of depth pixels finite "
            f"(< {MIN_FINITE_DEPTH_FRAC:.0%}) — renderer depth broken")
        if forbidden_gids.size:
            leaked = np.intersect1d(np.unique(seg[seg >= 0]), forbidden_gids)
            assert leaked.size == 0, (
                f"{tag}: seg contains {forbidden_what} geom ids "
                f"{leaked.tolist()} ({self.geom_names(leaked)}) — hide/graveyard "
                f"mechanism leaked (facts F4); refusing impure captures")

    def render_orbit_view(self, pose: cap.OrbitPose, tag: str
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Render one orbit pose; per-view GL extrinsics assert; returns
        (rgb, depth, seg, c2w)."""
        rgb, depth, seg = cap.render_view(
            self.renderer, self.env.sim.data, pose.mjv_camera(), self.scn_opt)
        c2w = pose.c2w_opencv()
        cap.assert_free_camera_pose(self.renderer.scene, c2w, tag)
        return rgb, depth, seg, c2w

    def base_transforms(self, component: str, views_meta: list,
                        validation: dict, **extra) -> dict:
        tf = {
            "image_size": int(self.args.image_size),
            "K": np.asarray(self.K).tolist(),
            "views": views_meta,
            "component": component,
            "task": self.task_name,
            "model_xml_sha1": self.model_xml_sha1,  # G9
            "pose_validation": validation,
        }
        tf.update(extra)
        return tf


# ── background ───────────────────────────────────────────────────────────────

def capture_background(ctx: TaskContext) -> None:
    env, addr = ctx.env, ctx.addr
    out_dir = ctx.out_dir("background")
    tag = f"{ctx.task_name}/background"

    # variable keep-count (skip rule below): stale view files from a previous
    # run must not outlive the new transforms.json
    for stale in glob.glob(os.path.join(out_dir, "view_*")):
        os.remove(stale)

    cap.set_flat_state(env, ctx.base_state)
    cap.graveyard_movables(env, addr)  # F4-a

    # Interior-safe orbit: the plan radius (1.6× table diag) can exceed the
    # room — LIVING_ROOM walls sit ~1.5 m (+x) from the lookat while the
    # requested radius is ~3.3 m, leaving most ring cameras OUTSIDE the room
    # staring at wall backfaces. Clamp per azimuth to wall distance − 0.30 m,
    # floored so the camera never sits over the table itself.
    model, data = env.sim.model, env.sim.data
    requested_radius = 1.6 * ctx.table_diag
    lookat_xy = np.asarray(ctx.table_lookat[:2], dtype=np.float64)

    def interior_radius(az_deg: float) -> float:
        wall_d = cap.wall_distance_2d(model, data, lookat_xy, az_deg)
        return cap.interior_orbit_radius(requested_radius, wall_d,
                                         ctx.table_diag / 2.0)

    poses = cap.background_orbit_poses(ctx.table_lookat, requested_radius,
                                       radius_for_azimuth=interior_radius)

    masks_dir_name: Optional[str] = None
    hide: Optional[cap.RobotHide] = None
    if ctx.robot_hide == "alpha0":
        hide = cap.hide_robot_alpha0(env)  # F4-b
        forbidden = np.concatenate(
            [np.asarray(g, dtype=np.int64) for g in ctx.movable_gids.values()]
            + [ctx.robot_gids])
        forbidden_what = "movable/robot"
    else:
        # facts F4-b failed: robot stays at its reset (stow) config; per-pixel
        # robot masks let the trainer exclude it (plan §5.1 masked mode).
        masks_dir_name = "masks"
        os.makedirs(os.path.join(out_dir, masks_dir_name), exist_ok=True)
        forbidden = np.concatenate(
            [np.asarray(g, dtype=np.int64) for g in ctx.movable_gids.values()])
        forbidden_what = "movable"

    validator = cap.PoseValidator(env.sim.model, ctx.K)
    views_meta = []
    view_radii = []
    skipped_views = []
    try:
        for i, pose in enumerate(tqdm.tqdm(poses, desc=tag, unit="view")):
            k = len(views_meta)  # kept-view index = on-disk index
            vtag = f"{tag} view {k} (orbit pose {i})"
            rgb, depth, seg, c2w = ctx.render_orbit_view(pose, vtag)
            frac_close = float((depth < BG_POINT_BLANK_DEPTH_M).mean())
            if frac_close > BG_POINT_BLANK_FRAC:
                # point-blank wall/backface (camera pinched against a wall):
                # do not save, do not count.
                skipped_views.append({
                    "orbit_pose_idx": i,
                    "azimuth_deg": float(pose.azimuth_deg),
                    "elevation_deg": float(pose.elevation_deg),
                    "radius": float(pose.distance),
                    "frac_depth_below_0p5m": frac_close})
                continue
            ctx.check_view(vtag, depth, seg, forbidden, forbidden_what)
            validator.try_view(env.sim.data, c2w, seg, depth, k,
                               pose.elevation_deg, tag)
            prefix = cap.write_view_files(out_dir, k, rgb, depth, seg)
            if masks_dir_name:
                cap.write_mask_file(os.path.join(out_dir, masks_dir_name), k,
                                    np.isin(seg, ctx.robot_gids))
            views_meta.append({"file_prefix": prefix,
                               "c2w_opencv": c2w.tolist(),
                               "cam_params": pose.cam_params()})
            view_radii.append(float(pose.distance))
    finally:
        if hide is not None:
            hide.restore()

    validation = validator.finalize(tag)
    n_requested, n_kept = len(poses), len(views_meta)
    assert n_kept >= MIN_BACKGROUND_VIEWS, (
        f"{tag}: only {n_kept}/{n_requested} background views usable after "
        f"skipping {len(skipped_views)} point-blank wall views "
        f"(>{BG_POINT_BLANK_FRAC:.0%} of pixels closer than "
        f"{BG_POINT_BLANK_DEPTH_M} m); need >= {MIN_BACKGROUND_VIEWS} — "
        f"lookat/wall clamp misconfigured for this room")

    extra = {"capture_args": ctx.capture_args(
        radius_requested=float(requested_radius),
        n_views_requested=n_requested, n_views_kept=n_kept,
        view_radii=view_radii, skipped_views=skipped_views,
        lookat=ctx.table_lookat.tolist(),
        table_diag=ctx.table_diag, table_top_z=ctx.table_top_z,
        table_geoms=ctx.table_geoms)}
    if masks_dir_name:
        extra["masks_dir"] = masks_dir_name
        robot_qpos = {}  # stow config, for the masked-mode trainer
        for name in env.sim.model.joint_names:
            if name and name.startswith(("robot0_", "gripper0_")):
                a = env.sim.model.get_joint_qpos_addr(name)
                if isinstance(a, (int, np.integer)):
                    robot_qpos[name] = float(np.asarray(env.sim.data.qpos)[int(a)])
        extra["robot_stow_qpos"] = robot_qpos
    cap.write_transforms(out_dir, ctx.base_transforms(
        "background", views_meta, validation, **extra))
    print(f"[capture] {tag}: {n_kept}/{n_requested} views OK "
          f"({len(skipped_views)} point-blank wall views skipped, radii "
          f"[{min(view_radii):.2f}, {max(view_radii):.2f}] m; "
          f"validation: {validation}) -> {out_dir}")


# ── objects ──────────────────────────────────────────────────────────────────

def capture_object(ctx: TaskContext, joint_name: str) -> None:
    env, addr, args = ctx.env, ctx.addr, ctx.args
    comp_name = object_component_name(joint_name)
    component = f"objects/{comp_name}"
    out_dir = ctx.out_dir(component)
    tag = f"{ctx.task_name}/{component}"
    model = env.sim.model

    cap.set_flat_state(env, ctx.base_state)
    grave_state = cap.graveyard_movables(env, addr, exclude=(joint_name,))

    obj_gids = np.asarray(ctx.movable_gids[joint_name], dtype=np.int64)
    other_gids = np.concatenate(
        [np.asarray(g, dtype=np.int64)
         for n, g in ctx.movable_gids.items() if n != joint_name]
        or [np.zeros(0, dtype=np.int64)])

    # radius from the RESTING bbox (bounding spheres — works for meshes), then
    # float high enough that the -20 deg ring stays above the table top.
    mn, mx = cap.geoms_world_aabb(model, env.sim.data, obj_gids)
    diag = float(np.linalg.norm(mx - mn))
    radius = radius_requested = max(4.0 * diag, 0.35)
    # interior clamp (same wall logic as the background orbit): large objects
    # (e.g. the basket, diag 0.57 m -> 4x = 2.27 m) would put low-ring cameras
    # OUTSIDE the room (measured: wall at 1.99 m along az 0 vs camera at
    # 2.13 m -> object 0 px behind the wall). A uniform clamped radius keeps
    # the orbit rings coherent; the object stays fully in frame down to
    # ~1.3x diag at fovy 45 deg, so require 1.5x diag with margin.
    lookat_xy = ((mn + mx) / 2.0)[:2]
    wall_r = min(cap.wall_distance_2d(model, env.sim.data, lookat_xy, az)
                 for az in np.arange(0.0, 360.0, 7.5)) - 0.30
    if radius > wall_r:
        min_frame_r = max(1.5 * diag, 0.35)
        assert wall_r >= min_frame_r, (
            f"{tag}: wall-clamped orbit radius {wall_r:.2f} m cannot frame "
            f"object diag {diag:.2f} m (needs >= {min_frame_r:.2f} m) — "
            f"object too large for an interior orbit in this room")
        radius = wall_r
    ring_drop = radius * math.sin(math.radians(20.0))  # -20 deg camera drop
    # the -20 deg ring camera sits at lookat_z - ring_drop with lookat_z =
    # rest_center_z + dz; require camera_z >= table_top_z + 0.03, anchored on
    # the ACTUAL table top — not the bounding-sphere aabb bottom, which dips
    # ~4 cm below the surface for mesh geoms and left the ring cameras below
    # the tabletop (measured: a table edge 4 cm in front of a below-top
    # camera blanks the whole view).
    rest_center_z = float(mn[2] + mx[2]) / 2.0
    dz = max(float(args.float_dz),
             ring_drop + 0.03 + ctx.table_top_z - rest_center_z)

    cap.set_flat_state(env, cap.float_object_state(grave_state, addr, joint_name, dz))
    # objects are always captured with the robot hidden (plan §5.1); in
    # 'masked' contingency mode any alpha-0 seg leakage is masked, not fatal.
    hide = cap.hide_robot_alpha0(env)
    masks_dir_name = "masks" if ctx.robot_hide == "masked" else None
    if masks_dir_name:
        os.makedirs(os.path.join(out_dir, masks_dir_name), exist_ok=True)
    forbidden = (other_gids if masks_dir_name
                 else np.concatenate([other_gids, ctx.robot_gids]))

    # capture body pose (asset frame anchor: Gaussians are stored body-local
    # against exactly this pose, plan §5.2/§6.1)
    bid = cap.object_body_id(model, joint_name)
    body_p = np.asarray(env.sim.data.xpos[bid], dtype=np.float64).copy()
    body_q = np.asarray(env.sim.data.xquat[bid], dtype=np.float64).copy()  # wxyz

    fmn, fmx = cap.geoms_world_aabb(model, env.sim.data, obj_gids)
    lookat = (fmn + fmx) / 2.0
    poses = cap.object_orbit_poses(lookat, radius)

    validator = cap.PoseValidator(env.sim.model, ctx.K)
    views_meta = []
    try:
        for i, pose in enumerate(tqdm.tqdm(poses, desc=tag, unit="view")):
            vtag = f"{tag} view {i}"
            rgb, depth, seg, c2w = ctx.render_orbit_view(pose, vtag)
            ctx.check_view(vtag, depth, seg, forbidden,
                           "other-movable" if masks_dir_name
                           else "other-movable/robot")
            n_obj = int(np.isin(seg, obj_gids).sum())
            assert n_obj >= MIN_OBJECT_PIXELS, (
                f"{vtag}: object {joint_name!r} covers only {n_obj} px "
                f"(< {MIN_OBJECT_PIXELS}) — occluded or orbit radius "
                f"{radius:.2f} m / float dz {dz:.2f} m mis-fit")
            validator.try_view(env.sim.data, c2w, seg, depth, i,
                               pose.elevation_deg, tag)
            prefix = cap.write_view_files(out_dir, i, rgb, depth, seg)
            if masks_dir_name:
                cap.write_mask_file(os.path.join(out_dir, masks_dir_name), i,
                                    np.isin(seg, ctx.robot_gids))
            views_meta.append({"file_prefix": prefix,
                               "c2w_opencv": c2w.tolist(),
                               "cam_params": pose.cam_params()})
    finally:
        hide.restore()

    validation = validator.finalize(tag)
    assert len(views_meta) == 48, f"{tag}: {len(views_meta)} views != 48"

    extra = {
        "joint_name": joint_name,
        "body_name": model.body_id2name(bid),
        "body_pose": {"p": body_p.tolist(), "q_wxyz": body_q.tolist()},
        "object_geom_ids": obj_gids.tolist(),
        "capture_args": ctx.capture_args(
            radius=float(radius), radius_requested=float(radius_requested),
            lookat=lookat.tolist(), float_dz=float(dz), bbox_diag=diag),
    }
    if masks_dir_name:
        extra["masks_dir"] = masks_dir_name
    cap.write_transforms(out_dir, ctx.base_transforms(
        component, views_meta, validation, **extra))
    print(f"[capture] {tag}: 48 views OK, float dz={dz:.3f} m, "
          f"radius={radius:.2f} m (validation: {validation}) -> {out_dir}")


# ── robot ────────────────────────────────────────────────────────────────────

def robot_joint_qpos_addrs(model) -> Tuple[List[str], List[int]]:
    """1-dof robot-stack joints (arm + gripper) in qpos-address order."""
    pairs = []
    for name in model.joint_names:
        if not name or not name.startswith(("robot0_", "gripper0_")):
            continue
        a = model.get_joint_qpos_addr(name)
        if isinstance(a, (int, np.integer)):
            pairs.append((int(a), name))
    pairs.sort()
    assert pairs, "no 1-dof robot joints found — unexpected for a LIBERO scene"
    return [n for _a, n in pairs], [a for a, _n in pairs]


def load_demo_robot_qpos(hdf5_path: str, state_cols: List[int],
                         expected_state_len: int) -> np.ndarray:
    """(n_frames, n_joints) robot qpos pool from every demo's flattened MuJoCo
    states (layout [time, qpos, qvel] — state_cols already carry the +1)."""
    pool = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in f["data"].keys():
            states = f["data"][demo_key]["states"]
            assert states.shape[1] == expected_state_len, (
                f"{hdf5_path}:{demo_key}: state length {states.shape[1]} != "
                f"sim state length {expected_state_len} — demo/model mismatch")
            pool.append(np.asarray(states[:, state_cols], dtype=np.float64))
    assert pool, f"{hdf5_path}: no demos"
    return np.concatenate(pool, axis=0)


def farthest_point_select(pool: np.ndarray, n: int,
                          seed_idx: Sequence[int]) -> List[int]:
    """Deterministic farthest-point sampling (Euclidean in joint space),
    seeded with ``seed_idx`` (gripper extremes) — the plan's dedup."""
    assert len(pool) >= n, f"pool of {len(pool)} frames < {n} requested configs"
    sel = list(dict.fromkeys(int(i) for i in seed_idx))
    d = np.full(len(pool), np.inf)
    for s in sel:
        d = np.minimum(d, np.linalg.norm(pool - pool[s], axis=1))
    while len(sel) < n:
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(pool - pool[i], axis=1))
    return sel[:n]


def build_robot_configs(pool: np.ndarray, joint_names: List[str],
                        rng: np.random.Generator) -> List[dict]:
    """36 FPS demo configs + joint1-shifted copies of the first 12 at
    ±20°/±30° (alternating pairs → ~60 total); each entry
    {'qpos': (n_joints,), 'source', 'joint1_shift_deg'}."""
    if len(pool) > MAX_ROBOT_POOL:
        keep = np.sort(rng.choice(len(pool), size=MAX_ROBOT_POOL, replace=False))
        pool = pool[keep]

    grip_cols = [i for i, n in enumerate(joint_names) if n.startswith("gripper0_")]
    assert grip_cols, f"no gripper joints among {joint_names}"
    j1_col = joint_names.index("robot0_joint1")

    # seed FPS with each gripper joint's extreme frames so the span assertion
    # below holds by construction (open AND closed grippers in the config set)
    seed_idx = []
    for col in grip_cols:
        seed_idx += [int(np.argmin(pool[:, col])), int(np.argmax(pool[:, col]))]
    sel = farthest_point_select(pool, N_BASE_ROBOT_CONFIGS, seed_idx)
    configs = [{"qpos": pool[i].copy(), "source": "demo",
                "joint1_shift_deg": 0.0} for i in sel]

    # joint1-shifted copies: the θ-grid tails (±30°) must be inside the robot
    # asset's training distribution (plan §5.1)
    n_skipped = 0
    for i in range(N_SHIFTED_ROBOT_CONFIGS):
        base = configs[i]["qpos"]
        for shift_deg in JOINT1_SHIFT_PAIRS[i % len(JOINT1_SHIFT_PAIRS)]:
            q = base.copy()
            q[j1_col] += math.radians(shift_deg)
            if abs(q[j1_col]) > PANDA_JOINT1_LIMIT - JOINT1_SHIFT_MARGIN:
                n_skipped += 1
                continue
            configs.append({"qpos": q, "source": "joint1_shift",
                            "joint1_shift_deg": float(shift_deg)})
    n_shift = len(configs) - N_BASE_ROBOT_CONFIGS
    assert n_shift >= N_SHIFTED_ROBOT_CONFIGS, (
        f"only {n_shift} joint1-shifted configs inside the joint limit "
        f"({n_skipped} skipped) — demo joint1 values sit unexpectedly close to "
        f"±{PANDA_JOINT1_LIMIT:.2f} rad; θ-grid tails would be out of "
        f"distribution for the robot asset")

    # gripper-range span assertion (plan §5.1): open AND closed must be present
    cfg = np.stack([c["qpos"] for c in configs])
    for col in grip_cols:
        pmin, pmax = float(pool[:, col].min()), float(pool[:, col].max())
        span = pmax - pmin
        tol = GRIPPER_SPAN_TOL * span + 1e-6
        cmin, cmax = float(cfg[:, col].min()), float(cfg[:, col].max())
        assert cmin <= pmin + tol and cmax >= pmax - tol, (
            f"robot config set does not span the demo gripper range for "
            f"{joint_names[col]}: configs [{cmin:.4f}, {cmax:.4f}] vs demos "
            f"[{pmin:.4f}, {pmax:.4f}] — gripper open/closed both required")
    return configs


def capture_robot(ctx: TaskContext) -> None:
    env, addr, args = ctx.env, ctx.addr, ctx.args
    out_dir = ctx.out_dir("robot")
    tag = f"{ctx.task_name}/robot"
    model = env.sim.model

    cap.set_flat_state(env, ctx.base_state)
    grave_state = cap.graveyard_movables(env, addr)  # robot stays visible

    joint_names, qpos_addrs = robot_joint_qpos_addrs(model)
    hdf5_path = os.path.join(args.hdf5_dir, f"{ctx.task_name}_demo.hdf5")
    assert os.path.exists(hdf5_path), (
        f"{hdf5_path} not found — robot configs are sampled from the task's "
        f"demo frames (--hdf5_dir)")
    state_cols = [1 + a for a in qpos_addrs]  # +1: time slot
    pool = load_demo_robot_qpos(hdf5_path, state_cols, len(ctx.base_state))
    rng = np.random.default_rng(args.seed)
    configs = build_robot_configs(pool, joint_names, rng)

    # link inventory (F6 companion): every robot-stack body with geoms gets its
    # world pose recorded per config, and a geom->link map for seg labeling.
    link_bids = cap.robot_body_ids(model, with_geoms_only=True)
    link_names = [model.body_id2name(b) for b in link_bids]
    geom_bodyid = np.asarray(model.geom_bodyid)
    geom_to_link = {str(g): model.body_id2name(int(geom_bodyid[g]))
                    for g in ctx.robot_gids.tolist()}
    forbidden = np.concatenate(
        [np.asarray(g, dtype=np.int64) for g in ctx.movable_gids.values()])

    validator = cap.PoseValidator(env.sim.model, ctx.K)
    views_meta: list = []
    configs_meta: list = []
    pbar = tqdm.tqdm(total=len(configs) * 16, desc=tag, unit="view")
    for cfg_idx, cfg in enumerate(configs):
        state = grave_state.copy()
        for a, v in zip(qpos_addrs, cfg["qpos"]):
            state[1 + a] = v
        cap.set_flat_state(env, state)

        mn, mx = cap.geoms_world_aabb(model, env.sim.data, ctx.robot_gids)
        diag = float(np.linalg.norm(mx - mn))
        radius = float(np.clip(1.25 * diag, 1.0, 2.0))
        lookat = (mn + mx) / 2.0
        az_offset = (cfg_idx * 137.508) % 45.0  # golden-ish per-config spread
        poses = cap.robot_orbit_poses(lookat, radius, az_offset_deg=az_offset)

        view_ids = []
        for pose in poses:
            i = len(views_meta)
            vtag = f"{tag} config {cfg_idx} view {i}"
            rgb, depth, seg, c2w = ctx.render_orbit_view(pose, vtag)
            n_robot = int(np.isin(seg, ctx.robot_gids).sum())
            frac_close = float((depth < BG_POINT_BLANK_DEPTH_M).mean())
            # room furniture/walls can pinch an orbit camera (measured: one
            # config's az-12.5 deg eye landed INSIDE the living_room shelf
            # mesh — 98% of pixels closer than 0.5 m, robot 0 px, while the
            # neighboring config's eye 27 cm away was clear). Pull the camera
            # in along the same ray until the view clears; direction coverage
            # is preserved, the robot just fills more of the frame.
            while (n_robot < MIN_ROBOT_PIXELS
                   or frac_close > BG_POINT_BLANK_FRAC):
                new_r = pose.distance * ROBOT_RADIUS_SHRINK
                if new_r < ROBOT_MIN_RADIUS:
                    break
                pose = cap.OrbitPose(pose.lookat, new_r, pose.azimuth_deg,
                                     pose.elevation_deg)
                rgb, depth, seg, c2w = ctx.render_orbit_view(pose, vtag)
                n_robot = int(np.isin(seg, ctx.robot_gids).sum())
                frac_close = float((depth < BG_POINT_BLANK_DEPTH_M).mean())
            ctx.check_view(vtag, depth, seg, forbidden, "movable")
            assert n_robot >= MIN_ROBOT_PIXELS, (
                f"{vtag}: robot covers only {n_robot} px (< {MIN_ROBOT_PIXELS}) "
                f"even after pulling the camera in to {pose.distance:.2f} m "
                f"(ring radius {radius:.2f} m) — occluded by room geometry or "
                f"hidden by mistake")
            validator.try_view(env.sim.data, c2w, seg, depth, i,
                               pose.elevation_deg, tag)
            prefix = cap.write_view_files(out_dir, i, rgb, depth, seg)
            views_meta.append({"file_prefix": prefix,
                               "c2w_opencv": c2w.tolist(),
                               "cam_params": pose.cam_params()})
            view_ids.append(i)
            pbar.update(1)

        link_poses = {
            name: {"p": np.asarray(env.sim.data.xpos[b], dtype=np.float64).tolist(),
                   "q_wxyz": np.asarray(env.sim.data.xquat[b], dtype=np.float64).tolist()}
            for name, b in zip(link_names, link_bids)}
        configs_meta.append({
            "qpos": np.asarray(cfg["qpos"], dtype=np.float64).tolist(),
            "view_ids": view_ids,
            "link_poses": link_poses,
            "source": cfg["source"],
            "joint1_shift_deg": cfg["joint1_shift_deg"],
        })
    pbar.close()

    validation = validator.finalize(tag)
    assert len(views_meta) == 16 * len(configs), (
        f"{tag}: {len(views_meta)} views != 16 x {len(configs)} configs")

    cap.write_transforms(out_dir, ctx.base_transforms(
        "robot", views_meta, validation,
        configs=configs_meta,
        robot_joint_names=joint_names,
        link_names=link_names,
        geom_to_link=geom_to_link,
        capture_args=ctx.capture_args(
            n_configs=len(configs), n_base_configs=N_BASE_ROBOT_CONFIGS,
            hdf5=hdf5_path, pool_frames=int(len(pool)))))
    print(f"[capture] {tag}: {len(configs)} configs x 16 views OK "
          f"(validation: {validation}) -> {out_dir}")


# ── driver ───────────────────────────────────────────────────────────────────

def capture_task(task_name: str, components: List[str],
                 args: argparse.Namespace, facts: dict) -> None:
    print(f"[capture] === {task_name} ({', '.join(components)}) ===")
    env = build_control_env(task_name, args.image_size)
    ctx = None
    try:
        ctx = TaskContext(task_name, env, args, facts)
        if "background" in components:
            capture_background(ctx)
        if "objects" in components:
            for joint_name in ctx.addr.obj_qpos_slices:
                capture_object(ctx, joint_name)
        if "robot" in components:
            capture_robot(ctx)
    finally:
        if ctx is not None:
            ctx.close()
        else:
            env.close()


def main() -> None:
    args = parse_args()
    facts = load_render_facts(args.facts)  # G7 gate: refuses pass!=true

    if args.task == "all":
        files = sorted(glob.glob(os.path.join(args.hdf5_dir, "*_demo.hdf5")))
        assert files, f"no *_demo.hdf5 in {args.hdf5_dir!r}"
        tasks = sorted((task_name_from_hdf5(f) for f in files),
                       key=lambda n: task_name_to_suite_and_ids[n][2])
    else:
        assert args.task in task_name_to_suite_and_ids, (
            f"unknown task {args.task!r}; known LIBERO tasks include e.g. "
            f"{list(task_name_to_suite_and_ids)[:3]} ...")
        tasks = [args.task]
    components = list(COMPONENTS) if args.component == "all" else [args.component]

    for task_name in tasks:
        capture_task(task_name, components, args, facts)
    print(f"[capture] done: {len(tasks)} task(s), components {components}")


if __name__ == "__main__":
    main()

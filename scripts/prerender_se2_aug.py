"""Offline SE(2) pre-render driver.

For every (episode, angle) pair, rewrites the demo's flattened MuJoCo states by
a world-frame yaw R_z(theta) about the robot base (objects + robot joint 1),
re-renders both cameras through the SAME LIBERO ControlEnv pipeline that
produced the base zarr, and writes the images to a separate augmentation zarr:

    images/{agentview_rgb,robot0_eye_in_hand_rgb}/angle_00..NN  uint8 [n_steps,H,W,3]
    meta/{angles_deg, episode_ends, valid_mask, done_mask, p_base, state_offset}

Numerics (actions, proprio) are NOT stored here -- the training dataset keeps
them in RAM from the base zarr and rotates them analytically
(oat.equi.se2_transforms); only images need the simulator.

Angle index 0 is always theta=0 (re-rendered too, so both aug-on and aug-off
arms share render provenance). Resumable via meta/done_mask; shardable across
processes via --tasks (image chunks and per-episode meta chunks make writes of
different tasks disjoint -- start shards only after one run has created the
output zarr).

Render sources (plan IMPLEMENTATION_PLAN_gs_render_phase0.md §7, M6):
    --renderer oracle (default)  MuJoCo re-render, byte-identical to before.
    --renderer gs                Gaussian-Splatting composite renderer
                                 (oat.gsaug.compose.GSCompositeRenderer). MuJoCo
                                 stays in the loop for states/contacts/camera
                                 poses (G3); only rasterization is replaced.
                                 Validity is oracle-owned (G4): delta is COPIED
                                 from --oracle-zarr meta/state_offset, and
                                 valid_mask / p_base / angles_deg / episode_ends
                                 are hard-asserted equal to the oracle aug zarr.

Provenance meta (G9) -- written by BOTH modes; documented choice: the value is
stored twice, (a) meta/render_source, a shape-(1,) VLenUTF8 zarr string array
(travels with meta through zarr copies), and (b) root.attrs['render_source']
(cheap to read without touching arrays); absence of both == 'oracle' (zarrs
predating GS support; resuming such a zarr creates the meta). Values:
{'oracle', 'gs', 'gs_hybrid0', 'gs_oracle_robot'}. GS mode additionally writes
root.attrs['gs_manifest_sha1'] = {task_name: manifest_sha1} per task.
--hybrid-zero-from ORACLE_ZARR (arm A5): after all tasks complete, copies
images/*/angle_00 verbatim from the oracle zarr and sets render_source
'gs_hybrid0'.
GS-shard provenance caveat: with concurrent GS-mode --tasks shards, the
meta/render_source creation and the root.attrs['gs_manifest_sha1'] merge are
NOT concurrency-safe -- run GS shards sequentially or accept last-writer-wins
on the attrs (each shard's report JSON carries its own authoritative sha1 map).

Usage:
    MUJOCO_GL=egl python scripts/prerender_se2_aug.py \
        --base_zarr data/libero/libero10_N500.zarr \
        --hdf5_dir third_party/LIBERO/libero/datasets/libero_10 \
        --out data/libero/libero10_N500_se2aug.zarr --resume

GS mode:
    MUJOCO_GL=egl python scripts/prerender_se2_aug.py --renderer gs \
        --gs-assets-dir data/libero/gs_assets \
        --oracle-zarr data/libero/libero10_N500_se2aug.zarr \
        --out data/libero/libero10_N500_se2aug_gs.zarr
"""

import os

# Must be set before robosuite / mujoco / libero are imported (one GL context
# per process).
os.environ.setdefault("MUJOCO_GL", "egl")

if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import datetime
import json
import shutil
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Optional, Set, Tuple

import h5py
import numcodecs
import numpy as np
import tqdm
import zarr

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.common.replay_buffer import ReplayBuffer
from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.demo_alignment import calibrate_state_offset, match_episodes
from oat.env.libero.se2_state_rewrite import (
    check_joint1_limit,
    check_object_penetration,
    check_objects_in_bounds,
    check_support_contacts,
    object_xy_from_state,
    resolve_addresses,
    rewrite_state,
    table_top_xy_aabb,
)
from oat.equi.se2_transforms import rotate_xy


# zarr image key -> raw robosuite observation key (un-flipped)
CAMERAS = OrderedDict(
    [
        ("agentview_rgb", "agentview_image"),
        ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand_image"),
    ]
)

# zarr image key -> mujoco camera name for GSCompositeRenderer; the compose
# contract takes explicit names (robosuite obs key minus '_image' is NOT
# assumed there)
GS_CAMERAS = OrderedDict(
    [
        ("agentview_rgb", "agentview"),
        ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand"),
    ]
)

N_SAMPLED_FRAMES = 5           # per (episode, angle) for inline stats
CALIB_EPISODES = 3             # episodes per task for delta calibration (probe 2 pattern)
RENDER_WARN_M = 1e-3           # eef-pos consistency warn threshold (probe 2 owns the hard gate)
PIXEL_DIFF_WARN = 1.0          # theta=0 mean |render - base| per task, uint8 units
PIXEL_DIFF_FAIL = 5.0
# GS-mode theta=0 gate (plan §7.4): gross-error gate only -- GS-vs-stored MAD
# legitimately sits in the 5-20 band (baked GS appearance vs mujoco shading);
# 25 still catches a wrong delta / flip / camera. WARN raised to 20 (not 1) so
# the expected appearance gap does not spam a warning for every task.
PIXEL_DIFF_WARN_GS = 20.0
PIXEL_DIFF_FAIL_GS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base_zarr", type=str, default="data/libero/libero10_N500.zarr")
    parser.add_argument(
        "--hdf5_dir", type=str, default="third_party/LIBERO/libero/datasets/libero_10",
        help="directory of LIBERO *_demo.hdf5 files (per-demo MuJoCo states)")
    parser.add_argument("--out", type=str, default="data/libero/libero10_N500_se2aug.zarr")
    parser.add_argument(
        "--angles", type=str, default="0,10,-10,20,-20,30,-30",
        help="comma-separated yaw angles in DEGREES; must include 0")
    parser.add_argument(
        "--tasks", type=str, default="all",
        help="'all' or comma-separated task names (shard unit)")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="skip (episode, angle) pairs already in meta/done_mask; --no-resume wipes --out")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--preview_dir", type=str, default=None,
        help="optional dir for one agentview PNG per (task, angle)")
    parser.add_argument(
        "--no-support-check", dest="support_check", action="store_false", default=True,
        help="disable the post-rewrite support-contact validity check")
    parser.add_argument("--xy_margin", type=float, default=0.05,
                        help="INSET from the table edge in meters: rotated object "
                             "centers must stay inside the physical table-top xy "
                             "AABB shrunk by this margin")
    parser.add_argument("--joint_margin", type=float, default=0.05,
                        help="joint-1 limit margin in radians")
    parser.add_argument(
        "--renderer", type=str, choices=("oracle", "gs"), default="oracle",
        help="pixel source: 'oracle' = MuJoCo re-render (default, unchanged "
             "behavior); 'gs' = Gaussian-Splatting composite renderer (plan §7)")
    parser.add_argument(
        "--gs-assets-dir", dest="gs_assets_dir", type=str,
        default="data/libero/gs_assets",
        help="per-task GS asset root (<dir>/<task>/{assets,manifest.json})")
    parser.add_argument(
        "--oracle-zarr", dest="oracle_zarr", type=str, default=None,
        help="ORACLE aug zarr for G4 cross-checks in GS mode (delta source; "
             "valid_mask/p_base/angles_deg/episode_ends must match)")
    parser.add_argument(
        "--oracle-crosscheck", dest="oracle_crosscheck",
        action=argparse.BooleanOptionalAction, default=True,
        help="G4 cross-asserts against --oracle-zarr; --no-oracle-crosscheck "
             "is for smoke tests ONLY (delta falls back to 1, report tainted)")
    parser.add_argument(
        "--hybrid-zero-from", dest="hybrid_zero_from", type=str, default=None,
        help="arm A5: after all tasks complete, copy images/*/angle_00 verbatim "
             "from this ORACLE aug zarr and set render_source='gs_hybrid0' "
             "(requires --renderer gs)")
    args = parser.parse_args()
    if args.renderer == "gs" and args.oracle_crosscheck and not args.oracle_zarr:
        parser.error("--renderer gs requires --oracle-zarr (the ORACLE aug zarr "
                     "to copy delta from and cross-assert against, G4); pass "
                     "--no-oracle-crosscheck only for smoke tests")
    if args.hybrid_zero_from is not None and args.renderer != "gs":
        parser.error("--hybrid-zero-from requires --renderer gs")
    return args


def parse_angles(angles_str: str) -> np.ndarray:
    angles_deg = [float(a) for a in angles_str.split(",") if a.strip() != ""]
    if len(set(angles_deg)) != len(angles_deg):
        raise ValueError(f"duplicate angles in --angles: {angles_str}")
    if 0.0 not in angles_deg:
        raise ValueError("--angles must include 0 (angle-0 pass records eef "
                         "references and support/penetration baselines, and both "
                         "training arms read angle-0 images)")
    if angles_deg[0] != 0.0:
        # angle-0 must run first (eef references + per-frame check baselines)
        angles_deg.remove(0.0)
        angles_deg.insert(0, 0.0)
        print(f"[prerender] reordered angles so 0 is first: {angles_deg}")
    return np.asarray(angles_deg, dtype=np.float64)


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


def open_out_zarr(
    out_path: str,
    resume: bool,
    n_steps: int,
    n_episodes: int,
    n_angles: int,
    image_size: int,
    angles_deg: np.ndarray,
    base_episode_ends: np.ndarray,
    render_source: str,
    accept_render_sources: Tuple[str, ...],
) -> zarr.Group:
    if not resume and os.path.exists(out_path):
        print(f"[prerender] --no-resume: wiping {out_path}")
        shutil.rmtree(out_path)

    root = zarr.open(out_path, mode="a")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.NOSHUFFLE)

    images = root.require_group("images")
    for cam_key in CAMERAS:
        cam_group = images.require_group(cam_key)
        for k in range(n_angles):
            cam_group.require_dataset(
                f"angle_{k:02d}",
                shape=(n_steps, image_size, image_size, 3),
                chunks=(1, image_size, image_size, 3),
                dtype=np.uint8,
                compressor=compressor,
            )

    meta = root.require_group("meta")
    # per-episode chunks -> concurrent per-task shards touch disjoint chunks
    meta.require_dataset("angles_deg", shape=(n_angles,), chunks=(n_angles,), dtype=np.float64)
    meta.require_dataset("episode_ends", shape=(n_episodes,), chunks=(n_episodes,), dtype=np.int64)
    meta.require_dataset("valid_mask", shape=(n_episodes, n_angles), chunks=(1, n_angles), dtype=bool)
    meta.require_dataset("done_mask", shape=(n_episodes, n_angles), chunks=(1, n_angles), dtype=bool)
    meta.require_dataset("p_base", shape=(n_episodes, 3), chunks=(1, 3), dtype=np.float32)
    meta.require_dataset("state_offset", shape=(n_episodes,), chunks=(1,), dtype=np.int8,
                         fill_value=-1)

    fresh = bool(np.all(meta["episode_ends"][:] == 0))
    if fresh:
        meta["angles_deg"][:] = angles_deg
        meta["episode_ends"][:] = base_episode_ends
    else:
        assert np.array_equal(meta["episode_ends"][:], base_episode_ends), (
            f"{out_path} was built against a different base zarr "
            "(meta/episode_ends mismatch); use --no-resume to rebuild")
        assert np.allclose(meta["angles_deg"][:], angles_deg), (
            f"{out_path} was built with different --angles "
            f"({meta['angles_deg'][:].tolist()} vs {angles_deg.tolist()}); "
            "use --no-resume to rebuild")

    # provenance (G9): meta/render_source shape-(1,) string array mirrored into
    # root.attrs['render_source'] (see module docstring for the choice). A
    # pre-existing zarr WITHOUT the meta predates GS support and is by
    # definition 'oracle' (backward compat) -- create the meta accordingly,
    # then refuse to resume it under a different render source.
    if "render_source" not in meta:
        rs = meta.create_dataset(
            "render_source", shape=(1,), chunks=(1,), dtype=object,
            object_codec=numcodecs.VLenUTF8())
        rs[0] = render_source if fresh else "oracle"
    stored_source = str(meta["render_source"][0])
    assert stored_source in accept_render_sources, (
        f"{out_path} carries meta/render_source={stored_source!r} but this run "
        f"would write {render_source!r}; refusing to mix render sources in one "
        "zarr -- use --no-resume or a different --out")
    root.attrs["render_source"] = stored_source
    return root


def save_preview(preview_dir: str, task_name: str, k: int, angle_deg: float,
                 img: np.ndarray) -> None:
    os.makedirs(preview_dir, exist_ok=True)
    path = os.path.join(preview_dir, f"{task_name}_angle_{k:02d}_{angle_deg:+.0f}deg.png")
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(img).save(path)
        except ImportError:
            print("[prerender] WARN: neither imageio nor PIL available; skipping previews")


def eef_pos_from_sim(env: ControlEnv) -> np.ndarray:
    """World eef position after set_state + sim.forward(), identical to the
    'robot0_eef_pos' observable (site_xpos of the grip site) but without
    triggering camera renders -- used to rebuild angle-0 references on resume."""
    robot = env.env.robots[0]
    return np.array(env.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64)


def sampled_frames(ep_len: int) -> np.ndarray:
    return np.unique(np.round(np.linspace(0, ep_len - 1, N_SAMPLED_FRAMES)).astype(int))


def render_episode_angle(
    env: ControlEnv,
    addr,
    states: np.ndarray,
    delta: int,
    theta: float,
    k: int,
    ep_start: int,
    ep_len: int,
    out_images: Dict[str, zarr.Array],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    frames_for_stats: np.ndarray,
    eef_refs: Dict[int, np.ndarray],
    support_fail_frames: Set[int],
    penetration_fail_frames: Set[int],
    on_sampled_frame: Optional[Callable[[int, dict], None]] = None,
    gs_renderer: Optional[object] = None,
) -> Tuple[bool, str, int, int]:
    """Render one (episode, angle): rewrite every frame's state, run validity
    checks, write both flipped camera images at global index ep_start + t.

    At k == 0 the support and penetration checks are statistics only (counted,
    never reject) and every failing frame index is recorded into
    ``support_fail_frames`` / ``penetration_fail_frames``; at k != 0 a failure
    rejects the pair ONLY IF the same frame passed that check at k == 0 (the
    rotation broke support / rotated an object into a fixture) -- frames
    already failing at angle 0 are rotation-independent (e.g. object briefly
    airborne mid-drop, or squeezed in the gripper) and only counted. Every
    other failed check rejects at k != 0 as before.
    Returns (valid, reject_reason, n_support_fail, n_penetration_fail) where
    the counts cover angle-0 failures at k == 0 and rotation-independent ones
    at k != 0.

    With ``gs_renderer`` set (GS mode, plan §7) the ONLY pixel-path change is
    that ``env.regenerate_obs_from_state`` is replaced by
    ``gs_renderer.render(env, state_rw)``, which returns DATASET-ORIENTED uint8
    images per zarr camera key (compose.py already applies the F2 flip -- no
    extra flip here). render() itself does set_state_from_flattened + forward
    (G3), so the contact-based validity checks below still read the rewritten
    state; eef refs come from the forwarded sim (obs observables do not exist
    without a mujoco render), and ``on_sampled_frame`` receives the GS image
    dict instead of an obs dict.
    """
    frame_set = set(int(t) for t in frames_for_stats)
    n_support_fail = 0
    n_pen_fail = 0
    for t in range(ep_len):
        state = states[min(t + delta, len(states) - 1)]
        state_rw = rewrite_state(state, theta, addr)

        ok, reason = check_joint1_limit(state_rw, addr, margin=args.joint_margin)
        if not ok:
            return False, f"joint1_limit: {reason}", n_support_fail, n_pen_fail
        if k != 0:
            ok, reason = check_objects_in_bounds(
                state_rw, addr, bounds[0], bounds[1], margin=args.xy_margin)
            if not ok:
                return (False, f"objects_out_of_bounds: {reason}",
                        n_support_fail, n_pen_fail)

        if gs_renderer is None:
            obs = env.regenerate_obs_from_state(state_rw)
        else:
            # G3: render() runs set_state_from_flattened + forward internally
            # and never touches the mujoco renderer; env.sim below is at the
            # rewritten state for the contact checks.
            gs_images = gs_renderer.render(env, state_rw)
            obs = None

        if args.support_check:
            ok, reason = check_support_contacts(env, addr)
            if not ok:
                if k == 0:
                    n_support_fail += 1  # angle 0 is never rejected by this check
                    support_fail_frames.add(t)
                elif t in support_fail_frames:
                    n_support_fail += 1  # also failed at angle 0: rotation-independent
                else:
                    return (False, f"support_contact: {reason}",
                            n_support_fail, n_pen_fail)
            # shallow-penetration guard: objects rotated INTO a fixture still
            # have contacts (so support passes) but are unphysically embedded
            ok, reason = check_object_penetration(env, addr)
            if not ok:
                if k == 0:
                    n_pen_fail += 1  # angle 0 is never rejected by this check
                    penetration_fail_frames.add(t)
                elif t in penetration_fail_frames:
                    n_pen_fail += 1  # also failed at angle 0: rotation-independent
                else:
                    return False, reason, n_support_fail, n_pen_fail

        if gs_renderer is None:
            for cam_key, obs_key in CAMERAS.items():
                # match dataset_conversion's vertical flip of raw robosuite frames
                img = np.flip(obs[obs_key], axis=0).astype(np.uint8)
                out_images[cam_key][ep_start + t] = img
        else:
            for cam_key in CAMERAS:
                # already dataset-oriented uint8 (compose applies F2) -- no flip
                out_images[cam_key][ep_start + t] = gs_images[cam_key]

        if t in frame_set:
            if k == 0:
                if gs_renderer is None:
                    eef_refs[t] = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                else:
                    # same site_xpos the observable reads, minus the render
                    eef_refs[t] = eef_pos_from_sim(env)
            if on_sampled_frame is not None:
                on_sampled_frame(t, obs if gs_renderer is None else gs_images)
    return True, "", n_support_fail, n_pen_fail


def main() -> None:
    args = parse_args()
    angles_deg = parse_angles(args.angles)
    thetas = np.deg2rad(angles_deg)
    n_angles = len(angles_deg)

    # ── base zarr: small meta into RAM, images stay lazy ────────────────────
    base_root = zarr.open(args.base_zarr, mode="r")
    episode_ends = base_root["meta/episode_ends"][:]
    n_episodes = len(episode_ends)
    n_steps = int(episode_ends[-1])
    episode_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    base_images = {cam: base_root["data"][cam] for cam in CAMERAS}
    for cam, arr in base_images.items():
        assert arr.shape[0] == n_steps, f"base zarr {cam} length != episode_ends[-1]"
    if args.renderer == "gs":
        # G8: GS mode must render at the dataset-native resolution --
        # intrinsics come from cam_fovy at --image_size and the theta=0 frames
        # are MAD-gated pixelwise against the base zarr; checked at startup,
        # BEFORE any renderer is built.
        for cam, arr in base_images.items():
            assert int(arr.shape[1]) == int(args.image_size), (
                f"[G8] base zarr '{cam}' stores {arr.shape[1]}x{arr.shape[2]} "
                f"frames but --image_size is {args.image_size}; GS mode must "
                f"render at the dataset-native resolution -- rerun with "
                f"--image_size {int(arr.shape[1])}")

    # theta=0 pixel gate thresholds (GS mode uses the gross-error gate, §7.4)
    pixel_warn = PIXEL_DIFF_WARN_GS if args.renderer == "gs" else PIXEL_DIFF_WARN
    pixel_fail = PIXEL_DIFF_FAIL_GS if args.renderer == "gs" else PIXEL_DIFF_FAIL

    # ── oracle aug zarr: G4 cross-check source (GS mode only) ───────────────
    # validity is oracle-owned: delta is COPIED from here, and valid_mask /
    # p_base must match exactly (checked per task and at the end)
    oracle_meta: Optional[Dict[str, np.ndarray]] = None
    if args.renderer == "gs" and args.oracle_crosscheck:
        assert os.path.exists(args.oracle_zarr), (
            f"--oracle-zarr {args.oracle_zarr} does not exist; run the ORACLE "
            "prerender first (GS mode copies delta from it and cross-asserts "
            "validity against it, G4)")
        oracle_root = zarr.open(args.oracle_zarr, mode="r")
        assert np.array_equal(oracle_root["meta/episode_ends"][:], episode_ends), (
            f"[G4] {args.oracle_zarr} meta/episode_ends != base zarr episode "
            "ends -- the oracle aug zarr was built against a different base zarr")
        assert np.allclose(oracle_root["meta/angles_deg"][:], angles_deg), (
            f"[G4] {args.oracle_zarr} meta/angles_deg "
            f"{oracle_root['meta/angles_deg'][:].tolist()} != --angles "
            f"{angles_deg.tolist()} -- GS run must use the oracle's angle grid")
        oracle_meta = {
            "valid_mask": oracle_root["meta/valid_mask"][:],
            "done_mask": oracle_root["meta/done_mask"][:],
            "state_offset": oracle_root["meta/state_offset"][:],
            "p_base": oracle_root["meta/p_base"][:].astype(np.float64),
        }
    elif args.renderer == "gs":
        print("[prerender] WARN: --no-oracle-crosscheck: G4 asserts DISABLED and "
              "delta falls back to 1 -- smoke tests only, report is tainted")

    # ── episode -> hdf5 demo (content matching; conversion scrambled order) ─
    print(f"[prerender] matching {n_episodes} zarr episodes to demos in {args.hdf5_dir} ...")
    replay_buffer = ReplayBuffer.create_from_path(args.base_zarr, mode="r")
    matches = match_episodes(replay_buffer, args.hdf5_dir)
    assert len(matches) == n_episodes

    # task from the matched hdf5 FILE NAME (authoritative), not zarr task_uid
    ep_task = [task_name_from_hdf5(path) for path, _ in matches]
    zarr_task_uid = base_root["data/task_uid"][:, 0][episode_starts]
    for e in range(n_episodes):
        expected_uid = task_name_to_suite_and_ids[ep_task[e]][2]
        if int(zarr_task_uid[e]) != expected_uid:
            print(f"[prerender] WARN: episode {e} zarr task_uid {int(zarr_task_uid[e])} "
                  f"!= {expected_uid} ({ep_task[e]}); trusting the matched hdf5")
    task_to_eps: Dict[str, List[int]] = OrderedDict()
    for t in sorted(set(ep_task), key=lambda name: task_name_to_suite_and_ids[name][2]):
        task_to_eps[t] = [e for e in range(n_episodes) if ep_task[e] == t]

    if args.tasks != "all":
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in selected if t not in task_to_eps]
        assert not unknown, (
            f"unknown --tasks {unknown}; available: {list(task_to_eps.keys())}")
        task_to_eps = OrderedDict((t, task_to_eps[t]) for t in task_to_eps if t in selected)
    scope_eps = [e for eps in task_to_eps.values() for e in eps]

    # ── A5 --hybrid-zero-from startup checks (plan §7.6) ────────────────────
    if args.hybrid_zero_from:
        # the 'gs_hybrid0' stamp covers the WHOLE zarr; plan §7.6 runs the A5
        # copy after ALL tasks complete -- refusing task shards here keeps a
        # partially rendered zarr from being stamped as a finished arm
        assert args.tasks == "all", (
            "--hybrid-zero-from stamps render_source='gs_hybrid0' on the "
            "whole zarr (plan §7.6: after all tasks complete); refusing to "
            "combine it with --tasks shards -- run the shards first, then one "
            "final --hybrid-zero-from pass with --tasks all")
        hz_src_root = zarr.open(args.hybrid_zero_from, mode="r")
        hz_src = hz_src_root.attrs.get("render_source")
        if hz_src is None and "meta/render_source" in hz_src_root:
            hz_src = str(hz_src_root["meta/render_source"][0])
        hz_src = "oracle" if hz_src is None else str(hz_src)
        assert hz_src == "oracle", (
            f"--hybrid-zero-from {args.hybrid_zero_from} carries "
            f"render_source={hz_src!r}, not 'oracle' -- A5 copies ORACLE "
            "theta=0 frames; point it at the oracle aug zarr")
        assert np.asarray(hz_src_root["meta/done_mask"][:, 0]).all(), (
            f"--hybrid-zero-from {args.hybrid_zero_from}: meta/done_mask[:,0] "
            "has unfinished episodes -- finish the oracle prerender first")
        assert np.asarray(hz_src_root["meta/valid_mask"][:, 0]).all(), (
            f"--hybrid-zero-from {args.hybrid_zero_from}: meta/valid_mask"
            "[:,0] has invalid theta=0 entries -- the oracle aug zarr is "
            "corrupt (theta=0 must be valid for every episode)")

    # render source this run writes; a finished A5 zarr ('gs_hybrid0') may be
    # re-opened only when --hybrid-zero-from is passed again
    render_source = "gs" if args.renderer == "gs" else "oracle"
    accept_sources = (render_source,) + (
        ("gs_hybrid0",) if args.hybrid_zero_from else ())
    out_root = open_out_zarr(
        args.out, args.resume, n_steps, n_episodes, n_angles,
        args.image_size, angles_deg, episode_ends,
        render_source=render_source, accept_render_sources=accept_sources)
    out_images_by_angle = [
        {cam: out_root["images"][cam][f"angle_{k:02d}"] for cam in CAMERAS}
        for k in range(n_angles)
    ]
    valid_mask = out_root["meta/valid_mask"]
    done_mask = out_root["meta/done_mask"]
    p_base_arr = out_root["meta/p_base"]
    state_offset_arr = out_root["meta/state_offset"]

    # ── report accumulators ─────────────────────────────────────────────────
    reject_reasons: Counter = Counter()
    support_fail_angle0 = 0
    support_fail_rot_independent = 0
    penetration_fail_angle0 = 0
    penetration_fail_rot_independent = 0
    per_task_delta: Dict[str, int] = {}
    per_task_deltas: Dict[str, List[int]] = {}
    per_task_support_fail0: Dict[str, int] = {}
    per_task_support_fail_rotind: Dict[str, int] = {}
    per_task_pen_fail0: Dict[str, int] = {}
    per_task_pen_fail_rotind: Dict[str, int] = {}
    per_task_table_aabb: Dict[str, dict] = {}
    per_task_occupancy: Dict[str, dict] = {}
    per_task_pixel_diff: Dict[str, List[float]] = {}
    per_task_pixel_diff_wrist: Dict[str, List[float]] = {}
    per_task_render_err: Dict[str, List[float]] = {}
    gs_manifest_sha1_by_task: Dict[str, str] = {}
    render_warn_count = 0
    episode_to_demo = [
        {"episode": e, "task": ep_task[e],
         "hdf5": matches[e][0], "demo_key": matches[e][1]}
        for e in range(n_episodes)
    ]
    report_path = f"{args.out}.report.json"
    if args.tasks != "all":
        # per-shard report; don't clobber other shards' reports
        shard_tag = "_".join(sorted(task_to_eps.keys()))[:60]
        report_path = f"{args.out}.report.{shard_tag}.json"

    def write_report() -> None:
        vm = valid_mask[:]
        dm = done_mask[:]
        scope = np.asarray(scope_eps, dtype=np.int64)
        render_err_all = [x for errs in per_task_render_err.values() for x in errs]
        pixel_all = [x for diffs in per_task_pixel_diff.values() for x in diffs]

        def _pct(values, qs=(50, 90, 99)):
            if not values:
                return None
            v = np.asarray(values)
            out = {f"p{q}": float(np.percentile(v, q)) for q in qs}
            out.update(n=int(len(v)), max=float(v.max()), mean=float(v.mean()))
            return out

        report = {
            "created": datetime.datetime.now().isoformat(),
            "args": vars(args),
            "renderer": args.renderer,
            # current zarr value ('gs_hybrid0' once the A5 copy has run)
            "render_source": str(out_root["meta/render_source"][0]),
            # False taints the report: G4 asserts were skipped, delta assumed
            "oracle_crosscheck": bool(args.oracle_crosscheck),
            "pixel_diff_thresholds": {"warn": pixel_warn, "fail": pixel_fail},
            "gs_manifest_sha1": gs_manifest_sha1_by_task or None,
            "angles_deg": angles_deg.tolist(),
            "tasks": list(task_to_eps.keys()),
            "n_episodes_in_scope": len(scope_eps),
            "per_angle_done": dm[scope].sum(axis=0).tolist(),
            "per_angle_valid_rate": [
                float(vm[scope, k][dm[scope, k]].mean()) if dm[scope, k].any() else None
                for k in range(n_angles)
            ],
            "reject_reasons": dict(reject_reasons),
            "support_fail_angle0_frames": support_fail_angle0,
            "support_fail_rotation_independent_frames": support_fail_rot_independent,
            "penetration_fail_angle0_frames": penetration_fail_angle0,
            "penetration_fail_rotation_independent_frames":
                penetration_fail_rot_independent,
            "per_task": {
                t: {
                    "state_offset_delta": per_task_delta.get(t),
                    "state_offset_deltas": per_task_deltas.get(t),
                    "support_fail_angle0_frames": per_task_support_fail0.get(t),
                    "support_fail_rotation_independent_frames":
                        per_task_support_fail_rotind.get(t),
                    "penetration_fail_angle0_frames": per_task_pen_fail0.get(t),
                    "penetration_fail_rotation_independent_frames":
                        per_task_pen_fail_rotind.get(t),
                    # physical bounds used for rejection (objects must stay
                    # inside, inset by --xy_margin)
                    "table_xy_aabb": per_task_table_aabb.get(t),
                    # empirical theta=0 occupancy box: REPORT STATISTIC ONLY
                    "object_occupancy_bounds": per_task_occupancy.get(t),
                    "n_episodes": len(eps),
                    "gs_manifest_sha1": gs_manifest_sha1_by_task.get(t),
                    "valid_per_angle": vm[np.asarray(eps)].sum(axis=0).tolist(),
                    "pixel_diff_theta0": _pct(per_task_pixel_diff.get(t, [])),
                    "pixel_diff_theta0_wrist":
                        _pct(per_task_pixel_diff_wrist.get(t, [])),
                    "render_consistency_m": _pct(per_task_render_err.get(t, [])),
                }
                for t, eps in task_to_eps.items()
            },
            "render_consistency_m": _pct(render_err_all),
            "render_consistency_warns_gt_1mm": render_warn_count,
            "pixel_diff_theta0": _pct(pixel_all),
            "pixel_diff_theta0_wrist": _pct(
                [x for d in per_task_pixel_diff_wrist.values() for x in d]),
            "episode_to_demo": episode_to_demo,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"[prerender] report written to {report_path}")

    def fail(msg: str) -> None:
        write_report()
        raise RuntimeError(msg)

    def check_valid_mask_vs_oracle(eps_list: List[int], where: str) -> None:
        """G4: GS run must reproduce oracle validity exactly (validity is
        decided in state space, never from GS output). No-op outside GS mode
        or with --no-oracle-crosscheck."""
        if oracle_meta is None:
            return
        eps_arr = np.asarray(eps_list, dtype=np.int64)
        if not oracle_meta["done_mask"][eps_arr].all():
            fail(f"[G4] oracle zarr {args.oracle_zarr} has unfinished "
                 f"(episode, angle) pairs among episodes of {where} "
                 "(meta/done_mask False) -- finish the ORACLE prerender before "
                 "the GS pass")
        vm = valid_mask[:][eps_arr]
        ovm = oracle_meta["valid_mask"][eps_arr]
        if not np.array_equal(vm, ovm):
            bad = np.argwhere(vm != ovm)
            examples = [(int(eps_arr[i]), int(k)) for i, k in bad[:10]]
            fail(f"[G4] valid_mask mismatch vs oracle after {where}: "
                 f"{len(bad)} differing (episode, angle_idx) entries, first "
                 f"{examples} -- state-space checks must be renderer-independent")

    total_units = sum(len(eps) for eps in task_to_eps.values()) * n_angles
    pbar = tqdm.tqdm(total=total_units, desc="prerender", unit="ep-angle")

    # ── per task ────────────────────────────────────────────────────────────
    for task_name, eps in task_to_eps.items():
        pbar.set_postfix_str(task_name[:40])
        env = build_control_env(task_name, args.image_size)
        env.reset()  # once per task
        sim_state_len = len(env.sim.get_state().flatten())

        h5_cache: Dict[str, h5py.File] = {}
        states_by_ep: Dict[int, np.ndarray] = {}
        addr_by_ep: Dict[int, object] = {}
        eef_refs_by_ep: Dict[int, Dict[int, np.ndarray]] = {}
        support_fail_frames_by_ep: Dict[int, Set[int]] = {}
        penetration_fail_frames_by_ep: Dict[int, Set[int]] = {}
        frames_by_ep: Dict[int, np.ndarray] = {}
        task_xy_min = np.full(2, np.inf)
        task_xy_max = np.full(2, -np.inf)
        preview_done = set()
        support_fail_local = 0
        support_fail_rotind_local = 0
        pen_fail_local = 0
        pen_fail_rotind_local = 0

        gs_renderer = None
        try:
            if args.renderer == "gs":
                # imported HERE, not at module top: oracle runs must never
                # import gsplat/torch. compose loads gs_render_facts.json
                # (asserts pass, G7), the per-task manifest (G9) and asserts
                # model_xml_sha1 vs the live env on first render.
                from oat.gsaug.compose import GSCompositeRenderer
                gs_renderer = GSCompositeRenderer(
                    task_assets_dir=os.path.join(args.gs_assets_dir, task_name),
                    cameras=OrderedDict(GS_CAMERAS),
                    resolution=args.image_size,
                    device="cuda:0",
                )
                sha1 = str(gs_renderer.manifest["manifest_sha1"])
                gs_manifest_sha1_by_task[task_name] = sha1
                # G9: per-task provenance on the zarr itself (attrs merge so
                # per-task shards accumulate rather than clobber). A resumed
                # zarr recording a DIFFERENT sha for this task means its
                # already-written images came from other assets -- never
                # silently overwrite the recorded provenance.
                stored = dict(out_root.attrs.get("gs_manifest_sha1", {}))
                prev_sha1 = stored.get(task_name)
                if prev_sha1 is not None and str(prev_sha1) != sha1:
                    fail(f"{task_name}: this zarr records gs_manifest_sha1 "
                         f"{str(prev_sha1)[:12]}… but the current manifest is "
                         f"{sha1[:12]}… -- assets retrained since this zarr's "
                         "images were rendered; rebuild with --no-resume or "
                         "restore the original assets")
                stored[task_name] = sha1
                out_root.attrs["gs_manifest_sha1"] = stored

            # physical table-top xy extent from the constructed model (objects
            # are at their reset placement here); this -- inset by --xy_margin
            # -- is what rotated object centers are checked against. The arm
            # co-rotates exactly with the world, so reachability needs no check.
            table_min_xy, table_max_xy, table_geoms = table_top_xy_aabb(
                env, resolve_addresses(env))
            bounds = (table_min_xy, table_max_xy)
            print(f"[prerender] {task_name}: table-top xy AABB "
                  f"x=[{table_min_xy[0]:.4f}, {table_max_xy[0]:.4f}] "
                  f"y=[{table_min_xy[1]:.4f}, {table_max_xy[1]:.4f}] m "
                  f"(reject outside inset {args.xy_margin} m) "
                  f"from geoms {table_geoms}")
            per_task_table_aabb[task_name] = {
                "min_xy": table_min_xy.tolist(),
                "max_xy": table_max_xy.tolist(),
                "geoms": table_geoms,
                "inset_margin": args.xy_margin,
            }

            # load all demo states of the task (small: ~100 KB / episode)
            for e in eps:
                hdf5_path, demo_key = matches[e]
                if hdf5_path not in h5_cache:
                    h5_cache[hdf5_path] = h5py.File(hdf5_path, "r")
                demo = h5_cache[hdf5_path]["data"][demo_key]
                states = np.asarray(demo["states"][:], dtype=np.float64)
                if states.shape[1] != sim_state_len:
                    # defensive fallback -- not expected for LIBERO (same model
                    # per task; placement is entirely in qpos)
                    print(f"[prerender] WARN: state length {states.shape[1]} != "
                          f"sim {sim_state_len} for ep {e}; reloading demo model XML")
                    env.reset_from_xml_string(demo.attrs["model_file"])
                    sim_state_len = len(env.sim.get_state().flatten())
                    assert states.shape[1] == sim_state_len, (
                        f"episode {e} ({task_name}): demo state length "
                        f"{states.shape[1]} != sim state length {sim_state_len} "
                        "even after model_file reload")
                states_by_ep[e] = states
                addr_by_ep[e] = resolve_addresses(env)
                p_base_arr[e] = addr_by_ep[e].p_base.astype(np.float32)
                if oracle_meta is not None:
                    # G4: fresh resolve_addresses must agree with the oracle's
                    # recorded rotation center to 1e-6 (same model, same base)
                    p_err = float(np.max(np.abs(
                        addr_by_ep[e].p_base - oracle_meta["p_base"][e])))
                    if p_err > 1e-6:
                        fail(f"[G4] episode {e} ({task_name}): p_base from fresh "
                             f"resolve_addresses differs from oracle zarr by "
                             f"{p_err:.2e} m > 1e-6 -- model/base mismatch")
                ep_len = int(episode_ends[e] - episode_starts[e])
                frames_by_ep[e] = sampled_frames(ep_len)
                eef_refs_by_ep[e] = {}
                support_fail_frames_by_ep[e] = set()
                penetration_fail_frames_by_ep[e] = set()

            # theta=0 must already be valid for anything resumed
            for e in eps:
                if done_mask[e, 0] and not valid_mask[e, 0]:
                    fail(f"episode {e} ({task_name}) has done_mask[.,0] set but "
                         "valid_mask[.,0]=False from a previous run; theta=0 must be "
                         "valid for every episode -- rebuild with --no-resume")

            if args.renderer == "gs":
                # G4: delta is COPIED from the oracle zarr's meta/state_offset,
                # never recalibrated here -- calibration compares ORACLE renders
                # against stored frames and is renderer-independent; both arms
                # must index demo states identically.
                if oracle_meta is not None:
                    offs = [int(oracle_meta["state_offset"][e]) for e in eps]
                    if len(set(offs)) != 1:
                        fail(f"{task_name}: oracle meta/state_offset disagrees "
                             f"across episodes (episode -> delta: "
                             f"{dict(zip(eps, offs))}) -- oracle aug zarr "
                             "inconsistent for this task")
                    delta = offs[0]
                    if delta not in (0, 1):
                        fail(f"{task_name}: oracle meta/state_offset={delta} not "
                             "in (0, 1) -- fill value -1 means the ORACLE "
                             "prerender never processed this task; finish it "
                             "first")
                else:
                    delta = 1
                    print(f"[prerender] WARN {task_name}: --no-oracle-crosscheck "
                          "-- assuming state offset delta=1 (measured on all 10 "
                          "LIBERO-10 tasks); smoke tests only")
                per_task_delta[task_name] = delta
                for e in eps:
                    state_offset_arr[e] = delta
            else:
                # calibrate obs/state offset delta over up to CALIB_EPISODES episodes
                # per task, asserted consistent (probe 2 pattern; expected delta=1
                # everywhere -- confirmed on all 10 tasks by probe_render_consistency)
                def render_fn(state):
                    obs = env.regenerate_obs_from_state(np.asarray(state, dtype=np.float64))
                    return np.flip(obs["agentview_image"], axis=0).astype(np.uint8)

                calib_eps = eps[:CALIB_EPISODES]
                deltas = []
                for e in calib_eps:
                    s, e_end = int(episode_starts[e]), int(episode_ends[e])
                    deltas.append(int(calibrate_state_offset(
                        env, states_by_ep[e], base_images["agentview_rgb"][s:e_end],
                        render_fn)))
                if len(set(deltas)) != 1:
                    fail(f"{task_name}: inconsistent state offset across calibration "
                         f"episodes (episode -> delta: {dict(zip(calib_eps, deltas))}) "
                         "-- demo/zarr alignment broken for this task")
                delta = deltas[0]
                assert delta in (0, 1), f"unexpected state offset {delta} for {task_name}"
                per_task_delta[task_name] = delta
                per_task_deltas[task_name] = deltas
                for e in eps:
                    state_offset_arr[e] = delta

            # per-task empirical object-occupancy box over the theta=0 states:
            # REPORT STATISTIC ONLY -- rejection uses the physical table AABB
            # above (the empirical box, +-margin, rejected exactly the rotated
            # placements the augmentation exists to create)
            for e in eps:
                states, addr = states_by_ep[e], addr_by_ep[e]
                ep_len = int(episode_ends[e] - episode_starts[e])
                for t in range(ep_len):
                    xy = object_xy_from_state(states[min(t + delta, len(states) - 1)], addr)
                    if len(xy):
                        task_xy_min = np.minimum(task_xy_min, xy.min(axis=0))
                        task_xy_max = np.maximum(task_xy_max, xy.max(axis=0))
            if np.all(np.isfinite(task_xy_min)):
                per_task_occupancy[task_name] = {
                    "min_xy": task_xy_min.tolist(),
                    "max_xy": task_xy_max.tolist(),
                }

            per_task_pixel_diff.setdefault(task_name, [])
            per_task_render_err.setdefault(task_name, [])

            # ── pass 1: angle 0 for ALL episodes of the task ────────────────
            for e in eps:
                states, addr = states_by_ep[e], addr_by_ep[e]
                ep_start = int(episode_starts[e])
                ep_len = int(episode_ends[e] - episode_starts[e])
                eef_refs = eef_refs_by_ep[e]

                if done_mask[e, 0]:
                    # resumed: rebuild angle-0 eef references and the per-frame
                    # support/penetration-fail sets without rendering (set_state
                    # + sim.forward is enough for all; no camera involved). With
                    # the support check off only the sampled eef-ref frames need
                    # sim calls.
                    support_fails = support_fail_frames_by_ep[e]
                    pen_fails = penetration_fail_frames_by_ep[e]
                    ref_frames = set(int(t) for t in frames_by_ep[e])
                    rebuild_ts = range(ep_len) if args.support_check else sorted(ref_frames)
                    for t in rebuild_ts:
                        rw = rewrite_state(states[min(t + delta, len(states) - 1)], 0.0, addr)
                        env.set_state(rw)
                        env.sim.forward()
                        if args.support_check:
                            ok, _ = check_support_contacts(env, addr)
                            if not ok:
                                support_fails.add(int(t))
                                support_fail_local += 1
                            ok, _ = check_object_penetration(env, addr)
                            if not ok:
                                pen_fails.add(int(t))
                                pen_fail_local += 1
                        if int(t) in ref_frames:
                            eef_refs[int(t)] = eef_pos_from_sim(env)
                    pbar.update(1)
                    continue

                if gs_renderer is None:
                    def on_theta0_frame(t, obs, _ep_start=ep_start):
                        # theta=0 render vs base zarr images (uint8 mean abs diff).
                        # Only the world-fixed agentview feeds the delta gate: the
                        # wrist camera rides the arm, so sub-mm settling
                        # differences shift its whole close-up image (mean MAD ~8
                        # on contact-rich kitchen scenes) without indicating a
                        # wrong state offset. Wrist diffs are reported only.
                        for cam_key, obs_key in CAMERAS.items():
                            rendered = np.flip(obs[obs_key], axis=0).astype(np.int16)
                            stored = base_images[cam_key][_ep_start + t].astype(np.int16)
                            mad = float(np.abs(rendered - stored).mean())
                            if cam_key == "agentview_rgb":
                                per_task_pixel_diff[task_name].append(mad)
                            else:
                                per_task_pixel_diff_wrist.setdefault(
                                    task_name, []).append(mad)
                else:
                    def on_theta0_frame(t, gs_imgs, _ep_start=ep_start):
                        # GS renders arrive dataset-oriented (compose applies
                        # F2) -- compare directly, no flip. Same agentview-only
                        # gating, but against the gross-error thresholds (§7.4).
                        for cam_key in CAMERAS:
                            rendered = gs_imgs[cam_key].astype(np.int16)
                            stored = base_images[cam_key][_ep_start + t].astype(np.int16)
                            mad = float(np.abs(rendered - stored).mean())
                            if cam_key == "agentview_rgb":
                                per_task_pixel_diff[task_name].append(mad)
                            else:
                                per_task_pixel_diff_wrist.setdefault(
                                    task_name, []).append(mad)

                ok, reason, n_support_fail, n_pen_fail = render_episode_angle(
                    env, addr, states, delta, float(thetas[0]), 0, ep_start, ep_len,
                    out_images_by_angle[0], bounds, args, frames_by_ep[e],
                    eef_refs, support_fail_frames_by_ep[e],
                    penetration_fail_frames_by_ep[e],
                    on_sampled_frame=on_theta0_frame, gs_renderer=gs_renderer)
                support_fail_local += n_support_fail
                pen_fail_local += n_pen_fail
                if not ok:
                    reject_reasons[f"angle0/{reason.split(':')[0]}"] += 1
                    valid_mask[e, 0] = False
                    done_mask[e, 0] = True
                    fail(f"theta=0 rejected for episode {e} ({task_name}): {reason} "
                         "-- theta=0 must be valid for every episode")
                valid_mask[e, 0] = True
                done_mask[e, 0] = True
                if args.preview_dir and 0 not in preview_done:
                    save_preview(args.preview_dir, task_name, 0, float(angles_deg[0]),
                                 out_images_by_angle[0]["agentview_rgb"][ep_start])
                    preview_done.add(0)
                pbar.update(1)

            # theta=0 pixel-diff gate before burning time on the other angles
            diffs = per_task_pixel_diff[task_name]
            if diffs:
                task_diff = float(np.mean(diffs))
                if task_diff > pixel_fail:
                    fail(f"{task_name}: theta=0 mean pixel diff vs base zarr = "
                         f"{task_diff:.2f} > {pixel_fail} (uint8) -- wrong state "
                         "offset or render mismatch")
                if task_diff > pixel_warn:
                    print(f"[prerender] WARN {task_name}: theta=0 mean pixel diff "
                          f"{task_diff:.2f} > {pixel_warn}")

            # ── pass 2: nonzero angles ──────────────────────────────────────
            for e in eps:
                states, addr = states_by_ep[e], addr_by_ep[e]
                ep_start = int(episode_starts[e])
                ep_len = int(episode_ends[e] - episode_starts[e])
                p_base_xy = addr.p_base[:2]
                eef_refs = eef_refs_by_ep[e]

                for k in range(1, n_angles):
                    if done_mask[e, k]:
                        pbar.update(1)
                        continue
                    theta = float(thetas[k])

                    if gs_renderer is None:
                        def on_frame(t, obs, _theta=theta):
                            # inline render-consistency: eef pos must map by
                            # R_z(theta) about the base (WARN only; probe 2 gates)
                            nonlocal render_warn_count
                            ref = eef_refs.get(int(t))
                            if ref is None:
                                return
                            expected = rotate_xy(ref, _theta, center_xy=p_base_xy)
                            err = float(np.linalg.norm(
                                np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - expected))
                            per_task_render_err[task_name].append(err)
                            if err > RENDER_WARN_M:
                                render_warn_count += 1
                                if render_warn_count <= 10:
                                    print(f"[prerender] WARN render-consistency: ep {e} "
                                          f"angle {angles_deg[k]:+.0f} frame {t}: "
                                          f"eef err {err:.4f} m")
                    else:
                        def on_frame(t, _gs_imgs, _theta=theta):
                            # same check, eef from the forwarded sim (G3): obs
                            # observables do not exist without a mujoco render
                            nonlocal render_warn_count
                            ref = eef_refs.get(int(t))
                            if ref is None:
                                return
                            expected = rotate_xy(ref, _theta, center_xy=p_base_xy)
                            err = float(np.linalg.norm(
                                eef_pos_from_sim(env) - expected))
                            per_task_render_err[task_name].append(err)
                            if err > RENDER_WARN_M:
                                render_warn_count += 1
                                if render_warn_count <= 10:
                                    print(f"[prerender] WARN render-consistency: ep {e} "
                                          f"angle {angles_deg[k]:+.0f} frame {t}: "
                                          f"eef err {err:.4f} m")

                    ok, reason, n_rot_indep, n_pen_rot_indep = render_episode_angle(
                        env, addr, states, delta, theta, k, ep_start, ep_len,
                        out_images_by_angle[k], bounds, args, frames_by_ep[e],
                        eef_refs, support_fail_frames_by_ep[e],
                        penetration_fail_frames_by_ep[e],
                        on_sampled_frame=on_frame, gs_renderer=gs_renderer)
                    support_fail_rotind_local += n_rot_indep
                    pen_fail_rotind_local += n_pen_rot_indep
                    if not ok:
                        reject_reasons[reason.split(":")[0]] += 1
                    valid_mask[e, k] = ok
                    done_mask[e, k] = True
                    if ok and args.preview_dir and k not in preview_done:
                        save_preview(args.preview_dir, task_name, k, float(angles_deg[k]),
                                     out_images_by_angle[k]["agentview_rgb"][ep_start])
                        preview_done.add(k)
                    pbar.update(1)
        finally:
            for f in h5_cache.values():
                f.close()
            env.close()
        support_fail_angle0 += support_fail_local
        support_fail_rot_independent += support_fail_rotind_local
        penetration_fail_angle0 += pen_fail_local
        penetration_fail_rot_independent += pen_fail_rotind_local
        per_task_support_fail0[task_name] = support_fail_local
        per_task_support_fail_rotind[task_name] = support_fail_rotind_local
        per_task_pen_fail0[task_name] = pen_fail_local
        per_task_pen_fail_rotind[task_name] = pen_fail_rotind_local
        if support_fail_local:
            print(f"[prerender] note {task_name}: {support_fail_local} theta=0 frames "
                  "failed the support check (statistic only; angle 0 never rejected)")
        if support_fail_rotind_local:
            print(f"[prerender] note {task_name}: {support_fail_rotind_local} k!=0 frames "
                  "failed the support check rotation-independently (also failed at "
                  "theta=0; statistic only, never rejects)")
        if pen_fail_local:
            print(f"[prerender] note {task_name}: {pen_fail_local} theta=0 frames "
                  "failed the penetration check (statistic only; angle 0 never rejected)")
        if pen_fail_rotind_local:
            print(f"[prerender] note {task_name}: {pen_fail_rotind_local} k!=0 frames "
                  "failed the penetration check rotation-independently (also failed "
                  "at theta=0; statistic only, never rejects)")
        # G4: the completed task's valid_mask rows must equal the oracle's
        # (no-op outside GS mode / with --no-oracle-crosscheck)
        check_valid_mask_vs_oracle(eps, f"task {task_name}")

    pbar.close()

    # final hard invariant: theta=0 valid for every episode in scope
    vm0 = valid_mask[:, 0]
    bad = [e for e in scope_eps if not vm0[e]]
    if bad:
        fail(f"theta=0 not valid for episodes {bad}")
    # G4 final invariant over everything in scope
    check_valid_mask_vs_oracle(scope_eps, "final invariant")

    # ── arm A5 assembly: theta=0 images verbatim from the oracle zarr ───────
    if args.hybrid_zero_from:
        hz_root = zarr.open(args.hybrid_zero_from, mode="r")
        assert np.array_equal(hz_root["meta/episode_ends"][:], episode_ends), (
            f"--hybrid-zero-from {args.hybrid_zero_from} meta/episode_ends != "
            "this run's episode_ends -- different base zarr, refusing to copy")
        assert float(hz_root["meta/angles_deg"][0]) == 0.0, (
            f"--hybrid-zero-from {args.hybrid_zero_from} angle_00 is "
            f"{float(hz_root['meta/angles_deg'][0])} deg, not 0 -- refusing")
        print(f"[prerender] hybrid0: copying images/*/angle_00 verbatim from "
              f"{args.hybrid_zero_from}")
        for cam_key in CAMERAS:
            src = hz_root["images"][cam_key]["angle_00"]
            dst = out_root["images"][cam_key]["angle_00"]
            assert src.shape == dst.shape and src.dtype == dst.dtype, (
                f"hybrid0 {cam_key}: shape/dtype mismatch {src.shape}/{src.dtype}"
                f" vs {dst.shape}/{dst.dtype} (different --image_size?)")
            # per-episode slabs; image chunks are (1, H, W, 3) so any episode
            # boundary is chunk-aligned
            for e in tqdm.trange(n_episodes, desc=f"hybrid0 {cam_key}",
                                 unit="ep", leave=False):
                s, e_end = int(episode_starts[e]), int(episode_ends[e])
                dst[s:e_end] = src[s:e_end]
        out_root["meta/render_source"][0] = "gs_hybrid0"
        out_root.attrs["render_source"] = "gs_hybrid0"

    write_report()
    dm = done_mask[:]
    vm = valid_mask[:]
    scope = np.asarray(scope_eps, dtype=np.int64)
    print(f"[prerender] done: {int(dm[scope].sum())}/{len(scope_eps) * n_angles} "
          f"(episode, angle) pairs done, "
          f"{int(vm[scope].sum())} valid; report: {report_path}")


if __name__ == "__main__":
    main()

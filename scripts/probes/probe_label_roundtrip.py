"""Probe 3: label round-trip — replay SE(2)-transformed actions in the
rotated scene and check the demo still succeeds.

Samples ``--n`` (episode, angle) pairs, stratified by goal anchoring (see
``TASK_GOAL_ANCHORING``): >= ~2/3 from tasks whose goal predicates reference
only CO-ROTATING entities (movable objects / the arm, which the SE(2) state
rewrite rotates), the rest from tasks anchored to NON-ROTATING fixtures
(stove/cabinet/microwave/caddy/table regions live in model.body_pos and do
not rotate) — those rotated replays CANNOT satisfy the goal, so that stratum
is reported as informational only (expected-fail per the plan's
non-rotating-fixture residual; training excludes such pairs anyway).

The SAME episodes are also replayed at theta=0 as controls. Per episode
(demo via content matching):

  * control replay: regenerate ``states[delta]``, step the raw demo actions;
    record the EEF trajectory + any-step success. Control DIVERGENCE = the
    per-step gap between this replay and the demo's own stored obs ee_pos
    (native playback already drifts — this calibrates the tolerance).
  * rotated replay: ``rewrite_state(states[delta], theta)``, step
    ``rotate_action_chunk(actions, theta, rotate_rotation)``; per-step error
    = | p_rot[t] - R_z(theta)-about-base( p_ref[t] ) |.

``rotate_rotation`` comes from probe 1's ``controller_frame_rot`` (warns and
defaults to world if the probe file is missing).

OSC nullspace protocol: robosuite's OSC adds nullspace torques toward
``controller.initial_joint``, captured at reset — the UNROTATED posture. So
after EVERY state injection (control and rotated alike, for symmetry) we call
``controller.update_initial_joints`` with the just-injected arm qpos;
otherwise the controller drags joint 1 back toward theta=0 for the whole
rotated rollout.

PASS iff (configurable), evaluated on the COROTATING stratum only: rotated
success >= --success_ratio * control success AND median per-step error <
--median_tol AND rotated p90 error <= --p90_factor * control-divergence p90.

Usage:
    MUJOCO_GL=egl python scripts/probes/probe_label_roundtrip.py --n 20
"""

import os

# must be set before any robosuite / mujoco / libero import
os.environ.setdefault("MUJOCO_GL", "egl")

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import datetime
import json

import h5py
import numpy as np
import zarr

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.common.replay_buffer import ReplayBuffer
from oat.env.libero.demo_alignment import calibrate_state_offset, match_episodes
from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.factory import get_subtasks
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.equi.se2_transforms import rotate_action_chunk, rotate_xy


# ── goal-anchoring stratification ────────────────────────────────────────────
# Under the SE(2) world rotation only movable objects (free-jointed, in the
# BDDL ``:objects`` section) and the arm rotate; fixtures (``:fixtures``,
# baked into model.body_pos/body_quat) do NOT. A rotated replay can therefore
# only satisfy goals whose predicates reference co-rotating entities.
# Classification derived by reading the ``(:goal ...)`` sections of
# third_party/LIBERO/libero/libero/bddl_files/libero_10/*.bddl (predicate
# cited per entry); keys match oat.env.libero.factory.get_subtasks('libero10')
# and the ``*_demo.hdf5`` basenames.
COROTATING = "corotating"
FIXTURE_ANCHORED = "fixture_anchored"
TASK_GOAL_ANCHORING = {
    # (Turnon flat_stove_1) & (On moka_pot_1 flat_stove_1_cook_region);
    # flat_stove_1 is a :fixture.
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it":
        FIXTURE_ANCHORED,
    # (Close white_cabinet_1_bottom_region) &
    # (In akita_black_bowl_1 white_cabinet_1_bottom_region);
    # white_cabinet_1 is a :fixture.
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it":
        FIXTURE_ANCHORED,
    # (In white_yellow_mug_1 microwave_1_heating_region) & (Close microwave_1);
    # microwave_1 is a :fixture.
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it":
        FIXTURE_ANCHORED,
    # (On moka_pot_1 flat_stove_1_cook_region) & (On moka_pot_2 ...) &
    # (Turnon flat_stove_1); flat_stove_1 is a :fixture.
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove":
        FIXTURE_ANCHORED,
    # (In alphabet_soup_1 basket_1_contain_region) & (In cream_cheese_1 ...);
    # basket_1 is a movable :object (co-rotates).
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket":
        COROTATING,
    # (In alphabet_soup_1 basket_1_contain_region) & (In tomato_sauce_1 ...);
    # basket_1 is a movable :object (co-rotates).
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket":
        COROTATING,
    # (In cream_cheese_1 basket_1_contain_region) & (In butter_1 ...);
    # basket_1 is a movable :object (co-rotates).
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket":
        COROTATING,
    # (On porcelain_mug_1 plate_1) & (On white_yellow_mug_1 plate_2);
    # plate_1/plate_2 are movable :objects (co-rotate).
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate":
        COROTATING,
    # (On porcelain_mug_1 plate_1) & (On chocolate_pudding_1
    # living_room_table_plate_right_region); the second conjunct is anchored
    # to the living_room_table :fixture, which does not rotate.
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate":
        FIXTURE_ANCHORED,
    # (In black_book_1 desk_caddy_1_back_contain_region);
    # desk_caddy_1 is a :fixture.
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy":
        FIXTURE_ANCHORED,
}

FIXTURE_STRATUM_NOTE = (
    "informational, expected-fail: under the SE(2) world rotation only "
    "movable objects and the arm rotate, fixtures do not, so rotated replays "
    "cannot satisfy fixture-anchored goal predicates (plan's "
    "non-rotating-fixture residual); training excludes such (task, theta) "
    "pairs anyway"
)


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def task_name_from_path(path: str) -> str:
    """Task name from an HDF5 demo path basename (``<task>_demo.hdf5``)."""
    base = os.path.basename(path)
    assert base.endswith("_demo.hdf5"), f"unrecognized demo file name: {path}"
    return base[: -len("_demo.hdf5")]


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def build_control_env(task_name: str, image_size: int = 128, seed: int = 42) -> ControlEnv:
    """Build a ControlEnv directly, mirroring oat/env/libero/env.py:45-59."""
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
    env.seed(seed)
    return env


def render_agentview(env: ControlEnv, state: np.ndarray) -> np.ndarray:
    """Re-render one flattened state; vertically flipped to match the zarr
    training images (dataset_conversion applies np.flip on the height axis)."""
    obs = env.regenerate_obs_from_state(np.asarray(state, dtype=np.float64))
    return np.flip(obs["agentview_image"], axis=0).astype(np.uint8)


def task_name_from_hdf5(h5file: h5py.File, path: str) -> str:
    bddl = str(h5file["data"].attrs.get("bddl_file_name", ""))
    name = os.path.basename(bddl)[:-len(".bddl")] if bddl else ""
    if name in task_name_to_suite_and_ids:
        return name
    base = os.path.basename(path)
    assert base.endswith("_demo.hdf5"), f"unrecognized demo file name: {path}"
    return base[: -len("_demo.hdf5")]


def load_rotate_rotation(probe_results_path: str) -> tuple:
    """rotate_rotation flag from probe 1's controller-frame verdict; warns
    and defaults to world-frame (True) when the probe file is missing."""
    if not os.path.exists(probe_results_path):
        print(f"[probe_label_roundtrip] WARNING: {probe_results_path} not found; "
              f"defaulting to controller_frame=world (rotate_rotation=True). "
              f"Run probe_controller_frame.py first.")
        return True, "world (default, probe file missing)"
    with open(probe_results_path) as fh:
        pr = json.load(fh)
    frame = pr.get("controller_frame_rot", pr.get("controller_frame"))
    if frame not in ("world", "ee"):
        print(f"[probe_label_roundtrip] WARNING: controller_frame_rot={frame!r} "
              f"in {probe_results_path}; defaulting to world (rotate_rotation=True)")
        return True, f"world (default, probe verdict {frame!r})"
    return frame == "world", frame


def sync_controller_initial_joints(env: ControlEnv) -> None:
    """Re-anchor the OSC nullspace to the just-injected arm posture.

    robosuite's OSC adds nullspace torques pulling the arm toward
    ``controller.initial_joint``, captured at reset
    (robosuite/controllers/base_controller.py:91) — i.e. the UNROTATED
    posture. After injecting a rotated state the controller would otherwise
    drag joint 1 back toward theta=0 for the whole rollout.
    ``update_initial_joints`` (base_controller.py:175) also refreshes the
    cached ee pos/ori via ``update(force=True)``. Attribute chain verified on
    the live classes: ``ControlEnv.env`` is the robosuite env
    (env_wrapper.py:56) and ``robots[0].controller`` is the arm's OSC;
    ``controller.qpos_index`` are exactly the sim.data.qpos addresses of the
    7 arm joints (``Robot._ref_joint_pos_indexes``), gripper excluded.
    """
    controller = env.env.robots[0].controller
    arm_qpos = np.asarray(
        env.sim.data.qpos[controller.qpos_index], dtype=np.float64)
    assert arm_qpos.shape == (controller.joint_dim,)
    controller.update_initial_joints(arm_qpos)


def replay(env: ControlEnv, state0: np.ndarray, actions: np.ndarray):
    """Reset, regenerate state0, step all actions; returns the (T, 3) EEF
    position trajectory and any-step success.

    After the state injection the OSC nullspace anchor is re-synced to the
    injected posture — applied to control (theta=0) and rotated replays alike
    so the two runs follow the same controller protocol."""
    env.reset()
    env.regenerate_obs_from_state(np.asarray(state0, dtype=np.float64))
    sync_controller_initial_joints(env)
    traj = np.empty((len(actions), 3), dtype=np.float64)
    success = False
    for t in range(len(actions)):
        obs, _, _, _ = env.step(np.asarray(actions[t], dtype=np.float64))
        traj[t] = obs["robot0_eef_pos"]
        success = success or bool(env.check_success())
    return traj, success


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5_dir",
                        default="third_party/LIBERO/libero/datasets/libero_10")
    parser.add_argument("--base_zarr", default="data/libero/libero10_N500.zarr")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--angles", default="10,-20,30",
                        help="comma-separated degrees; one is drawn per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/libero/probe_roundtrip_report.json")
    parser.add_argument("--probe_results", default="data/libero/probe_results.json")
    # pass criteria (configurable)
    parser.add_argument("--success_ratio", type=float, default=0.9,
                        help="rotated success >= ratio * control success")
    parser.add_argument("--median_tol", type=float, default=0.02,
                        help="median per-step EEF error tolerance, meters")
    parser.add_argument("--p90_factor", type=float, default=2.0,
                        help="rotated p90 error <= factor * control-divergence p90")
    parser.add_argument("--p90_floor", type=float, default=0.0,
                        help="floor (m) on the control-divergence p90 used in the "
                             "p90 criterion, guards a degenerate all-zero control")
    parser.add_argument("--calib_episodes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    hdf5_dir = _abspath(args.hdf5_dir)
    base_zarr = _abspath(args.base_zarr)
    angles = [float(a) for a in args.angles.split(",")]

    rotate_rotation, frame_str = load_rotate_rotation(_abspath(args.probe_results))
    print(f"[probe_label_roundtrip] controller_frame={frame_str} "
          f"-> rotate_rotation={rotate_rotation}")

    print(f"[probe_label_roundtrip] loading actions from {base_zarr}")
    replay_buffer = ReplayBuffer.copy_from_path(base_zarr, keys=["action"])
    matches = match_episodes(replay_buffer, hdf5_dir)
    episode_ends = replay_buffer.episode_ends[:]
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_lens = np.diff(np.concatenate([[0], episode_ends]))
    n_episodes = len(episode_ends)

    zroot = zarr.open(base_zarr, "r")
    agentview = zroot["data/agentview_rgb"]

    # sanity: the classification table covers exactly the libero10 tasks
    libero10_tasks = set(get_subtasks("libero10"))
    assert set(TASK_GOAL_ANCHORING) == libero10_tasks, (
        f"TASK_GOAL_ANCHORING out of sync with get_subtasks('libero10'): "
        f"missing={libero10_tasks - set(TASK_GOAL_ANCHORING)} "
        f"extra={set(TASK_GOAL_ANCHORING) - libero10_tasks}"
    )

    # stratified sampling by goal anchoring: >= ~2/3 of --n from tasks whose
    # goal predicates are anchored to co-rotating entities; the rest from
    # fixture-anchored tasks (informational stratum)
    ep_stratum = [TASK_GOAL_ANCHORING[task_name_from_path(matches[e][0])]
                  for e in range(n_episodes)]
    pool = {
        COROTATING: [e for e in range(n_episodes)
                     if ep_stratum[e] == COROTATING],
        FIXTURE_ANCHORED: [e for e in range(n_episodes)
                           if ep_stratum[e] == FIXTURE_ANCHORED],
    }
    budget = {COROTATING: int(np.ceil(args.n * 2 / 3))}
    budget[FIXTURE_ANCHORED] = args.n - budget[COROTATING]
    for s, other in ((COROTATING, FIXTURE_ANCHORED),
                     (FIXTURE_ANCHORED, COROTATING)):
        if not pool[s] and budget[s]:
            print(f"[probe_label_roundtrip] WARNING: no {s} episodes in the "
                  f"dataset; moving their budget ({budget[s]}) to {other}")
            budget[other] += budget[s]
            budget[s] = 0

    rng = np.random.default_rng(args.seed)
    episodes = []
    for s in (COROTATING, FIXTURE_ANCHORED):
        if budget[s]:
            episodes.extend(rng.choice(
                pool[s], size=budget[s],
                replace=budget[s] > len(pool[s])).tolist())
    print(f"[probe_label_roundtrip] stratified sample: "
          f"{budget[COROTATING]} corotating + "
          f"{budget[FIXTURE_ANCHORED]} fixture_anchored episodes")
    pairs = [(int(e), angles[int(rng.integers(0, len(angles)))])
             for e in episodes]

    by_path = {}
    for i, (e, _) in enumerate(pairs):
        by_path.setdefault(matches[e][0], []).append(i)

    per_episode = []
    all_err, all_div = [], []
    succ_control, succ_rotated = [], []
    strat_acc = {s: {"err": [], "div": [], "succ_c": [], "succ_r": []}
                 for s in (COROTATING, FIXTURE_ANCHORED)}
    per_task_delta = {}
    for path in sorted(by_path):
        pair_idxs = by_path[path]
        eps = sorted({pairs[i][0] for i in pair_idxs})
        with h5py.File(path, "r") as f:
            task_name = task_name_from_hdf5(f, path)
            stratum = TASK_GOAL_ANCHORING[task_name]
            print(f"[probe_label_roundtrip] task {task_name} [{stratum}]: "
                  f"{len(pair_idxs)} episodes")
            env = build_control_env(task_name, image_size=args.image_size)
            env.reset()

            # defensive fallback (plan: not expected for LIBERO): rebuild
            # from the demo's model XML on a state-length mismatch
            states0 = f["data"][matches[eps[0]][1]]["states"][:]
            sim_len = env.sim.get_state().flatten().shape[0]
            if states0.shape[1] != sim_len:
                print(f"[probe_label_roundtrip] WARNING: demo state dim "
                      f"{states0.shape[1]} != sim state dim {sim_len}; "
                      f"falling back to reset_from_xml_string")
                env.reset_from_xml_string(
                    f["data"][matches[eps[0]][1]].attrs["model_file"])
                assert f["data"][matches[eps[0]][1]]["states"].shape[1] \
                    == env.sim.get_state().flatten().shape[0]
            addr = resolve_addresses(env)
            p_base_xy = np.asarray(addr.p_base, dtype=np.float64)[:2]

            # per-task delta calibration
            render_fn = lambda s: render_agentview(env, s)  # noqa: E731
            deltas = []
            for e in eps[: args.calib_episodes]:
                states = f["data"][matches[e][1]]["states"][:]
                imgs = agentview[ep_starts[e]: ep_starts[e] + ep_lens[e]]
                deltas.append(int(calibrate_state_offset(
                    env, states, imgs, render_fn)))
            delta = deltas[0]
            if len(set(deltas)) != 1:
                print(f"[probe_label_roundtrip] WARNING: inconsistent delta "
                      f"{deltas} for task {task_name}; using {delta}")
            per_task_delta[task_name] = delta

            for i in pair_idxs:
                e, angle_deg = pairs[i]
                theta = float(np.deg2rad(angle_deg))
                demo = f["data"][matches[e][1]]
                states = demo["states"][:]
                actions = demo["actions"][:].astype(np.float64)
                demo_ee = demo["obs"]["ee_pos"][:].astype(np.float64)
                state0 = states[min(delta, len(states) - 1)]

                # control replay at theta=0 + divergence vs the demo's own obs
                p_ref, ok_c = replay(env, state0, actions)
                div = np.linalg.norm(p_ref - demo_ee, axis=1)

                # rotated replay with transformed actions
                acts_rot = rotate_action_chunk(actions, theta, rotate_rotation)
                p_rot, ok_r = replay(
                    env, rewrite_state(state0, theta, addr), acts_rot)
                p_expected = rotate_xy(p_ref, theta, center_xy=p_base_xy)
                err = np.linalg.norm(p_rot - p_expected, axis=1)

                all_err.append(err)
                all_div.append(div)
                succ_control.append(ok_c)
                succ_rotated.append(ok_r)
                acc = strat_acc[stratum]
                acc["err"].append(err)
                acc["div"].append(div)
                acc["succ_c"].append(ok_c)
                acc["succ_r"].append(ok_r)
                per_episode.append({
                    "episode": e,
                    "task": task_name,
                    "stratum": stratum,
                    "demo_key": matches[e][1],
                    "angle_deg": angle_deg,
                    "n_steps": len(actions),
                    "delta": delta,
                    "success_control": ok_c,
                    "success_rotated": ok_r,
                    "err_median_m": float(np.median(err)),
                    "err_p90_m": float(np.percentile(err, 90)),
                    "err_max_m": float(np.max(err)),
                    "control_div_median_m": float(np.median(div)),
                    "control_div_p90_m": float(np.percentile(div, 90)),
                })
                print(f"[probe_label_roundtrip]   ep {e:3d} theta={angle_deg:+6.1f} deg "
                      f"ctrl_succ={ok_c} rot_succ={ok_r} "
                      f"err_med={np.median(err):.4f} m div_med={np.median(div):.4f} m")
            env.close()

    all_err = np.concatenate(all_err)
    all_div = np.concatenate(all_div)
    success_rate_control = float(np.mean(succ_control))
    success_rate_rotated = float(np.mean(succ_rotated))
    err_median = float(np.median(all_err))
    err_p90 = float(np.percentile(all_err, 90))
    div_median = float(np.median(all_div))
    div_p90 = float(np.percentile(all_div, 90))

    def stratum_block(s):
        acc = strat_acc[s]
        eps = [rec for rec in per_episode if rec["stratum"] == s]
        if not acc["succ_c"]:
            return {"n_episodes": 0, "per_episode": []}
        e = np.concatenate(acc["err"])
        d = np.concatenate(acc["div"])
        return {
            "n_episodes": len(acc["succ_c"]),
            "tasks": sorted({rec["task"] for rec in eps}),
            "success_rate_control": float(np.mean(acc["succ_c"])),
            "success_rate_rotated": float(np.mean(acc["succ_r"])),
            "err_median_m": float(np.median(e)),
            "err_p90_m": float(np.percentile(e, 90)),
            "err_max_m": float(np.max(e)),
            "control_div_median_m": float(np.median(d)),
            "control_div_p90_m": float(np.percentile(d, 90)),
            "per_episode": eps,
        }

    strata = {s: stratum_block(s) for s in (COROTATING, FIXTURE_ANCHORED)}
    strata[FIXTURE_ANCHORED]["note"] = FIXTURE_STRATUM_NOTE

    # PASS criteria apply to the corotating stratum ONLY: fixture-anchored
    # goals are structurally unattainable under the rotation (see note)
    coro = strata[COROTATING]
    assert coro["n_episodes"] > 0, (
        "no corotating episodes sampled -- PASS criteria are undefined"
    )
    p90_limit = args.p90_factor * max(coro["control_div_p90_m"], args.p90_floor)
    crit_success = (coro["success_rate_rotated"]
                    >= args.success_ratio * coro["success_rate_control"])
    crit_median = coro["err_median_m"] < args.median_tol
    crit_p90 = coro["err_p90_m"] <= p90_limit
    ok = crit_success and crit_median and crit_p90

    result = {
        "probe": "label_roundtrip",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "hdf5_dir": hdf5_dir,
        "base_zarr": base_zarr,
        "n": args.n,
        "seed": args.seed,
        "angles_deg": angles,
        "controller_frame": frame_str,
        "rotate_rotation": rotate_rotation,
        "nullspace_resync": True,  # update_initial_joints after every injection
        # top-level rates/percentiles aggregate ALL episodes (backward
        # compat); PASS + criteria come from the corotating stratum only
        "success_rate_control": success_rate_control,
        "success_rate_rotated": success_rate_rotated,
        "err_median_m": err_median,
        "err_p90_m": err_p90,
        "control_div_median_m": div_median,
        "control_div_p90_m": div_p90,
        "task_goal_anchoring": TASK_GOAL_ANCHORING,
        "strata": strata,
        "criteria": {
            "stratum": COROTATING,
            "success_ratio": args.success_ratio,
            "median_tol_m": args.median_tol,
            "p90_factor": args.p90_factor,
            "p90_floor_m": args.p90_floor,
            "p90_limit_m": p90_limit,
            "crit_success": crit_success,
            "crit_median": crit_median,
            "crit_p90": crit_p90,
        },
        "per_task_delta": per_task_delta,
        "per_episode": per_episode,
        "pass": ok,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"[probe_label_roundtrip] wrote {out_path}")

    print(f"[probe_label_roundtrip] COROTATING stratum "
          f"(n={coro['n_episodes']}, PASS criteria apply here):")
    print(f"[probe_label_roundtrip]   success_control="
          f"{coro['success_rate_control']:.3f} "
          f"success_rotated={coro['success_rate_rotated']:.3f} "
          f"(need >= {args.success_ratio * coro['success_rate_control']:.3f}) "
          f"-> {'ok' if crit_success else 'FAIL'}")
    print(f"[probe_label_roundtrip]   err_median={coro['err_median_m']:.4f} m "
          f"(tol {args.median_tol}) -> {'ok' if crit_median else 'FAIL'}")
    print(f"[probe_label_roundtrip]   err_p90={coro['err_p90_m']:.4f} m "
          f"(limit {p90_limit:.4f} = {args.p90_factor} x control p90 "
          f"{coro['control_div_p90_m']:.4f}) -> {'ok' if crit_p90 else 'FAIL'}")
    fix = strata[FIXTURE_ANCHORED]
    if fix["n_episodes"]:
        print(f"[probe_label_roundtrip] FIXTURE_ANCHORED stratum "
              f"(n={fix['n_episodes']}, informational only):")
        print(f"[probe_label_roundtrip]   success_control="
              f"{fix['success_rate_control']:.3f} "
              f"success_rotated={fix['success_rate_rotated']:.3f} "
              f"err_median={fix['err_median_m']:.4f} m "
              f"err_p90={fix['err_p90_m']:.4f} m "
              f"(control p90 {fix['control_div_p90_m']:.4f} m)")
        print(f"[probe_label_roundtrip]   note: {FIXTURE_STRATUM_NOTE}")
    verdict = "PASS" if ok else "FAIL"
    print(f"[probe_label_roundtrip] {verdict} (corotating stratum)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

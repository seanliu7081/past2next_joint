"""Probe 2: render consistency of the SE(2) state rewrite.

Samples ~``--n`` (episode, frame, angle) triples across the LIBERO-10 tasks
(episode -> HDF5 demo via content matching). For each triple, the demo state
is re-rendered at theta=0 and after ``rewrite_state(theta)``; the rewritten
EEF pose must equal the R_z(theta)-mapped original within ``--pos_tol`` m
and ``--quat_tol`` rad (sign-insensitive geodesic).

Also reports, per task: the obs/state offset delta (calibrated over up to
``--calib_episodes`` sampled episodes, asserted consistent) and theta=0
pixel-diff stats of re-renders vs the stored zarr agentview images
(flip-corrected).

PASS iff 100% of the triples are within tolerance.

Usage:
    MUJOCO_GL=egl python scripts/probes/probe_render_consistency.py \
        --hdf5_dir third_party/LIBERO/libero/datasets/libero_10 --n 200
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
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.equi.se2_transforms import (
    quat_geodesic_angle_wxyz,
    quat_mul_xyzw,
    quat_z_xyzw,
    rotate_xy,
)

WXYZ = [3, 0, 1, 2]  # xyzw -> wxyz reordering


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


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


def prepare_env_for_demos(env: ControlEnv, states: np.ndarray, model_file: str) -> bool:
    """Defensive fallback (plan: not expected for LIBERO): if the demo state
    length mismatches the live sim, rebuild from the demo's model XML.
    Returns True iff the fallback was used."""
    # demo states are flattened [time(1), qpos, qvel] just like the live sim
    sim_len = env.sim.get_state().flatten().shape[0]
    if states.shape[1] == sim_len:
        return False
    print(f"[probe_render_consistency] WARNING: demo state dim {states.shape[1]} "
          f"!= sim state dim {sim_len}; falling back to reset_from_xml_string")
    env.reset_from_xml_string(model_file)
    sim_len = env.sim.get_state().flatten().shape[0]
    assert states.shape[1] == sim_len, (
        f"state dim still mismatched after XML fallback: {states.shape[1]} vs {sim_len}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5_dir",
                        default="third_party/LIBERO/libero/datasets/libero_10")
    parser.add_argument("--base_zarr", default="data/libero/libero10_N500.zarr")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/libero/probe_render_consistency.json")
    parser.add_argument("--angles_deg", default="10,-10,20,-20,30,-30")
    parser.add_argument("--pos_tol", type=float, default=1e-4, help="meters")
    parser.add_argument("--quat_tol", type=float, default=1e-3, help="radians")
    parser.add_argument("--calib_episodes", type=int, default=3)
    parser.add_argument("--calib_frames", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    hdf5_dir = _abspath(args.hdf5_dir)
    base_zarr = _abspath(args.base_zarr)
    angles = [float(a) for a in args.angles_deg.split(",")]

    print(f"[probe_render_consistency] loading actions from {base_zarr}")
    replay_buffer = ReplayBuffer.copy_from_path(base_zarr, keys=["action"])
    matches = match_episodes(replay_buffer, hdf5_dir)
    episode_ends = replay_buffer.episode_ends[:]
    ep_starts = np.concatenate([[0], episode_ends[:-1]])
    ep_lens = np.diff(np.concatenate([[0], episode_ends]))
    n_episodes = len(episode_ends)

    # zarr images stay on disk; read lazily per frame / episode
    zroot = zarr.open(base_zarr, "r")
    agentview = zroot["data/agentview_rgb"]

    rng = np.random.default_rng(args.seed)
    triples = []
    for _ in range(args.n):
        e = int(rng.integers(0, n_episodes))
        # keep t <= T-2 so states[t + delta] exists for delta in {0, 1}
        t = int(rng.integers(0, ep_lens[e] - 1))
        a = angles[int(rng.integers(0, len(angles)))]
        triples.append((e, t, a))

    by_path = {}
    for i, (e, _, _) in enumerate(triples):
        by_path.setdefault(matches[e][0], []).append(i)

    per_task = {}
    records = []
    for path in sorted(by_path):
        triple_idxs = by_path[path]
        eps = sorted({triples[i][0] for i in triple_idxs})
        with h5py.File(path, "r") as f:
            task_name = task_name_from_hdf5(f, path)
            print(f"[probe_render_consistency] task {task_name}: "
                  f"{len(triple_idxs)} triples, {len(eps)} episodes")
            env = build_control_env(task_name, image_size=args.image_size)
            env.reset()

            states_cache = {
                e: f["data"][matches[e][1]]["states"][:] for e in eps}
            for e in eps:
                assert len(states_cache[e]) == ep_lens[e], (
                    f"episode {e}: zarr length {ep_lens[e]} != demo states "
                    f"length {len(states_cache[e])} ({path} {matches[e][1]})")
            xml_fallback = prepare_env_for_demos(
                env, states_cache[eps[0]],
                f["data"][matches[eps[0]][1]].attrs["model_file"])
            addr = resolve_addresses(env)
            p_base_xy = np.asarray(addr.p_base, dtype=np.float64)[:2]

            # per-task delta calibration over up to --calib_episodes episodes
            render_fn = lambda s: render_agentview(env, s)  # noqa: E731
            deltas = []
            for e in eps[: args.calib_episodes]:
                imgs = agentview[ep_starts[e]: ep_starts[e] + ep_lens[e]]
                deltas.append(int(calibrate_state_offset(
                    env, states_cache[e], imgs, render_fn)))
            delta = deltas[0]
            delta_consistent = len(set(deltas)) == 1
            if not delta_consistent:
                print(f"[probe_render_consistency] WARNING: inconsistent delta "
                      f"{deltas} for task {task_name}; using {delta}")

            # theta=0 pixel-diff stats vs stored zarr images (calib episode)
            e0 = eps[0]
            states0 = states_cache[e0]
            mads = []
            calib_ts = np.unique(np.linspace(
                0, ep_lens[e0] - 1, num=min(args.calib_frames, ep_lens[e0]),
                dtype=int))
            for t in calib_ts:
                s_idx = min(t + delta, len(states0) - 1)
                img_r = render_agentview(env, states0[s_idx]).astype(np.int16)
                img_z = np.asarray(agentview[ep_starts[e0] + t]).astype(np.int16)
                mads.append(float(np.abs(img_r - img_z).mean()))
            per_task[task_name] = {
                "hdf5_path": path,
                "delta": delta,
                "deltas": deltas,
                "delta_consistent": delta_consistent,
                "xml_fallback": xml_fallback,
                "theta0_pixel_mad_mean": float(np.mean(mads)),
                "theta0_pixel_mad_max": float(np.max(mads)),
                "n_triples": len(triple_idxs),
            }

            # core pose-consistency check
            for i in triple_idxs:
                e, t, angle_deg = triples[i]
                states = states_cache[e]
                state = states[min(t + delta, len(states) - 1)]
                theta = float(np.deg2rad(angle_deg))

                obs0 = env.regenerate_obs_from_state(
                    np.asarray(state, dtype=np.float64))
                p0 = np.asarray(obs0["robot0_eef_pos"], dtype=np.float64)
                q0 = np.asarray(obs0["robot0_eef_quat"], dtype=np.float64)  # xyzw

                obs1 = env.regenerate_obs_from_state(
                    rewrite_state(state, theta, addr))
                p1 = np.asarray(obs1["robot0_eef_pos"], dtype=np.float64)
                q1 = np.asarray(obs1["robot0_eef_quat"], dtype=np.float64)

                p_expected = rotate_xy(p0, theta, center_xy=p_base_xy)
                pos_err = float(np.max(np.abs(p1 - p_expected)))
                q_expected = quat_mul_xyzw(quat_z_xyzw(theta), q0)
                quat_err = float(quat_geodesic_angle_wxyz(
                    q1[WXYZ], q_expected[WXYZ]))
                ok = pos_err < args.pos_tol and quat_err < args.quat_tol
                records.append({
                    "episode": e,
                    "task": task_name,
                    "frame": t,
                    "angle_deg": angle_deg,
                    "pos_err_m": pos_err,
                    "quat_err_rad": quat_err,
                    "ok": ok,
                })
            env.close()

    n_pass = sum(r["ok"] for r in records)
    pass_fraction = n_pass / len(records)
    max_pos_err = max(r["pos_err_m"] for r in records)
    max_quat_err = max(r["quat_err_rad"] for r in records)
    failures = [r for r in records if not r["ok"]]
    ok = pass_fraction == 1.0

    result = {
        "probe": "render_consistency",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "hdf5_dir": hdf5_dir,
        "base_zarr": base_zarr,
        "n": args.n,
        "seed": args.seed,
        "pos_tol_m": args.pos_tol,
        "quat_tol_rad": args.quat_tol,
        "angles_deg": angles,
        "pass_fraction": pass_fraction,
        "n_pass": n_pass,
        "n_fail": len(failures),
        "max_pos_err_m": max_pos_err,
        "max_quat_err_rad": max_quat_err,
        "per_task": per_task,
        "failures": failures,
        "triples": records,
        "pass": ok,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"[probe_render_consistency] wrote {out_path}")

    verdict = "PASS" if ok else "FAIL"
    print(f"[probe_render_consistency] {verdict}: {n_pass}/{len(records)} triples ok "
          f"(max_pos_err={max_pos_err:.2e} m, max_quat_err={max_quat_err:.2e} rad)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

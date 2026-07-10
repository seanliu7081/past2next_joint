"""M2 GATING probe 1: does OSC_POSE interpret action deltas in the world
frame or in the end-effector frame?

Protocol, per theta in ``--thetas`` (default 0, +-30, +-60 deg — covering the
+-30 deg augmentation range with margin): reset, rigidly rotate the
scene + arm by theta via ``rewrite_state`` (objects about the base, joint 1
+= theta), ``regenerate_obs_from_state``, then command a pure +x position
delta (resp. a pure +rx axis-angle delta) for ``--n_steps`` steps.

  * world-frame controller: the EEF displacement (resp. the world axis of
    the measured delta rotation) stays along world +x for every theta.
  * ee-frame controller: it tracks R_z(theta) @ [1, 0] instead.

Thetas in ``--extra_thetas`` (default 90 deg) are measured and reported in
the json under ``extra_theta_metrics`` but EXCLUDED from the verdicts: at
extreme rotations far outside the augmentation range the +x position push
can run into the workspace boundary, deflecting the displacement (a boundary
artifact, not controller-frame evidence — e.g. at theta=90 dot_ee is
decisively negative, rejecting the ee hypothesis, yet dot_world dips below
threshold too).

Writes ``probe_results.json`` (key ``controller_frame``) consumed — gating,
decision D7 — by the SE(2) aug dataset and by probe 3.

Usage:
    MUJOCO_GL=egl python scripts/probes/probe_controller_frame.py \
        --out data/libero/probe_results.json
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

import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.factory import get_subtasks, is_multitask
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.equi.se2_transforms import rot2d

DOT_THRESHOLD = 0.95
MIN_DISP_XY = 5e-3    # m; smaller xy displacement => degenerate position run
MIN_ROT_ANGLE = 2e-2  # rad; smaller delta rotation => degenerate rotation run


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def _parse_thetas(spec: str):
    """Comma-separated degrees -> tuple of floats; empty string -> ()."""
    return tuple(float(tok) for tok in spec.split(",") if tok.strip())


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def build_control_env(task_name: str, image_size: int = 128, seed: int = 42) -> ControlEnv:
    """Build a ControlEnv directly, mirroring oat/env/libero/env.py:45-59.
    LiberoEnv is deliberately not used: it hides the state setters and its
    reset runs _let_objects_fall."""
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


def run_push(env: ControlEnv, theta: float, action: np.ndarray,
             n_steps: int, settle_steps: int):
    """Reset, rotate the flattened sim state by theta, re-render, then step
    ``action`` n_steps times. Returns (p0, q0, p1, q1); quats are xyzw."""
    env.reset()
    addr = resolve_addresses(env)
    state = env.sim.get_state().flatten()
    obs = env.regenerate_obs_from_state(rewrite_state(state, theta, addr))
    settle = np.array([0.0] * 6 + [-1.0])
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(settle)
    p0 = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    q0 = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    for _ in range(n_steps):
        obs, _, _, _ = env.step(action)
    p1 = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    q1 = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    return p0, q0, p1, q1


def classify(dots_world, dots_ee, degenerate) -> str:
    if any(degenerate):
        return "inconclusive"
    if all(d > DOT_THRESHOLD for d in dots_world):
        return "world"
    if all(d > DOT_THRESHOLD for d in dots_ee):
        return "ee"
    return "inconclusive"


def measure_theta(env: ControlEnv, theta_deg: float, mag: float,
                  n_steps: int, settle_steps: int) -> dict:
    """Run the position and rotation pushes at one theta; return the metrics
    record (same schema as the ``per_theta`` json entries)."""
    ex = np.array([1.0, 0.0])
    ex3 = np.array([1.0, 0.0, 0.0])
    theta = float(np.deg2rad(theta_deg))
    e_ee = rot2d(theta) @ ex  # ee-frame hypothesis: +x tracks R(theta)@[1,0]

    # position protocol: pure +x position delta
    action = np.array([mag, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    p0, _, p1, _ = run_push(env, theta, action, n_steps, settle_steps)
    disp = p1 - p0
    disp_norm_xy = float(np.linalg.norm(disp[:2]))
    p_degen = disp_norm_xy < MIN_DISP_XY
    u = disp[:2] / disp_norm_xy if not p_degen else np.zeros(2)
    p_dot_world = float(u @ ex)
    p_dot_ee = float(u @ e_ee)

    # rotation protocol: pure +rx axis-angle delta; measure the world
    # axis of R_delta = R_end @ R_start^-1 (obs quats are xyzw = scipy order)
    action = np.array([0.0, 0.0, 0.0, mag, 0.0, 0.0, -1.0])
    _, q0, _, q1 = run_push(env, theta, action, n_steps, settle_steps)
    r_delta = Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()
    rotvec = r_delta.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    r_degen = angle < MIN_ROT_ANGLE
    axis = rotvec / angle if not r_degen else np.zeros(3)
    r_dot_world = float(axis @ ex3)
    r_dot_ee = float(axis @ np.array([e_ee[0], e_ee[1], 0.0]))

    return {
        "theta_deg": theta_deg,
        "pos": {
            "disp": disp.tolist(),
            "disp_norm_xy": disp_norm_xy,
            "dot_world": p_dot_world,
            "dot_ee": p_dot_ee,
            "degenerate": p_degen,
        },
        "rot": {
            "axis": axis.tolist(),
            "angle_rad": angle,
            "dot_world": r_dot_world,
            "dot_ee": r_dot_ee,
            "degenerate": r_degen,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_name", default=None,
                        help="LIBERO task name (default: first libero10 subtask); "
                             "a suite name like 'libero10' also resolves to its first subtask")
    parser.add_argument("--out", default="data/libero/probe_results.json")
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--mag", type=float, default=0.3)
    parser.add_argument("--settle_steps", type=int, default=0,
                        help="zero-action settle steps after re-render, before measuring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thetas", default="0,30,-30,60,-60",
                        help="comma-separated degrees over which the pos/rot "
                             "VERDICTS are computed; default covers the +-30 deg "
                             "augmentation range with margin")
    parser.add_argument("--extra_thetas", default="90",
                        help="comma-separated degrees measured and reported as "
                             "extra_theta_metrics but EXCLUDED from verdicts: at "
                             "extremes outside the augmentation range the +x push "
                             "can hit the workspace boundary, deflecting the "
                             "displacement (boundary artifact, not controller-"
                             "frame evidence)")
    args = parser.parse_args()

    verdict_thetas = _parse_thetas(args.thetas)
    extra_thetas = _parse_thetas(args.extra_thetas)
    if not verdict_thetas:
        parser.error("--thetas must contain at least one angle")

    task_name = args.task_name or get_subtasks("libero10")[0]
    if is_multitask(task_name):
        task_name = get_subtasks(task_name)[0]

    env = build_control_env(task_name, seed=args.seed)

    per_theta = []
    for theta_deg in verdict_thetas:
        rec = measure_theta(env, theta_deg, args.mag, args.n_steps,
                            args.settle_steps)
        per_theta.append(rec)
        print(f"[probe_controller_frame] theta={theta_deg:+7.1f} deg  "
              f"pos dot_world={rec['pos']['dot_world']:+.3f} "
              f"dot_ee={rec['pos']['dot_ee']:+.3f}  "
              f"rot dot_world={rec['rot']['dot_world']:+.3f} "
              f"dot_ee={rec['rot']['dot_ee']:+.3f}")

    # measured for the record only, never folded into the verdicts (see
    # module docstring: workspace-boundary artifacts outside the aug range)
    extra_theta_metrics = []
    for theta_deg in extra_thetas:
        rec = measure_theta(env, theta_deg, args.mag, args.n_steps,
                            args.settle_steps)
        extra_theta_metrics.append(rec)
        print(f"[probe_controller_frame] theta={theta_deg:+7.1f} deg  "
              f"pos dot_world={rec['pos']['dot_world']:+.3f} "
              f"dot_ee={rec['pos']['dot_ee']:+.3f}  "
              f"rot dot_world={rec['rot']['dot_world']:+.3f} "
              f"dot_ee={rec['rot']['dot_ee']:+.3f}  "
              f"[extra: excluded from verdict]")
    env.close()

    pos_verdict = classify([r["pos"]["dot_world"] for r in per_theta],
                           [r["pos"]["dot_ee"] for r in per_theta],
                           [r["pos"]["degenerate"] for r in per_theta])
    rot_verdict = classify([r["rot"]["dot_world"] for r in per_theta],
                           [r["rot"]["dot_ee"] for r in per_theta],
                           [r["rot"]["degenerate"] for r in per_theta])
    ok = pos_verdict == rot_verdict and pos_verdict != "inconclusive"

    result = {
        "probe": "controller_frame",
        "task_name": task_name,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_steps": args.n_steps,
        "mag": args.mag,
        "settle_steps": args.settle_steps,
        "seed": args.seed,
        "dot_threshold": DOT_THRESHOLD,
        "thetas_deg": list(verdict_thetas),
        "extra_thetas_deg": list(extra_thetas),
        "per_theta": per_theta,
        # measured at --extra_thetas for the record; EXCLUDED from the
        # verdicts above (workspace-boundary artifacts outside the +-30 deg
        # augmentation range, e.g. the +x push deflecting at theta=90)
        "extra_theta_metrics": extra_theta_metrics,
        "controller_frame_pos": pos_verdict,
        "controller_frame_rot": rot_verdict,
        # convenience key: what downstream consumers (dataset gating, probe 3)
        # actually read; rotation labels are the frame-sensitive part
        "controller_frame": rot_verdict,
        "pass": ok,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"[probe_controller_frame] wrote {out_path}")

    verdict = "PASS" if ok else "FAIL"
    print(f"[probe_controller_frame] {verdict}: controller_frame_pos={pos_verdict} "
          f"controller_frame_rot={rot_verdict}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

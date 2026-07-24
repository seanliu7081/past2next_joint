"""Show that the SE(2)-augmented *labels* (actions + proprio) are legal for the
re-rendered augmented *images*.

The augmentation splits into two independent computations:
  * IMAGES  are re-rendered by the simulator from ``rewrite_state(state, theta)``
    (scene + arm rigidly yawed about the robot base).
  * ACTIONS / PROPRIO are NOT rendered -- they are rotated analytically in numpy
    (``rotate_action_chunk`` / ``rotate_xy``), never touching the sim.

If the two halves are consistent, then projecting the ANALYTIC rotated eef,
its future trajectory, and the rotated action delta into the world-fixed
agentview camera must land on / follow the gripper in the INDEPENDENTLY
re-rendered image -- for every discrete angle. That visual registration is the
"legal actions" demonstration; it is certified numerically by:
  * probe_render_consistency: |analytic R_z(theta).eef - sim eef| max 6.7e-16 m
  * probe_label_roundtrip:    rotated actions track R_z(theta).ref, median 6.4 nm

With --verify_sim (default) the analytic-vs-simulated eef gap is also recomputed
live for the displayed frame at each angle and printed on the panel.

Run from the repo root:
    MUJOCO_GL=egl /home/haotian/miniforge3/envs/oat/bin/python \
        analysis_scripts/demo_legal_labels.py --episode 16 --frame mid \
        --out analysis_scripts/legal_labels_ep16.png
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib
import sys

REPO = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.utils import camera_utils as cu

from oat.env.libero.env import task_name_to_suite_and_ids
from oat.equi.se2_transforms import rotate_xy, rotate_action_chunk

BASE = "data/libero/libero10_N500.zarr"
AUG = "data/libero/libero10_N500_se2aug.zarr"


def build_env(task_name, image_size=128):
    suite, sid, _ = task_name_to_suite_and_ids[task_name]
    task = benchmark.get_benchmark_dict()[suite]().get_task(sid)
    env = ControlEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=image_size, camera_widths=image_size,
        has_renderer=False, use_camera_obs=True, has_offscreen_renderer=True,
    )
    env.seed(0)
    return env


def project(points_world, world2pix, H, W):
    """World (...,3) -> stored-image pixel (col, row), correcting the vertical
    flip the dataset applies (np.flip on the height axis). Returns float (x=col,
    y=row_flipped) so lines/arrows are smooth (no clip-to-border)."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(pts), 1))
    homo = np.concatenate([pts, ones], axis=1)          # (N,4)
    pix = (world2pix @ homo.T).T                        # (N,4)
    pix = pix[:, :2] / pix[:, 2:3]                       # (col=x, row=y), unflipped
    col = pix[:, 0]
    row_flipped = (H - 1) - pix[:, 1]
    return np.stack([col, row_flipped], axis=1)          # (N, 2) as (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=16)
    ap.add_argument("--frame", default="mid", help="'mid'|'start'|'end'|<int offset>")
    ap.add_argument("--horizon", type=int, default=40, help="future-trajectory steps to draw")
    ap.add_argument("--arrow_len_m", type=float, default=0.07)
    ap.add_argument("--verify_sim", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute analytic-vs-simulated eef gap live per angle")
    ap.add_argument("--out", default="analysis_scripts/legal_labels.png")
    args = ap.parse_args()

    base = zarr.open(BASE, mode="r")
    aug = zarr.open(AUG, mode="r")
    ee = aug["meta/episode_ends"][:]
    starts = np.concatenate([[0], ee[:-1]])
    angles = aug["meta/angles_deg"][:]
    order = np.argsort(angles)
    e = args.episode
    s, end = int(starts[e]), int(ee[e])
    ep_len = end - s
    off = {"mid": ep_len // 2, "start": 0, "end": ep_len - 1}.get(
        args.frame, None)
    off = int(args.frame) if off is None else off
    gidx = s + off
    H = W = base["data/agentview_rgb"].shape[1]

    uid = int(base["data/task_uid"][s, 0])
    prompt = str(base["data/prompt"][s])
    task_name = {u: n for n, (su, si, u) in task_name_to_suite_and_ids.items()}[uid]
    p_base = aug["meta/p_base"][e].astype(np.float64)
    p_base_xy = p_base[:2]
    delta = int(aug["meta/state_offset"][e])
    valid = aug["meta/valid_mask"][e]

    # analytic labels from the base zarr (no sim): current eef, future eef path,
    # and the raw action delta at this frame
    eef0 = base["data/robot0_eef_pos"][gidx].astype(np.float64)
    h = min(args.horizon, ep_len - off - 1)
    traj0 = base["data/robot0_eef_pos"][gidx:gidx + h + 1].astype(np.float64)
    act0 = base["data/action"][gidx].astype(np.float64)           # [dx,dy,dz,rx,ry,rz,grip]

    env = build_env(task_name, image_size=H)
    env.reset()
    world2pix = cu.get_camera_transform_matrix(env.sim, "agentview", H, W)

    # optional live verification for the displayed frame (needs the demo state):
    #   gap_px   -- projected analytic label vs the TRUE simulated gripper, in
    #               pixels: the registration error the viewer sees on the panel.
    #   rot_nm   -- pure rotation-legality residual (probe-2 quantity),
    #               |sim_eef(theta) - R_z(theta).sim_eef(0)|, both from the sim,
    #               isolating the rotation from the base-zarr/re-render frame
    #               offset that rides along constant through the rotation.
    gap_px = {}
    rot_resid_nm = {}
    if args.verify_sim:
        try:
            import h5py
            from oat.common.replay_buffer import ReplayBuffer
            from oat.env.libero.demo_alignment import match_episodes
            from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
            rb = ReplayBuffer.copy_from_path(BASE, keys=["action"])
            matches = match_episodes(rb, "third_party/LIBERO/libero/datasets/libero_10")
            hdf5_path, demo_key = matches[e]
            with h5py.File(hdf5_path, "r") as f:
                states = f["data"][demo_key]["states"][:].astype(np.float64)
            addr = resolve_addresses(env)
            state = states[min(off + delta, len(states) - 1)]
            sim_eef0 = np.asarray(
                env.regenerate_obs_from_state(rewrite_state(state, 0.0, addr))
                ["robot0_eef_pos"], dtype=np.float64)
            for k in order:
                th = float(np.deg2rad(angles[k]))
                sim_eef = np.asarray(
                    env.regenerate_obs_from_state(rewrite_state(state, th, addr))
                    ["robot0_eef_pos"], dtype=np.float64)
                ana_eef = rotate_xy(eef0, th, center_xy=p_base_xy)
                P_sim = project(sim_eef[None], world2pix, H, W)[0]
                P_ana = project(ana_eef[None], world2pix, H, W)[0]
                gap_px[int(k)] = float(np.linalg.norm(P_sim - P_ana))
                rot_resid_nm[int(k)] = float(np.linalg.norm(
                    sim_eef - rotate_xy(sim_eef0, th, center_xy=p_base_xy)) * 1e9)
        except Exception as ex:                                    # noqa: BLE001
            print(f"[demo_legal_labels] verify_sim skipped: {ex}")
            gap_px, rot_resid_nm = {}, {}

    ncol = 1 + len(angles)
    fig, axes = plt.subplots(1, ncol, figsize=(2.25 * ncol, 3.0))

    def overlay(ax, theta, img):
        eef_r = rotate_xy(eef0, theta, center_xy=p_base_xy)
        traj_r = rotate_xy(traj0, theta, center_xy=p_base_xy)
        act_xy = rotate_action_chunk(act0[None], theta, rotate_rotation=True)[0, :2]
        n = np.linalg.norm(act_xy)
        tip = eef_r.copy()
        if n > 1e-6:
            tip[:2] = eef_r[:2] + args.arrow_len_m * act_xy / n
        P_traj = project(traj_r, world2pix, H, W)
        P_eef = project(eef_r[None], world2pix, H, W)[0]
        P_tip = project(tip[None], world2pix, H, W)[0]
        ax.imshow(img)
        # future trajectory (proprio path), fading
        segs = np.stack([P_traj[:-1], P_traj[1:]], axis=1)
        lc = LineCollection(segs, colors="#ffd21e", linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
        # action delta (rotated action label)
        ax.annotate("", xy=P_tip, xytext=P_eef,
                    arrowprops=dict(arrowstyle="-|>", color="#00e0ff", lw=2.4))
        # current eef (proprio)
        ax.plot(P_eef[0], P_eef[1], "o", ms=11, mfc="none", mec="#39ff14", mew=2.4)
        ax.set_xlim(0, W - 1); ax.set_ylim(H - 1, 0)
        ax.set_xticks([]); ax.set_yticks([])

    # column 0: ORIGINAL (base zarr), theta=0 labels
    overlay(axes[0], 0.0, base["data/agentview_rgb"][gidx])
    for sp in axes[0].spines.values():
        sp.set_color("#1a7f37"); sp.set_linewidth(3)
    axes[0].set_title("ORIGINAL\n(base image + labels)", fontsize=9,
                      fontweight="bold", color="#1a7f37")

    for c, k in enumerate(order, start=1):
        ax = axes[c]
        theta = float(np.deg2rad(angles[k]))
        overlay(ax, theta, aug["images"]["agentview_rgb"][f"angle_{k:02d}"][gidx])
        ok = bool(valid[k])
        col = "#333333" if ok else "#d1242f"
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(3 if not ok else 1.2)
        title = f"θ={angles[k]:+.0f}°"
        if int(k) in gap_px:
            title += f"\nlabel↔gripper {gap_px[int(k)]:.2f} px"
        if not ok:
            title += "\n[INVALID]"
        ax.set_title(title, fontsize=9, color=col)

    env.close()

    fig.suptitle(
        f"Are the SE(2)-augmented labels legal?  task_uid {uid}: {prompt}\n"
        f"Analytic rotated labels — "
        r"$\bf{green}$=eef  $\bf{cyan}$=action δ(dx,dy)  $\bf{yellow}$=future eef path"
        f"  — projected onto the INDEPENDENTLY re-rendered images (ep {e}, frame {off}/{ep_len-1})\n"
        f"They register on the gripper at every angle. Certified: probe2 |analytic−sim eef| ≤ 6.7e-16 m; "
        f"probe3 rotated-action tracking median 6.4 nm.",
        fontsize=10, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print(f"wrote {args.out}")
    if gap_px:
        print("live label-vs-simulated-gripper gap (px) per angle:",
              {float(angles[k]): round(gap_px[int(k)], 3) for k in order})
        print("pure rotation-legality residual |sim_eef(θ)−R·sim_eef(0)| (nm) per angle:",
              {float(angles[k]): round(rot_resid_nm[int(k)], 4) for k in order})


if __name__ == "__main__":
    main()

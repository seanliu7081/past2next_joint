"""Visualize SE(2) data augmentation: original (base zarr) vs each discrete
angle (aug zarr), for both the global (agentview) and wrist (eye-in-hand)
cameras, for one (episode, frame).

Usage:
    python viz_se2_aug.py --episode 16 --frame mid --out out.png
"""
import argparse
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "data/libero/libero10_N500.zarr"
AUG = "data/libero/libero10_N500_se2aug.zarr"
CAMS = [("agentview_rgb", "Global (agentview)"),
        ("robot0_eye_in_hand_rgb", "Wrist (eye-in-hand)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=16)
    ap.add_argument("--frame", default="mid",
                    help="'mid', 'start', 'end', or an integer offset into the episode")
    ap.add_argument("--out", default="se2_aug_viz.png")
    args = ap.parse_args()

    base = zarr.open(BASE, mode="r")
    aug = zarr.open(AUG, mode="r")

    ee = aug["meta/episode_ends"][:]
    starts = np.concatenate([[0], ee[:-1]])
    e = args.episode
    s, end = int(starts[e]), int(ee[e])
    ep_len = end - s

    if args.frame == "mid":
        off = ep_len // 2
    elif args.frame == "start":
        off = 0
    elif args.frame == "end":
        off = ep_len - 1
    else:
        off = int(args.frame)
    gidx = s + off  # global step index

    angles = aug["meta/angles_deg"][:]
    valid = aug["meta/valid_mask"][e]           # (n_angles,) bool
    order = np.argsort(angles)                  # display -30..+30
    n_ang = len(angles)

    prompt = str(base["data/prompt"][s])
    task_uid = int(base["data/task_uid"][s, 0])

    ncols = 1 + n_ang                           # original + each angle
    nrows = len(CAMS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 2.35 * nrows))

    for r, (cam_key, cam_label) in enumerate(CAMS):
        # column 0: ORIGINAL, straight from the base zarr
        ax = axes[r, 0]
        ax.imshow(base["data"][cam_key][gidx])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#1a7f37"); sp.set_linewidth(2.5)
        if r == 0:
            ax.set_title("ORIGINAL\n(base zarr)", fontsize=10, fontweight="bold",
                         color="#1a7f37")
        ax.set_ylabel(cam_label, fontsize=11, fontweight="bold")

        # columns 1..: each discrete angle from the aug zarr
        for c, k in enumerate(order, start=1):
            ax = axes[r, c]
            img = aug["images"][cam_key][f"angle_{k:02d}"][gidx]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ok = bool(valid[k])
            col = "#333333" if ok else "#d1242f"
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(2.5 if not ok else 1.0)
            if r == 0:
                deg = angles[k]
                tag = "  (θ=0 re-render)" if deg == 0 else ""
                title = f"θ={deg:+.0f}°{tag}"
                if not ok:
                    title += "\n[INVALID]"
                ax.set_title(title, fontsize=10,
                             color=col, fontweight="bold" if not ok else "normal")

    fig.suptitle(
        f"SE(2) yaw augmentation — task_uid {task_uid}: {prompt}\n"
        f"episode {e}, frame {off}/{ep_len-1} (global idx {gidx})   "
        f"green=original · red border=INVALID(θ) pair",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}  ({ncols} cols x {nrows} rows; angles {angles[order].tolist()})")
    print(f"valid mask (display order): "
          f"{dict((f'{angles[k]:+.0f}', bool(valid[k])) for k in order)}")


if __name__ == "__main__":
    main()

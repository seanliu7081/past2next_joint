"""Render one video per demo for the SE(2) augmentation: the ORIGINAL demo
(base zarr) plus each discrete-angle augmented demo (aug zarr), each showing the
global (agentview) and wrist (eye-in-hand) cameras side by side over the whole
episode.

    original.mp4                     base zarr images
    aug_theta_+00deg.mp4  ...        aug zarr images/{cam}/angle_kk

Only angles whose (episode, angle) pair is VALID are rendered by default
(invalid pairs were rejected mid-episode during prerender, so their later
frames are blank); pass --all_angles to force-render them anyway.

Run from the repo root (no sim needed -- reads pre-rendered pixels):
    /home/haotian/miniforge3/envs/oat/bin/python \
        analysis_scripts/render_aug_videos.py --episode 16 --fps 20
"""
import argparse
import os
import pathlib
import sys

REPO = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import zarr
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from oat.env.libero.env import task_name_to_suite_and_ids

BASE = "data/libero/libero10_N500.zarr"
AUG = "data/libero/libero10_N500_se2aug.zarr"
CAMS = [("agentview_rgb", "Global (agentview)"),
        ("robot0_eye_in_hand_rgb", "Wrist (eye-in-hand)")]


def _font(size):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:                                              # noqa: BLE001
        return ImageFont.load_default()


def make_frame(cam_imgs, scale, banner, captions, accent, font, cap_font):
    """cam_imgs: list of HxWx3 uint8 (native res). Returns a composed uint8
    frame: title banner + [cam | cam ...] upscaled + per-camera captions."""
    ups = [np.asarray(Image.fromarray(im).resize(
        (im.shape[1] * scale, im.shape[0] * scale), Image.NEAREST)) for im in cam_imgs]
    ch = ups[0].shape[0]
    gap = 10
    body_w = sum(u.shape[1] for u in ups) + gap * (len(ups) - 1)
    body = np.full((ch, body_w, 3), 20, np.uint8)
    x = 0
    xs = []
    for u in ups:
        body[:, x:x + u.shape[1]] = u
        xs.append(x + u.shape[1] // 2)
        x += u.shape[1] + gap

    banner_h, cap_h = 44, 26
    W = body_w
    W += W % 2                                                     # even for libx264
    total_h = banner_h + ch + cap_h
    total_h += total_h % 2
    canvas = np.full((total_h, W, 3), 20, np.uint8)
    canvas[banner_h:banner_h + ch, :body_w] = body

    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, banner_h - 1], fill=(15, 15, 18))
    d.rectangle([0, 0, 6, banner_h - 1], fill=accent)
    d.text((14, 11), banner, fill=(240, 240, 240), font=font)
    for cx, cap in zip(xs, captions):
        w = d.textlength(cap, font=cap_font)
        d.text((cx - w / 2, banner_h + ch + 4), cap, fill=(200, 200, 200), font=cap_font)
    return np.asarray(img)


def write_video(path, frames_iter, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                           macro_block_size=1, ffmpeg_log_level="error")
    for fr in frames_iter:
        w.append_data(fr)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=16)
    ap.add_argument("--out_dir", default=None,
                    help="default analysis_scripts/videos/ep<E>")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=3, help="pixel upscale (nearest)")
    ap.add_argument("--all_angles", action="store_true",
                    help="also render angles whose pair is INVALID (blank tail)")
    ap.add_argument("--format", default="mp4", choices=["mp4", "gif"])
    args = ap.parse_args()

    base = zarr.open(BASE, mode="r")
    aug = zarr.open(AUG, mode="r")
    ee = aug["meta/episode_ends"][:]
    starts = np.concatenate([[0], ee[:-1]])
    angles = aug["meta/angles_deg"][:]
    e = args.episode
    s, end = int(starts[e]), int(ee[e])
    ep_len = end - s
    valid = aug["meta/valid_mask"][e]
    uid = int(base["data/task_uid"][s, 0])
    prompt = str(base["data/prompt"][s])
    task_name = {u: n for n, (su, si, u) in task_name_to_suite_and_ids.items()}[uid]

    out_dir = args.out_dir or f"analysis_scripts/videos/ep{e}"
    os.makedirs(out_dir, exist_ok=True)
    ext = args.format
    font, cap_font = _font(18), _font(15)
    captions = [c[1] for c in CAMS]

    def frames_for(get_imgs, banner_prefix, accent):
        for t in range(ep_len):
            cams = [get_imgs(cam_key, s + t) for cam_key, _ in CAMS]
            banner = (f"{banner_prefix}   ·   frame {t+1:3d}/{ep_len}   ·   "
                      f"ep{e}  {task_name[:20]}")
            yield make_frame(cams, args.scale, banner, captions, accent, font, cap_font)

    written = []
    # ORIGINAL (base zarr)
    p = f"{out_dir}/original.{ext}"
    write_video(p, frames_for(lambda k, i: base["data"][k][i],
                              "ORIGINAL demo (base)", (26, 160, 60)), args.fps)
    written.append(p)
    print(f"[render] {p}")

    # each augmented angle
    order = np.argsort(angles)
    for k in order:
        k = int(k)
        deg = angles[k]
        if not bool(valid[k]) and not args.all_angles:
            print(f"[render] skip θ={deg:+.0f}° (invalid pair; use --all_angles)")
            continue
        tag = "re-render θ=0" if deg == 0 else f"AUGMENTED θ={deg:+.0f}°"
        accent = (200, 40, 60) if not bool(valid[k]) else (0, 150, 220)
        p = f"{out_dir}/aug_theta_{deg:+03.0f}deg.{ext}"
        write_video(
            p,
            frames_for(
                (lambda kk: (lambda ck, i: aug["images"][ck][f"angle_{kk:02d}"][i]))(k),
                tag, accent),
            args.fps)
        written.append(p)
        print(f"[render] {p}")

    print(f"\n[render] {len(written)} videos in {out_dir}  "
          f"(ep {e}, {ep_len} frames @ {args.fps} fps, task_uid {uid}: {prompt})")


if __name__ == "__main__":
    main()

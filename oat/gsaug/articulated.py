"""Articulated per-task robot asset (plan §6.4, M4): labeled per-link depth
init + multi-config FK finetune. One asset, frame='link', SH degree 1
(``rotate_sh_l1`` under full SO(3), G5), per-Gaussian ``link_id`` indexing
``meta['link_names']``.

Init (M4a): from the CANONICAL config (the capture's first config) — per-pixel
link ids from the seg render + the capture's ``geom_to_link`` map, backproject
robot pixels, express each point in its link's LOCAL frame at the canonical
config (``x_l = R_wl^T (x_w − p_wl)``), voxel-downsample per link (4 mm).

Finetune (M4b): each iteration samples a random (config, view) pair from the
training split, poses ALL links differentiably from that config's recorded
``link_poses`` (means: ``p + R(q) x_l``; quats: ``q ⊗ q_l``; SH rotated by the
link's delta rotation ``R(q) R(q_canonical)^T`` through
``sh_rotation.rotate_sh_l1`` — MANDATORY: it carries the signed-permutation
l=1 basis verified against gsplat's SH evaluator), rasterizes jointly, and
optimizes the LOCAL parameters through the pose transform. No densification
(the labeled init is dense); one opacity prune at iter 2000. Per-config
rotation/SH-rotation matrices are cached (only ~60 configs).

The saved asset's meta matches exactly what ``compose.GSCompositeRenderer``
loads for frame='link' assets: ordered ``link_names``, dict-valued
``p_capture``/``q_capture`` keyed by link name (canonical-config world poses),
SH degree 1, task / model_xml_sha1 / versions / metrics.
"""

import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import gsplat
from gsplat.strategy.ops import remove as _gs_remove

from oat.gsaug.cameras import c2w_from_w2c, project
from oat.gsaug.capture import CaptureBundle
from oat.gsaug.components import quat_mul, quat_normalize, quat_to_R
from oat.gsaug.gaussian_asset import EXPECTED_CONVENTIONS, GaussianAsset
from oat.gsaug.sh_rotation import rotate_sh_l1
from oat.gsaug.trainer import (
    SH_C0,
    backproject,
    build_optimizers,
    masked_psnr,
    rasterize_views,
    scene_extent_of,
    silhouette_iou,
    ssim_map,
    voxel_downsample,
)

SH_DEGREE_ROBOT = 1                  # G5: robot links are SH degree 1
K_REST_ROBOT = 3
DEFAULT_VOXEL_M = 0.004              # plan §6.4 per-link init voxel
DEFAULT_ITERS = 15000                # plan §6.4
INIT_STRIDE = 2                      # denser than the static stride-4: one
                                     # config's 16 views must cover thin parts

_FIT_DEFAULTS = dict(
    voxel_m=DEFAULT_VOXEL_M,
    stride=INIT_STRIDE,
    seed=0,
    canonical_index=0,               # plan: canonical = first config
    n_holdout=4,                     # held-out configs (>=1 joint1-shifted)
    ssim_weight=0.2,
    silhouette_weight=1.0,
    lrs=None,
    prune_iter=2000,                 # single opacity prune (plan §6.4)
    prune_opacity=0.005,
    eef_link=None,                   # None -> first link name containing 'hand'
    log_every=1000,
)


# ── capture-schema helpers ──────────────────────────────────────────────────

def _geom_link_lut(bundle: CaptureBundle) -> Tuple[List[str], np.ndarray]:
    """(link_names ordered list, geom-id -> link-index LUT (−1 = not robot))
    from the capture transforms' ``link_names`` + ``geom_to_link``."""
    tf = bundle.transforms
    assert "link_names" in tf and "geom_to_link" in tf and "configs" in tf, (
        f"'{bundle.directory}' transforms lack link_names/geom_to_link/"
        f"configs — not a robot capture (re-run capture_assets.py)")
    link_names = list(tf["link_names"])
    index = {n: i for i, n in enumerate(link_names)}
    g2l = tf["geom_to_link"]
    max_gid = max(int(g) for g in g2l)
    lut = np.full(max_gid + 1, -1, dtype=np.int64)
    for g, name in g2l.items():
        assert name in index, (
            f"geom_to_link maps geom {g} to unknown link {name!r} "
            f"(link_names: {link_names})")
        lut[int(g)] = index[name]
    return link_names, lut


def _link_index_image(seg: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Per-pixel link index (−1 = not a robot pixel) from a geom-id seg."""
    out = np.full(seg.shape, -1, dtype=np.int64)
    valid = (seg >= 0) & (seg < len(lut))
    out[valid] = lut[seg[valid]]
    return out


def _config_shift_deg(cfg: dict) -> float:
    try:
        return abs(float(cfg.get("joint1_shift_deg", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def select_holdout_configs(configs: List[dict], n_holdout: int = 4,
                           canonical_index: int = 0) -> List[int]:
    """Deterministic held-out config pick (plan §6.4: 4 configs, at least one
    joint1-shifted). Uses the capture's ``source``/``joint1_shift_deg`` fields;
    if absent, the most-shifted config is inferred from qpos[0]'s deviation
    from the pool median. The canonical config is never held out; at least two
    training configs always remain (n_holdout is clamped)."""
    n = len(configs)
    n_holdout = min(int(n_holdout), n - 2)
    if n_holdout <= 0:
        return []
    shifted = [i for i, c in enumerate(configs)
               if i != canonical_index
               and (c.get("source") == "joint1_shift"
                    or _config_shift_deg(c) > 0.0)]
    if not shifted:
        q0 = np.array([float(c["qpos"][0]) for c in configs])
        dev = np.abs(q0 - np.median(q0))
        dev[canonical_index] = -np.inf
        shifted = [int(np.argmax(dev))]
    shifted.sort(key=lambda i: (-_config_shift_deg(configs[i]), i))
    held = [shifted[0]]

    rest = [i for i in range(n)
            if i != canonical_index and i not in held]
    non_shifted = [i for i in rest if i not in shifted] or rest
    k = n_holdout - len(held)
    if k > 0:
        pick = np.round(np.linspace(0, len(non_shifted) - 1,
                                    min(k, len(non_shifted)))).astype(int)
        for j in dict.fromkeys(int(p) for p in pick):
            if non_shifted[j] not in held:
                held.append(non_shifted[j])
    return sorted(held)[:n_holdout]


# ── init (M4a) ──────────────────────────────────────────────────────────────

def init_robot_from_depth(bundle: CaptureBundle,
                          voxel_m: float = DEFAULT_VOXEL_M,
                          stride: int = INIT_STRIDE,
                          canonical_index: int = 0
                          ) -> Dict[str, torch.Tensor]:
    """Labeled per-link depth init from the canonical config's views.

    Returns CPU tensors: means (LINK-local), identity quats,
    log_scales = log(2·voxel_m), opacity_logits = 0, sh_dc from pixel color,
    sh_rest zeros (N, 3, 3) (degree 1), and ``link_id`` int32 (sorted
    ascending — gaussians are contiguous per link) indexing the capture's
    ordered ``link_names``.
    """
    link_names, lut = _geom_link_lut(bundle)
    configs = bundle.configs
    assert 0 <= canonical_index < len(configs), (canonical_index, len(configs))
    cfg0 = configs[canonical_index]
    poses = cfg0["link_poses"]
    missing = [n for n in link_names if n not in poses]
    assert not missing, f"canonical config lacks link poses for {missing}"

    L = len(link_names)
    pts_l: List[List[np.ndarray]] = [[] for _ in range(L)]
    cols_l: List[List[np.ndarray]] = [[] for _ in range(L)]
    for vi in cfg0["view_ids"]:
        v = bundle.views[vi]
        H, W = v.depth.shape
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        ys, xs = ys.ravel(), xs.ravel()
        d = v.depth[ys, xs]
        li = _link_index_image(v.seg, lut)[ys, xs]
        keep = np.isfinite(d) & (d > 1e-4) & (li >= 0)
        if not keep.any():
            continue
        ys, xs, li = ys[keep], xs[keep], li[keep]
        pts_w = backproject(v.K, v.c2w, v.depth, ys, xs)
        cols = v.rgb[ys, xs].astype(np.float64) / 255.0
        for l in np.unique(li):
            sel = li == l
            name = link_names[int(l)]
            p_wl = np.asarray(poses[name]["p"], dtype=np.float64)
            q_wl = np.asarray(poses[name]["q_wxyz"], dtype=np.float64)
            R_wl = quat_to_R(torch.as_tensor(q_wl, dtype=torch.float64)).numpy()
            pts_l[int(l)].append((pts_w[sel] - p_wl) @ R_wl)  # R^T (x − p)
            cols_l[int(l)].append(cols[sel])

    means, colors, link_id = [], [], []
    counts = {}
    for l in range(L):
        if not pts_l[l]:
            continue
        m, c = voxel_downsample(np.concatenate(pts_l[l]),
                                np.concatenate(cols_l[l]), voxel_m)
        means.append(m)
        colors.append(c)
        link_id.append(np.full(len(m), l, dtype=np.int32))
        counts[link_names[l]] = len(m)
    assert means, "no robot pixels found in the canonical config's views"
    means = np.concatenate(means)
    colors = np.concatenate(colors)
    link_id = np.concatenate(link_id)
    n = len(means)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    print(f"[articulated] init: {n} gaussians over {len(counts)}/{L} links "
          f"({counts})")
    return {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "quats": torch.from_numpy(quats),
        "log_scales": torch.full((n, 3), math.log(2.0 * float(voxel_m)),
                                 dtype=torch.float32),
        "opacity_logits": torch.zeros(n, dtype=torch.float32),
        "sh_dc": torch.as_tensor((colors - 0.5) / SH_C0, dtype=torch.float32),
        "sh_rest": torch.zeros(n, K_REST_ROBOT, 3, dtype=torch.float32),
        "link_id": torch.from_numpy(link_id),
    }


# ── finetune (M4b) + save ───────────────────────────────────────────────────

def fit_robot(bundle: CaptureBundle, out_path: str,
              iters: int = DEFAULT_ITERS, device: str = "cuda:0",
              **overrides) -> Tuple[GaussianAsset, Dict]:
    """Init + multi-config FK finetune + save of the per-task robot asset
    (plan §6.4). ``overrides`` may set any key of ``_FIT_DEFAULTS``.

    Returns (asset, metrics): robot-region PSNR, per-link silhouette IoU and
    EEF projection error over the held-out configs.
    """
    cfg = dict(_FIT_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"fit_robot: unknown override(s) {sorted(unknown)}; "
                        f"allowed: {sorted(cfg)}")
    cfg.update(overrides)
    iters = int(iters)
    dev = torch.device(device)
    torch.manual_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))

    link_names, lut = _geom_link_lut(bundle)
    L = len(link_names)
    configs = bundle.configs
    assert len(configs) >= 3, (
        f"robot capture has only {len(configs)} configs — need >= 3 "
        f"(train + held-out)")
    canonical_index = int(cfg["canonical_index"])
    size = bundle.image_size

    holdout_cfgs = select_holdout_configs(configs, int(cfg["n_holdout"]),
                                          canonical_index)
    train_cfgs = [i for i in range(len(configs)) if i not in holdout_cfgs]
    train_pairs = [(ci, vi) for ci in train_cfgs
                   for vi in configs[ci]["view_ids"]]
    assert train_pairs, "empty training split"

    # ── per-view CPU tensors + device cameras ───────────────────────────────
    rgb_t = [torch.from_numpy(np.ascontiguousarray(v.rgb))
             for v in bundle.views]
    robot_t = [torch.from_numpy(_link_index_image(v.seg, lut) >= 0)
               for v in bundle.views]
    viewmats = torch.as_tensor(
        np.stack([c2w_from_w2c(v.c2w) for v in bundle.views]),
        dtype=torch.float32, device=dev)
    Ks = torch.as_tensor(np.stack([v.K for v in bundle.views]),
                         dtype=torch.float32, device=dev)

    # ── per-config pose caches (p, q, R, and SH delta R·R0^T) ──────────────
    q0s = torch.stack([quat_normalize(torch.as_tensor(
        configs[canonical_index]["link_poses"][n]["q_wxyz"],
        dtype=torch.float32)) for n in link_names])
    R0 = quat_to_R(q0s)                                       # (L,3,3)
    pose_cache = []
    for c in configs:
        lp = c["link_poses"]
        p = torch.stack([torch.as_tensor(lp[n]["p"], dtype=torch.float32)
                         for n in link_names]).to(dev)
        q = torch.stack([quat_normalize(torch.as_tensor(
            lp[n]["q_wxyz"], dtype=torch.float32)) for n in link_names]).to(dev)
        R = quat_to_R(q)
        Rd = R @ R0.to(dev).transpose(-1, -2)                 # delta vs canonical
        pose_cache.append({"p": p, "q": q, "R": R, "Rd": Rd})

    # ── init + parameters ───────────────────────────────────────────────────
    init = init_robot_from_depth(bundle, float(cfg["voxel_m"]),
                                 int(cfg["stride"]), canonical_index)
    link_id = init["link_id"].to(dev)
    assert bool((link_id[1:] >= link_id[:-1]).all()), \
        "init link_id must be sorted (per-link contiguity)"
    link_idx = link_id.to(torch.long)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(init["means"].to(dev)),
        "quats": torch.nn.Parameter(init["quats"].to(dev)),
        "scales": torch.nn.Parameter(init["log_scales"].to(dev)),
        "opacities": torch.nn.Parameter(init["opacity_logits"].to(dev)),
        "sh0": torch.nn.Parameter(init["sh_dc"].to(dev)),
        "shN": torch.nn.Parameter(init["sh_rest"].to(dev)),
    })
    extent = scene_extent_of(bundle.views)
    optimizers = build_optimizers(params, extent, cfg["lrs"])

    def _slices(lid: torch.Tensor) -> List[Tuple[int, int]]:
        b = np.searchsorted(lid.detach().cpu().numpy(), np.arange(L + 1))
        return [(int(b[l]), int(b[l + 1])) for l in range(L)]
    slices = _slices(link_id)

    def _pose(cache: dict, s: int = 0, e: Optional[int] = None):
        """Differentiable FK posing of gaussians [s:e) for one config; the SH
        of each link goes through rotate_sh_l1 (signed l=1 basis, G5)."""
        e = int(params["means"].shape[0]) if e is None else e
        idx = link_idx[s:e]
        R_g = cache["R"][idx]                                   # (n,3,3)
        means_w = torch.bmm(R_g, params["means"][s:e].unsqueeze(-1)
                            ).squeeze(-1) + cache["p"][idx]
        quats_w = quat_mul(cache["q"][idx], params["quats"][s:e])
        sh = torch.cat([params["sh0"][s:e, None, :], params["shN"][s:e]],
                       dim=1)                                   # (n,4,3)
        parts = []
        for l in range(L):
            ls, le = max(slices[l][0], s), min(slices[l][1], e)
            if le > ls:
                parts.append(rotate_sh_l1(sh[ls - s:le - s], cache["Rd"][l]))
        sh_w = parts[0] if len(parts) == 1 else torch.cat(parts)
        return means_w, quats_w, sh_w

    def _raster(cache: dict, vi: int, s: int = 0, e: Optional[int] = None):
        means_w, quats_w, sh_w = _pose(cache, s, e)
        e_ = int(params["means"].shape[0]) if e is None else e
        return rasterize_views(
            means_w, quats_w, params["scales"][s:e_].exp(),
            params["opacities"][s:e_].sigmoid(), sh_w,
            viewmats[vi:vi + 1], Ks[vi:vi + 1], size, size, SH_DEGREE_ROBOT)

    # ── finetune loop ───────────────────────────────────────────────────────
    recent = {k: deque(maxlen=100) for k in ("total", "l1", "ssim", "sil")}
    for step in range(iters):
        ci, vi = train_pairs[int(rng.integers(len(train_pairs)))]
        gt = rgb_t[vi].to(dev).float() / 255.0
        m = robot_t[vi].to(dev)
        img, alpha, _info = _raster(pose_cache[ci], vi)
        img0, a0 = img[0], alpha[0, :, :, 0]
        l1 = (img0 - gt).abs()[m].mean()
        sloss = 1.0 - ssim_map(img0.clamp(0, 1), gt)[m].mean()
        sil = (a0 - m.float()).abs().mean()
        loss = (l1 + float(cfg["ssim_weight"]) * sloss
                + float(cfg["silhouette_weight"]) * sil)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for k, v in (("total", loss), ("l1", l1), ("ssim", sloss), ("sil", sil)):
            recent[k].append(float(v))

        if step + 1 == int(cfg["prune_iter"]):  # single prune (plan §6.4)
            with torch.no_grad():
                drop = params["opacities"].sigmoid() < float(cfg["prune_opacity"])
            if bool(drop.any()):
                _gs_remove(params, optimizers, state={}, mask=drop)
                keep = ~drop
                link_id = link_id[keep]
                link_idx = link_idx[keep]
                slices = _slices(link_id)
                print(f"[articulated] step {step + 1}: pruned "
                      f"{int(drop.sum())} -> {params['means'].shape[0]} gaussians")
        if cfg["log_every"] and (step + 1) % int(cfg["log_every"]) == 0:
            print(f"[articulated] step {step + 1}/{iters}: "
                  f"loss {np.mean(recent['total']):.4f} "
                  f"n={params['means'].shape[0]}")

    # ── held-out metrics (plan §6.4) ────────────────────────────────────────
    eef_link = cfg["eef_link"] or next(
        (n for n in link_names if "hand" in n), link_names[-1])
    assert eef_link in link_names, f"eef_link {eef_link!r} not in link_names"
    eef_l = link_names.index(eef_link)
    eval_cfgs = holdout_cfgs if holdout_cfgs else train_cfgs

    psnrs, eef_px = [], []
    iou_per_link: Dict[str, List[float]] = {n: [] for n in link_names}
    with torch.no_grad():
        for ci in eval_cfgs:
            cache = pose_cache[ci]
            for vi in configs[ci]["view_ids"]:
                gt = rgb_t[vi].to(dev).float() / 255.0
                m = robot_t[vi].to(dev)
                if not bool(m.any()):
                    continue
                img, _alpha, _ = _raster(cache, vi)
                psnrs.append(masked_psnr(img[0], gt, m))

                li_img = _link_index_image(bundle.views[vi].seg, lut)
                for l in range(L):
                    s, e = slices[l]
                    gt_mask = torch.from_numpy(li_img == l).to(dev)
                    if e <= s:
                        if bool(gt_mask.any()):
                            iou_per_link[link_names[l]].append(0.0)
                        continue
                    _i, al, _ = _raster(cache, vi, s, e)
                    iou = silhouette_iou(al[0, :, :, 0], gt_mask)
                    if iou is not None:
                        iou_per_link[link_names[l]].append(iou)
                    if l == eef_l:
                        uv = project(bundle.views[vi].K,
                                     c2w_from_w2c(bundle.views[vi].c2w),
                                     np.asarray(
                                         configs[ci]["link_poses"][eef_link]["p"],
                                         dtype=np.float64))
                        pred = (al[0, :, :, 0] > 0.5).cpu().numpy()
                        if np.all(np.isfinite(uv)) and pred.any():
                            u, v_ = float(uv[0]), float(uv[1])
                            iu, iv = int(round(u - 0.5)), int(round(v_ - 0.5))
                            if 0 <= iv < size and 0 <= iu < size and pred[iv, iu]:
                                eef_px.append(0.0)
                            else:
                                pys, pxs = np.nonzero(pred)
                                eef_px.append(float(np.sqrt(
                                    ((pys + 0.5 - v_) ** 2
                                     + (pxs + 0.5 - u) ** 2)).min()))

    ious_flat = {n: (float(np.mean(v)) if v else None)
                 for n, v in iou_per_link.items()}
    iou_vals = [v for v in ious_flat.values() if v is not None]
    metrics = {
        "psnr_robot": float(np.mean(psnrs)),
        "psnr_robot_min": float(np.min(psnrs)),
        "silhouette_iou_per_link": ious_flat,
        "silhouette_iou_mean": float(np.mean(iou_vals)) if iou_vals else None,
        "silhouette_iou_min": float(np.min(iou_vals)) if iou_vals else None,
        "eef_link": eef_link,
        "eef_median_px": float(np.median(eef_px)) if eef_px else None,
        "n_gaussians": int(params["means"].shape[0]),
        "n_configs": len(configs),
        "heldout_config_indices": [int(i) for i in holdout_cfgs],
        "heldout_is_train_fallback": not holdout_cfgs,
        "scene_extent_m": float(extent),
        "final_losses": {k: float(np.mean(v)) for k, v in recent.items()},
    }

    # ── save (the compose.py frame='link' contract) ─────────────────────────
    poses0 = configs[canonical_index]["link_poses"]
    train_args = {
        "iters": iters,
        **{k: cfg[k] for k in ("voxel_m", "stride", "seed", "canonical_index",
                               "n_holdout", "ssim_weight", "silhouette_weight",
                               "prune_iter", "prune_opacity", "lrs")},
        "eef_link": eef_link,
    }
    meta = {
        "frame": "link",
        "link_names": list(link_names),
        "p_capture": {n: [float(x) for x in poses0[n]["p"]]
                      for n in link_names},
        "q_capture": {n: [float(x) for x in poses0[n]["q_wxyz"]]
                      for n in link_names},
        "task": bundle.task, "component": "robot",
        "model_xml_sha1": bundle.model_xml_sha1,
        "canonical_config_index": canonical_index,
        "versions": {"gsplat": gsplat.__version__,
                     "torch": torch.__version__},
        "train_args": train_args, "metrics": metrics,
    }
    asset = GaussianAsset(
        means=params["means"].detach().cpu().contiguous(),
        quats=quat_normalize(params["quats"].detach().cpu()).contiguous(),
        log_scales=params["scales"].detach().cpu().contiguous(),
        opacity_logits=params["opacities"].detach().cpu().contiguous(),
        sh_dc=params["sh0"].detach().cpu().contiguous(),
        sh_rest=params["shN"].detach().cpu().contiguous(),
        conventions=dict(EXPECTED_CONVENTIONS),
        meta=meta,
        link_id=link_id.detach().to("cpu", torch.int32).contiguous())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    digest = asset.save(out_path)
    print(f"[articulated] robot: saved {out_path} "
          f"({metrics['n_gaussians']} gaussians, sha1 {digest[:12]}…, "
          f"PSNR {metrics['psnr_robot']:.2f}, "
          f"IoU_min {metrics['silhouette_iou_min']}, "
          f"EEF {metrics['eef_median_px']} px)")
    return asset, metrics

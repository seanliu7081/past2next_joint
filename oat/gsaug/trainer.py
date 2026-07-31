"""Static GS asset training (plan §5.2, M3): depth-based init + masked
photometric fitting for the task background (frame='world') and each movable
object (frame='body').

Consumes a ``CaptureBundle`` (oat/gsaug/capture.py), produces a
``GaussianAsset`` (oat/gsaug/gaussian_asset.py) that ``compose.py`` can pose:

* training happens in WORLD frame at the single static capture pose;
* at save time, background assets stay world-frame (identity capture pose) and
  object assets are re-expressed in the capture BODY frame (means/quats only —
  SH stays world-at-capture per ``EXPECTED_CONVENTIONS['sh_frame']``, G5), so
  ``PosedComponent.posed(p_capture, q_capture)`` reproduces the trained world
  Gaussians exactly.

Loss (plan §5.2): L1 + 0.2·(1−SSIM) over the component's seg-derived pixel
mask, plus a full-image silhouette term ``|alpha − mask|`` for objects and for
the robot-masked background contingency (detected via the capture transforms'
``masks_dir``). In masked mode the robot pixels are excluded from every term —
the true background continues behind the robot, so forcing alpha to 0 there
would be wrong. SSIM is the standard window-11 gaussian form, torch-native
(no torchvision/skimage dependency).

Densification uses gsplat 1.5.3's ``DefaultStrategy`` with its documented
usage (check_sanity / step_pre_backward / step_post_backward on the
``packed=False`` info dict). NOTE: 1.5.3's ``step_post_backward`` has an
operator-precedence bug (``step % reset_every == 0 & step > 0`` is always
False), so its periodic opacity reset never fires; the loop below applies
``gsplat.strategy.ops.reset_opa`` itself on the plan §5.2 schedule.
"""

import math
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import gsplat
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa

from oat.gsaug.cameras import c2w_from_w2c
from oat.gsaug.capture import CaptureBundle
from oat.gsaug.components import quat_conj, quat_mul, quat_normalize, quat_to_R
from oat.gsaug.gaussian_asset import EXPECTED_CONVENTIONS, GaussianAsset

SH_C0 = 0.28209479177387814          # Y_00 constant (3DGS DC color transform)
INIT_STRIDE = 4                      # plan §5.2 depth-init pixel stride
SH_DEGREE_STATIC = 3                 # static assets: SH degree 3 (G5)
K_REST_STATIC = (SH_DEGREE_STATIC + 1) ** 2 - 1   # 15

DEFAULT_VOXEL_M = {"background": 0.01, "object": 0.003}   # plan §5.2
DEFAULT_ITERS = {"background": 7000, "object": 5000}      # plan §5.2

# Adam learning rates (plan §5.2 / 3DGS conventions); 'means' is additionally
# multiplied by the scene extent.
LR = {"means": 1.6e-4, "sh0": 2.5e-3, "shN": 1.25e-4,
      "opacities": 5e-2, "scales": 5e-3, "quats": 1e-3}

_FIT_DEFAULTS = dict(
    voxel_m=None,            # None -> DEFAULT_VOXEL_M[kind]
    max_init_views=None,
    seed=0,
    holdout_every=8,         # plan §5.2: every 8th view held out
    ssim_weight=0.2,
    silhouette_weight=1.0,
    lrs=None,                # dict overriding LR entries by name
    refine_start=500,
    refine_stop=None,        # None -> iters // 2 (plan §5.2 schedule)
    refine_every=100,
    reset_every=3000,
    reset_opacity=0.05,      # plan §5.2: logits reset to logit(0.05)
    prune_opa=0.005,
    grow_grad2d=2e-4,
    log_every=1000,
)


# ── SSIM (window-11 gaussian, torch-native) ─────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, device,
                     dtype) -> torch.Tensor:
    x = torch.arange(window_size, dtype=dtype, device=device) \
        - (window_size - 1) / 2.0
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g).expand(3, 1, window_size, window_size).contiguous()


def ssim_map(img: torch.Tensor, ref: torch.Tensor, window_size: int = 11,
             sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM map (H, W) — channel mean — between two (H, W, 3) float
    images in [0, 1]. Standard constants C1=0.01², C2=0.03²; zero-padded
    'same' convolution (border pixels slightly biased, uniformly for both)."""
    assert img.shape == ref.shape and img.dim() == 3 and img.shape[2] == 3, \
        (img.shape, ref.shape)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x = img.permute(2, 0, 1)[None]
    y = ref.permute(2, 0, 1)[None]
    k = _gaussian_kernel(window_size, sigma, img.device, img.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, k, padding=pad, groups=3)
    mu_y = F.conv2d(y, k, padding=pad, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, k, padding=pad, groups=3) - mu_x2
    sig_y = F.conv2d(y * y, k, padding=pad, groups=3) - mu_y2
    sig_xy = F.conv2d(x * y, k, padding=pad, groups=3) - mu_xy
    num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    return (num / den)[0].mean(0)


# ── depth init building blocks (shared with articulated.py) ─────────────────

def backproject(K: np.ndarray, c2w: np.ndarray, depth: np.ndarray,
                ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Backproject pixel (ys, xs) metric z-depths into world points (M, 3)
    float64 — pixel-center (+0.5) convention, the same recipe capture.py's
    ``table_depth_error_m`` validates against rendered pixels (G7)."""
    d = np.asarray(depth, dtype=np.float64)[ys, xs]
    uv1 = np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))], axis=1)
    rays = uv1 @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T
    pts_c = rays * d[:, None]
    c2w = np.asarray(c2w, dtype=np.float64)
    return pts_c @ c2w[:3, :3].T + c2w[:3, 3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                     voxel_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool points (M, 3) and colors (M, 3) over a voxel grid of pitch
    ``voxel_m``; returns per-voxel means (N, 3) and mean colors (N, 3)."""
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape == cols.shape and len(pts) > 0, \
        (pts.shape, cols.shape)
    ijk = np.floor(pts / float(voxel_m)).astype(np.int64)
    ijk -= ijk.min(axis=0)
    assert (ijk.max(axis=0) < (1 << 21)).all(), \
        f"voxel grid too large for packing: {ijk.max(axis=0)} @ {voxel_m} m"
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    _uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    n = len(counts)
    pm = np.zeros((n, 3), dtype=np.float64)
    cm = np.zeros((n, 3), dtype=np.float64)
    np.add.at(pm, inv, pts)
    np.add.at(cm, inv, cols)
    pm /= counts[:, None]
    cm /= counts[:, None]
    return pm, cm


def scene_extent_of(views) -> float:
    """Scene extent: max camera-center distance from their centroid × 1.1
    (the 3DGS convention gsplat's strategy/means-lr are calibrated against)."""
    c = np.stack([np.asarray(v.c2w, dtype=np.float64)[:3, 3] for v in views])
    return float(np.linalg.norm(c - c.mean(axis=0), axis=1).max()) * 1.1


def init_from_depth(bundle: CaptureBundle,
                    component_geom_ids: Optional[Sequence[int]],
                    voxel_m: float,
                    max_views: Optional[int] = None,
                    view_indices: Optional[Sequence[int]] = None
                    ) -> Dict[str, torch.Tensor]:
    """Depth-based init (plan §5.2, skip SfM): backproject stride-4 pixels of
    every view (K + metric depth + c2w from the bundle), seg-filter to the
    component's geom ids (None = every rendered geom, the background case),
    voxel-downsample, and build the initial parameter tensors.

    ``view_indices`` restricts the init to those views of the bundle (applied
    BEFORE ``max_views`` subsampling); default None = all views (backward
    compatible). ``fit_static`` passes its TRAINING split — held-out views
    must not leak into the init, or the held-out metrics are biased.

    Robot-masked pixels (``view.mask``) are always excluded. Returns CPU
    float32 tensors: means (world frame), identity quats,
    log_scales = log(2·voxel_m), opacity_logits = logit(0.5) = 0,
    sh_dc = (rgb/255 − 0.5)/Y00 from the source pixel color, sh_rest zeros at
    degree 3 (K=15).
    """
    views = list(bundle.views)
    if view_indices is not None:
        views = [views[int(i)] for i in view_indices]
        assert views, "init_from_depth: empty view_indices"
    if max_views is not None and len(views) > int(max_views):
        sel = np.round(np.linspace(0, len(views) - 1, int(max_views))).astype(int)
        views = [views[i] for i in sel]
    gids = (None if component_geom_ids is None
            else np.asarray(sorted(int(g) for g in component_geom_ids),
                            dtype=np.int64))

    pts_all: List[np.ndarray] = []
    col_all: List[np.ndarray] = []
    for v in views:
        H, W = v.depth.shape
        ys, xs = np.mgrid[0:H:INIT_STRIDE, 0:W:INIT_STRIDE]
        ys, xs = ys.ravel(), xs.ravel()
        d = v.depth[ys, xs]
        seg = v.seg[ys, xs]
        keep = np.isfinite(d) & (d > 1e-4)
        keep &= (seg >= 0) if gids is None else np.isin(seg, gids)
        if v.mask is not None:
            keep &= ~v.mask[ys, xs]
        if not keep.any():
            continue
        ys, xs = ys[keep], xs[keep]
        pts_all.append(backproject(v.K, v.c2w, v.depth, ys, xs))
        col_all.append(v.rgb[ys, xs].astype(np.float64) / 255.0)
    assert pts_all, (
        f"init_from_depth: no component pixels in any view of "
        f"'{bundle.directory}' (geom ids {None if gids is None else gids.tolist()})")

    means, cols = voxel_downsample(np.concatenate(pts_all),
                                   np.concatenate(col_all), voxel_m)
    n = len(means)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "quats": torch.from_numpy(quats),
        "log_scales": torch.full((n, 3), math.log(2.0 * float(voxel_m)),
                                 dtype=torch.float32),
        "opacity_logits": torch.zeros(n, dtype=torch.float32),  # logit(0.5)
        "sh_dc": torch.as_tensor((cols - 0.5) / SH_C0, dtype=torch.float32),
        "sh_rest": torch.zeros(n, K_REST_STATIC, 3, dtype=torch.float32),
    }


# ── shared training plumbing (also used by articulated.py) ──────────────────

def build_optimizers(params: "torch.nn.ParameterDict", scene_extent: float,
                     lrs: Optional[Dict[str, float]] = None
                     ) -> Dict[str, torch.optim.Optimizer]:
    """One Adam per parameter (the gsplat strategy contract): plan §5.2 rates,
    'means' scaled by the scene extent."""
    table = dict(LR)
    table.update(lrs or {})
    unknown = set(table) - set(LR)
    assert not unknown, f"unknown lr override keys {sorted(unknown)}"
    opts = {}
    for name, p in params.items():
        lr = table[name] * (scene_extent if name == "means" else 1.0)
        opts[name] = torch.optim.Adam([{"params": [p], "lr": lr, "name": name}],
                                      eps=1e-15)
    return opts


def rasterize_views(means, quats, scales, opacities, sh, viewmats, Ks,
                    width: int, height: int, sh_degree: int):
    """One ``gsplat.rasterization`` call, activated inputs, ``packed=False``
    (the DefaultStrategy info contract). Returns (img, alpha, info)."""
    return gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=sh, viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, packed=False, render_mode="RGB")


def masked_psnr(img: torch.Tensor, gt: torch.Tensor,
                mask: torch.Tensor) -> float:
    """PSNR over ``mask`` pixels; images (H, W, 3) float in [0, 1]."""
    assert bool(mask.any()), "masked_psnr: empty mask"
    mse = float(((img.clamp(0, 1) - gt) ** 2)[mask].mean())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def silhouette_iou(alpha: torch.Tensor, mask: torch.Tensor,
                   thresh: float = 0.5) -> Optional[float]:
    """IoU of (alpha > thresh) vs a boolean mask; None if the union is empty."""
    pred = alpha > thresh
    union = int((pred | mask).sum())
    if union == 0:
        return None
    return float((pred & mask).sum()) / union


# ── the static fit ──────────────────────────────────────────────────────────

def fit_static(bundle: CaptureBundle, component: str, out_path: str,
               iters: Optional[int] = None, device: str = "cuda:0",
               **overrides) -> Tuple[GaussianAsset, Dict]:
    """Train one static asset from a capture bundle and save it (plan §5.2).

    Args:
        bundle: loaded capture directory (background or one object).
        component: 'background' or 'object' (the bundle's own
            'objects/<name>' component string is also accepted).
        out_path: where the ``GaussianAsset`` .pt is written.
        iters: training iterations (default: 7000 background, 5000 object).
        overrides: any key of ``_FIT_DEFAULTS`` (unknown keys raise).

    Returns (asset, metrics): metrics carries held-out component-region PSNR,
    silhouette IoU, n_gaussians and final loss terms.
    """
    if component == "background":
        kind = "background"
    elif component == "object" or component.startswith("objects/"):
        kind = "object"
    else:
        raise ValueError(
            f"component={component!r}: expected 'background', 'object', or an "
            f"'objects/<name>' bundle component string")
    cfg = dict(_FIT_DEFAULTS)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"fit_static: unknown override(s) {sorted(unknown)}; "
                        f"allowed: {sorted(cfg)}")
    cfg.update(overrides)
    iters = int(iters) if iters is not None else DEFAULT_ITERS[kind]
    voxel_m = float(cfg["voxel_m"] or DEFAULT_VOXEL_M[kind])
    refine_stop = int(cfg["refine_stop"] or iters // 2)
    dev = torch.device(device)
    torch.manual_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))

    tf = bundle.transforms
    if kind == "object":
        assert "object_geom_ids" in tf and "body_pose" in tf, (
            f"object capture '{bundle.directory}' transforms lack "
            f"object_geom_ids/body_pose — re-run capture_assets.py")
        gids: Optional[List[int]] = [int(g) for g in tf["object_geom_ids"]]
    else:
        gids = None  # background: every rendered geom is the component
    # silhouette term: objects always; background only in robot-masked mode
    masked_bg = kind == "background" and bool(tf.get("masks_dir"))
    use_sil = kind == "object" or masked_bg

    size = bundle.image_size
    gids_np = None if gids is None else np.asarray(gids, dtype=np.int64)

    # per-view CPU tensors (moved to device per iteration) + device cameras
    rgb_t, comp_t, excl_t = [], [], []
    for v in bundle.views:
        rgb_t.append(torch.from_numpy(np.ascontiguousarray(v.rgb)))
        comp = (v.seg >= 0) if gids_np is None else np.isin(v.seg, gids_np)
        excl = None
        if v.mask is not None:
            comp = comp & ~v.mask
            excl = torch.from_numpy(np.ascontiguousarray(v.mask))
        comp_t.append(torch.from_numpy(comp))
        excl_t.append(excl)
    viewmats = torch.as_tensor(
        np.stack([c2w_from_w2c(v.c2w) for v in bundle.views]),
        dtype=torch.float32, device=dev)                     # rigid inverse
    Ks = torch.as_tensor(np.stack([v.K for v in bundle.views]),
                         dtype=torch.float32, device=dev)

    n_views = len(bundle.views)
    every = int(cfg["holdout_every"])
    holdout = [i for i in range(n_views) if i % every == 0]
    train = [i for i in range(n_views)
             if i % every != 0 and bool(comp_t[i].any())]
    assert train, f"no training views with component pixels ({bundle.directory})"
    holdout = [i for i in holdout if bool(comp_t[i].any())]
    assert holdout, f"no held-out views with component pixels ({bundle.directory})"

    # ── init + parameters (strategy naming: means/quats/scales/opacities) ──
    # init sees ONLY the training split: held-out views must not leak into
    # the depth init (they seed means/colors — the held-out metrics would be
    # evaluated on views that shaped the model).
    init = init_from_depth(bundle, gids, voxel_m,
                           max_views=cfg["max_init_views"],
                           view_indices=train)
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

    strategy = DefaultStrategy(
        prune_opa=float(cfg["prune_opa"]),
        grow_grad2d=float(cfg["grow_grad2d"]),
        refine_start_iter=int(cfg["refine_start"]),
        refine_stop_iter=refine_stop,
        reset_every=int(cfg["reset_every"]),
        refine_every=int(cfg["refine_every"]))
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=extent)

    def _colors():
        return torch.cat([params["sh0"][:, None, :], params["shN"]], dim=1)

    def _raster(view_idx: int):
        return rasterize_views(
            params["means"], params["quats"], params["scales"].exp(),
            params["opacities"].sigmoid(), _colors(),
            viewmats[view_idx:view_idx + 1], Ks[view_idx:view_idx + 1],
            size, size, SH_DEGREE_STATIC)

    # ── loop ────────────────────────────────────────────────────────────────
    recent = {k: deque(maxlen=100) for k in ("total", "l1", "ssim", "sil")}
    for step in range(iters):
        vi = train[int(rng.integers(len(train)))]
        gt = rgb_t[vi].to(dev).float() / 255.0
        m = comp_t[vi].to(dev)
        img, alpha, info = _raster(vi)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        img0, a0 = img[0], alpha[0, :, :, 0]

        l1 = (img0 - gt).abs()[m].mean()
        sloss = 1.0 - ssim_map(img0.clamp(0, 1), gt)[m].mean()
        loss = l1 + float(cfg["ssim_weight"]) * sloss
        sil = img0.new_zeros(())
        if use_sil:
            resid = (a0 - m.float()).abs()
            if excl_t[vi] is not None:   # masked mode: robot pixels excluded
                resid = resid[~excl_t[vi].to(dev)]
            sil = resid.mean()
            loss = loss + float(cfg["silhouette_weight"]) * sil
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    packed=False)
        # 1.5.3 DefaultStrategy never fires its own reset (precedence bug,
        # module docstring); apply the plan §5.2 reset schedule manually.
        if 0 < step <= refine_stop and step % int(cfg["reset_every"]) == 0:
            reset_opa(params, optimizers, state,
                      value=float(cfg["reset_opacity"]))
        for k, v in (("total", loss), ("l1", l1), ("ssim", sloss), ("sil", sil)):
            recent[k].append(float(v.detach()))
        if cfg["log_every"] and (step + 1) % int(cfg["log_every"]) == 0:
            print(f"[trainer] {bundle.component} step {step + 1}/{iters}: "
                  f"loss {np.mean(recent['total']):.4f} "
                  f"n={params['means'].shape[0]}")

    # ── held-out metrics ────────────────────────────────────────────────────
    psnrs, ious = [], []
    with torch.no_grad():
        for vi in holdout:
            img, alpha, _ = _raster(vi)
            gt = rgb_t[vi].to(dev).float() / 255.0
            m = comp_t[vi].to(dev)
            psnrs.append(masked_psnr(img[0], gt, m))
            iou = silhouette_iou(alpha[0, :, :, 0], m)
            if iou is not None:
                ious.append(iou)
    metrics = {
        "psnr_component": float(np.mean(psnrs)),
        "psnr_component_min": float(np.min(psnrs)),
        "silhouette_iou": float(np.mean(ious)) if ious else None,
        "silhouette_iou_min": float(np.min(ious)) if ious else None,
        "n_gaussians": int(params["means"].shape[0]),
        "n_views": int(n_views),
        "n_train_views": len(train),
        "n_holdout_views": len(holdout),
        "scene_extent_m": float(extent),
        "final_losses": {k: float(np.mean(v)) for k, v in recent.items()},
    }

    # ── frame conversion + save (plan §5.2 / §6.1) ─────────────────────────
    means_w = params["means"].detach().cpu()
    quats_w = quat_normalize(params["quats"].detach().cpu())
    if kind == "background":
        frame = "world"
        p_cap = [0.0, 0.0, 0.0]
        q_cap = [1.0, 0.0, 0.0, 0.0]
        means_s, quats_s = means_w, quats_w
    else:
        frame = "body"
        bp = tf["body_pose"]
        p_t = torch.as_tensor(bp["p"], dtype=torch.float32)
        q_t = quat_normalize(torch.as_tensor(bp["q_wxyz"], dtype=torch.float32))
        R = quat_to_R(q_t)                       # (3,3)
        means_s = (means_w - p_t) @ R            # R^T x  ==  x @ R
        quats_s = quat_mul(quat_conj(q_t).expand_as(quats_w), quats_w)
        p_cap = [float(x) for x in bp["p"]]
        q_cap = [float(x) for x in bp["q_wxyz"]]

    train_args = {
        "iters": iters, "kind": kind, "voxel_m": voxel_m,
        "refine_stop": refine_stop, "use_silhouette": bool(use_sil),
        **{k: cfg[k] for k in ("seed", "holdout_every", "ssim_weight",
                               "silhouette_weight", "refine_start",
                               "refine_every", "reset_every", "reset_opacity",
                               "prune_opa", "grow_grad2d", "max_init_views",
                               "lrs")},
    }
    meta = {
        "frame": frame, "p_capture": p_cap, "q_capture": q_cap,
        "task": bundle.task, "component": bundle.component,
        "model_xml_sha1": bundle.model_xml_sha1,
        "versions": {"gsplat": str(gsplat.__version__),
                     "torch": str(torch.__version__)},
        "train_args": train_args, "metrics": metrics,
    }
    if kind == "object":
        meta["joint_name"] = tf.get("joint_name")
        meta["body_name"] = tf.get("body_name")

    asset = GaussianAsset(
        means=means_s.contiguous(),
        quats=quats_s.contiguous(),
        log_scales=params["scales"].detach().cpu().contiguous(),
        opacity_logits=params["opacities"].detach().cpu().contiguous(),
        sh_dc=params["sh0"].detach().cpu().contiguous(),
        sh_rest=params["shN"].detach().cpu().contiguous(),
        conventions=dict(EXPECTED_CONVENTIONS),
        meta=meta)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    digest = asset.save(out_path)
    print(f"[trainer] {bundle.component}: saved {out_path} "
          f"({metrics['n_gaussians']} gaussians, sha1 {digest[:12]}…, "
          f"PSNR {metrics['psnr_component']:.2f}, "
          f"IoU {metrics['silhouette_iou']})")
    return asset, metrics

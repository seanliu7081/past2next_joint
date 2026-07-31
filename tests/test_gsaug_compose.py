"""GPU tests for the GS compositing math (plan §6.5 T1–T6, M5 gate).

Synthetic scenes only — no env, no sim: components are built directly as
``PosedComponent`` / ``WorldGaussians`` and rasterized through gsplat following
``oat.gsaug.compose``'s exact conventions (OpenCV w2c viewmats via
``cameras.c2w_from_w2c``, Ks from ``fovy_to_K``, one rasterization call,
``sh_degree`` = concatenated max). ``GSCompositeRenderer`` construction failure
modes that need no env (missing manifest, non-PASSing facts) are covered with
temp files.

T1 rigid-transform consistency, T2 z-rotation SH invariance, T2b full-SO(3)
deg-3 SH invariance (R7), T3 SO(3) l=1 SH invariance, T4 occlusion (concat ==
merged; image-space compositing fails), T5 camera-math round-trip (<= 0.5 px),
T6 repeatability (uint8 tol 1).
"""

import json
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="gsplat rasterization needs CUDA")

if torch.cuda.is_available():
    import gsplat  # noqa: F401  (compiled extension; import gated on CUDA)

    from oat.gsaug.compose import GSCompositeRenderer

from oat.gsaug import cameras as cam
from oat.gsaug.components import PosedComponent, WorldGaussians

DEVICE = "cuda:0"
SIZE = 128
FOVY = 45.0
IDENT_Q = np.array([1.0, 0.0, 0.0, 0.0])


# ── helpers ──────────────────────────────────────────────────────────────────

def rodrigues(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * Kx + (1.0 - math.cos(angle)) * Kx @ Kx


def aa_quat(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def rigid4(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def K_matrix() -> np.ndarray:
    return cam.fovy_to_K(FOVY, SIZE, SIZE)


def rasterize(world: WorldGaussians, c2ws, sh_degree=None):
    """One gsplat call, compose.py conventions: float32 OpenCV w2c viewmats,
    shared K, square image; returns (img (C,H,W,3) float 0..1-ish, alpha)."""
    viewmats = torch.as_tensor(
        np.stack([cam.c2w_from_w2c(np.asarray(c)) for c in c2ws]),
        dtype=torch.float32, device=DEVICE)
    Ks = torch.as_tensor(np.stack([K_matrix()] * len(c2ws)),
                         dtype=torch.float32, device=DEVICE)
    img, alpha, _meta = gsplat.rasterization(
        means=world.means, quats=world.quats, scales=world.scales,
        opacities=world.opacities, colors=world.sh,
        viewmats=viewmats, Ks=Ks, width=SIZE, height=SIZE,
        sh_degree=world.sh_degree if sh_degree is None else sh_degree,
        render_mode="RGB")
    return img, alpha


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """compose.render's exact float->uint8 recipe."""
    return img.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def make_component(mode, deg, n=3, seed=0, sh_scale=0.3) -> PosedComponent:
    g = torch.Generator().manual_seed(seed)
    return PosedComponent(
        f"synthetic_{mode}",
        means_l=torch.randn(n, 3, generator=g) * 0.25,
        quats_l=torch.randn(n, 4, generator=g),
        log_scales=torch.rand(n, 3, generator=g) * 1.0 - 3.2,  # anisotropic
        opacity_logits=torch.full((n,), 1.5),
        sh=torch.randn(n, (deg + 1) ** 2, 3, generator=g) * sh_scale,
        sh_rot_mode=mode,
        p_capture=torch.zeros(3),
        q_capture=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    ).to(DEVICE)


def solid_gaussians(means, color, scale, opac, deg) -> WorldGaussians:
    """World-frame isotropic constant-color gaussians (DC-only appearance)."""
    means = torch.as_tensor(np.asarray(means, dtype=np.float64),
                            dtype=torch.float32, device=DEVICE).reshape(-1, 3)
    n = means.shape[0]
    K = (deg + 1) ** 2
    sh = torch.zeros(n, K, 3, device=DEVICE)
    # gsplat colors = SH(dir) + 0.5, so DC = (rgb - 0.5) / Y00
    sh[:, 0, :] = (torch.as_tensor(color, dtype=torch.float32,
                                   device=DEVICE) - 0.5) / 0.2820947917738781
    return WorldGaussians(
        means=means,
        quats=torch.tensor([[1.0, 0, 0, 0]] * n, device=DEVICE),
        scales=torch.full((n, 3), float(scale), device=DEVICE),
        opacities=torch.full((n,), float(opac), device=DEVICE),
        sh=sh,
    )


C2W0 = cam.lookat_c2w(np.array([1.6, -1.1, 1.3]), np.zeros(3))


# ── T1: rigid-transform consistency ──────────────────────────────────────────

def test_t1_rigid_transform_consistency():
    """render(posed(T), C) == render(posed(I), T^-1 C) — the world transform
    and the camera transform must be the same group action (PSNR >= 45)."""
    comp = make_component("so3_deg1", 1, n=3, seed=0)
    axis, ang = [0.3, -0.5, 0.8], 0.9
    R, t = rodrigues(axis, ang), [0.35, -0.2, 0.45]
    T = rigid4(R, t)

    img_t, _ = rasterize(comp.posed(np.asarray(t), aa_quat(axis, ang)), [C2W0])
    img_0, _ = rasterize(comp.posed(np.zeros(3), IDENT_Q),
                         [np.linalg.inv(T) @ C2W0])
    p = psnr(img_t[0], img_0[0])
    assert p >= 45.0, f"T1 rigid-transform consistency PSNR {p:.1f} < 45"


# ── T2: SH z-rotation invariance ─────────────────────────────────────────────

def test_t2_sh_z_rotation_invariance():
    """Anisotropic deg-3 SH: rotating component AND camera by the same R_z
    about the same center leaves the image unchanged — pins so3_deg3's
    closed-form z fast-path sign conventions end-to-end through the real
    rasterizer (G5)."""
    comp = make_component("so3_deg3", 3, n=4, seed=1, sh_scale=0.35)
    theta = math.radians(50.0)
    Rz4 = rigid4(rodrigues([0, 0, 1], theta))

    img_ref, _ = rasterize(comp.posed(np.zeros(3), IDENT_Q), [C2W0])
    img_rot, _ = rasterize(comp.posed(np.zeros(3), aa_quat([0, 0, 1], theta)),
                           [Rz4 @ C2W0])
    p = psnr(img_rot[0], img_ref[0])
    assert p >= 45.0, f"T2 z-rotation invariance PSNR {p:.1f} < 45"


def test_t2b_sh_so3_rotation_invariance_deg3():
    """T2b (R7): deg-3 'so3_deg3' component under a random FULL SO(3) rotation
    about its pose center with the co-rotated camera: image unchanged. This is
    the end-to-end proof that the exact projection path (rotate_sh_so3) is
    correct in gsplat's own basis — the tumbled-object case that used to
    assert now renders exactly (G5)."""
    comp = make_component("so3_deg3", 3, n=4, seed=4, sh_scale=0.35)
    axis, ang = [0.5, -0.3, 0.6], 1.0   # mixes z with x/y — off the fast path
    R4 = rigid4(rodrigues(axis, ang))

    img_ref, _ = rasterize(comp.posed(np.zeros(3), IDENT_Q), [C2W0])
    img_rot, _ = rasterize(comp.posed(np.zeros(3), aa_quat(axis, ang)),
                           [R4 @ C2W0])
    p = psnr(img_rot[0], img_ref[0])
    assert p >= 45.0, f"T2b full-SO(3) deg-3 invariance PSNR {p:.1f} < 45"


# ── T3: SH SO(3) l=1 invariance ──────────────────────────────────────────────

def test_t3_sh_so3_rotation_invariance():
    """so3_deg1 component under a random R (about the component's pose origin)
    with the co-rotated camera: image unchanged. Catches any sign/permutation
    error in rotate_sh_l1 against the real gsplat SH basis."""
    comp = make_component("so3_deg1", 1, n=4, seed=2, sh_scale=0.4)
    axis, ang = [0.2, 0.7, -0.4], 1.1   # mixes z with x/y
    R4 = rigid4(rodrigues(axis, ang))

    img_ref, _ = rasterize(comp.posed(np.zeros(3), IDENT_Q), [C2W0])
    img_rot, _ = rasterize(comp.posed(np.zeros(3), aa_quat(axis, ang)),
                           [R4 @ C2W0])
    p = psnr(img_rot[0], img_ref[0])
    assert p >= 45.0, f"T3 SO(3) l=1 invariance PSNR {p:.1f} < 45"


# ── T4: occlusion — concat-then-rasterize vs image-space compositing ─────────

def test_t4_occlusion_concat_vs_composite():
    # A: two red gaussians sandwiching B's blue one along the view axis; every
    # per-component 2D composite order is wrong somewhere (G2).
    c2w = cam.lookat_c2w(np.array([0.0, 0.0, 3.0]), np.zeros(3))
    A = solid_gaussians([[0, 0, 0.6], [0, 0, -0.6]],
                        color=[0.9, 0.15, 0.1], scale=0.12, opac=0.7, deg=1)
    B = solid_gaussians([[0, 0, 0.0]],
                        color=[0.1, 0.15, 0.9], scale=0.12, opac=0.7, deg=3)

    joint = WorldGaussians.concat([A, B])
    assert joint.sh_degree == 3
    img_joint, _ = rasterize(joint, [c2w])

    # reference merged set: A's SH zero-padded to deg 3 by hand, same order
    pad = torch.zeros(2, 16 - 4, 3, device=DEVICE)
    merged = WorldGaussians(
        means=torch.cat([A.means, B.means]),
        quats=torch.cat([A.quats, B.quats]),
        scales=torch.cat([A.scales, B.scales]),
        opacities=torch.cat([A.opacities, B.opacities]),
        sh=torch.cat([torch.cat([A.sh, pad], dim=1), B.sh]),
    )
    img_merged, _ = rasterize(merged, [c2w])
    diff = float((img_joint - img_merged).abs().max())
    assert diff <= 2.0 / 255.0, f"concat vs merged max diff {diff:.4f}"

    # NEGATIVE control: alpha-compositing separate renders breaks occlusion.
    img_a, alpha_a = rasterize(A, [c2w])
    img_b, alpha_b = rasterize(B, [c2w])
    over_ab = img_a + (1.0 - alpha_a) * img_b   # A over B (premultiplied over)
    over_ba = img_b + (1.0 - alpha_b) * img_a
    d_ab = float((over_ab - img_joint).abs().max())
    d_ba = float((over_ba - img_joint).abs().max())
    assert d_ab > 10.0 / 255.0, f"A-over-B composite too close ({d_ab:.4f})"
    assert d_ba > 10.0 / 255.0, f"B-over-A composite too close ({d_ba:.4f})"


# ── T5: camera math round-trip ───────────────────────────────────────────────

def test_t5_camera_math_roundtrip():
    """A gaussian at a known world point lands where project() says, to
    <= 0.5 px (blob centroid, pixel centers at index + 0.5)."""
    point = np.array([0.15, -0.08, 0.1])
    c2w = cam.lookat_c2w(np.array([0.9, 0.6, 0.8]), np.zeros(3))
    w2c = cam.c2w_from_w2c(c2w)

    world = solid_gaussians([point], color=[0.95, 0.95, 0.95],
                            scale=0.008, opac=0.97, deg=0)
    _img, alpha = rasterize(world, [c2w])
    a = alpha[0, :, :, 0].double().cpu().numpy()
    assert a.max() > 0.2, "blob not visible — camera/point setup broken"

    uv = cam.project(K_matrix(), w2c, point)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    m = a.sum()
    cu = float((a * (xs + 0.5)).sum() / m)
    cv = float((a * (ys + 0.5)).sum() / m)
    err = math.hypot(cu - uv[0], cv - uv[1])
    assert err <= 0.5, (
        f"blob centroid ({cu:.2f}, {cv:.2f}) vs projected ({uv[0]:.2f}, "
        f"{uv[1]:.2f}): {err:.3f} px > 0.5")


# ── T6: repeatability ────────────────────────────────────────────────────────

def test_t6_repeatability_uint8_tol1():
    comp = make_component("so3_deg3", 3, n=4, seed=3)
    world = comp.posed(np.zeros(3), IDENT_Q)
    img1, _ = rasterize(world, [C2W0])
    img2, _ = rasterize(comp.posed(np.zeros(3), IDENT_Q), [C2W0])
    d = np.abs(to_uint8(img1).astype(np.int16)
               - to_uint8(img2).astype(np.int16)).max()
    assert d <= 1, f"repeat render uint8 max diff {d} > 1"


# ── GSCompositeRenderer ctor failure modes (no env needed) ───────────────────

def _write_facts(path, ok=True):
    with open(path, "w") as f:
        json.dump({"pass": bool(ok),
                   "F1": {"flip_name": "gl_to_cv"},
                   "F2": {"gsplat_flip_ud": True}}, f)
    return str(path)


CAMS = OrderedDict([("agentview_rgb", "agentview")])


def test_ctor_missing_assets_dir(tmp_path):
    facts = _write_facts(tmp_path / "facts.json", ok=True)
    with pytest.raises(FileNotFoundError, match="manifest"):
        GSCompositeRenderer(str(tmp_path / "no_such_task"), CAMS, SIZE,
                            facts_path=facts, device=DEVICE)


def test_ctor_refuses_failing_facts(tmp_path):
    facts = _write_facts(tmp_path / "facts.json", ok=False)
    with pytest.raises(RuntimeError, match="PASS"):
        GSCompositeRenderer(str(tmp_path / "no_such_task"), CAMS, SIZE,
                            facts_path=facts, device=DEVICE)


def test_ctor_missing_facts_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="probe_render_facts"):
        GSCompositeRenderer(str(tmp_path / "no_such_task"), CAMS, SIZE,
                            facts_path=str(tmp_path / "no_facts.json"),
                            device=DEVICE)

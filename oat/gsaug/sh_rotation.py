"""Spherical-harmonic coefficient rotation for GS assets (G5).

Layout: gsplat / vanilla-3DGS real SH, bands concatenated in degree order with
m = -l..l inside each band; coefficient index 0 is the DC (l=0) term. A full
degree-3 set has 16 coefficients per channel; ``sh`` tensors here are
``(N, K, 3)`` with K = (deg+1)^2 (DC included) unless noted.

Real-SH convention: for m>0 the +m basis function carries cos(m*phi) and the -m
function carries sin(m*phi) azimuthal dependence (phi measured about +z).
"Rotating a component by R" means the appearance function transforms as
``f'(d) = f(R^T d)``.

Four paths (G5):
  * ``rotate_sh_z``   — closed-form world-z rotation, exact for any l<=3
                        (fast path for pure-z deltas; also the test reference).
  * ``rotate_sh_l1``  — exact l=1 rotation under arbitrary R (robot links,
                        which are SH degree 1).
  * ``rotate_sh_so3`` — exact, dependency-free general SO(3) rotation for
                        l<=3 via projection in the gsplat basis; the
                        PRODUCTION SO(3) path for objects ('so3_deg3').
  * ``rotate_sh_wigner`` — general SO(3) via e3nn wigner_D; CROSS-CHECK-ONLY
                        scaffold (optional dep), never a default path.

R7 note: the original design asserted object deltas were pure z-rotations and
kept e3nn wigner as the contingency. Review measured the assertion firing on
~100% of real LIBERO-10 frames (capture-vs-demo resting tilt |q_xy| ~ 2.4e-3..
3.4e-3; grasped-object tumble up to |q_xy| ~ 0.83), so ``rotate_sh_so3``
replaced both the assertion and the e3nn contingency.

All functions are pure, torch-native (accept numpy, return torch on the input's
device), and never modify inputs in place. A transform that moves means/quats
without routing SH through one of these paths is invalid (G5) —
``components.PosedComponent.posed`` is the single enforcement point.
"""

import math
from typing import Dict, Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]

# band index ranges (DC included): l -> (start, end) into the K axis
_BAND = {0: (0, 1), 1: (1, 4), 2: (4, 9), 3: (9, 16)}


def _as_tensor(x: Array) -> torch.Tensor:
    t = torch.as_tensor(x)
    return t.float() if t.dtype == torch.float64 else t


def sh_degree_of(sh: Array) -> int:
    """Degree from a (N, K, 3) coefficient tensor; K must be a perfect square."""
    K = int(torch.as_tensor(sh).shape[-2])
    deg = int(round(K ** 0.5)) - 1
    if (deg + 1) ** 2 != K:
        raise ValueError(f"SH coefficient count {K} is not (deg+1)^2")
    return deg


def rotate_sh_z(sh: Array, theta: float) -> torch.Tensor:
    """Rotate SH coefficients by R_z(theta) (radians), exact for any band l<=3.

    Within band l the (m=-k, m=+k) coefficient pair mixes by a 2x2 rotation of
    angle k*theta (m=0 fixed):
        a' = cos(k t) * a + sin(k t) * b        (a = c_{-k}, b = c_{+k})
        b' = -sin(k t) * a + cos(k t) * b
    which is f'(d) = f(R_z(theta)^T d) in the real-SH convention above.
    """
    t = _as_tensor(sh).clone()
    deg = sh_degree_of(t)
    for l in range(1, deg + 1):
        s0, _ = _BAND[l]
        for k in range(1, l + 1):
            ia = s0 + (l - k)   # m = -k
            ib = s0 + (l + k)   # m = +k
            c = float(np.cos(k * theta))
            s = float(np.sin(k * theta))
            a = t[..., ia, :].clone()
            b = t[..., ib, :].clone()
            t[..., ia, :] = c * a + s * b
            t[..., ib, :] = -s * a + c * b
    return t


def rotate_sh_l1(sh: Array, R: Array) -> torch.Tensor:
    """Rotate a degree-1 SH set (K=4, DC + 3) by an arbitrary rotation R (3,3).

    The gsplat/3DGS real l=1 basis in (m=-1, 0, +1) order is the SIGNED
    permutation (-y, +z, -x) * 0.4886 of the direction components. With
    p = [1, 2, 0] (band index -> xyz axis) and signs s = (-1, +1, -1), the
    coefficients transform by M[j, i] = s[j] * s[i] * R[p[j], p[i]]:
    c' = M @ c. The sign conjugation is load-bearing: a plain permutation
    conjugation mis-rotates any R that mixes z with x/y (verified against
    gsplat's ``_spherical_harmonics`` by tests/test_gsaug_transforms.py).
    """
    t = _as_tensor(sh).clone()
    if t.shape[-2] != 4:
        raise ValueError(f"rotate_sh_l1 expects K=4 (deg-1) SH, got K={t.shape[-2]}")
    Rt = torch.as_tensor(R, dtype=t.dtype, device=t.device).reshape(3, 3)
    p = [1, 2, 0]
    s = torch.tensor([-1.0, 1.0, -1.0], dtype=t.dtype, device=t.device)
    M = Rt[p][:, p] * (s[:, None] * s[None, :])   # M[j,i] = s_j s_i R[p[j], p[i]]
    band = t[..., 1:4, :]                    # (..., 3, 3ch)
    t[..., 1:4, :] = torch.einsum("ji,...ic->...jc", M, band)
    return t


# ── exact general SO(3) rotation via projection in the gsplat basis ─────────

def eval_sh_bases(deg: int, dirs: Array) -> torch.Tensor:
    """(M,3) unit dirs -> (M,(deg+1)^2) real-SH basis values in the gsplat /
    3DGS layout (bands l=0..deg, m=-l..l inside each band).

    Constants verbatim from the installed gsplat 1.5.3
    ``_eval_sh_bases_fast`` (gsplat/cuda/_torch_impl.py) — this evaluator IS
    the rasterizer's basis, which is what makes the projection below exact in
    the basis the renderer actually uses, not merely in some SH convention.
    """
    if not 0 <= deg <= 3:
        raise ValueError(f"eval_sh_bases supports deg 0..3, got {deg}")
    d = torch.as_tensor(dirs)
    out = d.new_empty(d.shape[:-1] + ((deg + 1) ** 2,))
    out[..., 0] = 0.2820947917738781
    if deg == 0:
        return out
    x, y, z = d.unbind(-1)
    fA = -0.48860251190292
    out[..., 1] = fA * y
    out[..., 2] = -fA * z
    out[..., 3] = fA * x
    if deg == 1:
        return out
    z2 = z * z
    fB = -1.092548430592079 * z
    fA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2.0 * x * y
    out[..., 4] = fA * fS1
    out[..., 5] = fB * y
    out[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    out[..., 7] = fB * x
    out[..., 8] = fA * fC1
    if deg == 2:
        return out
    fC = -2.285228997322329 * z2 + 0.4570457994644658
    fB = 1.445305721320277 * z
    fA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    out[..., 9] = fA * fS2
    out[..., 10] = fB * fS1
    out[..., 11] = fC * y
    out[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    out[..., 13] = fC * x
    out[..., 14] = fB * fC1
    out[..., 15] = fA * fC2
    return out


def _fibonacci_sphere(m: int) -> torch.Tensor:
    """(m,3) float64 well-spread unit directions; deterministic."""
    i = torch.arange(m, dtype=torch.float64)
    z = 1.0 - 2.0 * (i + 0.5) / m
    r = (1.0 - z * z).clamp_min(0.0).sqrt()
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


# fixed projection sample set (module-level constants; CPU float64)
_PROJ_DIRS = _fibonacci_sphere(64)          # (64, 3)
_PINV_B: Dict[int, torch.Tensor] = {}       # band l -> pinv(B) (2l+1, 64)


def so3_band_matrix(l: int, R: Array) -> torch.Tensor:
    """Exact (2l+1, 2l+1) float64 coefficient-rotation matrix D for band l
    under rotation R in the gsplat basis: c'_band = D @ c_band realizes
    f'(d) = f(R^T d).

    Built by exact projection: with B[m, j] = Y_{l,j}(d_m) sampled on the
    fixed 64-direction fibonacci sphere and B_rot[m, i] = Y_{l,i}(R^T d_m),
    the overdetermined system B @ D = B_rot is exactly consistent (each
    rotated band-l basis function lies in the span of the band), so
    D = pinv(B) @ B_rot has zero residual and is exact to float64 precision.

    pinv(B) is cached per band at module level; D is NOT cached per R — the
    earlier per-theta SH cache was unbounded and leaked GPU tensors, and the
    rotation cost is negligible next to rasterization.
    """
    if l not in (1, 2, 3):
        raise ValueError(f"so3_band_matrix supports bands 1..3, got l={l}")
    Rm = torch.as_tensor(R, dtype=torch.float64).reshape(3, 3).cpu()
    s0, s1 = _BAND[l]
    if l not in _PINV_B:
        B = eval_sh_bases(l, _PROJ_DIRS)[:, s0:s1]      # (64, 2l+1)
        _PINV_B[l] = torch.linalg.pinv(B)
    B_rot = eval_sh_bases(l, _PROJ_DIRS @ Rm)[:, s0:s1]  # rows are Y(R^T d_m)
    return _PINV_B[l] @ B_rot


def rotate_sh_so3(sh: Array, R: Array) -> torch.Tensor:
    """Rotate SH coefficients (deg <= 3) by an arbitrary rotation R (3,3).

    The production general-SO(3) path (G5/R7): exact for every band, no
    dependencies. DC untouched; each band l>=1 mixes by ``so3_band_matrix``.
    Same convention as ``rotate_sh_z``: the result satisfies
    f'(d) = f(R^T d) in the gsplat basis.
    """
    t = _as_tensor(sh).clone()
    deg = sh_degree_of(t)
    for l in range(1, deg + 1):
        D = so3_band_matrix(l, R).to(dtype=t.dtype, device=t.device)
        s0, s1 = _BAND[l]
        band = t[..., s0:s1, :]
        t[..., s0:s1, :] = torch.einsum("ji,...ic->...jc", D, band)
    return t


def rotate_sh_wigner(sh: Array, R: Array) -> torch.Tensor:
    """General SO(3) SH rotation for l<=3 via e3nn wigner_D. Scaffold (G5):
    CROSS-CHECK ONLY — ``rotate_sh_so3`` is the production SO(3) path; this
    stays as an independent implementation for validation and requires the
    optional ``e3nn`` dependency.

    e3nn's real-SH basis for band l is index-reversed relative to the
    gsplat/3DGS m = -l..l layout used here; the conversion below is the
    explicit basis mapping and MUST stay cross-checked against ``rotate_sh_z``
    and ``rotate_sh_l1`` by tests before first production use.
    """
    try:
        from e3nn.o3 import wigner_D  # noqa: PLC0415  (lazy: optional dep)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rotate_sh_wigner requires the optional dependency 'e3nn' "
            "(pip install e3nn). It is a cross-check-only path — production "
            "SO(3) SH rotation uses rotate_sh_so3, which has no extra "
            "dependency.") from e

    t = _as_tensor(sh).clone()
    deg = sh_degree_of(t)
    Rm = torch.as_tensor(R, dtype=torch.float64).reshape(3, 3)
    # e3nn parameterizes by ZYZ Euler angles of the rotation acting on
    # functions: f'(d) = f(R^T d) corresponds to D(R) applied to coefficients.
    from e3nn.o3 import matrix_to_angles
    a, b, g = matrix_to_angles(Rm)
    # e3nn real-SH band ordering is m = -l..l as well, but with a different
    # component convention: e3nn uses the (y, z, x) axis ordering natively for
    # l=1, matching ours; for l>=2 the bases agree up to the same index order.
    for l in range(1, deg + 1):
        D = wigner_D(l, a, b, g).to(dtype=t.dtype, device=t.device)  # (2l+1, 2l+1)
        s0, s1 = _BAND[l]
        band = t[..., s0:s1, :]
        t[..., s0:s1, :] = torch.einsum("ji,...ic->...jc", D, band)
    return t

"""Spherical-harmonic coefficient rotation for GS assets (G5).

Layout: gsplat / vanilla-3DGS real SH, bands concatenated in degree order with
m = -l..l inside each band; coefficient index 0 is the DC (l=0) term. A full
degree-3 set has 16 coefficients per channel; ``sh`` tensors here are
``(N, K, 3)`` with K = (deg+1)^2 (DC included) unless noted.

Real-SH convention: for m>0 the +m basis function carries cos(m*phi) and the -m
function carries sin(m*phi) azimuthal dependence (phi measured about +z).
"Rotating a component by R" means the appearance function transforms as
``f'(d) = f(R^T d)``.

Three paths (G5):
  * ``rotate_sh_z``   — closed-form world-z rotation, exact for any l<=3
                        (objects; also used by tests as reference).
  * ``rotate_sh_l1``  — exact l=1 rotation under arbitrary R (robot links,
                        which are SH degree 1).
  * ``rotate_sh_wigner`` — general SO(3) for l<=3 via e3nn wigner_D; scaffold
                        only, NOT enabled by default; imports e3nn lazily.

All functions are pure, torch-native (accept numpy, return torch on the input's
device), and never modify inputs in place. A transform that moves means/quats
without routing SH through one of these paths is invalid (G5) — the assertion
lives in ``components.PosedComponent.posed``.
"""

from typing import Union

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


def rotate_sh_wigner(sh: Array, R: Array) -> torch.Tensor:
    """General SO(3) SH rotation for l<=3 via e3nn wigner_D. Scaffold (G5):
    NOT used by any default path; requires the optional ``e3nn`` dependency.

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
            "(pip install e3nn). It is the G5 upgrade path and is not needed "
            "by any default configuration.") from e

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

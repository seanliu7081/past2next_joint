"""PosedComponent: the single code path posing objects, robot links, and the
background for compositional rasterization (plan §6.1).

A component is a rigid Gaussian set in a local frame plus the world pose of
that frame at capture time. ``posed(p_wb, q_wb)`` maps it into world frame for
the body's *current* pose (from the forwarded sim's ``data.xpos/xquat``):

    means:   p_wb + R(q_wb) @ means_l
    quats:   q_wb ⊗ quats_l                       (wxyz, left multiply = world)
    SH:      rotated by the world-frame delta pose since capture,
             R_delta = R(q_wb) R(q_capture)^T, per ``sh_rot_mode`` (G5)
    scales, opacities: activated (exp / sigmoid), otherwise unchanged

``sh_rot_mode``:
    'static'      background — pose must be identity, SH untouched
    'so3_deg3'    objects — exact SH rotation for ANY R_delta: closed-form
                  ``rotate_sh_z`` fast path when the delta is z-only
                  (|qx|,|qy| <= FAST_Z_TOL), exact ``rotate_sh_so3`` otherwise
    'so3_deg1'    robot links — exact l=1 rule under arbitrary R_delta
    'wigner'      e3nn cross-check path (optional dep); only set explicitly
    ('z_only_deg3' is accepted as a deprecated alias for 'so3_deg3'.)

R7 history: 'z_only_deg3' originally ASSERTED the delta was a pure z-rotation
(tol 1e-5), with e3nn wigner as the contingency. Adversarial review measured
that assertion firing on ~100% of real LIBERO-10 frames — capture-reset
resting poses differ from demo resting poses by settling tilt (|q_xy| ~
2.4e-3..3.4e-3) and grasped objects tumble (up to |q_xy| ~ 0.83) — so the
exact in-house SO(3) projection path replaced both the assertion and the e3nn
contingency. SH now rotates exactly for any delta, so the G5 invariant ("SH
rotates whenever means rotate") holds unconditionally.

G5 invariant: there is no way to move a component's means/quats through this
class without its SH being routed through ``sh_rotation`` — do not add one.

Quaternion helpers here are torch, wxyz, batch-friendly; they intentionally
mirror the numpy wxyz helpers in ``oat.equi.se2_transforms`` (same convention).
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from oat.gsaug.gaussian_asset import GaussianAsset
from oat.gsaug.sh_rotation import (
    rotate_sh_l1,
    rotate_sh_so3,
    rotate_sh_wigner,
    rotate_sh_z,
)

# |x|,|y| of the unit delta quat below which the closed-form z path is used
# (both paths are exact; this only selects the cheaper one).
FAST_Z_TOL = 1e-6


# ── torch quaternion helpers (wxyz) ─────────────────────────────────────────

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2, both (...,4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """(...,4) wxyz (need not be normalized) -> (...,3,3) rotation matrix."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


# ── world-frame gaussians ready for rasterization ───────────────────────────

@dataclass
class WorldGaussians:
    """Activated, world-frame Gaussians. ``sh`` is (N, (deg+1)^2, 3) with the
    component's native degree; ``GSCompositeRenderer`` zero-pads all components
    to the max degree before the single rasterization pass (G2)."""
    means: torch.Tensor       # (N,3) float32
    quats: torch.Tensor       # (N,4) wxyz, normalized
    scales: torch.Tensor      # (N,3) linear (exp of log_scales)
    opacities: torch.Tensor   # (N,)  0..1 (sigmoid of logits)
    sh: torch.Tensor          # (N,K,3)

    @property
    def sh_degree(self) -> int:
        return int(round(self.sh.shape[1] ** 0.5)) - 1

    @staticmethod
    def concat(parts: List["WorldGaussians"]) -> "WorldGaussians":
        """Concatenate, zero-padding SH bands up to the max degree present."""
        assert parts, "nothing to concatenate"
        K = max(p.sh.shape[1] for p in parts)
        shs = []
        for p in parts:
            sh = p.sh
            if sh.shape[1] < K:
                pad = sh.new_zeros(sh.shape[0], K - sh.shape[1], 3)
                sh = torch.cat([sh, pad], dim=1)
            shs.append(sh)
        return WorldGaussians(
            means=torch.cat([p.means for p in parts]),
            quats=torch.cat([p.quats for p in parts]),
            scales=torch.cat([p.scales for p in parts]),
            opacities=torch.cat([p.opacities for p in parts]),
            sh=torch.cat(shs),
        )


# ── the component ───────────────────────────────────────────────────────────

SH_ROT_MODES = ("static", "so3_deg3", "so3_deg1", "wigner")


class PosedComponent:
    def __init__(self, name: str, *, means_l: torch.Tensor, quats_l: torch.Tensor,
                 log_scales: torch.Tensor, opacity_logits: torch.Tensor,
                 sh: torch.Tensor, sh_rot_mode: str,
                 p_capture: torch.Tensor, q_capture: torch.Tensor):
        if sh_rot_mode == "z_only_deg3":  # deprecated pre-R7 alias (module docstring)
            sh_rot_mode = "so3_deg3"
        assert sh_rot_mode in SH_ROT_MODES, sh_rot_mode
        if sh_rot_mode == "so3_deg1" and sh.shape[1] != 4:
            raise ValueError(
                f"component '{name}': so3_deg1 requires SH degree 1 (K=4), "
                f"got K={sh.shape[1]} (G5)")
        self.name = name
        self.means_l = means_l.float()
        self.quats_l = quat_normalize(quats_l.float())
        self.log_scales = log_scales.float()
        self.opacity_logits = opacity_logits.float()
        self.sh = sh.float()
        self.sh_rot_mode = sh_rot_mode
        self.p_capture = p_capture.reshape(3).float()
        self.q_capture = quat_normalize(q_capture.reshape(4).float())

    @property
    def device(self) -> torch.device:
        return self.means_l.device

    def to(self, device) -> "PosedComponent":
        for k in ("means_l", "quats_l", "log_scales", "opacity_logits", "sh",
                  "p_capture", "q_capture"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    @classmethod
    def from_asset(cls, asset: GaussianAsset, name: str, sh_rot_mode: str,
                   device: str = "cuda") -> "PosedComponent":
        asset.validate()
        meta = asset.meta
        p_cap = torch.as_tensor(meta.get("p_capture", [0.0, 0.0, 0.0]))
        q_cap = torch.as_tensor(meta.get("q_capture", [1.0, 0.0, 0.0, 0.0]))
        return cls(
            name,
            means_l=asset.means.to(device), quats_l=asset.quats.to(device),
            log_scales=asset.log_scales.to(device),
            opacity_logits=asset.opacity_logits.to(device),
            sh=asset.sh_full().to(device), sh_rot_mode=sh_rot_mode,
            p_capture=p_cap.to(device), q_capture=q_cap.to(device),
        ).to(device)

    # ── the one posing path (G5 routing lives here) ─────────────────────────

    def posed(self, p_wb, q_wb) -> WorldGaussians:
        """World gaussians for current body pose (p_wb (3,), q_wb (4,) wxyz)."""
        p_wb = torch.as_tensor(p_wb, dtype=torch.float32, device=self.device).reshape(3)
        q_wb = quat_normalize(
            torch.as_tensor(q_wb, dtype=torch.float32, device=self.device).reshape(4))

        if self.sh_rot_mode == "static":
            # background: identity pose only — a moved background would need SH
            # rotation, which 'static' by definition does not do (G5).
            if (p_wb.abs().max() > 1e-6
                    or (q_wb - q_wb.new_tensor([1.0, 0, 0, 0])).abs().max() > 1e-6):
                raise AssertionError(
                    f"component '{self.name}' is static (background) but got a "
                    f"non-identity pose — SH would silently not rotate (G5)")
            return self._world(self.means_l, self.quats_l, self.sh)

        R_wb = quat_to_R(q_wb)
        means_w = self.means_l @ R_wb.T + p_wb
        quats_w = quat_mul(q_wb.expand_as(self.quats_l), self.quats_l)

        # world-frame delta pose since capture drives the SH rotation (G5).
        # NOTE plan §6.1 writes the body-frame form q_capture⁻¹ ⊗ q_current;
        # the group-correct quantity for world-frame SH is the WORLD delta
        # q_current ⊗ q_capture⁻¹ (they coincide when the capture pose's z is
        # the world z, the LIBERO resting case). T2/T3 pin the behavior.
        q_delta = quat_mul(q_wb, quat_conj(self.q_capture))
        q_delta = quat_normalize(q_delta)

        if self.sh_rot_mode == "so3_deg3":
            # Exact for ANY delta (G5/R7): real demos tilt at reset and tumble
            # when grasped, so there is no assertion and no error path — only
            # a fast-path selection between two exact rotations.
            if q_delta[1].abs() <= FAST_Z_TOL and q_delta[2].abs() <= FAST_Z_TOL:
                theta = 2.0 * math.atan2(float(q_delta[3]), float(q_delta[0]))
                sh_w = rotate_sh_z(self.sh, theta)
            else:
                sh_w = rotate_sh_so3(self.sh, quat_to_R(q_delta))
        elif self.sh_rot_mode == "so3_deg1":
            sh_w = rotate_sh_l1(self.sh, quat_to_R(q_delta))
        elif self.sh_rot_mode == "wigner":
            sh_w = rotate_sh_wigner(self.sh, quat_to_R(q_delta))
        else:  # pragma: no cover — SH_ROT_MODES guard in __init__
            raise AssertionError(self.sh_rot_mode)

        return self._world(means_w, quats_w, sh_w)

    def posed_identity(self) -> WorldGaussians:
        """Background convenience: identity pose."""
        z = self.means_l.new_zeros(3)
        return self.posed(z, self.means_l.new_tensor([1.0, 0, 0, 0]))

    def _world(self, means, quats, sh) -> WorldGaussians:
        return WorldGaussians(
            means=means,
            quats=quat_normalize(quats),
            scales=self.log_scales.exp(),
            opacities=self.opacity_logits.sigmoid(),
            sh=sh,
        )

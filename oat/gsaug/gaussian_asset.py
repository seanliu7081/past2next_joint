"""GaussianAsset: the on-disk container for one trained Gaussian component.

One asset = one rigid Gaussian set: a task background (world frame), one
movable object (body frame), or the whole robot (per-Gaussian ``link_id``,
link-local frames). Saved with ``torch.save`` as a plain dict.

Conventions block (G7/R4): every asset records the conventions it was trained
under; ``load`` refuses an asset whose block disagrees with this module's
``EXPECTED_CONVENTIONS``. Provenance (G9): ``sha1`` is computed over the raw
parameter tensor bytes in a fixed order and verified on load.

Parameter storage (matches plan §5.2):
    means           float32 (N, 3)   local frame (see ``frame``)
    quats           float32 (N, 4)   wxyz, unnormalized allowed (normalized on use)
    log_scales      float32 (N, 3)
    opacity_logits  float32 (N,)
    sh_dc           float32 (N, 3)
    sh_rest         float32 (N, K, 3)   K = (deg+1)^2 - 1  (0 for deg 0)
    link_id         int32   (N,)     robot assets only (else absent)

Meta block: ``frame`` ('world' | 'body' | 'link'), ``sh_degree``, capture body
pose ``p_capture``/``q_capture`` (world pose of the body at capture; identity
for world-frame assets), per-link capture poses for robot assets, task name,
model-XML sha1, gsplat/torch versions, training args.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

# The conventions this codebase is written against. Assets recording anything
# else are refused at load (R4: convention landmines fail loudly).
EXPECTED_CONVENTIONS: Dict[str, str] = {
    "quat_order": "wxyz",
    "sh_layout": "3dgs_mneg_to_mpos",   # bands l=0..deg, m=-l..l (gsplat layout)
    "scales": "log",
    "opacity": "logit",
    "camera_model": "opencv_w2c",
    "sh_frame": "world_at_capture",     # SH captured in world frame; rotated at
                                        # render time by the delta pose (G5)
}

_PARAM_ORDER = ("means", "quats", "log_scales", "opacity_logits", "sh_dc", "sh_rest")


def _params_sha1(params: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha1()
    for k in _PARAM_ORDER:
        t = params[k].detach().to("cpu", torch.float32).contiguous()
        h.update(k.encode())
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    if "link_id" in params and params["link_id"] is not None:
        t = params["link_id"].detach().to("cpu", torch.int32).contiguous()
        h.update(b"link_id")
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    return h.hexdigest()


@dataclass
class GaussianAsset:
    means: torch.Tensor
    quats: torch.Tensor
    log_scales: torch.Tensor
    opacity_logits: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    conventions: Dict[str, str]
    meta: Dict = field(default_factory=dict)
    link_id: Optional[torch.Tensor] = None

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        K = int(self.sh_rest.shape[1]) + 1
        deg = int(round(K ** 0.5)) - 1
        assert (deg + 1) ** 2 == K, f"sh_rest K+1={K} is not a perfect square"
        return deg

    @property
    def frame(self) -> str:
        return self.meta["frame"]

    def sh_full(self) -> torch.Tensor:
        """(N, (deg+1)^2, 3): DC concatenated ahead of the rest bands."""
        return torch.cat([self.sh_dc[:, None, :], self.sh_rest], dim=1)

    def validate(self) -> None:
        n = self.n
        assert self.means.shape == (n, 3), self.means.shape
        assert self.quats.shape == (n, 4), self.quats.shape
        assert self.log_scales.shape == (n, 3), self.log_scales.shape
        assert self.opacity_logits.shape == (n,), self.opacity_logits.shape
        assert self.sh_dc.shape == (n, 3), self.sh_dc.shape
        assert self.sh_rest.ndim == 3 and self.sh_rest.shape[0] == n \
            and self.sh_rest.shape[2] == 3, self.sh_rest.shape
        _ = self.sh_degree
        if self.link_id is not None:
            assert self.link_id.shape == (n,), self.link_id.shape
        assert self.meta.get("frame") in ("world", "body", "link"), \
            f"asset meta.frame={self.meta.get('frame')!r}"
        for k, v in EXPECTED_CONVENTIONS.items():
            got = self.conventions.get(k)
            if got != v:
                raise RuntimeError(
                    f"asset conventions[{k!r}] = {got!r}, this code expects "
                    f"{v!r} — refusing to use an asset trained under different "
                    f"conventions (R4/G7).")

    def sha1(self) -> str:
        params = {k: getattr(self, k) for k in _PARAM_ORDER}
        params["link_id"] = self.link_id
        return _params_sha1(params)

    # ── IO ──────────────────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Write the asset; returns its content sha1 (also stored inside)."""
        self.validate()
        digest = self.sha1()
        payload = {
            "format": "oat.gsaug.GaussianAsset.v1",
            "conventions": dict(self.conventions),
            "meta": dict(self.meta),
            "sha1": digest,
            "params": {k: getattr(self, k).detach().cpu() for k in _PARAM_ORDER},
        }
        if self.link_id is not None:
            payload["params"]["link_id"] = self.link_id.detach().cpu()
        torch.save(payload, path)
        return digest

    @classmethod
    def load(cls, path: str, expected_frame: Optional[str] = None,
             device: str = "cpu") -> "GaussianAsset":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "oat.gsaug.GaussianAsset.v1":
            raise RuntimeError(f"'{path}' is not a GaussianAsset v1 file")
        p = payload["params"]
        asset = cls(
            means=p["means"].to(device),
            quats=p["quats"].to(device),
            log_scales=p["log_scales"].to(device),
            opacity_logits=p["opacity_logits"].to(device),
            sh_dc=p["sh_dc"].to(device),
            sh_rest=p["sh_rest"].to(device),
            conventions=dict(payload["conventions"]),
            meta=dict(payload["meta"]),
            link_id=p.get("link_id").to(device) if p.get("link_id") is not None else None,
        )
        asset.validate()
        digest = asset.sha1()
        if digest != payload["sha1"]:
            raise RuntimeError(
                f"'{path}' sha1 mismatch: stored {payload['sha1'][:12]}…, "
                f"recomputed {digest[:12]}… — corrupt or hand-edited asset (G9).")
        if expected_frame is not None and asset.frame != expected_frame:
            raise RuntimeError(
                f"'{path}' has frame={asset.frame!r}, caller expects "
                f"{expected_frame!r}")
        return asset

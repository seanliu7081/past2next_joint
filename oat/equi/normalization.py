"""Group-compatible normalization (Workstream N) + frozen NormalizationSpec.

Why this exists
---------------
Per-dim min-max normalization warps the planar (rho1) action blocks, so the
SO(2) group action in normalized space is NOT an orthogonal R(theta). This
module builds a ``LinearNormalizer`` whose normalized-space group action IS
exactly orthogonal with zero offset:

  * rho1 pair   -- one shared, symmetric, zero-centered scale
                   s = max over both dims of |min|, |max|  (= max ||(a_i,a_j)||_inf),
                   scale = 1/s tied, offset = 0
  * rho0 dim    -- the existing per-dim limits affine, formula vendored from
                   ``normalizer._fit`` (mode='limits', fit_offset=True)
  * free_iso    -- one shared scale over the whole block, offset = 0
  * identity    -- scale 1, offset 0 (unit quaternions)
  * rgb keys    -- untouched: whatever the fitted baseline produced (0..255
                   per-channel limits)

Consequences: (i) plain per-block ``randn`` is exactly P1-correct, (ii) raw-
space label rotation commutes exactly with normalization, (iii) the stats are
G-invariant, so one frozen spec serves every experimental arm.

Freezing
--------
Stats are computed ONCE (``scripts/build_normalization_spec.py``), persisted
as a JSON ``NormalizationSpec`` with a dataset fingerprint, and only ever
LOADED afterwards (``normalizer_from_spec``). ``assert_spec_matches`` is the
stats-frozen guard used by the policy/dataset at construction time.

The normalizer convention is ``x_norm = x * scale + offset``
(``oat.model.common.normalizer._normalize``).
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from oat.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from oat.equi.blocks import (
    RHO0,
    RHO1,
    FREE_ISO,
    IDENTITY,
    BlockSpec,
    assert_blocks_cover,
    libero_action_blocks,
    libero_obs_blocks,
)

SPEC_VERSION = 1
MODE_GROUP_COMPATIBLE = "group_compatible"
MODE_PER_DIM_MINMAX = "per_dim_minmax"
MODES = (MODE_GROUP_COMPATIBLE, MODE_PER_DIM_MINMAX)

# match normalizer._fit defaults (mode='limits', fit_offset=True)
_OUTPUT_MAX = 1.0
_OUTPUT_MIN = -1.0
_RANGE_EPS = 1e-4


# ── per-block scale/offset rules ─────────────────────────────────────────────

def _limits_scale_offset(vmin: np.ndarray, vmax: np.ndarray):
    """Per-dim limits affine, vendored from ``normalizer._fit`` lines 216-223
    (mode='limits', fit_offset=True) so rho0 dims match the baseline exactly."""
    vmin = np.asarray(vmin, dtype=np.float64)
    vmax = np.asarray(vmax, dtype=np.float64)
    input_range = vmax - vmin
    ignore_dim = input_range < _RANGE_EPS
    input_range = np.where(ignore_dim, _OUTPUT_MAX - _OUTPUT_MIN, input_range)
    scale = (_OUTPUT_MAX - _OUTPUT_MIN) / input_range
    offset = _OUTPUT_MIN - scale * vmin
    offset = np.where(ignore_dim, (_OUTPUT_MAX + _OUTPUT_MIN) / 2.0 - vmin, offset)
    return scale, offset


def _shared_symmetric_scale(vmin: np.ndarray, vmax: np.ndarray) -> float:
    """Shared zero-centered scale for rho1/free_iso: s = max |min|,|max| over
    the block's dims; scale = 1/s. Guards degenerate all-zero blocks."""
    s = float(max(np.max(np.abs(vmin)), np.max(np.abs(vmax))))
    if s < _RANGE_EPS:
        s = 1.0  # constant-zero block: leave unscaled
    return 1.0 / s


def group_compatible_scale_offset(
    stats: Dict[str, np.ndarray], blocks: Sequence[BlockSpec]
):
    """Build (scale, offset) float64 vectors for one key from its raw per-dim
    ``stats`` ({'min','max','mean','std'}) and its block spec."""
    vmin = np.asarray(stats["min"], dtype=np.float64).flatten()
    vmax = np.asarray(stats["max"], dtype=np.float64).flatten()
    dim = vmin.shape[0]
    assert_blocks_cover(blocks, dim)

    scale = np.ones(dim, dtype=np.float64)
    offset = np.zeros(dim, dtype=np.float64)
    for b in blocks:
        idx = np.asarray(b.idx, dtype=int)
        if b.rep == RHO0:
            s, o = _limits_scale_offset(vmin[idx], vmax[idx])
            scale[idx], offset[idx] = s, o
        elif b.rep in (RHO1, FREE_ISO):
            scale[idx] = _shared_symmetric_scale(vmin[idx], vmax[idx])
            offset[idx] = 0.0
        elif b.rep == IDENTITY:
            scale[idx] = 1.0
            offset[idx] = 0.0
    return scale, offset


# ── normalizer builders ──────────────────────────────────────────────────────

def _stats_np(params_stats) -> Dict[str, np.ndarray]:
    return {
        k: params_stats[k].detach().cpu().numpy().astype(np.float64).flatten()
        for k in ("min", "max", "mean", "std")
    }


def _manual_field(scale: np.ndarray, offset: np.ndarray, stats: Dict[str, np.ndarray]):
    dtype = torch.float32
    field = SingleFieldLinearNormalizer.create_manual(
        scale=torch.as_tensor(scale, dtype=dtype),
        offset=torch.as_tensor(offset, dtype=dtype),
        input_stats_dict={k: torch.as_tensor(v, dtype=dtype) for k, v in stats.items()},
    )
    for p in field.params_dict.parameters():
        p.requires_grad_(False)
    return field


def build_group_compatible_normalizer(
    fitted: LinearNormalizer,
    rgb_keys: Sequence[str],
    action_blocks: Optional[List[BlockSpec]] = None,
    obs_blocks: Optional[Dict[str, List[BlockSpec]]] = None,
    world_frame_rotation: bool = False,
) -> LinearNormalizer:
    """Derive the group-compatible normalizer from a baseline-fitted one.

    ``fitted`` supplies the raw per-dim input stats for every key (so both
    modes share bit-identical ``input_stats``); block rules override
    scale/offset for the action and low-dim obs keys; rgb keys are copied
    through untouched.
    """
    if action_blocks is None:
        action_blocks = libero_action_blocks(world_frame_rotation)
    if obs_blocks is None:
        obs_blocks = libero_obs_blocks()

    out = LinearNormalizer()
    for key in fitted.params_dict.keys():
        params = fitted.params_dict[key]
        stats = _stats_np(params["input_stats"])
        if key in rgb_keys:
            scale = params["scale"].detach().cpu().numpy().astype(np.float64).flatten()
            offset = params["offset"].detach().cpu().numpy().astype(np.float64).flatten()
        else:
            blocks = action_blocks if key == "action" else obs_blocks.get(key)
            if blocks is None:
                # unknown low-dim key: fall back to the baseline per-dim affine
                scale = params["scale"].detach().cpu().numpy().astype(np.float64).flatten()
                offset = params["offset"].detach().cpu().numpy().astype(np.float64).flatten()
            else:
                scale, offset = group_compatible_scale_offset(stats, blocks)
        out[key] = _manual_field(scale, offset, stats)
    return out


# ── spec (de)serialization + freeze guards ───────────────────────────────────

def fingerprint_replay_buffer(replay_buffer, zarr_path: str) -> Dict:
    """Cheap identity of the dataset the stats were computed from. The action
    array (~4 MB) is hashed in full; images are covered indirectly via
    n_steps/n_episodes."""
    action = np.ascontiguousarray(np.asarray(replay_buffer["action"], dtype=np.float32))
    return {
        "zarr_path": os.path.normpath(str(zarr_path)),
        "n_steps": int(action.shape[0]),
        "n_episodes": int(replay_buffer.n_episodes),
        "action_sha1": hashlib.sha1(action.tobytes()).hexdigest(),
    }


def spec_from_normalizer(
    normalizer: LinearNormalizer,
    mode: str,
    fingerprint: Dict,
    world_frame_rotation: bool = False,
) -> Dict:
    assert mode in MODES, mode
    keys = {}
    for key in normalizer.params_dict.keys():
        params = normalizer.params_dict[key]
        keys[key] = {
            "scale": params["scale"].detach().cpu().numpy().astype(np.float64).flatten().tolist(),
            "offset": params["offset"].detach().cpu().numpy().astype(np.float64).flatten().tolist(),
            "input_stats": {
                k: v.tolist() for k, v in _stats_np(params["input_stats"]).items()
            },
        }
    return {
        "version": SPEC_VERSION,
        "mode": mode,
        "world_frame_rotation": bool(world_frame_rotation),
        "fingerprint": fingerprint,
        "keys": keys,
    }


def save_spec(spec: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(spec, f, sort_keys=True, indent=1)
    os.replace(tmp, path)


def load_spec(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NormalizationSpec not found at '{path}'. Build it once with "
            f"scripts/build_normalization_spec.py (stats are frozen across all arms)."
        )
    with open(path) as f:
        spec = json.load(f)
    assert spec.get("version") == SPEC_VERSION, f"unsupported spec version in {path}"
    assert spec.get("mode") in MODES, f"unknown mode in {path}"
    return spec


def normalizer_from_spec(spec: Dict) -> LinearNormalizer:
    out = LinearNormalizer()
    for key, entry in spec["keys"].items():
        out[key] = _manual_field(
            np.asarray(entry["scale"], dtype=np.float64),
            np.asarray(entry["offset"], dtype=np.float64),
            {k: np.asarray(v, dtype=np.float64) for k, v in entry["input_stats"].items()},
        )
    return out


def verify_fingerprint(spec: Dict, replay_buffer, zarr_path: str) -> None:
    """Raise if the spec was built from different data (stats-frozen guard)."""
    fp = fingerprint_replay_buffer(replay_buffer, zarr_path)
    ref = dict(spec["fingerprint"])
    mismatches = {
        k: (ref.get(k), fp[k])
        for k in ("n_steps", "n_episodes", "action_sha1")
        if ref.get(k) != fp[k]
    }
    if mismatches:
        raise RuntimeError(
            f"NormalizationSpec fingerprint mismatch (spec built from different data): {mismatches}"
        )


def assert_spec_matches(normalizer: LinearNormalizer, spec: Dict, atol: float = 1e-6) -> None:
    """Assert the live normalizer's scale/offset equal the frozen spec's, for
    every key in the spec that the normalizer carries."""
    for key, entry in spec["keys"].items():
        if key not in normalizer.params_dict:
            continue
        params = normalizer.params_dict[key]
        for name in ("scale", "offset"):
            live = params[name].detach().cpu().numpy().astype(np.float64).flatten()
            ref = np.asarray(entry[name], dtype=np.float64)
            if live.shape != ref.shape or not np.allclose(live, ref, atol=atol):
                raise RuntimeError(
                    f"normalizer['{key}'].{name} deviates from the frozen "
                    f"NormalizationSpec (max abs diff "
                    f"{np.max(np.abs(live - ref)) if live.shape == ref.shape else 'shape mismatch'}). "
                    f"All arms must load the same persisted spec."
                )

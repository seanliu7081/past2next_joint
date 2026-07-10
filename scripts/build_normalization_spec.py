"""
One-shot builder of BOTH frozen NormalizationSpec JSONs (D3).

Emits, from one pass over the training zarr:
  <out_dir>/norm_spec_per_dim_minmax_<zarrstem>.json
      bit-for-bit serialization of today's ``ZarrDataset.get_normalizer()``
      output (per-dim min-max limits fit, incl. rgb 0..255)
  <out_dir>/norm_spec_group_compatible_<zarrstem>.json
      the group-compatible normalizer derived from the same stats
      (``oat.equi.normalization.build_group_compatible_normalizer``)

Both specs carry the same dataset fingerprint. After this script runs once,
every experimental arm only LOADS a spec (never refits), so the stats are
provably frozen across arms.

RAM note: low-dim keys are fitted exactly like ``ZarrDataset.get_normalizer()``
(arrays in RAM, ``LinearNormalizer.fit(..., last_n_dims=1, mode='limits')``).
The rgb keys are NOT materialized as float tensors (~27 GB each); instead
per-channel min/max are accumulated exactly over uint8 chunks and the limits
scale/offset are computed from them via the vendored ``_limits_scale_offset``
formula -- identical to what ``fit`` would produce. The rgb mean/std stored in
``input_stats`` come from a float64 streaming pass (Welford/Chan, ddof=1 to
match torch's unbiased default) and are informational only: scale/offset
depend solely on min/max.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib

import click
import numpy as np
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.model.common.normalizer import LinearNormalizer
from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    MODE_PER_DIM_MINMAX,
    _limits_scale_offset,
    _manual_field,
    build_group_compatible_normalizer,
    fingerprint_replay_buffer,
    save_spec,
    spec_from_normalizer,
)

# libero10 task obs keys (oat/config/task/policy/libero/libero10.yaml)
DEFAULT_OBS_KEYS = (
    "agentview_rgb",
    "robot0_eye_in_hand_rgb",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "task_uid",
)

RGB_CHUNK_FRAMES = 64


def streaming_rgb_stats(arr, chunk_frames: int = RGB_CHUNK_FRAMES):
    """Exact per-channel uint8 min/max + float64 running mean/M2 over a lazy
    [N, H, W, C] zarr array, ``chunk_frames`` frames at a time.

    Returns {'min','max','mean','std'} float64 (std with ddof=1, matching the
    unbiased default of ``torch.Tensor.std`` used by ``normalizer._fit``).
    """
    n_frames, c = arr.shape[0], arr.shape[-1]
    vmin = np.full(c, np.iinfo(arr.dtype).max, dtype=np.int64)
    vmax = np.full(c, np.iinfo(arr.dtype).min, dtype=np.int64)
    n = 0
    mean = np.zeros(c, dtype=np.float64)
    m2 = np.zeros(c, dtype=np.float64)
    for start in range(0, n_frames, chunk_frames):
        flat = arr[start:start + chunk_frames].reshape(-1, c)
        vmin = np.minimum(vmin, flat.min(axis=0))
        vmax = np.maximum(vmax, flat.max(axis=0))
        # Chan et al. pairwise merge of (mean, M2)
        cn = flat.shape[0]
        flat64 = flat.astype(np.float64)
        cmean = flat64.mean(axis=0)
        cm2 = np.square(flat64 - cmean).sum(axis=0)
        tot = n + cn
        delta = cmean - mean
        mean = mean + delta * (cn / tot)
        m2 = m2 + cm2 + np.square(delta) * (n * cn / tot)
        n = tot
    std = np.sqrt(m2 / (n - 1)) if n > 1 else np.zeros(c, dtype=np.float64)
    return {
        "min": vmin.astype(np.float64),
        "max": vmax.astype(np.float64),
        "mean": mean,
        "std": std,
    }


def print_spec_table(title: str, spec: dict) -> None:
    print(f"\n=== {title} (mode={spec['mode']}) ===")
    fmt = lambda v: np.array2string(
        np.asarray(v, dtype=np.float64), precision=6, separator=", ",
        max_line_width=200,
    )
    for key in sorted(spec["keys"]):
        entry = spec["keys"][key]
        print(f"  {key:24s} scale  = {fmt(entry['scale'])}")
        print(f"  {'':24s} offset = {fmt(entry['offset'])}")


@click.command()
@click.option("--zarr", "zarr_path", type=str,
              default="data/libero/libero10_N500.zarr", show_default=True)
@click.option("--out_dir", type=str, default="data/libero", show_default=True)
@click.option("--obs_keys", "--obs-keys", "obs_keys", multiple=True,
              default=DEFAULT_OBS_KEYS, show_default=True)
@click.option("--rgb_chunk_frames", type=int, default=RGB_CHUNK_FRAMES,
              show_default=True, help="frames per rgb streaming read")
@click.option("--world_frame_rotation", is_flag=True, default=False,
              show_default=True,
              help="build the group-compatible spec with the rot_xy/rot_z "
                   "block split (post-M2 ablation); output files gain a "
                   "'_wfr' suffix so the default specs are not clobbered")
def build_normalization_specs(zarr_path, out_dir, obs_keys, rgb_chunk_frames,
                              world_frame_rotation):
    obs_keys = list(obs_keys)
    root = zarr.open(zarr_path, "r")  # lazy: rgb never fully materialized

    # split requested keys: rgb (4-D image stacks) / low-dim numeric / skipped
    rgb_keys, lowdim_keys = [], []
    for key in obs_keys:
        arr = root["data"][key]
        if arr.ndim == 4 and arr.dtype == np.uint8:
            rgb_keys.append(key)
        elif arr.dtype.kind in "biuf":
            lowdim_keys.append(key)
        else:
            print(f"WARNING: skipping non-numeric key '{key}' (dtype {arr.dtype}) "
                  f"-- ZarrDataset does not normalize it either")

    # low-dim keys: into RAM via ReplayBuffer (same code path as ZarrDataset);
    # this buffer also carries meta/episode_ends for the fingerprint
    replay_buffer = ReplayBuffer.copy_from_path(
        zarr_path, keys=["action", *lowdim_keys])
    fingerprint = fingerprint_replay_buffer(replay_buffer, zarr_path)
    print(f"fingerprint: {fingerprint}")

    # baseline per-dim fit, EXACTLY as ZarrDataset.get_normalizer(mode='limits')
    lowdim_fitted = LinearNormalizer()
    lowdim_fitted.fit(
        data={
            "action": replay_buffer["action"],
            **{k: replay_buffer[k] for k in lowdim_keys},
        },
        last_n_dims=1, mode="limits",
    )

    # rgb keys: streaming stats; scale/offset from exact min/max via the
    # vendored limits formula (== what fit would compute); mean/std informational
    rgb_fields = {}
    for key in rgb_keys:
        arr = root["data"][key]
        print(f"streaming stats over '{key}' {arr.shape} ...")
        stats = streaming_rgb_stats(arr, chunk_frames=rgb_chunk_frames)
        if not (np.all(stats["min"] == 0.0) and np.all(stats["max"] == 255.0)):
            print(f"WARNING: '{key}' per-channel limits are not exactly [0, 255] "
                  f"(min={stats['min']}, max={stats['max']}); the spec will bake "
                  f"in these observed limits -- proceeding anyway")
        scale, offset = _limits_scale_offset(stats["min"], stats["max"])
        rgb_fields[key] = _manual_field(scale, offset, stats)

    # assemble the per-dim normalizer in ZarrDataset key order (action first)
    per_dim = LinearNormalizer()
    for key in ["action", *obs_keys]:
        if key in rgb_keys:
            per_dim[key] = rgb_fields[key]
        elif key == "action" or key in lowdim_keys:
            per_dim[key] = lowdim_fitted[key]

    group_compatible = build_group_compatible_normalizer(
        per_dim, rgb_keys=rgb_keys, world_frame_rotation=world_frame_rotation)

    zarr_stem = pathlib.Path(str(zarr_path).rstrip("/")).stem
    suffix = "_wfr" if world_frame_rotation else ""
    os.makedirs(out_dir, exist_ok=True)
    out_paths = {}
    for mode, normalizer in (
        (MODE_PER_DIM_MINMAX, per_dim),
        (MODE_GROUP_COMPATIBLE, group_compatible),
    ):
        spec = spec_from_normalizer(
            normalizer, mode=mode, fingerprint=fingerprint,
            world_frame_rotation=world_frame_rotation,
        )
        path = os.path.join(out_dir, f"norm_spec_{mode}{suffix}_{zarr_stem}.json")
        save_spec(spec, path)
        out_paths[mode] = path
        print_spec_table(os.path.basename(path), spec)

    print("\nwrote:")
    for mode, path in out_paths.items():
        print(f"  [{mode}] {path}")


if __name__ == "__main__":
    build_normalization_specs()

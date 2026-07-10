"""Content-based alignment between the training zarr and the source LIBERO
HDF5 demos, plus obs/state off-by-one calibration.

The zarr conversion (``oat.env.libero.dataset_conversion``) shuffles demos
with an unseeded RNG, so episode ORDER carries no information -- the float32
action bytes are the only reliable link (the zarr ``action`` array is an
exact float32 cast of each HDF5 ``demo['actions']``). Alignment is therefore
recovered by sha1 content matching, asserted unique 1:1 for every episode.

Off-by-one: HDF5 demos store pre-action states while the converted obs are
post-action, so stored obs[t] corresponds to states[t + delta] with
delta in {0, 1} -- calibrated empirically per task by pixel-comparing theta=0
renders against stored zarr images (expected delta = 1).

Module-level imports are numpy/h5py/stdlib only; anything sim-related enters
via the ``render_fn`` callback so this module stays importable in tests and
dataloader workers.
"""

import glob
import hashlib
import os
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np


def _action_sha1(actions: np.ndarray) -> str:
    """sha1 of the little-endian float32 bytes of an (T, 7) action array."""
    a = np.ascontiguousarray(np.asarray(actions).astype("<f4"))
    return hashlib.sha1(a.tobytes()).hexdigest()


def index_hdf5_demos(hdf5_dir: str) -> Dict[str, Tuple[str, str]]:
    """Index every demo under ``hdf5_dir/*.hdf5`` by the sha1 of its float32
    action bytes. Returns ``{sha1hex: (hdf5_path, demo_key)}``; asserts no
    hash collisions across demos."""
    paths = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    assert paths, f"no *.hdf5 files found under {hdf5_dir!r}"

    index: Dict[str, Tuple[str, str]] = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            for demo_key in sorted(f["data"].keys()):
                digest = _action_sha1(f["data"][demo_key]["actions"][:])
                assert digest not in index, (
                    f"action sha1 collision: {path}:{demo_key} and "
                    f"{index[digest][0]}:{index[digest][1]} hash to {digest} -- "
                    "duplicate demos in the HDF5 set?"
                )
                index[digest] = (path, demo_key)
    return index


def match_episodes(replay_buffer, hdf5_dir: str) -> List[Tuple[str, str]]:
    """Match every zarr episode to its source HDF5 demo by action content.

    Returns one ``(hdf5_path, demo_key)`` per zarr episode, in episode order.
    Raises ``ValueError`` unless the match is a unique 1:1 for ALL episodes.
    """
    index = index_hdf5_demos(hdf5_dir)
    actions = np.ascontiguousarray(np.asarray(replay_buffer["action"], dtype="<f4"))
    episode_ends = np.asarray(replay_buffer.episode_ends)
    assert len(episode_ends) > 0 and int(episode_ends[-1]) == len(actions), (
        f"episode_ends[-1]={episode_ends[-1] if len(episode_ends) else None} "
        f"inconsistent with action length {len(actions)}"
    )

    matches: List[Tuple[str, str]] = []
    unmatched: List[Tuple[int, int]] = []          # (episode_idx, length)
    seen: Dict[str, int] = {}                      # sha1 -> first episode idx
    duplicates: List[Tuple[int, int]] = []         # (episode_idx, first_episode_idx)
    start = 0
    for ep_idx, end in enumerate(episode_ends):
        end = int(end)
        digest = _action_sha1(actions[start:end])
        hit = index.get(digest)
        if hit is None:
            unmatched.append((ep_idx, end - start))
        else:
            if digest in seen:
                duplicates.append((ep_idx, seen[digest]))
            seen[digest] = ep_idx
            matches.append(hit)
        start = end

    if unmatched or duplicates:
        lines = [
            f"zarr<->HDF5 episode matching failed: "
            f"{len(episode_ends) - len(unmatched)}/{len(episode_ends)} episodes matched "
            f"against {len(index)} demos under {hdf5_dir!r}."
        ]
        if unmatched:
            lines.append(
                f"  unmatched episodes (idx, length): {unmatched[:10]}"
                + (" ..." if len(unmatched) > 10 else "")
            )
            lines.append(
                "  likely causes: wrong --hdf5_dir (different suite / demo set), "
                "or the zarr was built from actions that are not exact float32 "
                "casts of demo['actions']."
            )
        if duplicates:
            lines.append(
                f"  duplicate episodes mapping to the same demo "
                f"(episode, earlier episode): {duplicates[:10]}"
            )
        raise ValueError("\n".join(lines))
    return matches


# ── obs/state off-by-one calibration ─────────────────────────────────────────

def calibrate_state_offset_detailed(
    env,
    states: np.ndarray,
    episode_zarr_images: np.ndarray,
    render_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[int, Dict[int, float]]:
    """Pick delta in {0, 1} such that ``render(states[t + delta])`` best matches
    the stored zarr image at t, probing a handful of frames spread across the
    episode. Returns ``(delta, {delta: mean_abs_pixel_diff})``.

    ``render_fn(state)`` must return the FLIPPED (np.flip axis=0) agentview
    uint8 image at the zarr resolution -- i.e. directly comparable to stored
    frames. ``env`` is unused here; it is accepted so call sites keep the live
    env (which ``render_fn`` closes over) explicit.
    """
    del env
    states = np.asarray(states)
    images = np.asarray(episode_zarr_images)

    # t must index both images[t] and states[t + 1]
    max_t = min(len(images), len(states) - 1) - 1
    assert max_t >= 0, (
        f"episode too short to calibrate: {len(states)} states, {len(images)} images"
    )
    probe_ts = np.unique(np.linspace(0, max_t, num=min(8, max_t + 1)).round().astype(int))

    diffs: Dict[int, float] = {}
    for delta in (0, 1):
        total = 0.0
        for t in probe_ts:
            rendered = np.asarray(render_fn(states[t + delta]), dtype=np.float32)
            stored = images[t].astype(np.float32)
            assert rendered.shape == stored.shape, (
                f"render_fn output shape {rendered.shape} != stored image shape "
                f"{stored.shape} -- wrong camera/size/flip?"
            )
            total += float(np.mean(np.abs(rendered - stored)))
        diffs[delta] = total / len(probe_ts)

    delta = min(diffs, key=diffs.get)  # ties resolve to delta=0
    return delta, diffs


def calibrate_state_offset(
    env,
    states: np.ndarray,
    episode_zarr_images: np.ndarray,
    render_fn: Callable[[np.ndarray], np.ndarray],
) -> int:
    """See :func:`calibrate_state_offset_detailed`; returns just delta."""
    delta, _diffs = calibrate_state_offset_detailed(env, states, episode_zarr_images, render_fn)
    return delta

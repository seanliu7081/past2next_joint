"""SE(2)-augmented LIBERO zarr dataset (Workstream A).

Mirrors ``ZarrDataset`` exactly (sampler geometry, val split, output schema) but
serves per-item world-frame SE(2)-rotated (image, proprio, action) triples:

  * numerics (action + low-dim obs) live in RAM via ``ReplayBuffer`` from the
    BASE zarr, exactly like ``ZarrDataset``;
  * rgb frames are read lazily from disk -- either the pre-rendered aug zarr
    (``images/<key>/angle_XX``, one array per angle) or the base zarr;
  * each ``__getitem__`` draws exactly ONE angle index ``k`` (``k = 0`` when
    ``augment=False``) uniformly over that episode's valid angles, fetches the
    angle-``k`` images for the ``To`` obs frames, and rotates the RAW action
    chunk / proprio by ``theta = radians(angles_deg[k])``
    (``oat.equi.se2_transforms``). Raw-space rotation commutes exactly with the
    frozen group-compatible normalizer, which is only ever LOADED from a spec
    (``oat.equi.normalization``) -- never refit.

Matched budget: ``__len__`` counts sampler windows over the base numerics, so
it is independent of ``augment`` by construction (aug arms see the same number
of samples per epoch as no-aug arms).

Per-worker RNG caveat: the angle RNG is lazily seeded from
``torch.utils.data.get_worker_info().seed`` on first use in each worker. With
``persistent_workers=True`` the worker (and hence the angle stream) survives
across epochs instead of being re-seeded per epoch; angle draws stay i.i.d.
uniform either way.
"""

import copy
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.common.seq_sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from oat.dataset.base_dataset import BaseDataset
from oat.equi.normalization import (
    MODES,
    load_spec,
    normalizer_from_spec,
    verify_fingerprint,
)
from oat.equi.se2_transforms import rotate_action_chunk, rotate_proprio


def is_numeric_dtype(x):
    return x.dtype.kind in 'biuf'  # bool, int, uint, float


class SE2AugZarrDataset(BaseDataset):
    def __init__(
        self,
        zarr_path: str,
        obs_keys: List[str] = [],
        action_key: str = 'action',
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        augment: bool = False,
        aug_zarr_path: Optional[str] = None,
        image_source: str = 'aug',
        controller_frame: str = 'world',
        rotate_rotation_labels: bool = True,
        probe_results_path: str = 'data/libero/probe_results.json',
        norm_mode: str = 'group_compatible',
        norm_spec_path: Optional[str] = None,
        world_frame_rotation: bool = False,
        emit_angle_pair: bool = False,
        naive_image_rotation: bool = False,
        expected_render_source: Optional[str] = None,
        probe_gs_geometry_path: str = 'data/libero/probe_gs_geometry.json',
    ):
        super().__init__()
        assert n_obs_steps + n_action_steps > 0, "should have at least one frame"
        assert image_source in ('aug', 'base'), f"unknown image_source '{image_source}'"
        assert norm_mode in MODES, f"unknown norm_mode '{norm_mode}'"
        if augment and image_source == 'base' and not naive_image_rotation:
            raise ValueError(
                "augment=True with image_source='base' would pair rotated labels "
                "with un-rotated images; set image_source='aug' (or "
                "naive_image_rotation=True for the deliberately-wrong control arm)."
            )

        # frozen normalization spec (D3): required always, fail fast at ctor.
        if norm_spec_path is None:
            raise ValueError(
                "norm_spec_path is required (stats are frozen across all arms); "
                "build it once with scripts/build_normalization_spec.py."
            )
        spec = load_spec(norm_spec_path)
        assert spec['mode'] == norm_mode, (
            f"NormalizationSpec at '{norm_spec_path}' has mode '{spec['mode']}', "
            f"dataset expects '{norm_mode}'"
        )
        assert bool(spec['world_frame_rotation']) == bool(world_frame_rotation), (
            f"NormalizationSpec at '{norm_spec_path}' has world_frame_rotation="
            f"{spec['world_frame_rotation']}, dataset expects {world_frame_rotation}"
        )

        # probe gating (D7): world-frame rotation labels require the
        # controller-frame probe to have confirmed rot='world' AND to have
        # PASSed overall (pos/rot verdicts both conclusive and agreeing).
        rotate_rot_labels = rotate_rotation_labels and controller_frame == 'world'
        if augment and rotate_rot_labels:
            if not os.path.exists(probe_results_path):
                raise RuntimeError(
                    f"probe results not found at '{probe_results_path}': world-frame "
                    f"rotation labels are gated on the controller-frame probe. Run "
                    f"scripts/probes/probe_controller_frame.py first."
                )
            with open(probe_results_path) as f:
                probe = json.load(f)
            if probe.get('controller_frame_rot') != 'world':
                raise RuntimeError(
                    f"'{probe_results_path}' reports controller_frame_rot="
                    f"'{probe.get('controller_frame_rot')}', not 'world': refusing to "
                    f"rotate rotation labels. Re-run "
                    f"scripts/probes/probe_controller_frame.py or set "
                    f"rotate_rotation_labels=False / controller_frame accordingly."
                )
            if probe.get('pass') is not True:
                raise RuntimeError(
                    f"'{probe_results_path}' reports controller_frame_rot='world' "
                    f"but its overall verdict is pass={probe.get('pass')!r} (the "
                    f"probe run did not PASS: pos/rot verdicts inconclusive or "
                    f"disagreeing, or the file predates the 'pass' key): refusing "
                    f"to rotate rotation labels. Re-run "
                    f"scripts/probes/probe_controller_frame.py (e.g. MUJOCO_GL=egl "
                    f"python scripts/probes/probe_controller_frame.py "
                    f"--task_name libero10 --out {probe_results_path}) until it "
                    f"PASSes, or set rotate_rotation_labels=False."
                )

        # rgb keys are served lazily from disk; only numerics go to RAM.
        rgb_keys = [k for k in obs_keys if k.endswith('_rgb')]
        low_dim_keys = [k for k in obs_keys if not k.endswith('_rgb')]

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[action_key, *low_dim_keys],
        )
        episode_ends = self.replay_buffer.episode_ends[:]

        # aug zarr: required iff images or angle metadata are needed. When not
        # required (augment=False, image_source='base') the dataset must
        # construct with no aug zarr on disk (M1 smoke-train precedes M3).
        self._aug_required = bool(augment or image_source == 'aug')
        if self._aug_required:
            if aug_zarr_path is None:
                raise ValueError(
                    "aug_zarr_path is required when augment=True or "
                    "image_source='aug'; pre-render it with "
                    "scripts/prerender_se2_aug.py."
                )
            aug_root = zarr.open(aug_zarr_path, 'r')

            # render-source gating (G9, plan §8.1/§9): a GS training arm must
            # not start on a zarr rendered by a different pipeline. The aug
            # zarr self-describes its renderer via root attrs 'render_source'
            # (mirrored in a meta/render_source string array by
            # scripts/prerender_se2_aug.py); zarrs that predate the key are
            # oracle by definition.
            if expected_render_source is not None:
                actual_source = aug_root.attrs.get('render_source')
                if actual_source is None and 'meta/render_source' in aug_root:
                    v = aug_root['meta/render_source'][0]
                    actual_source = (
                        v.decode('utf-8') if isinstance(v, bytes) else str(v))
                if actual_source is None:
                    actual_source = 'oracle'
                if actual_source != expected_render_source:
                    raise RuntimeError(
                        f"aug zarr '{aug_zarr_path}' carries render_source="
                        f"'{actual_source}' but this dataset expects "
                        f"expected_render_source='{expected_render_source}': "
                        f"the pre-rendered images do not come from the "
                        f"required renderer. Point aug_zarr_path at a zarr "
                        f"pre-rendered with the matching "
                        f"scripts/prerender_se2_aug.py --renderer mode, or "
                        f"fix expected_render_source."
                    )
                # GS renders are additionally gated on the geometry probe
                # (D7 pattern, mirrors the controller-frame gate above).
                if expected_render_source.startswith('gs'):
                    if not os.path.exists(probe_gs_geometry_path):
                        raise RuntimeError(
                            f"probe results not found at "
                            f"'{probe_gs_geometry_path}': GS-rendered "
                            f"training data (expected_render_source="
                            f"'{expected_render_source}') is gated on the GS "
                            f"geometry probe. Run "
                            f"scripts/probes/probe_gs_geometry.py first."
                        )
                    with open(probe_gs_geometry_path) as f:
                        gs_probe = json.load(f)
                    if gs_probe.get('pass') is not True:
                        raise RuntimeError(
                            f"'{probe_gs_geometry_path}' reports pass="
                            f"{gs_probe.get('pass')!r} (the GS geometry probe "
                            f"did not PASS: silhouette IoU / EEF projection / "
                            f"wrist transform-stack check failed, or the file "
                            f"predates the 'pass' key): refusing to train on "
                            f"GS renders. Re-run "
                            f"scripts/probes/probe_gs_geometry.py (e.g. "
                            f"MUJOCO_GL=egl python "
                            f"scripts/probes/probe_gs_geometry.py --out "
                            f"{probe_gs_geometry_path}) until it PASSes."
                        )

            aug_ends = np.asarray(aug_root['meta/episode_ends'][:])
            assert np.array_equal(aug_ends, np.asarray(episode_ends)), (
                "aug zarr meta/episode_ends differs from the base zarr's -- it was "
                "pre-rendered from different data"
            )
            angles_deg = np.asarray(aug_root['meta/angles_deg'][:], dtype=np.float64)
            valid_mask = np.asarray(aug_root['meta/valid_mask'][:], dtype=bool)
            p_base = np.asarray(aug_root['meta/p_base'][:], dtype=np.float64)
            n_ep = len(episode_ends)
            n_steps = int(episode_ends[-1])
            assert angles_deg.ndim == 1 and angles_deg[0] == 0.0, \
                "angle index 0 must be theta=0"
            assert valid_mask.shape == (n_ep, len(angles_deg))
            assert p_base.shape == (n_ep, 3)
            assert valid_mask[:, 0].all(), \
                "aug zarr has invalid angle-0 episodes; the pre-render must " \
                "guarantee angle 0 valid for every episode"
            for k in rgb_keys:
                for j in range(len(angles_deg)):
                    arr = aug_root[f'images/{k}/angle_{j:02d}']
                    assert arr.shape[0] == n_steps, (
                        f"aug zarr images/{k}/angle_{j:02d} has {arr.shape[0]} "
                        f"frames, expected {n_steps}"
                    )
        else:
            # placeholders so the k=0 code path is uniform
            angles_deg = np.zeros(1, dtype=np.float64)
            valid_mask = np.ones((len(episode_ends), 1), dtype=bool)
            p_base = None

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        pad_before = max(n_obs_steps - 1, 0)
        pad_after = max(n_action_steps - 1, 0)
        seq_len = pad_before + 1 + pad_after
        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=seq_len,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=[action_key, *low_dim_keys],
        )

        # text observation keys (numerics only; rgb handled separately)
        sample0 = self.seq_sampler.sample_sequence(0)
        numeric_obs_keys = []
        text_obs_keys = []
        for k in low_dim_keys:
            if is_numeric_dtype(sample0[k]):
                numeric_obs_keys.append(k)
            else:
                text_obs_keys.append(k)

        self.train_mask = train_mask
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.seq_len = seq_len
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.numeric_obs_keys = numeric_obs_keys
        self.text_obs_keys = text_obs_keys
        self.action_key = action_key

        self.zarr_path = zarr_path
        self.aug_zarr_path = aug_zarr_path
        self.augment = augment
        self.image_source = image_source
        self.controller_frame = controller_frame
        self.rotate_rotation_labels = rotate_rotation_labels
        self._rotate_rot_labels = rotate_rot_labels
        self.probe_results_path = probe_results_path
        self.norm_mode = norm_mode
        self.norm_spec_path = norm_spec_path
        self.world_frame_rotation = world_frame_rotation
        self.emit_angle_pair = emit_angle_pair
        self.naive_image_rotation = naive_image_rotation
        self.expected_render_source = expected_render_source
        self.probe_gs_geometry_path = probe_gs_geometry_path
        self.seed = seed

        self.episode_ends = np.asarray(episode_ends)
        self.angles_deg = angles_deg
        self.valid_mask = valid_mask
        self.p_base = p_base

        self._rng = None        # per-worker, lazy (see module docstring)
        self._rgb_cache = None  # (pid, base arrays, aug arrays), lazy per process

        print(
            f"[SE2AugZarrDataset] matched-budget: {len(self.seq_sampler)} "
            f"samples/epoch (augment={augment})"
        )

    # ── lazy handles / per-worker state ──────────────────────────────────────

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = info.seed if info is not None else self.seed
            self._rng = np.random.default_rng(seed)
        return self._rng

    def _rgb_arrays(self):
        """Open zarr image arrays lazily, once per process (fork/spawn safe)."""
        pid = os.getpid()
        if self._rgb_cache is None or self._rgb_cache[0] != pid:
            base_root = zarr.open(self.zarr_path, 'r')
            base = {k: base_root['data'][k] for k in self.rgb_keys}
            aug = None
            if self._aug_required:
                aug_root = zarr.open(self.aug_zarr_path, 'r')
                aug = {
                    k: [aug_root[f'images/{k}/angle_{j:02d}']
                        for j in range(len(self.angles_deg))]
                    for k in self.rgb_keys
                }
            self._rgb_cache = (pid, base, aug)
        return self._rgb_cache[1], self._rgb_cache[2]

    # ── dataset protocol ─────────────────────────────────────────────────────

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            keys=[self.action_key, *self.low_dim_keys],
        )
        val_set.train_mask = ~self.train_mask
        # val loss is computed at theta=0 in every arm (comparable across arms)
        val_set.augment = False
        val_set.emit_angle_pair = False
        val_set._rng = None
        return val_set

    def get_normalizer(self, **kwargs):
        """Load the frozen NormalizationSpec -- NEVER fits (D3)."""
        spec = load_spec(self.norm_spec_path)
        assert spec['mode'] == self.norm_mode, (
            f"spec mode '{spec['mode']}' != dataset norm_mode '{self.norm_mode}'"
        )
        assert bool(spec['world_frame_rotation']) == bool(self.world_frame_rotation)
        verify_fingerprint(spec, self.replay_buffer, self.zarr_path)
        # Coverage guard: the obs encoders silently pass RAW values through for
        # keys the normalizer lacks (print-only warning) -- fail loudly here.
        needed = {'action', *self.numeric_obs_keys, *self.rgb_keys}
        missing = sorted(needed - set(spec['keys']))
        if missing:
            raise RuntimeError(
                f"NormalizationSpec '{self.norm_spec_path}' is missing keys "
                f"{missing} required by this dataset's obs_keys; rebuild it with "
                f"scripts/build_normalization_spec.py --obs_keys ..."
            )
        return normalizer_from_spec(spec)

    def __len__(self):
        # independent of `augment` by construction: one sampler window per item
        return len(self.seq_sampler)

    # ── sample assembly ──────────────────────────────────────────────────────

    def _sample_to_data(self, sample):
        To = self.n_obs_steps
        Ta = self.n_action_steps

        obs = {}
        for k in self.numeric_obs_keys:
            if sample[k].dtype.kind == 'f': # floatX -> float32
                obs[k] = sample[k][:To].astype(np.float32)
            else:                           # remain the same dtype
                obs[k] = sample[k][:To]
        for k in self.text_obs_keys:
            obs[k] = sample[k][0]           # every frame is the same

        start = max(To - 1, 0)
        end = start + Ta
        act = sample[self.action_key][start:end].astype(np.float32)
        assert np.allclose(act, sample[self.action_key][-Ta:]), "action mismatch"

        return {'obs': obs, 'action': act}

    def _obs_frame_indices(self, buffer_start, buffer_end, sample_start):
        """Global frame index of each of the To obs positions, replicating the
        SequenceSampler edge padding: sample position i holds buffer frame
        clip(buffer_start + (i - sample_start), buffer_start, buffer_end - 1)."""
        idxs = []
        for i in range(self.n_obs_steps):
            g = buffer_start + (i - sample_start)
            idxs.append(int(min(max(g, buffer_start), buffer_end - 1)))
        return idxs

    def _fetch_images(self, frame_idxs, k, theta):
        """(To, H, W, 3) uint8 per rgb key, from angle-k arrays (or base)."""
        base, aug = self._rgb_arrays()
        out = {}
        for key in self.rgb_keys:
            if self.naive_image_rotation and self.augment:
                # deliberately-wrong control arm: in-plane rotation of the BASE
                # agentview pixels (no re-render); wrist cam left unchanged.
                frames = np.stack([base[key][g] for g in frame_idxs], axis=0)
                if k != 0 and 'eye_in_hand' not in key:
                    from scipy.ndimage import rotate as nd_rotate
                    frames = np.stack([
                        nd_rotate(f, np.degrees(theta), axes=(0, 1),
                                  reshape=False, order=1, mode='nearest')
                        for f in frames
                    ], axis=0)
            elif self.image_source == 'aug':
                arr = aug[key][k]
                frames = np.stack([arr[g] for g in frame_idxs], axis=0)
            else:
                frames = np.stack([base[key][g] for g in frame_idxs], axis=0)
            out[key] = np.ascontiguousarray(frames)
        return out

    def _build_item(self, data, frame_idxs, episode, k):
        theta = math.radians(float(self.angles_deg[k]))
        obs = dict(data['obs'])
        act = data['action']
        if k != 0:
            # RAW-space label rotation (D2): commutes exactly with the frozen
            # group-compatible normalizer applied inside the policy.
            act = rotate_action_chunk(act, theta, self._rotate_rot_labels)
            obs = rotate_proprio(obs, theta, self.p_base[episode, :2])
        obs.update(self._fetch_images(frame_idxs, k, theta))

        torch_obs = {}
        for key, v in obs.items():
            if isinstance(v, np.ndarray) and is_numeric_dtype(v):
                torch_obs[key] = torch.from_numpy(v)
            elif isinstance(v, bytes):
                torch_obs[key] = v.decode('utf-8')
            else:
                torch_obs[key] = v
        return {'obs': torch_obs, 'action': torch.from_numpy(act)}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buffer_start, buffer_end, sample_start, _ = self.seq_sampler.indices[idx]
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        episode = int(np.searchsorted(self.episode_ends, buffer_start, side='right'))
        frame_idxs = self._obs_frame_indices(
            int(buffer_start), int(buffer_end), int(sample_start))

        # exactly one angle per item; k=0 (theta=0) when augmentation is off
        k = 0
        if self.augment:
            valid = np.nonzero(self.valid_mask[episode])[0]
            k = int(self._get_rng().choice(valid))

        item = self._build_item(data, frame_idxs, episode, k)

        # Phase-2 scaffold (unused by training configs): a second independently
        # drawn angle for the paired-consistency loss; collate with
        # oat.dataset.se2_collate.paired_angle_collate.
        if self.emit_angle_pair and self.augment:
            valid = np.nonzero(self.valid_mask[episode])[0]
            k2 = int(self._get_rng().choice(valid))
            pair = self._build_item(data, frame_idxs, episode, k2)
            pair['theta'] = math.radians(float(self.angles_deg[k2]))
            item['theta'] = math.radians(float(self.angles_deg[k]))
            item['pair'] = pair

        return item

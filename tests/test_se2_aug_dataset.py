"""Tests for ``oat.dataset.se2_aug_zarr_dataset`` + ``oat.dataset.se2_collate``
on a tiny synthetic base/aug zarr pair.

Provenance is stamped into pixels so image routing is observable: pixel
``[0, 0, 0]`` carries the global frame index in BOTH zarrs, pixel ``[0, 0, 1]``
carries 255 in the base zarr and the angle index k in the aug zarr. Key
asserts: matched budget across augment on/off, validity-respecting angle
sampling, exact raw-space label rotation (vs the real ``se2_transforms``
functions), sampler edge padding, the frozen-spec ``get_normalizer`` guards,
probe gating (D7), val-split disjointness, and paired-angle collate shapes.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.dataset.se2_aug_zarr_dataset import SE2AugZarrDataset
from oat.dataset.se2_collate import paired_angle_collate
from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    MODE_PER_DIM_MINMAX,
    build_group_compatible_normalizer,
    fingerprint_replay_buffer,
    load_spec,
    normalizer_from_spec,
    save_spec,
    spec_from_normalizer,
)
from oat.equi.se2_transforms import rotate_action_chunk, rotate_proprio
from oat.model.common.normalizer import LinearNormalizer

EP_LEN = 30
N_EPISODES = 3
N_STEPS = EP_LEN * N_EPISODES
H = W = 16
RGB_KEYS = ["agentview_rgb", "robot0_eye_in_hand_rgb"]
LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "task_uid"]
OBS_KEYS = RGB_KEYS + LOW_DIM_KEYS
ANGLES_DEG = np.array([0.0, 10.0, -10.0])
INVALID_EP, INVALID_ANGLE = 1, 2  # exactly one rejected (episode, angle != 0)
BASE_MARK = 255                   # base-zarr provenance stamp (pixel [0,0,1])
To, Ta = 2, 16
# window count per episode = ep_len - seq_len + pad_before + pad_after + 1
# = EP_LEN with the ZarrDataset pad geometry (pad_before=To-1, pad_after=Ta-1)


def _stamped(frames: np.ndarray, mark: int) -> np.ndarray:
    """(n, H, W, 3) uint8: pixel [0,0,0] = global frame idx, [0,0,1] = mark."""
    img = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
    img[:, 0, 0, 0] = frames
    img[:, 0, 0, 1] = mark
    return img


@pytest.fixture(scope="session")
def se2fx(tmp_path_factory):
    """Tiny base zarr + matching aug zarr + both frozen norm specs + probe file."""
    root = tmp_path_factory.mktemp("se2_aug_dataset")
    rng = np.random.default_rng(0)

    # (1) base zarr in the real layout (data/ + meta/episode_ends)
    rb = ReplayBuffer.create_empty_numpy()
    for e in range(N_EPISODES):
        frames = np.arange(e * EP_LEN, (e + 1) * EP_LEN)
        quat = rng.normal(size=(EP_LEN, 4))
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        rb.add_episode({
            "action": rng.uniform(-1.0, 1.0, size=(EP_LEN, 7)).astype(np.float32),
            "agentview_rgb": _stamped(frames, BASE_MARK),
            "robot0_eye_in_hand_rgb": _stamped(frames, BASE_MARK),
            "robot0_eef_pos": rng.uniform(-0.5, 0.5, size=(EP_LEN, 3)).astype(np.float32),
            "robot0_eef_quat": quat.astype(np.float32),
            "robot0_gripper_qpos": rng.uniform(0.0, 0.04, size=(EP_LEN, 2)).astype(np.float32),
            "task_uid": np.full((EP_LEN, 1), float(e), dtype=np.float32),
        })
    base_zarr = str(root / "base.zarr")
    rb.save_to_path(base_zarr)
    episode_ends = np.asarray(rb.episode_ends[:]).copy()

    # (2) matching aug zarr: K=3 angles, angle idx stamped into pixel [0,0,1]
    valid_mask = np.ones((N_EPISODES, len(ANGLES_DEG)), dtype=bool)
    valid_mask[INVALID_EP, INVALID_ANGLE] = False
    p_base = np.array([
        [-0.15, 0.05, 0.90],
        [-0.12, 0.02, 0.90],
        [-0.18, -0.03, 0.90],
    ])
    aug_zarr = str(root / "aug.zarr")
    aug = zarr.open(aug_zarr, mode="w")
    meta = aug.create_group("meta")
    meta.array("angles_deg", ANGLES_DEG.astype(np.float64))
    meta.array("episode_ends", episode_ends)
    meta.array("valid_mask", valid_mask)
    meta.array("p_base", p_base)
    meta.array("done_mask", np.ones_like(valid_mask))
    images = aug.create_group("images")
    for key in RGB_KEYS:
        g = images.create_group(key)
        for j in range(len(ANGLES_DEG)):
            g.array(f"angle_{j:02d}", _stamped(np.arange(N_STEPS), j),
                    chunks=(1, H, W, 3))

    # (3) both frozen norm specs via the real builder pipeline
    fitted = LinearNormalizer()
    fitted.fit(
        {"action": rb["action"], **{k: rb[k] for k in OBS_KEYS}},
        last_n_dims=1, mode="limits",
    )
    fp = fingerprint_replay_buffer(rb, base_zarr)
    gc = build_group_compatible_normalizer(fitted, rgb_keys=RGB_KEYS)
    spec_gc = str(root / "norm_spec_group_compatible.json")
    save_spec(spec_from_normalizer(gc, MODE_GROUP_COMPATIBLE, fp), spec_gc)
    spec_pd = str(root / "norm_spec_per_dim_minmax.json")
    save_spec(spec_from_normalizer(fitted, MODE_PER_DIM_MINMAX, fp), spec_pd)

    # (4) controller-frame probe result (gates world-frame rotation labels)
    probe_json = str(root / "probe_results.json")
    with open(probe_json, "w") as f:
        json.dump({"controller_frame_rot": "world",
                   "controller_frame_pos": "world",
                   "pass": True}, f)

    return SimpleNamespace(
        base_zarr=base_zarr,
        aug_zarr=aug_zarr,
        episode_ends=episode_ends,
        valid_mask=valid_mask,
        p_base=p_base,
        spec_gc=spec_gc,
        spec_pd=spec_pd,
        probe_json=probe_json,
    )


def make_ds(fx, **overrides) -> SE2AugZarrDataset:
    kwargs = dict(
        zarr_path=fx.base_zarr,
        obs_keys=list(OBS_KEYS),
        n_obs_steps=To,
        n_action_steps=Ta,
        val_ratio=0.0,
        aug_zarr_path=fx.aug_zarr,
        image_source="aug",
        norm_spec_path=fx.spec_gc,
        probe_results_path=fx.probe_json,
    )
    kwargs.update(overrides)
    return SE2AugZarrDataset(**kwargs)


def episodes_of(ds) -> np.ndarray:
    """Episode index of every sampler window (from its buffer_start)."""
    return np.searchsorted(ds.episode_ends, ds.seq_sampler.indices[:, 0],
                           side="right")


def expected_obs_frames(ds, idx):
    """Independent re-derivation of the To obs frame indices for window idx."""
    bs, be, ss, _ = (int(v) for v in ds.seq_sampler.indices[idx])
    return [int(np.clip(bs + (i - ss), bs, be - 1)) for i in range(To)]


def stamp_angle(item, key="agentview_rgb") -> int:
    return int(item["obs"][key][0, 0, 0, 1])


# ── matched budget ───────────────────────────────────────────────────────────

def test_matched_budget(se2fx):
    ds_aug = make_ds(se2fx, augment=True)
    ds_off = make_ds(se2fx, augment=False)
    assert len(ds_aug) == len(ds_off) == N_EPISODES * EP_LEN


# ── augment off ──────────────────────────────────────────────────────────────

def test_augment_off_base_images_and_raw_labels(se2fx, tmp_path):
    # aug zarr is NOT touched (and may not exist) in this configuration
    ds = make_ds(se2fx, augment=False, image_source="base",
                 aug_zarr_path=str(tmp_path / "nonexistent.zarr"))

    # a full interior window (no edge padding on either side)
    idxs = ds.seq_sampler.indices
    idx = int(np.nonzero((idxs[:, 2] == 0) & (idxs[:, 3] == ds.seq_len))[0][0])
    bs = int(idxs[idx, 0])
    item = ds[idx]

    for key in RGB_KEYS:
        img = item["obs"][key].numpy()
        assert img.shape == (To, H, W, 3) and img.dtype == np.uint8
        np.testing.assert_array_equal(img[:, 0, 0, 0], [bs, bs + 1])
        # base-zarr provenance, not the aug zarr's angle-0 render
        assert (img[:, 0, 0, 1] == BASE_MARK).all()

    # labels are the raw buffer slices, untouched
    rb = ds.replay_buffer
    np.testing.assert_array_equal(
        item["action"].numpy(), rb["action"][bs + To - 1: bs + To - 1 + Ta])
    for key in LOW_DIM_KEYS:
        np.testing.assert_array_equal(item["obs"][key].numpy(), rb[key][bs: bs + To])


# ── augment on ───────────────────────────────────────────────────────────────

def test_augment_respects_valid_mask(se2fx):
    ds = make_ds(se2fx, augment=True)
    eps = episodes_of(ds)

    # many draws over the invalidated episode: the rejected angle never appears
    inv_windows = np.nonzero(eps == INVALID_EP)[0]
    ks = {stamp_angle(ds[int(i)]) for _ in range(10) for i in inv_windows}
    valid = set(np.nonzero(se2fx.valid_mask[INVALID_EP])[0].tolist())
    assert INVALID_ANGLE not in ks
    assert ks == valid  # and every remaining valid angle is actually drawn

    # a fully-valid episode eventually sees all K angles
    ok_windows = np.nonzero(eps == 2)[0]
    ks2 = {stamp_angle(ds[int(i)]) for _ in range(5) for i in ok_windows}
    assert ks2 == set(range(len(ANGLES_DEG)))


def test_augment_rotates_labels_and_serves_angle_images(se2fx):
    ds = make_ds(se2fx, augment=True)
    ref = make_ds(se2fx, augment=False)  # same geometry, k=0 labels
    eps = episodes_of(ds)
    idx = int(np.nonzero(eps == 2)[0][3])  # interior window, fully-valid episode
    frames = expected_obs_frames(ds, idx)

    ref_item = ref[idx]
    raw_act = ref_item["action"].numpy()
    raw_obs = {k: ref_item["obs"][k].numpy() for k in LOW_DIM_KEYS}

    seen = set()
    for _ in range(40):
        item = ds[idx]
        k = stamp_angle(item)
        seen.add(k)
        theta = math.radians(float(ds.angles_deg[k]))

        # images come from the sampled angle's arrays, correct frames
        for key in RGB_KEYS:
            img = item["obs"][key].numpy()
            np.testing.assert_array_equal(img[:, 0, 0, 0], frames)
            assert (img[:, 0, 0, 1] == k).all()

        # labels match the real transform functions exactly (raw space)
        exp_act = raw_act if k == 0 else rotate_action_chunk(raw_act, theta, True)
        np.testing.assert_array_equal(item["action"].numpy(), exp_act)
        exp_obs = raw_obs if k == 0 else rotate_proprio(
            raw_obs, theta, se2fx.p_base[2, :2])
        for key in LOW_DIM_KEYS:
            np.testing.assert_array_equal(item["obs"][key].numpy(), exp_obs[key])

    assert {1, 2} <= seen  # non-zero angles were actually exercised above


# ── sampler edge padding ─────────────────────────────────────────────────────

def test_edge_padding_repeats_first_frame(se2fx):
    ds = make_ds(se2fx, augment=False)  # angle-0 aug images
    idxs = ds.seq_sampler.indices
    padded = np.nonzero(idxs[:, 2] > 0)[0]
    assert len(padded) == N_EPISODES  # pad_before=1: one padded window per episode
    for i in padded:
        bs = int(idxs[i, 0])
        item = ds[int(i)]
        for key in RGB_KEYS:
            # both obs frames are the episode's FIRST frame, repeated
            np.testing.assert_array_equal(
                item["obs"][key][:, 0, 0, 0].numpy(), [bs, bs])
        np.testing.assert_array_equal(
            item["obs"]["robot0_eef_pos"][0].numpy(),
            item["obs"]["robot0_eef_pos"][1].numpy())


# ── get_normalizer: frozen-spec guards ───────────────────────────────────────

@pytest.mark.parametrize("mode", [MODE_GROUP_COMPATIBLE, MODE_PER_DIM_MINMAX])
def test_get_normalizer_matches_frozen_spec(se2fx, mode):
    spec_path = se2fx.spec_gc if mode == MODE_GROUP_COMPATIBLE else se2fx.spec_pd
    ds = make_ds(se2fx, augment=False, norm_mode=mode, norm_spec_path=spec_path)
    normalizer = ds.get_normalizer()

    ref = normalizer_from_spec(load_spec(spec_path))
    assert set(normalizer.params_dict.keys()) == set(ref.params_dict.keys())
    for key in ref.params_dict.keys():
        for name in ("scale", "offset"):
            np.testing.assert_array_equal(
                normalizer.params_dict[key][name].numpy(),
                ref.params_dict[key][name].numpy())

    if mode == MODE_GROUP_COMPATIBLE:
        sc = normalizer.params_dict["action"]["scale"].numpy()
        off = normalizer.params_dict["action"]["offset"].numpy()
        assert sc[0] == sc[1] and off[0] == off[1] == 0.0  # tied rho1 xy


def test_ctor_rejects_wrong_norm_mode(se2fx):
    # group_compatible spec on disk, dataset configured for per_dim_minmax
    with pytest.raises(AssertionError, match="mode"):
        make_ds(se2fx, augment=False, norm_mode=MODE_PER_DIM_MINMAX)


def test_get_normalizer_rejects_tampered_fingerprint(se2fx, tmp_path):
    with open(se2fx.spec_gc) as f:
        spec = json.load(f)
    spec["fingerprint"]["action_sha1"] = "deadbeef" * 5
    bad_path = str(tmp_path / "tampered_spec.json")
    with open(bad_path, "w") as f:
        json.dump(spec, f)

    ds = make_ds(se2fx, augment=False, norm_spec_path=bad_path)  # ctor is fine
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        ds.get_normalizer()


# ── probe gating (D7) ────────────────────────────────────────────────────────

def test_probe_gating(se2fx, tmp_path):
    missing = str(tmp_path / "missing_probe.json")
    with pytest.raises(RuntimeError, match="probe"):
        make_ds(se2fx, augment=True, probe_results_path=missing)

    wrong = str(tmp_path / "ee_probe.json")
    with open(wrong, "w") as f:
        json.dump({"controller_frame_rot": "ee", "pass": True}, f)
    with pytest.raises(RuntimeError, match="controller_frame_rot"):
        make_ds(se2fx, augment=True, probe_results_path=wrong)

    # rot verdict is 'world' but the probe run FAILED overall: still gated
    failed = str(tmp_path / "failed_probe.json")
    with open(failed, "w") as f:
        json.dump({"controller_frame_rot": "world",
                   "controller_frame_pos": "inconclusive",
                   "pass": False}, f)
    with pytest.raises(RuntimeError, match="pass"):
        make_ds(se2fx, augment=True, probe_results_path=failed)

    ds = make_ds(se2fx, augment=True)  # with the probe file: constructs
    assert ds._rotate_rot_labels is True
    # not gated when rotation labels are off
    ds2 = make_ds(se2fx, augment=True, probe_results_path=missing,
                  rotate_rotation_labels=False)
    assert ds2._rotate_rot_labels is False


# ── validation split ─────────────────────────────────────────────────────────

def test_validation_dataset_unaugmented_and_disjoint(se2fx):
    ds = make_ds(se2fx, augment=True, val_ratio=0.34)  # exactly 1 val episode
    val = ds.get_validation_dataset()
    assert val.augment is False and val.emit_angle_pair is False

    train_eps = set(episodes_of(ds).tolist())
    val_eps = set(episodes_of(val).tolist())
    assert train_eps.isdisjoint(val_eps)
    assert train_eps | val_eps == set(range(N_EPISODES))
    assert len(val_eps) == 1
    assert len(ds) + len(val) == N_EPISODES * EP_LEN  # windows merely re-split

    assert stamp_angle(val[0]) == 0  # val loss is computed at theta=0


# ── paired-angle collate (Phase-2 scaffold) ──────────────────────────────────

def test_paired_collate_shapes(se2fx):
    ds = make_ds(se2fx, augment=True, emit_angle_pair=True)
    eps = episodes_of(ds)
    pick = [int(np.nonzero(eps == e)[0][j])
            for e, j in ((0, 0), (0, 5), (1, 2), (2, 7))]
    out = paired_angle_collate([ds[i] for i in pick])
    B = len(pick)

    for side in (out, out["pair"]):
        assert side["action"].shape == (B, Ta, 7)
        assert side["action"].dtype == torch.float32
        for key in RGB_KEYS:
            assert side["obs"][key].shape == (B, To, H, W, 3)
            assert side["obs"][key].dtype == torch.uint8
        assert side["obs"]["robot0_eef_pos"].shape == (B, To, 3)
        assert side["theta"].shape == (B,)
        assert side["theta"].dtype == torch.float32
    assert "pair" not in out["pair"]

    # plain items (no 'theta'/'pair') pass through like the default collate
    ds_off = make_ds(se2fx, augment=False)
    out_off = paired_angle_collate([ds_off[i] for i in pick])
    assert "pair" not in out_off and "theta" not in out_off
    assert out_off["action"].shape == (B, Ta, 7)

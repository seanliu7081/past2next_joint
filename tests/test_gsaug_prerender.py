"""Tests for the GS pre-render integration surface (plan §6.5 / §7, M5 gate):

1. ``scripts/prerender_se2_aug.py`` argparse contract — the GS flags exist,
   their guard rails fire (gs + crosscheck without --oracle-zarr;
   --hybrid-zero-from without gs), and oracle-mode defaults are unchanged.
   The script is imported via ``spec_from_file_location`` (its ``__main__``
   guard keeps chdir/sys.path edits out); no sim object is ever created.
2. ``SE2AugZarrDataset.expected_render_source`` gating (plan §8.1 consumer
   gate, D7 pattern) on a tiny synthetic base/aug zarr pair built locally —
   fixtures are intentionally NOT shared with test_se2_aug_dataset.py.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numcodecs
import numpy as np
import pytest
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.dataset.se2_aug_zarr_dataset import SE2AugZarrDataset
from oat.equi.normalization import (
    MODE_GROUP_COMPATIBLE,
    build_group_compatible_normalizer,
    fingerprint_replay_buffer,
    save_spec,
    spec_from_normalizer,
)
from oat.model.common.normalizer import LinearNormalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PRERENDER_PATH = REPO_ROOT / "scripts" / "prerender_se2_aug.py"


# ── argparse contract ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prerender():
    """Import the script as a module. Heavy (pulls libero/robosuite for its
    module-level imports) but sim-free: the ``__main__`` guard keeps the
    chdir/sys.path side effects out and nothing renders."""
    spec = importlib.util.spec_from_file_location(
        "prerender_se2_aug_under_test", str(PRERENDER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse(prerender, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prerender_se2_aug.py", *argv])
    return prerender.parse_args()


def test_defaults_unchanged(prerender, monkeypatch):
    args = parse(prerender, monkeypatch, [])
    # new flags default to full-oracle behavior (zero behavior change)
    assert args.renderer == "oracle"
    assert args.gs_assets_dir == "data/libero/gs_assets"
    assert args.oracle_zarr is None
    assert args.oracle_crosscheck is True
    assert args.hybrid_zero_from is None
    # pre-existing defaults untouched
    assert args.base_zarr == "data/libero/libero10_N500.zarr"
    assert args.out == "data/libero/libero10_N500_se2aug.zarr"
    assert args.image_size == 128
    assert args.resume is True
    assert "0" in args.angles.split(",")


def test_gs_requires_oracle_zarr(prerender, monkeypatch):
    with pytest.raises(SystemExit):
        parse(prerender, monkeypatch, ["--renderer", "gs"])


def test_gs_with_oracle_zarr_parses(prerender, monkeypatch):
    args = parse(prerender, monkeypatch,
                 ["--renderer", "gs", "--oracle-zarr", "oracle.zarr"])
    assert args.renderer == "gs" and args.oracle_zarr == "oracle.zarr"
    assert args.oracle_crosscheck is True


def test_gs_no_crosscheck_smoke_path(prerender, monkeypatch):
    args = parse(prerender, monkeypatch,
                 ["--renderer", "gs", "--no-oracle-crosscheck"])
    assert args.renderer == "gs" and args.oracle_crosscheck is False
    assert args.oracle_zarr is None


def test_hybrid_zero_requires_gs(prerender, monkeypatch):
    with pytest.raises(SystemExit):
        parse(prerender, monkeypatch, ["--hybrid-zero-from", "oracle.zarr"])
    args = parse(prerender, monkeypatch,
                 ["--renderer", "gs", "--oracle-zarr", "oracle.zarr",
                  "--hybrid-zero-from", "oracle.zarr"])
    assert args.hybrid_zero_from == "oracle.zarr"


def test_unknown_renderer_rejected(prerender, monkeypatch):
    with pytest.raises(SystemExit):
        parse(prerender, monkeypatch, ["--renderer", "nerf"])


def test_gs_constants_pinned(prerender):
    # plan §7.4: theta=0 gate is a gross-error gate (25) in GS mode, 5 oracle
    assert prerender.PIXEL_DIFF_FAIL == 5.0
    assert prerender.PIXEL_DIFF_FAIL_GS == 25.0
    assert prerender.PIXEL_DIFF_WARN_GS == 20.0
    # explicit zarr-key -> mujoco-camera mapping for GSCompositeRenderer
    assert dict(prerender.GS_CAMERAS) == {
        "agentview_rgb": "agentview",
        "robot0_eye_in_hand_rgb": "robot0_eye_in_hand",
    }


# ── dataset render-source gate (plan §8.1, D7 pattern) ───────────────────────

EP_LEN = 8
N_EPISODES = 2
N_STEPS = EP_LEN * N_EPISODES
H = W = 8
RGB_KEYS = ["agentview_rgb", "robot0_eye_in_hand_rgb"]
OBS_KEYS = RGB_KEYS + ["robot0_eef_pos"]
ANGLES_DEG = np.array([0.0, 15.0])


@pytest.fixture(scope="module")
def gatefx(tmp_path_factory):
    """Tiny base zarr + frozen norm spec (real builders); aug zarrs are built
    per-test via ``build_aug_zarr`` so provenance metadata can vary."""
    root = tmp_path_factory.mktemp("gsaug_gate")
    rng = np.random.default_rng(0)

    rb = ReplayBuffer.create_empty_numpy()
    for _ in range(N_EPISODES):
        rb.add_episode({
            "action": rng.uniform(-1, 1, size=(EP_LEN, 7)).astype(np.float32),
            "agentview_rgb": np.zeros((EP_LEN, H, W, 3), dtype=np.uint8),
            "robot0_eye_in_hand_rgb": np.zeros((EP_LEN, H, W, 3), dtype=np.uint8),
            "robot0_eef_pos": rng.uniform(-0.5, 0.5, size=(EP_LEN, 3)).astype(np.float32),
        })
    base_zarr = str(root / "base.zarr")
    rb.save_to_path(base_zarr)

    fitted = LinearNormalizer()
    fitted.fit({"action": rb["action"], **{k: rb[k] for k in OBS_KEYS}},
               last_n_dims=1, mode="limits")
    gc = build_group_compatible_normalizer(fitted, rgb_keys=RGB_KEYS)
    fp = fingerprint_replay_buffer(rb, base_zarr)
    spec_path = str(root / "norm_spec_group_compatible.json")
    save_spec(spec_from_normalizer(gc, MODE_GROUP_COMPATIBLE, fp), spec_path)

    passing_probe = str(root / "probe_gs_geometry.json")
    with open(passing_probe, "w") as f:
        json.dump({"probe": "gs_geometry", "pass": True}, f)
    failing_probe = str(root / "probe_gs_geometry_fail.json")
    with open(failing_probe, "w") as f:
        json.dump({"probe": "gs_geometry", "pass": False}, f)

    return SimpleNamespace(
        root=root,
        base_zarr=base_zarr,
        spec=spec_path,
        episode_ends=np.asarray(rb.episode_ends[:]).copy(),
        passing_probe=passing_probe,
        failing_probe=failing_probe,
        missing_probe=str(root / "no_such_probe.json"),
    )


def build_aug_zarr(fx, name, render_source=None, via="attrs") -> str:
    """Aug zarr in the pre-render layout; ``render_source`` provenance written
    to root attrs and/or the meta string array per ``via`` (matching the two
    channels scripts/prerender_se2_aug.py writes). None = legacy zarr."""
    path = str(fx.root / f"{name}.zarr")
    aug = zarr.open(path, mode="w")
    meta = aug.create_group("meta")
    meta.array("angles_deg", ANGLES_DEG.astype(np.float64))
    meta.array("episode_ends", fx.episode_ends)
    meta.array("valid_mask", np.ones((N_EPISODES, len(ANGLES_DEG)), dtype=bool))
    meta.array("p_base", np.tile([-0.15, 0.05, 0.9], (N_EPISODES, 1)))
    images = aug.create_group("images")
    for key in RGB_KEYS:
        g = images.create_group(key)
        for j in range(len(ANGLES_DEG)):
            g.array(f"angle_{j:02d}",
                    np.zeros((N_STEPS, H, W, 3), dtype=np.uint8),
                    chunks=(1, H, W, 3))
    if render_source is not None:
        if via in ("attrs", "both"):
            aug.attrs["render_source"] = render_source
        if via in ("meta", "both"):
            meta.array("render_source",
                       np.array([render_source], dtype=object),
                       dtype=object, object_codec=numcodecs.VLenUTF8())
    return path


def make_ds(fx, aug_zarr_path, **overrides) -> SE2AugZarrDataset:
    kwargs = dict(
        zarr_path=fx.base_zarr,
        obs_keys=list(OBS_KEYS),
        n_obs_steps=2,
        n_action_steps=4,
        augment=False,
        image_source="aug",
        aug_zarr_path=aug_zarr_path,
        norm_spec_path=fx.spec,
    )
    kwargs.update(overrides)
    return SE2AugZarrDataset(**kwargs)


def test_gate_none_is_todays_behavior(gatefx):
    aug = build_aug_zarr(gatefx, "aug_plain")
    ds = make_ds(gatefx, aug)   # no expected_render_source: no gate at all
    assert ds.expected_render_source is None
    assert len(ds) == N_EPISODES * EP_LEN


def test_gate_absent_metadata_means_oracle(gatefx):
    aug = build_aug_zarr(gatefx, "aug_legacy")
    ds = make_ds(gatefx, aug, expected_render_source="oracle")
    assert ds.expected_render_source == "oracle"


def test_gate_gs_missing_probe_refused(gatefx):
    aug = build_aug_zarr(gatefx, "aug_gs_noprobe", render_source="gs")
    with pytest.raises(RuntimeError, match="probe_gs_geometry"):
        make_ds(gatefx, aug, expected_render_source="gs",
                probe_gs_geometry_path=gatefx.missing_probe)


def test_gate_gs_failing_probe_refused(gatefx):
    aug = build_aug_zarr(gatefx, "aug_gs_failprobe", render_source="gs")
    with pytest.raises(RuntimeError, match="pass=False"):
        make_ds(gatefx, aug, expected_render_source="gs",
                probe_gs_geometry_path=gatefx.failing_probe)


def test_gate_gs_with_passing_probe_constructs(gatefx):
    aug = build_aug_zarr(gatefx, "aug_gs_ok", render_source="gs", via="both")
    ds = make_ds(gatefx, aug, expected_render_source="gs",
                 probe_gs_geometry_path=gatefx.passing_probe)
    assert ds.expected_render_source == "gs"
    # get_validation_dataset's copy carries the gate config unchanged
    val = ds.get_validation_dataset()
    assert val.expected_render_source == "gs"
    assert val.probe_gs_geometry_path == gatefx.passing_probe


def test_gate_meta_array_fallback(gatefx):
    # provenance only in meta/render_source (no root attr): still recognized
    aug = build_aug_zarr(gatefx, "aug_gs_meta", render_source="gs", via="meta")
    ds = make_ds(gatefx, aug, expected_render_source="gs",
                 probe_gs_geometry_path=gatefx.passing_probe)
    assert ds.expected_render_source == "gs"


def test_gate_mismatch_names_both_sources(gatefx):
    # legacy zarr (implicit oracle) fed to a GS arm
    aug = build_aug_zarr(gatefx, "aug_mismatch1")
    with pytest.raises(RuntimeError,
                       match=r"render_source='oracle'.*'gs'"):
        make_ds(gatefx, aug, expected_render_source="gs",
                probe_gs_geometry_path=gatefx.passing_probe)

    # gs zarr fed to an arm expecting the hybrid variant
    aug2 = build_aug_zarr(gatefx, "aug_mismatch2", render_source="gs")
    with pytest.raises(RuntimeError,
                       match=r"render_source='gs'.*'gs_hybrid0'"):
        make_ds(gatefx, aug2, expected_render_source="gs_hybrid0",
                probe_gs_geometry_path=gatefx.passing_probe)

    # gs zarr fed to an oracle arm
    aug3 = build_aug_zarr(gatefx, "aug_mismatch3", render_source="gs")
    with pytest.raises(RuntimeError,
                       match=r"render_source='gs'.*'oracle'"):
        make_ds(gatefx, aug3, expected_render_source="oracle")


def test_gate_inactive_without_aug_zarr(gatefx, tmp_path):
    # augment=False + image_source='base': the aug zarr is never opened, so
    # the gate must not fire (and the zarr need not exist) — zero behavior
    # change for arms that do not touch pre-rendered images.
    ds = make_ds(gatefx, str(tmp_path / "nonexistent.zarr"),
                 image_source="base", expected_render_source="gs",
                 probe_gs_geometry_path=gatefx.missing_probe)
    assert ds.expected_render_source == "gs"
    assert len(ds) == N_EPISODES * EP_LEN

"""GS geometry probe (plan §8.1) -- GATING.

Samples ``--n`` random VALID (episode, frame, angle) triples from the GS aug
zarr's ``meta/valid_mask`` (mirroring probe 2's sampling; theta=0 is excluded
because the wrist transform-stack check is degenerate there) and gates three
geometric properties of the GS composite renderer against the MuJoCo oracle:

1. **Silhouette IoU** per component: each component's Gaussians are rasterized
   ALONE (``GSCompositeRenderer.render_component_alpha`` -- a probe-only path,
   composite frames stay single-pass, G2), thresholded at alpha >= 0.5, and
   compared against the oracle seg mask of the same rewritten state (raw
   ``mujoco.Renderer`` seg under the measured F2b vis flags + the F6 geom map).
   PASS: p5 >= ``--iou_obj_p5`` (objects), p5 >= ``--iou_robot_p5`` (whole-robot
   union). Components whose oracle mask is smaller than ``--min_mask_px``
   (fully occluded / out of frame -- no silhouette to measure) are skipped and
   counted. An occlusion-corrected IoU (occluder pixels removed from the GS
   mask) is recorded as a non-gated diagnostic.
2. **EEF projection**: the ``robot0_right_hand`` body origin of the FORWARDED
   rewritten state (G3: set_state + forward, no MuJoCo render feeds the GS
   path) is projected through the renderer's camera math (``fovy_to_K`` +
   ``mujoco_cam_to_w2c`` with the measured F1 flip) and compared against the
   hand's oracle seg mask by containment + distance-to-mask (mask centroids are
   biased -- F1 pattern). PASS: median <= ``--eef_median_px``,
   p95 <= ``--eef_p95_px``. The eef grip-site anchor (plan §8.1 wording) is
   recorded alongside as ``eef_site_px`` (NOT gated). CAVEAT (measured on
   oracle GT over 103 demo triples / 8 tasks before GS assets existed): this
   metric has a structural floor -- the hand origin projects onto arm-link
   pixels ~3 px off its subtree silhouette and fixtures occlude the hand in
   the tail (hand anchor median 3.5 / p95 12 px; site anchor 2.6 / 7 px) --
   so the plan-default 2/4 px gate fails even for a perfect renderer; real
   camera-chain bugs sit at tens of px. Adjust the flags accordingly.
3. **Wrist transform-stack check**: under the group action the wrist camera
   co-rotates with (movables ∪ robot), so within that mask the GS wrist render
   at theta must equal the GS wrist render of the SAME frame at theta=0 up to
   rasterization noise -- one check that catches SH-rotation, covariance-quat
   and camera-extrinsics bugs at once (G5). PASS: masked PSNR >=
   ``--wrist_psnr_min`` for every triple (min gated; p5/median reported). The
   same masked PSNR of ORACLE re-renders is reported (NOT gated): the oracle's
   deficit from perfection is the fixed-light shading variation baked GS
   appearance structurally cannot reproduce.

delta is COPIED from the GS zarr's ``meta/state_offset`` (itself copied from
the oracle zarr, G4) -- never recalibrated here.

Orientation (G7, measured -- see data/libero/gs_render_facts.json):
* F2b measured MAD 0.0 between the raw ``mujoco.Renderer`` and
  ``np.flip(obs, axis=0)`` -- the exact expression dataset conversion stores --
  so RAW renderer orientation == stored-zarr (dataset) orientation, and F1
  pins raw row 0 == OpenCV top, so projected (u, v) indexes the raw seg
  directly. EEF math and all oracle-vs-oracle comparisons therefore run in raw
  orientation with no flip.
* ``GSCompositeRenderer`` transforms its output by the F2
  ``gsplat_flip_ud`` fact; seg masks compared against GS outputs (IoU, wrist
  mask) get the SAME facts-keyed transform, so GS-vs-seg comparisons stay
  aligned with whatever compose does. A per-task orientation anchor (theta=0
  GS agentview render vs the stored base frame, direct vs flipped MAD) is
  recorded so a global-flip inconsistency in that chain is visible in the
  report (the pre-render theta=0 MAD gate is what hard-fails on it).

PASS = all three gates. Exit code 0 iff PASS. The output JSON is what
``SE2AugZarrDataset(expected_render_source='gs*')`` gates on (D7 pattern).

Usage:
    export PATH=/home/haotian/miniforge3/envs/oat/bin:/usr/local/cuda/bin:$PATH
    MUJOCO_GL=egl python scripts/probes/probe_gs_geometry.py \
        --gs_zarr data/libero/libero10_N500_se2aug_gs.zarr --n 200
"""

import os

# must be set before any robosuite / mujoco / libero import
os.environ.setdefault("MUJOCO_GL", "egl")

if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.insert(0, ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import datetime
import json
import pathlib
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import h5py
import mujoco
import numpy as np
import zarr

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.common.replay_buffer import ReplayBuffer
from oat.env.libero.demo_alignment import match_episodes
from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.gsaug.capture import (
    body_subtree_ids,
    geom_ids_of_bodies,
    movable_geom_ids,
    robot_geom_ids,
)
from oat.gsaug.cameras import (
    facts_flip,
    facts_orientation_flip_ud,
    fovy_to_K,
    load_render_facts,
    mujoco_cam_to_w2c,
    project,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEOM_T = int(mujoco.mjtObj.mjOBJ_GEOM)

# zarr image key -> mujoco camera name (compose contract: explicit, no
# '_image'-suffix convention) -- must match scripts/prerender_se2_aug.py
GS_CAMERAS = OrderedDict(
    [
        ("agentview_rgb", "agentview"),
        ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand"),
    ]
)
AGENT_KEY = "agentview_rgb"
WRIST_KEY = "robot0_eye_in_hand_rgb"

# per-task obs-pipeline sanity bound: an oracle theta=0 re-render vs the stored
# base frame is normally MAD <~ 2; > 25 means a wrong delta or the measured
# EGL-context failure mode (robosuite obs re-renders come back from a wrong
# viewpoint once extra mujoco.Renderer contexts exist -- which is why ALL obs
# renders of a task run BEFORE the seg renderer is created below)
OBS_SANITY_MAD_MAX = 25.0
ALPHA_THRESH = 0.5


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def build_control_env(task_name: str, image_size: int, seed: int = 0) -> ControlEnv:
    """Build a ControlEnv directly, mirroring oat/env/libero/env.py:45-59."""
    libero_suite, task_suite_id, _ = task_name_to_suite_and_ids[task_name]
    task = benchmark.get_benchmark_dict()[libero_suite]().get_task(task_suite_id)
    env = ControlEnv(
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        camera_names=list(GS_CAMERAS.values()),
        camera_heights=image_size,
        camera_widths=image_size,
        has_renderer=False,
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    env.seed(seed)
    return env


def task_name_from_hdf5(hdf5_path: str) -> str:
    base = os.path.basename(hdf5_path)
    suffix = "_demo.hdf5"
    assert base.endswith(suffix), f"unexpected demo file name: {base}"
    task_name = base[: -len(suffix)]
    assert task_name in task_name_to_suite_and_ids, (
        f"hdf5 file name {base!r} does not map to a known LIBERO task")
    return task_name


class SegRenderer:
    """Exact-seg ``mujoco.Renderer`` under the measured F2b vis flags.

    Created from a context with ``model.vis.quality.offsamples = 0``: MSAA
    blends per-geom seg ID colors at silhouette edges into unrelated (possibly
    out-of-range) geom ids (measured; the ProbeRenderers.geo pattern from
    scripts/gsaug/probe_render_facts.py). F2b ``flags_off`` entries are
    ``mjRND_*`` scene flags from that probe's sweep space (measured empty),
    applied after ``update_scene`` exactly as the probe applied them.
    """

    def __init__(self, env, image_size: int, facts: dict):
        raw_model = env.sim.model._model
        # the offscreen framebuffer must fit the render (measured env fact)
        raw_model.vis.global_.offwidth = max(
            int(raw_model.vis.global_.offwidth), int(image_size))
        raw_model.vis.global_.offheight = max(
            int(raw_model.vis.global_.offheight), int(image_size))
        saved = int(raw_model.vis.quality.offsamples)
        raw_model.vis.quality.offsamples = 0  # read at MjrContext creation only
        try:
            self.r = mujoco.Renderer(raw_model, height=int(image_size),
                                     width=int(image_size))
        finally:
            raw_model.vis.quality.offsamples = saved
        flags = facts["F2b"]["flags"]
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = np.asarray(flags["geomgroup"], dtype=np.uint8)
        self.opt.sitegroup[:] = np.asarray(flags["sitegroup"], dtype=np.uint8)
        self._flags_off = [getattr(mujoco.mjtRndFlag, n)
                           for n in flags.get("flags_off", [])]
        self._default_flags = np.array(self.r.scene.flags, dtype=np.uint8).copy()

    def seg(self, env, cam_name: str) -> np.ndarray:
        """(H, W, 2) int32 [geom id, mjtObj type] at the CURRENT (forwarded)
        sim state, RAW renderer orientation (== dataset orientation, F2b)."""
        self.r.enable_segmentation_rendering()
        try:
            self.r.update_scene(env.sim.data._data, camera=cam_name,
                                scene_option=self.opt)
            self.r.scene.flags[:] = self._default_flags
            for f in self._flags_off:
                self.r.scene.flags[f] = 0
            seg2 = self.r.render().copy()
        finally:
            self.r.disable_segmentation_rendering()
        assert seg2.ndim == 3 and seg2.shape[2] == 2, (
            f"unexpected seg render shape {seg2.shape}; expected (H, W, 2)")
        bad = [int(t) for t in np.unique(seg2[..., 1])
               if int(t) not in (-1, GEOM_T)]
        assert not bad, (
            f"seg render contains non-geom object types {bad} -- the F2b flags "
            f"should hide sites/decor; re-run probe_render_facts (G7)")
        return seg2

    def close(self):
        self.r.close()


def geom_mask(seg2: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of pixels showing any geom in ``gids``."""
    return (seg2[..., 1] == GEOM_T) & np.isin(seg2[..., 0], gids)


def mask_iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum()) / union


def masked_psnr(a: np.ndarray, b: np.ndarray,
                mask: np.ndarray) -> Optional[float]:
    """PSNR (dB, uint8 range) over mask pixels; None if the mask is empty;
    capped at 99 (identical inputs)."""
    if int(mask.sum()) == 0:
        return None
    d = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    mse = float(np.mean(d * d))
    if mse <= 0.0:
        return 99.0
    return min(99.0, float(10.0 * np.log10(255.0 ** 2 / mse)))


def dist_to_mask_px(uv: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Containment + distance-to-mask (F1 pattern): 0 if the projected OpenCV
    pixel falls inside the mask, else distance to the nearest mask pixel
    center; inf for behind-camera / non-finite projections; None if the mask
    is empty (nothing to measure)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    if not np.all(np.isfinite(uv)):
        return float("inf")
    H, W = mask.shape
    r, c = int(np.floor(uv[1])), int(np.floor(uv[0]))
    if 0 <= r < H and 0 <= c < W and mask[r, c]:
        return 0.0
    return float(np.hypot(xs + 0.5 - uv[0], ys + 0.5 - uv[1]).min())


def _stats(vals: List[Optional[float]], qs=(5, 50, 95)) -> Optional[dict]:
    v = [float(x) for x in vals if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    out = {f"p{q}": float(np.percentile(a, q)) for q in qs}
    out.update(n=int(a.size), min=float(a.min()), max=float(a.max()),
               mean=float(a.mean()))
    return out


def read_render_source(root: "zarr.Group") -> str:
    src = root.attrs.get("render_source")
    if src is None and "meta/render_source" in root:
        v = root["meta/render_source"][0]
        src = v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return str(src) if src is not None else "oracle"


def task_delta(state_offset: np.ndarray, eps: List[int], task_name: str,
               gs_zarr: str) -> int:
    """The obs/state offset delta the GS pre-render recorded for this task
    (G4: copied from the oracle zarr, never recalibrated here)."""
    offs = sorted({int(state_offset[e]) for e in eps})
    if -1 in offs:
        raise RuntimeError(
            f"{task_name}: meta/state_offset is -1 (fill value) for episodes "
            f"of '{gs_zarr}' -- the GS pre-render never processed this task; "
            f"finish scripts/prerender_se2_aug.py --renderer gs first")
    if len(offs) != 1:
        raise RuntimeError(
            f"{task_name}: inconsistent meta/state_offset {offs} across "
            f"episodes in '{gs_zarr}' -- aug zarr is corrupt for this task")
    if offs[0] not in (0, 1):
        raise RuntimeError(
            f"{task_name}: meta/state_offset={offs[0]} not in (0, 1) in "
            f"'{gs_zarr}'")
    return offs[0]


def require_input(path: str, what: str, hint: str) -> None:
    """Graceful startup failure: missing inputs exit 2 with a clear message
    instead of an exception mid-run."""
    if not os.path.exists(path):
        print(f"[probe_gs_geometry] MISSING INPUT: {what} not found at "
              f"'{path}'.\n  {hint}")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gs_zarr",
                        default="data/libero/libero10_N500_se2aug_gs.zarr",
                        help="GS aug zarr (valid_mask / state_offset source)")
    parser.add_argument("--base_zarr", default="data/libero/libero10_N500.zarr")
    parser.add_argument("--hdf5_dir",
                        default="third_party/LIBERO/libero/datasets/libero_10")
    parser.add_argument("--assets_root", default="data/libero/gs_assets",
                        help="per-task GS asset root (<root>/<task>/manifest.json)")
    parser.add_argument("--facts", default="data/libero/gs_render_facts.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/libero/probe_gs_geometry.json")
    # gate thresholds (plan §8.1 defaults)
    parser.add_argument("--iou_obj_p5", type=float, default=0.90)
    parser.add_argument("--iou_robot_p5", type=float, default=0.85)
    parser.add_argument("--eef_median_px", type=float, default=2.0)
    parser.add_argument("--eef_p95_px", type=float, default=4.0)
    parser.add_argument("--wrist_psnr_min", type=float, default=32.0)
    parser.add_argument("--min_mask_px", type=int, default=20,
                        help="skip IoU/wrist measurements whose oracle mask is "
                             "smaller than this (nothing to measure)")
    args = parser.parse_args()

    gs_zarr_path = _abspath(args.gs_zarr)
    base_zarr_path = _abspath(args.base_zarr)
    hdf5_dir = _abspath(args.hdf5_dir)
    assets_root = _abspath(args.assets_root)
    facts_path = _abspath(args.facts)

    require_input(gs_zarr_path, "GS aug zarr",
                  "Render it first: MUJOCO_GL=egl python "
                  "scripts/prerender_se2_aug.py --renderer gs "
                  "--gs-assets-dir data/libero/gs_assets "
                  "--oracle-zarr data/libero/libero10_N500_se2aug.zarr "
                  f"--out {args.gs_zarr}")
    require_input(base_zarr_path, "base training zarr",
                  "The probe needs it for episode->demo matching and the "
                  "stored theta=0 frames.")
    require_input(hdf5_dir, "LIBERO demo hdf5 directory",
                  "Per-step MuJoCo states come from the *_demo.hdf5 files.")
    require_input(assets_root, "GS assets root",
                  "Train assets first (scripts/gsaug/train_gs_assets.py).")
    require_input(facts_path, "renderer facts",
                  "Run scripts/gsaug/probe_render_facts.py first (M1 gate).")

    # G7: measured conventions, asserted before anything renders
    facts = load_render_facts(facts_path)  # raises unless pass == true
    flip_mat = facts_flip(facts)
    flip_ud = facts_orientation_flip_ud(facts)

    def to_gs_orientation(m: np.ndarray) -> np.ndarray:
        """Align a raw-orientation array with GSCompositeRenderer output:
        the SAME facts-keyed F2 transform compose applies (see docstring)."""
        return m[::-1] if flip_ud else m

    # ── GS aug zarr: sampling source (valid_mask) + delta (state_offset) ────
    gs_root = zarr.open(gs_zarr_path, mode="r")
    render_source = read_render_source(gs_root)
    if not render_source.startswith("gs"):
        print(f"[probe_gs_geometry] WARN: '{gs_zarr_path}' carries "
              f"render_source={render_source!r} (not gs*) -- probing GS "
              f"geometry against a non-GS zarr's valid_mask")
    angles_deg = np.asarray(gs_root["meta/angles_deg"][:], dtype=np.float64)
    assert angles_deg[0] == 0.0, (
        f"{gs_zarr_path}: meta/angles_deg[0] = {angles_deg[0]} != 0")
    episode_ends = np.asarray(gs_root["meta/episode_ends"][:], dtype=np.int64)
    ep_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ep_lens = np.diff(np.concatenate([[0], episode_ends]))
    n_episodes = len(episode_ends)
    valid_mask = np.asarray(gs_root["meta/valid_mask"][:], dtype=bool)
    state_offset = np.asarray(gs_root["meta/state_offset"][:])

    base_root = zarr.open(base_zarr_path, mode="r")
    assert np.array_equal(
        np.asarray(base_root["meta/episode_ends"][:], dtype=np.int64),
        episode_ends), (
        f"{gs_zarr_path} meta/episode_ends != {base_zarr_path}'s -- the GS aug "
        f"zarr was built against a different base zarr")
    base_agent = base_root["data"][AGENT_KEY]
    image_size = int(base_agent.shape[1])  # G8: resolution from the zarr
    gs_agent0 = gs_root["images"][AGENT_KEY]["angle_00"]
    assert gs_agent0.shape[1] == image_size, (
        f"GS zarr image size {gs_agent0.shape[1]} != base zarr {image_size}")

    print(f"[probe_gs_geometry] matching {n_episodes} episodes to demos in "
          f"{hdf5_dir} ...")
    replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=["action"])
    matches = match_episodes(replay_buffer, hdf5_dir)
    assert len(matches) == n_episodes

    # ── sample triples from the VALID nonzero-angle entries ─────────────────
    rng = np.random.default_rng(args.seed)
    eligible = [e for e in range(n_episodes) if valid_mask[e, 1:].any()]
    assert eligible, f"{gs_zarr_path}: no episode has a valid nonzero angle"
    triples = []
    for _ in range(args.n):
        e = int(eligible[rng.integers(0, len(eligible))])
        ks = np.nonzero(valid_mask[e, 1:])[0] + 1
        k = int(ks[rng.integers(0, len(ks))])
        t = int(rng.integers(0, ep_lens[e] - 1))
        triples.append((e, t, k))

    by_task: Dict[str, List[int]] = {}
    for i, (e, _, _) in enumerate(triples):
        by_task.setdefault(task_name_from_hdf5(matches[e][0]), []).append(i)

    # GS-specific heavies stay out of module scope (mirrors prerender's lazy
    # import: --help and startup checks never touch gsplat/torch)
    import torch
    from oat.gsaug.compose import GSCompositeRenderer

    per_task: Dict[str, dict] = {}
    records: List[dict] = []
    obj_ious: List[float] = []
    robot_ious: List[float] = []
    eef_errs: List[float] = []
    eef_site_errs: List[float] = []
    gs_psnrs: List[float] = []
    oracle_psnrs: List[float] = []
    n_iou_skipped = 0
    n_eef_skipped = 0
    n_wrist_skipped = 0

    for task_name in sorted(by_task,
                            key=lambda t: task_name_to_suite_and_ids[t][2]):
        idxs = by_task[task_name]
        eps = sorted({triples[i][0] for i in idxs})
        print(f"[probe_gs_geometry] task {task_name}: {len(idxs)} triples, "
              f"{len(eps)} episodes")
        delta = task_delta(state_offset, eps, task_name, gs_zarr_path)

        hdf5_path = matches[eps[0]][0]
        with h5py.File(hdf5_path, "r") as f:
            states_cache = {
                e: np.asarray(f["data"][matches[e][1]]["states"][:],
                              dtype=np.float64) for e in eps}
        for e in eps:
            assert len(states_cache[e]) == ep_lens[e], (
                f"episode {e}: zarr length {ep_lens[e]} != demo states length "
                f"{len(states_cache[e])} ({hdf5_path} {matches[e][1]})")

        env = build_control_env(task_name, image_size)
        env.reset()  # final reset before any handle is taken (measured fact)
        sim_len = len(env.sim.get_state().flatten())
        for e in eps:
            assert states_cache[e].shape[1] == sim_len, (
                f"episode {e} ({task_name}): demo state length "
                f"{states_cache[e].shape[1]} != sim state length {sim_len}; an "
                f"XML fallback would break the manifest model_xml_sha1 (G9) -- "
                f"investigate instead")
        addr = resolve_addresses(env)
        obj_names = list(addr.obj_qpos_slices)

        # ctor loads + sha1-verifies assets (G9) and re-asserts the facts (G7);
        # the first render() asserts model_xml_sha1 vs the live env (G9)
        gs_renderer = GSCompositeRenderer(
            task_assets_dir=os.path.join(assets_root, task_name),
            cameras=OrderedDict(GS_CAMERAS),
            resolution=image_size,
            facts_path=facts_path,
            device="cuda:0",
        )
        manifest_sha1 = gs_renderer.manifest_sha1

        # ── phase A: ALL robosuite obs renders of the task, BEFORE the seg
        # renderer exists (measured EGL fact -- see OBS_SANITY_MAD_MAX) ──────
        rw_cache: Dict[int, tuple] = {}
        oracle_wrist: Dict[int, tuple] = {}
        obs_sanity_mad = None
        for j, i in enumerate(idxs):
            e, t, k = triples[i]
            states = states_cache[e]
            state = states[min(t + delta, len(states) - 1)]
            theta = float(np.deg2rad(angles_deg[k]))
            srw = rewrite_state(state, theta, addr)
            srw0 = rewrite_state(state, 0.0, addr)
            rw_cache[i] = (srw, srw0)
            obs_t = env.regenerate_obs_from_state(srw)
            w_t = np.flip(obs_t["robot0_eye_in_hand_image"], axis=0
                          ).astype(np.uint8).copy()
            obs_0 = env.regenerate_obs_from_state(srw0)
            w_0 = np.flip(obs_0["robot0_eye_in_hand_image"], axis=0
                          ).astype(np.uint8).copy()
            oracle_wrist[i] = (w_t, w_0)
            if j == 0:
                a_0 = np.flip(obs_0["agentview_image"], axis=0).astype(np.int16)
                stored = np.asarray(
                    base_agent[ep_starts[e] + t]).astype(np.int16)
                obs_sanity_mad = float(np.abs(a_0 - stored).mean())
                if obs_sanity_mad > OBS_SANITY_MAD_MAX:
                    raise RuntimeError(
                        f"{task_name}: oracle theta=0 re-render vs stored base "
                        f"frame MAD {obs_sanity_mad:.1f} > "
                        f"{OBS_SANITY_MAD_MAX} -- wrong state offset "
                        f"(delta={delta}) or the obs pipeline is rendering "
                        f"from a wrong viewpoint (extra EGL contexts alive?); "
                        f"oracle seg masks would be untrustworthy")

        # ── phase B: GS renders + oracle seg (offsamples=0 context) ─────────
        segr = SegRenderer(env, image_size, facts)
        model = env.sim.model
        mov_gids = {name: np.asarray(g, dtype=np.int64)
                    for name, g in movable_geom_ids(model, addr).items()}
        all_movable = (np.concatenate(list(mov_gids.values()))
                       if mov_gids else np.empty(0, dtype=np.int64))
        robot_gids = np.asarray(robot_geom_ids(model), dtype=np.int64)
        hand_bid = model.body_name2id("robot0_right_hand")
        hand_gids = np.asarray(
            geom_ids_of_bodies(model, body_subtree_ids(model, hand_bid)),
            dtype=np.int64)
        eef_sid = env.env.robots[0].eef_site_id  # grip site (plan §8.1 anchor)
        agent_cid = model.camera_name2id(GS_CAMERAS[AGENT_KEY])
        K_agent = fovy_to_K(float(model.cam_fovy[agent_cid]),
                            image_size, image_size)
        anchor = None
        t_obj, t_robot, t_eef, t_eef_site = [], [], [], []
        t_gs_psnr, t_or_psnr = [], []

        try:
            for j, i in enumerate(idxs):
                e, t, k = triples[i]
                srw, srw0 = rw_cache[i]
                gs0 = gs_renderer.render(env, srw0)
                gs_t = gs_renderer.render(env, srw)
                # G2 note: solo alpha is a probe-only path; composite frames
                # above come from the single concatenated pass
                alphas = {name: gs_renderer.render_component_alpha(
                    env, srw, name, AGENT_KEY) for name in obj_names}
                alphas["robot"] = gs_renderer.render_component_alpha(
                    env, srw, "robot", AGENT_KEY)

                if j == 0:
                    # orientation anchor (record-only; see module docstring)
                    stored = np.asarray(
                        base_agent[ep_starts[e] + t]).astype(np.int16)
                    g = gs0[AGENT_KEY].astype(np.int16)
                    mad_d = float(np.abs(g - stored).mean())
                    mad_f = float(np.abs(g[::-1] - stored).mean())
                    anchor = {"mad_direct": mad_d, "mad_flipped": mad_f,
                              "gs_matches_stored_orientation": mad_d <= mad_f}
                    if mad_f < mad_d:
                        print(f"[probe_gs_geometry] WARN {task_name}: GS "
                              f"theta=0 render matches the stored base frame "
                              f"better FLIPPED (MAD {mad_f:.1f} < {mad_d:.1f})"
                              f" -- compose/F2 orientation chain suspect; the "
                              f"pre-render theta=0 gate is the hard fail for "
                              f"this")

                # oracle seg of the rewritten state (raw orientation == dataset
                # orientation, F2b): the last render_component_alpha left the
                # sim at srw, but set it explicitly -- no hidden coupling
                env.sim.set_state_from_flattened(srw)
                env.sim.forward()
                seg_a = segr.seg(env, GS_CAMERAS[AGENT_KEY])
                seg_w = segr.seg(env, GS_CAMERAS[WRIST_KEY])

                # 1. silhouette IoU (gated) + occlusion-corrected (diagnostic)
                all_mov_mask = geom_mask(seg_a, all_movable)
                robot_mask_a = geom_mask(seg_a, robot_gids)
                iou_obj_rec, iou_obj_occl_rec = {}, {}
                for name in obj_names:
                    vis = geom_mask(seg_a, mov_gids[name])
                    if int(vis.sum()) < args.min_mask_px:
                        iou_obj_rec[name] = None
                        iou_obj_occl_rec[name] = None
                        n_iou_skipped += 1
                        continue
                    vis_gs = to_gs_orientation(vis)
                    am = alphas[name] >= ALPHA_THRESH
                    iou = mask_iou(am, vis_gs)
                    occ = to_gs_orientation(
                        robot_mask_a | (all_mov_mask & ~vis))
                    iou_occl = mask_iou(am & ~occ, vis_gs)
                    iou_obj_rec[name] = iou
                    iou_obj_occl_rec[name] = iou_occl
                    obj_ious.append(iou)
                    t_obj.append(iou)
                rob_vis = robot_mask_a
                iou_robot = iou_robot_occl = None
                if int(rob_vis.sum()) < args.min_mask_px:
                    n_iou_skipped += 1
                else:
                    vis_gs = to_gs_orientation(rob_vis)
                    am = alphas["robot"] >= ALPHA_THRESH
                    iou_robot = mask_iou(am, vis_gs)
                    iou_robot_occl = mask_iou(
                        am & ~to_gs_orientation(all_mov_mask), vis_gs)
                    robot_ious.append(iou_robot)
                    t_robot.append(iou_robot)

                # 2. EEF projection (raw orientation: F1 pins raw row 0 ==
                # OpenCV top, so (u, v) indexes the raw seg directly)
                d = env.sim.data
                w2c = mujoco_cam_to_w2c(d.cam_xpos[agent_cid],
                                        d.cam_xmat[agent_cid], flip_mat)
                hand_mask = geom_mask(seg_a, hand_gids)
                uv = project(K_agent, w2c,
                             np.asarray(d.xpos[hand_bid], dtype=np.float64))
                eef_px = dist_to_mask_px(uv, hand_mask)
                uv_site = project(K_agent, w2c, np.asarray(
                    d.site_xpos[eef_sid], dtype=np.float64))
                eef_site_px = dist_to_mask_px(uv_site, hand_mask)
                if eef_px is None:
                    n_eef_skipped += 1
                else:
                    eef_errs.append(eef_px)
                    t_eef.append(eef_px)
                if eef_site_px is not None:
                    eef_site_errs.append(eef_site_px)
                    t_eef_site.append(eef_site_px)

                # 3. wrist transform-stack check (movables ∪ robot mask from
                # the theta render's seg; co-rotating content is wrist-static)
                mask_w_raw = (geom_mask(seg_w, all_movable)
                              | geom_mask(seg_w, robot_gids))
                wrist_mask_px = int(mask_w_raw.sum())
                psnr_gs = psnr_or = None
                if wrist_mask_px < args.min_mask_px:
                    n_wrist_skipped += 1
                else:
                    psnr_gs = masked_psnr(gs_t[WRIST_KEY], gs0[WRIST_KEY],
                                          to_gs_orientation(mask_w_raw))
                    w_t, w_0 = oracle_wrist[i]
                    # oracle wrist frames are np.flip(obs) == raw orientation:
                    # the mask applies UNFLIPPED here (report-only quantity)
                    psnr_or = masked_psnr(w_t, w_0, mask_w_raw)
                    gs_psnrs.append(psnr_gs)
                    oracle_psnrs.append(psnr_or)
                    t_gs_psnr.append(psnr_gs)
                    t_or_psnr.append(psnr_or)

                records.append({
                    "episode": e, "task": task_name, "frame": t,
                    "angle_deg": float(angles_deg[k]),
                    "iou_objects": iou_obj_rec,
                    "iou_objects_occl": iou_obj_occl_rec,
                    "iou_robot": iou_robot,
                    "iou_robot_occl": iou_robot_occl,
                    "eef_px": eef_px,
                    "eef_site_px": eef_site_px,
                    "wrist_psnr_gs": psnr_gs,
                    "wrist_psnr_oracle": psnr_or,
                    "wrist_mask_px": wrist_mask_px,
                })
        finally:
            segr.close()
            env.close()
            del gs_renderer
            torch.cuda.empty_cache()

        per_task[task_name] = {
            "hdf5_path": hdf5_path,
            "n_triples": len(idxs),
            "n_objects": len(obj_names),
            "delta": delta,
            "gs_manifest_sha1": manifest_sha1,
            "obs_sanity_mad": obs_sanity_mad,
            "orientation_anchor": anchor,
            "iou_objects": _stats(t_obj),
            "iou_robot": _stats(t_robot),
            "eef_px": _stats(t_eef),
            "eef_site_px": _stats(t_eef_site),
            "wrist_psnr_gs": _stats(t_gs_psnr),
            "wrist_psnr_oracle": _stats(t_or_psnr),
        }

    # ── gates ───────────────────────────────────────────────────────────────
    iou_gate = {
        "objects_p5": (float(np.percentile(obj_ious, 5)) if obj_ious else None),
        "objects_thresh": args.iou_obj_p5,
        "robot_p5": (float(np.percentile(robot_ious, 5)) if robot_ious else None),
        "robot_thresh": args.iou_robot_p5,
        "n_object_measurements": len(obj_ious),
        "n_robot_measurements": len(robot_ious),
        "n_skipped_small_mask": n_iou_skipped,
    }
    iou_gate["pass"] = bool(
        obj_ious and robot_ious
        and iou_gate["objects_p5"] >= args.iou_obj_p5
        and iou_gate["robot_p5"] >= args.iou_robot_p5)

    eef_gate = {
        "median_px": (float(np.median(eef_errs)) if eef_errs else None),
        "p95_px": (float(np.percentile(eef_errs, 95)) if eef_errs else None),
        "median_thresh": args.eef_median_px,
        "p95_thresh": args.eef_p95_px,
        "n": len(eef_errs),
        "n_skipped": n_eef_skipped,
        # plan §8.1 grip-site anchor, recorded but NOT gated (see docstring)
        "site_median_px": (float(np.median(eef_site_errs))
                           if eef_site_errs else None),
        "site_p95_px": (float(np.percentile(eef_site_errs, 95))
                        if eef_site_errs else None),
    }
    eef_gate["pass"] = bool(
        eef_errs and eef_gate["median_px"] <= args.eef_median_px
        and eef_gate["p95_px"] <= args.eef_p95_px)

    wrist_gate = {
        "gs_psnr_min": (float(np.min(gs_psnrs)) if gs_psnrs else None),
        "gs_psnr_p5": (float(np.percentile(gs_psnrs, 5)) if gs_psnrs else None),
        "gs_psnr_median": (float(np.median(gs_psnrs)) if gs_psnrs else None),
        "min_thresh": args.wrist_psnr_min,
        "oracle_psnr_p5": (float(np.percentile(oracle_psnrs, 5))
                           if oracle_psnrs else None),
        "oracle_psnr_median": (float(np.median(oracle_psnrs))
                               if oracle_psnrs else None),
        "n": len(gs_psnrs),
        "n_skipped_empty_mask": n_wrist_skipped,
    }
    wrist_gate["pass"] = bool(
        gs_psnrs and wrist_gate["gs_psnr_min"] >= args.wrist_psnr_min)

    ok = bool(iou_gate["pass"] and eef_gate["pass"] and wrist_gate["pass"])

    failures = []
    for r in records:
        reasons = []
        for name, v in r["iou_objects"].items():
            if v is not None and v < args.iou_obj_p5:
                reasons.append(f"iou_object[{name}]={v:.3f}")
        if r["iou_robot"] is not None and r["iou_robot"] < args.iou_robot_p5:
            reasons.append(f"iou_robot={r['iou_robot']:.3f}")
        if r["eef_px"] is not None and r["eef_px"] > args.eef_p95_px:
            reasons.append(f"eef_px={r['eef_px']:.2f}")
        if (r["wrist_psnr_gs"] is not None
                and r["wrist_psnr_gs"] < args.wrist_psnr_min):
            reasons.append(f"wrist_psnr_gs={r['wrist_psnr_gs']:.2f}")
        if reasons:
            failures.append({"episode": r["episode"], "task": r["task"],
                             "frame": r["frame"],
                             "angle_deg": r["angle_deg"], "reasons": reasons})

    result = {
        "probe": "gs_geometry",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(args).items()},
        "gs_zarr": gs_zarr_path,
        "base_zarr": base_zarr_path,
        "render_source": render_source,
        "image_size": image_size,
        "n_triples": len(records),
        "gates": {
            "silhouette_iou": iou_gate,
            "eef_projection": eef_gate,
            "wrist_transform_stack": wrist_gate,
        },
        "per_task": per_task,
        "failures": failures,
        "triples": records,
        "pass": ok,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"[probe_gs_geometry] wrote {out_path}")

    verdict = "PASS" if ok else "FAIL"
    print(f"[probe_gs_geometry] {verdict}: "
          f"iou(obj p5={iou_gate['objects_p5']}, "
          f"robot p5={iou_gate['robot_p5']}) "
          f"eef(med={eef_gate['median_px']}, p95={eef_gate['p95_px']}) "
          f"wrist(min={wrist_gate['gs_psnr_min']}, "
          f"oracle med={wrist_gate['oracle_psnr_median']})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

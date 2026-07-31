"""GS photometric probe (plan §8.2) -- REPORT-ONLY (always exits 0 on a
completed run; only missing inputs exit nonzero).

For every task: up to ``--frames_per_task`` sampled frames are re-rendered at
theta=0 through ``GSCompositeRenderer`` (the same pixel source the GS
pre-render uses) and compared against the STORED base zarr frames the oracle
arms train on. Reported per task and pooled:

* **partitioned PSNR / SSIM** over the oracle seg masks {robot, movables,
  background} (mask-exact means; SSIM is a channel-averaged gaussian-window
  map -- Wang et al. defaults, sigma 1.5 / truncate 3.5, skimage-compatible --
  averaged over each mask, so partition edges share window support);
* **full-frame LPIPS** (lpips package, net='alex', batched on cuda);
* **movable-bbox-crop LPIPS** (movable-silhouette bbox padded 4 px, expanded
  to >= 32 px sides);
* **contact-band PSNR**: pixels within ``CONTACT_BAND_PX`` rows BELOW the
  movable silhouette's lower edge (mask dilated downward minus the mask) --
  the shadow-gap tracker (risk R1: baked GS casts no contact shadows);
* extras (clearly non-gating): wrist-camera full-frame PSNR/LPIPS, the
  orientation anchor (GS render vs stored frame, direct vs flipped MAD), and
  -- when the GS zarr's render_source is 'gs' -- the MAD between the live GS
  render and the zarr's stored angle_00 frame (T6 repeatability/provenance).

delta is copied from the GS zarr's ``meta/state_offset`` (G4). Oracle seg
masks come from a raw ``mujoco.Renderer`` with the measured F2b flags on an
offsamples=0 context; raw renderer orientation == stored-zarr orientation
(F2b measured MAD 0.0 against the exact expression dataset conversion
stores), so masks index the stored frames with no flip. GS frames are
compared AS DELIVERED by compose (what training would see).

``scripts/gsaug/report_factorial.py`` later correlates these per-task numbers
against the per-task A3-A4 success-rate deltas.

Usage:
    export PATH=/home/haotian/miniforge3/envs/oat/bin:/usr/local/cuda/bin:$PATH
    MUJOCO_GL=egl python scripts/probes/probe_gs_photometric.py \
        --gs_zarr data/libero/libero10_N500_se2aug_gs.zarr
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
from typing import Dict, List, Optional, Tuple

import h5py
import mujoco
import numpy as np
import zarr
from scipy.ndimage import gaussian_filter

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from oat.common.replay_buffer import ReplayBuffer
from oat.env.libero.demo_alignment import match_episodes
from oat.env.libero.env import task_name_to_suite_and_ids
from oat.env.libero.se2_state_rewrite import resolve_addresses, rewrite_state
from oat.gsaug.capture import movable_geom_ids, robot_geom_ids
from oat.gsaug.cameras import load_render_facts

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GEOM_T = int(mujoco.mjtObj.mjOBJ_GEOM)

# zarr image key -> mujoco camera name (must match scripts/prerender_se2_aug.py)
GS_CAMERAS = OrderedDict(
    [
        ("agentview_rgb", "agentview"),
        ("robot0_eye_in_hand_rgb", "robot0_eye_in_hand"),
    ]
)
AGENT_KEY = "agentview_rgb"
WRIST_KEY = "robot0_eye_in_hand_rgb"

CONTACT_BAND_PX = 6     # rows below the movable silhouette lower edge
BBOX_PAD_PX = 4         # movable-crop LPIPS bbox padding
BBOX_MIN_SIDE = 32      # alexnet features need a sane minimum crop
LPIPS_BATCH = 32


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
    """Exact-seg ``mujoco.Renderer`` under the measured F2b vis flags on an
    offsamples=0 context (MSAA blends seg ID colors at silhouette edges into
    unrelated geom ids -- the ProbeRenderers.geo pattern from
    scripts/gsaug/probe_render_facts.py). F2b ``flags_off`` entries are
    ``mjRND_*`` scene flags (measured empty), applied after ``update_scene``."""

    def __init__(self, env, image_size: int, facts: dict):
        raw_model = env.sim.model._model
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
        sim state, RAW renderer orientation (== stored-zarr orientation)."""
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
    return (seg2[..., 1] == GEOM_T) & np.isin(seg2[..., 0], gids)


def masked_psnr(a: np.ndarray, b: np.ndarray,
                mask: np.ndarray) -> Optional[float]:
    """PSNR (dB, uint8 range) over mask pixels; None on an empty mask; capped
    at 99 (identical inputs)."""
    if int(mask.sum()) == 0:
        return None
    d = a[mask].astype(np.float64) - b[mask].astype(np.float64)
    mse = float(np.mean(d * d))
    if mse <= 0.0:
        return 99.0
    return min(99.0, float(10.0 * np.log10(255.0 ** 2 / mse)))


def ssim_map(a: np.ndarray, b: np.ndarray, data_range: float = 255.0,
             sigma: float = 1.5) -> np.ndarray:
    """Per-pixel SSIM map, channel-averaged (Wang et al. defaults with a
    gaussian window; matches skimage's gaussian_weights=True settings)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def f(im):
        return gaussian_filter(im, sigma, truncate=3.5)

    maps = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mx, my = f(x), f(y)
        vx = f(x * x) - mx * mx
        vy = f(y * y) - my * my
        cxy = f(x * y) - mx * my
        maps.append(((2 * mx * my + C1) * (2 * cxy + C2))
                    / ((mx * mx + my * my + C1) * (vx + vy + C2)))
    return np.mean(maps, axis=0)


def masked_mean(m: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if int(mask.sum()) == 0:
        return None
    return float(m[mask].mean())


def contact_band(mov_mask: np.ndarray, px: int = CONTACT_BAND_PX) -> np.ndarray:
    """Pixels within ``px`` rows BELOW the movable silhouette lower edge:
    the mask dilated downward ``px`` rows, minus the mask (stored-zarr
    orientation: row 0 is the image top, larger row = visually lower)."""
    band = np.zeros_like(mov_mask)
    shifted = mov_mask.copy()
    for _ in range(px):
        shifted = np.roll(shifted, 1, axis=0)
        shifted[0] = False
        band |= shifted
    return band & ~mov_mask


def movable_bbox(mov_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(r0, r1, c0, c1) crop window: mask bbox + BBOX_PAD_PX, expanded to at
    least BBOX_MIN_SIDE per side, clipped to the image. None on empty mask."""
    ys, xs = np.nonzero(mov_mask)
    if ys.size == 0:
        return None
    H, W = mov_mask.shape
    r0, r1 = int(ys.min()) - BBOX_PAD_PX, int(ys.max()) + 1 + BBOX_PAD_PX
    c0, c1 = int(xs.min()) - BBOX_PAD_PX, int(xs.max()) + 1 + BBOX_PAD_PX
    while (r1 - r0) < min(BBOX_MIN_SIDE, H):
        r0, r1 = r0 - 1, r1 + 1
    while (c1 - c0) < min(BBOX_MIN_SIDE, W):
        c0, c1 = c0 - 1, c1 + 1
    return max(0, r0), min(H, r1), max(0, c0), min(W, c1)


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
    """delta the GS pre-render recorded (G4: copied from the oracle zarr)."""
    offs = sorted({int(state_offset[e]) for e in eps})
    if -1 in offs:
        raise RuntimeError(
            f"{task_name}: meta/state_offset is -1 (fill value) in "
            f"'{gs_zarr}' -- the GS pre-render never processed this task; "
            f"finish scripts/prerender_se2_aug.py --renderer gs first")
    if len(offs) != 1 or offs[0] not in (0, 1):
        raise RuntimeError(
            f"{task_name}: bad meta/state_offset {offs} in '{gs_zarr}'")
    return offs[0]


def require_input(path: str, what: str, hint: str) -> None:
    """Graceful startup failure: missing inputs exit 2 with a clear message."""
    if not os.path.exists(path):
        print(f"[probe_gs_photometric] MISSING INPUT: {what} not found at "
              f"'{path}'.\n  {hint}")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gs_zarr",
                        default="data/libero/libero10_N500_se2aug_gs.zarr",
                        help="GS aug zarr (state_offset / provenance source)")
    parser.add_argument("--base_zarr", default="data/libero/libero10_N500.zarr")
    parser.add_argument("--hdf5_dir",
                        default="third_party/LIBERO/libero/datasets/libero_10")
    parser.add_argument("--assets_root", default="data/libero/gs_assets")
    parser.add_argument("--facts", default="data/libero/gs_render_facts.json")
    parser.add_argument("--frames_per_task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/libero/probe_gs_photometric.json")
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
                  "The probe compares GS renders against its stored frames.")
    require_input(hdf5_dir, "LIBERO demo hdf5 directory",
                  "Per-step MuJoCo states come from the *_demo.hdf5 files.")
    require_input(assets_root, "GS assets root",
                  "Train assets first (scripts/gsaug/train_gs_assets.py).")
    require_input(facts_path, "renderer facts",
                  "Run scripts/gsaug/probe_render_facts.py first (M1 gate).")

    facts = load_render_facts(facts_path)  # G7: raises unless pass == true

    gs_root = zarr.open(gs_zarr_path, mode="r")
    render_source = read_render_source(gs_root)
    episode_ends = np.asarray(gs_root["meta/episode_ends"][:], dtype=np.int64)
    ep_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    n_episodes = len(episode_ends)
    state_offset = np.asarray(gs_root["meta/state_offset"][:])

    base_root = zarr.open(base_zarr_path, mode="r")
    assert np.array_equal(
        np.asarray(base_root["meta/episode_ends"][:], dtype=np.int64),
        episode_ends), (
        f"{gs_zarr_path} meta/episode_ends != {base_zarr_path}'s -- the GS aug "
        f"zarr was built against a different base zarr")
    base_images = {cam: base_root["data"][cam] for cam in GS_CAMERAS}
    image_size = int(base_images[AGENT_KEY].shape[1])  # G8
    # stored GS theta=0 frames (repeatability diagnostic; only meaningful when
    # angle_00 actually holds GS renders, i.e. render_source == 'gs')
    gs_stored_agent = gs_root["images"][AGENT_KEY]["angle_00"]
    check_repeat = render_source == "gs"

    print(f"[probe_gs_photometric] matching {n_episodes} episodes to demos in "
          f"{hdf5_dir} ...")
    replay_buffer = ReplayBuffer.copy_from_path(base_zarr_path, keys=["action"])
    matches = match_episodes(replay_buffer, hdf5_dir)
    assert len(matches) == n_episodes

    ep_task = [task_name_from_hdf5(path) for path, _ in matches]
    task_to_eps: Dict[str, List[int]] = OrderedDict()
    for t in sorted(set(ep_task), key=lambda n: task_name_to_suite_and_ids[n][2]):
        task_to_eps[t] = [e for e in range(n_episodes) if ep_task[e] == t]

    rng = np.random.default_rng(args.seed)

    # GS/torch heavies out of module scope (--help and startup checks stay
    # light); lpips downloads/loads the alexnet trunk once for the whole run
    import torch
    import lpips as lpips_pkg
    from oat.gsaug.compose import GSCompositeRenderer

    device = torch.device("cuda:0")
    lpips_model = lpips_pkg.LPIPS(net="alex").to(device).eval()

    def lpips_pairs(pairs: List[Tuple[np.ndarray, np.ndarray]],
                    batch: int = LPIPS_BATCH) -> List[float]:
        """LPIPS for uint8 HWC image pairs (same shape within a call)."""
        vals: List[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch):
                chunk = pairs[i:i + batch]
                A = torch.stack([
                    torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
                    for a, _ in chunk]).float().to(device) / 127.5 - 1.0
                B = torch.stack([
                    torch.from_numpy(np.ascontiguousarray(b)).permute(2, 0, 1)
                    for _, b in chunk]).float().to(device) / 127.5 - 1.0
                vals.extend(float(x) for x in
                            lpips_model(A, B).flatten().cpu().numpy())
        return vals

    per_task: Dict[str, dict] = {}
    frame_records: List[dict] = []
    pooled: Dict[str, List[Optional[float]]] = {}

    def pool(key: str, val: Optional[float]) -> Optional[float]:
        pooled.setdefault(key, []).append(val)
        return val

    for task_name, eps in task_to_eps.items():
        delta = task_delta(state_offset, eps, task_name, gs_zarr_path)
        hdf5_path = matches[eps[0]][0]
        with h5py.File(hdf5_path, "r") as f:
            states_cache = {
                e: np.asarray(f["data"][matches[e][1]]["states"][:],
                              dtype=np.float64) for e in eps}

        # sample <= frames_per_task global frame indices across the task
        frame_pool = np.concatenate([
            np.arange(ep_starts[e], episode_ends[e], dtype=np.int64)
            for e in eps])
        take = min(args.frames_per_task, len(frame_pool))
        gidxs = np.sort(rng.choice(frame_pool, size=take, replace=False))
        gidx_to_ep = np.searchsorted(episode_ends, gidxs, side="right")
        print(f"[probe_gs_photometric] task {task_name}: {take} frames, "
              f"delta={delta}")

        env = build_control_env(task_name, image_size)
        env.reset()  # final reset before any handle is taken (measured fact)
        sim_len = len(env.sim.get_state().flatten())
        for e in eps:
            assert states_cache[e].shape[1] == sim_len, (
                f"episode {e} ({task_name}): demo state length "
                f"{states_cache[e].shape[1]} != sim state length {sim_len}")
        addr = resolve_addresses(env)

        gs_renderer = GSCompositeRenderer(
            task_assets_dir=os.path.join(assets_root, task_name),
            cameras=OrderedDict(GS_CAMERAS),
            resolution=image_size,
            facts_path=facts_path,
            device="cuda:0",
        )
        # no robosuite obs render is ever needed here (stored frames are the
        # reference), so the seg renderer can exist from the start
        segr = SegRenderer(env, image_size, facts)
        model = env.sim.model
        mov_gids = movable_geom_ids(model, addr)
        all_movable = (np.concatenate([np.asarray(g) for g in mov_gids.values()])
                       if mov_gids else np.empty(0, dtype=np.int64))
        robot_gids = np.asarray(robot_geom_ids(model), dtype=np.int64)

        t_metrics: Dict[str, List[Optional[float]]] = {}

        def rec(key: str, val: Optional[float]) -> Optional[float]:
            t_metrics.setdefault(key, []).append(val)
            pool(key, val)
            return val

        full_pairs, wrist_pairs, crop_pairs = [], [], []
        crop_frame_ids: List[int] = []
        try:
            for fi, gidx in enumerate(gidxs.tolist()):
                e = int(gidx_to_ep[fi])
                t = int(gidx - ep_starts[e])
                states = states_cache[e]
                state = states[min(t + delta, len(states) - 1)]
                srw0 = rewrite_state(state, 0.0, addr)

                gs = gs_renderer.render(env, srw0)
                gs_agent = gs[AGENT_KEY]
                gs_wrist = gs[WRIST_KEY]
                stored_agent = np.asarray(base_images[AGENT_KEY][gidx])
                stored_wrist = np.asarray(base_images[WRIST_KEY][gidx])

                # oracle seg masks at the same state (raw orientation ==
                # stored-zarr orientation, F2b -- no flip anywhere here)
                env.sim.set_state_from_flattened(srw0)
                env.sim.forward()
                seg_a = segr.seg(env, GS_CAMERAS[AGENT_KEY])
                m_robot = geom_mask(seg_a, robot_gids)
                m_mov = geom_mask(seg_a, all_movable)
                m_bg = ~(m_robot | m_mov)

                r = {"task": task_name, "episode": e, "frame": t,
                     "global_idx": int(gidx)}
                r["psnr_full"] = rec("psnr_full", masked_psnr(
                    gs_agent, stored_agent, np.ones_like(m_bg)))
                r["psnr_robot"] = rec("psnr_robot",
                                      masked_psnr(gs_agent, stored_agent, m_robot))
                r["psnr_movables"] = rec("psnr_movables",
                                         masked_psnr(gs_agent, stored_agent, m_mov))
                r["psnr_background"] = rec("psnr_background",
                                           masked_psnr(gs_agent, stored_agent, m_bg))

                smap = ssim_map(gs_agent, stored_agent)
                r["ssim_full"] = rec("ssim_full", float(smap.mean()))
                r["ssim_robot"] = rec("ssim_robot", masked_mean(smap, m_robot))
                r["ssim_movables"] = rec("ssim_movables", masked_mean(smap, m_mov))
                r["ssim_background"] = rec("ssim_background",
                                           masked_mean(smap, m_bg))

                band = contact_band(m_mov)
                r["contact_band_psnr"] = rec(
                    "contact_band_psnr",
                    masked_psnr(gs_agent, stored_agent, band))
                r["contact_band_px"] = int(band.sum())

                r["wrist_psnr_full"] = rec("wrist_psnr_full", masked_psnr(
                    gs_wrist, stored_wrist, np.ones_like(m_bg)))

                # orientation anchor + repeatability (see module docstring)
                mad_d = float(np.mean(np.abs(
                    gs_agent.astype(np.int16) - stored_agent.astype(np.int16))))
                mad_f = float(np.mean(np.abs(
                    gs_agent[::-1].astype(np.int16)
                    - stored_agent.astype(np.int16))))
                r["anchor_mad_direct"] = rec("anchor_mad_direct", mad_d)
                r["anchor_mad_flipped"] = rec("anchor_mad_flipped", mad_f)
                if check_repeat:
                    r["gs_repeat_mad"] = rec("gs_repeat_mad", float(np.mean(
                        np.abs(gs_agent.astype(np.int16)
                               - np.asarray(gs_stored_agent[gidx]
                                            ).astype(np.int16)))))

                full_pairs.append((gs_agent, stored_agent))
                wrist_pairs.append((gs_wrist, stored_wrist))
                bbox = movable_bbox(m_mov)
                if bbox is not None:
                    r0_, r1_, c0_, c1_ = bbox
                    crop_pairs.append((gs_agent[r0_:r1_, c0_:c1_],
                                       stored_agent[r0_:r1_, c0_:c1_]))
                    crop_frame_ids.append(len(frame_records))
                frame_records.append(r)

            # batched LPIPS (full frames share a shape; crops do not)
            for v, r_idx in zip(lpips_pairs(full_pairs),
                                range(len(frame_records) - len(full_pairs),
                                      len(frame_records))):
                frame_records[r_idx]["lpips_full"] = rec("lpips_full", v)
            for v, r_idx in zip(lpips_pairs(wrist_pairs),
                                range(len(frame_records) - len(wrist_pairs),
                                      len(frame_records))):
                frame_records[r_idx]["wrist_lpips_full"] = rec(
                    "wrist_lpips_full", v)
            for (a, b), r_idx in zip(crop_pairs, crop_frame_ids):
                v = lpips_pairs([(a, b)], batch=1)[0]
                frame_records[r_idx]["lpips_movable_crop"] = rec(
                    "lpips_movable_crop", v)
        finally:
            segr.close()
            env.close()
            del gs_renderer
            torch.cuda.empty_cache()

        per_task[task_name] = {
            "hdf5_path": hdf5_path,
            "n_frames": int(take),
            "delta": delta,
            "metrics": {k: _stats(v) for k, v in sorted(t_metrics.items())},
        }

    result = {
        "probe": "gs_photometric",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(args).items()},
        "gs_zarr": gs_zarr_path,
        "base_zarr": base_zarr_path,
        "render_source": render_source,
        "image_size": image_size,
        "n_frames": len(frame_records),
        "contact_band_px_below": CONTACT_BAND_PX,
        "per_task": per_task,
        "pooled": {k: _stats(v) for k, v in sorted(pooled.items())},
        "frames": frame_records,
    }
    out_path = _abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=_json_default)
    print(f"[probe_gs_photometric] wrote {out_path}")

    pooled_stats = result["pooled"]

    def _fmt(key):
        s = pooled_stats.get(key)
        return f"{s['mean']:.3f}" if s else "n/a"

    print(f"[probe_gs_photometric] REPORT (never gates): "
          f"psnr full/robot/movables/bg = {_fmt('psnr_full')}/"
          f"{_fmt('psnr_robot')}/{_fmt('psnr_movables')}/"
          f"{_fmt('psnr_background')}  ssim_full={_fmt('ssim_full')}  "
          f"lpips_full={_fmt('lpips_full')}  "
          f"lpips_movable_crop={_fmt('lpips_movable_crop')}  "
          f"contact_band_psnr={_fmt('contact_band_psnr')}")
    sys.exit(0)


if __name__ == "__main__":
    main()

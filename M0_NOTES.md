# M0 Notes — GS Render Phase 0 discovery (2026-07-31)

Plan: `IMPLEMENTATION_PLAN_gs_render_phase0.md`. Every item below was measured on this
machine, in the `oat` conda env (`/home/haotian/miniforge3/envs/oat/bin/python`).

## File map / repo confirmation

- All functions this plan names exist as documented in `docs/se2_data_augmentation.md` §9:
  `rewrite_state`, `resolve_addresses`, `regenerate_obs_from_state`, `table_top_xy_aabb`,
  the four validity checks (`check_joint1_limit`, `check_objects_in_bounds`,
  `check_support_contacts`, `check_object_penetration`), the two-pass driver in
  `scripts/prerender_se2_aug.py` (`render_episode_angle`, `open_out_zarr`, `write_report`,
  resume/shard logic).
- **Plan §15 path fix**: arm configs live under `oat/config/` (root) and
  `oat/config/task/policy/libero/` — not `conf/...`. New configs go there.

## Oracle aug zarr (parameterizes everything downstream)

- `data/libero/libero10_N500_se2aug.zarr` + `.report.json` exist.
- `angles_deg = [0, 10, -10, 20, -20, 30, -30]` (K=7).
- per-angle valid rate `[1.0, .532, .558, .518, .588, .504, .634]`; overall 61.9%.
- 500 episodes, 10 tasks, 138090 frames; images 128², cameras
  `{agentview_rgb, robot0_eye_in_hand_rgb}`; `meta/state_offset` is uniformly **1**.
- Prereqs present: `probe_results.json` (PASS), both frozen norm specs.

## GPU / packages

- 2× RTX 4090 (24 GB, sm_89), driver 580.173.02; system CUDA toolkit **12.8** at
  `/usr/local/cuda` (matches torch cu128).
- torch 2.10.0+cu128 (cuda ok, 2 devices), mujoco 3.2.6 (`mujoco.Renderer` present),
  robosuite 1.4.0, zarr 2.18.3, numpy 2.2.6.
- Installed: **gsplat 1.5.3**, **lpips 0.1.4** (no `__version__` attr), ninja 1.13.0.
  e3nn NOT installed (optional; the G5 wigner scaffold must import it lazily).
- gsplat JIT compile: needs `PATH` to include the env `bin/` (ninja) and
  `/usr/local/cuda/bin` (nvcc) on first import; compiled in 76 s, cached afterwards
  (`~/.cache/torch_extensions`). gsplat's fast path through torch's private
  `_jit_compile` API is signature-incompatible with torch 2.10; its fallback path works.

## gsplat smoke render (F7 preview)

- 3-Gaussian scene: **7384 FPS @128²**, 7342 FPS @512².
- 300k Gaussians, sh_degree=3: **1328 FPS @128²** → the §12 budget (≥150 FPS composited)
  has ~9× headroom before kinematics overhead.
- API facts recorded for the conventions block: `gsplat.rasterization(means[N,3],
  quats[N,4] **wxyz**, scales[N,3] (linear, not log), opacities[N] (0..1, not logits),
  colors (RGB or SH [N,K,3]), viewmats[C,4,4] **world-to-camera, OpenCV (+z forward)**,
  Ks[C,3,3], width, height, sh_degree=...)` → returns `(img [C,H,W,3] float 0..1, alpha,
  meta)`.

## Raw-MuJoCo render recipe from LIBERO `ControlEnv` (used by every capture/probe stage)

- `env.sim` is `robosuite.utils.binding_utils.MjSim`; raw structs at `sim.model._model`
  (`mujoco.MjModel`) and `sim.data._data` (`mujoco.MjData`).
- `mujoco.Renderer(sim.model._model, height, width)` works with **named cameras**
  (`update_scene(sim.data._data, camera="agentview")`) and **programmable free cameras**
  (`mujoco.MjvCamera`, type `mjCAMERA_FREE`).
- **Depth is already metric** (float32 meters; verified 1.24–6.15 m on a reset scene):
  the plan's F5 znear/zfar/extent conversion is unnecessary — F5 reduces to the
  plane-fit validation. (`vis.map.znear=0.001·extent`, `zfar=50·extent`,
  `stat.extent≈11.83` recorded for completeness.)
- Segmentation render returns `(H, W, 2)` int32 (geom id / type channels).
- `cam_fovy`: agentview **45°**, robot0_eye_in_hand **75°** (vertical FOV; square pixels).
- Headlight (stock): ambient 0.1, diffuse 0.4, specular 0.5 (per-channel equal).
- Orientation: raw renderer output ≈ `np.flip(obs, axis=0)` of robosuite obs
  (MAD 15.4 flipped vs 98.1 unflipped) — flip chain confirmed.
- **Discovery (new fact F2b needed)**: even at the same named camera, raw
  `mujoco.Renderer` vs robosuite obs differ by MAD ≈ 15 — a *visualization-flag*
  mismatch (robosuite renders geom group 1 visuals only; default `MjvOption` also draws
  other groups/sites). `probe_render_facts.py` must sweep `MjvOption`
  (`geomgroup`, `sitegroup`, relevant `mjVIS_*` flags) to find the parity setting,
  record it as **F2b**, and all capture/oracle-comparison renders must apply it.

## Plan deviations recorded (per the M0 gate, resolved here, not improvised)

1. **F5 simplification** — depth already metric (above).
2. **F2b addition** — renderer vis-flag parity fact (above).
3. **Config path** — `oat/config/...`, not `conf/...`.
4. **M9 / arm A6** (`gs_oracle_robot` depth composite): deferred per §10
   ("build only if A3−A4 turns out large"); the `meta/render_source` enum reserves the
   value. A5 hybrid assembly IS built (cheap, §7.6).
5. **e3nn** — not installed; `sh_rotation.rotate_sh_wigner` raises a clear
   ImportError-with-instructions unless e3nn is present (G5 default off, unchanged).

## Gate

All M0 items confirmed. Proceed to M1.

---

# Post-implementation review addendum (2026-07-31)

Adversarial review of the finished implementation against real demo data produced
further measured corrections (full list in the plan's M0/post-review block):

- **Pure-z object deltas do not exist in practice** — capture-reset vs demo resting
  poses differ by settling tilt on every frame, grasped objects tumble; object SH now
  rotates through an exact in-house SO(3) path (`rotate_sh_so3`, no e3nn), mode
  `so3_deg3` with a pure-z fast path.
- **Orientation chain corrected**: raw Renderer == stored zarr orientation (measured
  bit-exact), so `gsplat_flip_ud = false`; an earlier inverted derivation would have
  written upside-down GS frames — fixed and re-measured.
- EEF probe thresholds recalibrated (oracle GT floor: median 3.45 px / p95 12.3 px →
  defaults 6/16 px); unbounded per-θ SH cache removed; EEF metric anchor fixed
  (was silently the robot mount); F2b `flags_off` is mjRND (scene-flag) namespace;
  A5 hybrid copy now validates source provenance + completeness; GS resume hard-fails
  on retrained assets (manifest sha mismatch); holdout views excluded from depth init;
  manifest writes atomic.
- gsplat 1.5.3 quirk: `DefaultStrategy`'s opacity reset never fires (precedence bug);
  the trainer calls `reset_opa` on schedule itself.

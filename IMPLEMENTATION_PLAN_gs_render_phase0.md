# Implementation Plan — Phase 0: Gaussian-Splatting Render Source for SE(2) Augmentation (LIBERO)

**Target:** coding agent working in the OAT repo (the codebase documented in `docs/se2_data_augmentation.md`; that doc is the authoritative reference for every existing component named below).
**Scope:** Phase 0 only. Replace the *pixel source* of the SE(2) augmentation — MuJoCo oracle re-render — with a compositional 3D Gaussian Splatting renderer built from sim-captured assets, and run the renderer × augmentation factorial (arms A1–A4, optional A5/A6). Everything downstream of pixels is reused unchanged: label math (`rotate_action_chunk`/`rotate_proprio`), frozen group-compatible normalization (D2/D3), probe-1 gating (D7), valid-mask semantics, matched budget, vanilla-env rollout eval.
**Out of scope (deferred, do not build):** scene-level reconstruction with SAM segmentation + inpainting (Level-1, separate plan), real-robot capture, relightable GS, dynamic/4D reconstruction, diffusion polish, shadow proxies.

---

## 0. The one property that governs everything

Phase 0 exists to measure **the cost of neural rendering in policy space**, cleanly:

```
ΔSR(rendering) = SR(A3: aug @ oracle renderer) − SR(A4: aug @ GS renderer)
```

For that subtraction to mean anything, the two arms must differ in **pixels only**. Every design decision below serves this:

1. **Same states.** Both arms render the identical rewritten states from `rewrite_state(states[t+δ], θ)`.
2. **Same support.** `valid_mask`, `p_base`, `state_offset` are identical across arms — validity is decided in state space (MuJoCo contacts), never from GS output, and the GS pre-render hard-asserts equality against the oracle aug zarr.
3. **Same labels, same normalizer, same budget.** Untouched by construction: the dataset only swaps which zarr it reads images from.
4. **Same eval.** All arms roll out in the vanilla LIBERO env. GS-trained arms therefore face a train→eval visual domain shift *by design* — that shift is the sim analogue of the GS→real gap at deployment, and is part of what Phase 0 measures.

Any code path where a GS artifact could leak into states, labels, validity, or normalization is a bug by definition.

---

## 1. Design decisions (G-tags)

Existing decisions D1–D3, D7 remain in force. New decisions carry G-tags; reference them in code comments the same way.

| Tag | Decision |
|---|---|
| G1 | **Asset-based reconstruction**, not scene-based: per-task capture of background (movables removed), each movable object solo, and the robot — composed at render time from GT poses. No segmentation-lifting, no inpainting, no residual shadows from removed objects. Scene-based reconstruction is Level-1 and out of scope. |
| G2 | **Single rasterization pass** per camera per frame over the concatenated Gaussian set (background + posed objects + posed robot links). Image-space compositing of separately rendered components is forbidden — it breaks occlusion. (Exception: the optional A6 depth-composite mode, §10, which is explicitly a hybrid.) |
| G3 | **MuJoCo stays in the loop for everything except rasterization**: `set_state_from_flattened → forward()` still runs per frame to provide object body poses, robot link poses, eye-in-hand camera extrinsics, and contact-based validity checks. Only `render` is replaced. |
| G4 | **Validity is oracle-owned.** The GS pre-render recomputes the state-space checks and hard-asserts `valid_mask == oracle_zarr.meta/valid_mask` (likewise `p_base`, `angles_deg`, `episode_ends`). δ is **copied** from the oracle zarr's `meta/state_offset`, not recalibrated (calibration compares oracle renders to stored frames and is renderer-independent); a fresh `resolve_addresses` run must agree with the copied `p_base` to 1e-6. |
| G5 | **SH policy.** Background: degree 3, never rotated. Objects: degree 3, rotated only by `R_z(θ)` via the closed-form per-degree ±m block rotation (§6.3). Robot links: degree 1, rotated under full SO(3) via the exact l=1 vector rule. General SO(3) degree-3 rotation (e3nn `wigner_D`) is a scaffolded upgrade path, enabled only if partitioned metrics show the robot-region gap is dominated by view-dependent shading. A transform that moves a component's means/quats but not its SH is invalid and asserted against. |
| G6 | **Provenance uniformity within an arm** (D1 generalized): GS arms read GS renders at θ=0 too. Arm A5 (θ=0 from oracle, θ≠0 from GS) is the *controlled exception* that tests this decision. |
| G7 | **Conventions are measured, then asserted — never assumed.** Camera axis convention, image orientation, headlight contribution, depth mapping, and hide-mechanism behavior are Stage-0 facts recorded in `gs_render_facts.json`; every downstream constructor loads the facts file and asserts `pass==true`. Mirrors the probe-1 gating pattern. |
| G8 | **Capture at 512², render at dataset-native cameras.** Assets are trained on 512² orbit captures with GT intrinsics/extrinsics; composite renders use the base zarr's camera names, resolution (read from the zarr, do not hard-code), and per-frame extrinsics from the forwarded sim. |
| G9 | **Content-addressed provenance.** Every asset file carries a SHA-1 over its parameter tensors; the per-task manifest hashes assets + capture/training params + the model-XML hash of the task env. The GS aug zarr records `meta/render_source` and `meta/gs_manifest_sha1`; the dataset can assert an expected render source (§9). |
| G10 | **Per-task robot asset**, not one shared asset: LIBERO-10 scenes have different lighting and baked appearance must match the scene it will be composited into. (The kinematic structure is shared; only appearance differs.) Revisit only if capture/training cost becomes a problem — it should not (§12). |

---

## 2. New package layout

```
oat/gsaug/
├── cameras.py          # fovy→K, MuJoCo cam_xpos/xmat → OpenCV w2c, GL/CV flip per facts F1
├── capture.py          # orbit pose generation, raw mujoco.Renderer RGB+depth+seg capture
├── gaussian_asset.py   # GaussianAsset container: params, conventions block, IO, sha1
├── sh_rotation.py      # closed-form z-rotation (l≤3), exact l=1 SO(3) rotation, e3nn scaffold
├── components.py       # PosedComponent abstraction (§6.1) — objects, links, background
├── trainer.py          # gsplat training loop: depth init, masked loss, silhouette loss, densify
├── articulated.py      # RobotAsset: per-link locals, FK posing, multi-config finetune
└── compose.py          # GSCompositeRenderer (§6.2)

scripts/gsaug/
├── probe_render_facts.py    # Stage 0 → data/libero/gs_render_facts.json
├── capture_assets.py        # Stage 1
├── train_gs_assets.py       # Stages 2–3
└── report_factorial.py      # Stage 8

scripts/probes/
├── probe_gs_geometry.py     # GATING (§8.1)
└── probe_gs_photometric.py  # report-only (§8.2)

tests/
├── test_gsaug_transforms.py
├── test_gsaug_compose.py
└── test_gsaug_prerender.py

data/libero/gs_assets/<task_name>/
├── captures/{background, objects/<obj_name>, robot}/   # imgs, depth.npy, seg.png, transforms.json
├── assets/{background.pt, objects/<obj_name>.pt, robot.pt}
└── manifest.json
```

Touched existing files (and nothing else): `scripts/prerender_se2_aug.py` (§7), `oat/dataset/se2_aug_zarr_dataset.py` (one optional ctor arg, §9), Hydra configs (§9). New Python dependencies: `gsplat` (pin the tested version in the manifest), `lpips`; `e3nn` optional behind the G5 upgrade flag.

---

## 3. M0 — Discovery & environment verification (½ day)

Do not assume; verify. Deliverable: a short `M0_NOTES.md` recording each item.

- Confirm the file map in `docs/se2_data_augmentation.md` §9 against the live repo (paths, function names used in this plan: `rewrite_state`, `resolve_addresses`, `regenerate_obs_from_state`, `table_top_xy_aabb`, the four validity checks, the two-pass driver structure, report writer, resume/shard logic).
- Confirm the oracle aug zarr exists with a report (`libero10_N500_se2aug.zarr` + `.report.json`); record its `angles_deg`, per-angle valid rates, image resolution and camera names — these parameterize everything downstream.
- Install `gsplat`; smoke-render a synthetic 3-Gaussian scene on one 4090; record version + FPS at 128² and 512².
- Locate the raw-MuJoCo render path available from the LIBERO `ControlEnv` (`env.sim.model` / `env.sim.data` handles; ability to instantiate `mujoco.Renderer` with a programmable camera, depth and segmentation modes). If the installed robosuite/mujoco version routes rendering differently, record the working recipe here — every later stage uses it.
- **Gate:** all items confirmed; any mismatch with this plan is resolved by editing the plan, not by improvising downstream.

> **M0 outcome (2026-07-31): PASSED — see `M0_NOTES.md`.** Deviations folded into this
> plan: (a) F5 needs no depth conversion (mujoco.Renderer depth is already metric);
> (b) new fact **F2b** — raw-Renderer visualization flags must be swept to match
> robosuite obs rendering (MAD ≈ 15 at stock flags); (c) configs live under
> `oat/config/...`, not `conf/...`; (d) arm A6 (§10) deferred per its own build-only-if
> criterion; (e) e3nn stays uninstalled — the wigner path imports it lazily.
>
> **Post-implementation review outcome (2026-07-31).** Measured corrections to this
> plan's assumptions, found by adversarial review against real demo data:
> (f) **§6.1's pure-z premise is false on real data** — the world delta between an
> object's *capture* pose (fresh reset) and its *demo* pose is never a pure
> z-rotation (resting-settle tilt |q_xy| ≈ 2e-3–3e-3 on 100% of frames; grasped
> objects tumble to |q_xy| ≈ 0.8). R7 therefore fired universally, and the remedy is
> an **exact dependency-free SO(3) real-SH rotation for l ≤ 3**
> (`sh_rotation.rotate_sh_so3`, per-band matrices built by exact projection): object
> components use mode `so3_deg3` (closed-form z fast path when the delta *is* pure-z,
> exact SO(3) otherwise); the pure-z assertion and the e3nn contingency are replaced —
> G5's "SH rotates whenever means rotate" now holds unconditionally.
> (g) The F2 orientation chain in §4 is corrected by measurement: raw-Renderer
> orientation **equals** stored-zarr orientation (raw = flip(obs) bit-exactly under the
> F2b parity flags, and the zarr stores flip(obs)), so gsplat output needs **no** flip
> (`gsplat_flip_ud = false`).
> (h) §8.1's EEF thresholds (2 px / 4 px) are below the measured oracle-ground-truth
> floor (median 3.45 px / p95 12.3 px); probe defaults recalibrated to 6 px / 16 px per
> R6/§11.
> (i) §7.4's "§8.2 metrics written into the report JSON" live canonically in
> `probe_gs_photometric.json` (same ≤64-frame budget); the pre-render report carries the
> θ=0 MAD stats only.
> (j) Concurrent GS-mode `--tasks` shards are NOT provenance-safe (zarr root-attrs
> merges are last-writer-wins): run GS shards sequentially.

---

## 4. M1 — Renderer facts (Stage 0, GATING)

`scripts/gsaug/probe_render_facts.py --out data/libero/gs_render_facts.json`. JSON with top-level `pass` and one record per fact; nonzero exit on failure. All downstream ctors (`GSCompositeRenderer`, both probes, the GS pre-render path) load this file and assert `pass`.

| Fact | Method | PASS |
|---|---|---|
| F1 camera convention | Project the EEF site of a random demo state through our math (`cam_fovy→K`, `cam_xpos/xmat → w2c`, candidate GL→CV flips) for both cameras; compare against the site's pixel location from a MuJoCo seg render. Record the winning flip matrix. | ≤ 0.5 px both cameras |
| F2 image orientation | Establish the flip chain raw-MuJoCo → stored-zarr (known: `np.flip(axis=0)`) and gsplat-output → stored-zarr on one rendered state. Record both. | orientation resolved & unique |
| F2b renderer vis parity | Sweep `MjvOption` (geom groups, site groups, `mjVIS_*` flags) to minimize raw-Renderer vs robosuite-obs MAD at the same camera/state; record the winning flags. | MAD ≤ 2.0 with winning flags |
| F3 headlight | Render one state from agentview with `model.vis.headlight.{ambient,diffuse,specular}` zeroed vs stock; MAD in uint8 units. | record-only. If MAD > 1.0: keep stock lighting during capture (SH view-dependence absorbs a camera headlight approximately — that is what SH is for); note residual in the risk register. |
| F4 hide mechanisms | (a) Movables teleported to graveyard (`xy += 50`): their seg ids absent; rgb diff vs baseline confined to former silhouette + shadow region. (b) Robot geoms `rgba[...,3]=0`: robot seg ids absent AND rgb diff confined to robot silhouette + shadow (i.e. alpha-0 removes cast shadows too). | (a) must pass. If (b) fails (shadow residue or seg leakage): set `facts.robot_hide='masked'` — Stages 1–2 then capture background *with* the robot at a stow config and train with per-pixel robot masks. |
| F5 depth sanity | Depth from `mujoco.Renderer` is already metric (M0). Backproject table-plane pixels; fit plane. | planarity ≤ 2 mm RMS, height matches `table_top_xy_aabb` z |
| F6 id maps | Build geom→body→{object name \| robot link} from the model; every movable free joint from `resolve_addresses` and every robot link must resolve. | complete, no orphans |
| F7 perf | gsplat FPS at dataset resolution with a ~300k-Gaussian scene. | record-only |

---

## 5. M2–M3 — Capture and static assets

### 5.1 Capture (`scripts/gsaug/capture_assets.py --task T --component {background|objects|robot|all}`)

Per task, on a fresh `ControlEnv` reset, using the raw `mujoco.Renderer` with a programmable camera (RGB + depth + seg per view, 512², intrinsics/extrinsics written to `transforms.json` in OpenCV c2w per F1):

- **Background**: movables graveyarded; robot hidden per F4 (alpha-0, or present-at-stow + masks if `facts.robot_hide=='masked'`). Orbit: 2 elevation rings (25°, 50°) × 24 azimuths + 8 top-down → 56 views, radius fitted to 1.6× the table-AABB diagonal, look-at = table center.
- **Objects** (one asset per movable free joint, names from `resolve_addresses`): solo on the table, **floated** `z += 0.15 m` so the underside is visible (held objects expose bottoms at render time; sim gives full-sphere coverage for free). 3 rings (−20°, 20°, 55°) × 16 azimuths = 48 close-radius views.
- **Robot** (per task, G10): movables graveyarded, robot visible. Configs: 36 qpos sampled from that task's demo frames, farthest-point-deduplicated in joint space, **plus** joint1-shifted copies of 12 of them at ±20°, ±30° (the θ-grid tails must be inside the training distribution), **plus** an assertion that the config set spans the demo gripper-qpos range (open and closed). ~60 configs × 16 views. Per-view per-pixel link ids from the seg render (F6).

Fail-fast per asset: min view counts; seg purity (background captures contain zero movable/robot ids under alpha-0 mode); ≥ 99% finite depth; `transforms.json` schema; capture manifest records the task model-XML SHA-1 (G9 — render time asserts the same hash).

### 5.2 Static asset training (`train_gs_assets.py`, `oat/gsaug/trainer.py`)

- **Init from depth** (skip SfM entirely): backproject stride-4 pixels across all views, seg-filtered to the component, voxel-downsampled (background 1 cm, objects 3 mm).
- **Frames**: object Gaussians are stored in the object **body frame** at capture (backproject into body-local using the captured body pose) — see §6.1 for why. Background Gaussians live in world frame.
- **Loss**: L1 + 0.2·D-SSIM on component pixels; silhouette loss `|α_rendered − mask|` over the full image (weight 1.0) for objects (and for the robot-masked background mode). Standard densify/prune schedule; 7k iters background, 5k per object.
- **Held-out**: every 8th view. Acceptance (provisional floors — gross-failure catches, recalibrated after the M6 dry run and recorded in the manifest, not in code): object-region PSNR ≥ 30 (objects), full-image PSNR ≥ 29 (background); object held-out silhouette IoU ≥ 0.95.
- **Asset file** (`gaussian_asset.py`): `{means f32[N,3], quats f32[N,4], log_scales, opacity_logits, sh_dc[N,3], sh_rest[N,K,3]}` + a `conventions` block (quat order, SH ordering/normalization as used by the pinned gsplat version, frame = body|world, capture body pose) + sha1. Refuse to load an asset whose conventions block disagrees with the running code's expectations.

---

## 6. M4–M5 — Articulated robot asset and the compositional renderer

### 6.1 The `PosedComponent` abstraction (`components.py`)

Objects and robot links are the same thing: **a rigid Gaussian set in a body-local frame, posed at render time by that body's current world pose from the forwarded sim** (`data.xpos/xquat`). Background is the degenerate identity-posed component. One code path, one test surface:

```python
class PosedComponent:
    # local params: means_l, quats_l, sh_dc, sh_rest, log_scales, opacity_logits
    # sh_rot_mode: 'z_only_deg3' (objects) | 'so3_deg1' (links) | 'static' (background)
    def posed(self, p_wb: np.ndarray, q_wb: np.ndarray) -> WorldGaussians:
        # means:  p_wb + R(q_wb) @ means_l
        # quats:  q_wb ⊗ quats_l            (left multiply = world-frame; both wxyz here)
        # SH:     rotate per sh_rot_mode; 'z_only_deg3' additionally asserts R(q_wb)
        #         relative to the capture pose is a pure z-rotation (tol 1e-5) — G5
        # scales, opacities: unchanged
```

Because object Gaussians are stored in the body frame, the render-time transform **is** the rewritten body pose — no `T_capture⁻¹` bookkeeping, and the pure-z assertion for objects is checked against `q_capture⁻¹ ⊗ q_current` being a z-rotation, which is exactly what `rewrite_state` guarantees (it left-multiplies `q_z(θ)` onto poses whose support surface is horizontal-preserving). If a task ever violates this (an object whose demo poses tumble), the assertion fires and that object's `sh_rot_mode` gets upgraded to the e3nn path — measured, not assumed (G7 spirit).

### 6.2 `GSCompositeRenderer` (`compose.py`)

```python
class GSCompositeRenderer:
    def __init__(self, assets_dir, cam_names, resolution, facts_path, device): ...
    def render(self, env, state_rw) -> dict[str, np.ndarray]:   # uint8 (H,W,3), dataset orientation
        # 1) env.sim.set_state_from_flattened(state_rw); env.sim.forward()   — no MuJoCo render (G3)
        # 2) object body poses + robot link poses from data.xpos/xquat
        #    (object body ids via resolve_addresses' addr map; link ids via facts F6)
        # 3) camera K from cam_fovy (G8 res), w2c from data.cam_{xpos,xmat} + F1 flip
        #    → eye-in-hand extrinsics thereby come from the *rewritten* state's FK, exact
        # 4) world_gaussians = concat(background, *(c.posed(...) for c in components))
        # 5) one gsplat rasterization per camera (G2), sh_degree = per-component (padded)
        # 6) orientation per facts F2 → uint8
```

Model-hash assert on construction (asset manifest vs live env, G9). Unknown free joint at render time (an object with no asset) → hard error naming the joint.

### 6.3 SH rotation (`sh_rotation.py`)

- `rotate_sh_z(sh, theta)`: for each degree l, the real-SH block transforms by 2×2 rotations of the ±m coefficient pairs by angle m·θ (m=0 fixed). Ten lines; exact for any l.
- `rotate_sh_l1(sh_l1, R)`: exact l=1 rule — the (y, z, x) permutation-conjugated application of R to the 3-vector of coefficients.
- `rotate_sh_wigner(sh, R)` (scaffold, `requires e3nn`): general SO(3) via `wigner_D`, with an explicit basis-ordering conversion to the gsplat SH layout; cross-checked against the two exact paths in tests before first use. **Not enabled by default** (G5).

### 6.4 Robot asset (`articulated.py`)

- **M4a init**: canonical-config capture → per-pixel link id → backproject each pixel into its link's local frame (`x_l = R_wlᵀ (x_w − p_wl)` at the capture config). Gaussians carry `link_id`; one `PosedComponent` per link.
- **M4b finetune**: over shuffled (config, view) pairs — pose all links from that config's FK (poses precomputed from the capture run's `data.xpos/xquat`), rasterize jointly, masked L1+SSIM on robot pixels + silhouette loss vs the robot mask; optimize **local** params (gradients flow through the differentiable pose transform). 15k iters. SH degree 1 (G5).
- **Acceptance**: 4 held-out configs, at least one joint1-shifted: robot-region PSNR ≥ 27 (provisional), per-link silhouette IoU ≥ 0.85, EEF-site projection error ≤ 2 px median at dataset resolution.

### 6.5 Unit tests (M5 gate — all must pass before M6)

- T1 rigid-transform consistency: rendering a transformed component equals rendering the original from the inversely-transformed camera (synthetic 3-Gaussian scene, PSNR ≥ 45).
- T2 SH z-rotation invariance: anisotropic-SH component; rotate component *and* camera jointly → image unchanged.
- T3 same as T2 for the l=1 SO(3) path with a random R.
- T4 occlusion: two overlapping components rendered via concat-then-rasterize equals a single merged reference set (bit-comparable modulo sort ties); an image-space composite of separate renders must *fail* this test (negative control for G2).
- T5 camera math round-trip vs facts F1 (≤ 0.5 px).
- T6 repeatability: two renders of the same inputs within uint8 tolerance 1.

---

## 7. M6 — Pre-render integration (touching `scripts/prerender_se2_aug.py`)

New flags: `--renderer {oracle,gs}` (default `oracle` — **zero behavior change** for existing runs), `--gs-assets-dir`, `--oracle-zarr PATH` (required with `gs` unless `--no-oracle-crosscheck`, which exists only for smoke tests and taints the report).

With `--renderer gs`:

1. The **only** pixel-path change: `env.regenerate_obs_from_state(state_rw)` → `gs_renderer.render(env, state_rw)`, then the F2 orientation fix into dataset orientation. Two-pass structure, sha1 episode matching, resume/shard, report writer: unchanged.
2. δ **copied** from `oracle_zarr.meta/state_offset` (G4), not recalibrated; `p_base` recomputed via `resolve_addresses` and asserted equal to the oracle's within 1e-6.
3. Validity checks run exactly as today (state-space, MuJoCo contacts — they never render). After each task, and again as a final invariant: `valid_mask == oracle.valid_mask` element-wise, plus `angles_deg` / `episode_ends` equality. Mismatch = hard failure of the run.
4. θ=0 gate re-scoped: the MAD ≤ 5.0 oracle gate becomes a **gross-error gate MAD ≤ 25** for `gs` (catches wrong δ / flip / camera; GS-vs-stored MAD will legitimately sit in the 5–20 band). Quality is reported, not gated, via §8.2 metrics on ≤ 64 sampled frames/task written into the report JSON.
5. New meta: `meta/render_source ∈ {oracle, gs, gs_hybrid0, gs_oracle_robot}`, `meta/gs_manifest_sha1` (per task, G9). Oracle-mode writes `render_source='oracle'` so old and new zarrs are self-describing.
6. `--hybrid-zero-from ORACLE_ZARR` (arm A5 assembly): after rendering, copy `images/*/angle_00` from the oracle zarr verbatim; `render_source='gs_hybrid0'`.

Output: `data/libero/libero10_N500_se2aug_gs.zarr` (+ `_gs_hybrid0` view).

---

## 8. M7 — Probe additions

### 8.1 `probe_gs_geometry.py` — GATING (mirrors probe 2's sampling: 200 random valid (episode, frame, θ) triples)

- **Silhouette IoU** per component: rasterize that component's Gaussians alone, threshold α ≥ 0.5, IoU against the oracle seg mask of the same rewritten state. PASS: p5 ≥ 0.90 (objects), p5 ≥ 0.85 (robot).
- **EEF projection error**: EEF site projected through the renderer's camera math vs its oracle seg-render location. PASS: median ≤ 2 px, p95 ≤ 4 px.
- **Wrist transform-stack check** (this is the free lunch — use it): under the group action the wrist camera co-rotates with everything that rotates, so *within the (movables ∪ robot) mask* the GS wrist render at θ must be pixel-identical to the GS wrist render at θ=0 up to rasterization noise. PASS: masked PSNR ≥ 32 over all sampled triples. This single check catches SH-rotation bugs, covariance-quaternion bugs, and camera-extrinsics bugs at once. Also compute the same quantity for **oracle** renders and report it (NOT a gate): the oracle's deficit from perfection is the fixed-light shading variation that baked GS appearance structurally cannot reproduce — a number worth having in the paper.
- **Consumer gating** (D7 pattern): `SE2AugZarrDataset` gains an optional `expected_render_source` ctor arg; when set to a `gs*` value it additionally requires a PASSing `probe_gs_geometry.json` and asserts the zarr's `meta/render_source` matches. No config change for existing arms.

### 8.2 `probe_gs_photometric.py` — report-only

θ=0 GS renders vs stored base frames, per task: **partitioned PSNR/SSIM** over oracle-seg masks {robot, movables, background} (mask-exact), **full-frame LPIPS**, **movable-bbox-crop LPIPS**, and a **contact-band PSNR** (pixels within 6 px of the movable silhouette's lower edge — the shadow-gap tracker). Written per task; `report_factorial.py` later correlates these against per-task A3−A4 SR deltas.

---

## 9. M8 — Configs, arms, training, report

Dataset change: the single optional `expected_render_source` arg (§8.1). Everything else is config wiring.

| Arm | Config | zarr | augment | Answers |
|---|---|---|---|---|
| A1 | existing no-aug | oracle | false | baseline (θ=0 oracle renders, D1) |
| A2 | `libero10_se2aug_gs` + `aug.enable=false` | gs | false | pure GS render-domain cost, no augmentation |
| A3 | existing aug | oracle | true | augmentation value @ perfect renderer (upper bound) |
| A4 | `libero10_se2aug_gs` | gs | true | **the arm**: GS augmentation |
| A5 (opt) | `..._gs_hybrid0` | gs_hybrid0 | true | is provenance uniformity (G6/D1) worth more than in-domain θ=0 frames? |
| A6 (opt) | `..._gs_oracle_robot` | gs_oracle_robot | true | attributes the A3−A4 gap: robot rendering vs scene rendering |

All arms: same frozen norm spec, probe-1 gate, matched budget (dataset `__len__` construction guarantees it), vanilla-env rollout eval with the existing protocol, ≥ 2 seeds (3 preferred). Key contrasts the report must print: `A3−A1` (aug value @ oracle), `A4−A2` (aug value @ GS), `A3−A4` (rendering cost under augmentation), `A1−A2` (pure domain cost), per task and pooled, with seed std; plus the correlation of per-task `A3−A4` against §8.2 partitioned metrics.

`scripts/gsaug/report_factorial.py` reads eval jsons + both prerender reports + photometric probe output → one markdown/CSV table.

---

## 10. M9 (optional) — A6 renderer mode `gs_oracle_robot`

Per frame: oracle-render the rewritten state with movables graveyarded (F4-a) → robot RGB + depth + robot mask; GS-render background+objects with expected depth; per-pixel: robot pixel wins where `depth_robot < depth_gs`. This is the one sanctioned image-space composite (explicitly exempted from G2 because both sides carry depth). ~1 day; build only if A3−A4 turns out large and needs attribution.

---

## 11. Runbook (dependency order)

```bash
# prerequisites (already done, unchanged): norm specs → probe 1 PASS → ORACLE prerender
python scripts/gsaug/probe_render_facts.py --out data/libero/gs_render_facts.json          # M1
python scripts/gsaug/capture_assets.py --task TASK --component all                          # M2, shardable per task
python scripts/gsaug/train_gs_assets.py --task TASK --component background|objects|robot    # M3–M4
pytest tests/test_gsaug_*.py                                                                # M5 gate
python scripts/prerender_se2_aug.py --renderer gs \
    --gs-assets-dir data/libero/gs_assets \
    --oracle-zarr  data/libero/libero10_N500_se2aug.zarr \
    --out          data/libero/libero10_N500_se2aug_gs.zarr                                 # M6
python scripts/probes/probe_gs_geometry.py   ... # GATING                                    # M7
python scripts/probes/probe_gs_photometric.py ...
# M8: train A2/A4 (A1/A3 exist), then:
python scripts/gsaug/report_factorial.py
```

**Sequencing rule:** run M1–M8 end-to-end on ONE task first (pick a LIVING_ROOM task — ~100% valid rate), including a short A4 smoke train, before fanning out to all 10 tasks. Recalibrate every provisional threshold from that dry run and record the final values in the facts file / manifests — thresholds live in data, not in code comments.

---

## 12. Budget (dual RTX 4090; GS stages fit on one)

- Assets: 10 tasks × (1 bg + 2–4 obj + 1 robot) ≈ 50–70 trainings × 2–4 min → **3–5 GPU-h**; capture ≈ 1–2 h.
- GS pre-render: ~500 ep × ~250 f × 7 angles × 2 cams × ~62% valid → **1.4 M images**; at ≥ 150 FPS composited → **3–4 h** including kinematics overhead.
- Training: only A2/A4 (+A5/A6) are new runs; reuse the existing per-arm budget × seeds.

---

## 13. Risk register

| Risk | Handling |
|---|---|
| R1 No light transport: composed movables cast no contact shadows; baked shading doesn't respond to rotation. | Expected headline artifact. Tracked by the contact-band metric and the oracle wrist-deficit number (§8.1); mitigation (shadow proxy / relightable GS) is Phase-1 material, not built here. |
| R2 Headlight view-dependence (F3). | SH absorbs a camera headlight approximately; residual is measured at the θ=0 report and listed as a known limitation. No protocol change by default. |
| R3 Robot articulation quality: joint-boundary tearing, thin fingers. | Labeled per-link init + multi-config FK finetune + gripper-range assertion; A6 exists to isolate its SR impact if needed. |
| R4 Convention landmines: MuJoCo wxyz vs dataset eef xyzw vs gsplat quat order; GL vs CV cameras; three flip conventions. | All pinned by M1 facts + T5; every asset carries a conventions block; loaders refuse mismatches. |
| R5 gsplat/CUDA versioning (sm_89). | Pin + record versions in manifests. Renders are cached to zarr offline, so rasterizer nondeterminism is not load-bearing (T6 only asserts tolerance-1 repeatability). |
| R6 Provisional thresholds mis-set. | They are gross-failure catches only; the one-task dry run (§11) recalibrates them before the fleet run. |
| R7 An object whose demo poses are not z-rotations of its capture pose (tumbled object). | The `z_only_deg3` assertion in `PosedComponent.posed` fires with the object name; upgrade that object to the e3nn SH path (G5) rather than silently mis-shading. |

---

## 14. Invariants (extends `docs/se2_data_augmentation.md` §8)

- **Pixels are the only difference between oracle and GS runs.** Any divergence in `valid_mask`, `p_base`, `state_offset`, `angles_deg`, `episode_ends` is a bug by definition (G4 asserts make it unrepresentable).
- **One rasterization pass per camera per frame** over concatenated Gaussians (G2); the only sanctioned exception is A6's depth composite.
- **SH rotates whenever means rotate** — asserted inside `PosedComponent.posed` (G5).
- **The facts file is load-bearing**: renderer, probes, and the GS pre-render path all assert `gs_render_facts.json` pass; conventions are never assumed twice (G7).
- **Assets are bound to tasks by model-XML hash** (G9/G10); a renderer constructed against the wrong task env must fail at construction, not produce subtly wrong pixels.
- **`expected_render_source` follows the D7 pattern**: a GS training arm cannot start without a PASSing geometry probe and a self-describing zarr.

## 15. File map (delta)

| File | Role |
|---|---|
| `scripts/gsaug/probe_render_facts.py` | M1: measured renderer facts → `gs_render_facts.json` (gating) |
| `scripts/gsaug/capture_assets.py` | M2: per-task background/object/robot capture (RGB+depth+seg+poses) |
| `scripts/gsaug/train_gs_assets.py` | M3–M4: static + articulated asset training driver |
| `oat/gsaug/cameras.py` | intrinsics/extrinsics/convention math (facts-asserted) |
| `oat/gsaug/capture.py` | orbit generation + raw-renderer capture backend |
| `oat/gsaug/gaussian_asset.py` | asset container, conventions block, sha1, IO |
| `oat/gsaug/sh_rotation.py` | z-rotation closed form; exact l=1 SO(3); e3nn scaffold |
| `oat/gsaug/components.py` | `PosedComponent` — the single object/link/background code path |
| `oat/gsaug/articulated.py` | per-link robot asset: labeled init, FK finetune |
| `oat/gsaug/trainer.py` | gsplat loop: depth init, masked + silhouette loss, densify |
| `oat/gsaug/compose.py` | `GSCompositeRenderer` (G2/G3) |
| `scripts/prerender_se2_aug.py` | **touched**: `--renderer` flag, G4 cross-asserts, gate re-scope, provenance meta, A5 assembly |
| `scripts/probes/probe_gs_geometry.py` | GATING: silhouette IoU, EEF projection, wrist transform-stack check |
| `scripts/probes/probe_gs_photometric.py` | report-only partitioned metrics |
| `oat/dataset/se2_aug_zarr_dataset.py` | **touched**: optional `expected_render_source` (D7-pattern gate) |
| `scripts/gsaug/report_factorial.py` | M8: factorial table + metric→SR correlation |
| `tests/test_gsaug_{transforms,compose,prerender}.py` | T1–T6 + prerender integration pins |
| `oat/config/.../libero10_se2aug_gs*.yaml` | arm configs A2/A4/A5/A6 |

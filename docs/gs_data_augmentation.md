# Gaussian Splatting for Data Augmentation — What Goes In, What Comes Out

How 3D Gaussian Splatting (3DGS) is used in this repo to manufacture augmented
LIBERO demonstrations: what the GS pipeline consumes, how assets are built, how
a frame is composited at render time, and how the result becomes new training
data. Companion docs: [`se2_data_augmentation.md`](se2_data_augmentation.md)
(the augmentation itself — labels, validity, dataset), the plan of record
[`../IMPLEMENTATION_PLAN_gs_render_phase0.md`](../IMPLEMENTATION_PLAN_gs_render_phase0.md)
(G-tag decisions), and [`../M0_NOTES.md`](../M0_NOTES.md) (measured environment
facts).

---

## 0. The one-paragraph mental model

GS **does not invent new trajectories**. The augmentation is a world-frame yaw
`R_z(θ)` about the robot base: the demo's stored MuJoCo states are rewritten
exactly (objects rotated about `p_base`, robot joint 1 shifted by θ), and the
labels are rotated analytically in the dataloader. All GS changes is **where the
pixels come from**. The baseline ("oracle") augmentation re-renders the rewritten
state with MuJoCo; the GS arm replaces that one call with a compositional
Gaussian rasterizer built from sim-captured assets. Everything else — states,
validity, labels, normalization, budget, evaluation — is identical by
construction and hard-asserted to be so (G4).

```
ΔSR(rendering) = SR(aug @ MuJoCo renderer) − SR(aug @ GS renderer)
```

That subtraction is the entire point of the Phase-0 build: GS is the *stand-in
for a real-world reconstruction*, and this measures what neural rendering costs
in policy space when everything else is held fixed.

Why an asset-based reconstruction and not a scene reconstruction (G1): the
augmentation moves objects and the arm. A single baked scene cannot be
re-posed. So the scene is decomposed into rigid pieces — background, one asset
per movable object, one per robot link — captured separately and **re-composed
at render time from ground-truth poses the simulator hands us**.

---

## 1. Pipeline at a glance

```
                       ┌──────────────────── inputs ────────────────────┐
LIBERO BDDL task ──► ControlEnv (MuJoCo)          demo HDF5 (robot qpos pool)
        │                     │                              │
        │  M1 probe_render_facts.py  ──► gs_render_facts.json (camera flip, vis
        │                     │            flags, hide mechanism, depth, id maps)
        ▼                     ▼                              ▼
  M2  capture_assets.py: orbit RGB + metric depth + geom-id seg @512²
        │   background (movables teleported away, robot hidden)
        │   objects    (one per free joint, solo + floated)
        │   robot      (~60 demo qpos configs, movables away)
        ▼
      captures/<component>/{view_XXXX.png, _depth.npy, _seg.png, transforms.json}
        │
  M3-4  train_gs_assets.py → trainer.fit_static / articulated.fit_robot
        │   depth-backprojection init → gsplat photometric fit
        ▼
      assets/{background.pt, objects/<free_joint>.pt, robot.pt} + manifest.json
        │
  M6  prerender_se2_aug.py --renderer gs
        │   per (episode, angle, frame):
        │     rewrite_state(states[t+δ], θ) ──► set_state + forward (MuJoCo, G3)
        │     poses of every object body + robot link + both cameras
        │     ──► GSCompositeRenderer: concat all Gaussians, ONE rasterization
        ▼
      libero10_N500_se2aug_gs.zarr  (images only, + provenance meta)
        │
  M7  probe_gs_geometry.py (GATING) / probe_gs_photometric.py (report)
        ▼
  M8  SE2AugZarrDataset — images from the GS zarr, labels rotated analytically
```

---

## 2. What the GS pipeline takes as input

Three distinct input layers; conflating them is the usual source of confusion.

### 2.1 Inputs to **asset building** (offline, once per task)

| Input | Source | Used for |
|---|---|---|
| LIBERO task (BDDL file) | `benchmark.get_benchmark_dict()` → `ControlEnv` with `camera_names=[agentview, robot0_eye_in_hand]`, `has_offscreen_renderer=True` | the scene to be reconstructed; a **fresh reset**, not a demo frame |
| Renderer facts | `data/libero/gs_render_facts.json` (M1 probe) | GL→CV camera flip (F1), image orientation (F2), MuJoCo visualization-flag parity with robosuite obs (F2b), robot hide mechanism (F4), depth sanity (F5), geom→body→link maps (F6) |
| Demo HDF5 (`*_demo.hdf5`) | `third_party/LIBERO/libero/datasets/libero_10` | **only** the robot `qpos` pool: ~60 arm/gripper configurations for the robot asset. No images, no actions |
| Free-joint inventory | `resolve_addresses(env)` (shared with the SE(2) rewrite) | which movable objects exist, and their qpos slices |
| Capture resolution | `--image_size 512` (G8) | orbit captures are 512²; composite rendering later happens at the dataset's native 128² |

Note what is **not** an input: no real photos, no COLMAP/SfM, no dataset frames.
Geometry initialization comes from MuJoCo's metric depth buffer, and camera
poses are known analytically — the classic SfM stage is skipped entirely.

### 2.2 The on-disk capture format (the trainer's actual input)

Written by [`scripts/gsaug/capture_assets.py`](../scripts/gsaug/capture_assets.py),
loaded by `CaptureBundle` ([`oat/gsaug/capture.py:939`](../oat/gsaug/capture.py#L939)):

```
data/libero/gs_assets/<task>/captures/<component>/
├── view_0000.png          uint8 (512,512,3)   RGB
├── view_0000_depth.npy    float32 (512,512)   METRIC meters (M0: no conversion needed)
├── view_0000_seg.png      uint16              geom id + 1 (0 = nothing)
├── masks/…                                    robot masks, only if facts.F4 robot_hide=='masked'
└── transforms.json        K, per-view OpenCV c2w, cam params, component/task,
                           model-XML sha1, plus per-component extras:
                             objects → joint_name, body_name, body_pose{p,q_wxyz}, object_geom_ids
                             robot   → configs[{qpos, view_ids, link_poses}], link_names, geom_to_link
```

`<component>` ∈ `background`, `objects/<name>`, `robot`.

### 2.3 Inputs at **render time** (per augmented frame)

| Input | Source |
|---|---|
| Rewritten flattened MuJoCo state `[time, qpos, qvel]` | `rewrite_state(states[t+δ], θ, addr)` — the same function the oracle arm uses |
| Object body poses `data.xpos/xquat`, robot link poses, camera `cam_xpos/cam_xmat` | the **forwarded** sim (`set_state_from_flattened → forward()`); MuJoCo stays in the loop for everything except rasterization (G3) |
| Trained assets + `manifest.json` | `data/libero/gs_assets/<task>/assets/` |
| Facts file | `gs_render_facts.json` (must PASS, else the renderer refuses to construct) |
| Output resolution + camera names | read from the base zarr (128², `agentview_rgb` / `robot0_eye_in_hand_rgb`), never hard-coded (G8) |

---

## 3. Stage 1 — Capture (`scripts/gsaug/capture_assets.py`)

All three components are captured from **one freshly reset env** per task, with
a raw `mujoco.Renderer` driven by a programmable free camera
(`MjvCamera`, `mjCAMERA_FREE`) under the measured F2b visualization flags, so
capture pixels match the robosuite observation pipeline the dataset was built
with.

| Component | Scene setup | Orbit | Per-view checks |
|---|---|---|---|
| **background** | every movable free joint teleported `xy += 50 m` (F4-a); robot hidden by zeroing geom alpha (F4-b — this also removes its cast shadows) | rings at 25°/50° × 24 azimuths + 8 top-down (80°) = 56 poses; requested radius `1.6 × table-AABB diagonal`, **clamped per azimuth** to the nearest wall − 0.30 m (LIBERO rooms are smaller than the nominal orbit); point-blank wall views dropped; ≥ 32 must survive | seg purity (no movable/robot ids), ≥ 99 % finite depth |
| **objects** (one per free joint) | that object alone on the table, all others graveyarded, robot hidden; object **floated** `+z` (≥ 0.15 m, auto-raised so the low ring clears the tabletop) so its underside is reconstructed — held objects expose their bottoms at render time | rings at −20°/20°/55° × 16 azimuths = 48 views, radius `max(4×object diag, 0.35 m)` wall-clamped | object ≥ 30 px in **every** view, seg purity, finite depth |
| **robot** (per task, G10) | movables graveyarded, robot visible, appearance baked under that task's lighting | ~60 configs × 16 views (rings 20°/50° × 8 azimuths, per-config azimuth offset). Configs = 36 farthest-point-sampled from the task's demo `qpos` frames + joint1-shifted copies of 12 at ±20°/±30° (so the θ-grid tails are in-distribution) + an assertion that gripper open **and** closed are both present | robot ≥ 100 px per view (camera pulled in if room furniture pinches it), seg purity |

Two validation layers guard the recorded camera chain (G7 — conventions are
measured, never assumed):

- **per view**: the analytic OpenCV `c2w` we write to `transforms.json` is
  compared against the GL camera MuJoCo actually placed in the scene
  (`assert_free_camera_pose`, tol 2e-3);
- **once per capture**: `PoseValidator` checks the full `(K, c2w)` chain against
  *pixels* — a projected small-primitive geom center vs its seg centroid (≤ 1 px),
  or, where no such landmark exists, backprojected depth planarity against the
  table-top box / floor plane (median ≤ 5 mm).

Every capture also records the task's canonicalized model-XML SHA-1 (G9), which
binds the assets to that exact scene build.

---

## 4. Stage 2 — Asset training

### 4.1 Static assets — background and objects ([`oat/gsaug/trainer.py`](../oat/gsaug/trainer.py))

- **Initialization from depth, not SfM**: backproject stride-4 pixels of every
  *training* view into world points, seg-filter to the component's geom ids,
  voxel-downsample (1 cm background, 3 mm objects). Each surviving voxel becomes
  a Gaussian: mean = voxel center, identity rotation, `log_scale = log(2·voxel)`,
  opacity logit 0 (α = 0.5), SH DC from the source pixel color, higher SH bands
  zero. Held-out views are excluded from the init so the held-out metrics are
  honest.
- **Loss** per iteration (one random training view):
  `L1 + 0.2·(1 − SSIM)` restricted to the component's seg mask, plus a
  full-image **silhouette** term `|α − mask|` (weight 1.0) for objects — this is
  what stops an object asset from growing floaters outside its true outline.
- **Densify/prune**: gsplat 1.5.3 `DefaultStrategy`, refine 500 → iters/2 every
  100 steps. (The trainer applies periodic opacity resets itself: 1.5.3's
  strategy has an operator-precedence bug that makes its own reset never fire.)
- **Budget**: 7 000 iterations background, 5 000 per object. SH degree 3.
- **Held-out**: every 8th view. Provisional acceptance floors (recorded as data
  in the manifest, WARN-only in code): object-region PSNR ≥ 30, background PSNR
  ≥ 29, object silhouette IoU ≥ 0.95.
- **Frame conversion at save time** — this is the load-bearing detail. Training
  happens in world frame at the single capture pose; object assets are then
  re-expressed in the **capture body frame** (`means ← R(q_cap)ᵀ(x − p_cap)`,
  `quats ← q_cap⁻¹ ⊗ q`) while SH stays world-at-capture. Background assets stay
  in world frame with an identity capture pose.

### 4.2 Articulated robot asset ([`oat/gsaug/articulated.py`](../oat/gsaug/articulated.py))

One asset per task holding **all links**, with a per-Gaussian `link_id`.

- **Init (M4a)**: from the canonical (first) config only — per-pixel link id via
  the seg render + the capture's `geom_to_link` map, backproject at stride 2,
  and express each point in *its link's local frame* at that config
  (`x_l = R_wlᵀ(x_w − p_wl)`), voxel-downsampled at 4 mm per link.
- **Finetune (M4b)**: 15 000 iterations over random (config, view) pairs. Each
  iteration poses **all** links differentiably from that config's recorded FK
  (`means = p + R(q)·x_l`, `quats = q ⊗ q_l`, SH rotated by the link's delta
  rotation vs canonical), rasterizes jointly, and backprops masked L1 + SSIM +
  silhouette into the **local** parameters. No densification (the labeled init is
  already dense); one opacity prune at iteration 2 000. SH degree 1 (G5).
- **Acceptance** (provisional, WARN-only): robot-region PSNR ≥ 27, per-link
  silhouette IoU ≥ 0.85, EEF projection error ≤ 2 px at 128².

### 4.3 The asset file ([`oat/gsaug/gaussian_asset.py`](../oat/gsaug/gaussian_asset.py))

```
means f32[N,3] · quats f32[N,4] wxyz · log_scales[N,3] · opacity_logits[N]
sh_dc[N,3] · sh_rest[N,K,3]  (+ link_id int32[N] for the robot)
conventions{quat_order, sh_layout, scales, opacity, camera_model, sh_frame}
meta{frame: world|body|link, p_capture, q_capture, task, model_xml_sha1,
     versions{gsplat, torch}, train_args, metrics}
sha1  — over the parameter bytes, verified on load
```

`load()` **refuses** an asset whose conventions block disagrees with the running
code, whose sha1 doesn't match, or whose frame isn't what the caller expects
(R4: convention landmines fail loudly). `manifest.json` per task records every
asset's path + sha1 + metrics, the model-XML sha1, pinned library versions, and
the provisional thresholds as data.

---

## 5. Stage 3 — Rendering one augmented frame (`GSCompositeRenderer`)

[`oat/gsaug/compose.py`](../oat/gsaug/compose.py). Per frame, per rewritten
state:

1. **`set_state_from_flattened(state_rw)` + `sim.forward()`** — kinematics only,
   no MuJoCo render (G3). This is also what makes the contact-based validity
   checks downstream see the rewritten state.
2. **Read ground-truth poses** from the forwarded sim: `data.xpos/xquat` for
   every movable object body and every robot link body; `data.cam_xpos/cam_xmat`
   for both cameras.
3. **Pose every component** through the single `PosedComponent.posed()` path
   ([`components.py:186`](../oat/gsaug/components.py#L186)):

   ```
   means_w = p_wb + R(q_wb) · means_l
   quats_w = q_wb ⊗ quats_l                   (wxyz, left-multiply = world frame)
   sh_w    = rotate(sh, R_delta),  R_delta = R(q_wb) · R(q_capture)ᵀ
   scales  = exp(log_scales),  opacities = sigmoid(logits)
   ```

   | Component | frame | `sh_rot_mode` | SH degree |
   |---|---|---|---|
   | background | world | `static` (identity pose enforced; SH never rotated) | 3 |
   | object (per free joint) | body | `so3_deg3` — closed-form z rotation when the delta is pure-z, exact SO(3) band projection otherwise | 3 |
   | robot link | link | `so3_deg1` — exact l=1 vector rule under arbitrary R | 1 |

   **G5 invariant**: there is no code path that moves a component's means/quats
   without routing its SH through `sh_rotation`. This matters because view-
   dependent appearance is stored in a *world-oriented* basis; rotating geometry
   without rotating SH silently mis-shades everything.

   > Historical note worth knowing: objects were originally assumed to differ
   > from their capture pose by a pure z-rotation, with an assertion to enforce
   > it. Measured on real demos, that assertion fires on ~100 % of frames —
   > resting objects settle with |q_xy| ≈ 2–3e-3 tilt, and grasped objects
   > tumble to |q_xy| ≈ 0.8. The assertion was replaced by an exact dependency-
   > free SO(3) SH rotation for l ≤ 3 (`rotate_sh_so3`), so the invariant now
   > holds unconditionally.

4. **Concatenate** background + all posed objects + all posed links into **one**
   Gaussian set (SH bands zero-padded to the max degree present).
5. **One `gsplat.rasterization` call** covering both cameras (G2). Intrinsics
   `K` from the model's `cam_fovy` at the dataset resolution; extrinsics
   world-to-camera in OpenCV convention from `cam_xmat` with the measured F1
   GL→CV flip — so the **eye-in-hand camera's pose comes from the rewritten
   state's own FK, exactly**.

   Image-space compositing of separately rendered components is forbidden: it
   breaks occlusion. Test T4 pins this — a composite of separate renders must
   *fail* the equivalence check that concat-then-rasterize passes.
6. **Orientation + dtype**: apply the measured F2 orientation (measured
   `gsplat_flip_ud = false`), clamp, `×255`, uint8. Output is directly writable
   next to the oracle's `np.flip(obs, axis=0)` frames.

Construction-time gates: facts file must PASS; every asset's sha1 must match the
manifest; on first render the live env's canonicalized model-XML sha1 must equal
the manifest's; every free joint in the live model must have an asset and vice
versa (a missing object errors by name rather than rendering a hole).

---

## 6. How new demos actually get generated

This is the part where GS turns into training data
([`scripts/prerender_se2_aug.py`](../scripts/prerender_se2_aug.py), unchanged
except for the pixel source).

For each task, each episode, each angle `k` in
`θ ∈ {0, ±10, ±20, ±30}°` (K = 7), each frame `t`:

1. `state_rw = rewrite_state(states[t+δ], θ, addr)` — objects' xy rotated about
   `p_base`, their quats left-multiplied by `q_z(θ)`, robot `qpos[joint1] += θ`.
   `δ` (the obs/state off-by-one) is **copied** from the oracle zarr's
   `meta/state_offset`, never recalibrated (G4).
2. Validity checks — joint-1 limit, objects inside the table AABB, support
   contacts, penetration. These read the **state**, never pixels, so they are
   renderer-independent.
3. `gs_renderer.render(env, state_rw)` → both camera images.
4. Write to `images/<cam>/angle_{k:02d}[ep_start + t]`.

A rejected `(episode, angle)` pair is marked invalid and never sampled. Coverage
on LIBERO-10: 2167/3500 pairs valid (~62 %) — the 5 LIVING_ROOM tasks are ~100 %
valid; tasks whose goals anchor to non-rotating fixtures correctly reject most
rotations.

**What "a new demo" means here.** The action sequence is unchanged; the
augmented sample is the same manipulation performed in a world yawed by θ. The
images are new pixels; the labels are the old numbers rotated analytically at
`__getitem__` time (`rotate_action_chunk`, `rotate_proprio`) — never re-rendered,
never re-simulated. One angle is drawn per item and used for *both* halves, so
images and labels are consistent by construction. See
[`se2_data_augmentation.md` §4](se2_data_augmentation.md) for the online path.

### Cross-assertions that keep the comparison clean (G4)

With `--renderer gs`, the run hard-fails unless, for every episode:
`valid_mask`, `p_base`, `angles_deg`, `episode_ends` are **element-wise equal**
to the oracle aug zarr's, and `δ` is copied rather than derived. The θ=0 pixel
gate is re-scoped: the oracle's MAD ≤ 5 becomes a gross-error gate MAD ≤ 25
(warn at 20) — a GS-vs-stored MAD legitimately sits in the 5–20 band because
baked GS appearance is not MuJoCo shading, while a wrong δ / flip / camera lands
far above 25.

### Provenance recorded in the output zarr (G9)

```
meta/render_source ∈ {oracle, gs, gs_hybrid0, gs_oracle_robot}   (+ root.attrs mirror)
root.attrs['gs_manifest_sha1'] = {task_name: manifest_sha1}
```

Resuming a zarr whose recorded manifest sha1 differs from the current assets is
a hard failure — retrained assets can never silently mix with older frames.
(Concurrent GS `--tasks` shards are *not* provenance-safe: zarr root-attr merges
are last-writer-wins. Run GS shards sequentially.)

---

## 7. Gating chain — nothing downstream trusts an assumption

| Gate | Enforced where | Effect |
|---|---|---|
| `gs_render_facts.json` `pass == true` | `cameras.load_render_facts`, called by the renderer, both probes, capture | no GS component is built on unverified conventions (G7) |
| Asset conventions block + sha1 | `GaussianAsset.load` | refuses assets trained under different conventions or hand-edited files |
| Model-XML sha1 | capture manifest → renderer ctor → first render | a renderer built against the wrong task env fails loudly instead of producing subtly wrong pixels |
| Unit tests T1–T6 | `tests/test_gsaug_{transforms,compose,prerender}.py` | rigid-transform consistency, SH rotation invariance (z, SO(3) deg-3, l=1), occlusion (with the image-space composite as a negative control), camera round-trip, uint8 repeatability |
| `probe_gs_geometry.py` (**GATING**) | run after the pre-render | per-component silhouette IoU vs the oracle seg mask; EEF projection error; the **wrist transform-stack check** — within the (movables ∪ robot) mask the GS wrist render at θ must equal the GS wrist render at θ=0, which catches SH-rotation, covariance-quaternion and camera-extrinsics bugs at once |
| `expected_render_source` | `SE2AugZarrDataset` ctor | a `gs*` arm refuses to start unless the zarr self-reports the matching render source **and** `probe_gs_geometry.json` PASSes (the D7 pattern) |
| `probe_gs_photometric.py` | report-only | partitioned PSNR/SSIM over {robot, movables, background} masks, full-frame + movable-crop LPIPS, contact-band PSNR (the shadow-gap tracker) |

Measured caveat baked into the geometry probe's defaults: the EEF projection
metric has a structural floor (oracle ground truth itself sits at median 3.45 px
/ p95 12.3 px), so the plan's original 2/4 px thresholds were recalibrated to
6/16 px. Real camera-chain bugs sit at tens of pixels.

---

## 8. The experimental arms

| Arm | Config | zarr | augment | Question it answers |
|---|---|---|---|---|
| A1 | existing no-aug | oracle | false | baseline (θ=0 oracle re-renders) |
| A2 | `train_flowpolicy_gs_noaug` | gs | false | pure GS render-domain cost, no augmentation |
| A3 | existing aug | oracle | true | augmentation value at a perfect renderer (upper bound) |
| A4 | `train_flowpolicy_gs_aug` | gs | true | **the arm** — GS-rendered augmentation |
| A5 | `train_flowpolicy_gs_hybrid0` | gs_hybrid0 | true | is provenance uniformity worth more than in-domain θ=0 frames? |
| A6 | (deferred) | gs_oracle_robot | true | attributes the A3−A4 gap: robot rendering vs scene rendering |

All arms share the frozen normalization spec, the probe-1 gate, a matched sample
budget, and **vanilla-env rollout evaluation** — so GS-trained arms face a
train→eval visual domain shift by design. That shift is the sim analogue of the
GS→real gap at deployment, and measuring it is the point. Key contrasts the
report prints: `A3−A1` (aug value @ oracle), `A4−A2` (aug value @ GS), `A3−A4`
(rendering cost under augmentation), `A1−A2` (pure domain cost), plus the
correlation of per-task `A3−A4` against the photometric probe's partitioned
metrics.

---

## 9. Runbook

```bash
# prerequisites (unchanged): norm specs → probe 1 PASS → ORACLE prerender exist
python scripts/gsaug/probe_render_facts.py --out data/libero/gs_render_facts.json
MUJOCO_GL=egl python scripts/gsaug/capture_assets.py --task TASK --component all
python scripts/gsaug/train_gs_assets.py --task TASK --component all
pytest tests/test_gsaug_*.py

MUJOCO_GL=egl python scripts/prerender_se2_aug.py --renderer gs \
    --gs-assets-dir data/libero/gs_assets \
    --oracle-zarr  data/libero/libero10_N500_se2aug.zarr \
    --out          data/libero/libero10_N500_se2aug_gs.zarr

MUJOCO_GL=egl python scripts/probes/probe_gs_geometry.py    # GATING
MUJOCO_GL=egl python scripts/probes/probe_gs_photometric.py # report-only
# train A2/A4 (+A5), then:
python scripts/gsaug/report_factorial.py
```

Sequencing rule: run the whole chain on **one** task first (pick a LIVING_ROOM
task — ~100 % valid rate), including a short A4 smoke train, before fanning out.
Thresholds recalibrated from that dry run live in the facts file / manifests, not
in code comments.

Budget (dual RTX 4090, GS stages fit on one): ~50–70 asset trainings ≈ 3–5 GPU-h
plus 1–2 h capture; the GS pre-render is ~1.4 M images at > 150 FPS composited ≈
3–4 h. gsplat measured 1328 FPS at 128² on a 300 k-Gaussian scene, so kinematics
overhead dominates, not rasterization.

---

## 10. Known limitations (what GS renders *cannot* do here)

| # | Limitation | Status |
|---|---|---|
| R1 | **No light transport.** Composed movables cast no contact shadows, and baked shading does not respond to rotation. | The expected headline artifact. Tracked by the photometric probe's contact-band PSNR; shadow proxies / relightable GS are Phase-1 material. |
| R2 | **Headlight view-dependence.** SH absorbs a camera headlight only approximately. | Measured at the θ=0 report, listed as a known limitation; no protocol change. |
| R3 | **Robot articulation quality** — joint-boundary tearing, thin fingers. | Mitigated by labeled per-link init + multi-config FK finetune + gripper-range assertion; arm A6 exists to isolate its impact if needed. |
| — | **Fixtures don't rotate.** Stove, cabinet, microwave and the robot base live in `model.body_pos/body_quat`, not qpos, so the yaw leaves them in place. | Pre-existing property of the SE(2) augmentation, not of GS; the validity checks reject the rewrites this breaks. |
| — | **Assets are per task** (including the robot, G10), because baked appearance must match the scene it is composited into. | Deliberate; capture/training cost is small enough that sharing is not worth the appearance mismatch. |

---

## 11. File map

| File | Role |
|---|---|
| [`scripts/gsaug/probe_render_facts.py`](../scripts/gsaug/probe_render_facts.py) | M1: measures renderer facts F1–F7 → `gs_render_facts.json` (gating) |
| [`scripts/gsaug/capture_assets.py`](../scripts/gsaug/capture_assets.py) | M2: per-task background/object/robot orbit capture (RGB + depth + seg + poses) |
| [`scripts/gsaug/train_gs_assets.py`](../scripts/gsaug/train_gs_assets.py) | M3–M4: training driver + per-task `manifest.json` (G9) |
| [`oat/gsaug/capture.py`](../oat/gsaug/capture.py) | orbit geometry, raw-renderer backend, hide/graveyard mechanisms, pose validation, `CaptureBundle` |
| [`oat/gsaug/cameras.py`](../oat/gsaug/cameras.py) | `fovy → K`, MuJoCo cam → OpenCV w2c, facts loading |
| [`oat/gsaug/trainer.py`](../oat/gsaug/trainer.py) | static fit: depth init, masked L1+SSIM + silhouette, densify, body-frame conversion |
| [`oat/gsaug/articulated.py`](../oat/gsaug/articulated.py) | robot asset: labeled per-link init + FK finetune |
| [`oat/gsaug/gaussian_asset.py`](../oat/gsaug/gaussian_asset.py) | asset container, conventions block, sha1, IO |
| [`oat/gsaug/components.py`](../oat/gsaug/components.py) | `PosedComponent` — the single posing path for objects, links, background |
| [`oat/gsaug/sh_rotation.py`](../oat/gsaug/sh_rotation.py) | closed-form z rotation, exact l=1 SO(3), exact SO(3) for l ≤ 3, e3nn cross-check scaffold |
| [`oat/gsaug/compose.py`](../oat/gsaug/compose.py) | `GSCompositeRenderer` — the render-time pixel source |
| [`scripts/prerender_se2_aug.py`](../scripts/prerender_se2_aug.py) | pre-render driver; `--renderer {oracle,gs}`, G4 cross-asserts, provenance meta, A5 assembly |
| [`scripts/probes/probe_gs_geometry.py`](../scripts/probes/probe_gs_geometry.py) | GATING: silhouette IoU, EEF projection, wrist transform-stack check |
| [`scripts/probes/probe_gs_photometric.py`](../scripts/probes/probe_gs_photometric.py) | report-only partitioned photometric metrics |
| [`oat/dataset/se2_aug_zarr_dataset.py`](../oat/dataset/se2_aug_zarr_dataset.py) | online dataset; `expected_render_source` gate |
| [`scripts/gsaug/report_factorial.py`](../scripts/gsaug/report_factorial.py) | M8: factorial table + metric→success-rate correlation |
| `oat/config/train_flowpolicy_gs_{noaug,aug,hybrid0}.yaml` | arms A2/A4/A5 |
| [`tests/`](../tests/) | `test_gsaug_transforms.py`, `test_gsaug_compose.py` (T1–T6), `test_gsaug_prerender.py` |

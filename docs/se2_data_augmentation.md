# SE(2) Data Augmentation — Detailed Procedure

How the world-frame SE(2) (yaw) augmentation for LIBERO-10 actually works, end to
end: where every image and label comes from, which code path produces it, and
which guard catches each way it could silently go wrong. Training arms, EquiNoise,
and launch commands are covered separately in
[`se2_aug_equinoise_summary.md`](se2_aug_equinoise_summary.md).

**The one-paragraph mental model.** A training sample is made equivariant under a
world-frame yaw `R_z(θ)` about the robot base. The **pixels** for a rotated sample
come from an **offline simulator re-render**: the demo's stored MuJoCo states are
rewritten (objects + arm rotated exactly, no dynamics) and both cameras are
re-rendered once per angle into an augmentation zarr. The **labels** (action chunk
+ proprio) are **never re-rendered** — the dataloader rotates the raw numbers
analytically at `__getitem__` time. One angle `k` is drawn per sample and both
halves use the same `θ = angles_deg[k]`, so the images and labels of every item
are mutually consistent by construction. Raw-space label rotation is exact only
because normalization is *group-compatible* (tied scale, zero offset on rotating
blocks) and frozen in a spec that every arm loads but never refits.

```
offline (once)                                 online (every __getitem__)
──────────────                                 ───────────────────────────
LIBERO HDF5 states                             base zarr numerics (RAM)
      │  sha1-match episodes,                        │
      │  calibrate obs/state offset δ                │  draw k ~ U(valid angles)
      ▼                                              ▼
rewrite_state(states[t+δ], θ_k)                rotate_action_chunk(act, θ_k)
      │  objects: xy about p_base,             rotate_proprio(obs, θ_k, p_base)
      │  quat ⊗; robot: joint1 += θ                  │
      ▼                                              │
re-render both cameras ──► aug zarr ────────► fetch images/<cam>/angle_k ──► item
   (validity checks,       images/…/angle_00..NN     (same k for images
    valid_mask)            meta/{valid_mask,p_base,…} and labels)
```

---

## 1. The group action

- **Group**: planar rotations `R_z(θ)` about the robot-base vertical axis, applied
  in the **world frame**. (No translation component is used — "SE(2)" in this
  codebase means the yaw subgroup acting about `p_base`.)
- **Angle grid**: discrete, fixed at pre-render time. Default
  `θ ∈ {0, +10, −10, +20, −20, +30, −30}` degrees (`--angles`, K=7). Angle index
  0 is **always** `θ=0`; `parse_angles` reorders the grid if needed and refuses a
  grid without 0.
- **Rotation center `p_base`**: resolved per episode by
  `resolve_addresses` ([oat/env/libero/se2_state_rewrite.py](../oat/env/libero/se2_state_rewrite.py)) —
  preferably the joint-1 axis anchor `sim.data.xanchor[joint1]`, asserted to agree
  with the `robot0_base` body xy within 1e-4 (falls back to the body position if
  `xanchor` is unavailable). Stored in the aug zarr as `meta/p_base` and reused by
  the dataloader as the center for analytic proprio rotation.
- **What transforms how** (for a yaw by θ):

  | Quantity | Transform | Where |
  |---|---|---|
  | camera images | simulator re-render of the rewritten state | offline, `scripts/prerender_se2_aug.py` |
  | action `(dx,dy)` | `R(θ)` rotation | online, `rotate_action_chunk` |
  | action `(rx,ry)` | `R(θ)` rotation **iff** world-frame controller (probe-gated) | online, `rotate_action_chunk` |
  | action `dz`, `rz`, `grip` | invariant | — |
  | `robot0_eef_pos` | xy rotated about `p_base` | online, `rotate_proprio` |
  | `robot0_eef_quat` (xyzw) | left-multiplied by `q_{R_z(θ)}` | online, `rotate_proprio` |
  | `robot0_joint_pos` | joint 1 `+= θ` | online, `rotate_proprio` |
  | `robot0_gripper_qpos`, `task_uid` | invariant | — |

The action-space justification for `(rx,ry)`: an axis-angle delta transforms by
conjugation, `R_z e^{r̂} R_z^⊤ = e^{(R_z r)^}`, so under a world yaw the delta's
xy components rotate and `rz` is invariant — but only if the OSC_POSE controller
interprets the delta in the **world** frame. That hypothesis is not assumed; it is
measured by probe 1 and hard-gated (§6).

---

## 2. Provenance: why alignment work is needed before any rendering

The base zarr (`data/libero/libero10_N500.zarr`) was built in two steps: per-task
zarrs converted from the LIBERO HDF5 demos by `scripts/convert_libero_dataset.py`
→ `oat/env/libero/dataset_conversion.py`, then composed into the 10-task zarr by
`scripts/compose_libero_multitask_dataset.py` → `scripts/merge_data.py --shuffle`.
Two conversion facts drive the whole alignment machinery in
[`oat/env/libero/demo_alignment.py`](../oat/env/libero/demo_alignment.py):

1. **Episode order is meaningless.** The converter sampled demos with an
   *unseeded* `np.random.choice`, and the merge step interleaved tasks with a
   second unseeded shuffle — so zarr episode `i` says nothing about which
   HDF5 demo it came from. (Episodes are copied verbatim through the merge, so
   action bytes stay exact.) The pre-render needs the HDF5 `demo['states']`
   (flattened MuJoCo states — the zarr does not store them), so `match_episodes`
   re-identifies every episode **by content**: SHA-1 over the little-endian
   float32 bytes of the `(T,7)` action array. This works because the zarr's
   `action` is an exact float32 cast of `demo['actions']`. Matching is exact
   (no tolerance) and all-or-nothing: a single unmatched or duplicated episode
   raises `ValueError`. The task name is then taken from the matched HDF5
   *filename* (authoritative; a disagreeing zarr `task_uid` only warns).

2. **Obs and states are off by one.** HDF5 stores *pre-action* states while the
   converted obs are *post-action*, so stored `obs[t]` corresponds to
   `states[t+δ]` with `δ ∈ {0,1}` (expected: 1). `calibrate_state_offset` finds δ
   empirically per task: re-render `states[t+δ]` for both δ candidates at ≤8
   probe frames and pick the δ with the smaller mean-abs pixel difference against
   the stored agentview frames. The pre-render calibrates on the first 3 episodes
   of each task and **hard-fails if they disagree**; the chosen δ is applied as
   `states[min(t+δ, len(states)−1)]` everywhere and recorded per episode in
   `meta/state_offset`.

One more convention: the converter stored camera frames **vertically flipped**
relative to raw robosuite output, so every re-rendered frame is flipped
(`np.flip(axis=0)`) before being compared or written — calibration, the θ=0 pixel
gate, and the aug zarr all live in the flipped (dataset) orientation.

---

## 3. Offline stage: `scripts/prerender_se2_aug.py`

For every (episode, angle) pair: rewrite each frame's stored state, re-render
both cameras through the **same** LIBERO `ControlEnv` pipeline that produced the
base zarr, run validity checks, and write images + metadata to a separate
augmentation zarr. Numerics are deliberately **not** stored — only images need
the simulator.

### 3.1 State rewrite (`oat/env/libero/se2_state_rewrite.py`)

Input is a robosuite flattened state `[time(1), qpos(nq), qvel(nv)]`
(`MjSimState.flatten()`; note every qpos address gets a +1 offset for the time
slot). `rewrite_state(state, θ, addr)` is pure numpy:

- **Movable objects** = every non-robot free joint (name not prefixed
  `robot`/`gripper`/`mount`, 7-dof qpos span). For each: position xy
  `← R(θ)(xy − p_base) + p_base` (z untouched); orientation quaternion (MuJoCo
  **wxyz**) `← q_z(θ) ⊗ quat` (left multiply = world-frame rotation).
- **Robot**: `qpos[joint1] += θ`, nothing else. Joint 1's axis *is* the base z
  axis, so this rigidly co-rotates the entire arm exactly — no IK, no re-solve.
- **qvel is untouched**: rewritten states are render-only inputs; they are never
  used to continue dynamics.
- **What does not rotate**: fixtures (stove, cabinet, microwave, …) and the robot
  base live in `model.body_pos/body_quat`, not qpos, and stay put. This is an
  accepted residual — the contact checks below exist precisely to reject rewrites
  it breaks.

Rendering goes through `env.regenerate_obs_from_state(state_rw)`, which is
`set_state_from_flattened → sim.forward() → observables` — a kinematic
re-pose + render, **no physics stepping**, hence exact and drift-free.

### 3.2 Validity checks — when a rotation is rejected

Checked per frame; the first failure rejects the whole (episode, angle) pair and
rendering stops there (frames already written for that pair stay on disk —
consumers must gate on `meta/valid_mask`, never on image content).

| Check | Condition | Applies at | Needs sim? |
|---|---|---|---|
| `check_joint1_limit` | `abs(q1) ≤ 2.8973 − 0.05` rad (Panda limit − `--joint_margin`) | every angle, incl. θ=0 | no |
| `check_objects_in_bounds` | every object center xy inside the **physical table-top AABB** inset by `--xy_margin` (0.05 m) | **θ≠0 only** | no (bounds precomputed) |
| `check_support_contacts` | every object has ≥1 contact with normal within 30° of ±z, or touches a gripper geom (held) | θ≠0, **rotation-conditional** | yes (after `set_state` + `forward`) |
| `check_object_penetration` | no object contact deeper than 5 mm | θ≠0, **rotation-conditional** | yes |

Details that matter:

- The table bounds come from `table_top_xy_aabb`: the union xy-AABB of collision
  "table" box geoms whose top surface sits near the objects' resting height,
  computed once per task on a freshly reset env. Placement is judged against
  physical table geometry, **not** against the empirical data-occupancy box —
  the empirical box (recorded in the report for reference) would reject exactly
  the rotated placements the augmentation exists to create.
- The two contact checks use a **θ=0 exemption**: during the θ=0 pass, per-frame
  failures are recorded but never reject; at θ≠0 a failure rejects **only if the
  same frame passed at θ=0**. A frame that already fails un-rotated (object
  mid-drop, squeezed in the gripper) is a demo artifact, not a rotation problem.
- Reachability needs no check: the arm co-rotates exactly, so a reachable pose
  stays reachable.
- `--no-support-check` disables *both* contact checks (penetration is nested
  under the support-check flag).

### 3.3 Two passes per task, and the θ=0 self-checks

**Pass 1 — θ=0 for every episode.** Angle 0 is re-rendered too (not copied from
the base zarr) so that aug-on and aug-off arms share render provenance (decision
D1). This pass also records the reference EEF positions at ≤5 sampled frames per
episode (used by pass 2's consistency check) and the support/penetration
baseline fail-sets over **every** frame (for the θ=0 exemption). Self-checks:

- A rejected θ=0 pair is a **hard failure** of the whole run — `valid_mask[:,0]`
  must end up all-True (also re-asserted as a final invariant and by the dataset
  at load time).
- Per-task pixel gate: mean abs difference between the θ=0 re-render and the
  stored base-zarr agentview frames must be ≤ 5.0 (uint8 units; > 1.0 warns).
  This catches a wrong δ, camera, resolution, or flip. The wrist camera is
  report-only (it rides the arm; sub-mm settling makes small MADs normal).

**Pass 2 — all θ≠0 angles.** For each valid pair, every frame is rewritten,
checked, rendered, flipped, and written to `images/<cam>/angle_{k:02d}` at its
global frame index. Inline consistency check at the sampled frames: observed EEF
position vs `rotate_xy(eef_ref_θ0, θ, center=p_base)` — > 1 mm prints a WARN but
never rejects (probe 2 owns the hard gate on this property).

### 3.4 Output layout, resumability, sharding

```
<out>.zarr
├── images/{agentview_rgb, robot0_eye_in_hand_rgb}/angle_00..NN
│      uint8 (n_steps, H, W, 3), chunks (1,H,W,3), Blosc zstd-3
└── meta/
    ├── angles_deg     (K,)   float64   angle 0 first, always 0.0
    ├── episode_ends   (E,)   int64     copied verbatim from the base zarr
    ├── valid_mask     (E,K)  bool      pair passed all checks AND fully rendered
    ├── done_mask      (E,K)  bool      pair processed (done ≠ valid!)
    ├── p_base         (E,3)  float32   per-episode rotation center
    └── state_offset   (E,)   int8      calibrated δ (−1 = not yet calibrated)
```

- Frame indices in `images/*` line up 1:1 with the base zarr's `data/*` arrays —
  the aug zarr stores **no** numerics and shares the base's episode geometry.
- **Resume** (`--resume`, default): pairs with `done_mask=True` are skipped; the
  θ=0 EEF/baseline references are rebuilt without rendering. Resuming asserts the
  existing store has identical `episode_ends` and `angles_deg`; changing either
  requires `--no-resume`, which deletes the output tree.
- **Sharding** (`--tasks`): the task is the shard unit. Per-frame image chunks and
  per-episode meta chunks make different tasks' writes disjoint — but the store
  must be created by one run before shards start.
- A JSON report (`<out>.report.json`, or `.report.<shard>.json`) records CLI
  provenance, per-angle valid rates, rejection reasons, pixel-diff and
  render-consistency percentiles, and the episode→demo mapping.

Coverage on LIBERO-10 (from the recorded run; see the summary doc): 2167/3500
pairs valid. The 5 LIVING_ROOM tasks are ~100% valid at every angle; tasks whose
goals anchor to non-rotating fixtures correctly reject most θ≠0 renders.

---

## 4. Online stage: `SE2AugZarrDataset` (`oat/dataset/se2_aug_zarr_dataset.py`)

Mirrors `ZarrDataset` (sampler geometry, val split, output schema) but serves
per-item SE(2)-rotated triples.

### 4.1 Data placement

- **Numerics in RAM**: `action` + low-dim obs are copied from the **base** zarr
  via `ReplayBuffer`, exactly like the un-augmented dataset.
- **Images lazy from disk**: rgb keys (`*_rgb`) are read per item from the aug
  zarr's angle arrays (or the base zarr), with zarr handles cached per process
  (pid-keyed, fork/spawn safe).

### 4.2 Per-item assembly (`__getitem__`)

1. Sample a window from the base numerics (`SequenceSampler`, `To=n_obs_steps`
   obs frames, `Ta=n_action_steps` action steps, edge-padded at episode
   boundaries).
2. Draw **exactly one** angle index `k`, uniform over that episode's valid angles
   (`meta/valid_mask` row); `k=0` whenever `augment=False`.
3. `θ = radians(angles_deg[k])`. If `k≠0`, rotate the **raw** labels:
   `rotate_action_chunk(act, θ, rotate_rot_labels)` and
   `rotate_proprio(obs, θ, p_base[episode,:2])`
   ([oat/equi/se2_transforms.py](../oat/equi/se2_transforms.py) — float64 math,
   float32 out; `robot0_eef_quat` is **xyzw** here, unlike MuJoCo's wxyz qpos).
4. Fetch the `To` obs frames from `images/<key>/angle_{k:02d}` and attach.

Raw-space rotation before normalization is decision **D2**: it commutes exactly
with the frozen group-compatible normalizer inside the policy (§5), so the
dataset never touches normalized quantities.

### 4.3 Matched budget and comparability

- `__len__` counts sampler windows over the **base** numerics, so it is identical
  with `augment` on or off — aug and no-aug arms see the same number of samples
  per epoch by construction (printed at ctor).
- `get_validation_dataset()` forces `augment=False`: validation loss is computed
  at θ=0 in every arm, so val curves are comparable across arms.
- The angle RNG is per-dataloader-worker, lazily seeded from
  `get_worker_info().seed`; with `persistent_workers=True` it is *not* re-seeded
  each epoch (draws stay i.i.d. uniform either way).

### 4.4 Fail-fast guards at construction

The ctor refuses to produce silently-wrong data:

| Guard | Error |
|---|---|
| no `norm_spec_path` | `ValueError` — stats are frozen across arms (D3), build once with `scripts/build_normalization_spec.py` |
| spec `mode`/`world_frame_rotation` ≠ dataset args | `AssertionError` |
| `augment=True, image_source='base'` (without the naive control arm) | `ValueError` — would pair rotated labels with un-rotated images |
| rotating rotation labels without a PASSing probe 1 | `RuntimeError` — needs `probe_results.json` with `controller_frame_rot=='world'` **and** `pass==true` (D7) |
| aug zarr `episode_ends` ≠ base's, `angles_deg[0]≠0`, `valid_mask[:,0]` not all-True, or wrong image frame counts | `AssertionError` |

`get_normalizer()` additionally verifies the spec's data fingerprint
(`n_steps`, `n_episodes`, `action_sha1`) against the live buffer and fails if the
spec lacks any needed key — it **loads** the frozen spec, never fits.

### 4.5 Control arm and Phase-2 scaffold

- `naive_image_rotation=True` (+ `augment=True`): the deliberately-wrong baseline
  — in-plane `scipy.ndimage.rotate` of the **base** agentview pixels (no
  re-render), wrist camera served un-rotated. Exists to demonstrate why the real
  re-render is necessary. No config enables it; it is a CLI-override arm.
- `emit_angle_pair=True` (+ `augment=True`): draws a second independent angle `k2` and returns a
  full second item under `item['pair']` (plus `theta` floats) for a future
  paired-consistency loss; batch with `paired_angle_collate`
  (`oat/dataset/se2_collate.py`). Not used by any training config.

---

## 5. Why raw-space label rotation is exact: group-compatible normalization

`oat/equi/normalization.py` + `oat/equi/blocks.py`; frozen spec built once by
`scripts/build_normalization_spec.py` (decision D3).

Per-dim min-max normalization would warp the planar blocks: with different
scales on x and y (plus offsets), the induced group action in normalized space is
`D R D⁻¹ x + (o − D R D⁻¹ o)` — neither orthogonal nor offset-free, so rotating
raw labels and normalizing would *not* equal normalizing then rotating. Instead,
each block gets a representation-aware rule:

| Block | Rep | Rule |
|---|---|---|
| action `(dx,dy)` | ρ₁ | one shared symmetric scale `1/max‖·‖∞`, offset 0 (an all-zero block falls back to scale 1) |
| action `dz`, `grip` | ρ₀ | per-dim limits affine (baseline formula) |
| action `(rx,ry,rz)` | free-iso (default hedge) | one tied scale over all 3 dims, offset 0 |
| `robot0_eef_pos` xy / z | ρ₁ / ρ₀ | shared xy scale / per-dim z |
| `robot0_eef_quat` | identity | scale 1, offset 0 (unit quaternion) |
| rgb keys | — | baseline 0..255 limits, copied through |

On a ρ₁/free-iso block the normalizer is `N(x) = (1/s)·x` — a scalar multiple of
the identity with zero offset — so `N(Rx) = RN(x)` exactly, for any orthogonal
`R`. Consequences: (i) rotating **raw** labels in the dataloader commutes with
the policy-side normalize step; (ii) the stats are G-invariant, so **one frozen
spec serves every arm**; (iii) plain per-block `randn` source noise is already
correctly distributed in normalized space.

- `world_frame_rotation=False` (default) keeps `(rx,ry,rz)` as one isotropic
  block — safe under *both* controller-frame hypotheses. `True` (a post-probe
  ablation, `'_wfr'` spec files) splits it into a rotating `(rx,ry)` ρ₁ pair +
  invariant `rz`. The dataset asserts spec and dataset agree on this flag.
- The spec JSON stores per-key scale/offset/input-stats plus a **fingerprint**
  (`n_steps`, `n_episodes`, SHA-1 of the float32 action bytes; the zarr *path* is
  deliberately not compared, so moving the zarr is fine but changing its data is
  not). Policies re-assert their live normalizer against the same spec
  (`assert_spec_matches`), giving a two-sided stats-frozen guard.
- Known approximation: `robot0_eef_pos` rotates about `p_base ≠ 0`, so its
  *normalized-space* group action carries a constant offset. Labels remain exact
  because the dataset rotates the raw values about `p_base` itself.

---

## 6. The probe suite (`scripts/probes/`)

Three measured facts replace three assumptions. All write JSON with a top-level
`pass` and exit nonzero on failure.

**Probe 1 — controller frame (`probe_controller_frame.py`, GATING).**
*Question:* does OSC_POSE interpret `[dx..rz]` deltas in the world or EE frame?
*Method:* rigidly rotate scene+arm by θ (same `rewrite_state`), then command a
pure `+x` position delta and a pure `+rx` rotation delta; check whether the
measured EEF displacement / delta-rotation axis stays along world `+x`
(world-frame hypothesis) or tracks `R(θ)·x̂` (EE-frame) across θ ∈
{0,±30,±60}°, threshold dot > 0.95. `pass` requires the position and rotation
verdicts to be conclusive **and** agree. θ=90° is measured but excluded from the
verdict (workspace-boundary artifact). *Consumer:* `SE2AugZarrDataset` hard-gates
rotation-label rotation on `controller_frame_rot=='world' and pass==true` —
if the controller were EE-frame, rotated `(rx,ry)` labels would be silently
double-rotated. Result on LIBERO-10: both **world**, PASS.

**Probe 2 — render consistency (`probe_render_consistency.py`).**
*Question:* is `rewrite_state` + re-render *exactly* the rotation it claims?
*Method:* 200 random (episode, frame, angle) triples; compare the re-rendered
EEF pose against `rotate_xy(p₀)` / `q_z(θ) ⊗ q₀`. PASS iff **100%** of triples
are within 1e-4 m (per-component) and 1e-3 rad. This is the hard gate behind the
pre-render's WARN-only inline check; δ calibration and θ=0 pixel-match stats are
also reported (informational — they do not enter the pass verdict).
Result: 200/200, max error ~7e-16 m (exact FK).

**Probe 3 — label round-trip (`probe_label_roundtrip.py`).**
*Question:* do the *rotated action labels* actually drive the *rotated scene*
along the rotated trajectory (end-to-end consistency of state rewrite + label
transform under real controller dynamics)? *Method:* replay
`rotate_action_chunk(actions, θ)` from `rewrite_state(state₀, θ)` and compare
the EEF trajectory to `rotate_xy(reference, θ)`, with the un-rotated replay's own
drift calibrating tolerances; the OSC nullspace target is re-anchored after every
state injection (otherwise the controller drags joint 1 back toward θ=0 — a
controller artifact, not a label error). PASS criteria (success-rate ratio ≥ 0.9,
median error < 2 cm, p90 ≤ 2× control drift) are evaluated on the **corotating**
task stratum only; fixture-anchored tasks structurally cannot succeed when
rotated (fixtures don't rotate) and are reported as expected-fail. Result: PASS
(median rotated-trajectory error ~6.4 nm).

---

## 7. Configuration wiring

Hydra groups `aug.*` and `norm.*` (root config `train_equi_flowpolicy.yaml`) are
passed through `task/policy/libero/libero10_se2aug.yaml` into the dataset:

```yaml
aug:
  enable: true                                  # -> dataset.augment
  zarr_path: data/libero/libero10_N500_se2aug.zarr
  image_source: aug                             # 'aug' | 'base'
  controller_frame: world
  rotate_rotation_labels: true
  probe_results_path: data/libero/probe_results.json
  naive_image_rotation: false
norm:
  mode: group_compatible
  spec_path: data/libero/norm_spec_group_compatible_libero10_N500.json
  world_frame_rotation: false
```

Notes:

- The dataset's `n_action_steps` is bound to `${horizon}` (16), not the policy's
  top-level `n_action_steps` (8): the dataset serves the full 16-step raw chunk;
  the policy consumes 8 at inference.
- No-aug arms keep `image_source: aug` (decision D1): they read the θ=0
  *re-renders*, so aug-vs-no-aug compares learning signal, not render provenance.
  `aug.image_source=base` is a smoke-run escape hatch for before the aug zarr
  exists.
- Prerequisite order (fail-fast if violated): build the normalization specs →
  run probe 1 to PASS → pre-render the aug zarr → train. Commands are in the
  [summary doc](se2_aug_equinoise_summary.md#prerequisites-checklist-run-once-in-order).

### Decision tags referenced in code comments

| Tag | Meaning |
|---|---|
| D1 | both arms read angle-0 re-renders → shared render provenance |
| D2 | labels rotate in RAW space, before (frozen) normalization |
| D3 | normalization stats are built once and only ever loaded, never refit |
| D7 | rotation-label rotation is gated on probe 1 (`controller_frame_rot=='world'` and `pass`) |

---

## 8. Invariants and gotchas worth remembering

- **One θ per item.** Images and labels must come from the same angle; the ctor
  guard against `augment=True, image_source='base'` exists to make the
  inconsistent pairing unrepresentable.
- **`done_mask` ≠ `valid_mask`.** Rejected pairs are done-but-invalid and leave
  *partial image rows* on disk; always gate on `valid_mask`. The dataset never
  reads `done_mask` (it's a resume bookkeeping array).
- **Angle 0 is load-bearing.** `angles_deg[0]==0` and `valid_mask[:,0]` all-True
  are asserted by both the pre-render and the dataset; the `augment=False` and
  validation paths hard-code `k=0` as the identity.
- **Two quaternion conventions coexist.** MuJoCo qpos quats are **wxyz**
  (state rewrite); the dataset's `robot0_eef_quat` is **xyzw** (proprio
  rotation). Both are *left*-multiplied by the yaw quaternion (world-frame);
  right-multiplying would be a body-frame rotation.
- **+1 time offset.** Flattened states are `[time, qpos, qvel]`; every qpos
  address is shifted by one when indexing a flattened state.
- **Rewritten states are render-only.** qvel is stale; never roll dynamics from
  them (probe 3 does so deliberately — that is exactly what it tests).
- **δ clamps at the end.** `states[min(t+δ, len−1)]`: with δ=1 the last obs frame
  reuses the final state.
- **Everything is fail-fast.** Wrong spec mode, missing or failed probe,
  mismatched aug zarr, inconsistent δ, invalid θ=0 — each raises at
  construction/pre-render time (a tampered spec fingerprint raises slightly
  later, in `get_normalizer()`); a misconfigured arm cannot silently train on
  wrong data.

## 9. File map

| File | Role |
|---|---|
| [`scripts/prerender_se2_aug.py`](../scripts/prerender_se2_aug.py) | offline pre-render driver (two passes, checks, resume/shard, report) |
| [`oat/env/libero/se2_state_rewrite.py`](../oat/env/libero/se2_state_rewrite.py) | state rewrite + address resolution + table AABB + 4 validity checks |
| [`oat/env/libero/demo_alignment.py`](../oat/env/libero/demo_alignment.py) | sha1 episode↔demo matching + obs/state δ calibration |
| [`oat/equi/se2_transforms.py`](../oat/equi/se2_transforms.py) | pure-numpy rotations: actions, proprio, quats (wxyz & xyzw) |
| [`oat/dataset/se2_aug_zarr_dataset.py`](../oat/dataset/se2_aug_zarr_dataset.py) | online dataset: angle draw, label rotation, image fetch, guards |
| [`oat/dataset/se2_collate.py`](../oat/dataset/se2_collate.py) | Phase-2 paired-angle collate (scaffold) |
| [`oat/equi/normalization.py`](../oat/equi/normalization.py), [`oat/equi/blocks.py`](../oat/equi/blocks.py) | group-compatible normalizer, frozen spec, fingerprint |
| [`scripts/build_normalization_spec.py`](../scripts/build_normalization_spec.py) | one-shot builder of both frozen specs |
| [`scripts/probes/`](../scripts/probes/) | probe 1 (controller frame, gating), probe 2 (render consistency), probe 3 (label round-trip) |
| [`tests/test_se2_aug_dataset.py`](../tests/test_se2_aug_dataset.py) | pins matched budget, valid-mask sampling, bit-exact label rotation, probe gating, frozen-spec behavior |

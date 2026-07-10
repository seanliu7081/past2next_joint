# SE(2) Data Augmentation + EquiNoise — Phase 1 Summary

Flow-matching policy for LIBERO-10 with world-frame SE(2) augmentation and a
block-isotropic ("EquiNoise") noise source. New-files-only implementation on top
of the existing `FlowPolicy`; no existing file was modified.

Interpreter for every command below: `/home/haotian/miniforge3/envs/oat/bin/python`
(conda env `oat`; robosuite 1.4.0, mujoco 3.2.6, LIBERO editable in `third_party/`).
Run everything from the repo root `/home/haotian/code/oat`.

---

## 1. Policy overview

**`EquiFlowPolicy`** (`oat/policy/equi_flow_policy.py`) subclasses **`FlowPolicy`**
(`oat/policy/flow_policy.py`). It is a **rectified flow-matching** policy conditioned
only on observation tokens; the single change vs the baseline is *where the source
noise `x0` comes from* (see §2).

| | |
|---|---|
| Task suite | LIBERO-10 (10 tasks × 50 demos = 500 episodes, `libero10_N500.zarr`) |
| Action | 7-dim OSC_POSE delta `[dx, dy, dz, rx, ry, rz, grip]` (axis-angle rot delta, world frame) |
| Observation | `agentview_rgb` (128²), `robot0_eye_in_hand_rgb` (128²), `robot0_eef_pos` (3), `robot0_eef_quat` (4, xyzw), `robot0_gripper_qpos` (2), `task_uid` (1) |
| Horizon / act steps / obs steps | `horizon=16`, `n_action_steps=8`, `n_obs_steps=2` |
| Backbone | `TransformerForDiffusion`, `embed_dim=256`, `n_layers=4`, `n_heads=4`, `causal_attn=False` (whole chunk denoised jointly) |
| Flow steps | Euler integration, `num_inference_steps=10`, `prior_noise_scale=1.0` |
| Obs encoder | `FusedObservationEncoder` (robomimic RGB crop 76², projection state encoder) |

**Training objective (rectified flow), unchanged from the baseline:**

```
x1 = normalize(action chunk)              # (B, H, 7)
x0 = prior_noise_scale * source()         # structured source (§2); randn when disabled
t  ~ U(0,1)
xt = (1-t)*x0 + t*x1
loss = MSE( vθ(xt, t, obs),  x1 - x0 )
```

`x0` feeds **both** the interpolation `xt` and the regression target `v = x1 - x0`,
so training and inference must draw from the *same* source — `EquiFlowPolicy` vendors
`forward`/`predict_action` verbatim from `FlowPolicy`, swapping only the two
`torch.randn` sites for `self._source(...)`.

**Default-off contract:** with `source.enable: false` (or absent) the source is exactly
`torch.randn` and the policy is bit-for-bit identical to `FlowPolicy` (verified in tests).

**Two extra knobs** on `EquiFlowPolicy`:
- `freeze_obs_encoder: bool` — sets `requires_grad_(False)` on the obs encoder and keeps
  it in `eval()` (the parent's grouped optimizer already skips frozen params).
- `norm_spec_path` — arms the **stats-frozen guard**: `set_normalizer` asserts the
  incoming normalizer equals the persisted spec, so every experimental arm provably
  trains under identical normalization statistics.

---

## 2. Data augmentation + noise initialization

Both pieces exist to satisfy **P1 (source invariance)** and make **P2 (velocity
equivariance)** the optimum of the flow-matching loss under the world-frame yaw group
`R_z(θ)` about the robot base.

### 2a. Group-compatible normalization (Workstream N — the shared foundation)

`oat/equi/normalization.py` + `oat/equi/blocks.py`.

Per-dim min-max normalization *warps* the planar action blocks, so the group action in
normalized space would not be an orthogonal rotation. Instead each block gets a
representation-aware rule so that in normalized space the group action is an **exact
orthogonal `R(θ)` with zero offset**:

| Block (`BlockSpec`) | Dims | Rep | Rule |
|---|---|---|---|
| `xy` | 0,1 | ρ₁ (rotates) | one shared symmetric scale `1/max‖·‖∞`, offset 0 |
| `z` | 2 | ρ₀ (invariant) | per-dim limits affine (baseline formula) |
| `rot` | 3,4,5 | free-iso (hedge) | one shared scale, offset 0 |
| `grip` | 6 | ρ₀ | per-dim limits affine |
| `robot0_eef_pos` | xy / z | ρ₁ / ρ₀ | shared xy scale / per-dim z |
| `robot0_eef_quat` | 0–3 | identity | scale 1, offset 0 (unit quaternion) |

Consequences: (i) plain per-block `randn` is exactly P1-correct; (ii) rotating **raw**
action/proprio labels commutes exactly with normalization, so the dataset rotates raw
values and leaves the policy's normalize step untouched; (iii) stats are G-invariant, so
**one frozen spec serves every arm**. Default `world_frame_rotation=False` keeps the
axis-angle rot delta as a single isotropic 3-block (the "rot hedge", P1-safe under both
controller-frame hypotheses).

Stats are computed **once** and only ever loaded afterward (never refit):

```bash
python scripts/build_normalization_spec.py \
  --zarr data/libero/libero10_N500.zarr --out_dir data/libero
# -> data/libero/norm_spec_group_compatible_libero10_N500.json
# -> data/libero/norm_spec_per_dim_minmax_libero10_N500.json   (ablation baseline)
```

### 2b. SE(2) image–action augmentation (Workstream A)

The rotation is realized by **rewriting stored MuJoCo state and re-rendering** — never
by replaying dynamics (exact, zero drift). `θ ∈ {0, ±10°, ±20°, ±30°}` (K=7).

Per demo frame (`oat/env/libero/se2_state_rewrite.py`):
- every movable object's free-joint qpos is rotated about the base point
  (`pos ← R(θ)(pos − p_base) + p_base`; `quat ← q_{R_z(θ)} ⊗ quat`, **wxyz** in state);
- robot **joint-1 `qpos += θ`** — the joint-1 axis is base-z, so the whole arm
  co-rotates *exactly* with no IK re-solve;
- `set_state → sim.forward → render` both cameras, then vertical-flip to match the
  dataset's stored orientation.

Validity checks reject a `(episode, angle)` pair when the rotation breaks physics:
Panda joint-1 limit `±166°`, objects outside the **physical table-top AABB** (derived
from the sim model, inset by a margin), and support/penetration contact checks that are
**rotation-conditional** (a failure that also occurs at θ=0 is a demo artifact, not a
rotation problem, and does not reject).

Images are pre-rendered offline (below) and cached; **action/proprio labels are rotated
analytically on the fly** in the dataloader (`oat/equi/se2_transforms.py`):
`(dx,dy) ← R(θ)(dx,dy)`; `(rx,ry) ← R(θ)(rx,ry)` iff the controller rotation delta is
world-frame (gated by probe 1); `eef_pos` xy about `p_base`; `eef_quat ← q_{R_z(θ)} ⊗ q`
(**xyzw** in the dataset).

**Matched budget:** `SE2AugZarrDataset.__getitem__` draws **exactly one angle per item**
(k=0 when `augment=False`), so `len(dataset)` and steps/epoch are byte-identical with aug
on or off — the aug-vs-no-aug comparison is matched-compute by construction.

Pre-render the augmented image cache (once; ~34 GB, resumable, shardable with `--tasks`):

```bash
MUJOCO_GL=egl python scripts/prerender_se2_aug.py \
  --base_zarr data/libero/libero10_N500.zarr \
  --hdf5_dir  third_party/LIBERO/libero/datasets/libero_10 \
  --out       data/libero/libero10_N500_se2aug.zarr --resume
```

> Coverage note: 6 of 10 tasks have goals anchored to **non-rotating fixtures**
> (stove / cabinet / microwave / caddy / table-plate regions). Those scenes correctly
> reject most θ≠0 renders; the 5 LIVING_ROOM tasks are 100% valid at every angle. Overall
> 2167/3500 (episode, angle) pairs valid; angle-0 is always valid.

### 2c. EquiNoise noise initialization (Workstream B)

`oat/equi/sources.py`. Level-0 `BlockIsotropicSource` is a `torch.randn` drop-in whose
per-dim std is **tied within each block**; per-block scales `{s_xy, s_z, s_rot, s_grip}`
are hyperparameters (different physical quantities). Under group-compatible normalization
plain per-block randn is already P1-correct, so the `physical_so2` warp correction
degrades to identity (asserted at runtime). `physical_so2` remains active only under the
`per_dim_minmax` ablation. Levels 1–3 (learned source) are scaffolded (`SourceModule`
protocol + `Level1ScaleHeadSource` stub) but not wired into training.

### Verification probes (all PASS)

| Probe | Result |
|---|---|
| 1 — controller frame (GATING) | pos & rot both **world**, `pass=true` |
| 2 — render consistency | 200/200, max EEF error **6.7e-16 m** (exact FK), δ=1 all tasks |
| 3 — label round-trip | median rotated-trajectory error **6.4 nm** after OSC nullspace resync |

---

## 3. Train & eval commands

Five arms (Hydra composes each from `train_equi_flowpolicy.yaml`):

```bash
# aug ON,  full fine-tune          (group-compatible norm, block-isotropic source)
python scripts/run_workspace.py --config-name=train_equi_flowpolicy

# aug OFF, full fine-tune          (matched budget; still reads angle-0 aug renders)
python scripts/run_workspace.py --config-name=train_equi_flowpolicy_noaug

# aug ON,  frozen obs encoder
python scripts/run_workspace.py --config-name=train_equi_flowpolicy_frozen

# aug OFF, frozen obs encoder
python scripts/run_workspace.py --config-name=train_equi_flowpolicy_frozen_noaug

# ablation: per-dim min-max norm + physical_so2 source (aug on/off via override)
python scripts/run_workspace.py --config-name=train_equi_flowpolicy_perdim aug.enable=true
```

Smoke test any arm (short, offline, no rollout):

```bash
python scripts/run_workspace.py --config-name=train_equi_flowpolicy \
  training.num_epochs=2 training.max_train_steps=30 logging.mode=offline
```

Inspect a composed config without running:

```bash
python scripts/run_workspace.py --config-name=train_equi_flowpolicy --cfg job
```

**Evaluation** (rollout success rate). `lazy_eval: true` for libero10, so training does no
in-loop rollout — evaluate checkpoints afterward with the standalone script (it rebuilds
the env runner from the checkpoint's own config):

```bash
MUJOCO_GL=egl python scripts/eval_policy_sim.py \
  -c output/<run_dir>/checkpoints/            # a .ckpt file or a dir of them \
  -o output/<run_dir>/eval \
  -n 1 -d cuda:0
```

### Prerequisites checklist (run once, in order)

```bash
# 1. frozen normalization specs
python scripts/build_normalization_spec.py --zarr data/libero/libero10_N500.zarr --out_dir data/libero
# 2. GATING controller-frame probe  -> data/libero/probe_results.json  (must PASS)
MUJOCO_GL=egl python scripts/probes/probe_controller_frame.py --task_name libero10 --out data/libero/probe_results.json --settle_steps 15
# 3. augmented image cache          -> data/libero/libero10_N500_se2aug.zarr
MUJOCO_GL=egl python scripts/prerender_se2_aug.py --base_zarr data/libero/libero10_N500.zarr --hdf5_dir third_party/LIBERO/libero/datasets/libero_10 --out data/libero/libero10_N500_se2aug.zarr --resume
# 4. (optional) confidence probes
MUJOCO_GL=egl python scripts/probes/probe_render_consistency.py --hdf5_dir third_party/LIBERO/libero/datasets/libero_10 --n 200
MUJOCO_GL=egl python scripts/probes/probe_label_roundtrip.py     --hdf5_dir third_party/LIBERO/libero/datasets/libero_10 --n 20
# unit tests (no sim needed)
python -m pytest tests/ -q -m "not sim"
```

The dataset fails fast at construction if any prerequisite is missing (no frozen spec,
no aug zarr, or a probe that did not PASS), so a misconfigured arm cannot silently train
on wrong data.

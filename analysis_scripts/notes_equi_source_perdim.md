# How the noise source changes in `train_equi_flowpolicy_perdim.yaml`

Scope: what the `_perdim` ablation actually does to the **flow-matching noise source**,
with the real frozen normalizer numbers — not just the class wiring.

Files involved:
- config: [oat/config/train_equi_flowpolicy_perdim.yaml](../oat/config/train_equi_flowpolicy_perdim.yaml)
  (inherits [train_equi_flowpolicy.yaml](../oat/config/train_equi_flowpolicy.yaml))
- source: [oat/equi/sources.py](../oat/equi/sources.py) · blocks: [oat/equi/blocks.py](../oat/equi/blocks.py)
- policy: [oat/policy/equi_flow_policy.py](../oat/policy/equi_flow_policy.py)
- specs: `data/libero/norm_spec_{per_dim_minmax,group_compatible}_libero10_N500.json`

---

## 1. The config diff (only two lines change)

`_perdim` is the parent config with exactly two overrides:

```yaml
defaults: [train_equi_flowpolicy, _self_]
norm:
  mode: per_dim_minmax                                   # was: group_compatible
  spec_path: .../norm_spec_per_dim_minmax_libero10_N500.json
policy:
  source:
    warp_correction: physical_so2                        # was: none
```

Everything else about the source is inherited unchanged:
`enable: true`, `kind: block_isotropic`, `scales {xy,z,rot,grip}=1.0`,
`world_frame_rotation: false`, `prior_noise_scale: 1.0`.

So the noise-source question reduces to: **what does `warp_correction: physical_so2`
do to the sampled noise, once the normalizer is `per_dim_minmax`?**

---

## 2. Where the noise enters (unchanged by this ablation)

`EquiFlowPolicy` is `FlowPolicy` with `torch.randn` swapped for `self._source(...)` in
two spots (identical draw in both):

- **train** ([equi_flow_policy.py:186](../oat/policy/equi_flow_policy.py#L186)):
  `x0 = prior_noise_scale · source(x1.shape)` seeds the rectified-flow interpolant
  `xt = (1−t)·x0 + t·x1` **and** the regression target `v = x1 − x0`.
- **infer** ([equi_flow_policy.py:150](../oat/policy/equi_flow_policy.py#L150)):
  `x = prior_noise_scale · source(...)` is the Euler-integration seed.

The source is `randn × std`, one std **tied per action block**
([BlockIsotropicSource](../oat/equi/sources.py#L54)). The 7-D LIBERO action blocks
(`world_frame_rotation=false`, [blocks.py:65](../oat/equi/blocks.py#L65)):

| dims | block | rep | corrected by `physical_so2`? |
|------|-------|-----|------------------------------|
| 0,1  | `xy`  | **ρ₁** (SO(2) 2-vector) | **yes** |
| 2    | `z`   | ρ₀ scalar | no |
| 3,4,5| `rot` | free-iso | no (not ρ₁) |
| 6    | `grip`| ρ₀ scalar | no |

**Only the ρ₁ blocks get a correction** — here that is just `xy`.

---

## 3. What `physical_so2` computes

Per-dim std, from the **normalizer's scale vector** (not raw ranges),
[sources.py:96-105](../oat/equi/sources.py#L96):

```
std_i = block_scale · scale_i / geomean(scale over the block's dims)      (ρ₁ blocks only)
```

Under per-dim min-max, `scale_i = 2 / R_i`, so this is `geomean(R)/R_i`: the axis with the
**tighter** physical range gets **more** noise — exactly undoing the elliptical covariance
warp that per-dim min-max would otherwise stamp onto the (dx,dy) plane, restoring an
isotropic (SO(2)-equivariant) prior in physical units. It is the source-side twin of the
diffusion-side `EquiNoise` correction.

Under `group_compatible` the two ρ₁ scales are **tied by construction**, so the correction
is identically 1 — and the policy asserts that at train start
([equi_flow_policy.py:104-115](../oat/policy/equi_flow_policy.py#L104)). That is why the
parent (group_compatible) leaves `warp_correction: none`: it would be a no-op anyway.

---

## 4. The actual numbers on this dataset

Fitted, frozen action normalizer (LIBERO-10, N500):

| dim | 0 dx | 1 dy | 2 dz | 3 rx | 4 ry | 5 rz | 6 grip |
|-----|------|------|------|------|------|------|--------|
| `group_compatible` scale  | 1.0667 | 1.0667 | 1.0667 | **2.6667** | **2.6667** | **2.6667** | 1.0 |
| `group_compatible` offset | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `per_dim_minmax`  scale   | 1.0667 | 1.0667 | 1.0667 | **3.5376** | **2.9630** | **2.6936** | 1.0 |
| `per_dim_minmax`  offset  | 0 | 0 | 0 | **−0.1636** | **−0.0952** | **−0.0101** | 0 |

Feeding those scales through `build_source(...)` gives the **noise std the policy actually draws**:

| source std (per dim) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|----------------------|---|---|---|---|---|---|---|
| PARENT — group_compatible, `warp=none`     | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| PERDIM — per_dim_minmax, `warp=physical_so2` (`std_correction`) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| PERDIM — final `std`                        | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

### The punchline

**On this dataset the noise source is numerically unchanged** — both emit `randn × [1,1,1,1,1,1,1]`,
i.e. plain block-isotropic Gaussian. The reason: `physical_so2` only touches the `xy` (ρ₁)
block, and LIBERO's dx/dy min-max ranges are **symmetric and equal** (both scale `1.0667`),
so `correction_xy = [1.0667/1.0667, 1.0667/1.0667] = [1, 1]`. The rot dims (3,4,5) — whose
per-dim scales *do* differ (3.54 / 2.96 / 2.69) — are a **free-iso** block, not ρ₁, so
`physical_so2` deliberately leaves them alone.

So what the `_perdim` run really changes vs the parent is **the target normalization**, not the
noise:
- rot-block scales become **untied** (3.54 / 2.96 / 2.69 instead of a shared 2.6667) and pick up
  **nonzero offsets** (−0.16 / −0.10 / −0.01). That reshapes the normalized target `x1`
  anisotropically, hence the interpolant and `v = x1 − x0` the policy regresses.
- the `physical_so2` **path is activated** but evaluates to identity here; it exists to de-warp
  the ρ₁ block *if* those ranges were asymmetric, and the runtime assert guarantees it never
  silently does anything under group_compatible.

---

## 5. When `physical_so2` would bite (illustrative)

If the dx/dy ranges were asymmetric — say `R_x = 0.5`, `R_y = 1.5`
(`scale = [4.0, 1.333]`) — the same code yields:

```
correction_xy = scale_xy / geomean(scale_xy)
              = [4.0, 1.333] / sqrt(4.0·1.333)
              = [√3, 1/√3] ≈ [1.732, 0.577]
```

i.e. the noise std on the tight x-axis is boosted ~3× relative to y, making the prior a circle
again in physical space. That is the effect `_perdim` is built to test; the LIBERO xy action
ranges just happen to be symmetric, so it is inert here and the ablation isolates the
per-dim-minmax **target** reshaping (untied rot scales + offsets) with the noise held fixed.

---

## TL;DR

- `_perdim` flips the source from the `warp=none` path to `warp=physical_so2`, which rescales the
  **ρ₁ (`xy`) noise std** by `scale_i / geomean(scale)` read off the per-dim-minmax normalizer.
- On LIBERO-10 that correction is **exactly identity** (symmetric dx/dy ranges) → the emitted noise
  is still `randn × 1` per dim, same as the parent.
- The genuine difference this config makes is in **normalization of the target action**
  (rot scales untied `2.667 → 3.54/2.96/2.69`, offsets `0 → −0.16/−0.10/−0.01`), not in the noise
  draws — with the source-side warp mechanism activated but numerically dormant on this data.

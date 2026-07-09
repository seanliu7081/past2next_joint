# OAT Joint Training **with Enriched Past**: Co-Training the Tokenizer + an AR Policy Conditioned on Past Actions

## 1. TL;DR

This variant combines two existing ideas in the repo:

1. **Joint OAT training** ([JOINT_TRAINING.md](JOINT_TRAINING.md)) — instead of freezing a pre-trained
   `OATTok` action tokenizer and only training the AR policy on top of it, build a **trainable** tokenizer
   from scratch *inside* the policy and co-train it end-to-end with the backbone. The objective is
   `recon_loss_weight * recon_loss + ce_loss_weight * ce_loss`, where the tokenizer learns **only** from
   reconstruction (FSQ straight-through estimator) and the backbone/obs-encoder learn **only** from
   cross-entropy against the tokenizer's *detached* token indices.

2. **Enriched-past conditioning** ([oat_policy_with_enriched_past.py](oat/policy/oat_policy_with_enriched_past.py)) —
   in addition to the usual observation features, condition the AR policy on the **raw past `past_n` action
   steps** plus two **explicit higher-order derivative features** (acceleration & jerk) computed from those
   past actions. This gives the policy exact velocity / temporal-pattern / command-inertia information that
   observations alone (position + coarse 2-frame velocity) do not expose.

The new policy **[`OATPolicyWithEnrichedPastJoint`](oat/policy/oat_policy_with_enriched_past_joint.py)** is
to `OATPolicyWithEnrichedPast` exactly what `OATPolicyJoint` is to `OATPolicy`: same enriched conditioning &
inference, but the tokenizer is no longer frozen — it is built from scratch and co-trained. It **reuses the
existing joint workspace** [`TrainOATJointWorkspace`](oat/workspace/train_oat_joint.py) unchanged (warmup,
joint optimizer, separate recon/CE logging all work as-is).

Files added (nothing existing was modified):

| File | Role |
|---|---|
| [oat/policy/oat_policy_with_enriched_past_joint.py](oat/policy/oat_policy_with_enriched_past_joint.py) | `OATPolicyWithEnrichedPastJoint` — the joint, enriched-past policy |
| [oat/config/train_oat_joint_with_enriched_past.yaml](oat/config/train_oat_joint_with_enriched_past.yaml) | Training config (reuses `TrainOATJointWorkspace`, `libero10_with_past` task) |

---

## 2. At a Glance (shipped config)

Concrete instantiated values for [train_oat_joint_with_enriched_past.yaml](oat/config/train_oat_joint_with_enriched_past.yaml)
on `task/policy: libero/libero10_with_past`.

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `Da` / `action_dim` | Action feature dim | **7** | `shape_meta.action.shape[0]` (libero10) |
| `Ta` / `horizon` / `sample_horizon` | Action chunk length fed to the tokenizer | **16** | `horizon: 16` |
| `To` / `n_obs_steps` | Observation history window | **2** | `n_obs_steps: 2` |
| `past_n` | # raw past action steps used for conditioning | **7** | `past_n: 7` |
| `n_action_steps` | Receding-horizon actions executed per inference | **8** | `n_action_steps: 8` |
| `N_EXPLICIT_FEATURES` | Explicit derivative tokens (acc, jerk) | **2** | class const |
| **`max_cond_len`** | **AR condition length = `To + 2 + past_n`** | **2+2+7 = 11** | `__init__` |
| `L` / `num_registers` / `latent_horizon` | # latent register tokens = AR target length | **8** | `encoder.num_registers: 8` |
| FSQ `levels` | Per-dim quantization grid | **[8, 5, 5, 5, 5]** | `quantizer.levels` |
| `Dl` / `latent_dim` | Latent dim = `len(levels)` | **5** | `${eval:'len(levels)'}` |
| `codebook_size` | `prod(levels)` = 8·5·5·5·5 | **5000** | `FSQ._levels.prod()` |
| AR `vocab_size` | `codebook_size + 1` (incl. `<BOS>`) | **5001** | `__init__` |
| `bos_id` | `<BOS>` token id (= `codebook_size`) | **5000** | `__init__` |
| `embed_dim` (`n_emb`) | AR backbone hidden dim | **256** | `policy.embed_dim` |
| `n_layers` / `n_heads` | AR backbone depth / heads | **4 / 4** | `policy.n_layers/n_heads` |
| tokenizer `emb_dim` / `head_dim` | Encoder & decoder hidden / per-head dim | **256 / 64** | `encoder/decoder` |
| encoder `depth` / decoder `depth` | RegisterEncoder / SinglePassDecoder blocks | **2 / 4** | `encoder/decoder.depth` |
| `d` / obs feature dim (`cond_dim`) | Fused vision+state per obs step | *(printed at runtime)* | `FusedObservationEncoder.output_feature_dim()` |
| batch size `B` | Per-step batch | **64** | `dataloader.batch_size` |

> **Note on `d`.** Every enriched-past projection (`acc_proj`, `jerk_proj`, `raw_proj`) outputs the **same**
> `d = obs_feature_dim` so the explicit/raw tokens concatenate cleanly with the obs feature tokens along the
> sequence axis. The exact value is the fused vision(ResNet18)+state dim and is printed by
> `OATPolicyWithEnrichedPastJoint.__init__` at startup (the `obs enc / act tok / policy` report). FSQ itself
> is **parameter-free**; the tokenizer's learnable weights live entirely in the encoder + decoder and are
> **100% trainable** here (vs frozen in the non-joint `OATPolicyWithEnrichedPast`).

---

## 3. What's different from plain joint (`OATPolicyJoint`)

Everything in the **reconstruction branch** and the joint-training machinery is identical to
[JOINT_TRAINING.md](JOINT_TRAINING.md). The **only** additions are in the **conditioning** of the
cross-entropy branch:

| Aspect | `OATPolicyJoint` | `OATPolicyWithEnrichedPastJoint` |
|---|---|---|
| AR condition | obs features only, `[B, To, d]` | `[obs, acc, jerk, raw_past]`, `[B, To+2+past_n, d]` |
| Extra batch input | — | `batch['past_action']` `[B, past_n, Da]` |
| Extra trainable modules | — | `acc_proj`, `jerk_proj`, `raw_proj` (small MLPs) |
| Normalizers | obs enc + tokenizer | obs enc + tokenizer + **`action_normalizer`** (for past actions) |
| `max_cond_len` | `To` | `To + N_EXPLICIT_FEATURES + past_n` |
| Dataset / task | `libero/libero10` (`ZarrDataset`) | `libero/libero10_with_past` (`LazyZarrDatasetWithPastAction`) |
| Inference | one-shot from obs | rolling **past-action buffer** (`_past_buffer`), see §8 |

The **tokenizer is trained exactly the same way** (recon-only), and the **past-action conditioning never
flows gradient into the tokenizer** — see §6.

---

## 4. End-to-End Data Flow (training `forward`)

Source: [oat_policy_with_enriched_past_joint.py → `forward`](oat/policy/oat_policy_with_enriched_past_joint.py).
Two branches share **one** optimizer step.

```
══════════════ RECONSTRUCTION BRANCH (trains TOKENIZER only) ══════════════════════════════════════

 raw action a        normalize        RegisterEncoder      FSQ quantizer         SinglePassDecoder       recon
 [B,Ta=16,Da=7] ───► nsamples ──────►  latents  ─────────► quant (STE,float) ──► recons ─────────────► MSE(recons, nsamples)
                     [B,16,7]         [B,L=8,Dl=5]         [B,8,5]  ──┐         [B,16,7]               = recon_loss
                  tok.normalizer                          tokens(long)│        (nested 'pow2' dropout    ▲ grad flows back through
                  ['action']                              [B,8] ◄─────┘         active in train mode)     quant via STE → encoder
                                                              │
                                       tokens.detach().long() │  ◄── STOP-GRAD (FSQ indices non-differentiable)
                                                              ▼
══════════════ CROSS-ENTROPY BRANCH (trains BACKBONE + OBS ENC + acc/jerk/raw PROJECTIONS) ═════════

 obs_dict ──► FusedObservationEncoder ──► obs_features [B,To=2,d] ─────┐
 {rgb×2,state}                                                         │
                                                                       ├─► _build_condition ─► cond
 batch['past_action'] [B,past_n=7,Da=7] ─► action_normalizer.normalize │      (concat on seq) [B, 11, d]
        │                                       │ norm_past [B,7,7]     │
        │   a_{t-1},a_{t-2},a_{t-3} = norm_past[:,-1],[:,-2],[:,-3]     │
        │   acc  = a_{t-1} − a_{t-2}            ─► acc_proj  ─► [B,1,d] ─┤   (explicit derivative tokens)
        │   jerk = a_{t-1} −2a_{t-2}+a_{t-3}    ─► jerk_proj ─► [B,1,d] ─┤
        └─────────────────  norm_past          ─► raw_proj  ─► [B,7,d] ─┘   (shared per-step projection)

                       prepend <BOS>=5000:  action_tokens = [BOS, t0..t7]   [B, L+1 = 9]
                                                  │
                       logits = model( action_tokens[:, :-1]=[B,8],  cond=[B,11,d] )   ─► [B, 8, vocab=5001]
                                                  │
                       ce_loss = CE( logits.reshape(-1,5001),  action_tokens[:,1:].reshape(-1) )

══════════════ COMBINE ════════════════════════════════════════════════════════════════════════════
 total = recon_loss_weight · recon_loss  +  current_ce_weight · ce_loss
 returns {'loss': total, 'recon_loss': recon_loss, 'ce_loss': ce_loss}
```

The dict return shape is exactly what `TrainOATJointWorkspace` expects (it backward()s `out['loss']` and
logs `recon_loss` / `ce_loss` separately).

---

## 5. The enriched condition: `_build_condition` (inherited)

Source: [oat_policy_with_enriched_past.py → `_build_condition`](oat/policy/oat_policy_with_enriched_past.py).
Reused unchanged by the joint subclass.

```python
norm_past = self.action_normalizer["action"].normalize(past_actions)   # [B, past_n, Da]

# explicit higher-order derivatives from the nearest 3 past steps
a_t1, a_t2, a_t3 = norm_past[:, -1], norm_past[:, -2], norm_past[:, -3]
acc  = a_t1 - a_t2                    # acceleration-level   [B, Da]
jerk = a_t1 - 2.0 * a_t2 + a_t3       # jerk-level           [B, Da]
explicit = stack([acc_proj(acc), jerk_proj(jerk)], dim=1)              # [B, 2, d]

# raw history, shared projection over all past_n steps
raw_feat = self.raw_proj(norm_past)                                    # [B, past_n, d]

cond = cat([obs_features, explicit, raw_feat], dim=1)                  # [B, To + 2 + past_n, d]
```

- **Why explicit acc/jerk?** Observations give position + a coarse 2-frame velocity; acceleration and jerk
  require cross-timestep differencing the model would otherwise have to learn. Providing them directly is a
  cheap, strong inductive bias.
- **Why also the raw history?** The raw `past_n` actions preserve the full temporal pattern / command inertia
  / task progress, beyond the 3-step derivatives.
- All three projections are `Linear(Da→d) → GELU → Linear(d→d)`. `acc_proj` and `jerk_proj` are independent
  (different physical scales); `raw_proj` is shared across all `past_n` steps.

---

## 6. Gradient flow — what trains what

| Module | Trained by | Reaches it? |
|---|---|---|
| Tokenizer encoder + decoder | `recon_loss` (FSQ STE) | ✅ recon only — CE targets are `.detach()`ed integer indices |
| FSQ quantizer | — | parameter-free (buffers only) |
| AR backbone (`self.model`) | `ce_loss` | ✅ |
| Obs encoder | `ce_loss` | ✅ |
| `acc_proj` / `jerk_proj` / `raw_proj` | `ce_loss` | ✅ (they feed `cond`, which CE depends on) |
| `action_normalizer` | — | frozen buffers, loaded from the dataset normalizer |

Crucially, the **past-action conditioning path does *not* touch the tokenizer**: `past_action` only feeds the
projections → `cond` → CE branch. The tokenizer's gradient comes solely from `recon_loss` through `quant`
(straight-through). This was verified empirically (recon-only backward → tokenizer grad > 0, backbone grad
== 0; CE-only backward → tokenizer grad **exactly 0**, backbone + all three projections grad > 0).

**Warmup.** `TrainOATJointWorkspace` calls `policy.set_ce_weight(0.0)` for the first
`training.tokenizer_warmup_epochs` (= **50**) epochs, so only the tokenizer trains early (the codebook
settles before the policy starts chasing it). The CE branch is still *computed* (× 0.0) so all backbone /
projection params stay in the autograd graph and DDP `find_unused_parameters=False` remains valid.

> **Zero-init transient (benign).** `OATTok` zero-initializes its output projections, so for the first few
> steps the encoder emits ~zero latents and only the decoder's output layer receives gradient; latents and
> per-param gradients fill in within a handful of steps. This is a property of `OATTok` (identical for the
> plain `OATPolicyJoint`), which is exactly why a tokenizer-only warmup is used.

---

## 7. Data plumbing: `past_action` & time alignment

Task: `task/policy: libero/libero10_with_past`; dataset
[`LazyZarrDatasetWithPastAction`](oat/dataset/lazy_zarr_dataset_with_past.py) (extends the lazy zarr dataset,
adds `past_n`). Each item is `{obs, action, past_action}`. Time alignment (`To=2, Ta=16, past_n=7`):

```
obs:         [t,   t+1]                    (To = 2 frames)
past_action: [t-6, t-5, ..., t]            (past_n = 7 frames, ending at current step t)
action:      [t+1, t+2, ..., t+16]         (Ta = horizon = 16 frames, the tokenizer target chunk)
```

At episode starts the sequence sampler zero-pads, so `past_action` is naturally all-zeros early in an
episode (mirrored at inference by the zero-initialized `_past_buffer`). The dataset's `n_action_steps` field
is bound to `${horizon}` (= 16), i.e. the tokenizer always sees a full `horizon`-length chunk.

The single dataset normalizer (`dataset.get_normalizer()`) is shared three ways by
`set_normalizer` — obs encoder, the **trainable tokenizer's internal** normalizer, and the policy's
`action_normalizer` (past-action path) — keeping the recon (`normalize`) and detokenize (`unnormalize`)
spaces consistent.

---

## 8. Inference path (inherited, rolling past buffer)

`predict_action` is inherited unchanged from `OATPolicyWithEnrichedPast`:

1. Encode obs → `features`; build `cond` from `features` + the current `_past_buffer` (zeros on the first
   call after `reset()`).
2. Autoregressively generate `L` latent tokens from `<BOS>`, conditioned on `cond` (temperature / top-k
   sampling), then `clamp` to valid codebook ids.
3. `detokenize` (codebook lookup → decoder → unnormalize) → `action_pred` `[B, Ta, Da]`; execute the first
   `n_action_steps`.
4. **Update `_past_buffer`** with the just-predicted actions (receding-horizon roll) so the next call sees a
   correct past window.

The env runner calls `reset()` at episode start, which clears `_past_buffer`.

---

## 9. Code structure

`OATPolicyWithEnrichedPastJoint(OATPolicyWithEnrichedPast)` **subclasses** the enriched-past policy and:

- **`__init__`** — skips the parent's tokenizer-freezing constructor via
  `super(OATPolicyWithEnrichedPast, self).__init__()` (the same pattern `OATPolicyJoint` uses to skip
  `OATPolicy.__init__`), then rebuilds the projections / `action_normalizer` / AR model **without** freezing
  the tokenizer, and adds the joint loss weights (`recon_loss_weight`, `ce_loss_weight`, `current_ce_weight`).
- **`forward`** — returns the `{loss, recon_loss, ce_loss}` dict (recon branch + enriched-cond CE branch).
- **`get_optimizer`** — adds a `tokenizer_lr` AdamW group alongside backbone(+projections) and obs-encoder
  groups (2D params → weight decay, 1D → none).
- **`set_normalizer`** — also feeds the trainable tokenizer's internal normalizer.
- **`set_ce_weight`** — the warmup toggle the workspace calls.
- **Inherited unchanged**: `_build_condition`, `predict_action`, `reset`, `create_dummy_observation`,
  `get_observation_*`.

The workspace [`TrainOATJointWorkspace`](oat/workspace/train_oat_joint.py) is **reused as-is** — it is generic
over the policy (reads `cfg.task.policy.dataset`, calls `set_normalizer` / `set_ce_weight`, expects the dict
forward, and during the `sample_every` eval calls `predict_action(batch['obs'])` + `action_tokenizer.autoencode`).

---

## 10. Running it

```bash
HYDRA_FULL_ERROR=1 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 scripts/run_workspace.py \
    --config-name=train_oat_joint_with_enriched_past \
    training.num_demo=500 task.policy.lazy_eval=false
```

- No dataset override is needed (unlike the plain `train_oat_joint` command): the `_with_past` task already
  defaults to the lazy `LazyZarrDatasetWithPastAction`, and that dataset is **required** (it provides
  `past_action`). `training.num_demo=500` selects `data/libero/libero10_N500.zarr` via the `zarr_path`
  interpolation.
- Drop `task.policy.lazy_eval=false` (and the `MUJOCO_*` vars) to skip in-training Libero rollouts.
- Key knobs: `policy.recon_loss_weight`, `policy.ce_loss_weight`, `optimizer.tokenizer_lr`,
  `training.tokenizer_warmup_epochs`, `policy.past_n`, `policy.action_tokenizer.quantizer.levels`.

---

## 11. Caveats

- **Eval diagnostic quirk (pre-existing in the joint workspace).** The `sample_every` reconstruction metric
  calls `predict_action` across consecutive validation batches **without** `reset()`, so the rolling
  `_past_buffer` carries over between (unrelated) batches. This only perturbs the logged `test_reconst_mse`
  diagnostic — it does **not** affect training or real env rollouts (which do `reset()` per episode). Left as
  the existing workspace behavior; a `reset()`-per-batch workspace variant would fix the metric if desired.
- **`past_n ≥ 3` required** for the explicit acc/jerk features (they index `norm_past[:, -3]`). The shipped
  `past_n = 7` satisfies this.
- **Consistency:** `horizon` (tokenizer chunk length), the encoder/decoder `sample_horizon`, and the
  dataset's `n_action_steps` must all match (`${horizon}` ties them together). `latent_dim` auto-derives from
  `len(levels)`; `latent_horizon` mirrors `num_registers`.
```

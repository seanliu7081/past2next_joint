# OAT Joint Training: End-to-End Co-Training of the Action Tokenizer (`oattok`) + Autoregressive Policy (`oatpolicy`)

## 1. TL;DR

**Joint OAT training** simultaneously optimizes a continuous-action **tokenizer** (an FSQ autoencoder, `OATTok`) and an **autoregressive policy backbone** (`AutoregressiveModel`) that predicts the tokenizer's discrete codes from observations. In the *default* OAT pipeline these are two separate stages: first you train `OATTok` to convergence as a standalone action autoencoder, then you **freeze** it and train the AR policy to predict its (fixed) token vocabulary via cross-entropy. This repo's joint variant ([oat_policy_joint.py](oat/policy/oat_policy_joint.py) + [train_oat_joint.py](oat/workspace/train_oat_joint.py)) instead builds a **trainable** tokenizer from scratch inside the policy and co-trains everything end to end. The total objective is a weighted sum of (a) a tokenizer **reconstruction** loss (MSE between decoded and ground-truth normalized actions, made differentiable by the FSQ straight-through estimator) and (b) a backbone/obs-encoder **cross-entropy** loss against the tokenizer's *detached* discrete token indices. Because the CE targets are detached and FSQ indices are non-differentiable, the **tokenizer is trained only by reconstruction**, while the **backbone + obs encoder are trained only by CE** — but both gradients coexist in one optimizer step. A configurable tokenizer-only **warmup** (`tokenizer_warmup_epochs`) zeros the CE weight for the first K epochs so the codebook can settle before the policy starts chasing it. The motivation: a jointly-shaped token vocabulary can be made easier for the AR policy to predict (and remain reconstructable) rather than being fixed in advance by a tokenizer that never saw the policy's prediction difficulty.

---

## 2. At a Glance (shipped `libero10` config)

All numbers below are the **concrete instantiated values** for the shipped config [train_oat_joint.yaml](oat/config/train_oat_joint.yaml) on the `task/policy: libero/libero10` task.

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `Da` / `action_dim` | Action feature dim | **7** | `shape_meta.action.shape[0]` (libero10) |
| `Ta` / `horizon` / `sample_horizon` | Action sequence length fed to tokenizer | **32** | `horizon: 32` |
| `n_obs_steps` (`To`) | Observation history window (AR cond length) | **2** | `n_obs_steps: 2` |
| `n_action_steps` | Receding-horizon actions executed per inference | **16** | `n_action_steps: 16` |
| `L` / `num_registers` / `latent_horizon` | # latent register tokens (AR sequence length) | **8** | `encoder.num_registers: 8`, `decoder.latent_horizon` mirrors it |
| FSQ `levels` | Per-dim quantization grid | **[8, 5, 5, 5]** | `quantizer.levels` |
| `Dl` / `latent_dim` | Latent dim = `len(levels)` | **4** | `latent_dim: ${eval:'len(levels)'}` |
| `codebook_size` | `prod(levels)` = 8·5·5·5 | **1000** | `FSQ._levels.prod()` |
| AR `vocab_size` | `codebook_size + 1` (incl. `<BOS>`) | **1001** | `OATPolicyJoint.__init__` |
| `bos_id` | `<BOS>` token id (= `codebook_size`) | **1000** | `OATPolicyJoint.__init__` |
| `embed_dim` (`n_emb`) | AR backbone hidden dim | **256** | `policy.embed_dim: 256` |
| `n_layers` | AR backbone decoder blocks | **4** | `policy.n_layers: 4` |
| `n_heads` | AR backbone attention heads (head_dim = 64) | **4** | `policy.n_heads: 4` |
| tokenizer `emb_dim` | Encoder & decoder hidden dim | **256** | `encoder/decoder.emb_dim: 256` |
| tokenizer `head_dim` | Per-head dim (→ 4 heads at emb_dim 256) | **64** | `encoder/decoder.head_dim: 64` |
| encoder `depth` | RegisterEncoder transformer blocks | **2** | `encoder.depth: 2` |
| decoder `depth` | SinglePassDecoder transformer-decoder layers | **4** | `decoder.depth: 4` |
| `token_dropout_mode` | Nested (Matryoshka) dropout mode | **`pow2`** | `decoder.token_dropout_mode` |
| `use_causal_decoder` | Causal self-attn over action queries in decoder | **true** | `decoder.use_causal_decoder` |
| obs feature dim (`cond_dim`) | Fused vision(64)+state(10) per obs step | **74** | `FusedObservationEncoder.output_feature_dim()` |
| batch size `B` | Per-step batch | **64** | `dataloader.batch_size` |

**Notes / parameter counts.** Exact parameter counts are printed at runtime by `OATPolicyJoint.__init__` (see the `obs enc / act tok / policy` report), and are not statically asserted in config — the orders of magnitude here are small: the AR backbone is 4 layers × 256 dim, the tokenizer encoder/decoder are 256-dim transformers of depth 2/4, and FSQ itself is **parameter-free** (0 learnable params; it contributes only `codebook_size`-derived buffers). The obs encoder's biggest block is a ResNet18 vision backbone. The trainable-ratio report distinguishes that, unlike `OATPolicy` (tokenizer frozen, ~0% trainable), here the tokenizer is **100% trainable**.

---

## 3. End-to-End Data Flow (two loss branches)

```
                          ┌──────────────────────────  RECONSTRUCTION BRANCH (trains TOKENIZER)  ──────────────────────────┐
                          │                                                                                                │
 raw action a             │   normalize          RegisterEncoder            FSQ quantizer            SinglePassDecoder       recon
 [B, Ta=32, Da=7] ────────┼──► nsamples ─────────►  latents  ───────────►  quant (STE, float)  ───►  recons ──────────────► MSE(recons, nsamples)
                          │   [B,32,7]            [B, L=8, Dl=4]           [B, 8, 4]  ──┐           [B,32,7]                 = recon_loss
                          │   (normalizer['action']                       tokens (long)│           (nested 'pow2' dropout    ▲ gradient flows back
                          │    .normalize)                                [B, 8]  ◄─────┘            active in train mode)     through quant via STE
                          │                                                   │                                                into encoder
                          └───────────────────────────────────────────────── │ ──────────────────────────────────────────────────────────────────┘
                                                                              │  tokens.detach().long()   (STOP-GRAD: FSQ indices non-differentiable)
                                                                              ▼
                          ┌──────────────────────────  CROSS-ENTROPY BRANCH (trains BACKBONE + OBS ENCODER)  ──────────────┐
 obs_dict                 │   FusedObservationEncoder              prepend <BOS>=1000                                       │
 {rgb x2, state} ─────────┼──►  features (cond)  ─────────┐    action_tokens = [BOS, t0..t7]                                │
 [B, To=2, ...]           │     [B, 2, 74]                │    [B, L+1 = 9]                                                  │
                          │                               │         │ (input = tokens[:, :-1] = [B, 8])                     │
                          │                               ▼         ▼                                                       │
                          │                       AutoregressiveModel (cross-attn cond + causal self-attn)                  │
                          │                               │                                                                 │
                          │                               ▼                                                                 │
                          │                          logits [B, L=8, vocab=1001]  ──► CE( logits, action_tokens[:,1:] )     │
                          │                                                              = ce_loss                          │
                          │                                                              ▲ gradient flows into backbone     │
                          │                                                                + obs encoder ONLY               │
                          └─────────────────────────────────────────────────────────────────────────────────────────────┘

 total_loss = recon_loss_weight * recon_loss  +  current_ce_weight * ce_loss
            =        1.0         * recon_loss  +        {0.0 (warmup) | 1.0 (joint)} * ce_loss
```

**Gradient routing (exact):**

- **Tokenizer (encoder + decoder)** receives gradients **only** from `recon_loss`. The path `nsamples → encoder → latents → FSQ.quantize → quant → decoder → recons → MSE` is differentiable end-to-end because FSQ rounding uses a **straight-through estimator** (`round_ste`: forward = `round(z)`, backward = identity). FSQ has **no learnable parameters**, so the gradient passes straight through it into the encoder.
- **Backbone (`self.model`) + obs encoder** receive gradients **only** from `ce_loss`. The CE *targets* are `tokens.detach().long()` — detached and integer-valued — so CE cannot and does not backprop into the tokenizer.
- The two branches are summed into one scalar `total` and backpropped in a single `accelerator.backward(loss)` call; the joint AdamW optimizer then updates all three parameter groups in one step.

---

## 4. Action Tokenizer: `OATTok`

`OATTok` ([tokenizer.py](oat/tokenizer/oat/tokenizer.py)) is an FSQ autoencoder composed of a `RegisterEncoder`, an `FSQ` quantizer, and a `SinglePassDecoder`, plus a `LinearNormalizer` (identity until `set_normalizer` loads dataset stats). Key derived attributes: `latent_horizon = decoder.latent_horizon` (= 8) and `codebook_size = quantizer.codebook_size` (= 1000).

In the **joint forward** ([oat_policy_joint.py](oat/policy/oat_policy_joint.py) lines 196–201) the tokenizer is exercised **component-by-component** (not via `OATTok.forward`) so that `quant` and `tokens` are both available:

```python
nsamples = tok.normalizer['action'].normalize(action)   # [B, 32, 7]
latents  = tok.encoder(nsamples)                         # [B, 8, 4]
quant, tokens = tok.quantizer(latents)                   # quant: [B,8,4] STE float; tokens: [B,8] long
recons   = tok.decoder(quant)                            # [B, 32, 7]   (no eval_keep_k -> nested dropout sampled)
recon_loss = F.mse_loss(recons, nsamples)
```

> Reconstruction loss is computed in **normalized** space (`recons` vs `nsamples`), exactly as in the standalone `OATTok.forward`. The decoder is called with `eval_keep_k=None`, so in `train()` mode the nested `pow2` dropout **samples a random prefix length per sample** (see §4.4).

### 4.1 RegisterEncoder (`depth=2`, `emb_dim=256`, `head_dim=64`, `num_registers=8`)

[register_encoder.py](oat/tokenizer/oat/encoder/register_encoder.py). Compresses a variable-length action sequence into `L=8` fixed latent tokens via learnable **register** tokens.

| Stage | Operation | Output shape |
|---|---|---|
| Input | normalized action | `[B, 32, 7]` |
| Sample embed | `SampleEmbedder` = `Linear(7→256)` (xavier, bias 0) | `[B, 32, 256]` |
| Pos-emb | sincos `PositionalEmbeddingAdder(max_sizes=[32])`, added | `[B, 32, 256]` |
| Concat registers | `[action_emb ; registers.expand(B,8,256)]` | `[B, 40, 256]` |
| Transformer | `depth=2` pre-norm blocks, 4 heads (256/64), gated SiLU MLP, qk-norm; causal-last mask | `[B, 40, 256]` |
| Slice registers | `x[:, 32:]` (drop action tokens) | `[B, 8, 256]` |
| Head | `LinearHead(256→4)` (zero-init proj, optional Fp32LayerNorm) | `[B, 8, 4]` |

**Attention mask (`create_causal_last_mask`, lru-cached):** action tokens attend to all actions but **not** to registers; registers attend to **all actions** + **causally** to previous registers. This makes the 8 registers an ordered, causal summary of the action chunk — important because the AR policy will later generate them left-to-right.

### 4.2 FSQ quantizer (`levels=[8,5,5,5]`, parameter-free)

[fsq.py](oat/tokenizer/oat/quantizer/fsq.py). Finite Scalar Quantization ([Mentzer et al. 2023](https://arxiv.org/abs/2309.15505)): each of the `Dl=4` latent dims is independently rounded to one of its `levels[i]` grid points; no codebook embeddings, no commitment loss, no collapse.

- `dim = len(levels) = 4`; `codebook_size = prod(levels) = 1000`; `_basis = cumprod([1,8,5,5]) = [1,8,40,200]` (mixed-radix index basis).
- **Forward `quantizer(latents) → (quant, tokens)`:**
  - `quant` `[B,8,4]` float: `bound()` (tanh soft-clip to each level's range) → `round_ste` → normalize by `levels//2` → values in `[-1,1]`, **STE-differentiable**.
  - `tokens` `[B,8]` long: `codes_to_indices(quant)` = mixed-radix dot with `_basis`, in `[0, 999]`.
- **Inverse `indices_to_embedding(tokens) → codes`** `[B,8,4]`: `(idx // _basis) % _levels` then inverse scale-shift back to `[-1,1]`. Used at inference (`detokenize`).
- **Training-only stochastics:** `drop_quant_p` (skip quant per-sample) and `corrupt_tokens_p` (random token swap). **Both are 0.0** in the shipped config (FSQ instantiated with only `levels`), so neither is active here.

### 4.3 SinglePassDecoder (`depth=4`, `emb_dim=256`, `head_dim=64`, `use_causal_decoder=true`)

[single_pass_decoder.py](oat/tokenizer/oat/decoder/single_pass_decoder.py). A query-only cross-attention decoder that reconstructs all `Ta=32` action steps in **one pass** (no autoregression over actions).

| Stage | Operation | Output shape |
|---|---|---|
| Action queries | `sample_pos_emb([32])` sincos, expanded, transposed `B D T → B T D` | `[B, 32, 256]` |
| Latent project | `LinearLayer(latent_dim 4 → 256)` | `[B, 8, 256]` |
| Latent pos-emb | `PositionalEmbeddingAdder(max_sizes=[8])`, added | `[B, 8, 256]` |
| Nested dropout | `MaskedNestedDropout('pow2')` — replace tail latents with mask token | `[B, 8, 256]` |
| Decoder | `nn.TransformerDecoder`, `depth=4` layers, `nhead=256//64=4`, `dim_feedforward=4·256=1024`, `gelu`, `norm_first=True`, `batch_first=True` | `[B, 32, 256]` |
| Head | `LinearHead(256→7)` | `[B, 32, 7]` |

**Causal mode (`use_causal_decoder=true`):** a lower-triangular `tgt_mask` of shape `(32,32)` is applied to the action-query **self-attention** (`tgt_is_causal=True`); **cross-attention** to the 8 latents is **unrestricted** (every action step sees every latent). Queries are pure positional embeddings (no learnable content tokens) — each action position "knows" only its index and the latents.

### 4.4 Nested / token dropout (`pow2`) — coarse-to-fine latents

[token_dropout.py](oat/tokenizer/oat/model/token_dropout.py) `MaskedNestedDropout`. Implements **Matryoshka-style** prefix dropout: keep the first `k` latents, replace latents `k…L-1` with a single learnable `dropout_mask_token` `(256,)`. This forces a **nested/coarse-to-fine** structure where latent 0 carries the coarsest info and later latents add refinement.

- **`pow2` mode:** `keep_k` sampled uniformly from the powers of two in `[1, L]`. For `L=8`: `keep_k ∈ {1, 2, 4, 8}`, each with prob `1/4`. (Requires `L` to be a power of two — 8 qualifies.)
- **Train (`model.train()`):** a per-sample `keep_k` is sampled every forward; mask = positions `≥ keep_k` → mask token. This is exactly what happens in the joint reconstruction branch (decoder called with `eval_keep_k=None`).
- **Eval (`model.eval()`):** uses the explicit `eval_keep_k` list if provided, else no dropout (full 8 latents). At inference (`detokenize`), `eval_keep_k=token_lens` is passed so the decoder keeps exactly the number of generated tokens.

Practical payoff: the AR policy can stop generating early (e.g. 1/2/4 tokens) and still get a valid, monotonically-refining reconstruction — enabling compute-adaptive inference.

---

## 5. Shared Transformer Building Blocks

The tokenizer's `RegisterEncoder` is built from OAT's custom block library ([oat/tokenizer/oat/model/](oat/tokenizer/oat/model/)); the decoder uses PyTorch's native `nn.TransformerDecoder`. The AR backbone (§7) uses its own custom blocks. Shared conventions for the **OAT block library**:

- **Attention** (`SelfAttention`/`CrossAttention`): multi-head, `num_heads = dim // head_dim`; separate `wq/wk/wv` (bias default `False`), output proj (bias `False`); optional **QK-normalization** via `Fp32LayerNorm(bias=False, elementwise_affine=False)` over `head_dim` (enabled in encoder blocks). Uses `F.scaled_dot_product_attention`; masks are passed in externally (no built-in causality).
- **MLP**: standard `Mlp` (GELU) or **`GatedMlp`/SwiGLU** (SiLU, hidden ≈ `2·hidden/3`). Default expansion `mlp_ratio=4.0`. The encoder blocks use the **gated** SiLU variant.
- **Normalization**: `Fp32LayerNorm` (LayerNorm computed in fp32, cast back), **pre-norm** residual structure (`x → norm → attn/mlp → +residual`), affine disabled by default. Optional **AdaLN-Zero** modulation in `BlockAdaLN`/`LinearHead` (not used by the shipped tokenizer heads beyond optional norm).
- **Positional embeddings**: default **sincos** (`build_1d_sincos_posemb`, temperature 10000), `absolute` scaling (crop to length). `learnable_sum`/`learnable_prod` variants exist but are unused here. `PositionalEmbeddingAdder` rearranges `[B,D,…] → [B,…,D]` and adds.
- **Stochastic depth** (`DropPath`) and **`SampleEmbedder`** (patch/linear projection, xavier init, bias 0). DropPath defaults to 0.0.

---

## 6. Observation Encoder: `FusedObservationEncoder`

[fused_obs_encoder.py](oat/perception/fused_obs_encoder.py). Fuses **vision** + **state** per observation step; text disabled for libero10. `cond_dim = output_feature_dim() = 74`.

### 6.1 Vision (`RobomimicRgbEncoder`, crop `76×76`)

[robomimic_vision_encoder.py](oat/perception/robomimic_vision_encoder.py). Two RGB ports: `agentview_rgb`, `robot0_eye_in_hand_rgb`, each `[B, To=2, 128, 128, 3]`.

- **CropRandomizer:** train = random `76×76` crop; eval = center `76×76` crop.
- **Backbone:** `ResNet18Conv` (3→64 feature dim), BatchNorm replaced with **GroupNorm** (`groups = features//16`) for small-batch stability.
- **Pooling:** `SpatialSoftmax`, `num_kp=32`, temperature 1.0 → keypoint coords.
- **Per-port output feature dim:** 64; the encoder reports a **combined** `64` across the two ports (`output_feature_dim` returns `sum(D//N · N) = D`).

### 6.2 State (`ProjectionStateEncoder`, identity)

[state_encoder.py](oat/perception/state_encoder.py). Concatenated proprio + task id: `robot0_eef_pos(3) + robot0_eef_quat(4) + robot0_gripper_qpos(2) + task_uid(1) = 10`. `out_dim: null` → `nn.Identity()` (no params). Normalized per-port via the shared `LinearNormalizer` before fusion.

### 6.3 Fusion

Per step, modality features are **concatenated** along the feature axis: `cat([vision(64), state(10)], -1) = 74`. Output `[B, To=2, 74]`. `set_normalizer` fans out to both sub-encoders; missing per-port params log a warning and skip (graceful).

---

## 7. Autoregressive Policy Backbone: `AutoregressiveModel`

[transformer_cache.py](oat/model/autoregressive/transformer_cache.py). An encoder-decoder transformer that predicts the `L=8` latent tokens autoregressively, conditioned on obs features via **cross-attention**. Instantiated as:

```python
AutoregressiveModel(
    vocab_size = 1001,            # codebook_size(1000) + 1 (<BOS>)
    max_seq_len = 9,             # latent_horizon(8) + 1 (<BOS>)
    max_cond_len = 2,            # n_obs_steps
    cond_dim = 74,               # obs feature dim
    n_layer = 4, n_head = 4, n_emb = 256,
    p_drop_emb = 0.1, p_drop_attn = 0.1,
)
```

**Embeddings.** Token: `tok_emb = Embedding(1001, 256)`, weight-**tied** to the output `head`. Absolute learned positions `tok_pos_emb (1, 9, 256)` (zero-init). Condition: `cond_emb = Linear(74→256)`, `cond_pos_emb (1, 2, 256)`.

**Condition injection (cross-attention, not prefix).** `memory = encoder(cond_emb(cond) + cond_pos_emb)` where `encoder` is an MLP `256→1024→256` with **Mish**. Each of the 4 blocks then cross-attends queries (token stream) to this `memory` (keys/values). Obs and token sequence lengths are fully decoupled.

**Block structure (pre-norm, RMSNorm):** `x + CausalSelfAttn(RMSNorm(x))` → `x + CrossAttn(RMSNorm(x), memory)` → `x + MLP(RMSNorm(x))`. MLP is 4× GELU. `CausalSelfAttention`/`CrossAttention` use `bias=False` projections.

**Causal masking.** Self-attention is causal during the full-sequence training forward (`is_causal = (layer_past is None) and (T>1)`); cross-attention is never masked.

**KV-cache (generation only).** Cross-attention KV for `memory` is precomputed **once** (static across steps). Self-attention KV grows by one position per generated token. Training does **not** use the cache.

**Forward signature (training, used by the joint loss):**

```python
logits = model(tokens, cond)
# tokens: [B, 8]  (= action_tokens[:, :-1], i.e. [BOS, t0..t6])
# cond:   [B, 2, 74]
# logits: [B, 8, 1001]
```

In the joint forward the full sequence is `action_tokens = [BOS, t0…t7]` (`[B, 9]`); the model is fed `tokens[:, :-1]` (`[B, 8]`) and the CE target is `tokens[:, 1:]` (`[B, 8]`) — standard next-token shift.

---

## 8. Joint Training Objective

The exact loss from `OATPolicyJoint.forward` (returns a **dict**, unlike `OATPolicy.forward` which returns a scalar):

```python
total = recon_loss_weight * recon_loss  +  current_ce_weight * ce_loss
# shipped: recon_loss_weight = 1.0 ; current_ce_weight ∈ {0.0 warmup, 1.0 joint}
recon_loss = F.mse_loss( decoder(quant(encode(a))) , normalize(a) )      # normalized space
ce_loss    = F.cross_entropy( logits[B*8, 1001], detach(tokens)[B*8] )
return {'loss': total, 'recon_loss': recon_loss, 'ce_loss': ce_loss}
```

### 8.1 Precise gradient flow

| Loss term | Backprops into | Does **not** touch | Mechanism |
|---|---|---|---|
| `recon_loss` (MSE) | tokenizer **encoder** + **decoder** (+ decoder's `dropout_mask_token`, latent proj, heads) | backbone, obs encoder | differentiable via **FSQ STE** (`round_ste`); FSQ has 0 params |
| `ce_loss` (CE) | AR **backbone** + **obs encoder** | tokenizer (any part) | CE **targets are `tokens.detach().long()`** → stop-grad; indices are integer/non-diff anyway |

So **the tokenizer is shaped only by reconstruction**, and **the policy is shaped only by token prediction** — yet both updates happen in the same step on the same batch, letting the codebook drift in a direction the encoder/decoder still find reconstructable while the policy continuously re-targets the (slowly moving) detached tokens.

### 8.2 Why `ce_loss` is still **computed** during warmup

During the tokenizer-only warmup the workspace sets `current_ce_weight = 0.0`, so CE contributes nothing to `total`. **But the CE branch is still fully executed** (obs encoder forward, backbone forward, `F.cross_entropy`). This is deliberate ([train_oat_joint.py](oat/workspace/train_oat_joint.py) lines 187–194):

1. **DDP `find_unused_parameters=False`.** The accelerator is configured with `DistributedDataParallelKwargs(find_unused_parameters=False)`. If the backbone/obs-encoder params were absent from the autograd graph, DDP would raise an "unused parameters" error. Keeping CE in the graph (even ×0) guarantees every parameter participates in backward.
2. **`weight_decay = 0.0`.** With decay enabled, AdamW would shrink the unused backbone weights every step even with zero gradient; the joint config sets `weight_decay=0.0`, so multiplying CE by 0.0 truly freezes those params during warmup (no decay drift, zero gradient).

The `× 0.0` produces zero gradients for backbone/obs-encoder params, so they remain effectively frozen while still being valid graph nodes.

---

## 9. Tokenizer-Only Warmup Phase (`tokenizer_warmup_epochs = 50`)

[train_oat_joint.py](oat/workspace/train_oat_joint.py) lines 172–194. At the top of each epoch:

```python
tokenizer_warmup_epochs = cfg.training.get('tokenizer_warmup_epochs', 0)   # 50
is_warmup = self.epoch < tokenizer_warmup_epochs
current_ce_weight = 0.0 if is_warmup else unwrap(self.model).ce_loss_weight  # else 1.0
unwrap(self.model).set_ce_weight(current_ce_weight)                          # runtime toggle
```

- **Epochs 0–49:** `current_ce_weight = 0.0` → only `recon_loss` drives learning → the tokenizer (encoder/decoder) is trained to autoencode actions before the policy chases its tokens. Backbone/obs encoder receive zero-gradient (see §8.2).
- **Epoch ≥ 50:** `current_ce_weight = ce_loss_weight = 1.0` → full joint objective.

`set_ce_weight` simply assigns `self.current_ce_weight`; `ce_loss_weight` is the immutable configured target, `current_ce_weight` is the live value read inside `forward`. Setting `tokenizer_warmup_epochs: 0` disables warmup entirely.

---

## 10. Optimization

`OATPolicyJoint.get_optimizer` builds **six** AdamW param groups (three components × decay/no-decay), called with `**cfg.optimizer`:

| Group | Params | LR | weight_decay |
|---|---|---|---|
| policy decay | `model` params with `dim ≥ 2` | `policy_lr = 5e-5` | `weight_decay = 0.0` |
| policy no-decay | `model` params with `dim < 2` | `5e-5` | `0.0` |
| obs-enc decay | obs encoder `dim ≥ 2` | `obs_enc_lr = 1e-5` | `0.0` |
| obs-enc no-decay | obs encoder `dim < 2` | `1e-5` | `0.0` |
| tokenizer decay | tokenizer `dim ≥ 2` | `tokenizer_lr = 5e-5` | `0.0` |
| tokenizer no-decay | tokenizer `dim < 2` | `5e-5` | `0.0` |

- **AdamW betas:** `[0.9, 0.95]`. **weight_decay:** `0.0` everywhere in the shipped config (the decay/no-decay split only matters if `weight_decay > 0`: weights/embeddings `dim≥2` would decay, biases/norms `dim<1` would not).
- **Grad clip:** `max_grad_norm = 1.0` via `accelerator.clip_grad_norm_` (on `sync_gradients` steps).
- **LR schedule:** `constant` with `lr_warmup_steps = 100` (HF `get_scheduler`; constant schedule, stepped per-batch; `last_epoch = global_step − 1` for resume). Note: the `constant` scheduler does not actually consume `lr_warmup_steps`/`num_training_steps`, so the LR is effectively flat at the per-group base from step 0.
- **Mixed precision:** `bf16` if `allow_bf16=True` and `detect_bf16_support()`; forward under `accelerator.autocast()`.
- **EMA** ([ema_model.py](oat/model/diffusion/ema_model.py)): `update_after_step=0`, `inv_gamma=1.0`, `power=0.75`, `min_value=0.0`, `max_value=0.9999`. Decay `= 1 − (1 + step)^(−0.75)`, clamped ≤ 0.9999 (with `power=0.75`, ≈0.999 by ~10k steps). BatchNorm/frozen params copied directly; trainable params EMA'd. The EMA model also gets `set_normalizer`.
- **DDP:** `find_unused_parameters=False`, `InitProcessGroupKwargs(timeout=2h)`, `gradient_accumulation_steps = gradient_accumulate_every = 1`.

---

## 11. Training Loop / Workspace Walkthrough

[train_oat_joint.py](oat/workspace/train_oat_joint.py) `TrainOATJointWorkspace.run()`. `include_keys = ['global_step', 'epoch']` (checkpoint-serialized). Per epoch (`num_epochs = 5001`):

1. **Set CE weight** for the epoch (warmup logic, §9). `model.train()` (and `ema_model.train()`).
2. **Train pass** over `train_dataloader` (`batch_size 64`, `drop_last`, shuffled, 2 workers): under `accelerator.accumulate`, device-transfer batch → autocast forward → `accelerator.backward(out['loss'])`. Accumulate `[loss, recon, ce, bs]` weighted by batch size. On `sync_gradients`: grad clip 1.0 → `optimizer.step` → `zero_grad` → `lr_scheduler.step` → `ema.step`. Per-step wandb/json log: `train_loss, recon_loss, ce_loss, ce_weight, lr, global_step, epoch`.
3. **Epoch-average losses** via `accelerator.reduce(..., 'sum')` across ranks → `train_loss/recon_loss/ce_loss`.
4. **Eval setup:** `policy = unwrap(ema_model if use_ema else model)`, `policy.eval()`.
5. **Rollout** every `rollout_every = 50` epochs (main process, if not `lazy_eval`): `env_runner.run(policy)` → merges `mean_success_rate` etc. into `step_log`.
6. **Validation** every `val_every = 10` epochs: under `inference_mode`, forward `policy(batch)` over `val_dataloader`, reduce → `val_loss/val_recon_loss/val_ce_loss`.
7. **Reconstruction eval** every `sample_every = 10` epochs: per val batch compute
   - `tok_reconst_mse` = MSE of **tokenizer-only** autoencode `policy.action_tokenizer.autoencode(gt_action)` vs `gt_action` (raw action space);
   - `test_reconst_mse` = MSE of **full-pipeline** `policy.predict_action(obs)['action_pred']` vs `gt_action` (AR-generate → detokenize).
   These separate "can the tokenizer reconstruct?" from "can the policy predict tokens that reconstruct?".
8. **Checkpoint** every `checkpoint_every = 10` epochs (main process): unwrap model/ema, `save_checkpoint()` (`save_last_ckpt=True`), then **Top-3** by `mean_success_rate` (`mode=max`, `k=3`, `format_str='ep-{epoch:04d}_sr-{mean_success_rate:.3f}.ckpt'`) via `TopKCheckpointManager`. `save_last_snapshot=False`.
9. **End of epoch:** log `step_log`; `epoch += 1`, `global_step += 1`.

`resume: True` loads the latest checkpoint and exits early if `epoch ≥ num_epochs`.

---

## 12. Inference Path

`predict_action` is **inherited from `OATPolicy`** ([oatpolicy.py](oat/policy/oatpolicy.py) lines 170–216) — the joint policy overrides only `__init__`, `set_normalizer`, `set_ce_weight`, `get_optimizer`, and `forward`.

```
obs_dict ─► obs_encoder ─► features [B, 2, 74]
          start = [<BOS>=1000]  [B,1]
          model.generate(start, cond=features, max_new_tokens=8,
                         temperature=1.0, top_k=10)            # KV-cached AR
          ─► tokens [B, 9] ─► drop BOS ─► [B, 8] ─► clamp(0, 999)
          ─► action_tokenizer.detokenize(tokens)              # indices→embedding→decoder→unnormalize
          ─► action_pred [B, 32, 7]
          ─► action = action_pred[:, :16]   (receding horizon n_action_steps)
return {'action': [B,16,7], 'action_pred': [B,32,7]}
```

- **Generation** (`AutoregressiveModel.generate`): precompute cross-attn KV from `memory` once; process `<BOS>` prefix; then loop `max_new_tokens = 8` steps. Sampling: `temperature=1.0` (no scaling), **top-k=10** (mask all but top-10 logits, softmax, `multinomial`). `temperature=0` would be greedy; no `eos_id` here so it always emits exactly 8 tokens.
- **`clamp(0, bos_id-1)`** guards against the model emitting `<BOS>` (id 1000) as a code.
- **`detokenize`** ([tokenizer.py](oat/tokenizer/oat/tokenizer.py) lines 121–143): pad to `latent_horizon` if short, `FSQ.indices_to_embedding` → latents `[B,8,4]`, `decode(latents, eval_keep_k=token_lens)` → decoder (eval mode, `eval_keep_k=8`) → `unnormalize` → `[B,32,7]`.

---

## 13. Full Config Reference (`train_oat_joint.yaml`)

| Field | Value | Meaning |
|---|---|---|
| `defaults` | `_self_`, `task/policy: libero/libero10` | Hydra composition: this file + libero10 task |
| `name` | `train_oat_joint` | Run / config name |
| `_target_` | `...TrainOATJointWorkspace` | Workspace class instantiated |
| `seed` | `42` | Global + accelerate seed |
| `horizon` | `32` | Action chunk length `Ta` (= tokenizer `sample_horizon`) |
| `n_action_steps` | `16` | Receding-horizon actions executed per call |
| `n_obs_steps` | `2` | Obs history window `To` (= AR `max_cond_len`) |
| `task_name` | `${task.policy.name}` | Resolved task name (for logging) |
| `shape_meta` | `${task.policy.shape_meta}` | Obs/action shapes & types |
| `action_dim` | `${...action.shape[0]}` → `7` | Action feature dim |
| `latent_dim` | `eval len(levels)` → `4` | FSQ latent dim |
| **policy** | `OATPolicyJoint` | The joint policy |
| `policy.shape_meta/n_action_steps/n_obs_steps` | interpolated | Wiring |
| `policy.obs_encoder` | `FusedObservationEncoder` (`_recursive_: false`) | Vision+state fusion |
| `obs_encoder.vision_encoder` | `RobomimicRgbEncoder`, `crop_shape: [76,76]` | RGB encoder |
| `obs_encoder.state_encoder` | `ProjectionStateEncoder`, `out_dim: null` | Identity state proj |
| `policy.action_tokenizer` | `OATTok` (trainable, built from scratch) | FSQ autoencoder |
| `...encoder` | `RegisterEncoder`: `sample_dim 7, sample_horizon 32, emb_dim 256, head_dim 64, depth 2, pdropout 0.1, latent_dim 4, num_registers 8` | Action→8 latents |
| `...decoder` | `SinglePassDecoder`: `sample_dim 7, sample_horizon 32, emb_dim 256, head_dim 64, depth 4, pdropout 0.1, token_dropout_mode 'pow2', latent_dim 4, latent_horizon 8, use_causal_decoder true` | Latents→32 actions |
| `...quantizer` | `FSQ`: `levels [8,5,5,5]` | 1000-code quantizer |
| `policy.embed_dim` | `256` | AR backbone hidden dim |
| `policy.n_layers` | `4` | AR backbone blocks |
| `policy.n_heads` | `4` | AR backbone heads |
| `policy.dropout` | `0.1` | AR emb/attn dropout |
| `policy.temperature` | `1.0` | Sampling temperature |
| `policy.topk` | `10` | Top-k sampling |
| `policy.recon_loss_weight` | `1.0` | Weight on MSE recon term |
| `policy.ce_loss_weight` | `1.0` | Target weight on CE term (post-warmup) |
| `training.resume` | `True` | Resume from latest ckpt |
| `training.allow_bf16` | `True` | Enable bf16 if HW supports |
| `training.max_grad_norm` | `1.0` | Grad-norm clip |
| `training.num_epochs` | `5001` | Total epochs |
| `training.tokenizer_warmup_epochs` | `50` | Tokenizer-only warmup (CE weight 0) |
| `training.val_every` | `10` | Validation cadence (epochs) |
| `training.sample_every` | `10` | Recon-eval cadence (epochs) |
| `training.rollout_every` | `50` | Env rollout cadence (epochs) |
| `training.checkpoint_every` | `10` | Checkpoint cadence (epochs) |
| `training.lr_scheduler` | `constant` | Flat LR schedule |
| `training.lr_warmup_steps` | `100` | Warmup steps (unused by `constant`) |
| `training.gradient_accumulate_every` | `1` | Grad-accum steps |
| `training.max_train_steps/max_val_steps/max_reconst_steps` | `null` | No per-epoch step caps |
| `training.seed` | `${seed}` → 42 | Training seed |
| `training.use_ema` | `True` | Maintain EMA model |
| `training.num_demo` | `500` | # demos (zarr path / run name) |
| `training.tqdm_interval_sec` | `1.0` | Progress-bar refresh |
| **ema** | `EMAModel` | `update_after_step 0, inv_gamma 1.0, power 0.75, min 0.0, max 0.9999` |
| **optimizer** | — | `policy_lr 5e-5, obs_enc_lr 1e-5, tokenizer_lr 5e-5, weight_decay 0.0, betas [0.9,0.95]` |
| **dataloader** | — | `batch_size 64, num_workers 2, shuffle True, pin_memory True, persistent_workers True, drop_last True` |
| **val_dataloader** | — | same but `shuffle False` |
| **logging** | wandb `oat_dev` | `mode online`, run name `${now}_..._N500` |
| **checkpoint.topk** | — | `monitor_key mean_success_rate, mode max, k 3, format ep-{epoch:04d}_sr-{...}.ckpt` |
| `checkpoint.save_last_ckpt` | `True` | Always save `latest.ckpt` |
| `checkpoint.save_last_snapshot` | `False` | No full-workspace snapshot |
| **multi_run / hydra** | — | Output dir `output/${now:%Y%m%d}/${now:%H%M%S}_...` |

---

## 14. File Map

| Path | Role |
|---|---|
| [oat/config/train_oat_joint.yaml](oat/config/train_oat_joint.yaml) | Joint-training Hydra config (entry config) |
| [oat/workspace/train_oat_joint.py](oat/workspace/train_oat_joint.py) | `TrainOATJointWorkspace` — full training loop, warmup, eval, checkpointing |
| [oat/policy/oat_policy_joint.py](oat/policy/oat_policy_joint.py) | `OATPolicyJoint` — trainable tokenizer + AR backbone; joint forward & 6-group optimizer |
| [oat/policy/oatpolicy.py](oat/policy/oatpolicy.py) | `OATPolicy` base (frozen-tokenizer variant); `predict_action` inherited by joint |
| [oat/policy/base_policy.py](oat/policy/base_policy.py) | `BasePolicy` (device prop, dummy obs) |
| [oat/tokenizer/oat/tokenizer.py](oat/tokenizer/oat/tokenizer.py) | `OATTok` — encode/decode/tokenize/detokenize/autoencode, normalizer |
| [oat/tokenizer/base_tokenizer.py](oat/tokenizer/base_tokenizer.py) | `BaseTokenizer` abstract interface + `from_checkpoint` |
| [oat/tokenizer/oat/encoder/register_encoder.py](oat/tokenizer/oat/encoder/register_encoder.py) | `RegisterEncoder` — actions→8 register latents |
| [oat/tokenizer/oat/decoder/single_pass_decoder.py](oat/tokenizer/oat/decoder/single_pass_decoder.py) | `SinglePassDecoder` — latents→32 actions, causal queries |
| [oat/tokenizer/oat/quantizer/fsq.py](oat/tokenizer/oat/quantizer/fsq.py) | `FSQ` — parameter-free finite scalar quantizer (STE) |
| [oat/tokenizer/oat/model/token_dropout.py](oat/tokenizer/oat/model/token_dropout.py) | `MaskedNestedDropout` — `pow2` coarse-to-fine latent dropout |
| [oat/tokenizer/oat/model/](oat/tokenizer/oat/model/) | OAT transformer block library (attn/MLP/norm/pos-emb/heads) |
| [oat/model/autoregressive/transformer_cache.py](oat/model/autoregressive/transformer_cache.py) | `AutoregressiveModel` — KV-cached cross-attn AR backbone + `generate` |
| [oat/perception/fused_obs_encoder.py](oat/perception/fused_obs_encoder.py) | `FusedObservationEncoder` — vision+state fusion (dim 74) |
| [oat/perception/robomimic_vision_encoder.py](oat/perception/robomimic_vision_encoder.py) | `RobomimicRgbEncoder` — crop + ResNet18 + SpatialSoftmax |
| [oat/perception/state_encoder.py](oat/perception/state_encoder.py) | `ProjectionStateEncoder` — identity state projection |
| [oat/model/common/normalizer.py](oat/model/common/normalizer.py) | `LinearNormalizer` — fit/normalize/unnormalize (limits→[-1,1]) |
| [oat/model/common/lr_scheduler.py](oat/model/common/lr_scheduler.py) | `get_scheduler` — HF scheduler factory (`constant`) |
| [oat/model/diffusion/ema_model.py](oat/model/diffusion/ema_model.py) | `EMAModel` — exponential moving average of weights |
| [oat/workspace/base_workspace.py](oat/workspace/base_workspace.py) | `BaseWorkspace` — checkpoint/snapshot save/load |
| [oat/common/checkpoint_util.py](oat/common/checkpoint_util.py) | `TopKCheckpointManager` — best-K checkpoint retention |
| [oat/dataset/zarr_dataset.py](oat/dataset/zarr_dataset.py) | `ZarrDataset` — libero10 sequence sampling + normalizer fit |
| [oat/config/task/policy/libero/libero10.yaml](oat/config/task/policy/libero/libero10.yaml) | libero10 task: shape_meta, dataset, env_runner |

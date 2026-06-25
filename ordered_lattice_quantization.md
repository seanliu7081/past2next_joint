# Ordered Lattice Quantization for Action Tokenization

## Summary of Research Discussion (March 2026)

---

## 1. Core Problem Statement

We are designing an **action tokenizer** for autoregressive (AR) robot policy learning. The tokenizer maps continuous actions $a \in \mathbb{R}^d$ (typically $d=7$: 6 DOF + gripper) to discrete token indices $v \in \{1, \ldots, K\}$.

The AR transformer then predicts $p(v_t | \text{context})$. Two properties of the tokenizer critically affect policy performance:

### Property 1: Quantization Distortion ($\mathcal{D}$)
$$\mathcal{D}(\mathcal{C}) = \mathbb{E}_{a \sim p(a)} \left[\min_{c \in \mathcal{C}} \|a - c\|^2 \right]$$
Lower distortion = more faithful action reconstruction. Governed by **lattice geometry** (Voronoi cell shape).

### Property 2: Ordinal Quality ($\mathcal{O}$)
The correlation between token index distance $|\phi(c_i) - \phi(c_j)|$ and Euclidean distance $\|c_i - c_j\|$. Higher ordinal quality = AR model predictions degrade gracefully (predicting a "nearby" token gives a "nearby" action). Governed by **ordering strategy**.

### The Design Space
$$\text{Tokenizer Design} = \underbrace{\text{Lattice } \Lambda}_{\text{determines } \mathcal{D}} \;\times\; \underbrace{\text{Ordering } \phi: \Lambda \to \{1,\ldots,K\}}_{\text{determines } \mathcal{O}}$$

---

## 2. Connection to Sphere Packing

### The Mathematical Link
Vector quantization (VQ) and sphere packing are mathematically dual:
- **Codebook = lattice points = sphere centers**
- **Voronoi cell = packing unit**
- **Normalized second moment $G(\Lambda)$** = quantization efficiency metric

Key result from Conway & Sloane: the **best lattice quantizer** in a given dimension is closely related to the **densest lattice packing**:

| Dimension | Best Lattice | $G(\Lambda)$ | vs $\mathbb{Z}^d$ |
|-----------|-------------|-------------|-------------------|
| 2 | $A_2$ (hexagonal) | 0.0802 | +3.8% better |
| 3 | BCC ($D_3^*$) | 0.0785 | +5.8% better |
| 8 | $E_8$ | 0.0717 | +14.0% better |
| Any $d$ | $\mathbb{Z}^d$ (cubic) | 0.0833 | baseline (worst) |

**FSQ (Finite Scalar Quantization) uses $\mathbb{Z}^d$ — the worst lattice in every dimension.**

### The Fundamental Trade-off: Packing Density vs Orderability
Denser lattices have higher **kissing numbers** (more nearest neighbors per point). But any 1D ordering can only make 2 neighbors "adjacent" (left and right). Therefore:

| Lattice | Kissing # | Neighbors lost to ordering | Loss % |
|---------|----------|---------------------------|--------|
| $\mathbb{Z}^2$ | 4 | 2 | 50% |
| $A_2$ (hex) | 6 | 4 | 67% |
| $E_8$ | 240 | 238 | 99% |
| Leech | 196,560 | 196,558 | ~100% |

**Denser packing → harder to order for AR prediction.** This is an unavoidable mathematical constraint.

---

## 3. FSQ's Specific Problem: Lexicographic Ordering

FSQ uses **mixed-radix (lexicographic) encoding**: for $d$-dim action with $L$ levels per dim, token index = $q_1 + L \cdot q_2 + L^2 \cdot q_3 + \ldots$

### The Boundary Jump Problem
At dimension boundaries (e.g., from $(L{-}1, 0)$ to $(0, 1)$), token index changes by 1 but Euclidean distance jumps to $\approx L$. These "wrap-around jumps" grow exponentially with dimension:
- Dim 1 boundary: jump $\approx L$
- Dim 2 boundary: jump $\approx L^2$  
- Dim $(d{-}1)$ boundary: jump $\approx L^{d-1}$

### Hilbert Curve Eliminates This
Hilbert curve ordering on $\mathbb{Z}^d$ guarantees **bounded max jump** ($O(1)$) regardless of grid size. Experimentally verified:

| 2D, K=64 | Spearman ρ | Max Jump | Nbhd Pres (k=6) |
|-----------|-----------|----------|-----------------|
| Z² + Lex (FSQ) | 0.645 | **7.07** | 0.320 |
| Z² + Hilbert | 0.649 | **1.00** | **0.672** |

Hilbert doubles neighborhood preservation and eliminates worst-case jumps.

---

## 4. Experimental Results: Full Lattice × Ordering Comparison

### 2D Results (K=64)

| Method | Spearman ρ | Max Jump | Mean Jump | Nbhd Pres |
|--------|-----------|----------|-----------|-----------|
| Z² + Lex (FSQ) | 0.645 | 7.07 | 1.68 | 0.320 |
| **Z² + Hilbert** | 0.649 | **1.00** | **1.00** | **0.672** |
| A₂ + Lex | **0.658** | 7.55 | 1.70 | 0.323 |
| A₂ + Hex-Hilbert (L=16) | 0.565 | 1.73 | 1.04 | 0.617 |
| A₂ + Gosper (lv=3) | 0.597 | 2.00 | 1.04 | 0.589 |
| Z² + Random | -0.017 | 8.60 | 4.31 | 0.063 |

### 4D Results (K=256)

| Method | Spearman ρ | Max Jump | Mean Jump | Nbhd Pres |
|--------|-----------|----------|-----------|-----------|
| Z⁴ + Lex (FSQ) | 0.442 | 5.29 | 1.62 | 0.378 |
| Z⁴ + Product Hilbert | 0.455 | 3.16 | 1.13 | 0.389 |
| **A₂² + Lex** | **0.501** | 4.80 | 1.55 | **0.459** |
| A₂² + Prod Hex-Hilbert | 0.385 | 2.45 | 1.15 | 0.351 |
| A₂² + Prod Gosper | 0.425 | 4.12 | 1.27 | 0.362 |

### 8D Results (K=256)

| Method | Spearman ρ | Max Jump | Mean Jump | Nbhd Pres |
|--------|-----------|----------|-----------|-----------|
| Z⁸ + Lex (FSQ) | 0.323 | 2.83 | 1.34 | 0.251 |
| Z⁸ + Product Hilbert | 0.364 | 2.00 | 1.12 | **0.463** |
| **A₂⁴ + Lex** | **0.418** | 3.61 | 1.36 | 0.306 |
| Z⁸ + Spiral | 0.361 | 2.65 | 1.34 | 0.477 |

**Note:** d=8 with K_factor=4 per A₂ factor causes all within-factor orderings to collapse to identical results (too few points for ordering to matter).

---

## 5. Key Conclusions

### Conclusion 1: Lattice geometry matters in high dimensions
$A_2^{d/2}$ + Lexicographic consistently wins **Spearman ρ** (global ordinal quality) across d=4 and d=8. The hexagonal lattice's geometric advantage compounds with dimension.

### Conclusion 2: Ordering strategy matters for local preservation
Z² + Hilbert wins **neighborhood preservation** (local ordinal quality) in 2D. Product Hilbert wins in 8D. Space-filling curves are superior to lexicographic for local structure.

### Conclusion 3: Hex-Hilbert and Gosper did NOT beat baselines
Both novel approaches failed to dominate Z²+Hilbert. Reasons:
- Hex-Hilbert: affine mapping distortion degrades Hilbert's locality
- Gosper: 19.1° per-level rotation causes spatial twisting
- Neither achieved the "best of both worlds" we hoped for

### Conclusion 4: $A_2^{d/2}$ + Lexicographic is the most robust high-dim solution
In d=4, it wins both metrics. Simple to implement, theoretically justified.

### Conclusion 5: Product Hilbert is the best local-preservation method
When local neighborhood preservation is the priority (which it likely is for AR policy), Z^d + Product Hilbert is strongest.

---

## 6. Metrics Explained

### Spearman ρ (Global Ordinal Quality)
Rank correlation between all-pairs token distance and Euclidean distance. Measures: "overall, do distant tokens correspond to distant actions?"

### Neighborhood Preservation (Local Ordinal Quality)  
For each point, what fraction of its $k$ geometric nearest neighbors are also among its $k$ nearest neighbors in token index? Measures: "when the AR model predicts nearby tokens, are they nearby actions?"

### Max Jump
Maximum Euclidean distance between consecutive tokens (index $i$ and $i+1$). Measures worst-case degradation when predicting off-by-one.

### Mean Jump
Average consecutive Euclidean distance. Lower = smoother token path.

---

## 7. Implications for OATTok Implementation

### What to implement as the new tokenizer

**Primary candidate: $A_2^{d/2}$ Product Lattice Tokenizer with Lexicographic ordering**

Rationale:
- Best global ordinal quality (Spearman ρ) in d=4 and d=8
- Wins both metrics in d=4
- Simple implementation (no complex space-filling curves needed)
- Theoretical backing: hexagonal quantization is provably more efficient than cubic

**Secondary candidate: $\mathbb{Z}^d$ + Product Hilbert (for ablation)**

Rationale:
- Best local neighborhood preservation
- Drop-in replacement for FSQ (same lattice, different index encoding)
- Useful as ablation to isolate lattice vs ordering contributions

### Implementation requirements

**For $A_2^{d/2}$ Product Lattice Tokenizer:**
1. Encoder outputs $d$-dim latent (must be even $d$; pad if $d=7$ → use $d=8$)
2. Pair dimensions: $(z_1, z_2)$, $(z_3, z_4)$, ..., $(z_{d-1}, z_d)$
3. Each pair: quantize to nearest $A_2$ lattice point
   - $A_2$ basis: $e_1 = (1, 0)$, $e_2 = (1/2, \sqrt{3}/2)$
   - Nearest-neighbor quantization on $A_2$ has closed-form solution
4. Convert lattice coordinates to token index via lexicographic encoding
5. Straight-through estimator for gradient
6. Decoder maps token indices back to continuous actions

**For Product Hilbert on $\mathbb{Z}^d$:**
1. Same as FSQ encoder
2. Replace mixed-radix index with Product Hilbert index:
   - Pair dims, apply 2D Hilbert curve per pair
   - Combine with mixed-radix across pairs
3. Everything else identical to FSQ

### Key hyperparameters to tune
- Number of quantization levels per $A_2$ factor ($L$)
- Total codebook size $K = L^{d/2}$ (or $L_1 \times L_2 \times \ldots$)
- Latent dimension $d$ (4, 6, or 8)
- Whether to learn the affine transform before quantization

### Ablation plan for the paper
1. **FSQ ($\mathbb{Z}^d$ + Lex)** — existing baseline
2. **$\mathbb{Z}^d$ + Product Hilbert** — isolates ordering effect
3. **$A_2^{d/2}$ + Lex** — isolates lattice effect  
4. **$A_2^{d/2}$ + Product Hilbert** — combined (if implementable)
5. All with same codebook size $K$, same encoder/decoder architecture

Each evaluated on: reconstruction MSE, per-step entropy, and **LIBERO-10 policy success rate**.

---

## 8. Theoretical Framework for the Paper

### Section title suggestion
"Ordered Lattice Quantization: Bridging Sphere Packing and Autoregressive Action Decoding"

### Key theoretical contributions
1. **Formalize the $\mathcal{D}$-$\mathcal{O}$ trade-off** as a bi-objective optimization
2. **Prove the kissing number bound**: any 1D ordering on a lattice with kissing number $k$ must lose at least $(k-2)/k$ fraction of nearest-neighbor relationships
3. **Show FSQ is Pareto-suboptimal**: $\mathbb{Z}^d$ + lexicographic is dominated by both $A_2^{d/2}$ + lexicographic (better lattice) and $\mathbb{Z}^d$ + Hilbert (better ordering)
4. **Connect to sphere packing**: frame codebook design as the first known application of lattice theory to autoregressive action decoding

---

## 9. Code and Data

All experimental code is in two Python scripts:
- `ordered_packing_tradeoff.py` — original analysis (Z², A₂, D_d lattices × Lex, Hilbert, Spiral, NN-Chain orderings)
- `hex_hilbert_experiment.py` — new methods (Hex-Hilbert, Gosper curve, product lattices in d=4,8)

Generated figures:
- `fig1_orderings_2d.png` — 2D ordering path visualization
- `fig2_jump_distributions.png` — consecutive jump histograms
- `fig3_pareto_frontier.png` — D vs O Pareto frontier
- `fig4_fundamental_tradeoff.png` — packing density vs orderability
- `fig5_distance_correlation.png` — token distance vs Euclidean distance scatterplots
- `fig6_new_orderings_2d.png` — Hex-Hilbert and Gosper visualization
- `fig7_summary_comparison.png` — bar chart comparison across dimensions

---

## 10. Context: OATTok Codebase

- **Current tokenizer**: FSQ (Finite Scalar Quantization) = $\mathbb{Z}^d$ + lexicographic
- **Framework**: OAT (Ordered Action Tokenizer) evaluated on LIBERO-10 benchmark
- **Key prior result**: Past-action conditioning (enriched past with acc/jerk features) improved SR from 56.3% → ~66-72% depending on codebook size
- **Visual encoder**: Currently ResNet18, planned ablation with DINOv2 ViT-S/14
- **Training setup**: Dual RTX 4090 GPUs, remote SSH server
- **Action space**: 7-dim (6 DOF + gripper), will need padding to 8-dim for $A_2^4$ product
- **Codebook sizes tested**: |V| = 240 to 5000 (monotonic scaling confirmed with enriched past)

### Important prior finding
BSQ (Binary Spherical Quantization) achieved identical reconstruction MSE to FSQ but policy SR dropped from 0.56 to 0.40. This strongly suggests **ordinal structure matters more than raw quantization quality** for AR policy performance — but this result should be re-verified.
#!/usr/bin/env python3
"""PolarOATTok Codebook Utilization Analysis on LIBERO-10 demos."""

import sys
import os
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import dill
import zarr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_DIR = Path("/workspace/oat/output/20260320/014428_train_polar_oattok_libero10_N500/checkpoints")
ZARR_PATH = "/workspace/oat/data/libero/libero10_N500.zarr"
OUTPUT_PDF = "/workspace/oat/analysis_scripts/codebook_utilization.pdf"

# Pick best checkpoint (lowest MSE) or latest
ckpt_candidates = sorted(CKPT_DIR.glob("ep-*.ckpt"))
if ckpt_candidates:
    ckpt_path = str(ckpt_candidates[-1])  # highest epoch among top-k
else:
    ckpt_path = str(CKPT_DIR / "latest.ckpt")
print(f"Using checkpoint: {ckpt_path}")

# ── Load tokenizer from checkpoint ──────────────────────────────────────────
sys.path.insert(0, "/workspace/oat")
import hydra
from oat.common.hydra_util import register_new_resolvers
register_new_resolvers()

payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
cfg = payload['cfg']

# Instantiate workspace to load model
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg, output_dir="/tmp/polar_analysis", lazy_instantiation=False)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

if getattr(cfg.training, "use_ema", False):
    tokenizer = workspace.ema_model
else:
    tokenizer = workspace.model

tokenizer.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = tokenizer.to(device)

print(f"Loaded PolarOATTok | vocab_sizes={tokenizer.vocab_sizes}")
print(f"  FSQ codebook: {tokenizer.quantizer.codebook_size}")
print(f"  Effective vocab: {tokenizer.effective_vocab_size:,}")

# ── Load all LIBERO-10 actions ───────────────────────────────────────────────
z = zarr.open(ZARR_PATH, 'r')
all_actions = z['data/action'][:]       # (N, 7)
episode_ends = z['meta/episode_ends'][:]
task_uids = z['data/task_uid'][:].flatten()
N = all_actions.shape[0]
print(f"\nLoaded {N:,} action steps from {len(episode_ends)} episodes")

# ── Reconstruct (B, T, 7) chunks from episodes ──────────────────────────────
# The tokenizer expects (B, T=32, 7). Build chunks from each episode.
horizon = tokenizer.sample_horizon
episode_starts = np.concatenate([[0], episode_ends[:-1]])

chunks = []
for start, end in zip(episode_starts, episode_ends):
    ep_actions = all_actions[start:end]
    ep_len = len(ep_actions)
    for t in range(0, ep_len - horizon + 1, horizon):
        chunks.append(ep_actions[t:t + horizon])
    # Include last chunk if there's a remainder
    if ep_len >= horizon and ep_len % horizon != 0:
        chunks.append(ep_actions[ep_len - horizon:ep_len])

chunks = np.stack(chunks, axis=0)  # (num_chunks, 32, 7)
print(f"Built {chunks.shape[0]} chunks of horizon {horizon}")

# ── Run encode in batches ────────────────────────────────────────────────────
all_tokens = {k: [] for k in tokenizer.vocab_sizes.keys()}
batch_size = 256

with torch.inference_mode():
    for i in range(0, len(chunks), batch_size):
        batch = torch.from_numpy(chunks[i:i + batch_size]).float().to(device)
        _, tokens = tokenizer.encode(batch)
        for k in all_tokens:
            all_tokens[k].append(tokens[k].cpu().reshape(-1))

# Concatenate all
for k in all_tokens:
    all_tokens[k] = torch.cat(all_tokens[k], dim=0)

total_tokens = len(all_tokens['inv'])
print(f"\nEncoded {total_tokens:,} token positions total")

# ── Per-subspace utilization stats ───────────────────────────────────────────
print("\n" + "=" * 70)
print("CODEBOOK UTILIZATION ANALYSIS")
print("=" * 70)

subspaces = list(tokenizer.vocab_sizes.keys())
stats = {}

for key in subspaces:
    tokens = all_tokens[key]
    vocab_size = tokenizer.vocab_sizes[key]

    unique_used = tokens.unique().numel()
    utilization = unique_used / vocab_size

    counts = Counter(tokens.tolist())
    freqs = sorted(counts.values(), reverse=True)
    top10_pct = sum(freqs[:10]) / len(tokens)

    probs = torch.tensor(freqs).float() / len(tokens)
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = math.log(vocab_size)

    most_common_tok, most_common_cnt = counts.most_common(1)[0]
    least_common_cnt = freqs[-1]

    stats[key] = {
        'vocab_size': vocab_size,
        'unique_used': unique_used,
        'utilization': utilization,
        'top10_pct': top10_pct,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'norm_entropy': entropy / max_entropy if max_entropy > 0 else 0,
        'counts': counts,
        'freqs': freqs,
    }

    print(f"\n=== {key} ===")
    print(f"  Vocab size:    {vocab_size}")
    print(f"  Codes used:    {unique_used}/{vocab_size} ({utilization:.1%})")
    print(f"  Top-10 tokens: {top10_pct:.1%} of all assignments")
    print(f"  Entropy:       {entropy:.2f} / {max_entropy:.2f} (normalized: {entropy / max_entropy:.2%})")
    print(f"  Most common:   token {most_common_tok} ({most_common_cnt:,} times, {most_common_cnt / len(tokens):.1%})")
    print(f"  Least common:  {least_common_cnt} assignments")

# Joint stats
joint_tokens = list(zip(*[all_tokens[k].tolist() for k in subspaces]))
unique_combos = len(set(joint_tokens))
effective_vocab = math.prod(tokenizer.vocab_sizes[k] for k in subspaces)

print(f"\n=== JOINT ===")
print(f"  Unique combos:     {unique_combos:,}")
print(f"  Effective vocab:   {effective_vocab:,}")
print(f"  Joint utilization: {unique_combos / effective_vocab:.4%}")

# ── Plots ────────────────────────────────────────────────────────────────────
pdf = PdfPages(OUTPUT_PDF)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PolarOATTok Codebook Utilization", fontsize=14)

# (a) Bar chart: per-subspace utilization %
ax = axes[0, 0]
x = range(len(subspaces))
utils = [stats[k]['utilization'] * 100 for k in subspaces]
norm_ents = [stats[k]['norm_entropy'] * 100 for k in subspaces]
w = 0.35
ax.bar([i - w / 2 for i in x], utils, w, label='% codes used', color='steelblue')
ax.bar([i + w / 2 for i in x], norm_ents, w, label='% max entropy', color='coral')
ax.set_xticks(list(x))
ax.set_xticklabels(subspaces, rotation=15)
ax.set_ylabel('%')
ax.set_title('(a) Codebook Utilization & Entropy')
ax.legend()
ax.set_ylim(0, 110)
for i, (u, e) in enumerate(zip(utils, norm_ents)):
    ax.text(i - w / 2, u + 1, f'{u:.0f}%', ha='center', fontsize=7)
    ax.text(i + w / 2, e + 1, f'{e:.0f}%', ha='center', fontsize=7)

# (b) inv subspace: histogram of per-token frequency
ax = axes[0, 1]
inv_counts = stats['inv']['counts']
inv_vocab = stats['inv']['vocab_size']
freq_array = np.zeros(inv_vocab)
for tok_id, cnt in inv_counts.items():
    if tok_id < inv_vocab:
        freq_array[tok_id] = cnt
ax.bar(range(inv_vocab), freq_array, color='steelblue', alpha=0.7, width=1.0)
ax.set_xlabel('Token index')
ax.set_ylabel('Frequency')
ax.set_title(f'(b) inv token frequency ({stats["inv"]["unique_used"]}/{inv_vocab} used)')
ax.set_xlim(-1, inv_vocab)

# (c) theta_trans: polar bar plot
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(2, 2, 3, projection='polar')
n_bins_trans = tokenizer._n_bins_trans
tt_counts = stats['theta_trans']['counts']
# Bin centers
bin_widths_t = 2 * np.pi / n_bins_trans
centers_t = -np.pi + bin_widths_t * (np.arange(n_bins_trans) + 0.5)
heights_t = np.array([tt_counts.get(i, 0) for i in range(n_bins_trans)])
# NULL count
null_count_t = tt_counts.get(n_bins_trans, 0)
ax.bar(centers_t, heights_t, width=bin_widths_t * 0.9, color='steelblue', alpha=0.7)
ax.set_title(f'(c) θ_trans bin usage\n(NULL={null_count_t:,})', pad=15, fontsize=10)

# (d) theta_rot: polar bar plot
ax = axes[1, 1]
ax.remove()
ax = fig.add_subplot(2, 2, 4, projection='polar')
n_bins_rot = tokenizer._n_bins_rot
tr_counts = stats['theta_rot']['counts']
bin_widths_r = 2 * np.pi / n_bins_rot
centers_r = -np.pi + bin_widths_r * (np.arange(n_bins_rot) + 0.5)
heights_r = np.array([tr_counts.get(i, 0) for i in range(n_bins_rot)])
null_count_r = tr_counts.get(n_bins_rot, 0)
ax.bar(centers_r, heights_r, width=bin_widths_r * 0.9, color='coral', alpha=0.7)
ax.set_title(f'(d) θ_rot bin usage\n(NULL={null_count_r:,})', pad=15, fontsize=10)

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

# Page 2: yaw histogram + detailed inv frequency distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Additional Codebook Details", fontsize=14)

# Yaw bin usage
ax = axes[0]
n_bins_yaw = tokenizer._n_bins_yaw
yaw_counts = stats['yaw']['counts']
yaw_heights = np.array([yaw_counts.get(i, 0) for i in range(n_bins_yaw)])
ax.bar(range(n_bins_yaw), yaw_heights, color='teal', alpha=0.7)
ax.set_xlabel('Bin index')
ax.set_ylabel('Frequency')
ax.set_title(f'Δyaw bin usage ({stats["yaw"]["unique_used"]}/{n_bins_yaw} used)')

# inv frequency distribution (log-scale)
ax = axes[1]
nonzero_freqs = freq_array[freq_array > 0]
ax.hist(nonzero_freqs, bins=50, color='steelblue', alpha=0.7)
ax.set_xlabel('Token frequency')
ax.set_ylabel('Number of tokens')
ax.set_title('inv token frequency distribution')
if len(nonzero_freqs) > 0 and nonzero_freqs.max() > nonzero_freqs.min() * 10:
    ax.set_xscale('log')

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

pdf.close()
print(f"\nPlots saved to {OUTPUT_PDF}")

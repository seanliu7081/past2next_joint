"""
Diagnose WHY codebook utilization is low in SpectralBasisEncoder.

Collects pre-FSQ latent vectors and visualizes their distribution
to understand why only 13-25% of FSQ codes are being used.

Usage:
  python experiments/diagnose_latent_distribution.py \
    --tokenizer-ckpt path/to/spectral_tok.ckpt \
    --tokenizer-ckpt-b path/to/register_tok.ckpt \
    --zarr-path data/libero/libero10_N500.zarr \
    -o output/latent_diagnosis
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import torch
import numpy as np
import json
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from oat.tokenizer.oat.tokenizer import OATTok
from oat.dataset.zarr_dataset import ZarrDataset


def build_tokenizer_dataloader(zarr_path, batch_size=256, horizon=32, seed=42):
    """Build dataset for tokenizer evaluation (no obs, no past_action)."""
    dataset = ZarrDataset(
        zarr_path=zarr_path,
        obs_keys=[],
        action_key="action",
        n_obs_steps=0,
        n_action_steps=horizon,
        seed=seed,
        val_ratio=0.1,
    )
    val_dataset = dataset.get_validation_dataset()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return dataset, val_dataset, loader


@torch.no_grad()
def collect_latent_stats(tokenizer, dataloader, device, max_batches=50):
    """
    Collect pre-FSQ and post-FSQ latent vectors.

    Hooks into the tokenizer to capture:
      1. encoder output (pre-FSQ): continuous latents
      2. FSQ output: quantized latents
      3. token IDs

    Returns dict with numpy arrays.
    """
    tokenizer.eval()
    tokenizer.to(device)

    all_pre_fsq = []    # [N, K, latent_dim]  continuous
    all_post_fsq = []   # [N, K, latent_dim]  quantized
    all_tokens = []     # [N, K]              token IDs
    all_coeffs = []     # [N, K, action_dim]  spectral coeffs (if available)

    has_spectral = hasattr(tokenizer.encoder, 'basis')

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting latents")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        actions = batch['action'].to(device)
        nactions = tokenizer.normalizer['action'].normalize(actions)

        # --- encoder forward (get pre-FSQ latents) ---
        latents_pre = tokenizer.encoder(nactions)  # [B, K, latent_dim]
        all_pre_fsq.append(latents_pre.cpu())

        # --- also collect spectral coefficients if available ---
        if has_spectral:
            coeffs = torch.einsum("kt, btd -> bkd",
                                  tokenizer.encoder.basis, nactions)
            all_coeffs.append(coeffs.cpu())

        # --- FSQ forward ---
        latents_post, tokens = tokenizer.quantizer(latents_pre)
        all_post_fsq.append(latents_post.cpu())
        all_tokens.append(tokens.cpu())

    result = {
        'pre_fsq': torch.cat(all_pre_fsq, dim=0).numpy(),      # [N, K, D]
        'post_fsq': torch.cat(all_post_fsq, dim=0).numpy(),     # [N, K, D]
        'tokens': torch.cat(all_tokens, dim=0).numpy(),          # [N, K]
    }
    if all_coeffs:
        result['coeffs'] = torch.cat(all_coeffs, dim=0).numpy()  # [N, K, 7]

    return result


def analyze_distribution(stats, name=""):
    """Print detailed statistics about latent distribution."""
    pre_fsq = stats['pre_fsq']   # [N, K, D]
    tokens = stats['tokens']      # [N, K]
    N, K, D = pre_fsq.shape

    print(f"\n{'='*70}")
    print(f"  LATENT DISTRIBUTION ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"  Samples: {N}, Tokens/sample: {K}, Latent dim: {D}")

    # --- Per-position, per-dim statistics ---
    print(f"\n  Pre-FSQ latent stats (per position, per dim):")
    print(f"  {'Pos':>3} {'Dim':>3} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|>2|%':>8}")
    for k in range(K):
        for d in range(D):
            vals = pre_fsq[:, k, d]
            mean = vals.mean()
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            pct_outside = (np.abs(vals) > 2.0).mean() * 100
            print(f"  {k:>3} {d:>3} {mean:>8.3f} {std:>8.3f} {vmin:>8.3f} {vmax:>8.3f} {pct_outside:>7.1f}%")
        if k < K - 1:
            print(f"  {'---':>3}")

    # --- Per-position codebook usage ---
    print(f"\n  Per-position codebook usage:")
    max_token = tokens.max() + 1
    for k in range(K):
        unique = len(np.unique(tokens[:, k]))
        print(f"    pos {k}: {unique}/{max_token} codes used "
              f"({unique/max_token*100:.1f}%)")

    # --- Per-position token distribution entropy ---
    print(f"\n  Per-position top-5 most frequent tokens:")
    for k in range(K):
        vals, counts = np.unique(tokens[:, k], return_counts=True)
        sorted_idx = np.argsort(-counts)[:5]
        total = counts.sum()
        top5 = [(int(vals[i]), counts[i]/total*100) for i in sorted_idx]
        top5_str = ", ".join([f"{v}({p:.1f}%)" for v, p in top5])
        top5_cumulative = sum(p for _, p in top5)
        print(f"    pos {k}: [{top5_str}] (top-5 covers {top5_cumulative:.1f}%)")

    # --- Spectral coefficient stats (if available) ---
    if 'coeffs' in stats:
        coeffs = stats['coeffs']  # [N, K, 7]
        print(f"\n  Spectral coefficient stats (pre-MLP):")
        print(f"  {'Pos':>3} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Energy':>8}")
        for k in range(K):
            vals = coeffs[:, k]  # [N, 7]
            mean = vals.mean()
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            energy = (vals ** 2).mean()
            print(f"  {k:>3} {mean:>8.3f} {std:>8.3f} {vmin:>8.3f} {vmax:>8.3f} {energy:>8.4f}")

    # --- Correlation between dims (collapse across positions) ---
    print(f"\n  Cross-dim correlation (averaged across positions):")
    corr_sum = np.zeros((D, D))
    for k in range(K):
        data_k = pre_fsq[:, k, :]  # [N, D]
        corr_k = np.corrcoef(data_k.T)
        corr_sum += corr_k
    corr_avg = corr_sum / K
    for i in range(D):
        row = " ".join([f"{corr_avg[i,j]:>6.3f}" for j in range(D)])
        print(f"    dim {i}: [{row}]")

    return {
        'per_pos_mean': pre_fsq.mean(axis=0).tolist(),      # [K, D]
        'per_pos_std': pre_fsq.std(axis=0).tolist(),         # [K, D]
        'per_pos_min': pre_fsq.min(axis=0).tolist(),         # [K, D]
        'per_pos_max': pre_fsq.max(axis=0).tolist(),         # [K, D]
        'cross_dim_corr': corr_avg.tolist(),
    }


@click.command()
@click.option('--tokenizer-ckpt', required=True,
              help="Tokenizer A (SpectralBasisEncoder)")
@click.option('--tokenizer-ckpt-b', default=None,
              help="Tokenizer B (RegisterEncoder) for comparison")
@click.option('--zarr-path', default='data/libero/libero10_N500.zarr')
@click.option('-o', '--output-dir', default='output/latent_diagnosis')
@click.option('-d', '--device', default='cuda:0')
@click.option('--max-batches', default=50, type=int)
@click.option('--horizon', default=32, type=int)
def main(
    tokenizer_ckpt: str,
    tokenizer_ckpt_b: Optional[str],
    zarr_path: str,
    output_dir: str,
    device: str,
    max_batches: int,
    horizon: int,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    # build dataset
    dataset, val_dataset, loader = build_tokenizer_dataloader(
        zarr_path, horizon=horizon)
    normalizer = dataset.get_normalizer()
    print(f"Dataset: {len(val_dataset)} val samples")

    all_results = {}

    # --- Tokenizer A ---
    print(f"\nLoading tokenizer A from {tokenizer_ckpt}")
    tok_a = OATTok.from_checkpoint(tokenizer_ckpt)
    tok_a.set_normalizer(normalizer)
    tok_a.to(device)

    print("\n>>> Collecting latent stats (A)...")
    stats_a = collect_latent_stats(tok_a, loader, device, max_batches)
    analysis_a = analyze_distribution(stats_a, name="Tokenizer A (Spectral)")
    all_results['analysis_a'] = analysis_a

    # save raw stats for potential visualization
    np.savez(os.path.join(output_dir, 'latent_stats_a.npz'), **stats_a)

    # --- Tokenizer B (optional) ---
    if tokenizer_ckpt_b is not None:
        print(f"\nLoading tokenizer B from {tokenizer_ckpt_b}")
        tok_b = OATTok.from_checkpoint(tokenizer_ckpt_b)
        tok_b.set_normalizer(normalizer)
        tok_b.to(device)

        print("\n>>> Collecting latent stats (B)...")
        stats_b = collect_latent_stats(tok_b, loader, device, max_batches)
        analysis_b = analyze_distribution(stats_b, name="Tokenizer B (Register)")
        all_results['analysis_b'] = analysis_b

        np.savez(os.path.join(output_dir, 'latent_stats_b.npz'), **stats_b)

        # --- Side by side comparison ---
        pre_a = stats_a['pre_fsq']
        pre_b = stats_b['pre_fsq']
        K = pre_a.shape[1]

        print(f"\n{'='*70}")
        print(f"  A vs B: KEY DIFFERENCES")
        print(f"{'='*70}")
        print(f"\n  Per-position std (higher = more spread = better FSQ usage):")
        print(f"  {'Pos':>3} {'A std':>10} {'B std':>10}")
        for k in range(K):
            std_a = pre_a[:, k].std()
            std_b = pre_b[:, k].std()
            print(f"  {k:>3} {std_a:>10.4f} {std_b:>10.4f}")

        print(f"\n  Per-position effective range (max - min):")
        print(f"  {'Pos':>3} {'A range':>10} {'B range':>10}")
        for k in range(K):
            range_a = pre_a[:, k].max() - pre_a[:, k].min()
            range_b = pre_b[:, k].max() - pre_b[:, k].min()
            print(f"  {k:>3} {range_a:>10.4f} {range_b:>10.4f}")

    # save analysis
    out_path = os.path.join(output_dir, 'diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
"""
Evaluate tokenizer-policy diagnostics WITHOUT retraining.

Metrics computed:
  1. Token Prediction Accuracy (per-position and overall)
     - Loads trained policy checkpoint, teacher-forcing on val set
     - Reports argmax + top-k match rate at each token position

  2. Token Entropy & Codebook Usage (tokenizer-only)
     - Per-position entropy and fraction of codebook used

  3. Overlap Rate (tokenizer-only)
     - Tokenizes temporally adjacent action chunks from same episode
     - Reports fraction of matching tokens

Usage:
  # All metrics (needs policy checkpoint + data):
  python experiments/eval_tokenizer_metrics.py \
    -c path/to/policy.ckpt \
    --zarr-path data/libero/libero10_N500.zarr

  # Compare two tokenizers' overlap rate + entropy:
  python experiments/eval_tokenizer_metrics.py \
    --tokenizer-ckpt path/to/spectral_tokenizer.ckpt \
    --tokenizer-ckpt-b path/to/register_tokenizer.ckpt \
    --zarr-path data/libero/libero10_N500.zarr
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
import json
import numpy as np
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.oat.tokenizer import OATTok
from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction


# ============================================================
# Dataset Construction
# ============================================================

def build_dataset_and_loader(
    zarr_path: str,
    batch_size: int = 64,
    n_obs_steps: int = 2,
    horizon: int = 32,
    past_n: int = 7,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Build validation dataset and dataloader from zarr path."""
    obs_keys = [
        "agentview_rgb",
        "robot0_eye_in_hand_rgb",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "task_uid",
    ]

    dataset = ZarrDatasetWithPastAction(
        zarr_path=zarr_path,
        obs_keys=obs_keys,
        action_key="action",
        n_obs_steps=n_obs_steps,
        n_action_steps=horizon,
        seed=seed,
        val_ratio=val_ratio,
        past_n=past_n,
    )

    val_dataset = dataset.get_validation_dataset()

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, val_dataset, val_dataloader


# ============================================================
# Metric 1: Token Prediction Accuracy
# ============================================================

@torch.no_grad()
def compute_token_prediction_accuracy(policy, dataloader, device, max_batches=None):
    """
    Token prediction accuracy via teacher forcing.

    Supports both OATPolicy and OATPolicyWithEnrichedPast:
      - OATPolicy:              cond = obs_features
      - OATPolicyWithEnrichedPast: cond = [obs_features, acc, jerk, raw_past]

    Returns dict with:
      - overall_acc:      float
      - per_pos_acc:      list[float], len=K
      - per_pos_topk_acc: {k: list[float]}  for k in [3, 5, 10]
      - ce_loss:          float (should match training val_loss)
      - num_samples:      int
    """
    policy.eval()
    has_enriched_past = hasattr(policy, '_build_condition')

    all_matches = []
    all_topk_matches = defaultdict(list)
    all_losses = []
    num_samples = 0
    topk_values = [3, 5, 10]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Token accuracy")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = _batch_to_device(batch, device)
        B = batch['action'].shape[0]

        # ground truth tokens
        action_tokens = policy.action_tokenizer.tokenize(batch['action'])  # [B, K]

        # encode observation
        features = policy.obs_encoder(batch['obs'])  # [B, To, d]

        # build condition (handles both policy variants)
        if has_enriched_past:
            cond = policy._build_condition(features, batch['past_action'])
        else:
            cond = features

        # prepend BOS, teacher forcing
        input_tokens = torch.cat([
            torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device),
            action_tokens
        ], dim=1)

        logits = policy.model(input_tokens[:, :-1], cond=cond)  # [B, K, vocab]

        # CE loss
        vocab_size = logits.size(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            action_tokens.reshape(-1),
        )
        all_losses.append(loss.item() * B)

        # argmax accuracy
        pred_tokens = logits.argmax(dim=-1)  # [B, K]
        matches = (pred_tokens == action_tokens)
        all_matches.append(matches.cpu())

        # top-k accuracy
        for k in topk_values:
            _, topk_preds = logits.topk(k, dim=-1)  # [B, K, k]
            topk_match = (topk_preds == action_tokens.unsqueeze(-1)).any(dim=-1)
            all_topk_matches[k].append(topk_match.cpu())

        num_samples += B

    # aggregate
    all_matches = torch.cat(all_matches, dim=0)  # [N, K]
    per_pos_acc = all_matches.float().mean(dim=0).tolist()
    overall_acc = all_matches.float().mean().item()
    ce_loss = sum(all_losses) / num_samples

    per_pos_topk_acc = {}
    for k in topk_values:
        cat = torch.cat(all_topk_matches[k], dim=0)
        per_pos_topk_acc[k] = cat.float().mean(dim=0).tolist()

    return {
        'overall_acc': overall_acc,
        'per_pos_acc': per_pos_acc,
        'per_pos_topk_acc': per_pos_topk_acc,
        'ce_loss': ce_loss,
        'num_samples': num_samples,
    }


# ============================================================
# Metric 2: Token Entropy
# ============================================================

@torch.no_grad()
def compute_token_entropy(tokenizer, dataloader, device, max_batches=None):
    """
    Per-position token entropy and codebook utilization.

    Returns dict with:
      - per_pos_entropy:        list[float], entropy in bits
      - per_pos_codebook_usage: list[float], fraction of codebook used
      - overall_entropy:        float
      - max_entropy:            float (log2 of codebook size)
      - entropy_ratio:          float (overall / max, 1.0 = uniform)
    """
    tokenizer.eval()
    codebook_size = tokenizer.quantizer.codebook_size
    all_token_counts = None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Token entropy")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        actions = batch['action'].to(device)
        tokens = tokenizer.tokenize(actions)  # [B, K]

        if all_token_counts is None:
            K = tokens.shape[1]
            all_token_counts = torch.zeros(K, codebook_size, device='cpu')

        for k in range(K):
            counts = torch.bincount(tokens[:, k].cpu(), minlength=codebook_size)
            all_token_counts[k] += counts.float()

    probs = all_token_counts / all_token_counts.sum(dim=-1, keepdim=True)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)  # [K]
    usage = (all_token_counts > 0).float().mean(dim=-1)
    max_entropy = np.log2(codebook_size)

    return {
        'per_pos_entropy': entropy.tolist(),
        'per_pos_codebook_usage': usage.tolist(),
        'overall_entropy': entropy.mean().item(),
        'max_entropy': max_entropy,
        'entropy_ratio': entropy.mean().item() / max_entropy,
    }


# ============================================================
# Metric 3: Overlap Rate
# ============================================================

@torch.no_grad()
def compute_overlap_rate(tokenizer, dataset, device, max_pairs=5000):
    """
    Overlap rate between temporally adjacent action chunks
    within the same episode.

    Returns dict with:
      - mean_overlap_rate:     float
      - std_overlap_rate:      float
      - per_pos_overlap_rate:  list[float]
      - num_pairs:             int
    """
    tokenizer.eval()
    tokenizer.to(device)

    all_per_pos_matches = []
    all_overlap_rates = []

    total = min(len(dataset) - 1, max_pairs)
    for i in tqdm(range(total), desc="Overlap rate"):
        sample_t = dataset[i]
        sample_t1 = dataset[i + 1]

        action_t = sample_t['action'].unsqueeze(0).to(device)    # [1, H, D]
        action_t1 = sample_t1['action'].unsqueeze(0).to(device)  # [1, H, D]

        tokens_t = tokenizer.tokenize(action_t)    # [1, K]
        tokens_t1 = tokenizer.tokenize(action_t1)  # [1, K]

        per_pos_match = (tokens_t == tokens_t1).float().squeeze(0)  # [K]
        all_per_pos_matches.append(per_pos_match.cpu())
        all_overlap_rates.append(per_pos_match.mean().item())

    if not all_overlap_rates:
        return {'mean_overlap_rate': 0.0, 'num_pairs': 0}

    all_per_pos_matches = torch.stack(all_per_pos_matches, dim=0)  # [N, K]

    return {
        'mean_overlap_rate': float(np.mean(all_overlap_rates)),
        'std_overlap_rate': float(np.std(all_overlap_rates)),
        'per_pos_overlap_rate': all_per_pos_matches.mean(dim=0).tolist(),
        'num_pairs': len(all_overlap_rates),
    }


# ============================================================
# Helpers
# ============================================================

def _batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [_batch_to_device(v, device) for v in batch]
    return batch


def _print_results(results, title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for key, value in results.items():
        if isinstance(value, list) and len(value) <= 12:
            formatted = [f"{v:.4f}" for v in value]
            print(f"  {key}: [{', '.join(formatted)}]")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    formatted = [f"{v:.4f}" for v in sub_value]
                    print(f"  {key}[top-{sub_key}]: [{', '.join(formatted)}]")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ============================================================
# Main
# ============================================================

@click.command()
@click.option('-c', '--checkpoint', default=None,
              help="Policy checkpoint (.ckpt) for token accuracy")
@click.option('--tokenizer-ckpt', default=None,
              help="Tokenizer A checkpoint (if no policy ckpt)")
@click.option('--tokenizer-ckpt-b', default=None,
              help="Tokenizer B checkpoint for A/B comparison")
@click.option('--zarr-path', default='data/libero/libero10_N500.zarr',
              help="Path to zarr dataset")
@click.option('-o', '--output-dir', default='output/eval_metrics',
              help="Output directory")
@click.option('-d', '--device', default='cuda:0')
@click.option('--batch-size', default=64, type=int)
@click.option('--max-batches', default=None, type=int,
              help="Max batches for accuracy/entropy (None=full)")
@click.option('--max-pairs', default=5000, type=int,
              help="Max pairs for overlap rate")
@click.option('--horizon', default=32, type=int)
@click.option('--n-obs-steps', default=2, type=int)
@click.option('--past-n', default=7, type=int)
def main(
    checkpoint: Optional[str],
    tokenizer_ckpt: Optional[str],
    tokenizer_ckpt_b: Optional[str],
    zarr_path: str,
    output_dir: str,
    device: str,
    batch_size: int,
    max_batches: Optional[int],
    max_pairs: int,
    horizon: int,
    n_obs_steps: int,
    past_n: int,
):
    assert checkpoint is not None or tokenizer_ckpt is not None, \
        "Provide either -c (policy ckpt) or --tokenizer-ckpt"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    all_results = {}

    # ---- Build dataset ----
    print(f"Building dataset from {zarr_path}")
    dataset, val_dataset, val_dataloader = build_dataset_and_loader(
        zarr_path=zarr_path,
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        past_n=past_n,
    )
    print(f"  Train samples: {len(dataset)}, Val samples: {len(val_dataset)}")

    # ---- Load policy (optional) ----
    policy = None
    if checkpoint is not None:
        print(f"\nLoading policy from {checkpoint}")
        policy = BasePolicy.from_checkpoint(checkpoint)
        policy.to(device)
        policy.eval()

        # detect policy variant
        if hasattr(policy, '_build_condition'):
            print(f"  Detected: OATPolicyWithEnrichedPast (past_n={policy.past_n})")
            print(f"  Condition: obs({policy.n_obs_steps}) + explicit(2) + raw({policy.past_n})")
        else:
            print(f"  Detected: OATPolicy (no past conditioning)")

    # ---- Resolve tokenizer A ----
    tokenizer_a = None
    if policy is not None:
        tokenizer_a = policy.action_tokenizer
    elif tokenizer_ckpt is not None:
        print(f"Loading tokenizer A from {tokenizer_ckpt}")
        tokenizer_a = OATTok.from_checkpoint(tokenizer_ckpt)
        normalizer = dataset.get_normalizer()
        tokenizer_a.set_normalizer(normalizer)

    if tokenizer_a is not None:
        tokenizer_a.to(device)
        tokenizer_a.eval()

    # ==== Metric 1: Token Prediction Accuracy ====
    if policy is not None:
        print("\n>>> Computing Token Prediction Accuracy...")
        acc_results = compute_token_prediction_accuracy(
            policy, val_dataloader, device, max_batches=max_batches,
        )
        _print_results(acc_results, "TOKEN PREDICTION ACCURACY")
        all_results['token_accuracy'] = acc_results

    # ==== Metric 2: Token Entropy (Tokenizer A) ====
    if tokenizer_a is not None:
        print("\n>>> Computing Token Entropy (A)...")
        entropy_a = compute_token_entropy(
            tokenizer_a, val_dataloader, device, max_batches=max_batches,
        )
        _print_results(entropy_a, "TOKEN ENTROPY (Tokenizer A)")
        all_results['token_entropy_a'] = entropy_a

    # ==== Metric 3: Overlap Rate (Tokenizer A) ====
    if tokenizer_a is not None:
        print("\n>>> Computing Overlap Rate (A)...")
        or_a = compute_overlap_rate(
            tokenizer_a, val_dataset, device, max_pairs=max_pairs,
        )
        _print_results(or_a, "OVERLAP RATE (Tokenizer A)")
        all_results['overlap_rate_a'] = or_a

    # ==== Tokenizer B comparison (optional) ====
    if tokenizer_ckpt_b is not None:
        print(f"\nLoading tokenizer B from {tokenizer_ckpt_b}")
        tokenizer_b = OATTok.from_checkpoint(tokenizer_ckpt_b)
        normalizer = dataset.get_normalizer()
        tokenizer_b.set_normalizer(normalizer)
        tokenizer_b.to(device)
        tokenizer_b.eval()

        print("\n>>> Computing Token Entropy (B)...")
        entropy_b = compute_token_entropy(
            tokenizer_b, val_dataloader, device, max_batches=max_batches,
        )
        _print_results(entropy_b, "TOKEN ENTROPY (Tokenizer B)")
        all_results['token_entropy_b'] = entropy_b

        print("\n>>> Computing Overlap Rate (B)...")
        or_b = compute_overlap_rate(
            tokenizer_b, val_dataset, device, max_pairs=max_pairs,
        )
        _print_results(or_b, "OVERLAP RATE (Tokenizer B)")
        all_results['overlap_rate_b'] = or_b

        # ---- Side-by-side ----
        print(f"\n{'='*60}")
        print(f"  A vs B COMPARISON")
        print(f"{'='*60}")
        print(f"  Overlap Rate:    A={or_a['mean_overlap_rate']:.4f}   B={or_b['mean_overlap_rate']:.4f}")
        print(f"  Entropy (bits):  A={entropy_a['overall_entropy']:.4f}   B={entropy_b['overall_entropy']:.4f}")
        print(f"  Codebook usage:  A={np.mean(entropy_a['per_pos_codebook_usage']):.4f}   "
              f"B={np.mean(entropy_b['per_pos_codebook_usage']):.4f}")
        print(f"  Per-pos OR (A):  {[f'{v:.3f}' for v in or_a['per_pos_overlap_rate']]}")
        print(f"  Per-pos OR (B):  {[f'{v:.3f}' for v in or_b['per_pos_overlap_rate']]}")
        print(f"{'='*60}\n")

    # ---- Save ----
    out_path = os.path.join(output_dir, 'tokenizer_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(_to_serializable(all_results), f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
"""
Standalone AR Entropy Analysis for OAT
=======================================
Usage:
    python analyze_entropy.py \
        --ckpt_dir /path/to/run_output_dir \
        --ckpt_name latest.ckpt \
        --n_batches 20 \
        --output_dir ./entropy_results

Or compare two models:
    python analyze_entropy.py \
        --compare \
        --ckpt_a /path/to/original/latest.ckpt \
        --ckpt_b /path/to/enriched/latest.ckpt \
        --label_a "Original OAT" \
        --label_b "+Past Actions" \
        --n_batches 20 \
        --output_dir ./entropy_results

What it does:
    1. Loads checkpoint + config (hydra style)
    2. Creates the dataset/dataloader
    3. Runs teacher-forced forward → gets logits at every AR step
    4. Computes H(z_k | z_{<k}, cond) per step
    5. Saves figures + CSV

No env/rollout needed. Pure offline analysis on the training data.
"""

import os
import sys
import json
import argparse
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# ── You need the oat package importable ──
# e.g. run from the repo root, or `pip install -e .`
import hydra
from omegaconf import OmegaConf


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load a trained policy from checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def load_policy_from_checkpoint(ckpt_path: str, device: str = "cuda:0"):
    """
    Load a trained OAT policy from a hydra-managed checkpoint.

    Expects the checkpoint directory to contain:
        - .hydra/config.yaml  (in the run output dir)
        - the .ckpt file (state_dict)

    Returns:
        policy: the loaded policy (eval mode, on device)
        cfg: the hydra config
        dataset: the training dataset (for dataloader creation)
    """
    ckpt_path = pathlib.Path(ckpt_path)

    # ── Find the config ──────────────────────────────────────────────
    # Walk up from checkpoint to find .hydra/config.yaml
    # Typical structure: output/YYYYMMDD/HHMMSS_name/.hydra/config.yaml
    #                    output/YYYYMMDD/HHMMSS_name/checkpoints/latest.ckpt
    search_dir = ckpt_path.parent
    cfg = None
    for _ in range(5):  # walk up at most 5 levels
        config_path = search_dir / ".hydra" / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            print(f"Found config at: {config_path}")
            break
        search_dir = search_dir.parent

    # ── Load checkpoint (needed for embedded config fallback) ────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if cfg is None:
        # Try embedded config inside the checkpoint
        if isinstance(ckpt, dict) and "cfg" in ckpt:
            cfg = OmegaConf.create(ckpt["cfg"])
            print(f"Using config embedded in checkpoint")
        else:
            raise FileNotFoundError(
                f"Cannot find .hydra/config.yaml starting from {ckpt_path.parent}. "
                f"You can also pass --config_path explicitly."
            )

    # ── Instantiate the policy ───────────────────────────────────────
    policy = hydra.utils.instantiate(cfg.policy)
    policy = policy.to(device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dicts" in ckpt:
            # workspace-style checkpoint
            if "ema_model" in ckpt["state_dicts"]:
                state_dict = ckpt["state_dicts"]["ema_model"]
                print("Loading EMA model weights")
            elif "model" in ckpt["state_dicts"]:
                state_dict = ckpt["state_dicts"]["model"]
                print("Loading model weights")
            else:
                raise KeyError(f"Unknown keys in state_dicts: {ckpt['state_dicts'].keys()}")
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Assume the dict itself is the state_dict
            state_dict = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Try loading, handle prefix mismatches
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Try stripping common prefixes
        for prefix in ["module.", "policy.", "ema_model."]:
            stripped = {k.removeprefix(prefix): v for k, v in state_dict.items()}
            try:
                policy.load_state_dict(stripped, strict=True)
                print(f"Loaded after stripping '{prefix}' prefix")
                break
            except RuntimeError:
                continue
        else:
            # Last resort: non-strict
            missing, unexpected = policy.load_state_dict(state_dict, strict=False)
            print(f"WARNING: non-strict load. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"  Unexpected (first 5): {unexpected[:5]}")

    policy.eval()
    print(f"Policy loaded: {policy.get_policy_name()}")

    # ── Create the dataset ───────────────────────────────────────────
    # Determine which dataset class to use
    is_enriched = hasattr(policy, 'past_n')
    if is_enriched:
        from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
        dataset = ZarrDatasetWithPastAction(
            past_n=cfg.get("past_n", 7),
            zarr_path=cfg.task.policy.dataset.zarr_path,
            obs_keys=list(cfg.shape_meta.obs.keys()),
            action_key="action",
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.horizon,
            seed=cfg.seed,
        )
    else:
        from oat.dataset.zarr_dataset import ZarrDataset
        dataset = ZarrDataset(
            zarr_path=cfg.task.policy.dataset.zarr_path,
            obs_keys=list(cfg.shape_meta.obs.keys()),
            action_key="action",
            n_obs_steps=cfg.n_obs_steps,
            n_action_steps=cfg.horizon,
            seed=cfg.seed,
        )

    # Set normalizer (must be done before moving policy to device, since
    # LinearNormalizer starts empty — set_normalizer adds CPU params that
    # _normalize() will use as the target device for input tensors)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)  # move normalizer params to device along with everything else

    return policy, cfg, dataset


# ══════════════════════════════════════════════════════════════════════════════
# 2. Collect teacher-forced logits
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_teacher_forced_entropy(
    policy,
    dataloader,
    n_batches: int = 20,
    device: str = "cuda:0",
):
    """
    Run teacher-forced forward passes and collect per-step entropy.

    This replicates the logic inside policy.forward() but saves the
    per-step logits instead of just computing the loss.

    Returns:
        dict with:
            'entropy_per_step':    (n_samples, K) — H at each AR step
            'topk_mass_per_step':  (n_samples, K) — top-10 prob mass
            'loss_per_step':       (n_samples, K) — per-token CE loss
            'vocab_size':          int
    """
    policy.eval()

    all_entropy = []
    all_topk_mass = []
    all_loss = []
    vocab_size = None

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        # Move batch to device
        batch_device = {}
        batch_device["action"] = batch["action"].to(device)
        batch_device["obs"] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["obs"].items()
        }
        if "past_action" in batch:
            batch_device["past_action"] = batch["past_action"].to(device)

        # ── Replicate policy.forward() to get logits ─────────────────
        # Tokenize
        action_tokens = policy.action_tokenizer.tokenize(batch_device["action"])
        B = batch_device["action"].shape[0]

        # Encode obs
        features = policy.obs_encoder(batch_device["obs"])  # (B, To, d)

        # Build condition
        if hasattr(policy, '_build_condition') and "past_action" in batch_device:
            cond = policy._build_condition(features, batch_device["past_action"])
        else:
            cond = features

        # Prepend BOS
        bos = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device)
        tokens_with_bos = torch.cat([bos, action_tokens], dim=1)

        # Forward → logits
        logits = policy.model(tokens_with_bos[:, :-1], cond=cond)  # (B, K, V)
        targets = tokens_with_bos[:, 1:]  # (B, K)

        if vocab_size is None:
            vocab_size = logits.shape[-1]

        # ── Compute per-step metrics ─────────────────────────────────
        # Entropy: H = -sum(p * log p)
        log_probs = F.log_softmax(logits, dim=-1)       # (B, K, V)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)       # (B, K)

        # Top-k mass
        topk_probs, _ = torch.topk(probs, k=min(10, probs.shape[-1]), dim=-1)
        topk_mass = topk_probs.sum(dim=-1)               # (B, K)

        # Per-token loss
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none',
        ).reshape(B, -1)                                  # (B, K)

        all_entropy.append(entropy.cpu().numpy())
        all_topk_mass.append(topk_mass.cpu().numpy())
        all_loss.append(loss_per_token.cpu().numpy())

        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{n_batches} done")

    return {
        'entropy_per_step': np.concatenate(all_entropy, axis=0),      # (N, K)
        'topk_mass_per_step': np.concatenate(all_topk_mass, axis=0),  # (N, K)
        'loss_per_step': np.concatenate(all_loss, axis=0),            # (N, K)
        'vocab_size': vocab_size,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_model(stats, label, output_dir):
    """Plot entropy analysis for a single model."""
    import matplotlib.pyplot as plt

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K = stats['entropy_per_step'].shape[1]
    V = stats['vocab_size']
    steps = np.arange(K)

    mean_H = stats['entropy_per_step'].mean(axis=0)
    std_H = stats['entropy_per_step'].std(axis=0)
    mean_topk = stats['topk_mass_per_step'].mean(axis=0)
    mean_loss = stats['loss_per_step'].mean(axis=0)
    eff_vocab = np.exp(mean_H)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'{label}  |  |V| = {V}', fontsize=14, fontweight='bold')

    # (a) Entropy per step
    ax = axes[0, 0]
    ax.plot(steps, mean_H, 'o-', lw=2, ms=6)
    ax.fill_between(steps, mean_H - std_H, mean_H + std_H, alpha=0.2)
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('H(z_k | z_{<k}, cond) [nats]')
    ax.set_title('Conditional Entropy per Step')
    ax.axhline(np.log(V), ls='--', color='gray', alpha=0.5, label=f'log|V|={np.log(V):.2f}')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Effective vocab
    ax = axes[0, 1]
    ax.bar(steps, eff_vocab, alpha=0.7)
    ax.axhline(V, ls='--', color='red', alpha=0.5, label=f'|V|={V}')
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('exp(H) — Effective Vocab')
    ax.set_title(f'Effective Vocab Size (ratio={eff_vocab.mean()/V:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Top-10 mass
    ax = axes[1, 0]
    ax.plot(steps, mean_topk, 's-', lw=2, ms=6, color='green')
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('P(top-10 tokens)')
    ax.set_title('Top-10 Probability Mass')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # (d) Per-token loss
    ax = axes[1, 1]
    ax.plot(steps, mean_loss, '^-', lw=2, ms=6, color='orange')
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Per-Step Loss')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'entropy_single_{label.replace(" ", "_")}.png', dpi=150)
    plt.close(fig)
    label_slug = label.replace(' ', '_')
    print(f"Saved: {output_dir / f'entropy_single_{label_slug}.png'}")


def plot_comparison(stats_a, stats_b, label_a, label_b, output_dir):
    """The money plot: compare two models side by side."""
    import matplotlib.pyplot as plt

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K = stats_a['entropy_per_step'].shape[1]
    V_a = stats_a['vocab_size']
    V_b = stats_b['vocab_size']
    steps = np.arange(K)

    H_a = stats_a['entropy_per_step'].mean(axis=0)
    H_b = stats_b['entropy_per_step'].mean(axis=0)
    std_a = stats_a['entropy_per_step'].std(axis=0)
    std_b = stats_b['entropy_per_step'].std(axis=0)

    topk_a = stats_a['topk_mass_per_step'].mean(axis=0)
    topk_b = stats_b['topk_mass_per_step'].mean(axis=0)

    C_a, C_b = '#d62728', '#2ca02c'

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f'Conditional Entropy Comparison\n{label_a} (|V|={V_a}) vs {label_b} (|V|={V_b})',
        fontsize=14, fontweight='bold',
    )

    # (a) Entropy per step — overlay
    ax = axes[0, 0]
    ax.plot(steps, H_a, 'o-', lw=2, ms=6, color=C_a, label=label_a)
    ax.fill_between(steps, H_a - std_a, H_a + std_a, alpha=0.15, color=C_a)
    ax.plot(steps, H_b, 's-', lw=2, ms=6, color=C_b, label=label_b)
    ax.fill_between(steps, H_b - std_b, H_b + std_b, alpha=0.15, color=C_b)
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('H(z_k | z_{<k}, cond) [nats]')
    ax.set_title('Per-Step Conditional Entropy')
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Delta entropy
    ax = axes[0, 1]
    delta_H = H_a - H_b
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta_H]
    ax.bar(steps, delta_H, color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('AR Step k')
    ax.set_ylabel(f'ΔH = H({label_a}) − H({label_b})')
    ax.set_title(f'Entropy Reduction per Step (avg ΔH = {delta_H.mean():.3f} nats)')
    ax.grid(alpha=0.3)

    # (c) Top-10 mass comparison
    ax = axes[1, 0]
    ax.plot(steps, topk_a, 'o-', lw=2, ms=6, color=C_a, label=label_a)
    ax.plot(steps, topk_b, 's-', lw=2, ms=6, color=C_b, label=label_b)
    ax.set_xlabel('AR Step k')
    ax.set_ylabel('P(top-10)')
    ax.set_title('Top-10 Probability Mass')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) Entropy distribution (histogram)
    ax = axes[1, 1]
    avg_H_a = stats_a['entropy_per_step'].mean(axis=1)  # per-sample avg
    avg_H_b = stats_b['entropy_per_step'].mean(axis=1)
    ax.hist(avg_H_a, bins=50, alpha=0.6, color=C_a, label=label_a, density=True)
    ax.hist(avg_H_b, bins=50, alpha=0.6, color=C_b, label=label_b, density=True)
    ax.set_xlabel('Sample-Avg Entropy [nats]')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Per-Sample Average Entropy')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / 'entropy_comparison.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Print summary table ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Metric':<30} {label_a:>15} {label_b:>15}")
    print("=" * 65)
    print(f"{'|V|':<30} {V_a:>15} {V_b:>15}")
    print(f"{'Avg Entropy (nats)':<30} {H_a.mean():>15.4f} {H_b.mean():>15.4f}")
    print(f"{'Avg exp(H) (eff vocab)':<30} {np.exp(H_a).mean():>15.1f} {np.exp(H_b).mean():>15.1f}")
    print(f"{'exp(H)/|V| ratio':<30} {np.exp(H_a).mean()/V_a:>15.5f} {np.exp(H_b).mean()/V_b:>15.5f}")
    print(f"{'Avg Top-10 mass':<30} {topk_a.mean():>15.4f} {topk_b.mean():>15.4f}")
    print(f"{'Avg per-token loss':<30} {stats_a['loss_per_step'].mean():>15.4f} {stats_b['loss_per_step'].mean():>15.4f}")
    print("=" * 65)

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = output_dir / 'entropy_summary.csv'
    with open(csv_path, 'w') as f:
        f.write("step,H_a,H_b,delta_H,topk_a,topk_b,effV_a,effV_b\n")
        for k in range(K):
            f.write(f"{k},{H_a[k]:.6f},{H_b[k]:.6f},{delta_H[k]:.6f},"
                    f"{topk_a[k]:.6f},{topk_b[k]:.6f},"
                    f"{np.exp(H_a[k]):.2f},{np.exp(H_b[k]):.2f}\n")
    print(f"Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════

def run_single(args):
    """Analyze a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading: {args.ckpt}")
    print(f"{'='*60}")

    policy, cfg, dataset = load_policy_from_checkpoint(
        args.ckpt, device=args.device,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"\nCollecting entropy over {args.n_batches} batches "
          f"(~{args.n_batches * args.batch_size} samples)...")
    stats = collect_teacher_forced_entropy(
        policy, dataloader,
        n_batches=args.n_batches,
        device=args.device,
    )

    V = stats['vocab_size']
    H = stats['entropy_per_step'].mean()
    print(f"\n  |V| = {V}")
    print(f"  Avg H = {H:.4f} nats")
    print(f"  exp(H) = {np.exp(H):.1f}")
    print(f"  exp(H)/|V| = {np.exp(H)/V:.5f}")

    label = args.label or f"model_V{V}"
    plot_single_model(stats, label, args.output_dir)

    # Save raw stats
    np.savez(
        pathlib.Path(args.output_dir) / f"stats_{label.replace(' ', '_')}.npz",
        **stats,
    )
    print(f"Saved raw stats to {args.output_dir}/stats_{label.replace(' ', '_')}.npz")


def run_compare(args):
    """Compare two checkpoints."""
    print(f"\n{'='*60}")
    print(f"Model A: {args.ckpt_a}")
    print(f"Model B: {args.ckpt_b}")
    print(f"{'='*60}")

    # Load A
    print(f"\n── Loading Model A ({args.label_a}) ──")
    policy_a, cfg_a, dataset_a = load_policy_from_checkpoint(
        args.ckpt_a, device=args.device,
    )
    dl_a = torch.utils.data.DataLoader(
        dataset_a, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Load B
    print(f"\n── Loading Model B ({args.label_b}) ──")
    policy_b, cfg_b, dataset_b = load_policy_from_checkpoint(
        args.ckpt_b, device=args.device,
    )
    dl_b = torch.utils.data.DataLoader(
        dataset_b, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Collect
    print(f"\nCollecting entropy — Model A...")
    stats_a = collect_teacher_forced_entropy(
        policy_a, dl_a, n_batches=args.n_batches, device=args.device,
    )
    print(f"Collecting entropy — Model B...")
    stats_b = collect_teacher_forced_entropy(
        policy_b, dl_b, n_batches=args.n_batches, device=args.device,
    )

    # Plot
    plot_comparison(stats_a, stats_b, args.label_a, args.label_b, args.output_dir)
    plot_single_model(stats_a, args.label_a, args.output_dir)
    plot_single_model(stats_b, args.label_b, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="OAT AR Entropy Analysis")
    sub = parser.add_subparsers(dest="mode", help="Run mode")

    # ── Single model ─────────────────────────────────────────────────
    p_single = sub.add_parser("single", help="Analyze one model")
    p_single.add_argument("--ckpt", type=str, required=True,
                          help="Path to .ckpt file")
    p_single.add_argument("--label", type=str, default=None,
                          help="Label for this model")

    # ── Compare two ──────────────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare two models")
    p_compare.add_argument("--ckpt_a", type=str, required=True)
    p_compare.add_argument("--ckpt_b", type=str, required=True)
    p_compare.add_argument("--label_a", type=str, default="Original")
    p_compare.add_argument("--label_b", type=str, default="+Past Actions")

    # ── Shared args ──────────────────────────────────────────────────
    for p in [p_single, p_compare]:
        p.add_argument("--n_batches", type=int, default=20,
                       help="Number of batches to evaluate")
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--device", type=str, default="cuda:0")
        p.add_argument("--output_dir", type=str, default="./entropy_results")

    args = parser.parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "compare":
        run_compare(args)
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  # Analyze one model:")
        print("  python analyze_entropy.py single \\")
        print("      --ckpt output/run_enriched_V1920/checkpoints/latest.ckpt \\")
        print("      --label 'Enriched V1920'")
        print()
        print("  # Compare original vs enriched:")
        print("  python analyze_entropy.py compare \\")
        print("      --ckpt_a output/run_original_V1920/checkpoints/latest.ckpt \\")
        print("      --ckpt_b output/run_enriched_V1920/checkpoints/latest.ckpt \\")
        print("      --label_a 'Original' --label_b '+Past Actions'")


if __name__ == "__main__":
    main()
"""
Analyze reconstruction quality of the original OATPolicy (no past actions).

Measures:
  1. action_pred vs GT (clean)   — total error: tokenizer + policy token prediction
  2. action_pred vs GT (detok)   — policy token prediction error only

Both per-timestep (0..31) and aggregate.

Usage:
  python analysis_scripts/analyze_reconstruction_quality.py \
      --ckpt /workspace/oat/output/20260316/191419_train_oatpolicy_libero10_N500/checkpoints/ep-0100_sr-0.516.ckpt \
      --dataset_path /workspace/oat/data/libero/libero10_N500.zarr \
      --num_batches 50 \
      --output_dir ./reconstruction_quality_results
"""

import argparse
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str):
    ckpt_path = pathlib.Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = None
    search_dir = ckpt_path.parent
    for _ in range(5):
        config_path = search_dir / ".hydra" / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            break
        search_dir = search_dir.parent
    if cfg is None:
        if isinstance(ckpt, dict) and "cfg" in ckpt:
            cfg = OmegaConf.create(ckpt["cfg"])
        else:
            raise FileNotFoundError("Cannot find config for checkpoint")

    policy = hydra.utils.instantiate(cfg.policy)

    state_dict = ckpt["state_dicts"].get("ema_model") or ckpt["state_dicts"]["model"]
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        policy.load_state_dict(state_dict, strict=False)

    return policy, cfg


def load_dataset(cfg, dataset_path: str):
    from oat.dataset.zarr_dataset import ZarrDataset
    dataset = ZarrDataset(
        zarr_path=dataset_path,
        obs_keys=list(cfg.shape_meta.obs.keys()),
        action_key="action",
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.horizon,
        seed=cfg.seed,
    )
    return dataset


@torch.no_grad()
def generate_actions(policy, obs_features):
    """
    Run one forward pass given pre-computed obs features.
    Returns action_pred: (B, horizon, action_dim)
    """
    B = obs_features.shape[0]

    action_tokens = torch.full(
        (B, 1), policy.bos_id,
        dtype=torch.long, device=policy.device,
    )
    action_tokens = policy.model.generate(
        action_tokens,
        cond=obs_features,
        max_new_tokens=policy.max_seq_len,
        temperature=1.0,
        top_k=policy.topk,
    )[:, 1:]
    action_tokens = action_tokens.clamp(0, policy.bos_id - 1)

    action_pred = policy.action_tokenizer.detokenize(tokens=action_tokens)
    return action_pred  # (B, horizon, action_dim)


# ── Main collection loop ──────────────────────────────────────────────────────

@torch.no_grad()
def collect(policy, dataloader, n_batches: int, device: str):
    """
    Returns dict of accumulated tensors (on CPU):
      action_pred, gt_clean, gt_detok
      each shape: (N, horizon, action_dim)
    """
    policy.eval()

    results = {k: [] for k in ["action_pred", "gt_clean", "gt_detok"]}

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        action_gt = batch["action"].to(device)       # (B, horizon, Da)
        obs_dict  = {k: v.to(device) for k, v in batch["obs"].items()
                     if isinstance(v, torch.Tensor)}

        # encode observations
        obs_features = policy.obs_encoder(obs_dict)  # (B, To, d)

        # predict actions
        action_pred = generate_actions(policy, obs_features)  # (B, horizon, Da)

        # GT detok: best the tokenizer can reconstruct
        gt_detok = policy.action_tokenizer.autoencode(action_gt)  # (B, horizon, Da)

        results["action_pred"].append(action_pred.cpu())
        results["gt_clean"].append(action_gt.cpu())
        results["gt_detok"].append(gt_detok.cpu())

        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{n_batches}")

    return {k: torch.cat(v, dim=0) for k, v in results.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mse(pred, target):
    """
    pred, target: (N, horizon, Da)
    Returns:
      per_step: (horizon,)
      aggregate: scalar
    """
    err = (pred - target).pow(2)
    per_step  = err.mean(dim=(0, 2)).numpy()
    aggregate = err.mean().item()
    return per_step, aggregate


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(metrics, n_action_steps: int, output_dir: pathlib.Path):
    horizon = len(next(iter(metrics.values()))["per_step"])
    steps = np.arange(horizon)

    comparisons = [
        ("pred_vs_gt_clean", "action_pred vs GT (clean)", "tab:blue"),
        ("pred_vs_gt_detok", "action_pred vs GT (detok)", "tab:orange"),
    ]

    # ── Per-timestep MSE ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in comparisons:
        if key in metrics:
            ax.plot(steps, metrics[key]["per_step"], label=label, color=color)
    ax.axvline(x=n_action_steps - 1, color="gray", linestyle="--",
               alpha=0.5, label=f"n_action_steps={n_action_steps}")
    ax.set_xlabel("Timestep within predicted horizon")
    ax.set_ylabel("MSE")
    ax.set_title("Per-timestep Reconstruction Quality (Original OATPolicy)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_step_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_step_mse.png'}")

    # ── Aggregate bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    agg_labels, agg_vals, agg_colors = [], [], []
    for key, label, color in comparisons:
        if key in metrics:
            agg_labels.append(label)
            agg_vals.append(metrics[key]["aggregate"])
            agg_colors.append(color)

    bars = ax.bar(range(len(agg_vals)), agg_vals, color=agg_colors)
    ax.set_xticks(range(len(agg_labels)))
    ax.set_xticklabels(agg_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MSE")
    ax.set_title("Aggregate Reconstruction MSE (Original OATPolicy)")
    for bar, val in zip(bars, agg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'aggregate_mse.png'}")


def print_summary(metrics):
    print("\n" + "=" * 60)
    print(f"{'Comparison':<30} {'Aggregate MSE':>15} {'RMSE':>12}")
    print("=" * 60)
    labels = {
        "pred_vs_gt_clean": "action_pred vs GT (clean)",
        "pred_vs_gt_detok": "action_pred vs GT (detok)",
    }
    for key, label in labels.items():
        if key in metrics:
            mse = metrics[key]["aggregate"]
            print(f"  {label:<28} {mse:>15.6f} {mse**0.5:>12.6f}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./reconstruction_quality_results")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {args.ckpt} ...")
    policy, cfg = load_policy(args.ckpt, args.device)

    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = load_dataset(cfg, args.dataset_path)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(args.device)
    policy.eval()
    print(f"Policy: {policy.get_policy_name()}")
    print(f"  n_action_steps={policy.n_action_steps}, horizon={policy.max_seq_len}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"\nCollecting over {args.num_batches} batches ...")
    data = collect(policy, dataloader, args.num_batches, args.device)
    print(f"Collected {data['action_pred'].shape[0]} samples")

    metrics = {}
    metrics["pred_vs_gt_clean"] = dict(zip(["per_step", "aggregate"],
                                      compute_mse(data["action_pred"], data["gt_clean"])))
    metrics["pred_vs_gt_detok"] = dict(zip(["per_step", "aggregate"],
                                      compute_mse(data["action_pred"], data["gt_detok"])))

    print_summary(metrics)
    plot_results(metrics, policy.n_action_steps, output_dir)

    csv_path = output_dir / "per_step_mse.csv"
    horizon = data["action_pred"].shape[1]
    header = "step," + ",".join(metrics.keys())
    rows = np.column_stack([np.arange(horizon)] + [metrics[k]["per_step"] for k in metrics])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

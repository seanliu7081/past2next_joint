"""
Compare action prediction error between Original OATPolicy and
OATPolicyWithEnrichedPast using realistic inference conditions.

Original OAT:     obs → predict → action_pred
Enriched Past:    obs + own predicted past → predict → action_pred
  (own past = last past_n steps of the previously predicted action chunk,
   mimicking the past buffer state during real rollout evaluation)

Measurements (per-timestep 0..31 and aggregate):
  1. original_oat  vs GT (clean)
  2. enriched_past vs GT (clean)
  3. original_oat  vs enriched_past  (direct difference)

Usage:
  python analysis_scripts/compare_oat_vs_enriched.py \
      --ckpt_oat <path_to_oat_ckpt> \
      --ckpt_enriched <path_to_enriched_ckpt> \
      --dataset_path /workspace/oat/data/libero/libero10_N500.zarr \
      --num_batches 50 \
      --output_dir ./oat_vs_enriched_results
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
            raise FileNotFoundError(f"Cannot find config for {ckpt_path}")

    policy = hydra.utils.instantiate(cfg.policy)
    state_dict = ckpt["state_dicts"].get("ema_model") or ckpt["state_dicts"]["model"]
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        policy.load_state_dict(state_dict, strict=False)

    return policy, cfg


@torch.no_grad()
def predict_oat(policy, obs_features):
    """Original OAT: obs features only."""
    B = obs_features.shape[0]
    action_tokens = torch.full(
        (B, 1), policy.bos_id, dtype=torch.long, device=policy.device
    )
    action_tokens = policy.model.generate(
        action_tokens,
        cond=obs_features,
        max_new_tokens=policy.max_seq_len,
        temperature=1.0,
        top_k=policy.topk,
    )[:, 1:]
    action_tokens = action_tokens.clamp(0, policy.bos_id - 1)
    return policy.action_tokenizer.detokenize(tokens=action_tokens)


@torch.no_grad()
def predict_enriched(policy, obs_features, past_actions):
    """Enriched past: obs features + past actions as condition."""
    cond = policy._build_condition(obs_features, past_actions)
    B = obs_features.shape[0]
    action_tokens = torch.full(
        (B, 1), policy.bos_id, dtype=torch.long, device=policy.device
    )
    action_tokens = policy.model.generate(
        action_tokens,
        cond=cond,
        max_new_tokens=policy.max_seq_len,
        temperature=1.0,
        top_k=policy.topk,
    )[:, 1:]
    action_tokens = action_tokens.clamp(0, policy.bos_id - 1)
    return policy.action_tokenizer.detokenize(tokens=action_tokens)


# ── Collection ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(policy_oat, policy_enriched, dataloader, n_batches: int, device: str):
    """
    For each batch:
      - Original OAT:    obs → predict → action_oat
      - Enriched past:
          step 1: obs + GT past → predict → action_warm  (warm-up prediction)
          step 2: obs + own past (from action_warm) → predict → action_enriched
                  (this mimics what the policy sees during rollout)

    Returns tensors of shape (N, horizon, action_dim):
      action_oat, action_enriched, gt_clean
    """
    policy_oat.eval()
    policy_enriched.eval()

    n_action_steps = policy_enriched.n_action_steps
    past_n         = policy_enriched.past_n

    results = {k: [] for k in ["action_oat", "action_enriched", "gt_clean"]}

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        action_gt   = batch["action"].to(device)       # (B, horizon, Da)
        past_action = batch["past_action"].to(device)  # (B, past_n, Da)
        obs_dict    = {k: v.to(device) for k, v in batch["obs"].items()
                       if isinstance(v, torch.Tensor)}

        # ── Original OAT ──────────────────────────────────────────────────
        obs_features_oat = policy_oat.obs_encoder(obs_dict)   # (B, To, d)
        action_oat = predict_oat(policy_oat, obs_features_oat)

        # ── Enriched Past ─────────────────────────────────────────────────
        obs_features_enc = policy_enriched.obs_encoder(obs_dict)

        # Step 1: warm-up with GT past to get a realistic first prediction
        action_warm = predict_enriched(policy_enriched, obs_features_enc, past_action)

        # Step 2: build own past from warm-up prediction (rollout-realistic)
        # same logic as receding horizon: take last past_n of executed chunk
        if n_action_steps >= past_n:
            own_past = action_warm[:, n_action_steps - past_n : n_action_steps]
        else:
            own_past = torch.cat([
                past_action[:, n_action_steps:],
                action_warm[:, :n_action_steps],
            ], dim=1)

        action_enriched = predict_enriched(policy_enriched, obs_features_enc, own_past)

        results["action_oat"].append(action_oat.cpu())
        results["action_enriched"].append(action_enriched.cpu())
        results["gt_clean"].append(action_gt.cpu())

        if (i + 1) % 10 == 0:
            print(f"  batch {i+1}/{n_batches}")

    return {k: torch.cat(v, dim=0) for k, v in results.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mse(pred, target):
    err = (pred - target).pow(2)
    per_step  = err.mean(dim=(0, 2)).numpy()   # (horizon,)
    aggregate = err.mean().item()
    return per_step, aggregate


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(metrics, n_action_steps: int, output_dir: pathlib.Path):
    horizon = len(next(iter(metrics.values()))["per_step"])
    steps = np.arange(horizon)

    comparisons = [
        ("oat_vs_gt",      "Original OAT vs GT",      "tab:blue"),
        ("enriched_vs_gt", "Enriched Past vs GT",      "tab:orange"),
        ("oat_vs_enriched","Original OAT vs Enriched", "tab:green"),
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
    ax.set_title("Per-timestep Action Error: Original OAT vs Enriched Past")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_step_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_step_mse.png'}")

    # ── Aggregate bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
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
    ax.set_title("Aggregate Action Error: Original OAT vs Enriched Past")
    for bar, val in zip(bars, agg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'aggregate_mse.png'}")


def print_summary(metrics):
    print("\n" + "=" * 62)
    print(f"{'Comparison':<32} {'Aggregate MSE':>15} {'RMSE':>12}")
    print("=" * 62)
    labels = {
        "oat_vs_gt":       "Original OAT vs GT",
        "enriched_vs_gt":  "Enriched Past vs GT",
        "oat_vs_enriched": "Original OAT vs Enriched",
    }
    for key, label in labels.items():
        if key in metrics:
            mse = metrics[key]["aggregate"]
            print(f"  {label:<30} {mse:>15.6f} {mse**0.5:>12.6f}")
    print("=" * 62)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_oat",      required=True, help="Original OAT checkpoint")
    parser.add_argument("--ckpt_enriched", required=True, help="Enriched past checkpoint")
    parser.add_argument("--dataset_path",  required=True)
    parser.add_argument("--num_batches",   type=int, default=50)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--device",        type=str, default="cuda:0")
    parser.add_argument("--output_dir",    type=str, default="./oat_vs_enriched_results")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policies ─────────────────────────────────────────────────────
    print(f"Loading Original OAT from {args.ckpt_oat} ...")
    policy_oat, cfg_oat = load_policy(args.ckpt_oat, args.device)

    print(f"Loading Enriched Past from {args.ckpt_enriched} ...")
    policy_enriched, cfg_enriched = load_policy(args.ckpt_enriched, args.device)

    # ── Dataset: use enriched past dataset (has past_action field) ────────
    print(f"Loading dataset from {args.dataset_path} ...")
    from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
    dataset = ZarrDatasetWithPastAction(
        past_n=cfg_enriched.get("past_n", 7),
        zarr_path=args.dataset_path,
        obs_keys=list(cfg_enriched.shape_meta.obs.keys()),
        action_key="action",
        n_obs_steps=cfg_enriched.n_obs_steps,
        n_action_steps=cfg_enriched.horizon,
        seed=cfg_enriched.seed,
    )

    # set normalizers
    normalizer = dataset.get_normalizer()
    policy_oat.set_normalizer(normalizer)
    policy_enriched.set_normalizer(normalizer)
    policy_oat.to(args.device).eval()
    policy_enriched.to(args.device).eval()

    print(f"Original OAT:    {policy_oat.get_policy_name()}")
    print(f"Enriched Past:   {policy_enriched.get_policy_name()}")
    print(f"  n_action_steps={policy_enriched.n_action_steps}, "
          f"past_n={policy_enriched.past_n}, horizon={policy_enriched.max_seq_len}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Collect ───────────────────────────────────────────────────────────
    print(f"\nCollecting over {args.num_batches} batches ...")
    data = collect(policy_oat, policy_enriched, dataloader, args.num_batches, args.device)
    print(f"Collected {data['action_oat'].shape[0]} samples")

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = {}
    metrics["oat_vs_gt"]       = dict(zip(["per_step", "aggregate"],
                                     compute_mse(data["action_oat"],      data["gt_clean"])))
    metrics["enriched_vs_gt"]  = dict(zip(["per_step", "aggregate"],
                                     compute_mse(data["action_enriched"], data["gt_clean"])))
    metrics["oat_vs_enriched"] = dict(zip(["per_step", "aggregate"],
                                     compute_mse(data["action_oat"],      data["action_enriched"])))

    print_summary(metrics)
    plot_results(metrics, policy_enriched.n_action_steps, output_dir)

    csv_path = output_dir / "per_step_mse.csv"
    horizon = data["action_oat"].shape[1]
    header = "step," + ",".join(metrics.keys())
    rows = np.column_stack([np.arange(horizon)] + [metrics[k]["per_step"] for k in metrics])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

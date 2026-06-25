"""
Compare the effect of past action conditioning on predicted actions.

Experiment A:  past condition = ground-truth past actions from dataset
Experiment B1: past condition = policy's own predicted actions
               (one receding-horizon step removed from A)

Measurements (per-timestep 0..31 and aggregate):
  1. action_A  vs GT_future (clean)
  2. action_A  vs GT_future (detok)
  3. action_B1 vs GT_future (clean)
  4. action_B1 vs GT_future (detok)
  5. action_A  vs action_B1

Usage:
  python analysis_scripts/compare_past_conditions.py \
      --ckpt /workspace/oat/ep-0800_sr-0.656.ckpt \
      --dataset_path /workspace/oat/data/libero/libero10_N500.zarr \
      --num_batches 50 \
      --output_dir ./past_condition_results
"""

import argparse
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str):
    ckpt_path = pathlib.Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # find config: walk up for .hydra/config.yaml, else use embedded cfg
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

    # load weights
    state_dict = ckpt["state_dicts"].get("ema_model") or ckpt["state_dicts"]["model"]
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        policy.load_state_dict(state_dict, strict=False)

    return policy, cfg


def load_dataset(cfg, dataset_path: str):
    from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction
    dataset = ZarrDatasetWithPastAction(
        past_n=cfg.get("past_n", 7),
        zarr_path=dataset_path,
        obs_keys=list(cfg.shape_meta.obs.keys()),
        action_key="action",
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.horizon,
        seed=cfg.seed,
    )
    return dataset


@torch.no_grad()
def generate_actions(policy, obs_features, past_actions):
    """
    Run one forward pass given pre-computed obs features and past actions.
    Returns action_pred: (B, horizon, action_dim)
    Does NOT touch policy._past_buffer.
    """
    cond = policy._build_condition(obs_features, past_actions)
    B = obs_features.shape[0]

    action_tokens = torch.full(
        (B, 1), policy.bos_id,
        dtype=torch.long, device=policy.device,
    )
    action_tokens = policy.model.generate(
        action_tokens,
        cond=cond,
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
      action_A, action_B1, gt_clean, gt_detok
      each shape: (N, horizon, action_dim)
    """
    policy.eval()
    n_action_steps = policy.n_action_steps
    past_n = policy.past_n

    results = {k: [] for k in ["action_A", "action_B1", "gt_clean", "gt_detok"]}

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        # move to device
        action_gt   = batch["action"].to(device)          # (B, horizon, Da)
        past_action = batch["past_action"].to(device)     # (B, past_n, Da)
        obs_dict    = {k: v.to(device) for k, v in batch["obs"].items()
                       if isinstance(v, torch.Tensor)}

        # encode observations once (shared for A and B1)
        obs_features = policy.obs_encoder(obs_dict)       # (B, To, d)

        # ── Experiment A: GT past ─────────────────────────────────────────
        action_A = generate_actions(policy, obs_features, past_action)
        # (B, horizon, Da)

        # ── Experiment B1: policy's own past (one receding-horizon step) ──
        # Mimic receding horizon: first n_action_steps of action_A get
        # "executed", then we take the last past_n of those as the new past.
        if n_action_steps >= past_n:
            past_B1 = action_A[:, n_action_steps - past_n : n_action_steps]
        else:
            # edge case: slide old past + append new executed steps
            past_B1 = torch.cat([
                past_action[:, n_action_steps:],
                action_A[:, :n_action_steps],
            ], dim=1)
        # past_B1: (B, past_n, Da)

        action_B1 = generate_actions(policy, obs_features, past_B1)
        # (B, horizon, Da)

        # ── GT round-trip ─────────────────────────────────────────────────
        gt_detok = policy.action_tokenizer.autoencode(action_gt)
        # (B, horizon, Da)

        results["action_A"].append(action_A.cpu())
        results["action_B1"].append(action_B1.cpu())
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
      per_step: (horizon,)  — MSE averaged over N and Da
      aggregate: scalar
    """
    err = (pred - target).pow(2)          # (N, horizon, Da)
    per_step = err.mean(dim=(0, 2))       # (horizon,)
    aggregate = err.mean().item()
    return per_step.numpy(), aggregate


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(metrics, output_dir: pathlib.Path):
    horizon = len(next(iter(metrics.values()))["per_step"])
    steps = np.arange(horizon)

    comparisons = [
        ("A_vs_gt_clean",    "action_A vs GT (clean)",     "tab:blue"),
        ("A_vs_gt_rt",       "action_A vs GT (detok)", "tab:cyan"),
        ("B1_vs_gt_clean",   "action_B1 vs GT (clean)",    "tab:orange"),
        ("B1_vs_gt_rt",      "action_B1 vs GT (detok)","tab:red"),
        ("A_vs_B1",          "action_A vs action_B1",      "tab:green"),
    ]

    # ── Per-timestep MSE plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in comparisons:
        if key in metrics:
            ax.plot(steps, metrics[key]["per_step"], label=label, color=color)
    ax.axvline(x=15, color="gray", linestyle="--", alpha=0.5, label="n_action_steps boundary")
    ax.set_xlabel("Timestep within predicted horizon")
    ax.set_ylabel("MSE")
    ax.set_title("Per-timestep MSE: Effect of Past Action Conditioning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_step_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_step_mse.png'}")

    # ── Aggregate bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    agg_labels = []
    agg_vals = []
    agg_colors = []
    for key, label, color in comparisons:
        if key in metrics:
            agg_labels.append(label)
            agg_vals.append(metrics[key]["aggregate"])
            agg_colors.append(color)

    bars = ax.bar(range(len(agg_vals)), agg_vals, color=agg_colors)
    ax.set_xticks(range(len(agg_labels)))
    ax.set_xticklabels(agg_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("Aggregate MSE by Comparison")
    for bar, val in zip(bars, agg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.5f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'aggregate_mse.png'}")


def print_summary(metrics):
    print("\n" + "=" * 65)
    print(f"{'Comparison':<35} {'Aggregate MSE':>15} {'RMSE':>12}")
    print("=" * 65)
    labels = {
        "A_vs_gt_clean":  "action_A vs GT (clean)",
        "A_vs_gt_rt":     "action_A vs GT (detok)",
        "B1_vs_gt_clean": "action_B1 vs GT (clean)",
        "B1_vs_gt_rt":    "action_B1 vs GT (detok)",
        "A_vs_B1":        "action_A vs action_B1",
    }
    for key, label in labels.items():
        if key in metrics:
            mse = metrics[key]["aggregate"]
            print(f"  {label:<33} {mse:>15.6f} {mse**0.5:>12.6f}")
    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./past_condition_results")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policy ───────────────────────────────────────────────────────
    print(f"Loading policy from {args.ckpt} ...")
    policy, cfg = load_policy(args.ckpt, args.device)

    # ── Load dataset + normalizer ─────────────────────────────────────────
    print(f"Loading dataset from {args.dataset_path} ...")
    dataset = load_dataset(cfg, args.dataset_path)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(args.device)
    policy.eval()
    print(f"Policy: {policy.get_policy_name()}")
    print(f"  n_action_steps={policy.n_action_steps}, past_n={policy.past_n}, horizon={policy.max_seq_len}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Collect ───────────────────────────────────────────────────────────
    print(f"\nCollecting over {args.num_batches} batches ...")
    data = collect(policy, dataloader, args.num_batches, args.device)

    N = data["action_A"].shape[0]
    print(f"Collected {N} samples")

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = {}
    metrics["A_vs_gt_clean"]  = dict(zip(["per_step", "aggregate"],
                                    compute_mse(data["action_A"],  data["gt_clean"])))
    metrics["A_vs_gt_rt"]     = dict(zip(["per_step", "aggregate"],
                                    compute_mse(data["action_A"],  data["gt_detok"])))
    metrics["B1_vs_gt_clean"] = dict(zip(["per_step", "aggregate"],
                                    compute_mse(data["action_B1"], data["gt_clean"])))
    metrics["B1_vs_gt_rt"]    = dict(zip(["per_step", "aggregate"],
                                    compute_mse(data["action_B1"], data["gt_detok"])))
    metrics["A_vs_B1"]        = dict(zip(["per_step", "aggregate"],
                                    compute_mse(data["action_A"],  data["action_B1"])))

    # ── Print + plot ──────────────────────────────────────────────────────
    print_summary(metrics)
    plot_results(metrics, output_dir)

    # save per-step arrays as csv
    csv_path = output_dir / "per_step_mse.csv"
    horizon = data["action_A"].shape[1]
    header = "step," + ",".join(metrics.keys())
    rows = np.column_stack([np.arange(horizon)] + [metrics[k]["per_step"] for k in metrics])
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

"""
Fair open-loop MSE comparison between Original OAT and OATPolicyWithEnrichedPast
by running both through the actual LIBERO simulator.

For each LIBERO-10 task we run N rollout episodes per policy, record the actions
each policy executes at every timestep, align them by timestep index against GT
demonstration actions from the zarr dataset, and compute MSE.

Outputs:
  - Aggregate MSE table (OAT vs EnrichedPast)
  - Per-timestep MSE curves (absolute step in episode)
  - Per-chunk-position MSE curves (step within each n_action_steps chunk)
  - Success rate per policy (sanity check)
  - PNG plots and CSV data files

Usage:
  python analysis_scripts/compare_rollout_mse_sim.py \
      --ckpt_oat <path> \
      --ckpt_enriched <path> \
      --dataset_path data/libero/libero10_N500.zarr \
      --num_episodes 5 \
      --device cuda:0 \
      --output_dir ./rollout_mse_results
"""

import argparse
import pathlib
import math
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import zarr
from collections import defaultdict
from omegaconf import OmegaConf
import hydra

from oat.env.libero.env import LiberoEnv
from oat.env.libero.factory import get_subtasks
from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.common.pytorch_util import dict_apply


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str):
    """Load a policy checkpoint and its config."""
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


def maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


def load_gt_actions_per_task(zarr_path: str):
    """
    Load all GT action episodes from the zarr dataset, grouped by task_uid.

    Returns:
        dict[int, list[np.ndarray]]: task_uid -> list of (ep_len, action_dim)
    """
    root = zarr.open(zarr_path, mode="r")
    actions = root["data"]["action"][:]          # (total_steps, action_dim)
    task_uids = root["data"]["task_uid"][:]      # (total_steps, 1)
    episode_ends = root["meta"]["episode_ends"][:].astype(int)

    gt_by_task = defaultdict(list)
    start = 0
    for end in episode_ends:
        ep_actions = actions[start:end]
        ep_uid = int(task_uids[start, 0])
        gt_by_task[ep_uid].append(ep_actions.astype(np.float32))
        start = end
    return gt_by_task


def make_env(task_name, seed, n_obs_steps, n_action_steps, max_episode_steps):
    """Create a single LIBERO env wrapped with MultiStepWrapper."""
    env = MultiStepWrapper(
        LiberoEnv(
            task_name=task_name,
            image_size=128,
            seed=seed,
            camera_names=["agentview", "robot0_eye_in_hand"],
            state_ports=[
                "robot0_joint_pos",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ],
            max_episode_steps=max_episode_steps,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        reward_agg_method="max",
    )
    return env


# ── Rollout ────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def rollout_episode(policy, env, device, dtype, max_episode_steps):
    """
    Run one rollout episode.

    Returns:
        executed_actions: list of np.ndarray, each (action_dim,) — one per sim step
        success: bool
    """
    obs, _ = env.reset()
    policy.reset()

    executed_actions = []
    done = False
    total_steps = 0

    while not done and total_steps < max_episode_steps:
        # Add batch dimension: single env returns (To, ...), policy expects (B, To, ...)
        def _to_batched_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device=device, dtype=dtype).unsqueeze(0)
            return x

        obs_dict = dict_apply(obs, _to_batched_torch)

        # Get action chunk from policy
        result = policy.predict_action(
            {port: obs_dict[port] for port in policy.get_observation_ports()}
        )
        action_chunk = result["action"].detach().cpu().numpy()  # (1, n_action_steps, Da)
        action_chunk = action_chunk[0]  # (n_action_steps, Da)

        # Count how many steps the env will actually execute before we step,
        # by checking the done state the same way MultiStepWrapper does.
        steps_before = len(env.reward)

        # Step the env with the full chunk (MultiStepWrapper handles sequential exec)
        obs, reward, done, _, _ = env.step(action_chunk)

        steps_after = len(env.reward)
        n_exec = steps_after - steps_before

        # Record the actions that were actually executed
        for i in range(n_exec):
            executed_actions.append(action_chunk[i].copy())
        total_steps += n_exec

    success = any(r >= 1.0 for r in env.reward) if env.reward else False
    return executed_actions, success


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Open-loop MSE comparison via LIBERO simulator rollouts"
    )
    parser.add_argument("--ckpt_oat", required=True, help="Original OAT checkpoint")
    parser.add_argument("--ckpt_enriched", required=True, help="EnrichedPast checkpoint")
    parser.add_argument("--dataset_path", required=True, help="Path to zarr dataset")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Episodes per task per policy")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./rollout_mse_results")
    parser.add_argument("--max_episode_steps", type=int, default=550)
    parser.add_argument("--seed_start", type=int, default=1000,
                        help="Starting seed for env (same seeds used for both policies)")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load policies ──────────────────────────────────────────────────────
    print(f"Loading Original OAT from {args.ckpt_oat} ...")
    policy_oat, cfg_oat = load_policy(args.ckpt_oat, args.device)

    print(f"Loading Enriched Past from {args.ckpt_enriched} ...")
    policy_enriched, cfg_enriched = load_policy(args.ckpt_enriched, args.device)

    # ── Load dataset for normalizer and GT actions ─────────────────────────
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

    normalizer = dataset.get_normalizer()
    policy_oat.set_normalizer(normalizer)
    policy_enriched.set_normalizer(normalizer)
    policy_oat.to(args.device).eval()
    policy_enriched.to(args.device).eval()

    device = policy_oat.device
    dtype = policy_oat.dtype

    n_obs_steps = cfg_oat.n_obs_steps
    n_action_steps_oat = cfg_oat.n_action_steps
    n_action_steps_enr = cfg_enriched.n_action_steps

    print(f"OAT:      n_action_steps={n_action_steps_oat}, "
          f"n_obs_steps={n_obs_steps}")
    print(f"Enriched: n_action_steps={n_action_steps_enr}, "
          f"past_n={cfg_enriched.get('past_n', 7)}")

    # ── Load GT actions grouped by task_uid ────────────────────────────────
    print("Loading GT actions from zarr ...")
    gt_by_task = load_gt_actions_per_task(args.dataset_path)
    print(f"  Found {sum(len(v) for v in gt_by_task.values())} GT episodes "
          f"across {len(gt_by_task)} tasks")

    # ── Get LIBERO-10 subtasks ─────────────────────────────────────────────
    task_name = cfg_oat.get("task_name", "libero10")
    subtask_names = get_subtasks(task_name)
    print(f"\nRunning rollouts on {len(subtask_names)} subtasks, "
          f"{args.num_episodes} episodes each ...")

    # ── Build task_name -> task_uid mapping ────────────────────────────────
    from oat.env.libero.env import task_name_to_suite_and_ids
    task_name_to_uid = {
        name: task_name_to_suite_and_ids[name][2] for name in subtask_names
    }

    # ── Run rollouts ──────────────────────────────────────────────────────
    # Both policies use the SAME tokenizer detokenize output space.
    # The tokenizer's detokenize() returns UN-normalized actions
    # (it calls normalizer['action'].unnormalize internally).
    # The GT actions in the zarr are also un-normalized.
    # So both are in the same (raw) action space — no conversion needed.

    all_results = {
        "oat": {"mse_per_step": [], "mse_per_chunk_pos": [], "success": []},
        "enriched": {"mse_per_step": [], "mse_per_chunk_pos": [], "success": []},
    }

    policies = {
        "oat": (policy_oat, n_action_steps_oat),
        "enriched": (policy_enriched, n_action_steps_enr),
    }

    for task_idx, subtask_name in enumerate(subtask_names):
        task_uid = task_name_to_uid[subtask_name]
        gt_episodes = gt_by_task.get(task_uid, [])
        if len(gt_episodes) == 0:
            print(f"  WARNING: No GT episodes for task {subtask_name} (uid={task_uid})")
            continue

        print(f"\n[{task_idx+1}/{len(subtask_names)}] {subtask_name} "
              f"(uid={task_uid}, {len(gt_episodes)} GT demos)")

        for policy_name, (policy, n_act_steps) in policies.items():
            print(f"  Running {policy_name} ...")

            for ep_idx in range(args.num_episodes):
                seed = args.seed_start + task_idx * args.num_episodes + ep_idx

                env = make_env(
                    task_name=subtask_name,
                    seed=seed,
                    n_obs_steps=n_obs_steps,
                    n_action_steps=n_act_steps,
                    max_episode_steps=args.max_episode_steps,
                )

                try:
                    executed_actions, success = rollout_episode(
                        policy, env, device, dtype, args.max_episode_steps,
                    )
                finally:
                    env.close()

                all_results[policy_name]["success"].append(success)

                if len(executed_actions) == 0:
                    continue

                exec_arr = np.stack(executed_actions, axis=0)  # (T_exec, Da)

                # ── Align with GT by averaging over all GT demos for this task ──
                # Use the shortest length for alignment
                gt_mses = []
                for gt_ep in gt_episodes:
                    T_align = min(len(exec_arr), len(gt_ep))
                    if T_align == 0:
                        continue
                    mse_per_t = np.mean(
                        (exec_arr[:T_align] - gt_ep[:T_align]) ** 2, axis=-1
                    )  # (T_align,)
                    gt_mses.append(mse_per_t)

                # Average MSE across all GT demos (per-timestep)
                if gt_mses:
                    # Pad to max length across GT comparisons, then average
                    max_len = max(len(m) for m in gt_mses)
                    padded = np.full((len(gt_mses), max_len), np.nan)
                    for i, m in enumerate(gt_mses):
                        padded[i, :len(m)] = m
                    avg_mse_per_step = np.nanmean(padded, axis=0)  # (max_len,)
                    all_results[policy_name]["mse_per_step"].append(avg_mse_per_step)

                    # ── Per-chunk-position MSE ────────────────────────────────
                    # Reshape executed actions into chunks of n_action_steps
                    n_full_chunks = len(exec_arr) // n_act_steps
                    if n_full_chunks > 0:
                        # Also align GT the same way
                        chunk_mses = []  # list of (n_act_steps,)
                        for gt_ep in gt_episodes:
                            T_align = min(
                                n_full_chunks * n_act_steps, len(gt_ep)
                            )
                            n_usable_chunks = T_align // n_act_steps
                            if n_usable_chunks == 0:
                                continue
                            T_use = n_usable_chunks * n_act_steps
                            exec_chunks = exec_arr[:T_use].reshape(
                                n_usable_chunks, n_act_steps, -1
                            )
                            gt_chunks = gt_ep[:T_use].reshape(
                                n_usable_chunks, n_act_steps, -1
                            )
                            # MSE per chunk position, averaged across chunks
                            mse_per_pos = np.mean(
                                (exec_chunks - gt_chunks) ** 2, axis=(0, 2)
                            )  # (n_act_steps,)
                            chunk_mses.append(mse_per_pos)
                        if chunk_mses:
                            all_results[policy_name]["mse_per_chunk_pos"].append(
                                np.mean(chunk_mses, axis=0)
                            )

                print(f"    ep {ep_idx+1}/{args.num_episodes}: "
                      f"steps={len(executed_actions)}, success={success}")

    # ── Compute aggregate metrics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name in ["oat", "enriched"]:
        sr = np.mean(all_results[name]["success"]) if all_results[name]["success"] else 0.0
        mse_steps = all_results[name]["mse_per_step"]
        if mse_steps:
            # Pad to max length and compute mean
            max_len = max(len(m) for m in mse_steps)
            padded = np.full((len(mse_steps), max_len), np.nan)
            for i, m in enumerate(mse_steps):
                padded[i, :len(m)] = m
            mean_mse_per_step = np.nanmean(padded, axis=0)
            agg_mse = np.nanmean(mean_mse_per_step)
        else:
            mean_mse_per_step = np.array([])
            agg_mse = float("nan")

        all_results[name]["mean_mse_per_step"] = mean_mse_per_step
        all_results[name]["agg_mse"] = agg_mse
        all_results[name]["success_rate"] = sr

        label = "Original OAT" if name == "oat" else "EnrichedPast"
        print(f"\n  {label}:")
        print(f"    Aggregate MSE:  {agg_mse:.6f}")
        print(f"    RMSE:           {agg_mse**0.5:.6f}" if not np.isnan(agg_mse)
              else "    RMSE:           N/A")
        print(f"    Success Rate:   {sr:.3f}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print(f"{'Policy':<20} {'Agg MSE':>12} {'RMSE':>12} {'Success':>10}")
    print("-" * 60)
    for name, label in [("oat", "Original OAT"), ("enriched", "EnrichedPast")]:
        mse = all_results[name]["agg_mse"]
        sr = all_results[name]["success_rate"]
        rmse = mse ** 0.5 if not np.isnan(mse) else float("nan")
        print(f"  {label:<18} {mse:>12.6f} {rmse:>12.6f} {sr:>10.3f}")
    print("-" * 60)

    # ── Plot 1: Per-timestep MSE ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"oat": "tab:blue", "enriched": "tab:orange"}
    labels = {"oat": "Original OAT", "enriched": "EnrichedPast"}

    for name in ["oat", "enriched"]:
        mse = all_results[name]["mean_mse_per_step"]
        if len(mse) > 0:
            ax.plot(np.arange(len(mse)), mse, label=labels[name],
                    color=colors[name], alpha=0.8)

    # Mark n_action_steps boundaries
    n_act = max(n_action_steps_oat, n_action_steps_enr)
    max_t = max(
        len(all_results["oat"]["mean_mse_per_step"]),
        len(all_results["enriched"]["mean_mse_per_step"]),
    )
    for k in range(1, max_t // n_act + 1):
        ax.axvline(x=k * n_act, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel("Timestep (absolute)")
    ax.set_ylabel("MSE (vs GT)")
    ax.set_title("Per-Timestep Open-Loop MSE: Rollout Actions vs GT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_timestep_mse.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'per_timestep_mse.png'}")

    # ── Plot 2: Per-chunk-position MSE ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["oat", "enriched"]:
        chunk_mses = all_results[name]["mse_per_chunk_pos"]
        if chunk_mses:
            n_act_steps = policies[name][1]
            # All entries should have same length (n_action_steps)
            mean_chunk_mse = np.mean(chunk_mses, axis=0)
            ax.plot(np.arange(len(mean_chunk_mse)), mean_chunk_mse,
                    label=labels[name], color=colors[name], marker="o",
                    markersize=3, alpha=0.8)

    ax.set_xlabel("Step within action chunk")
    ax.set_ylabel("MSE (vs GT)")
    ax.set_title("Per-Chunk-Position MSE: Rollout Actions vs GT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_chunk_position_mse.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'per_chunk_position_mse.png'}")

    # ── Plot 3: Aggregate bar chart ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # MSE bar
    ax = axes[0]
    names_list = ["oat", "enriched"]
    bar_labels = [labels[n] for n in names_list]
    bar_vals = [all_results[n]["agg_mse"] for n in names_list]
    bar_colors = [colors[n] for n in names_list]
    bars = ax.bar(range(len(bar_vals)), bar_vals, color=bar_colors)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("MSE")
    ax.set_title("Aggregate MSE (Rollout vs GT)")
    for bar, val in zip(bars, bar_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    # Success rate bar
    ax = axes[1]
    sr_vals = [all_results[n]["success_rate"] for n in names_list]
    bars = ax.bar(range(len(sr_vals)), sr_vals, color=bar_colors)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate (Sanity Check)")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, sr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_mse_and_success.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'aggregate_mse_and_success.png'}")

    # ── Save CSV: per-timestep ─────────────────────────────────────────────
    max_len = max(
        len(all_results["oat"]["mean_mse_per_step"]),
        len(all_results["enriched"]["mean_mse_per_step"]),
    )
    if max_len > 0:
        csv_data = np.full((max_len, 3), np.nan)
        csv_data[:, 0] = np.arange(max_len)
        oat_mse = all_results["oat"]["mean_mse_per_step"]
        enr_mse = all_results["enriched"]["mean_mse_per_step"]
        csv_data[:len(oat_mse), 1] = oat_mse
        csv_data[:len(enr_mse), 2] = enr_mse
        csv_path = output_dir / "per_timestep_mse.csv"
        np.savetxt(csv_path, csv_data, delimiter=",",
                   header="timestep,oat_mse,enriched_mse", comments="")
        print(f"Saved: {csv_path}")

    # ── Save CSV: per-chunk-position ───────────────────────────────────────
    for name in ["oat", "enriched"]:
        chunk_mses = all_results[name]["mse_per_chunk_pos"]
        if chunk_mses:
            mean_chunk_mse = np.mean(chunk_mses, axis=0)
            csv_path = output_dir / f"per_chunk_position_mse_{name}.csv"
            csv_data = np.column_stack([
                np.arange(len(mean_chunk_mse)), mean_chunk_mse
            ])
            np.savetxt(csv_path, csv_data, delimiter=",",
                       header="chunk_position,mse", comments="")
            print(f"Saved: {csv_path}")

    # ── Save CSV: summary ──────────────────────────────────────────────────
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("policy,aggregate_mse,rmse,success_rate\n")
        for name, label in [("oat", "Original OAT"), ("enriched", "EnrichedPast")]:
            mse = all_results[name]["agg_mse"]
            rmse = mse ** 0.5 if not np.isnan(mse) else float("nan")
            sr = all_results[name]["success_rate"]
            f.write(f"{label},{mse:.6f},{rmse:.6f},{sr:.3f}\n")
    print(f"Saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

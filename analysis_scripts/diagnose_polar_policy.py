#!/usr/bin/env python3
"""Diagnose PolarOATPolicy performance drop vs baseline OATPolicy."""

import sys
import os
sys.path.insert(0, "/workspace/oat")
os.chdir("/workspace/oat")

import torch
import torch.nn.functional as F
import dill
import hydra
from torch.utils.data import DataLoader

from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply
from oat.dataset.base_dataset import BaseDataset

register_new_resolvers()

POLAR_CKPT = "output/20260320/053027_train_polar_oatpolicy_libero10_N500/checkpoints/ep-0100_sr-0.150.ckpt"
ORIG_CKPT = "output/20260316/191419_train_oatpolicy_libero10_N500/checkpoints/ep-0100_sr-0.516.ckpt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load PolarOATPolicy ─────────────────────────────────────────────────────
print("Loading PolarOATPolicy...")
from oat.policy.base_policy import BasePolicy
polar_policy = BasePolicy.from_checkpoint(POLAR_CKPT)
polar_policy.eval().to(device)
print(f"  vocab_sizes: {polar_policy.vocab_sizes}")

# ── Load Original OATPolicy ─────────────────────────────────────────────────
print("Loading Original OATPolicy...")
orig_policy = BasePolicy.from_checkpoint(ORIG_CKPT)
orig_policy.eval().to(device)
print(f"  codebook_size: {orig_policy.action_tokenizer.quantizer.codebook_size}")

# ── Load validation dataset ─────────────────────────────────────────────────
print("Loading validation dataset...")
polar_payload = torch.load(open(POLAR_CKPT, 'rb'), pickle_module=dill, map_location='cpu')
polar_cfg = polar_payload['cfg']
dataset = hydra.utils.instantiate(polar_cfg.task.policy.dataset)
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2,
                            pin_memory=True, drop_last=False)
print(f"  Val dataset: {len(val_dataset)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Per-Head Accuracy for PolarOATPolicy
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Per-Head Prediction Accuracy (PolarOATPolicy)")
print("=" * 70)

per_head_correct = {name: 0 for name in polar_policy.vocab_sizes}
per_head_total = {name: 0 for name in polar_policy.vocab_sizes}
per_head_losses = {name: 0.0 for name in polar_policy.vocab_sizes}

# For joint analysis
all_joint_correct = 0
all_joint_total = 0

with torch.inference_mode():
    for batch in val_dataloader:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

        token_dict = polar_policy.action_tokenizer.tokenize(batch['action'])
        features = polar_policy.obs_encoder(batch['obs'])

        B = batch['action'].shape[0]

        input_dict = {}
        for name in polar_policy.vocab_sizes:
            bos = torch.full((B, 1), polar_policy.bos_ids[name], dtype=torch.long, device=device)
            input_dict[name] = torch.cat([bos, token_dict[name]], dim=1)[:, :-1]

        logits_dict = polar_policy.model(input_dict, cond=features)

        joint_correct = None
        for name in polar_policy.vocab_sizes:
            logits = logits_dict[name]
            targets = token_dict[name]
            preds = logits.argmax(dim=-1)
            head_correct = (preds == targets)

            per_head_correct[name] += head_correct.sum().item()
            per_head_total[name] += targets.numel()

            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            per_head_losses[name] += loss.item() * targets.numel()

            if joint_correct is None:
                joint_correct = head_correct
            else:
                joint_correct = joint_correct & head_correct

        all_joint_correct += joint_correct.sum().item()
        all_joint_total += joint_correct.numel()

print(f"\n{'Head':>15s}  {'Accuracy':>8s}  {'Loss':>8s}  {'Vocab':>6s}")
print("-" * 50)
expected_independent = 1.0
for name in polar_policy.vocab_sizes:
    acc = per_head_correct[name] / per_head_total[name]
    avg_loss = per_head_losses[name] / per_head_total[name]
    expected_independent *= acc
    print(f"{name:>15s}  {acc:>8.4f}  {avg_loss:>8.4f}  {polar_policy.vocab_sizes[name]:>6d}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Original OATPolicy Baseline Accuracy
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Original OATPolicy Baseline Accuracy")
print("=" * 70)

orig_correct = 0
orig_total = 0
orig_loss_sum = 0.0

with torch.inference_mode():
    for batch in val_dataloader:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

        token = orig_policy.action_tokenizer.tokenize(batch['action'])
        features = orig_policy.obs_encoder(batch['obs'])

        B = batch['action'].shape[0]
        bos = torch.full((B, 1), orig_policy.bos_id, dtype=torch.long, device=device)
        input_tokens = torch.cat([bos, token], dim=1)[:, :-1]
        logits = orig_policy.model(input_tokens, cond=features)

        preds = logits.argmax(dim=-1)
        orig_correct += (preds == token).sum().item()
        orig_total += token.numel()

        vocab_size = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), token.reshape(-1))
        orig_loss_sum += loss.item() * token.numel()

orig_acc = orig_correct / orig_total
orig_avg_loss = orig_loss_sum / orig_total
print(f"\n  Single head acc: {orig_acc:.4f}  loss: {orig_avg_loss:.4f}  vocab: {orig_policy.action_tokenizer.quantizer.codebook_size}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Joint Accuracy Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Joint Accuracy Analysis")
print("=" * 70)

joint_acc = all_joint_correct / all_joint_total
print(f"\n  Joint accuracy (all heads correct):  {joint_acc:.4f}")
print(f"  Expected if independent:             {expected_independent:.4f}")
ratio = joint_acc / max(expected_independent, 1e-8)
print(f"  Ratio (>1 = positive correlation):   {ratio:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Per-Head Gradient Magnitude
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Per-Head Gradient Magnitude")
print("=" * 70)

polar_policy.train()
# Get one batch
batch = next(iter(val_dataloader))
batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

with torch.no_grad():
    token_dict = polar_policy.action_tokenizer.tokenize(batch['action'])
features = polar_policy.obs_encoder(batch['obs'])

B = batch['action'].shape[0]
input_dict = {}
for name in polar_policy.vocab_sizes:
    bos = torch.full((B, 1), polar_policy.bos_ids[name], dtype=torch.long, device=device)
    input_dict[name] = torch.cat([bos, token_dict[name]], dim=1)[:, :-1]

logits_dict = polar_policy.model(input_dict, cond=features)

print(f"\n{'Head':>15s}  {'Loss':>8s}  {'Backbone Grad Norm':>20s}  {'Head Grad Norm':>16s}")
print("-" * 70)

for name in polar_policy.vocab_sizes:
    polar_policy.zero_grad()
    logits = logits_dict[name]
    targets = token_dict[name]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward(retain_graph=True)

    # Backbone gradient norm
    backbone_grad = 0.0
    for p in polar_policy.model.blocks.parameters():
        if p.grad is not None:
            backbone_grad += p.grad.norm().item() ** 2
    backbone_grad = backbone_grad ** 0.5

    # Head-specific gradient norm
    head_grad = 0.0
    for p in polar_policy.model.heads[name].parameters():
        if p.grad is not None:
            head_grad += p.grad.norm().item() ** 2
    head_grad = head_grad ** 0.5

    print(f"{name:>15s}  {loss.item():>8.4f}  {backbone_grad:>20.6f}  {head_grad:>16.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Per-Position Accuracy (early vs late in sequence)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Per-Position Accuracy (first 8 vs last 8 tokens)")
print("=" * 70)

polar_policy.eval()
early_correct = {name: 0 for name in polar_policy.vocab_sizes}
late_correct = {name: 0 for name in polar_policy.vocab_sizes}
early_total = 0
late_total = 0

with torch.inference_mode():
    for batch in val_dataloader:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        token_dict = polar_policy.action_tokenizer.tokenize(batch['action'])
        features = polar_policy.obs_encoder(batch['obs'])
        B = batch['action'].shape[0]

        input_dict = {}
        for name in polar_policy.vocab_sizes:
            bos = torch.full((B, 1), polar_policy.bos_ids[name], dtype=torch.long, device=device)
            input_dict[name] = torch.cat([bos, token_dict[name]], dim=1)[:, :-1]

        logits_dict = polar_policy.model(input_dict, cond=features)

        for name in polar_policy.vocab_sizes:
            preds = logits_dict[name].argmax(dim=-1)
            targets = token_dict[name]
            early_correct[name] += (preds[:, :8] == targets[:, :8]).sum().item()
            late_correct[name] += (preds[:, -8:] == targets[:, -8:]).sum().item()
        early_total += B * 8
        late_total += B * 8

print(f"\n{'Head':>15s}  {'Early (t<8)':>12s}  {'Late (t>24)':>12s}  {'Delta':>8s}")
print("-" * 55)
for name in polar_policy.vocab_sizes:
    e_acc = early_correct[name] / early_total
    l_acc = late_correct[name] / late_total
    print(f"{name:>15s}  {e_acc:>12.4f}  {l_acc:>12.4f}  {e_acc - l_acc:>+8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSIS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

inv_acc = per_head_correct['inv'] / per_head_total['inv']
angular_accs = {name: per_head_correct[name] / per_head_total[name]
                for name in polar_policy.vocab_sizes if name != 'inv'}
avg_angular_acc = sum(angular_accs.values()) / len(angular_accs)

print(f"""
Original OATPolicy single-head acc:  {orig_acc:.4f}
PolarOATPolicy inv head acc:         {inv_acc:.4f}
PolarOATPolicy avg angular head acc: {avg_angular_acc:.4f}
PolarOATPolicy joint acc:            {joint_acc:.4f}

Interpretation:
""")

if inv_acc < orig_acc * 0.7:
    print("  [!] inv accuracy significantly LOWER than original")
    print("      -> Embedding interference from summing is likely hurting inv predictions")
else:
    print("  [OK] inv accuracy comparable to original")

if avg_angular_acc < 0.5:
    print("  [!] Angular heads have LOW accuracy")
    print("      -> Angular subspaces may need more capacity or loss upweighting")
elif avg_angular_acc > 0.8:
    print("  [OK] Angular heads have high accuracy")

if ratio > 1.5:
    print("  [OK] Joint accuracy higher than expected -> heads are positively correlated")
elif ratio < 0.5:
    print("  [!] Joint accuracy much LOWER than expected -> subspace predictions conflict")
else:
    print("  [~] Joint accuracy roughly matches independent expectation")

# Check gradient imbalance
print(f"""
Decision guide:
  - If inv acc << original acc -> (B) Embedding interference from summing
  - If angular heads low, inv OK -> (A) Loss imbalance (upweight angular heads)
  - If all heads low -> (C) Capacity issue or (B) interference
  - If joint << expected independent -> Subspace dependency (need C2 cascade)
  - If late >> worse than early -> AR error compounding over 32-step horizon
""")

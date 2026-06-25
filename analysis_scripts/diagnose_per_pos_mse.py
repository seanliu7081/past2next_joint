"""
Diagnostic: Per-position reconstruction MSE + basis analysis.

Answers three questions:
  1. Which frequency bands lose the most information through the full
     encode→quantize→decode pipeline?
  2. How far has the learned basis drifted from the DCT initialization?
  3. How well is orthogonality preserved?

Usage:
  python diagnose_per_position_mse.py \
      --checkpoint /path/to/tokenizer_checkpoint.pt \
      --dataset_path /path/to/zarr_dataset \
      --num_batches 50

Adapt the checkpoint loading section if your serialization format differs.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from oat.tokenizer.oat.tokenizer_spectral import OATTok  # adjust path if needed
from oat.dataset.zarr_dataset import ZarrDataset


def load_tokenizer(checkpoint_path: str, device: str = "cuda") -> OATTok:
    """
    Load trained tokenizer from checkpoint.

    Supports:
      1. Full model saved via torch.save(tokenizer, path)
      2. Workspace checkpoint: {'cfg', 'state_dicts', 'pickles'} — reconstructs
         OATTok from cfg and loads state_dicts['ema_model'] (falls back to 'model').
      3. Dict with {"model": state_dict, ...}
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Pattern 1: full model object
    if isinstance(ckpt, OATTok):
        tokenizer = ckpt
    # Pattern 2: workspace checkpoint with cfg + state_dicts
    elif isinstance(ckpt, dict) and "cfg" in ckpt and "state_dicts" in ckpt:
        from hydra._internal.utils import _locate
        tok_cfg = ckpt["cfg"]["tokenizer"]
        tokenizer = _locate(tok_cfg["_target_"])(
            encoder=_locate(tok_cfg["encoder"]["_target_"])(**{k: v for k, v in tok_cfg["encoder"].items() if k != "_target_"}),
            decoder=_locate(tok_cfg["decoder"]["_target_"])(**{k: v for k, v in tok_cfg["decoder"].items() if k != "_target_"}),
            quantizer=_locate(tok_cfg["quantizer"]["_target_"])(**{k: v for k, v in tok_cfg["quantizer"].items() if k != "_target_"}),
        )
        state_dicts = ckpt["state_dicts"]
        sd_key = "ema_model" if "ema_model" in state_dicts else "model"
        tokenizer.load_state_dict(state_dicts[sd_key])
    # Pattern 3: state_dict inside a dict
    elif isinstance(ckpt, dict) and "model" in ckpt:
        raise NotImplementedError(
            "Checkpoint contains a state_dict under 'model' key. "
            "You need to construct OATTok with the right config first, "
            "then call tokenizer.load_state_dict(ckpt['model'])."
        )
    # Pattern 4: raw state_dict
    elif isinstance(ckpt, dict) and any("encoder" in k for k in ckpt.keys()):
        raise NotImplementedError(
            "Checkpoint appears to be a raw state_dict. "
            "Construct OATTok with the right config first, "
            "then call tokenizer.load_state_dict(ckpt)."
        )
    else:
        raise ValueError(
            f"Unrecognized checkpoint format: type={type(ckpt)}, "
            f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
        )

    tokenizer.to(device).eval()
    return tokenizer


def load_dataloader(
    dataset_path: str,
    batch_size: int = 256,
    n_action_steps: int = 32,
):
    """
    Load ZarrDataset in tokenizer-training mode (no obs, action only).
    """
    dataset = ZarrDataset(
        zarr_path=dataset_path,
        obs_keys=[],            # tokenizer training: no observations
        action_key="action",
        n_obs_steps=1,          # minimum (won't be used)
        n_action_steps=n_action_steps,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Per-Position Reconstruction MSE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_per_position_recon_mse(
    tokenizer,
    dataloader,
    num_batches: int = -1,
    device: str = "cuda",
):
    """
    Measure reconstruction error per frequency band.

    Method:
      - For each batch, get normalized action and its reconstruction
        (also normalized, i.e. BEFORE unnormalize).
      - Project both into spectral domain using the SAME learned basis.
      - Compute MSE per position k=0..K-1.

    This tells you which frequency bands are losing information through
    the full encode→MLP→FSQ→decode pipeline.

    Returns:
        per_pos_mse:   [K] mean MSE per spectral position
        per_pos_mae:   [K] mean MAE per spectral position (for reference)
        total_mse:     scalar, overall recon MSE in normalized action space
        energy_per_pos:[K] mean spectral energy per position (denominator
                       for understanding error *relative to signal*)
    """
    encoder = tokenizer.encoder
    basis = encoder.basis.detach()  # [K, T]
    K = basis.shape[0]

    # Accumulators
    sum_mse = torch.zeros(K, device=device)
    sum_mae = torch.zeros(K, device=device)
    sum_energy = torch.zeros(K, device=device)
    sum_total_mse = 0.0
    n_samples = 0

    for i, batch in enumerate(dataloader):
        if num_batches > 0 and i >= num_batches:
            break

        action = batch["action"].to(device)  # [B, T, D]
        B = action.shape[0]

        # ── Normalized action ───────────────────────────────────────────
        action_norm = tokenizer.normalizer["action"].normalize(action)

        # ── Full pipeline reconstruction (normalized space) ─────────────
        latents = encoder(action_norm)                          # [B, K, 4]
        latents_q, _ = tokenizer.quantizer(latents)             # (quantized, tokens)
        recon_norm = tokenizer.decoder(latents_q)               # [B, T, D]

        # ── Project both to spectral domain ─────────────────────────────
        z_orig = torch.einsum("kt, btd -> bkd", basis, action_norm)   # [B, K, D]
        z_recon = torch.einsum("kt, btd -> bkd", basis, recon_norm)   # [B, K, D]

        # ── Per-position statistics ─────────────────────────────────────
        diff = z_orig - z_recon  # [B, K, D]

        # MSE: mean over batch and action dims, keep position dim
        sum_mse += (diff ** 2).mean(dim=(0, 2))       # [K]
        sum_mae += diff.abs().mean(dim=(0, 2))         # [K]

        # Energy: how much signal is in each band (for relative error)
        sum_energy += (z_orig ** 2).mean(dim=(0, 2))   # [K]

        # Overall MSE in action space
        sum_total_mse += ((action_norm - recon_norm) ** 2).mean().item() * B
        n_samples += B

    n_batches_used = min(i + 1, num_batches) if num_batches > 0 else i + 1

    per_pos_mse = (sum_mse / n_batches_used).cpu()
    per_pos_mae = (sum_mae / n_batches_used).cpu()
    energy_per_pos = (sum_energy / n_batches_used).cpu()
    total_mse = sum_total_mse / n_samples

    return {
        "per_pos_mse": per_pos_mse,           # [K]
        "per_pos_mae": per_pos_mae,           # [K]
        "energy_per_pos": energy_per_pos,     # [K]
        "relative_error": per_pos_mse / (energy_per_pos + 1e-8),  # [K]
        "total_mse": total_mse,               # scalar
        "mse_fraction": per_pos_mse / (per_pos_mse.sum() + 1e-8),  # [K], sums to 1
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2 & 3. Basis Analysis: Orthogonality + DCT Drift
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_basis(tokenizer):
    """
    Run encoder.basis_analysis() + additional derived diagnostics.

    Returns everything from basis_analysis() plus:
      - per_pair_gram: full off-diagonal breakdown
      - max_offdiag: worst-case off-diagonal element
      - drift_summary: per-basis angular deviation from DCT (degrees)
    """
    encoder = tokenizer.encoder
    result = encoder.basis_analysis()

    # ── Additional: angular deviation from DCT ──────────────────────────
    cos_sims = result["cosine_similarities"]
    # clamp for numerical safety before arccos
    angles_deg = torch.acos(cos_sims.clamp(-1, 1)) * 180.0 / np.pi

    # ── Additional: per-pair off-diagonal Gram entries ──────────────────
    gram = result["gram_matrix"]
    K = gram.shape[0]
    offdiag_pairs = {}
    for i in range(K):
        for j in range(i + 1, K):
            offdiag_pairs[f"({i},{j})"] = gram[i, j].item()

    # ── Additional: condition number of Gram matrix ─────────────────────
    eigvals = torch.linalg.eigvalsh(gram)
    cond_number = (eigvals.max() / eigvals.min().clamp(min=1e-8)).item()

    result["angles_from_dct_deg"] = angles_deg
    result["offdiag_pairs"] = offdiag_pairs
    result["max_offdiag"] = max(abs(v) for v in offdiag_pairs.values())
    result["gram_condition_number"] = cond_number
    result["gram_eigenvalues"] = eigvals.cpu()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretty Print
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(recon_results: dict, basis_results: dict):
    K = len(recon_results["per_pos_mse"])

    print("=" * 72)
    print("  DIAGNOSTIC REPORT: SpectralBasisEncoder")
    print("=" * 72)

    # ── Section 1: Per-position recon MSE ───────────────────────────────
    print("\n[1] Per-Position Reconstruction MSE (spectral domain)")
    print("-" * 72)
    print(f"  {'Pos':>4}  {'MSE':>10}  {'MAE':>10}  {'Energy':>10}  "
          f"{'RelErr':>10}  {'%Total':>8}")
    print("-" * 72)

    mse = recon_results["per_pos_mse"]
    mae = recon_results["per_pos_mae"]
    energy = recon_results["energy_per_pos"]
    rel_err = recon_results["relative_error"]
    frac = recon_results["mse_fraction"]

    for k in range(K):
        print(f"  k={k:<2}  {mse[k]:10.6f}  {mae[k]:10.6f}  {energy[k]:10.4f}  "
              f"{rel_err[k]:10.6f}  {frac[k]*100:7.1f}%")

    print("-" * 72)
    print(f"  Total action-space MSE: {recon_results['total_mse']:.6f}")
    print(f"  Sum of spectral MSE:    {mse.sum():.6f}")

    # Interpretation hints
    peak_k = mse.argmax().item()
    peak_rel_k = rel_err.argmax().item()
    print(f"\n  → Highest absolute error: k={peak_k} "
          f"({frac[peak_k]*100:.1f}% of total)")
    print(f"  → Highest relative error: k={peak_rel_k} "
          f"(error/energy = {rel_err[peak_rel_k]:.4f})")

    # ── Section 2: Basis orthogonality ──────────────────────────────────
    print(f"\n[2] Basis Orthogonality")
    print("-" * 72)

    norms = basis_results["basis_norms"]
    print(f"  Gram off-diag Frobenius norm: {basis_results['gram_offdiag_norm']:.6f}")
    print(f"  Max off-diagonal |entry|:     {basis_results['max_offdiag']:.6f}")
    print(f"  Gram condition number:        {basis_results['gram_condition_number']:.2f}")
    print(f"  Energy ordered:               {basis_results['energy_ordered']}")
    print(f"\n  Basis norms (ideal=1.0):")
    for k in range(K):
        bar = "█" * int(norms[k].item() * 30)
        print(f"    k={k}: {norms[k]:.4f}  {bar}")

    # ── Section 3: DCT drift ────────────────────────────────────────────
    print(f"\n[3] Drift from DCT Initialization")
    print("-" * 72)

    cos_sims = basis_results["cosine_similarities"]
    angles = basis_results["angles_from_dct_deg"]
    print(f"  {'Pos':>4}  {'cos_sim':>10}  {'angle(°)':>10}")
    print("-" * 72)
    for k in range(K):
        drift_indicator = "  ⚠" if angles[k] > 10.0 else ""
        print(f"  k={k:<2}  {cos_sims[k]:10.6f}  {angles[k]:10.2f}°{drift_indicator}")

    mean_angle = angles.mean().item()
    max_angle = angles.max().item()
    print(f"\n  Mean angular drift: {mean_angle:.2f}°")
    print(f"  Max angular drift:  {max_angle:.2f}° (k={angles.argmax().item()})")

    if max_angle < 5.0:
        print("  → Basis is very close to DCT. ortho_loss is holding tight.")
    elif max_angle < 15.0:
        print("  → Moderate drift. Basis is adapting while staying structured.")
    else:
        print("  → Significant drift. Consider whether ortho_weight=0.01 is too weak.")

    # ── Section 4: Top off-diagonal Gram pairs ──────────────────────────
    print(f"\n[4] Largest Off-Diagonal Gram Entries (basis coupling)")
    print("-" * 72)
    pairs = basis_results["offdiag_pairs"]
    sorted_pairs = sorted(pairs.items(), key=lambda x: abs(x[1]), reverse=True)
    for pair_name, val in sorted_pairs[:5]:
        severity = "⚠" if abs(val) > 0.05 else " "
        print(f"  {severity} basis pair {pair_name}: {val:+.6f}")

    print("\n" + "=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose SpectralBasisEncoder per-position MSE + basis health"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained tokenizer checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to zarr dataset")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of batches to evaluate (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_action_steps", type=int, default=32,
                        help="Action chunk length (sample_horizon)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.checkpoint} ...")
    tokenizer = load_tokenizer(args.checkpoint, device=args.device)

    print(f"Loading dataset from {args.dataset_path} ...")
    dataloader = load_dataloader(
        args.dataset_path,
        batch_size=args.batch_size,
        n_action_steps=args.n_action_steps,
    )

    print(f"Computing per-position recon MSE ({args.num_batches} batches) ...")
    recon_results = compute_per_position_recon_mse(
        tokenizer, dataloader,
        num_batches=args.num_batches,
        device=args.device,
    )

    print("Analyzing basis orthogonality and DCT drift ...")
    basis_results = analyze_basis(tokenizer)

    print_report(recon_results, basis_results)

    # ── Save raw results for further analysis ───────────────────────────
    save_path = Path(args.checkpoint).parent / "diagnostic_per_position.pt"
    torch.save({
        "recon": {k: v if isinstance(v, float) else v for k, v in recon_results.items()},
        "basis": basis_results,
    }, save_path)
    print(f"\nRaw results saved to {save_path}")
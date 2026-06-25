"""
Phase 1: Residual Statistics Analysis on Real LIBERO Data
==========================================================
Usage:
    python phase1_real_data.py --data_dir /path/to/libero/zarr/datasets
    python phase1_real_data.py --data_dir /path/to/libero/zarr/datasets --tasks task1.zarr task2.zarr
    python phase1_real_data.py --zarr_path /path/to/single_task.zarr

The script expects zarr datasets with:
  - 'data/action' : (N, action_dim) array of actions
  - 'meta/episode_ends' : (num_episodes,) array of episode end indices

This is the standard format used by the OAT/diffusion policy codebase.
"""

import numpy as np
import math
import argparse
import os
import glob
import json
from pathlib import Path
from collections import defaultdict

try:
    import zarr
except ImportError:
    print("Installing zarr...")
    os.system("pip install zarr --break-system-packages -q")
    import zarr

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    print("Warning: matplotlib not available, skipping plots")
    HAS_MPL = False

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_zarr_episodes(zarr_path):
    """
    Load action trajectories from a zarr dataset.
    
    Returns:
        episodes: list of (T_i, d) arrays, one per episode
        metadata: dict with dataset info
    """
    root = zarr.open(zarr_path, mode='r')
    
    # Try different zarr layouts
    if 'data' in root and 'action' in root['data']:
        # Standard layout: data/action, meta/episode_ends
        all_actions = np.array(root['data']['action'])
        episode_ends = np.array(root['meta']['episode_ends'])
    elif 'action' in root:
        # Flat layout
        all_actions = np.array(root['action'])
        if 'episode_ends' in root:
            episode_ends = np.array(root['episode_ends'])
        elif 'meta' in root and 'episode_ends' in root['meta']:
            episode_ends = np.array(root['meta']['episode_ends'])
        else:
            # Single episode
            episode_ends = np.array([len(all_actions)])
    else:
        raise ValueError(f"Cannot find actions in zarr at {zarr_path}. Keys: {list(root.keys())}")
    
    # Split into episodes
    episodes = []
    prev_end = 0
    for end in episode_ends:
        ep_actions = all_actions[prev_end:end]
        if len(ep_actions) > 1:  # skip empty episodes
            episodes.append(ep_actions)
        prev_end = end
    
    metadata = {
        'path': str(zarr_path),
        'total_steps': len(all_actions),
        'num_episodes': len(episodes),
        'action_dim': all_actions.shape[1],
        'episode_lengths': [len(ep) for ep in episodes],
        'mean_episode_length': np.mean([len(ep) for ep in episodes]),
    }
    
    return episodes, metadata


def load_all_datasets(data_dir, task_names=None):
    """Load episodes from all zarr datasets in a directory."""
    all_episodes = []
    all_metadata = []
    
    if task_names:
        zarr_paths = [os.path.join(data_dir, t) for t in task_names]
    else:
        zarr_paths = sorted(glob.glob(os.path.join(data_dir, '*.zarr')))
        if not zarr_paths:
            zarr_paths = sorted(glob.glob(os.path.join(data_dir, '*', '*.zarr')))
        if not zarr_paths:
            # Maybe data_dir itself is a zarr
            if os.path.exists(os.path.join(data_dir, '.zarray')) or os.path.exists(os.path.join(data_dir, '.zgroup')):
                zarr_paths = [data_dir]
    
    for zp in zarr_paths:
        try:
            episodes, meta = load_zarr_episodes(zp)
            all_episodes.extend(episodes)
            all_metadata.append(meta)
            print(f"  Loaded {meta['num_episodes']} episodes from {os.path.basename(zp)} "
                  f"(d={meta['action_dim']}, mean_len={meta['mean_episode_length']:.0f})")
        except Exception as e:
            print(f"  Warning: Failed to load {zp}: {e}")
    
    return all_episodes, all_metadata


# ============================================================================
# 2. PREDICTION STRATEGIES
# ============================================================================

def predict_zero_order(past, dt, H):
    """Repeat last action (zero-order hold)."""
    return np.tile(past[-1:], (H, 1))


def predict_first_order(past, dt, H):
    """Linear extrapolation from last 2 actions."""
    vel = (past[-1] - past[-2]) / dt
    pred = np.zeros((H, past.shape[1]))
    for h in range(H):
        pred[h] = past[-1] + vel * (h + 1) * dt
    return pred


def predict_linear_smooth(past, dt, H):
    """Linear extrapolation with weighted least squares velocity estimate."""
    K = len(past)
    t_past = np.arange(K) * dt
    
    # Exponential recency weighting
    weights = np.exp(np.arange(K) * 0.5)
    weights /= weights.sum()
    
    d = past.shape[1]
    vel = np.zeros(d)
    t_mean = np.sum(weights * t_past)
    
    for dim in range(d):
        p_mean = np.sum(weights * past[:, dim])
        num = np.sum(weights * (t_past - t_mean) * (past[:, dim] - p_mean))
        den = np.sum(weights * (t_past - t_mean)**2) + 1e-10
        vel[dim] = num / den
    
    pred = np.zeros((H, d))
    for h in range(H):
        pred[h] = past[-1] + vel * (h + 1) * dt
    return pred


def predict_quadratic_smooth(past, dt, H):
    """Quadratic extrapolation with weighted polynomial fit."""
    K = len(past)
    if K < 3:
        return predict_first_order(past, dt, H)
    
    t_past = np.arange(K) * dt
    weights = np.exp(np.arange(K) * 0.3)
    weights /= weights.sum()
    
    d = past.shape[1]
    pred = np.zeros((H, d))
    
    for dim in range(d):
        deg = min(2, K - 1)
        try:
            coeffs = np.polyfit(t_past, past[:, dim], deg=deg, w=weights)
            poly = np.poly1d(coeffs)
            for h in range(H):
                tau = t_past[-1] + (h + 1) * dt
                pred[h, dim] = poly(tau)
        except np.linalg.LinAlgError:
            # Fallback to linear
            pred[:, dim] = predict_first_order(past[-2:], dt, H)[:, dim]
    
    return pred


def predict_taylor_K(past, dt, H, K_order):
    """
    Explicit Taylor prediction of order K_order.
    Uses backward finite differences to estimate derivatives.
    """
    K = len(past)
    d = past.shape[1]
    
    if K_order > K:
        K_order = K
    
    # Compute finite difference derivatives
    derivs = [past[-1].copy()]  # 0th: position
    
    if K_order >= 1 and K >= 2:
        derivs.append((past[-1] - past[-2]) / dt)
    if K_order >= 2 and K >= 3:
        derivs.append((past[-1] - 2*past[-2] + past[-3]) / dt**2)
    if K_order >= 3 and K >= 4:
        derivs.append((past[-1] - 3*past[-2] + 3*past[-3] - past[-4]) / dt**3)
    
    pred = np.zeros((H, d))
    for h in range(H):
        tau = (h + 1) * dt
        p = np.zeros(d)
        for n, deriv in enumerate(derivs):
            p += deriv * (tau**n) / math.factorial(n)
        pred[h] = p
    
    return pred


# ============================================================================
# 3. COMPUTE RESIDUALS
# ============================================================================

def compute_residuals_for_dataset(episodes, dt, H, max_K=7):
    """
    Compute residuals for all prediction strategies across all episodes.
    
    Returns:
        results: dict with raw chunks and residuals per strategy
    """
    strategies = {
        'zero_order_K1': {'K': 1},
        'linear_K2': {'K': 2},
        'linear_smooth_K3': {'K': 3},
        'linear_smooth_K5': {'K': 5},
        'linear_smooth_K7': {'K': 7},
        'quad_smooth_K4': {'K': 4},
        'quad_smooth_K7': {'K': 7},
        'taylor_order1_K2': {'K': 2},
        'taylor_order2_K3': {'K': 3},
        'taylor_order3_K4': {'K': 4},
    }
    
    results = {name: {'residuals': [], 'predictions': []} for name in strategies}
    results['raw'] = {'chunks': []}
    
    # Per-episode tracking for episode-level analysis
    episode_stats = []
    
    for ep_idx, ep_actions in enumerate(episodes):
        T = len(ep_actions)
        if T < max_K + H + 1:
            continue
        
        ep_raw = []
        ep_residuals = {name: [] for name in strategies}
        
        for t in range(max_K, T - H):
            chunk = ep_actions[t:t+H]
            results['raw']['chunks'].append(chunk)
            ep_raw.append(chunk)
            
            for name, info in strategies.items():
                K = info['K']
                past = ep_actions[t-K:t]
                
                if 'zero_order' in name:
                    pred = predict_zero_order(past, dt, H)
                elif 'linear_smooth' in name:
                    pred = predict_linear_smooth(past, dt, H)
                elif 'quad_smooth' in name:
                    pred = predict_quadratic_smooth(past, dt, H)
                elif 'taylor_order1' in name:
                    pred = predict_taylor_K(past, dt, H, K_order=1)
                elif 'taylor_order2' in name:
                    pred = predict_taylor_K(past, dt, H, K_order=2)
                elif 'taylor_order3' in name:
                    pred = predict_taylor_K(past, dt, H, K_order=3)
                elif 'linear_K2' in name:
                    pred = predict_first_order(past, dt, H)
                else:
                    pred = predict_zero_order(past, dt, H)
                
                residual = chunk - pred
                results[name]['residuals'].append(residual)
                results[name]['predictions'].append(pred)
                ep_residuals[name].append(residual)
    
    # Convert to arrays
    results['raw']['chunks'] = np.array(results['raw']['chunks'])
    for name in strategies:
        results[name]['residuals'] = np.array(results[name]['residuals'])
        results[name]['predictions'] = np.array(results[name]['predictions'])
    
    return results, strategies


# ============================================================================
# 4. ANALYSIS
# ============================================================================

def analyze_all(results, strategies, d, H):
    """Run all analyses on computed residuals."""
    
    raw = results['raw']['chunks']  # (N, H, d)
    N = len(raw)
    
    analysis = {}
    
    # --- Raw action statistics ---
    flat_raw = raw.reshape(-1, d)
    raw_var = np.var(flat_raw, axis=0)
    raw_std = np.sqrt(raw_var)
    raw_total_var = np.sum(raw_var)
    
    # PCA of raw
    cov_raw = np.cov(flat_raw.T)
    eigvals_raw = np.sort(np.linalg.eigvalsh(cov_raw))[::-1]
    eigvals_raw = np.maximum(eigvals_raw, 0)
    cumsum_raw = np.cumsum(eigvals_raw) / (eigvals_raw.sum() + 1e-20)
    
    analysis['raw'] = {
        'std_per_dim': raw_std,
        'var_per_dim': raw_var,
        'total_var': raw_total_var,
        'mean_std': np.mean(raw_std),
        'eigvals': eigvals_raw,
        'explained_var': eigvals_raw / (eigvals_raw.sum() + 1e-20),
        'pr': eigvals_raw.sum()**2 / (np.sum(eigvals_raw**2) + 1e-20),
        'd_eff_95': int(np.searchsorted(cumsum_raw, 0.95) + 1),
        'd_eff_99': int(np.searchsorted(cumsum_raw, 0.99) + 1),
        'mean_magnitude_per_h': np.mean(np.sqrt(np.sum(raw**2, axis=-1)), axis=0),
    }
    
    # --- Per-strategy analysis ---
    for name in strategies:
        res = results[name]['residuals']  # (N, H, d)
        
        # Full chunk stats
        flat_res = res.reshape(-1, d)
        res_var = np.var(flat_res, axis=0)
        res_std = np.sqrt(res_var)
        res_total_var = np.sum(res_var)
        
        # PCA full chunk
        cov_full = np.cov(flat_res.T)
        ev_full = np.sort(np.linalg.eigvalsh(cov_full))[::-1]
        ev_full = np.maximum(ev_full, 0)
        cs_full = np.cumsum(ev_full) / (ev_full.sum() + 1e-20)
        
        # First step (h=0)
        first = res[:, 0, :]
        first_var = np.var(first, axis=0)
        first_total_var = np.sum(first_var)
        cov_first = np.cov(first.T)
        ev_first = np.sort(np.linalg.eigvalsh(cov_first))[::-1]
        ev_first = np.maximum(ev_first, 0)
        cs_first = np.cumsum(ev_first) / (ev_first.sum() + 1e-20)
        
        # Short horizon h<4
        short_res = res[:, :min(4, H), :].reshape(-1, d)
        short_var = np.var(short_res, axis=0)
        short_total_var = np.sum(short_var)
        
        # Per-horizon magnitude
        mag_per_h = np.mean(np.sqrt(np.sum(res**2, axis=-1)), axis=0)  # (H,)
        
        # Per-horizon variance ratio
        var_ratio_per_h = []
        for h in range(H):
            h_var = np.sum(np.var(res[:, h, :], axis=0))
            raw_h_var = np.sum(np.var(raw[:, h, :], axis=0))
            var_ratio_per_h.append(h_var / (raw_h_var + 1e-20))
        var_ratio_per_h = np.array(var_ratio_per_h)
        
        analysis[name] = {
            # Variance stats
            'std_per_dim': res_std,
            'var_per_dim': res_var,
            'total_var': res_total_var,
            'var_ratio_full': res_total_var / (raw_total_var + 1e-20),
            'var_ratio_first': first_total_var / (raw_total_var + 1e-20),
            'var_ratio_short': short_total_var / (raw_total_var + 1e-20),
            'std_ratio_per_dim': res_std / (raw_std + 1e-10),
            'first_step_std_ratio': np.sqrt(first_var) / (raw_std + 1e-10),
            
            # Magnitude per horizon
            'mag_per_h': mag_per_h,
            'var_ratio_per_h': var_ratio_per_h,
            
            # PCA full
            'eigvals_full': ev_full,
            'explained_var_full': ev_full / (ev_full.sum() + 1e-20),
            'pr_full': ev_full.sum()**2 / (np.sum(ev_full**2) + 1e-20),
            'd_eff_95_full': int(np.searchsorted(cs_full, 0.95) + 1),
            
            # PCA first step
            'eigvals_first': ev_first,
            'explained_var_first': ev_first / (ev_first.sum() + 1e-20),
            'pr_first': ev_first.sum()**2 / (np.sum(ev_first**2) + 1e-20),
            'd_eff_95_first': int(np.searchsorted(cs_first, 0.95) + 1),
        }
    
    return analysis


# ============================================================================
# 5. PRINTING
# ============================================================================

def print_results(analysis, strategies, d, H, dt):
    """Print comprehensive results."""
    
    # Detect dimension names
    if d == 7:
        dim_names = ['Δpx', 'Δpy', 'Δpz', 'Δrx', 'Δry', 'Δrz', 'Δgrip']
    else:
        dim_names = [f'dim{i}' for i in range(d)]
    
    short_names = {
        'zero_order_K1': 'ZeroOrder(K=1)',
        'linear_K2': 'Linear(K=2)',
        'linear_smooth_K3': 'LinSmooth(K=3)',
        'linear_smooth_K5': 'LinSmooth(K=5)',
        'linear_smooth_K7': 'LinSmooth(K=7)',
        'quad_smooth_K4': 'QuadSmooth(K=4)',
        'quad_smooth_K7': 'QuadSmooth(K=7)',
        'taylor_order1_K2': 'Taylor-1(K=2)',
        'taylor_order2_K3': 'Taylor-2(K=3)',
        'taylor_order3_K4': 'Taylor-3(K=4)',
    }
    
    print("\n" + "=" * 95)
    print("RAW ACTION STATISTICS")
    print("=" * 95)
    print(f"  Total variance: {analysis['raw']['total_var']:.8f}")
    print(f"  Mean std: {analysis['raw']['mean_std']:.8f}")
    print(f"  Participation ratio: {analysis['raw']['pr']:.2f}")
    print(f"  d_eff(95%): {analysis['raw']['d_eff_95']}")
    print(f"\n  Per-dimension std:")
    for i, name in enumerate(dim_names):
        print(f"    {name}: {analysis['raw']['std_per_dim'][i]:.8f}")
    
    print("\n" + "=" * 95)
    print("VARIANCE COMPRESSION RATIOS (residual_var / raw_var)")
    print("=" * 95)
    header = f"{'Strategy':<22} {'Full Chunk':<14} {'First Step':<14} {'Short(h<4)':<14} {'Better?':<8}"
    print(f"\n{header}")
    print("-" * 72)
    
    for name in strategies:
        a = analysis[name]
        sn = short_names.get(name, name)
        full = a['var_ratio_full']
        first = a['var_ratio_first']
        short = a['var_ratio_short']
        better = "YES" if first < 0.5 else ("ok" if first < 1.0 else "NO")
        print(f"{sn:<22} {full:<14.4f} {first:<14.4f} {short:<14.4f} {better:<8}")
    
    print("\n" + "=" * 95)
    print("PER-DIMENSION STD RATIO AT FIRST STEP (h=0)")
    print("=" * 95)
    
    key_strats = ['zero_order_K1', 'linear_K2', 'linear_smooth_K5', 'taylor_order2_K3']
    key_strats = [s for s in key_strats if s in strategies]
    
    print(f"\n{'Strategy':<22}", end='')
    for dn in dim_names:
        print(f"{dn:<10}", end='')
    print(f"{'Mean':<10}")
    print("-" * (22 + 10*d + 10))
    
    for name in key_strats:
        a = analysis[name]
        sn = short_names.get(name, name)
        print(f"{sn:<22}", end='')
        for v in a['first_step_std_ratio']:
            print(f"{v:<10.4f}", end='')
        print(f"{np.mean(a['first_step_std_ratio']):<10.4f}")
    
    print("\n" + "=" * 95)
    print("EFFECTIVE DIMENSIONALITY")
    print("=" * 95)
    
    print(f"\n{'Setting':<22} {'PR(full)':<10} {'d95(full)':<10} {'PR(h=0)':<10} {'d95(h=0)':<10}")
    print("-" * 62)
    print(f"{'Raw actions':<22} {analysis['raw']['pr']:<10.2f} {analysis['raw']['d_eff_95']:<10}")
    for name in key_strats:
        a = analysis[name]
        sn = short_names.get(name, name)
        print(f"{sn:<22} {a['pr_full']:<10.2f} {a['d_eff_95_full']:<10} "
              f"{a['pr_first']:<10.2f} {a['d_eff_95_first']:<10}")
    
    print("\n" + "=" * 95)
    print("EIGENVALUE SPECTRUM (explained variance ratio)")
    print("=" * 95)
    
    print(f"\n  Raw: ", end='')
    for v in analysis['raw']['explained_var']:
        print(f"{v:.4f} ", end='')
    print()
    for name in key_strats:
        sn = short_names.get(name, name)
        print(f"  {sn} (full): ", end='')
        for v in analysis[name]['explained_var_full']:
            print(f"{v:.4f} ", end='')
        print()
        print(f"  {sn} (h=0): ", end='')
        for v in analysis[name]['explained_var_first']:
            print(f"{v:.4f} ", end='')
        print()
    
    print("\n" + "=" * 95)
    print("RESIDUAL MAGNITUDE GROWTH OVER HORIZON")
    print("=" * 95)
    
    for name in key_strats:
        a = analysis[name]
        sn = short_names.get(name, name)
        mag = a['mag_per_h']
        print(f"\n  {sn}:")
        h_indices = [0, min(3,H-1), min(7,H-1), H-1]
        for h in h_indices:
            print(f"    h={h}: ||ε||={mag[h]:.6f}", end='')
        print(f"\n    Growth h=0→h={H-1}: {mag[-1]/(mag[0]+1e-10):.1f}x")
    
    print("\n" + "=" * 95)
    print("PER-HORIZON VARIANCE RATIO")
    print("=" * 95)
    print(f"\n  Horizon steps where residual variance < raw variance (ratio < 1.0):")
    for name in key_strats:
        a = analysis[name]
        sn = short_names.get(name, name)
        good_h = np.where(a['var_ratio_per_h'] < 1.0)[0]
        if len(good_h) > 0:
            print(f"  {sn}: h=0..{good_h[-1]} (ratio at h=0: {a['var_ratio_per_h'][0]:.4f}, "
                  f"at h={good_h[-1]}: {a['var_ratio_per_h'][good_h[-1]]:.4f})")
        else:
            print(f"  {sn}: NONE (ratio at h=0: {a['var_ratio_per_h'][0]:.4f})")
    
    # Summary
    best_first = min(strategies.keys(), key=lambda n: analysis[n]['var_ratio_first'])
    best_sn = short_names.get(best_first, best_first)
    best_val = analysis[best_first]['var_ratio_first']
    
    print("\n" + "=" * 95)
    print("SUMMARY & DESIGN IMPLICATIONS")
    print("=" * 95)
    print(f"""
  Best strategy for first step (h=0):
    {best_sn}: variance ratio = {best_val:.4f} ({(1-best_val)*100:.1f}% reduction)

  Zero-order hold (simplest, most stable):
    First step: {analysis['zero_order_K1']['var_ratio_first']:.4f}
    Full chunk: {analysis['zero_order_K1']['var_ratio_full']:.4f}
    Beneficial horizon: h=0..{np.sum(analysis['zero_order_K1']['var_ratio_per_h'] < 1.0)-1}

  Effective dimensionality at h=0:
    Raw: PR={analysis['raw']['pr']:.2f}, d95={analysis['raw']['d_eff_95']}
    ZeroOrder residual: PR={analysis['zero_order_K1']['pr_first']:.2f}, d95={analysis['zero_order_K1']['d_eff_95_first']}
    Linear residual: PR={analysis['linear_K2']['pr_first']:.2f}, d95={analysis['linear_K2']['d_eff_95_first']}
""")


# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_all(analysis, results, strategies, d, H, dt, save_dir):
    """Generate all analysis plots."""
    if not HAS_MPL:
        print("Skipping plots (matplotlib not available)")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    if d == 7:
        dim_names = ['Δpx', 'Δpy', 'Δpz', 'Δrx', 'Δry', 'Δrz', 'Δgrip']
    else:
        dim_names = [f'd{i}' for i in range(d)]
    
    short_names = {
        'zero_order_K1': 'ZeroOrder(K=1)',
        'linear_K2': 'Linear(K=2)',
        'linear_smooth_K3': 'LinSmooth(K=3)',
        'linear_smooth_K5': 'LinSmooth(K=5)',
        'linear_smooth_K7': 'LinSmooth(K=7)',
        'quad_smooth_K4': 'QuadSmooth(K=4)',
        'quad_smooth_K7': 'QuadSmooth(K=7)',
        'taylor_order1_K2': 'Taylor-1(K=2)',
        'taylor_order2_K3': 'Taylor-2(K=3)',
        'taylor_order3_K4': 'Taylor-3(K=4)',
    }
    
    key_strats = ['zero_order_K1', 'linear_K2', 'linear_smooth_K5', 'taylor_order2_K3']
    key_strats = [s for s in key_strats if s in strategies]
    all_strats = [s for s in strategies if s in analysis]
    
    # ---- Fig 1: Variance ratio bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax_idx, (metric, title) in enumerate([
        ('var_ratio_full', '(a) Full Chunk Variance Ratio'),
        ('var_ratio_first', '(b) First Step (h=0) Variance Ratio'),
        ('var_ratio_short', '(c) Short Horizon (h<4) Variance Ratio'),
    ]):
        ax = axes[ax_idx]
        names = all_strats
        vals = [analysis[n][metric] for n in names]
        colors = ['green' if v < 1 else 'salmon' for v in vals]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.75)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([short_names.get(n, n) for n in names], fontsize=8)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Variance Ratio')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(vals):
            ax.text(max(v, 0) + 0.02, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_variance_ratios.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Fig 2: Residual magnitude growth ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    raw_mag = analysis['raw']['mean_magnitude_per_h']
    ax.plot(range(H), raw_mag, 'k--', linewidth=2, label='Raw ||a||', alpha=0.6)
    for name in key_strats:
        ax.plot(range(H), analysis[name]['mag_per_h'], '-o', markersize=3, 
                linewidth=2, label=short_names.get(name, name))
    ax.set_xlabel('Timestep h in chunk')
    ax.set_ylabel('Mean ||residual||')
    ax.set_title('(a) Residual Magnitude vs Horizon')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for name in key_strats:
        ax.plot(range(H), analysis[name]['var_ratio_per_h'], '-o', markersize=3,
                linewidth=2, label=short_names.get(name, name))
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Timestep h in chunk')
    ax.set_ylabel('Variance Ratio at h')
    ax.set_title('(b) Per-Horizon Variance Ratio')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_residual_growth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Fig 3: Per-dimension std ratio at h=0 ----
    fig, ax = plt.subplots(1, 1, figsize=(max(12, d*1.5), 5))
    
    x = np.arange(d)
    width = 0.8 / len(key_strats)
    for i, name in enumerate(key_strats):
        ratios = analysis[name]['first_step_std_ratio']
        ax.bar(x + i*width, ratios, width, label=short_names.get(name, name), alpha=0.8)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xticks(x + width * (len(key_strats)-1) / 2)
    ax.set_xticklabels(dim_names, fontsize=10)
    ax.set_ylabel('Std Ratio (residual / raw)')
    ax.set_title('Per-Dimension Std Compression at First Step (h=0)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_per_dim_std.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Fig 4: Effective dimensionality ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PCA spectrum full
    ax = axes[0]
    ax.semilogy(range(1, d+1), analysis['raw']['explained_var'], 'k-o', 
                markersize=7, linewidth=2, label='Raw')
    for name in key_strats:
        ax.semilogy(range(1, d+1), analysis[name]['explained_var_full'], '-o',
                    markersize=4, linewidth=1.5, label=short_names.get(name, name))
    ax.set_xlabel('PC Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('(a) PCA Spectrum (Full Chunk)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, d+1))
    
    # PCA spectrum first step
    ax = axes[1]
    for name in key_strats:
        ax.semilogy(range(1, d+1), analysis[name]['explained_var_first'], '-o',
                    markersize=4, linewidth=1.5, label=short_names.get(name, name))
    ax.set_xlabel('PC Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('(b) PCA Spectrum (First Step)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, d+1))
    
    # PR summary
    ax = axes[2]
    all_names = ['raw'] + key_strats
    pr_full = [analysis['raw']['pr']] + [analysis[n]['pr_full'] for n in key_strats]
    pr_first = [0] + [analysis[n]['pr_first'] for n in key_strats]
    
    x_pos = np.arange(len(all_names))
    width = 0.35
    ax.bar(x_pos - width/2, pr_full, width, label='Full Chunk', alpha=0.8)
    ax.bar(x_pos + width/2, pr_first, width, label='First Step', alpha=0.8)
    labels = ['Raw'] + [short_names.get(n, n).split('(')[0] for n in key_strats]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Participation Ratio')
    ax.set_title('(c) Effective Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_dimensionality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Fig 5: Example trajectory ----
    fig, axes = plt.subplots(3, min(4, d), figsize=(5*min(4, d), 10))
    if min(4, d) == 1:
        axes = axes.reshape(-1, 1)
    
    idx = min(50, len(results['raw']['chunks'])-1)
    chunk = results['raw']['chunks'][idx]
    t_steps = np.arange(H)
    
    for dim_i in range(min(4, d)):
        ax = axes[0, dim_i]
        ax.plot(t_steps, chunk[:, dim_i], 'k-', linewidth=2)
        ax.set_title(f'{dim_names[dim_i]} (Raw)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, dim_i]
        ax.plot(t_steps, chunk[:, dim_i], 'k-', linewidth=2, label='GT')
        for name in key_strats:
            pred = results[name]['predictions'][idx]
            ax.plot(t_steps, pred[:, dim_i], '--', linewidth=1.2, 
                    label=short_names.get(name, name).split('(')[0], alpha=0.7)
        ax.set_title(f'{dim_names[dim_i]} (Predictions)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if dim_i == 0:
            ax.legend(fontsize=6)
        
        ax = axes[2, dim_i]
        for name in key_strats:
            res = results[name]['residuals'][idx]
            ax.plot(t_steps, res[:, dim_i], linewidth=1.5, 
                    label=short_names.get(name, name).split('(')[0], alpha=0.8)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_title(f'{dim_names[dim_i]} (Residuals)', fontsize=10)
        ax.set_xlabel('h')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Example Chunk: Ground Truth, Predictions, Residuals', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_example.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ---- Fig 6: Distributions ----
    fig, axes = plt.subplots(2, min(4, d), figsize=(5*min(4, d), 8))
    if min(4, d) == 1:
        axes = axes.reshape(-1, 1)
    
    for dim_i in range(min(4, d)):
        # h=0
        ax = axes[0, dim_i]
        raw_v = results['raw']['chunks'][:, 0, dim_i]
        ax.hist(raw_v, bins=80, density=True, alpha=0.4, color='gray', label='Raw')
        for name in ['zero_order_K1', 'linear_K2']:
            if name in results:
                res_v = results[name]['residuals'][:, 0, dim_i]
                ax.hist(res_v, bins=80, density=True, alpha=0.4, 
                        label=short_names.get(name, name))
        ax.set_title(f'{dim_names[dim_i]} (h=0)', fontsize=10)
        ax.legend(fontsize=7)
        
        # All h
        ax = axes[1, dim_i]
        raw_v = results['raw']['chunks'][:, :, dim_i].flatten()
        ax.hist(raw_v, bins=80, density=True, alpha=0.4, color='gray', label='Raw')
        for name in ['zero_order_K1', 'linear_K2']:
            if name in results:
                res_v = results[name]['residuals'][:, :, dim_i].flatten()
                ax.hist(res_v, bins=80, density=True, alpha=0.4,
                        label=short_names.get(name, name))
        ax.set_title(f'{dim_names[dim_i]} (all h)', fontsize=10)
        ax.legend(fontsize=7)
    
    plt.suptitle('Distribution: Raw Actions vs Residuals', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


# ============================================================================
# 7. SAVE NUMERICAL RESULTS
# ============================================================================

def save_results(analysis, strategies, d, H, dt, save_dir):
    """Save numerical results to JSON for later reference."""
    os.makedirs(save_dir, exist_ok=True)
    
    output = {
        'params': {'d': d, 'H': H, 'dt': dt},
        'raw': {
            'std_per_dim': analysis['raw']['std_per_dim'].tolist(),
            'total_var': float(analysis['raw']['total_var']),
            'pr': float(analysis['raw']['pr']),
            'd_eff_95': int(analysis['raw']['d_eff_95']),
        },
        'strategies': {}
    }
    
    for name in strategies:
        a = analysis[name]
        output['strategies'][name] = {
            'var_ratio_full': float(a['var_ratio_full']),
            'var_ratio_first': float(a['var_ratio_first']),
            'var_ratio_short': float(a['var_ratio_short']),
            'first_step_std_ratio': a['first_step_std_ratio'].tolist(),
            'pr_full': float(a['pr_full']),
            'pr_first': float(a['pr_first']),
            'd_eff_95_full': int(a['d_eff_95_full']),
            'd_eff_95_first': int(a['d_eff_95_first']),
            'var_ratio_per_h': a['var_ratio_per_h'].tolist(),
        }
    
    path = os.path.join(save_dir, 'phase1_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Numerical results saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Residual Statistics on Real LIBERO Data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing zarr datasets')
    parser.add_argument('--zarr_path', type=str, default=None,
                        help='Path to a single zarr dataset')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific task zarr filenames to load')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Control timestep (default: 0.05 for 20Hz)')
    parser.add_argument('--H', type=int, default=16,
                        help='Prediction horizon (default: 16)')
    parser.add_argument('--save_dir', type=str, default='./phase1_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 1: Residual Statistics on Real LIBERO Data")
    print("=" * 70)
    
    # Load data
    print(f"\n--- Loading data ---")
    if args.zarr_path:
        episodes, metadata = load_zarr_episodes(args.zarr_path)
        metadata_list = [metadata]
    elif args.data_dir:
        episodes, metadata_list = load_all_datasets(args.data_dir, args.tasks)
    else:
        print("ERROR: Please provide --data_dir or --zarr_path")
        print("\nUsage examples:")
        print("  python phase1_real_data.py --zarr_path /path/to/task.zarr")
        print("  python phase1_real_data.py --data_dir /path/to/libero/zarr/")
        print("  python phase1_real_data.py --data_dir /path/to/zarr/ --tasks LIBERO_10_task1.zarr LIBERO_10_task2.zarr")
        return
    
    if not episodes:
        print("ERROR: No episodes loaded!")
        return
    
    d = episodes[0].shape[1]
    total_steps = sum(len(ep) for ep in episodes)
    print(f"\n  Total: {len(episodes)} episodes, {total_steps} steps, d={d}")
    print(f"  Episode lengths: min={min(len(e) for e in episodes)}, "
          f"max={max(len(e) for e in episodes)}, "
          f"mean={np.mean([len(e) for e in episodes]):.0f}")
    
    # Compute residuals
    print(f"\n--- Computing residuals (dt={args.dt}, H={args.H}) ---")
    results, strategies = compute_residuals_for_dataset(episodes, args.dt, args.H)
    N = len(results['raw']['chunks'])
    print(f"  Extracted {N} chunks")
    
    if N == 0:
        print("ERROR: No chunks extracted! Episodes may be too short.")
        return
    
    # Analyze
    print(f"\n--- Analyzing ---")
    analysis = analyze_all(results, strategies, d, args.H)
    
    # Print results
    print_results(analysis, strategies, d, args.H, args.dt)
    
    # Plot
    plot_all(analysis, results, strategies, d, args.H, args.dt, args.save_dir)
    
    # Save numerical results
    save_results(analysis, strategies, d, args.H, args.dt, args.save_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""LIBERO-10 Action Space Polar Decomposition Analysis."""

import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────
ZARR_PATH = '/workspace/oat/data/libero/libero10_N500.zarr'
z = zarr.open(ZARR_PATH, 'r')
actions = z['data/action'][:]          # (N, 7)
task_uids = z['data/task_uid'][:].flatten()  # (N,)
episode_ends = z['meta/episode_ends'][:]

N = actions.shape[0]
dim_names = ['Δx', 'Δy', 'Δz', 'Δroll', 'Δpitch', 'Δyaw', 'grip']
unique_tasks = np.unique(task_uids)
print(f"Loaded {N} timesteps, {len(episode_ends)} episodes, {len(unique_tasks)} tasks")
print(f"Task UIDs: {unique_tasks}")

# ── Polar transforms ─────────────────────────────────────────────────────────
dx, dy, dz = actions[:, 0], actions[:, 1], actions[:, 2]
droll, dpitch, dyaw = actions[:, 3], actions[:, 4], actions[:, 5]
grip = actions[:, 6]

r_trans = np.sqrt(dx**2 + dy**2)
theta_trans = np.arctan2(dy, dx)
r_rot = np.sqrt(droll**2 + dpitch**2)
theta_rot = np.arctan2(dpitch, droll)

# ── Helper: Rayleigh test ────────────────────────────────────────────────────
def rayleigh_test(angles):
    """Return (mean_resultant_length, p_value)."""
    n = len(angles)
    if n < 2:
        return 0.0, 1.0
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    Z = n * R**2
    # Approximation for p-value
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p = max(0.0, min(1.0, p))
    return R, p

# ── PDF Report ───────────────────────────────────────────────────────────────
pdf = PdfPages('/root/polar_analysis_report.pdf')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC STATISTICS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("1. BASIC STATISTICS (all demos, all tasks)")
print("="*80)

stat_data = []
for i, name in enumerate(dim_names):
    col = actions[:, i]
    row = [name, np.mean(col), np.std(col), np.min(col), np.max(col),
           np.median(col), np.percentile(col, 1), np.percentile(col, 99)]
    stat_data.append(row)
    print(f"  {name:>8s}: mean={row[1]:+.6f}  std={row[2]:.6f}  "
          f"min={row[3]:+.6f}  max={row[4]:+.6f}  "
          f"p1={row[6]:+.6f}  p99={row[7]:+.6f}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
headers = ['Dim', 'Mean', 'Std', 'Min', 'Max', 'Median', 'P1', 'P99']
cell_text = [[r[0]] + [f"{v:+.5f}" for v in r[1:]] for r in stat_data]
table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)
ax.set_title("Per-Dimension Statistics Across All LIBERO-10 Demos", fontsize=12, pad=20)
fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. POLAR TRANSFORM DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Polar Transform Distributions", fontsize=14)

# (a) Histogram of r_trans (log-scale)
ax = axes[0, 0]
r_trans_pos = r_trans[r_trans > 0]
if len(r_trans_pos) > 0:
    ax.hist(r_trans_pos, bins=100, log=False, color='steelblue', alpha=0.7)
    ax.set_xscale('log')
    for p, c, ls in [(1, 'red', '--'), (5, 'orange', '--'), (10, 'green', '--')]:
        val = np.percentile(r_trans, p)
        ax.axvline(val, color=c, linestyle=ls, label=f'P{p}={val:.1e}')
ax.set_xlabel('r_trans (log)'); ax.set_ylabel('Count')
ax.set_title('(a) r_trans Distribution'); ax.legend(fontsize=7)

# (b) Histogram of r_rot (log-scale)
ax = axes[0, 1]
r_rot_pos = r_rot[r_rot > 0]
if len(r_rot_pos) > 0:
    ax.hist(r_rot_pos, bins=100, color='coral', alpha=0.7)
    ax.set_xscale('log')
    for p, c, ls in [(1, 'red', '--'), (5, 'orange', '--'), (10, 'green', '--')]:
        val = np.percentile(r_rot, p)
        ax.axvline(val, color=c, linestyle=ls, label=f'P{p}={val:.1e}')
ax.set_xlabel('r_rot (log)'); ax.set_ylabel('Count')
ax.set_title('(b) r_rot Distribution'); ax.legend(fontsize=7)

# (c) Circular histogram of θ_trans
ax = axes[0, 2]
ax.remove()
ax = fig.add_subplot(2, 3, 3, projection='polar')
mask_trans = r_trans > 1e-4  # exclude near-zero for meaningful angles
if mask_trans.sum() > 0:
    ax.hist(theta_trans[mask_trans], bins=36, color='steelblue', alpha=0.7)
ax.set_title('(c) θ_trans (r>1e-4)', pad=15)

# (d) Circular histogram of θ_rot
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(2, 3, 4, projection='polar')
mask_rot = r_rot > 1e-4
if mask_rot.sum() > 0:
    ax.hist(theta_rot[mask_rot], bins=36, color='coral', alpha=0.7)
ax.set_title('(d) θ_rot (r>1e-4)', pad=15)

# (e) 2D hexbin (θ_trans, r_trans)
ax = axes[1, 1]
if mask_trans.sum() > 0:
    hb = ax.hexbin(theta_trans[mask_trans], r_trans[mask_trans],
                   gridsize=40, cmap='Blues', mincnt=1)
    plt.colorbar(hb, ax=ax)
ax.set_xlabel('θ_trans'); ax.set_ylabel('r_trans')
ax.set_title('(e) (θ_trans, r_trans) hexbin')

# (f) 2D hexbin (θ_rot, r_rot)
ax = axes[1, 2]
if mask_rot.sum() > 0:
    hb = ax.hexbin(theta_rot[mask_rot], r_rot[mask_rot],
                   gridsize=40, cmap='Oranges', mincnt=1)
    plt.colorbar(hb, ax=ax)
ax.set_xlabel('θ_rot'); ax.set_ylabel('r_rot')
ax.set_title('(f) (θ_rot, r_rot) hexbin')

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. NEAR-ZERO MOTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("3. NEAR-ZERO MOTION ANALYSIS")
print("="*80)

thresholds = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
pct_trans = []
pct_rot = []
pct_both = []

for eps in thresholds:
    pt = 100.0 * np.mean(r_trans < eps)
    pr = 100.0 * np.mean(r_rot < eps)
    pb = 100.0 * np.mean((r_trans < eps) & (r_rot < eps))
    pct_trans.append(pt)
    pct_rot.append(pr)
    pct_both.append(pb)
    print(f"  ε={eps:.0e}: r_trans<ε {pt:6.2f}%  r_rot<ε {pr:6.2f}%  both<ε {pb:6.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Near-Zero Motion Analysis", fontsize=14)

# Line chart
ax = axes[0]
ax.plot(thresholds, pct_trans, 'o-', label='r_trans < ε', color='steelblue')
ax.plot(thresholds, pct_rot, 's-', label='r_rot < ε', color='coral')
ax.plot(thresholds, pct_both, '^-', label='both < ε', color='purple')
ax.set_xscale('log')
ax.set_xlabel('Threshold ε'); ax.set_ylabel('% of timesteps')
ax.set_title('Near-Zero Motion Percentages')
ax.legend(); ax.grid(True, alpha=0.3)
ax.axhline(5, color='green', linestyle=':', alpha=0.5, label='5% threshold')
ax.axhline(20, color='red', linestyle=':', alpha=0.5, label='20% threshold')

# Distribution of (Δx, Δy) when r_trans < 1e-3
ax = axes[1]
near_zero_mask = r_trans < 1e-3
if near_zero_mask.sum() > 0:
    ax.scatter(dx[near_zero_mask], dy[near_zero_mask], s=1, alpha=0.3, color='steelblue')
    # Check how many are exactly zero
    exact_zero = np.sum((dx == 0) & (dy == 0))
    ax.set_title(f'(Δx,Δy) when r_trans<1e-3\n'
                 f'N={near_zero_mask.sum()}, exactly zero={exact_zero} '
                 f'({100*exact_zero/near_zero_mask.sum():.1f}%)')
else:
    ax.set_title('No near-zero r_trans samples')
ax.set_xlabel('Δx'); ax.set_ylabel('Δy')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-TASK BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("4. PER-TASK BREAKDOWN")
print("="*80)

task_stats = []
for tid in unique_tasks:
    mask = task_uids == tid
    rt = r_trans[mask]
    rr = r_rot[mask]
    tt = theta_trans[mask & (r_trans > 1e-4)]
    pct_nz = 100.0 * np.mean(rt < 1e-3)

    # Circular mean of theta_trans
    if len(tt) > 0:
        circ_mean = np.arctan2(np.mean(np.sin(tt)), np.mean(np.cos(tt)))
    else:
        circ_mean = 0.0

    task_stats.append({
        'tid': tid, 'med_r_trans': np.median(rt), 'med_r_rot': np.median(rr),
        'circ_mean_theta': circ_mean, 'pct_near_zero': pct_nz, 'n': mask.sum()
    })
    print(f"  Task {tid}: med_r_trans={np.median(rt):.5f}  med_r_rot={np.median(rr):.5f}  "
          f"θ_mean={np.degrees(circ_mean):+.1f}°  near_zero={pct_nz:.1f}%  N={mask.sum()}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Per-Task Breakdown", fontsize=14)

tids = [s['tid'] for s in task_stats]
x = np.arange(len(tids))
w = 0.35

# Median r_trans and r_rot
ax = axes[0, 0]
ax.bar(x - w/2, [s['med_r_trans'] for s in task_stats], w, label='med r_trans', color='steelblue')
ax.bar(x + w/2, [s['med_r_rot'] for s in task_stats], w, label='med r_rot', color='coral')
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in tids])
ax.set_xlabel('Task UID'); ax.set_ylabel('Median radius')
ax.set_title('Median Polar Radii per Task'); ax.legend()

# % near-zero
ax = axes[0, 1]
ax.bar(x, [s['pct_near_zero'] for s in task_stats], color='purple', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in tids])
ax.set_xlabel('Task UID'); ax.set_ylabel('% near-zero (r_trans<1e-3)')
ax.set_title('Near-Zero Translation %'); ax.axhline(5, color='green', ls=':'); ax.axhline(20, color='red', ls=':')

# Circular mean direction
ax = axes[1, 0]
ax.remove()
ax = fig.add_subplot(2, 2, 3, projection='polar')
for s in task_stats:
    ax.arrow(s['circ_mean_theta'], 0, 0, s['med_r_trans'],
             head_width=0.1, head_length=0.005, color='steelblue', alpha=0.7)
    ax.annotate(str(s['tid']), xy=(s['circ_mean_theta'], s['med_r_trans']),
                fontsize=8, ha='center')
ax.set_title('Dominant θ_trans per Task', pad=15)

# N samples per task
ax = axes[1, 1]
ax.bar(x, [s['n'] for s in task_stats], color='gray', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in tids])
ax.set_xlabel('Task UID'); ax.set_ylabel('N timesteps')
ax.set_title('Samples per Task')

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANGULAR UNIFORMITY TEST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("5. ANGULAR UNIFORMITY (Rayleigh Test)")
print("="*80)

# Global tests
R_trans_global, p_trans_global = rayleigh_test(theta_trans[r_trans > 1e-3])
R_rot_global, p_rot_global = rayleigh_test(theta_rot[r_rot > 1e-3])
print(f"  θ_trans (global, r>1e-3): R={R_trans_global:.4f}, p={p_trans_global:.2e}")
print(f"  θ_rot   (global, r>1e-3): R={R_rot_global:.4f}, p={p_rot_global:.2e}")

print("\n  Per-task:")
task_rayleigh = []
for tid in unique_tasks:
    mask_t = (task_uids == tid) & (r_trans > 1e-3)
    mask_r = (task_uids == tid) & (r_rot > 1e-3)
    Rt, pt = rayleigh_test(theta_trans[mask_t])
    Rr, pr = rayleigh_test(theta_rot[mask_r])
    task_rayleigh.append({'tid': tid, 'R_trans': Rt, 'p_trans': pt, 'R_rot': Rr, 'p_rot': pr})
    print(f"    Task {tid}: θ_trans R={Rt:.4f} p={pt:.2e}  |  θ_rot R={Rr:.4f} p={pr:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Angular Uniformity: Mean Resultant Length per Task", fontsize=14)

ax = axes[0]
ax.bar(x, [s['R_trans'] for s in task_rayleigh], color='steelblue', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in tids])
ax.set_xlabel('Task UID'); ax.set_ylabel('Mean Resultant Length R')
ax.set_title('θ_trans Uniformity (lower R = more uniform)')
ax.axhline(0.1, color='green', ls=':', label='R=0.1 (approx uniform)')
ax.legend()

ax = axes[1]
ax.bar(x, [s['R_rot'] for s in task_rayleigh], color='coral', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in tids])
ax.set_xlabel('Task UID'); ax.set_ylabel('Mean Resultant Length R')
ax.set_title('θ_rot Uniformity'); ax.axhline(0.1, color='green', ls=':')

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Δyaw AND GRIP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("6. Δyaw AND GRIP ANALYSIS")
print("="*80)

dyaw_std = np.std(dyaw)
dyaw_nonzero = np.mean(np.abs(dyaw) > 1e-5)
grip_unique = np.unique(grip)
grip_binary = len(grip_unique) <= 3 or (np.std(grip) < 0.1 and len(grip_unique) <= 10)

print(f"  Δyaw: mean={np.mean(dyaw):.6f}  std={dyaw_std:.6f}  "
      f"min={np.min(dyaw):.6f}  max={np.max(dyaw):.6f}")
print(f"  Δyaw non-zero (|v|>1e-5): {100*dyaw_nonzero:.1f}%")
print(f"  Grip unique values: {grip_unique}")
print(f"  Grip std: {np.std(grip):.4f}")
print(f"  Grip appears binary: {grip_binary}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Δyaw and Grip Analysis", fontsize=14)

ax = axes[0]
ax.hist(dyaw[dyaw != 0] if np.any(dyaw != 0) else dyaw, bins=100, color='teal', alpha=0.7)
ax.set_xlabel('Δyaw'); ax.set_ylabel('Count')
ax.set_title(f'Δyaw Distribution (std={dyaw_std:.5f})')

ax = axes[1]
ax.hist(grip, bins=50, color='orange', alpha=0.7)
ax.set_xlabel('Grip'); ax.set_ylabel('Count')
ax.set_title(f'Grip Distribution (unique={len(grip_unique)})')

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CROSS-DIMENSION CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("7. CROSS-DIMENSION CORRELATION")
print("="*80)

corr = np.corrcoef(actions.T)
print("  Correlation matrix:")
print("         " + "  ".join(f"{n:>8s}" for n in dim_names))
for i, name in enumerate(dim_names):
    print(f"  {name:>6s} " + "  ".join(f"{corr[i,j]:+8.4f}" for j in range(7)))

# r_trans vs r_rot correlation
corr_rr = np.corrcoef(r_trans, r_rot)[0, 1]
print(f"\n  Correlation r_trans vs r_rot: {corr_rr:+.4f}")

# θ_trans vs θ_rot (where both are well-defined)
both_valid = (r_trans > 1e-3) & (r_rot > 1e-3)
if both_valid.sum() > 10:
    # Circular correlation
    tt = theta_trans[both_valid]
    tr = theta_rot[both_valid]
    sin_tt = np.sin(tt - np.mean(tt))
    sin_tr = np.sin(tr - np.mean(tr))
    circ_corr = np.sum(sin_tt * sin_tr) / np.sqrt(np.sum(sin_tt**2) * np.sum(sin_tr**2))
    print(f"  Circular correlation θ_trans vs θ_rot: {circ_corr:+.4f} (N={both_valid.sum()})")
else:
    circ_corr = 0.0
    print(f"  Insufficient samples with both r > 1e-3 for θ correlation")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cross-Dimension Correlations", fontsize=14)

ax = axes[0]
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(7)); ax.set_xticklabels(dim_names, rotation=45, ha='right')
ax.set_yticks(range(7)); ax.set_yticklabels(dim_names)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax)
ax.set_title('7D Action Correlation Matrix')

ax = axes[1]
# Subsample for scatter
idx = np.random.choice(N, min(5000, N), replace=False)
ax.scatter(r_trans[idx], r_rot[idx], s=1, alpha=0.3, color='purple')
ax.set_xlabel('r_trans'); ax.set_ylabel('r_rot')
ax.set_title(f'r_trans vs r_rot (ρ={corr_rr:+.3f})')
ax.set_xscale('log'); ax.set_yscale('log')

ax = axes[2]
if both_valid.sum() > 10:
    idx2 = np.random.choice(np.where(both_valid)[0], min(5000, both_valid.sum()), replace=False)
    hb = ax.hexbin(theta_trans[idx2], theta_rot[idx2], gridsize=30, cmap='Purples', mincnt=1)
    plt.colorbar(hb, ax=ax)
ax.set_xlabel('θ_trans'); ax.set_ylabel('θ_rot')
ax.set_title(f'θ_trans vs θ_rot (circ_ρ={circ_corr:+.3f})')

fig.tight_layout()
pdf.savefig(fig); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
pdf.close()
print(f"\n✓ Report saved to ~/polar_analysis_report.pdf")

# Compute key metrics for decision
pct_nz_trans = 100.0 * np.mean(r_trans < 1e-3)
pct_nz_rot = 100.0 * np.mean(r_rot < 1e-3)

def viability(pct):
    if pct < 5: return "VIABLE"
    elif pct < 20: return "CAUTION"
    else: return "PROBLEMATIC"

def uniformity(R, p):
    if p < 0.01 and R > 0.1:
        return f"CLUSTERED (p={p:.2e}, R={R:.4f})"
    else:
        return f"UNIFORM (p={p:.2e}, R={R:.4f})"

dyaw_label = "INVARIANT" if dyaw_std < 0.01 else "NEEDS ENCODING"
grip_label = "BINARY" if grip_binary else "CONTINUOUS"

trans_viab = viability(pct_nz_trans)
rot_viab = viability(pct_nz_rot)

if trans_viab == "PROBLEMATIC" or rot_viab == "PROBLEMATIC":
    recommendation = "RECONSIDER"
elif trans_viab == "CAUTION" or rot_viab == "CAUTION":
    recommendation = "PROCEED WITH NULL-MOTION TOKEN"
else:
    recommendation = "PROCEED"

print("\n" + "="*80)
print("=== POLAR DECOMPOSITION VIABILITY ===")
print("="*80)
print(f"Near-zero r_trans (< 1e-3): {pct_nz_trans:.1f}% of timesteps → [{trans_viab}]")
print(f"Near-zero r_rot (< 1e-3):   {pct_nz_rot:.1f}% of timesteps → [{rot_viab}]")
print(f"θ_trans uniformity:          [{uniformity(R_trans_global, p_trans_global)}]")
print(f"θ_rot uniformity:            [{uniformity(R_rot_global, p_rot_global)}]")
print(f"Δyaw variance:               {dyaw_std:.5f} → [{dyaw_label}]")
print(f"Grip:                        [{grip_label}] (unique values: {grip_unique})")
print(f"\nRecommendation: [{recommendation}]")
print("="*80)

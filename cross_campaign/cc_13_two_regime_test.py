"""
cc_13_two_regime_test.py

Two-regime hypothesis test:
  Theoretical Nd_predicted = (re_MODIS / re_insitu)^-2.5  (Grosvenor formula)
  Compared to observed bias = Nd_MODIS / Nd_insitu

If Heavy fits theoretical prediction (microphysics-dominated regime),
   and Non does NOT fit (geometry-dominated regime),
THEN two-regime physics is demonstrated.

This addresses Femin's editorial framing:
  "Cloud microphysical evolution modulates how systematic MODIS re biases
   propagate into Nd retrieval space."

NOTE: Strict propagation test uses Re-only as predictor. The full Grosvenor
formula has ~6 inputs (re, tau, k, f_ad, c_w). Here we isolate the Re path
to test whether re-error alone predicts bias when other factors are matched.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===


m = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')
m['delta_re_vw'] = m['Re_MODIS_21'] - m['re_cas_median']

# Theoretical Nd ratio from Re ratio alone
# Nd ∝ re^-2.5  →  Nd_MODIS/Nd_insitu = (re_MODIS/re_insitu)^-2.5
# If we plug in Re_MODIS_21 and re_cas_median as proxies:
m['re_ratio'] = m['Re_MODIS_21'] / m['re_cas_median']
m['nd_predicted_ratio'] = m['re_ratio'] ** (-2.5)

STATE_COLORS = {'Non': '#3366cc', 'Transition': '#ff8c00', 'Heavy': '#cc3333'}
STATE_ORDER = ['Non', 'Transition', 'Heavy']

# ============================================================================
# Test: Predicted vs Observed bias by state
# ============================================================================
print("=" * 80)
print("TWO-REGIME HYPOTHESIS TEST")
print("=" * 80)
print(f"\n{'State':<12} {'n':>3} {'<re_ratio>':>12} {'<Nd_pred>':>11} "
       f"{'<Nd_obs>':>11} {'gap':>8} {'fit':>10}")
print("-" * 80)

results = []
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    n = len(sub)
    pred_med = sub['nd_predicted_ratio'].median()
    obs_med = sub['bias_21_calc'].median()
    gap = obs_med - pred_med
    rel_gap = gap / pred_med if pred_med > 0 else np.nan

    # Spearman r between predicted and observed (does theoretical track observed?)
    if n >= 4:
        r, p = stats.spearmanr(sub['nd_predicted_ratio'], sub['bias_21_calc'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        fit_str = f'r={r:+.2f} {sig}'
    else:
        fit_str = 'n too small'
        r, p = np.nan, np.nan

    re_ratio_med = sub['re_ratio'].median()
    print(f"{state:<12} {n:>3} {re_ratio_med:>12.3f} {pred_med:>11.3f} "
          f"{obs_med:>11.3f} {gap:>+8.3f}  {fit_str}")

    results.append(dict(state=state, n=n, re_ratio_med=re_ratio_med,
                        pred_med=pred_med, obs_med=obs_med, gap=gap,
                        rel_gap=rel_gap, spearman_r=r, p=p))

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("If a state's predicted matches observed (small gap, r positive):")
print("  → Re-error PROPAGATES directly = microphysics-dominated regime")
print("If a state's predicted does NOT match observed (large gap, r ns):")
print("  → Other factors dominate = geometry/heterogeneity-dominated regime")

# ============================================================================
# More rigorous: Per-profile predicted vs observed scatter
# ============================================================================
print("\n" + "=" * 80)
print("PER-PROFILE SCATTER STATISTICS")
print("=" * 80)
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) < 4:
        continue
    pred = sub['nd_predicted_ratio'].values
    obs = sub['bias_21_calc'].values
    # Linear fit: obs = a + b*pred
    slope, intercept, r_lin, p_lin, _ = stats.linregress(pred, obs)
    # 1:1 fit metric: how close are obs to pred?
    residuals_1to1 = obs - pred
    rmse_1to1 = np.sqrt(np.mean(residuals_1to1**2))
    print(f"\n{state} (n={len(sub)}):")
    print(f"  Pearson r (pred vs obs): {r_lin:+.3f}")
    print(f"  Linear slope:            {slope:+.3f}  (1.0 = perfect propagation)")
    print(f"  Linear intercept:         {intercept:+.3f}")
    print(f"  RMSE around 1:1 line:    {rmse_1to1:.3f}")

# ============================================================================
# Build figure: Two-regime test visualization
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ----- Panel (a): Predicted vs observed scatter -----
ax = axes[0]
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    color = STATE_COLORS[state]
    ax.scatter(sub['nd_predicted_ratio'], sub['bias_21_calc'],
                color=color, s=80, alpha=0.85,
                edgecolors='black', linewidth=0.5,
                label=f'{state} (n={len(sub)})', zorder=3)

# 1:1 line
ax.plot([0, 1.5], [0, 1.5], 'k--', alpha=0.5, lw=1.5, zorder=1,
         label='1:1 (perfect Re propagation)')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.4, lw=1)
ax.axvline(1.0, color='gray', linestyle=':', alpha=0.4, lw=1)

# Median markers per state
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    pred_med = sub['nd_predicted_ratio'].median()
    obs_med = sub['bias_21_calc'].median()
    color = STATE_COLORS[state]
    ax.scatter([pred_med], [obs_med], marker='*', s=400, color=color,
                edgecolors='black', linewidth=1.5, zorder=10)

# Annotations
ax.text(0.5, 1.6, 'OVERESTIMATE\n(geometry-dominated?)',
         ha='center', fontsize=10, color='darkblue',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0e8ff'))
ax.text(0.5, 0.5, 'Re-propagation\nregime', ha='center', fontsize=10,
         color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0d8'))

ax.set_xlabel(r'$N_d^{\,predicted}$ from Re-only: $(R_e^{\,MODIS}/r_e^{\,CAS})^{-2.5}$',
                fontsize=11)
ax.set_ylabel(r'$N_d^{\,observed}$ bias: $N_d^{\,MODIS}/N_d^{\,in-situ}$', fontsize=11)
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 6)
ax.set_title('(a) Two-Regime Test: Re-error propagation\n'
             'Stars = state medians',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# ----- Panel (b): Gap (observed - predicted) by state -----
ax = axes[1]
gaps = []
labels = []
colors = []
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    gap_vals = sub['bias_21_calc'] - sub['nd_predicted_ratio']
    gaps.append(gap_vals.values)
    labels.append(f'{state}\n(n={len(sub)})')
    colors.append(STATE_COLORS[state])

positions = list(range(len(gaps)))
bp = ax.boxplot(gaps, positions=positions, widths=0.6, patch_artist=True,
                 showmeans=True, meanprops=dict(marker='D',
                                                  markerfacecolor='white',
                                                  markeredgecolor='black',
                                                  markersize=9))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)

# Scatter individual points
for i, (g, c) in enumerate(zip(gaps, colors)):
    jitter = np.random.normal(i, 0.07, len(g))
    ax.scatter(jitter, g, color=c, s=40, alpha=0.7,
                edgecolors='black', linewidth=0.4)

# Median annotations
for i, g in enumerate(gaps):
    med = np.median(g)
    ax.text(i, np.percentile(g, 95) + 0.2,
            f'med={med:+.2f}', ha='center', fontsize=10,
            fontweight='bold')

ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.5,
            label='Perfect propagation (gap = 0)')
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r'Gap = bias$_{obs}$ - $N_d^{\,predicted}$',
                fontsize=11)
ax.set_title('(b) Re-Propagation Residual: how far observed is from theory\n'
             'Large +gap → other factors PUSH bias UP (other than Re)\n'
             'Near 0 → Re-error fully propagates',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Statistical test: gaps zero?
print(f"\nWilcoxon test: gap vs 0:")
for i, (state, g) in enumerate(zip(STATE_ORDER, gaps)):
    if len(g) >= 3:
        stat, p = stats.wilcoxon(g)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {state:<12}: median gap={np.median(g):+.3f}, p={p:.4f} {sig}")

fig.suptitle('Two-Regime Physics: Does Re-error propagation explain the bias?\n'
              '(Heavy near 1:1 line → microphysics regime; '
              'Non far from 1:1 → geometry regime)',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out = FIG_DIR / 'cc_fig_two_regime.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\n[SAVE] {out.name}")

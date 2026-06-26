"""
cc_13_two_regime_test_v2.py

Two-regime physics test - REVISED per Femin's editorial input:
  1. Softened panel titles (no "geometry-dominated" or "re-propagation regime"
     as final claims)
  2. Outlier annotation (POST RF10_P20)
  3. Transition reframed as instability regime
  4. Caveat-aware framing

Conceptual claims (carefully calibrated):
  - "consistent with Re-error propagation"
  - "deviates from Re-only prediction"
  - "instability regime" for Transition
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
m['re_ratio'] = m['Re_MODIS_21'] / m['re_cas_median']
m['nd_predicted_ratio'] = m['re_ratio'] ** (-2.5)
m['gap'] = m['bias_21_calc'] - m['nd_predicted_ratio']

STATE_COLORS = {'Non': '#3366cc', 'Transition': '#ff8c00', 'Heavy': '#cc3333'}
STATE_ORDER = ['Non', 'Transition', 'Heavy']

# Identify outlier (POST RF10_P20)
m['is_outlier'] = (m['campaign'] == 'POST') & (m['cloud_id'] == 'RF10_P20')
print(f"Outlier identified: POST RF10_P20")
print(f"  re_cas={m[m['is_outlier']]['re_cas_median'].values[0]:.2f} µm "
       f"(suspect — very small for cloud-core)")
print(f"  bias={m[m['is_outlier']]['bias_21_calc'].values[0]:.2f} (extreme)")

# ============================================================================
# Compute statistics with and without outlier
# ============================================================================
print("\n" + "=" * 80)
print("Gap statistics (median; robust to outlier)")
print("=" * 80)
for label, df in [('ALL', m), ('NO OUTLIER', m[~m['is_outlier']])]:
    print(f"\n{label}:")
    for state in STATE_ORDER:
        sub = df[df['drizzle_state'] == state]
        if len(sub) >= 3:
            gap_med = sub['gap'].median()
            gap_iqr = (sub['gap'].quantile(.25), sub['gap'].quantile(.75))
            stat, p = stats.wilcoxon(sub['gap'])
            sig = ('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns')
            print(f"  {state:<12} (n={len(sub)}): "
                  f"gap median={gap_med:+.3f}  IQR=[{gap_iqr[0]:+.2f}, {gap_iqr[1]:+.2f}]"
                  f"  Wilcoxon p={p:.3f} {sig}")

# ============================================================================
# Build figure (V2 - softened)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ----- Panel (a): Predicted vs observed -----
ax = axes[0]
for state in STATE_ORDER:
    sub = m[(m['drizzle_state'] == state) & (~m['is_outlier'])].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    color = STATE_COLORS[state]
    ax.scatter(sub['nd_predicted_ratio'], sub['bias_21_calc'],
                color=color, s=80, alpha=0.85,
                edgecolors='black', linewidth=0.5,
                label=f'{state} (n={len(sub)})', zorder=3)

# Show outlier as separate marker
out = m[m['is_outlier']]
ax.scatter(out['nd_predicted_ratio'], out['bias_21_calc'],
            marker='X', s=140, color='gray', edgecolors='black', linewidth=1.5,
            zorder=5,
            label=f"POST RF10_P20 (excluded:\n re_CAS=3.3 µm, suspect QC)")

# 1:1 line
lim = 1.5
ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, lw=1.5, zorder=1,
         label='1:1 (Re-error fully propagates)')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.4, lw=1)
ax.axvline(1.0, color='gray', linestyle=':', alpha=0.4, lw=1)

# Median markers per state (excluding outlier)
for state in STATE_ORDER:
    sub = m[(m['drizzle_state'] == state) & (~m['is_outlier'])].dropna(
        subset=['nd_predicted_ratio', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    pred_med = sub['nd_predicted_ratio'].median()
    obs_med = sub['bias_21_calc'].median()
    color = STATE_COLORS[state]
    ax.scatter([pred_med], [obs_med], marker='*', s=400, color=color,
                edgecolors='black', linewidth=1.5, zorder=10)

ax.set_xlabel(r'$N_d^{predicted}$ from Re-only: $(R_e^{MODIS}/r_e^{CAS})^{-2.5}$',
                fontsize=11)
ax.set_ylabel(r'$N_d^{observed}$ bias: $N_d^{MODIS}/N_d^{in-situ}$', fontsize=11)
ax.set_xlim(0, lim)
ax.set_ylim(0, 4.5)
ax.set_title('(a) Observed bias vs Re-only prediction\n'
             'Stars = state medians (outlier excluded)',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=8.5)
ax.grid(True, alpha=0.3)

# ----- Panel (b): Gap distribution by state -----
ax = axes[1]
gaps = []
labels = []
colors = []
for state in STATE_ORDER:
    sub = m[(m['drizzle_state'] == state) & (~m['is_outlier'])].dropna(
        subset=['gap'])
    gaps.append(sub['gap'].values)
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

for i, (g, c) in enumerate(zip(gaps, colors)):
    jitter = np.random.normal(i, 0.07, len(g))
    ax.scatter(jitter, g, color=c, s=40, alpha=0.7,
                edgecolors='black', linewidth=0.4)

# Median annotations + Wilcoxon p
for i, (state, g) in enumerate(zip(STATE_ORDER, gaps)):
    med = np.median(g)
    if len(g) >= 3:
        stat, p = stats.wilcoxon(g)
        sig = ('***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns')
        label_text = f'med={med:+.2f}\np={p:.3f} {sig}'
    else:
        label_text = f'med={med:+.2f}'
    ax.text(i, np.percentile(g, 95) + 0.15,
            label_text, ha='center', fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.5,
            label='Re-only prediction (gap = 0)')
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel(r'Gap = bias$_{obs}$ - $N_d^{predicted}$', fontsize=11)
ax.set_title('(b) Re-prediction residual distribution by state\n'
             '(positive gap → bias higher than Re-error alone predicts)',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Overall title — softened
fig.suptitle(
    'Re-error Propagation Test: Does observed bias track $r_e^{-2.5}$ scaling?\n'
    'Heavy state shows propagation consistency; Non state shows residual offset',
    fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out_path = FIG_DIR / 'cc_fig_re_propagation.png'
fig.savefig(out_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\n[SAVE] {out_path.name}")

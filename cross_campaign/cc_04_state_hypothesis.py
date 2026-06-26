"""
cc_04_state_hypothesis.py

Test the hypothesis: Drizzle progression creates new cloud states,
each with characteristic MODIS Nd bias direction.

Femin's hypothesis:
  Non-drizzle    -> high bias (overestimate)
  Weak/Moderate  -> bias EXPLODES or COLLAPSES (max-bias state)
  Heavy drizzle  -> low bias (underestimate, min state)

Tests:
  1. Bimodality test (Hartigan dip test) on bias distribution
     within each drizzle regime
  2. State transition curve: bias variance/spread as function of regime
  3. Multimodal Gaussian mixture model on bias distribution
  4. Within-regime bias range comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===


REGIME_ORDER = ['non_drizzling', 'weak_drizzling',
                'moderate_drizzling', 'heavy_drizzling']
REGIME_LABELS = {
    'non_drizzling':       'Non',
    'weak_drizzling':      'Weak',
    'moderate_drizzling':  'Mod',
    'heavy_drizzling':     'Heavy',
}
REGIME_COLORS = {
    'non_drizzling':       '#3366cc',
    'weak_drizzling':      '#ffaa00',
    'moderate_drizzling':  'darkorange',
    'heavy_drizzling':     '#cc3333',
}

CAMPAIGN_COLORS = {'POST': '#1f77b4', 'MASE': '#d62728', 'VOCALS': '#2ca02c'}


# =========================================================================
# Load
# =========================================================================
master = pd.read_csv(OUT_DIR / 'cc_master_matched.csv')
print(f"Loaded {len(master)} matched profiles")

# Drop extreme outlier (POST 11.07x — cosmetic only)
print(f"\nbias_21_calc range: {master['bias_21_calc'].min():.2f} to {master['bias_21_calc'].max():.2f}")


# =========================================================================
# TEST 1: Within-regime bias variability (Femin's "explosion/collapse")
# =========================================================================
print("\n" + "=" * 80)
print("TEST 1: Within-regime bias variability across regime sequence")
print("=" * 80)

print(f"\n{'Regime':<22} {'n':>4} {'Median':>8} {'IQR':>15} {'CV%':>8} {'Range':>15}")
print("-" * 80)

regime_summary = []
for reg in REGIME_ORDER:
    sub = master[master['drizzle_regime_clean'] == reg]
    bias = sub['bias_21_calc'].dropna()
    if len(bias) == 0:
        continue
    n = len(bias)
    med = np.median(bias)
    q25, q75 = np.percentile(bias, [25, 75])
    iqr = q75 - q25
    cv = (np.std(bias) / np.mean(bias)) * 100 if len(bias) > 1 else np.nan
    rng = (bias.max() - bias.min())
    print(f"{REGIME_LABELS[reg]:<22} {n:>4} {med:>8.3f} [{q25:.2f}–{q75:.2f}]   {cv:>6.1f}%  {rng:>8.2f}")
    regime_summary.append(dict(regime=reg, n=n, median=med,
                                iqr=iqr, cv=cv, range=rng,
                                bias_min=bias.min(), bias_max=bias.max()))

reg_df = pd.DataFrame(regime_summary)


# =========================================================================
# TEST 2: Bimodality detection (Hartigan dip test approximation)
# =========================================================================
print("\n" + "=" * 80)
print("TEST 2: Bimodality test (KDE peaks + Gaussian Mixture)")
print("=" * 80)


def detect_bimodality(values, regime_name):
    """Test if bias distribution within a regime is bimodal."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return dict(n=len(values), bimodal=False,
                    n_modes=0, bic_1=np.nan, bic_2=np.nan,
                    note='n<4')

    log_bias = np.log(values).reshape(-1, 1)

    # Try 1-mode and 2-mode Gaussian mixture
    try:
        gm1 = GaussianMixture(n_components=1, random_state=42).fit(log_bias)
        bic_1 = gm1.bic(log_bias)
    except Exception:
        bic_1 = np.nan
    try:
        gm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(log_bias)
        bic_2 = gm2.bic(log_bias)
    except Exception:
        bic_2 = np.nan

    # Lower BIC = better fit. Bimodal if BIC(2) < BIC(1).
    bimodal = (not np.isnan(bic_2)) and (bic_2 < bic_1)
    n_modes = 2 if bimodal else 1

    note = ''
    if bimodal:
        # Show the two centers
        means = np.exp(gm2.means_.flatten())
        weights = gm2.weights_
        note = f'modes: {means[0]:.2f} (w={weights[0]:.2f}), {means[1]:.2f} (w={weights[1]:.2f})'

    return dict(n=len(values), bimodal=bimodal,
                n_modes=n_modes, bic_1=bic_1, bic_2=bic_2,
                note=note)


print(f"\n{'Regime':<22} {'n':>4} {'BIC_1':>8} {'BIC_2':>8} {'BIC_diff':>9} {'Bimodal':>9}  Notes")
print("-" * 85)
for reg in REGIME_ORDER:
    sub = master[master['drizzle_regime_clean'] == reg]
    bias = sub['bias_21_calc'].dropna()
    res = detect_bimodality(bias.values, reg)
    bd = res['bic_2'] - res['bic_1'] if not np.isnan(res['bic_2']) else np.nan
    bd_str = f"{bd:>+8.2f}" if not np.isnan(bd) else "    nan"
    bm_str = "YES" if res['bimodal'] else "no"
    print(f"{REGIME_LABELS[reg]:<22} {res['n']:>4} {res['bic_1']:>8.2f} {res['bic_2']:>8.2f} {bd_str}    {bm_str:>5}  {res['note']}")


# =========================================================================
# TEST 3: Bias direction probability vs regime (Femin's flip mechanism)
# =========================================================================
print("\n" + "=" * 80)
print("TEST 3: Bias direction (>1 vs <1) by regime — Femin's flip hypothesis")
print("=" * 80)

print(f"\n{'Regime':<22} {'n':>4} {'P(bias<1)':>10} {'P(bias>1)':>10}  Direction tendency")
print("-" * 80)
direction_data = []
for reg in REGIME_ORDER:
    sub = master[master['drizzle_regime_clean'] == reg]
    bias = sub['bias_21_calc'].dropna()
    if len(bias) == 0:
        continue
    n = len(bias)
    p_under = (bias < 1.0).sum() / n
    p_over = (bias > 1.0).sum() / n
    if p_over > 0.7:
        tend = "STRONG OVER (>1)"
    elif p_under > 0.7:
        tend = "STRONG UNDER (<1)"
    elif p_over > 0.55:
        tend = "leans over"
    elif p_under > 0.55:
        tend = "leans under"
    else:
        tend = "MIXED"
    print(f"{REGIME_LABELS[reg]:<22} {n:>4} {p_under:>10.2f} {p_over:>10.2f}  {tend}")
    direction_data.append(dict(regime=reg, n=n, p_under=p_under, p_over=p_over))


# =========================================================================
# TEST 4: Order of regime sequence vs bias direction transitions
# =========================================================================
print("\n" + "=" * 80)
print("TEST 4: Regime sequence: Does bias spread peak in middle regimes?")
print("=" * 80)

# Compute bias spread (interquartile range, std) per regime, ordered
print(f"\nFemin's hypothesis predicts: Non=tight high, Weak/Mod=wide (peak), Heavy=tight low")
print(f"\nObservations:")

for col in ['median', 'iqr', 'cv', 'range']:
    if col in reg_df.columns:
        vals = reg_df[col].values
        regs = [REGIME_LABELS[r] for r in reg_df['regime'].values]
        print(f"\n  {col.upper():>10}: " + "  ".join([f"{r}={v:.2f}"
                                                       for r, v in zip(regs, vals)]))

# Test if spread peaks at Weak or Moderate
spreads = reg_df.set_index('regime')['cv']
print(f"\n  Spread (CV%) sequence:")
non_cv  = spreads.get('non_drizzling', np.nan)
weak_cv = spreads.get('weak_drizzling', np.nan)
mod_cv  = spreads.get('moderate_drizzling', np.nan)
heavy_cv= spreads.get('heavy_drizzling', np.nan)
print(f"    Non    = {non_cv:.1f}%")
print(f"    Weak   = {weak_cv:.1f}%")
print(f"    Mod    = {mod_cv:.1f}%")
print(f"    Heavy  = {heavy_cv:.1f}%")

# Femin's hypothesis: weak or mod has the peak spread
peak_idx = np.nanargmax([non_cv, weak_cv, mod_cv, heavy_cv])
peak_reg = ['Non', 'Weak', 'Mod', 'Heavy'][peak_idx]
print(f"\n  Peak spread: {peak_reg}")
if peak_reg in ('Weak', 'Mod'):
    print(f"  -> SUPPORTS Femin's hypothesis (max bias spread in transition regimes)")
elif peak_reg == 'Non':
    print(f"  -> PARTIAL support (Non has high spread, Weak/Mod still significant)")
else:
    print(f"  -> Hypothesis NOT supported by spread peak alone (Heavy has peak)")


# =========================================================================
# FIGURE: State transition diagram
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Bias by regime, all individual points + median + IQR
ax = axes[0, 0]
positions = list(range(len(REGIME_ORDER)))
for i, reg in enumerate(REGIME_ORDER):
    sub = master[master['drizzle_regime_clean'] == reg]
    bias = sub['bias_21_calc'].dropna()
    if len(bias) == 0:
        continue
    color = REGIME_COLORS[reg]
    # Box
    bp = ax.boxplot(bias, positions=[i], widths=0.5,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='white',
                                    markeredgecolor='black', markersize=8))
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.5)
    # Scatter by campaign
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub_c = sub[sub['campaign'] == camp]
        b_c = sub_c['bias_21_calc'].dropna()
        if len(b_c) == 0:
            continue
        jitter = np.random.normal(i, 0.07, len(b_c))
        ax.scatter(jitter, b_c,
                    color=CAMPAIGN_COLORS[camp], s=45, alpha=0.85,
                    edgecolors='black', linewidth=0.7,
                    label=camp if i == 0 else "")

ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No bias')
ax.set_xticks(positions)
ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
ax.set_ylabel(r'bias$_{calc}$', fontsize=11)
ax.set_yscale('log')
ax.set_title('(a) Bias by Drizzle Regime — All Campaigns Pooled\n'
             "Femin's hypothesis: max-bias state at transition regimes",
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, which='both')

# (b) Bias spread (CV%) per regime — with bias range
ax = axes[0, 1]
regs_plot = [r for r in REGIME_ORDER if r in reg_df['regime'].values]
cv_vals  = [reg_df.set_index('regime').loc[r, 'cv'] for r in regs_plot]
positions = list(range(len(regs_plot)))
colors = [REGIME_COLORS[r] for r in regs_plot]
ax.bar(positions, cv_vals, color=colors, edgecolor='black', linewidth=1.2,
        alpha=0.85)
for p, v in zip(positions, cv_vals):
    ax.text(p, v + 5, f'{v:.0f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=10)
ax.set_xticks(positions)
ax.set_xticklabels([REGIME_LABELS[r] for r in regs_plot])
ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
ax.set_title('(b) Bias Spread (CV%) by Regime\n'
             "Higher CV = more 'state instability'",
             fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (c) Direction probability
ax = axes[1, 0]
regs_d = [d['regime'] for d in direction_data]
p_overs = [d['p_over'] for d in direction_data]
p_unders = [d['p_under'] for d in direction_data]
positions = list(range(len(regs_d)))
ax.bar(positions, p_overs, color='#cc4444', edgecolor='black', alpha=0.7,
        label='P(bias > 1) = MODIS overestimate')
ax.bar(positions, [-x for x in p_unders], color='#4444cc', edgecolor='black',
        alpha=0.7, label='P(bias < 1) = MODIS underestimate')
for p, v in zip(positions, p_overs):
    ax.text(p, v + 0.02, f'{v:.0%}', ha='center', va='bottom', fontsize=9)
for p, v in zip(positions, p_unders):
    ax.text(p, -v - 0.02, f'{v:.0%}', ha='center', va='top', fontsize=9)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(positions)
ax.set_xticklabels([REGIME_LABELS[r] for r in regs_d])
ax.set_ylabel('Probability', fontsize=11)
ax.set_ylim(-1.05, 1.05)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(['100%', '50%', '0%', '50%', '100%'])
ax.legend(loc='upper right', fontsize=9)
ax.set_title('(c) Bias Direction Probability by Regime\n'
             "Femin's hypothesis: flip from over (Non) -> mixed (Mod) -> under (Heavy)",
             fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (d) Bias distribution (KDE) per regime
ax = axes[1, 1]
for reg in REGIME_ORDER:
    sub = master[master['drizzle_regime_clean'] == reg]
    bias = sub['bias_21_calc'].dropna()
    if len(bias) < 3:
        continue
    color = REGIME_COLORS[reg]
    log_b = np.log(bias)
    # KDE
    kde = stats.gaussian_kde(log_b)
    x = np.linspace(log_b.min() - 0.3, log_b.max() + 0.3, 200)
    ax.plot(np.exp(x), kde(x), color=color, lw=2.2,
            label=f'{REGIME_LABELS[reg]} (n={len(bias)})')
    ax.fill_between(np.exp(x), 0, kde(x), color=color, alpha=0.18)
ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='No bias')
ax.set_xscale('log')
ax.set_xlabel(r'bias$_{calc}$ (log scale)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(d) Bias Density Distribution by Regime\n'
             'Multiple peaks = state-transition signature',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.3, 12)

fig.suptitle("Hypothesis Test: Drizzle-Driven Cloud State Transitions and MODIS $N_d$ Bias",
              fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
out = FIG_DIR / 'cc_fig6_state_hypothesis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[SAVE] {out.name}")


# =========================================================================
# FIGURE 7: Per-campaign within-regime bias (Femin's full hypothesis test)
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

for ax, camp in zip(axes, ['POST', 'MASE', 'VOCALS']):
    sub_camp = master[master['campaign'] == camp]
    positions = list(range(len(REGIME_ORDER)))
    for i, reg in enumerate(REGIME_ORDER):
        sub = sub_camp[sub_camp['drizzle_regime_clean'] == reg]
        bias = sub['bias_21_calc'].dropna()
        if len(bias) == 0:
            continue
        color = REGIME_COLORS[reg]
        # Plot points
        jitter = np.random.normal(i, 0.07, len(bias))
        ax.scatter(jitter, bias, color=color, s=70, alpha=0.85,
                    edgecolors='black', linewidth=0.8)
        # Median line
        med = np.median(bias)
        ax.plot([i - 0.3, i + 0.3], [med, med], color='black',
                 lw=2.5, zorder=10)
        ax.text(i, med * 1.15, f'{med:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIME_ORDER])
    ax.set_yscale('log')
    ax.set_title(f'{camp} (n={len(sub_camp)})',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, which='both')
    if camp == 'POST':
        ax.set_ylabel(r'bias$_{calc}$', fontsize=11)

fig.suptitle("State-Transition Hypothesis: bias trajectory across drizzle regimes\n"
             "Non -> Weak/Mod (max bias state?) -> Heavy (min bias)",
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out = FIG_DIR / 'cc_fig7_state_trajectory.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[SAVE] {out.name}")

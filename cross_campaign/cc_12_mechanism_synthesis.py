"""
cc_12_mechanism_synthesis.py

4-panel synthesis figure of the 4 strongest evidence mechanisms.
M2 (photon penetration) shown as text annotation only — weak direct evidence.
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
all_master = pd.read_csv(OUT_DIR / 'cc_master_all.csv')

m['delta_re'] = m['re_full_median'] - m['re_cas_median']
m['delta_re_vw'] = m['Re_MODIS_21'] - m['re_cas_median']
m['f_ad_gap'] = 0.80 - m['f_ad_mean']

def regime_to_state(reg):
    if reg == 'non_drizzling': return 'Non'
    if reg in ('weak_drizzling', 'moderate_drizzling'): return 'Transition'
    if reg == 'heavy_drizzling': return 'Heavy'
    return 'Unknown'
all_master['drizzle_state'] = all_master['drizzle_regime_clean'].map(regime_to_state)

STATE_COLORS = {'Non': '#3366cc', 'Transition': '#ff8c00', 'Heavy': '#cc3333'}
STATE_ORDER = ['Non', 'Transition', 'Heavy']

plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# =========================================================================
# Panel (a): M1 — Vertical weighting (Re_MODIS - re_cas) by state
# =========================================================================
ax = axes[0, 0]
positions = list(range(3))
data = []
for i, state in enumerate(STATE_ORDER):
    sub = m[m['drizzle_state'] == state]
    d = sub['delta_re_vw'].dropna().values
    color = STATE_COLORS[state]
    bp = ax.boxplot(d, positions=[i], widths=0.5, patch_artist=True,
                     showmeans=True, meanprops=dict(marker='D',
                                                     markerfacecolor='white',
                                                     markeredgecolor='black',
                                                     markersize=8))
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.5)
    jitter = np.random.normal(i, 0.07, len(d))
    ax.scatter(jitter, d, color=color, s=40, alpha=0.7,
                edgecolors='black', linewidth=0.4)
    med = np.median(d)
    q3 = np.percentile(d, 75)
    ax.text(i - 0.18, q3 + 0.30, f'{med:+.2f}', ha='right', va='center',
             fontsize=14, fontweight='bold', color=color)
    data.append(d)

ax.axhline(0, color='black', linestyle='--', alpha=0.6, lw=1.5,
            label='No Re bias')
ax.set_xticks(positions)
ax.set_xticklabels(STATE_ORDER)
ax.set_ylabel(r'$r_{e,\mathrm{MODIS}} - r_{e,\mathrm{CAS}}$ (µm)', fontsize=15)
# K-W
kw, p = stats.kruskal(*data)
ax.set_title('(a) Vertical weighting', fontsize=16, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.3)

# =========================================================================
# Panel (b): M3 — Drizzle tail Δre × bias by state (Transition focus)
# =========================================================================
ax = axes[0, 1]
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state].dropna(subset=['delta_re', 'bias_21_calc'])
    if len(sub) == 0:
        continue
    color = STATE_COLORS[state]
    ax.scatter(sub['delta_re'], sub['bias_21_calc'],
                color=color, s=70, alpha=0.85,
                edgecolors='black', linewidth=0.5,
                label=f'{state} (n={len(sub)})')
    # Spearman per state
    if len(sub) >= 4:
        r, p = stats.spearmanr(sub['delta_re'], sub['bias_21_calc'])

# Annotate Transition correlation
trans = m[m['drizzle_state'] == 'Transition'].dropna(
    subset=['delta_re', 'bias_21_calc'])
r, p = stats.spearmanr(trans['delta_re'], trans['bias_21_calc'])
sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
ax.text(0.97, 0.97,
        f'Transition: Spearman $r$={r:+.2f} {sig} (n={len(trans)})',
        transform=ax.transAxes, va='top', ha='right',
        fontsize=12.5, fontweight='bold', color=STATE_COLORS['Transition'])

ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No bias')
ax.set_xlabel(r'$\Delta r_e = r_{e,\mathrm{full}} - r_{e,\mathrm{CAS}}$ (µm)', fontsize=15)
ax.set_ylabel(r'$B_{\mathrm{calc}}$', fontsize=15)
ax.set_yscale('log')
ax.set_title(r'(b) Drizzle-tail contribution to $r_e$', fontsize=16, fontweight='bold', loc='left')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.90), frameon=False, fontsize=12)
ax.grid(True, alpha=0.3, which='both')

# =========================================================================
# Panel (c): M4 — Match rate U-shape (subpixel heterogeneity)
# =========================================================================
ax = axes[1, 0]
match_rates = {}
totals = {}
for state in STATE_ORDER:
    sub = all_master[all_master['drizzle_state'] == state]
    n_total = len(sub)
    n_matched = (sub['match_status'] == 'MATCHED').sum()
    match_rates[state] = (n_matched / n_total * 100) if n_total else 0
    totals[state] = (n_matched, n_total)

positions = list(range(3))
colors = [STATE_COLORS[s] for s in STATE_ORDER]
rates = [match_rates[s] for s in STATE_ORDER]

bars = ax.bar(positions, rates, color=colors, edgecolor='black',
               linewidth=1.5, alpha=0.85)
for p, r, s in zip(positions, rates, STATE_ORDER):
    n_m, n_t = totals[s]
    ax.text(p, r + 2, f'{r:.0f}%\n({n_m}/{n_t})',
            ha='center', va='bottom', fontsize=14)

# Add U-shape annotation
ax.plot(positions, rates, 'k--', lw=1.5, alpha=0.4, zorder=10)
ax.axhline(np.mean(rates), color='gray', linestyle=':', alpha=0.5,
            label=f'Mean = {np.mean(rates):.1f}%')

ax.set_xticks(positions)
ax.set_xticklabels(STATE_ORDER, fontsize=15)
ax.set_ylabel('MODIS QC pass rate (%)', fontsize=15)
ax.set_ylim(0, 70)
ax.set_title('(c) Subpixel heterogeneity', fontsize=16, fontweight='bold', loc='left')
ax.legend(loc='upper right', frameon=False, fontsize=12)
ax.grid(axis='y', alpha=0.3)

# =========================================================================
# Panel (d): M5 — Nd inversion sensitivity (theoretical + f_ad gap)
# =========================================================================
ax = axes[1, 1]

# Sub-panel: theoretical re sensitivity curve
re_range = np.linspace(0, 4, 100)  # ΔRe range µm
re_base = 10.5  # Heavy state median
nd_ratio = (1 + re_range / re_base) ** (-2.5)

ax.plot(re_range, nd_ratio, 'k-', lw=2.5,
        label=r'Theoretical: $N_c \propto r_e^{-2.5}$')

# Mark each state's median ΔRe_vw and observed bias
for state in STATE_ORDER:
    sub = m[m['drizzle_state'] == state]
    dre_med = sub['delta_re_vw'].dropna().median()
    bias_med = sub['bias_21_calc'].dropna().median()
    color = STATE_COLORS[state]

    # Plot observed bias at this ΔRe
    ax.scatter([dre_med], [bias_med], color=color, s=200, marker='*',
                edgecolors='black', linewidth=1.5, zorder=10,
                label=f'{state}: $\\Delta r_e$={dre_med:.2f}, $B$={bias_med:.2f}')

    # Vertical line down to predicted curve
    if dre_med <= re_range.max():
        predicted = (1 + dre_med / re_base) ** (-2.5)
        ax.plot([dre_med, dre_med], [predicted, bias_med],
                color=color, linestyle=':', alpha=0.5, lw=1.5)
        # Annotation: gap between predicted and observed
        gap = bias_med - predicted
        ax.text(dre_med + 0.15, bias_med,
                f'Δ={gap:+.2f}\n(other factors)',
                fontsize=11.5, color=color, alpha=0.95, va='center', ha='left',
                fontweight='bold')

ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No bias')
ax.set_xlabel(r'$r_{e,\mathrm{MODIS}} - r_{e,\mathrm{CAS}}$ (µm)', fontsize=15)
ax.set_ylabel(r'$B_{\mathrm{calc}}$ (median)', fontsize=15)
ax.set_xlim(0, 4)
ax.set_ylim(0.4, 1.4)
ax.set_title(r'(d) $N_c$ inversion sensitivity', fontsize=16, fontweight='bold', loc='left')
ax.legend(loc='upper left', frameon=False, fontsize=11)
ax.grid(True, alpha=0.3)

# Title
# AGU: no in-figure title (moved to caption)
# fig.suptitle(...)
plt.tight_layout()
out = FIG_DIR / 'cc_fig_mechanisms.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"[SAVE] {out.name}")

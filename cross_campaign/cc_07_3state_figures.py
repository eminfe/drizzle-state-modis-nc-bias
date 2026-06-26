"""
cc_07_3state_figures.py

Cross-campaign figures using 3-state grouping (Non / Transition / Heavy).

Output figures:
  cc_3st_fig1_forest.png         - 3-state forest plot with bootstrap CI
  cc_3st_fig2_state_trajectory.png - State transition trajectory
  cc_3st_fig3_state_signature.png  - Physical state signature (CTT, fad, re, etc)
  cc_3st_fig4_state_density.png    - Bias density by 3-state
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


STATE_ORDER = ['Non', 'Transition', 'Heavy']

STATE_COLORS = {
    'Non':         '#3366cc',  # blue   - over baskın
    'Transition':  '#ff8c00',  # orange - max bias state
    'Heavy':       '#cc3333',  # red    - under, stable
}

CAMPAIGN_COLORS = {'POST': '#1f77b4', 'MASE': '#d62728', 'VOCALS': '#2ca02c'}

# Load
master = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')
ci = pd.read_csv(OUT_DIR / 'cc_bootstrap_3state.csv')
print(f"Loaded {len(master)} matched profiles, {len(ci)} CI rows")


# =========================================================================
# FIG 1: 3-state forest plot
# =========================================================================
def fig1_forest_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    metrics = ['bias_21_calc', 'bias_21_lit', 'inflation_21']
    titles = [
        r'(a) $B_{\mathrm{calc}}$ (in-situ assumptions)',
        r'(b) $B_{\mathrm{lit}}$ (literature defaults)',
        r'(c) Inflation factor ($B_{\mathrm{lit}}/B_{\mathrm{calc}}$)',
    ]
    no_bias = [1.0, 1.0, 1.0]

    for ax, m, title, x0 in zip(axes, metrics, titles, no_bias):
        rows = ci[(ci['group_type'] == 'state') & (ci['metric'] == m)]
        for i, state in enumerate(STATE_ORDER):
            r = rows[rows['group'] == state]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            color = STATE_COLORS[state]
            ax.errorbar(r['value'], i,
                         xerr=[[r['value'] - r['ci_lower']],
                               [r['ci_upper'] - r['value']]],
                         fmt='o', color=color, markersize=14,
                         capsize=10, elinewidth=3, capthick=3,
                         markerfacecolor=color,
                         markeredgecolor='white', markeredgewidth=2,
                         label=f"{state} (n={int(r['n'])})")
            ax.text(r['value'], i - 0.16,
                    f"{r['value']:.2f}", ha='center', va='top',
                    fontsize=14, fontweight='bold', color=color)
            ax.text(r['value'], i - 0.40,
                    f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
                    ha='center', va='top', fontsize=11.5, color=color)

        ax.axvline(x0, color='black', linestyle='--', alpha=0.5,
                    linewidth=1.5, label='No bias')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(STATE_ORDER, fontsize=14)
        ax.set_ylim(2.7, -0.6)
        ax.set_title(title, fontsize=17, fontweight='bold', loc='left')
        ax.set_xlabel(r'$N_c^{\mathrm{MODIS}} / N_c^{\mathrm{CAS}}$', fontsize=17)
        ax.tick_params(axis='x', labelsize=13)
        ax.grid(axis='x', alpha=0.3)
        if m == 'bias_21_lit':
            ax.set_xlim(0.4, 1.6)
        elif m == 'inflation_21':
            ax.set_xlim(0.8, 1.4)
        else:
            ax.set_xlim(0.4, 1.75)
        if ax is axes[0]:
            ax.legend(loc='lower right', frameon=False, fontsize=12)

    # AGU: no in-figure title (moved to caption)
    # fig.suptitle(...)
    plt.tight_layout()
    out = FIG_DIR / 'cc_3st_fig1_forest.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 2: State trajectory (per campaign)
# =========================================================================
def fig2_state_trajectory():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5),
                              gridspec_kw={'width_ratios': [1.2, 1, 1, 1]})

    # Panel 0: Pooled (all campaigns)
    ax = axes[0]
    positions = list(range(3))
    for i, state in enumerate(STATE_ORDER):
        sub = master[master['drizzle_state'] == state]
        bias = sub['bias_21_calc'].dropna()
        if len(bias) == 0:
            continue
        color = STATE_COLORS[state]
        bp = ax.boxplot(bias, positions=[i], widths=0.5,
                         patch_artist=True, showmeans=True,
                         meanprops=dict(marker='D',
                                        markerfacecolor='white',
                                        markeredgecolor='black',
                                        markersize=9))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.5)
        # Scatter by campaign
        for camp in ['POST', 'MASE', 'VOCALS']:
            sub_c = sub[sub['campaign'] == camp]
            b_c = sub_c['bias_21_calc'].dropna()
            if len(b_c) == 0:
                continue
            jitter = np.random.normal(i, 0.08, len(b_c))
            ax.scatter(jitter, b_c,
                        color=CAMPAIGN_COLORS[camp], s=55, alpha=0.85,
                        edgecolors='black', linewidth=0.7,
                        label=camp if i == 0 else "")
        med = np.median(bias)
        ax.text(i, med * 4.8, f'{med:.2f}\n(n={len(bias)})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No bias')
    ax.set_xticks(positions)
    ax.set_xticklabels(STATE_ORDER, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylabel(r'bias$_{calc}$', fontsize=12)
    ax.set_title('POOLED (n=52)\nAll campaigns', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=0.3, which='both')

    # Panels 1, 2, 3: each campaign separately
    for ax_idx, camp in enumerate(['POST', 'MASE', 'VOCALS']):
        ax = axes[ax_idx + 1]
        sub_camp = master[master['campaign'] == camp]
        for i, state in enumerate(STATE_ORDER):
            sub = sub_camp[sub_camp['drizzle_state'] == state]
            bias = sub['bias_21_calc'].dropna()
            if len(bias) == 0:
                continue
            color = STATE_COLORS[state]
            jitter = np.random.normal(i, 0.07, len(bias))
            ax.scatter(jitter, bias, color=color, s=80, alpha=0.85,
                        edgecolors='black', linewidth=0.8)
            med = np.median(bias)
            ax.plot([i - 0.3, i + 0.3], [med, med],
                     color='black', lw=2.5, zorder=10)
            ax.text(i, med * 4.2, f'{med:.2f}\nn={len(bias)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(STATE_ORDER)
        ax.set_yscale('log')
        ax.set_ylim(0.3, 12)
        ax.set_title(f'{camp} (n={len(sub_camp)})',
                      fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, which='both')

    fig.suptitle('State Transition Trajectory: bias evolves Non -> Transition -> Heavy',
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'cc_3st_fig2_state_trajectory.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 3: State signature - physical parameters by state
# =========================================================================
def fig3_state_signature():
    """What's the physical signature of each state?"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    params = [
        ('Nd_median',      r'$N_d^{in-situ}$ (cm$^{-3}$)', True),
        ('re_cas_median',  r'$r_e$ CAS (µm)',              False),
        ('f_ad_mean',      r'$f_{ad}$',                     False),
        ('drizzle_fraction', 'Drizzle fraction',            True),
        ('cloud_depth',    'Cloud depth (m)',               False),
        ('LWP_insitu',     'LWP in-situ (g/m²)',            False),
        ('CTT_MODIS',      'CTT MODIS (K)',                 False),
        ('CTP_MODIS',      'CTP MODIS (hPa)',               False),
    ]

    for ax, (col, ylabel, log_y) in zip(axes.flat, params):
        positions = list(range(3))
        for i, state in enumerate(STATE_ORDER):
            sub = master[master['drizzle_state'] == state]
            d = sub[col].dropna()
            if len(d) == 0:
                continue
            color = STATE_COLORS[state]
            bp = ax.boxplot(d, positions=[i], widths=0.5,
                             patch_artist=True, showmeans=True,
                             meanprops=dict(marker='D',
                                            markerfacecolor='white',
                                            markeredgecolor='black',
                                            markersize=8))
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.5)
            jitter = np.random.normal(i, 0.07, len(d))
            ax.scatter(jitter, d, color=color, s=35, alpha=0.7,
                        edgecolors='black', linewidth=0.4)

        if log_y:
            ax.set_yscale('log')
        ax.set_xticks(positions)
        ax.set_xticklabels(STATE_ORDER, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis='y', alpha=0.3, which='both' if log_y else 'major')

        # K-W test
        groups = [master[master['drizzle_state'] == s][col].dropna()
                  for s in STATE_ORDER]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                kw, p = stats.kruskal(*groups)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.set_title(f'K-W p={p:.3f} {sig}', fontsize=11, fontweight='bold')
            except Exception:
                ax.set_title('K-W: N/A', fontsize=11)

    fig.suptitle('Physical Signature of Each Cloud State',
                  fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = FIG_DIR / 'cc_3st_fig3_state_signature.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 4: Bias density distribution by state
# =========================================================================
def fig4_state_density():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: KDE per state
    ax = axes[0]
    for state in STATE_ORDER:
        sub = master[master['drizzle_state'] == state]
        bias = sub['bias_21_calc'].dropna()
        if len(bias) < 3:
            continue
        color = STATE_COLORS[state]
        log_b = np.log(bias)
        kde = stats.gaussian_kde(log_b)
        x = np.linspace(log_b.min() - 0.4, log_b.max() + 0.4, 200)
        ax.plot(np.exp(x), kde(x), color=color, lw=2.5,
                label=f'{state} (n={len(bias)})')
        ax.fill_between(np.exp(x), 0, kde(x), color=color, alpha=0.20)
        # Mark median
        med = np.median(bias)
        ax.axvline(med, color=color, linestyle=':', alpha=0.7)

    ax.axvline(1.0, color='black', linestyle='--', alpha=0.7, lw=1.5,
                label='No bias')
    ax.set_xscale('log')
    ax.set_xlabel(r'bias$_{calc}$ (log scale)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(a) Bias Density by State\n'
                 'Distribution evolves: wide-mixed -> narrow-converged',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.3, 12)

    # Right: Direction probability + CV
    ax = axes[1]
    p_overs = []
    p_unders = []
    cvs = []
    ns = []
    for state in STATE_ORDER:
        sub = master[master['drizzle_state'] == state]
        bias = sub['bias_21_calc'].dropna()
        if len(bias) == 0:
            continue
        n = len(bias)
        p_o = (bias > 1.0).sum() / n
        p_u = (bias < 1.0).sum() / n
        cv = (bias.std() / bias.mean()) * 100
        p_overs.append(p_o)
        p_unders.append(p_u)
        cvs.append(cv)
        ns.append(n)

    positions = list(range(3))
    width = 0.35

    # Direction probability bars
    bars_over = ax.bar([p - width/2 for p in positions], p_overs,
                       width=width, color='#cc4444', alpha=0.7,
                       edgecolor='black', linewidth=1,
                       label='P(bias > 1) overestimate')
    bars_under = ax.bar([p + width/2 for p in positions], p_unders,
                        width=width, color='#4444cc', alpha=0.7,
                        edgecolor='black', linewidth=1,
                        label='P(bias < 1) underestimate')

    for p, v in zip(positions, p_overs):
        ax.text(p - width/2, v + 0.02, f'{v:.0%}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    for p, v in zip(positions, p_unders):
        ax.text(p + width/2, v + 0.02, f'{v:.0%}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # CV labels (right axis)
    ax2 = ax.twinx()
    ax2.plot(positions, cvs, 'k-o', lw=2.5, markersize=12,
              markerfacecolor='gold', markeredgecolor='black',
              markeredgewidth=2, zorder=10)
    for p, cv, n in zip(positions, cvs, ns):
        ax2.text(p, cv + 8, f'CV={cv:.0f}%\nn={n}',
                  ha='center', va='bottom', fontsize=10,
                  fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax2.set_ylim(0, max(cvs) * 1.4)

    ax.set_xticks(positions)
    ax.set_xticklabels(STATE_ORDER, fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('(b) Bias Direction & Spread by State\n'
                 'Stability increases as drizzle matures',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper center', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Cloud State Hypothesis: bias direction flips with drizzle progression',
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'cc_3st_fig4_state_density.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    fig1_forest_plot()
    fig2_state_trajectory()
    fig3_state_signature()
    fig4_state_density()
    print("\nAll 3-state figures saved.")

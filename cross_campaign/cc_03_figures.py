"""
cc_03_figures.py  -  Cross-Campaign Paper Figures

Generates publication-quality figures comparing POST + MASE + VOCALS:

  Fig 1: Forest plot - bias_calc, bias_lit, inflation per campaign with 95% CI
  Fig 2: 3-panel comparison - bias_calc by drizzle regime per campaign
  Fig 3: Inflation factor consistency - bias_calc vs bias_lit, all campaigns
  Fig 4: Bias direction map - which conditions flip bias direction?
  Fig 5: Bias driver heatmap - Spearman correlations across campaigns

OUTPUT:
  {FIG_DIR}/cc_fig1_forest.png
  {FIG_DIR}/cc_fig2_regime_comparison.png
  {FIG_DIR}/cc_fig3_inflation_scatter.png
  {FIG_DIR}/cc_fig4_bias_direction.png
  {FIG_DIR}/cc_fig5_driver_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# Campaign colors
COLORS = {
    'POST'   : '#1f77b4',  # blue
    'MASE'   : '#d62728',  # red
    'VOCALS' : '#2ca02c',  # green
}

REGIME_COLORS = {
    'non_drizzling':       '#3366cc',
    'weak_drizzling':      '#ff9933',
    'moderate_drizzling':  '#cc6600',
    'heavy_drizzling':     '#cc3333',
}

REGIME_LABELS = {
    'non_drizzling':       'Non',
    'weak_drizzling':      'Weak',
    'moderate_drizzling':  'Mod',
    'heavy_drizzling':     'Heavy',
}

# Load data
master = pd.read_csv(OUT_DIR / 'cc_master_matched.csv')
ci = pd.read_csv(OUT_DIR / 'cc_bootstrap_ci.csv')

print(f"Loaded {len(master)} matched profiles")
print(f"Loaded {len(ci)} CI rows")


# =========================================================================
# FIG 1: Forest plot
# =========================================================================
def fig1_forest_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             sharey=False)
    metrics = ['bias_21_calc', 'bias_21_lit', 'inflation_21']
    titles = [
        '(a) bias_calc (in-situ assumptions)',
        '(b) bias_lit (literature defaults)',
        '(c) Inflation Factor (lit / calc)',
    ]
    xlabels = [
        r'$N_d^{MODIS} / N_d^{in-situ}$',
        r'$N_d^{MODIS} / N_d^{in-situ}$',
        r'bias$_{lit}$ / bias$_{calc}$',
    ]
    no_bias = [1.0, 1.0, 1.0]

    for ax, m, title, xlab, x0 in zip(axes, metrics, titles, xlabels, no_bias):
        rows = ci[(ci['group_type'] == 'campaign') & (ci['metric'] == m)]

        y_positions = []
        for i, camp in enumerate(['POST', 'VOCALS', 'MASE']):
            r = rows[rows['group'] == camp].iloc[0]
            y = i
            y_positions.append((y, camp))
            color = COLORS[camp]
            # Error bar
            ax.errorbar(r['value'], y,
                        xerr=[[r['value'] - r['ci_lower']],
                              [r['ci_upper'] - r['value']]],
                        fmt='o', color=color, markersize=12,
                        capsize=8, elinewidth=2.5, capthick=2.5,
                        markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=1.5,
                        label=f"{camp} (n={r['n']})")
            # Median value annotation
            ax.text(r['value'], y + 0.18,
                    f"{r['value']:.2f}",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color)
            ax.text(r['ci_upper'] + 0.05, y,
                    f"  [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
                    va='center', fontsize=9, color='#444')

        # POOLED row
        rows_p = ci[(ci['group_type'] == 'pooled') & (ci['metric'] == m)]
        if len(rows_p):
            r = rows_p.iloc[0]
            y = 3
            ax.errorbar(r['value'], y,
                        xerr=[[r['value'] - r['ci_lower']],
                              [r['ci_upper'] - r['value']]],
                        fmt='D', color='black', markersize=12,
                        capsize=8, elinewidth=2.5, capthick=2.5,
                        markerfacecolor='gold',
                        markeredgecolor='black', markeredgewidth=1.5)
            ax.text(r['value'], y + 0.18,
                    f"{r['value']:.2f}",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')
            ax.text(r['ci_upper'] + 0.05, y,
                    f"  [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]  (n={r['n']})",
                    va='center', fontsize=9, color='#222')

        # Reference line (no bias)
        ax.axvline(x0, color='black', linestyle='--', alpha=0.5, linewidth=1.5,
                   zorder=0, label='No bias' if m != 'inflation_21' else 'No inflation')

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['POST', 'VOCALS', 'MASE', 'POOLED'])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlab, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=0)
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle('Cross-Campaign MODIS $N_d$ Bias - Bootstrap 95% CI (10,000 iterations)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'cc_fig1_forest.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 2: 3-panel regime comparison
# =========================================================================
def fig2_regime_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    regimes = ['non_drizzling', 'weak_drizzling',
               'moderate_drizzling', 'heavy_drizzling']

    for ax, camp in zip(axes, ['POST', 'MASE', 'VOCALS']):
        sub = master[master['campaign'] == camp]

        data = []
        labels = []
        positions = []
        colors = []
        for i, reg in enumerate(regimes):
            d = sub[sub['drizzle_regime_clean'] == reg]['bias_21_calc'].dropna()
            if len(d) == 0:
                continue
            data.append(d.values)
            labels.append(f"{REGIME_LABELS[reg]}\n(n={len(d)})")
            positions.append(i)
            colors.append(REGIME_COLORS[reg])

        if not data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                         showmeans=True, meanprops=dict(marker='D',
                                                         markerfacecolor='white',
                                                         markeredgecolor='black',
                                                         markersize=7))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        # Scatter individual points
        for i, (d, pos, c) in enumerate(zip(data, positions, colors)):
            jitter = np.random.normal(pos, 0.08, len(d))
            ax.scatter(jitter, d, color=c, alpha=0.7, s=30,
                       edgecolors='black', linewidth=0.5)

        # No-bias line
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, lw=1.5)

        # Median annotation
        for d, pos in zip(data, positions):
            ax.text(pos, np.median(d) + 0.05, f"{np.median(d):.2f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # K-W test
        if len(data) >= 2 and all(len(d) >= 2 for d in data):
            try:
                kw, p = stats.kruskal(*data)
                p_str = f"K-W p={p:.3f}" + (' ***' if p < 0.001 else
                                              ' **' if p < 0.01 else
                                              ' *' if p < 0.05 else ' ns')
            except Exception:
                p_str = 'K-W: N/A'
        else:
            p_str = 'K-W: N/A'

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Non', 'Weak', 'Mod', 'Heavy'])
        ax.set_title(f'{camp} (n={len(sub)})\n{p_str}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Drizzle Regime', fontsize=11)
        if camp == 'POST':
            ax.set_ylabel(r'bias$_{calc}$ = $N_d^{MODIS} / N_d^{in-situ}$',
                          fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')

    fig.suptitle(
        'MODIS $N_d$ Bias by Drizzle Regime: Cross-Campaign Comparison\n'
        '(in-situ assumptions; 1:1 = no bias)',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'cc_fig2_regime_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 3: bias_calc vs bias_lit scatter (all campaigns)
# =========================================================================
def fig3_inflation_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # LEFT: bias_calc vs bias_lit, color by campaign
    ax = axes[0]
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub = master[master['campaign'] == camp]
        ax.scatter(sub['bias_21_calc'], sub['bias_21_lit'],
                   color=COLORS[camp], s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5,
                   label=f"{camp} (n={len(sub)})")
    # 1:1 line
    lim = [0.3, 12]
    ax.plot(lim, lim, 'k--', alpha=0.5, label='1:1')
    ax.plot(lim, [x * 1.5 for x in lim], color='gray', linestyle=':', alpha=0.6,
            label='1.5× (mean inflation)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'bias$_{calc}$ (in-situ assumptions)', fontsize=11)
    ax.set_ylabel(r'bias$_{lit}$ (literature defaults)', fontsize=11)
    ax.set_title('(a) bias$_{lit}$ ≥ bias$_{calc}$ in ALL profiles\n'
                 '(literature defaults always inflate)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # RIGHT: Inflation factor distribution per campaign
    ax = axes[1]
    data = []
    labels = []
    colors_list = []
    for camp in ['POST', 'VOCALS', 'MASE']:
        sub = master[master['campaign'] == camp]
        d = sub['inflation_21'].dropna()
        data.append(d.values)
        labels.append(f"{camp}\n(n={len(d)})")
        colors_list.append(COLORS[camp])

    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.6, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D',
                                                    markerfacecolor='white',
                                                    markeredgecolor='black',
                                                    markersize=8))
    for patch, c in zip(bp['boxes'], colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Scatter individual
    for i, (d, c) in enumerate(zip(data, colors_list)):
        jitter = np.random.normal(i, 0.08, len(d))
        ax.scatter(jitter, d, color=c, alpha=0.7, s=40,
                   edgecolors='black', linewidth=0.5)

    # Bootstrap CIs as horizontal bands
    inflation_ci = ci[(ci['group_type'] == 'campaign') &
                       (ci['metric'] == 'inflation_21')]
    for i, camp in enumerate(['POST', 'VOCALS', 'MASE']):
        r = inflation_ci[inflation_ci['group'] == camp].iloc[0]
        ax.plot([i - 0.4, i + 0.4], [r['value'], r['value']],
                color='black', linewidth=2, zorder=10)
        ax.text(i, r['ci_upper'] + 0.05,
                f"{r['value']:.2f}\n[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5,
               label='No inflation')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Inflation Factor (bias$_{lit}$ / bias$_{calc}$)',
                  fontsize=11)
    ax.set_title(f'(b) Inflation Factor: ~1.5× across all campaigns',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.7, 3.5)

    fig.suptitle(
        'Literature Default Assumptions Systematically Inflate MODIS $N_d$ Retrievals\n'
        'across Three Independent Marine Stratocumulus Campaigns',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = FIG_DIR / 'cc_fig3_inflation_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 4: Bias direction map
# =========================================================================
def fig4_bias_direction():
    """When does bias flip from > 1 to < 1?"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    drivers = [
        ('Nd_median',  r'$N_d^{in-situ}$ (cm$^{-3}$)', True),
        ('LWP_insitu', 'LWP in-situ (g/m²)',          True),
        ('CTT_MODIS',  'CTT MODIS (K)',                False),
        ('drizzle_fraction', 'Drizzle fraction',       True),
    ]

    for ax, (col, xlabel, log_x) in zip(axes.flat, drivers):
        for camp in ['POST', 'MASE', 'VOCALS']:
            sub = master[master['campaign'] == camp].dropna(
                subset=[col, 'bias_21_calc'])
            ax.scatter(sub[col], sub['bias_21_calc'],
                       color=COLORS[camp], s=60, alpha=0.6,
                       edgecolors='black', linewidth=0.5,
                       label=f"{camp} (n={len(sub)})")
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5,
                   label='No bias (flip line)')
        if log_x:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(r'bias$_{calc}$', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        # Spearman across all campaigns
        all_data = master.dropna(subset=[col, 'bias_21_calc'])
        if len(all_data) > 5:
            rho, p = stats.spearmanr(all_data[col], all_data['bias_21_calc'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(0.02, 0.98,
                    f'All campaigns:\n  Spearman r = {rho:+.3f}\n  p = {p:.4f} {sig}',
                    transform=ax.transAxes, fontsize=9,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    fig.suptitle(
        'Bias Direction Map: Conditions that Flip MODIS $N_d$ Bias Direction\n'
        '(below 1 = MODIS underestimate; above 1 = MODIS overestimate)',
        fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = FIG_DIR / 'cc_fig4_bias_direction.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# FIG 5: Driver heatmap
# =========================================================================
def fig5_driver_heatmap():
    """Spearman correlation of log(bias) vs drivers, by campaign."""
    drivers = [
        ('CTT_MODIS',          'CTT (K)'),
        ('CTP_MODIS',          'CTP (hPa)'),
        ('Nd_median',          r'$N_d^{in-situ}$'),
        ('LWP_insitu',         'LWP in-situ'),
        ('tau_main',           r'$\tau$ in-situ'),
        ('re_cas_median',      r'$r_e$ CAS'),
        ('f_ad_mean',          r'$f_{ad}$'),
        ('drizzle_fraction',   'Drizzle frac'),
        ('SZA_mean',           'SZA'),
        ('VZA_mean',           'VZA'),
    ]

    rho_matrix = np.full((len(drivers), 3), np.nan)
    p_matrix   = np.full((len(drivers), 3), np.nan)

    for j, camp in enumerate(['POST', 'MASE', 'VOCALS']):
        sub = master[master['campaign'] == camp].dropna(subset=['bias_21_calc'])
        for i, (col, _) in enumerate(drivers):
            if col not in sub.columns:
                continue
            d = sub.dropna(subset=[col, 'bias_21_calc'])
            if len(d) < 4:
                continue
            log_bias = np.log(d['bias_21_calc'].values)
            rho, p = stats.spearmanr(d[col].values, log_bias)
            rho_matrix[i, j] = rho
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Annotations
    for i in range(len(drivers)):
        for j in range(3):
            r = rho_matrix[i, j]
            p = p_matrix[i, j]
            if np.isnan(r):
                txt = '—'
            else:
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                txt = f'{r:+.2f}\n{sig}' if sig else f'{r:+.2f}'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=9,
                    color='white' if abs(r) > 0.5 else 'black')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['POST', 'MASE', 'VOCALS'])
    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels([d[1] for d in drivers])
    ax.set_title(r'Spearman $r$: log(bias$_{calc}$) vs Drivers',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label=r'Spearman $r$')
    plt.tight_layout()
    out = FIG_DIR / 'cc_fig5_driver_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {out.name}")


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    fig1_forest_plot()
    fig2_regime_comparison()
    fig3_inflation_scatter()
    fig4_bias_direction()
    fig5_driver_heatmap()
    print("\nAll figures saved.")

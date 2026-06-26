# =============================================================================
# step11_packageA.py — Package A: In-Situ Cloud Physics
# =============================================================================
# Campaign : VOCALS-REx 2008
# Goal     : Do drizzle regimes alter the physical structure of clouds?
# Data     : VOCALS_golden_case.csv  ->  gc (filtered profiles)
# Outputs  : outputs/figures/VOCALS_PackageA_main.png
#            outputs/figures/VOCALS_PackageA_bonus.png
#            outputs/figures/VOCALS_PackageA_bimodal.png
#            outputs/VOCALS_PackageA_notes.md
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import kruskal, pearsonr, spearmanr, gaussian_kde
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths (config-driven)
# =============================================================================
import config

NAME       = config.CAMPAIGN_NAME
FIG_DIR    = config.FIG_DIR

# Input CSV (filtered output from step07)
GC_CSV     = config.STEP04_GOLDEN_CASE_CSV

# Output figure paths
OUT_MAIN    = FIG_DIR / f'{NAME}_PackageA_main.png'
OUT_BONUS   = FIG_DIR / f'{NAME}_PackageA_bonus.png'
OUT_BIMODAL = FIG_DIR / f'{NAME}_PackageA_bimodal.png'
OUT_NOTES   = config.OUTPUT_DIR / f'{NAME}_PackageA_notes.md'



# ============================================================
# 0. DATA LOADING
# ============================================================
gc = pd.read_csv(GC_CSV)

gc['drizzle_regime'] = (
    gc['drizzle_regime'].astype(str).str.strip()
    .str.replace('_', ' ').str.title()
    .str.replace('Drizzling', 'drizzling')
    .str.replace('Non drizzling', 'Non-drizzling')
)

REGIME_ORDER  = ['Non-drizzling', 'Weak drizzling', 'Moderate drizzling', 'Heavy drizzling']
REGIME_COLORS = {
    'Non-drizzling'     : '#2E7D32',
    'Weak drizzling'    : '#F9A825',
    'Moderate drizzling': '#E65100',
    'Heavy drizzling'   : '#B71C1C',
}
SHORT_LABELS = ['Non-\ndrizzle', 'Weak\ndrizzle', 'Moderate\ndrizzle', 'Heavy\ndrizzle']

gc = gc[gc['drizzle_regime'].isin(REGIME_ORDER)].copy()
gc['drizzle_regime'] = pd.Categorical(
    gc['drizzle_regime'], categories=REGIME_ORDER, ordered=True)

print(f"Total profiles: {len(gc)}")
print(gc['drizzle_regime'].value_counts()[REGIME_ORDER])

# ============================================================
# 1. STYLE
# ============================================================
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#111111',
    'xtick.color': '#111111', 'ytick.color': '#111111',
    'text.color': '#111111', 'grid.color': '#dddddd',
    'grid.linestyle': '--', 'grid.linewidth': 0.6, 'font.size': 10,
})
TXT   = '#111111'
ANNOT = '#333333'
KW_BG = '#F5F5F5'
KW_ED = '#BBBBBB'

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def kw_label(groups):
    valid = [g.dropna().values for g in groups if g.dropna().shape[0] >= 2]
    if len(valid) < 2:
        return "n/a"
    _, p = kruskal(*valid)
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
    return f"K-W p<0.001 {stars}" if p < 0.001 else f"K-W p={p:.3f} {stars}"


def single_boxplot(ax, gc, col, ylabel, title, show_kw=True, hline=None, hline_label=None):
    data   = [gc[gc['drizzle_regime'] == r][col].dropna() for r in REGIME_ORDER]
    colors = [REGIME_COLORS[r] for r in REGIME_ORDER]

    bp = ax.boxplot(
        data, patch_artist=True, widths=0.52,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(color='#555555', linewidth=1.2),
        capprops=dict(color='#555555', linewidth=1.2),
        flierprops=dict(marker='o', markersize=5,
                        markerfacecolor='#999999', linestyle='none', alpha=0.6),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.uniform(-0.18, 0.18, size=len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                   color=color, s=30, alpha=0.70, zorder=4,
                   edgecolors='white', linewidths=0.5)
        if len(d) > 0:
            med = np.nanmedian(d)
            ax.text(i + 1.30, med, f'{med:.2f}',
                    va='center', ha='left', color=TXT,
                    fontsize=7.5, fontweight='bold')

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
    ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=TXT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)

    if hline is not None:
        ax.axhline(hline, color='#555555', lw=1.2, ls='--', alpha=0.6)
        if hline_label:
            ax.text(0.02, hline, f' {hline_label}',
                    transform=ax.get_yaxis_transform(),
                    color='#555555', fontsize=7.5, va='bottom')

    if show_kw:
        kw_str = kw_label(data)
        ax.text(0.98, 0.97, kw_str, transform=ax.transAxes,
                ha='right', va='top', color=ANNOT, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=KW_BG, edgecolor=KW_ED, alpha=0.95))


def dual_boxplot(ax, gc, col1, col2, label1, label2,
                 color1, color2, ylabel, title):
    x_pos = np.arange(len(REGIME_ORDER))
    w = 0.30
    for j, (col, color) in enumerate([(col1, color1), (col2, color2)]):
        if col not in gc.columns:
            continue
        vals = [gc[gc['drizzle_regime'] == r][col].dropna().values
                for r in REGIME_ORDER]
        bp = ax.boxplot(
            vals, positions=x_pos + (j - 0.5) * w * 1.15,
            widths=w, patch_artist=True,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='#555555'), capprops=dict(color='#555555'),
            flierprops=dict(marker='o', markersize=4,
                            markerfacecolor=color, alpha=0.55),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.80)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
    ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=TXT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        [Patch(facecolor=color1), Patch(facecolor=color2)],
        [label1, label2],
        fontsize=8.5, framealpha=0.9, edgecolor='#CCCCCC', facecolor='white',
    )


def scatter_by_regime(ax, gc, xcol, ycol, xlabel, ylabel, title):
    for reg in REGIME_ORDER:
        sub = gc[gc['drizzle_regime'] == reg]
        ax.scatter(sub[xcol], sub[ycol],
                   c=REGIME_COLORS[reg], s=65,
                   edgecolors='white', linewidths=0.5,
                   label=reg, zorder=3)
    ax.set_xlabel(xlabel, color=TXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=TXT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='#CCCCCC', facecolor='white')


# ============================================================
# 3. MAIN FIGURE — 6 panels
# ============================================================
np.random.seed(42)

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

single_boxplot(axes[0], gc, 'Nd_median',
               'Nd  (cm⁻³)', 'A1 · Cloud Droplet Number Concentration')

dual_boxplot(axes[1], gc, 're_cas_median', 're_full_median',
             'r$_e$ CAS (µm)', 'r$_e$ Full (µm)',
             '#1565C0', '#C62828',
             'Effective Radius  (µm)', 'A2 · Effective Radius (CAS vs Full)')

dual_boxplot(axes[2], gc, 'tau_main', 'tau_full',
             'τ main', 'τ full',
             '#2E7D32', '#6A1B9A',
             'Optical Depth  (τ)', 'A3 · Optical Depth (main vs full)')

single_boxplot(axes[3], gc, 'f_ad_mean',
               'Adiabatic Fraction  (f$_{ad}$)', 'A4 · Cloud Adiabaticity',
               hline=0.8, hline_label='f$_{ad}$ = 0.8')

dual_boxplot(axes[4], gc, 'min_z_drizzle', 'mean_z_drizzle',
             'min z$_{drizzle}$', 'mean z$_{drizzle}$',
             '#00838F', '#BF360C',
             'Normalised altitude', 'A5 · Drizzle Penetration Depth')

single_boxplot(axes[5], gc, 'cloud_depth',
               'Cloud Depth  (m)', 'A6 · Cloud Geometrical Thickness')

fig.suptitle(
    f'VOCALS-REx 2008 — Package A: In-Situ Cloud Physics\n'
    f'Do Drizzle Regimes Alter the Physical Structure of Clouds?',
    color=TXT, fontsize=14, fontweight='bold', y=1.02,
)
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Main figure  → {OUT_MAIN}")

# ============================================================
# 4. BONUS SCATTERS — 3 panels
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.patch.set_facecolor('white')

scatter_by_regime(axes2[0], gc, 're_cas_median', 'Nd_median',
                  'r$_e$ CAS  (µm)', 'Nd  (cm⁻³)', 'B1 · Nd vs Effective Radius')

scatter_by_regime(axes2[1], gc, 'LWP_insitu', 'tau_main',
                  'In-situ LWP  (g m⁻²)', 'τ main', 'B2 · Optical Depth vs LWP')

scatter_by_regime(axes2[2], gc, 'drizzle_fraction', 'f_ad_mean',
                  'Drizzle Fraction', 'f$_{ad}$ mean',
                  'B3 · Adiabaticity vs Drizzle Fraction')

fig2.suptitle(
    'VOCALS-REx 2008 — Package A: Bonus — Physical Relationships  (colour = drizzle regime)',
    color=TXT, fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_BONUS, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Bonus figure → {OUT_BONUS}")

# ============================================================
# 5. STATISTICS TABLE
# ============================================================
STAT_COLS = [
    ('Nd_median',        'Nd (cm⁻³)'),
    ('re_cas_median',    're CAS (µm)'),
    ('re_full_median',   're Full (µm)'),
    ('tau_main',         'tau main'),
    ('tau_full',         'tau full'),
    ('f_ad_mean',        'f_ad mean'),
    ('drizzle_fraction', 'drizzle fraction'),
    ('min_z_drizzle',    'min z drizzle (norm.)'),
    ('mean_z_drizzle',   'mean z drizzle (norm.)'),
    ('cloud_depth',      'cloud depth (m)'),
    ('LWP_insitu',       'LWP in-situ (g/m²)'),
]

print("\n" + "=" * 72)
print(f"{'PACKAGE A — STATISTICS TABLE':^72}")
print("=" * 72)

for col, label in STAT_COLS:
    if col not in gc.columns:
        print(f"\n{label}: COLUMN NOT FOUND")
        continue
    print(f"\n{label}")
    groups = []
    for reg in REGIME_ORDER:
        vals = gc[gc['drizzle_regime'] == reg][col].dropna()
        groups.append(vals)
        if len(vals) == 0:
            print(f"  {reg:22s}: —")
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {reg:22s}: {med:8.2f}  [{q1:.2f} – {q3:.2f}]  (n={len(vals)})")
    print(f"  → {kw_label(groups)}")

# ============================================================
# 6+7. BIMODAL ANALYSIS  +  NOTES FILE
#      In some campaigns most profiles fall into a single regime,
#      making bimodal analysis unreliable. Wrapped in try/except;
#      skipped silently on failure.
# ============================================================
N_MOD = (gc['drizzle_regime'] == 'Moderate drizzling').sum()
if N_MOD < 4:
    print(f'\n[INFO] Bimodal analysis skipped — Moderate drizzling: {N_MOD} profiles (<4)')
else:
    try:
        print("\n" + "=" * 65)
        print("BIMODAL ANALYSIS — MODERATE DRIZZLING")
        print("=" * 65)

        ND_THRESHOLD = 60
        mod_mask = gc['drizzle_regime'] == 'Moderate drizzling'
        gc['subgroup'] = 'Other'
        gc.loc[mod_mask & (gc['Nd_median'] <  ND_THRESHOLD), 'subgroup'] = 'Mod-Low  (<60)'
        gc.loc[mod_mask & (gc['Nd_median'] >= ND_THRESHOLD), 'subgroup'] = 'Mod-High (≥60)'

        SUB_COLORS = {
            'Other'         : '#AAAAAA',
            'Mod-Low  (<60)': '#1565C0',
            'Mod-High (≥60)': '#AD1457',
        }

        mod_df  = gc[mod_mask].copy()
        low_df  = mod_df[mod_df['subgroup'] == 'Mod-Low  (<60)']
        high_df = mod_df[mod_df['subgroup'] == 'Mod-High (≥60)']

        cv_stats = {}
        for reg in REGIME_ORDER:
            vals = gc[gc['drizzle_regime'] == reg]['Nd_median'].dropna()
            cv_stats[reg] = {
                'n': len(vals), 'median': np.median(vals),
                'mean': np.mean(vals), 'std': np.std(vals),
                'cv': np.std(vals) / np.mean(vals) * 100 if np.mean(vals) > 0 else np.nan,
                'q1': np.percentile(vals, 25), 'q3': np.percentile(vals, 75),
            }

        r_p, p_p = pearsonr(mod_df['Nd_median'], mod_df['re_cas_median'])
        r_s, p_s = spearmanr(mod_df['Nd_median'], mod_df['re_cas_median'])

        nd_mod_vals = mod_df['Nd_median'].dropna().values
        kde_obj     = gaussian_kde(nd_mod_vals, bw_method=0.4)
        x_kde       = np.linspace(0, 300, 1000)
        y_kde       = kde_obj(x_kde)
        peaks_idx   = argrelextrema(y_kde, np.greater, order=30)[0]
        peak_positions = x_kde[peaks_idx]

        print(f"\nCV by regime:")
        for reg in REGIME_ORDER:
            s = cv_stats[reg]
            print(f"  {reg:22s}: median={s['median']:6.1f}  CV={s['cv']:5.1f}%  (n={s['n']})")

        print(f"\nModerate subgroups (threshold={ND_THRESHOLD} cm⁻³):")
        print(f"  Low  (<60) : n={len(low_df)}, Nd={low_df['Nd_median'].mean():.1f}, "
              f"re={low_df['re_cas_median'].mean():.2f} µm, "
              f"f_ad={low_df['f_ad_mean'].mean():.3f}, "
              f"LWP={low_df['LWP_insitu'].mean():.1f} g/m²")
        print(f"  High (≥60) : n={len(high_df)}, Nd={high_df['Nd_median'].mean():.1f}, "
              f"re={high_df['re_cas_median'].mean():.2f} µm, "
              f"f_ad={high_df['f_ad_mean'].mean():.3f}, "
              f"LWP={high_df['LWP_insitu'].mean():.1f} g/m²")

        # ── Bimodal figure — 4 panels ────────────────────────────────
        fig3 = plt.figure(figsize=(18, 13))
        fig3.patch.set_facecolor('white')
        gs3  = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.48, wspace=0.38)

        # Panel 1 — KDE
        ax1 = fig3.add_subplot(gs3[0, 0])
        x_range = np.linspace(0, 310, 1000)
        for reg in REGIME_ORDER:
            vals = gc[gc['drizzle_regime'] == reg]['Nd_median'].dropna().values
            if len(vals) < 3:
                continue
            kde = gaussian_kde(vals, bw_method=0.45)
            y   = kde(x_range)
            lw  = 3.0 if reg == 'Moderate drizzling' else 1.8
            ax1.plot(x_range, y, color=REGIME_COLORS[reg], lw=lw,
                     label=f"{reg} (n={len(vals)})",
                     zorder=4 if reg == 'Moderate drizzling' else 3)
            ax1.fill_between(x_range, y,
                             alpha=0.18 if reg == 'Moderate drizzling' else 0.08,
                             color=REGIME_COLORS[reg])
            ax1.axvline(np.median(vals), color=REGIME_COLORS[reg], lw=1.2, ls='--', alpha=0.7)

        ax1.axvspan(0, ND_THRESHOLD, alpha=0.06, color='#1565C0',
                    label=f'Low mode (<{ND_THRESHOLD})')
        ax1.axvspan(ND_THRESHOLD, 310, alpha=0.06, color='#AD1457',
                    label=f'High mode (≥{ND_THRESHOLD})')
        ax1.axvline(ND_THRESHOLD, color='#555555', lw=1.2, ls=':', alpha=0.8)
        ax1.set_xlabel('Nd  (cm⁻³)', color=TXT, fontsize=10)
        ax1.set_ylabel('Density', color=TXT, fontsize=10)
        ax1.set_title('P1 · Nd KDE by Drizzle Regime\n(dashed = median; shading = bimodal zones)',
                      color=TXT, fontsize=10, fontweight='bold', pad=8)
        ax1.legend(fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
        ax1.set_axisbelow(True)

        # Panel 2 — Strip plot
        ax2 = fig3.add_subplot(gs3[0, 1])
        x_pos = {r: i for i, r in enumerate(REGIME_ORDER)}
        for reg in REGIME_ORDER:
            sub = gc[gc['drizzle_regime'] == reg]
            for _, row in sub.iterrows():
                sg    = row['subgroup']
                color = SUB_COLORS[sg]
                size  = 90 if sg != 'Other' else 55
                jit   = np.random.uniform(-0.18, 0.18)
                ax2.scatter(x_pos[reg] + jit, row['Nd_median'],
                            c=color, s=size, zorder=5 if sg != 'Other' else 3,
                            edgecolors='white', linewidths=0.6, alpha=0.90)
            med = sub['Nd_median'].median()
            ax2.plot([x_pos[reg]-0.35, x_pos[reg]+0.35], [med, med],
                     color=REGIME_COLORS[reg], lw=2.5, zorder=6)
            ax2.text(x_pos[reg]+0.38, med, f'{med:.0f}',
                     va='center', ha='left', color=TXT, fontsize=8, fontweight='bold')

        ax2.set_xticks(range(4))
        ax2.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
        ax2.set_ylabel('Nd  (cm⁻³)', color=TXT, fontsize=10)
        ax2.set_title('P2 · Nd Strip Plot\n(Moderate sub-groups: blue=Low, pink=High)',
                      color=TXT, fontsize=10, fontweight='bold', pad=8)
        ax2.legend(handles=[
            Patch(facecolor=SUB_COLORS['Mod-Low  (<60)'],  label=f'Mod-Low  (<{ND_THRESHOLD})'),
            Patch(facecolor=SUB_COLORS['Mod-High (≥60)'],  label=f'Mod-High (≥{ND_THRESHOLD})'),
            Patch(facecolor=SUB_COLORS['Other'],            label='Other regimes'),
        ], fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
        ax2.set_axisbelow(True)

        # Panel 3 — Moderate: Nd vs re
        ax3 = fig3.add_subplot(gs3[1, 0])
        for sg, color in SUB_COLORS.items():
            if sg == 'Other':
                continue
            sub = mod_df[mod_df['subgroup'] == sg]
            ax3.scatter(sub['re_cas_median'], sub['Nd_median'],
                        c=color, s=100, zorder=4,
                        edgecolors='white', linewidths=0.6, label=sg, alpha=0.92)
            for _, row in sub.iterrows():
                ax3.annotate(row['cloud_id'], (row['re_cas_median'], row['Nd_median']),
                             fontsize=7.2, color=TXT, alpha=0.85,
                             xytext=(5, 4), textcoords='offset points')

        x_fit  = mod_df['re_cas_median'].values
        y_fit  = mod_df['Nd_median'].values
        m, b   = np.polyfit(x_fit, y_fit, 1)
        x_line = np.linspace(x_fit.min()-0.3, x_fit.max()+0.3, 100)
        ax3.plot(x_line, m*x_line + b, 'k--', lw=1.3, alpha=0.5,
                 label=f'Linear fit  (r={r_p:.2f}, p={p_p:.3f})')
        ax3.set_xlabel('r$_e$ CAS  (µm)', color=TXT, fontsize=10)
        ax3.set_ylabel('Nd  (cm⁻³)', color=TXT, fontsize=10)
        ax3.set_title('P3 · Moderate Drizzling Only: Nd vs r$_e$',
                      color=TXT, fontsize=10, fontweight='bold', pad=8)
        ax3.legend(fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
        ax3.spines[['top', 'right']].set_visible(False)
        ax3.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
        ax3.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
        ax3.set_axisbelow(True)

        # Panel 4 — CV bar chart
        ax4 = fig3.add_subplot(gs3[1, 1])
        cv_vals    = [cv_stats[r]['cv'] for r in REGIME_ORDER]
        n_vals     = [cv_stats[r]['n']  for r in REGIME_ORDER]
        bar_colors = [REGIME_COLORS[r]  for r in REGIME_ORDER]

        ax4.bar(range(4), cv_vals, color=bar_colors,
                edgecolor='white', linewidth=0.8, alpha=0.85, width=0.55)
        for i, (v, n) in enumerate(zip(cv_vals, n_vals)):
            ax4.text(i, v + 1.5, f'{v:.0f}%\n(n={n})',
                     ha='center', va='bottom', color=TXT,
                     fontsize=9, fontweight='bold')

        mod_idx = REGIME_ORDER.index('Moderate drizzling')
        ax4.annotate('⚠ Bimodal\ndistribution',
                     xy=(mod_idx, cv_vals[mod_idx]),
                     xytext=(mod_idx + 0.6, cv_vals[mod_idx] + 8),
                     fontsize=8.5, color='#E65100', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                               edgecolor='#E65100', alpha=0.9))

        ax4.set_xticks(range(4))
        ax4.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
        ax4.set_ylabel('Coefficient of Variation  (%)', color=TXT, fontsize=10)
        ax4.set_title('P4 · Within-Regime Nd Variability (CV)',
                      color=TXT, fontsize=10, fontweight='bold', pad=8)
        ax4.spines[['top', 'right']].set_visible(False)
        ax4.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
        ax4.set_axisbelow(True)

        fig3.suptitle(
            'VOCALS-REx 2008 — Package A: Bimodal Analysis — Moderate Drizzling Nd Distribution',
            color=TXT, fontsize=14, fontweight='bold', y=1.02,
        )
        plt.savefig(OUT_BIMODAL, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Bimodal fig  → {OUT_BIMODAL}")

        # ============================================================
        # 7. NOTES FILE
        # ============================================================
        notes = f"""# VOCALS-REx 2008 — Bimodal Nd in Moderate Drizzling Regime

        ## CV by Regime
        | Regime              | n  | Median Nd | CV (%) |
        |---------------------|----|-----------|--------|
        | Non-drizzling       | {cv_stats['Non-drizzling']['n']}  | {cv_stats['Non-drizzling']['median']:.1f}      | {cv_stats['Non-drizzling']['cv']:.0f}     |
        | Weak drizzling      | {cv_stats['Weak drizzling']['n']}  | {cv_stats['Weak drizzling']['median']:.1f}      | {cv_stats['Weak drizzling']['cv']:.0f}     |
        | Moderate drizzling  | {cv_stats['Moderate drizzling']['n']}  | {cv_stats['Moderate drizzling']['median']:.1f}       | {cv_stats['Moderate drizzling']['cv']:.0f}     |
        | Heavy drizzling     | {cv_stats['Heavy drizzling']['n']}  | {cv_stats['Heavy drizzling']['median']:.1f}      | {cv_stats['Heavy drizzling']['cv']:.0f}     |

        ## Subgroups (threshold = {ND_THRESHOLD} cm⁻³)
        | Sub-group       | n | Nd mean | re_cas | f_ad | LWP |
        |-----------------|---|---------|--------|------|-----|
        | Mod-Low (<{ND_THRESHOLD})  | {len(low_df)} | {low_df['Nd_median'].mean():.1f} | {low_df['re_cas_median'].mean():.2f} µm | {low_df['f_ad_mean'].mean():.3f} | {low_df['LWP_insitu'].mean():.1f} g/m² |
        | Mod-High (≥{ND_THRESHOLD}) | {len(high_df)} | {high_df['Nd_median'].mean():.1f} | {high_df['re_cas_median'].mean():.2f} µm | {high_df['f_ad_mean'].mean():.3f} | {high_df['LWP_insitu'].mean():.1f} g/m² |

        ## Nd vs re_cas Correlation (Moderate only)
        - Pearson  r = {r_p:.3f}  (p = {p_p:.3f})
        - Spearman r = {r_s:.3f}  (p = {p_s:.3f})
        """

        with open(OUT_NOTES, 'w', encoding='utf-8') as f:
            f.write(notes)
        print(f"Notes        → {OUT_NOTES}")
    except Exception as _bimodal_err:
        print(f'\n[WARN] Error in Bimodal/Notes block: {_bimodal_err}')
        print('       Section skipped; main figures were created.')

print("\n" + "=" * 65)
print(f"✓  Package A complete  [{NAME}]")
print(f"   {OUT_MAIN}")
print(f"   {OUT_BONUS}")
if "OUT_BIMODAL" in dir(): print(f"   {OUT_BIMODAL}")
if "OUT_NOTES" in dir(): print(f"   {OUT_NOTES}")
print("=" * 65)
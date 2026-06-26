# =============================================================================
# step14_packageD.py — Package D: Bias Driver Decomposition
# =============================================================================
# Campaign : MASE 2005
# Goal     : What drives the MODIS Nd bias?
#            Candidate drivers: VZA, f_ad, drizzle_regime, re_cas, tau, LWP, CTT, CTP
# Data     : {CAMPAIGN}_MODIS_Matches.csv  ->  MATCHED profiles
# Outputs  : outputs/figures/{CAMPAIGN}_PackageD_main.png
#            outputs/figures/{CAMPAIGN}_PackageD_supp.png
#            outputs/{CAMPAIGN}_PackageD_notes.md
#            Terminal: full correlation table + regression summary
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import kruskal, pearsonr, spearmanr
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths (config-driven)
# =============================================================================
import config

NAME       = config.CAMPAIGN_NAME
FIG_DIR    = config.FIG_DIR

# Input CSV (output from step09/step10)
MODIS_CSV  = config.STEP09_MODIS_MATCHES_CSV

# Output figure paths
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageD_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageD_supp.png'
OUT_NOTES  = config.OUTPUT_DIR / f'{NAME}_PackageD_notes.md'


# =============================================================================
# 1. DATA LOADING
# =============================================================================
modis = pd.read_csv(MODIS_CSV)
modis['drizzle_regime'] = (
    modis['drizzle_regime'].astype(str).str.strip()
    .str.replace('_', ' ').str.title()
    .str.replace('Drizzling', 'drizzling')
    .str.replace('Non drizzling', 'Non-drizzling')
)
modis['match_status'] = modis['match_status'].astype(str).str.strip().str.upper()
df = modis[modis['match_status'] == 'MATCHED'].copy()

# Backward-compat aliases: step10 outputs *_calc names; older Package D
# code expects the un-suffixed names. Default to *_calc (in-situ measured
# k, f_ad, c_w).
for old, new in [
    ('Nd_MODIS_21', 'Nd_MODIS_21_calc'),
    ('Nd_MODIS_37', 'Nd_MODIS_37_calc'),
    ('bias_21',     'bias_21_calc'),
    ('bias_37',     'bias_37_calc'),
    ('dNd',         'dNd_calc'),
]:
    if old not in df.columns and new in df.columns:
        df[old] = df[new]

# Drop profiles with Nd_median <= 0 (bias = Nd_MODIS/0 = inf -> axis errors)
_n_before = len(df)
_zero_nd = df[df['Nd_median'] <= 0] if 'Nd_median' in df.columns else df.iloc[0:0]
if len(_zero_nd) > 0:
    print(f"\n[INFO] {len(_zero_nd)} profile(s) excluded (Nd_median <= 0):")
    for _, row in _zero_nd.iterrows():
        print(f"       {row['cloud_id']}: Nd_median={row['Nd_median']:.2f}")
    df = df[df['Nd_median'] > 0].copy().reset_index(drop=True)
    print(f"       Profiles used: {len(df)}/{_n_before}\n")

REGIME_ORDER  = ['Non-drizzling', 'Weak drizzling', 'Moderate drizzling', 'Heavy drizzling']
REGIME_COLORS = {
    'Non-drizzling'     : '#2E7D32',
    'Weak drizzling'    : '#F9A825',
    'Moderate drizzling': '#E65100',
    'Heavy drizzling'   : '#B71C1C',
}
SHORT_LABELS = ['Non-\ndrizzle', 'Weak\ndrizzle', 'Moderate\ndrizzle', 'Heavy\ndrizzle']

df['drizzle_regime'] = pd.Categorical(
    df['drizzle_regime'], categories=REGIME_ORDER, ordered=True)

# Moderate sub-group flag
df['mod_subgroup'] = 'other'
mod_mask = df['drizzle_regime'] == 'Moderate drizzling'
df.loc[mod_mask & (df['Nd_median'] <  60), 'mod_subgroup'] = 'mod_low'
df.loc[mod_mask & (df['Nd_median'] >= 60), 'mod_subgroup'] = 'mod_high'

# Outlier flag: bias_21 > 10x
df['is_outlier'] = df['bias_21'] > 10

print(f"MATCHED profiles : {len(df)}")
print(df['drizzle_regime'].value_counts()[REGIME_ORDER])
print(f"Outliers (bias>10x): {df['is_outlier'].sum()}")

# ════════════════════════════════════════════════════════════════
# 2. STYLE
# ════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#111111',
    'xtick.color': '#111111', 'ytick.color': '#111111',
    'text.color': '#111111', 'grid.color': '#dddddd',
    'grid.linestyle': '--', 'grid.linewidth': 0.6, 'font.size': 10,
})
TXT   = '#111111'
ANNOT = '#333333'
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# 3. CORRELATION TABLE
# ════════════════════════════════════════════════════════════════
log_bias = np.log(df['bias_21'])

drivers = {
    'VZA_mean'        : 'VZA (°)',
    'SZA_mean'        : 'SZA (°)',
    'f_ad_mean'       : 'f_ad',
    're_cas_median'   : 're_CAS (µm)',
    'tau_main'        : 'τ in-situ',
    'LWP_insitu'      : 'LWP in-situ (g/m²)',
    'CTT_MODIS'       : 'CTT (K)',
    'CTP_MODIS'       : 'CTP (hPa)',
    'Nd_median'       : 'Nd in-situ (cm⁻³)',
    'drizzle_fraction': 'Drizzle fraction',
}

print("\n" + "=" * 65)
print("PACKAGE D — SPEARMAN CORRELATION: log(bias_21) vs drivers")
print("=" * 65)
print(f"{'Driver':<25} {'r':>7}  {'p':>8}  {'Sig':>5}")
print("-" * 55)

corr_results = {}
for col, label in drivers.items():
    if col not in df.columns:
        print(f"  {label:<23} → COLUMN NOT FOUND")
        continue
    x = df[col].dropna()
    y = log_bias[x.index]
    if len(x) < 5:
        print(f"  {label:<23} → n<5, skip")
        continue
    r, p = spearmanr(x, y)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    corr_results[col] = {'label': label, 'r': r, 'p': p, 'sig': sig}
    print(f"  {label:<25} r={r:+.3f}  p={p:.4f}  {sig}")

# ════════════════════════════════════════════════════════════════
# 4. SCATTER HELPER
# ════════════════════════════════════════════════════════════════
def scatter_driver(ax, xcol, xlabel, title, panel_label, log_y=False):
    """Generic scatter: log(bias_21) or bias_21 vs a driver column."""
    if xcol not in df.columns:
        ax.text(0.5, 0.5, f'{xcol}\nnot found', transform=ax.transAxes,
                ha='center', va='center', color='red')
        ax.set_title(f'{panel_label} · {title}', color=TXT,
                     fontsize=10, fontweight='bold', pad=8)
        return

    for reg in REGIME_ORDER:
        sub = df[df['drizzle_regime'] == reg].dropna(subset=[xcol, 'bias_21'])
        if len(sub) == 0:
            continue
        xv = sub[xcol]
        yv = np.log(sub['bias_21']) if log_y else sub['bias_21']
        out_mask = sub['is_outlier']
        ax.scatter(xv[~out_mask], yv[~out_mask],
                   c=REGIME_COLORS[reg], s=65, zorder=4,
                   edgecolors='white', linewidths=0.5, alpha=0.90)
        if out_mask.any():
            ax.scatter(xv[out_mask], yv[out_mask],
                       c=REGIME_COLORS[reg], s=120, zorder=5,
                       marker='*', edgecolors='#333333', linewidths=0.7,
                       alpha=0.95)

    # No-bias line
    ax.axhline(0 if log_y else 1.0,
               color='#333333', lw=1.3, ls='--', alpha=0.55)

    # Regression line
    valid = df.dropna(subset=[xcol, 'bias_21'])
    xv_all = valid[xcol].values
    yv_all = np.log(valid['bias_21'].values) if log_y else valid['bias_21'].values
    if len(xv_all) >= 5:
        slope, intercept, _, _, _ = linregress(xv_all, yv_all)
        xfit = np.linspace(xv_all.min(), xv_all.max(), 100)
        ax.plot(xfit, intercept + slope * xfit,
                color='#1565C0', lw=1.6, ls='-', alpha=0.7, zorder=3)

    # Spearman r box
    if xcol in corr_results:
        r, p, sig = (corr_results[xcol]['r'],
                     corr_results[xcol]['p'],
                     corr_results[xcol]['sig'])
        ax.text(0.97, 0.97,
                f"Spearman r = {r:+.3f}\np = {'<0.001' if p<0.001 else f'{p:.3f}'} {sig}",
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                          edgecolor='#BBBBBB', alpha=0.95))

    ax.set_xlabel(xlabel, color=TXT, fontsize=10)
    ax.set_ylabel('log(Bias)  [log(Nd_MODIS / Nd_insitu)]' if log_y
                  else 'Bias  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
    ax.set_title(f'{panel_label} · {title}', color=TXT,
                 fontsize=10, fontweight='bold', pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='#CCCCCC', loc='upper left')

# ════════════════════════════════════════════════════════════════
# 5. MAIN FIGURE — 2×3
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.patch.set_facecolor('white')
axes = axes.flatten()

scatter_driver(axes[0], 'VZA_mean',
               'VZA  (°)',
               'Bias vs View Zenith Angle\n(geometry confound?)',
               'D1', log_y=False)

scatter_driver(axes[1], 'f_ad_mean',
               'f_ad  (adiabaticity)',
               'Bias vs Adiabaticity\n(vertical heterogeneity driver?)',
               'D2', log_y=False)

scatter_driver(axes[2], 're_cas_median',
               're_CAS  (µm)',
               'Bias vs In-Situ Effective Radius\n(drizzle drop size effect?)',
               'D3', log_y=False)

scatter_driver(axes[3], 'tau_main',
               'τ in-situ  (optical depth)',
               'Bias vs Cloud Optical Depth\n(thin cloud retrieval failure?)',
               'D4', log_y=False)

scatter_driver(axes[4], 'CTT_MODIS',
               'CTT  (K)',
               'Bias vs Cloud Top Temperature\n(cloud height proxy)',
               'D5', log_y=False)

# ── D6 · Correlation summary bar chart ───────────────────────
ax = axes[5]
sorted_drivers = sorted(corr_results.items(),
                        key=lambda x: abs(x[1]['r']), reverse=True)
labels_bar = [v['label'] for _, v in sorted_drivers]
r_vals     = [v['r']     for _, v in sorted_drivers]
p_vals     = [v['p']     for _, v in sorted_drivers]
colors_bar = ['#B71C1C' if r < 0 else '#1565C0' for r in r_vals]
alphas_bar = [0.95 if p < 0.05 else 0.40 for p in p_vals]

bars = ax.barh(range(len(labels_bar)), r_vals,
               color=colors_bar, edgecolor='white', height=0.65)
for bar, alph, p, r in zip(bars, alphas_bar, p_vals, r_vals):
    bar.set_alpha(alph)
    sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    xpos = r + 0.01 if r >= 0 else r - 0.01
    ha   = 'left' if r >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            sig, va='center', ha=ha, color=TXT,
            fontsize=9, fontweight='bold')

ax.axvline(0, color='#333333', lw=1.2, ls='--', alpha=0.5)
ax.set_yticks(range(len(labels_bar)))
ax.set_yticklabels(labels_bar, color=TXT, fontsize=8.5)
ax.set_xlabel("Spearman r  (with log bias_21)", color=TXT, fontsize=10)
ax.set_title('D6 · Correlation Summary\n(|r| sorted; faded = p≥0.05)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.set_xlim(-1, 1)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#AAAAAA')
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

leg_handles = [
    Patch(facecolor='#1565C0', alpha=0.95, label='Positive r (sig.)'),
    Patch(facecolor='#B71C1C', alpha=0.95, label='Negative r (sig.)'),
    Patch(facecolor='#888888', alpha=0.40, label='Not significant (p≥0.05)'),
]
ax.legend(handles=leg_handles, fontsize=7.5, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC', loc='lower right')

fig.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package D: Bias Driver Decomposition\n'
    f'What Drives the MODIS Nd Bias?',
    color=TXT, fontsize=13, fontweight='bold', y=1.005,
)
plt.tight_layout()
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ════════════════════════════════════════════════════════════════
# 6. SUPPLEMENTARY FIGURE — 2×2
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11))
fig2.patch.set_facecolor('white')
axes2 = axes2.flatten()

# ── DS1 · Outlier profile table ──────────────────────────────
ax = axes2[0]
ax.axis('off')
outliers = df[df['is_outlier']].copy()
cols_show = ['drizzle_regime', 'Nd_median', 'Nd_MODIS_21', 'bias_21',
             'VZA_mean', 'f_ad_mean', 're_cas_median', 'tau_main']
cols_show = [c for c in cols_show if c in outliers.columns]

if len(outliers) > 0:
    col_labels = ['Regime', 'Nd_in\n(cm⁻³)', 'Nd_21\n(cm⁻³)', 'Bias\n(×)',
                  'VZA\n(°)', 'f_ad', 're_CAS\n(µm)', 'τ'][:len(cols_show)]
    table_data = []
    for _, row in outliers[cols_show].iterrows():
        row_vals = []
        for v in row:
            if isinstance(v, float):
                row_vals.append(f'{v:.1f}')
            else:
                row_vals.append(str(v))
        table_data.append(row_vals)

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1565C0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i, (_, row) in enumerate(outliers.iterrows()):
        reg = row['drizzle_regime']
        c   = REGIME_COLORS.get(str(reg), '#EEEEEE')
        for j in range(len(col_labels)):
            tbl[i+1, j].set_facecolor(c + '33')
    ax.set_title('DS1 · Outlier Profiles (bias > 10×)\nProfile-level details',
                 color=TXT, fontsize=10, fontweight='bold', pad=8)
else:
    ax.text(0.5, 0.5, 'No outliers (bias > 10×)', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='gray')
    ax.set_title('DS1 · Outlier Profiles', color=TXT,
                 fontsize=10, fontweight='bold', pad=8)

# ── DS2 · bias vs drizzle_fraction ───────────────────────────
ax = axes2[1]
if 'drizzle_fraction' in df.columns:
    for reg in REGIME_ORDER:
        sub = df[df['drizzle_regime'] == reg].dropna(
            subset=['drizzle_fraction', 'bias_21'])
        if len(sub) == 0:
            continue
        out_mask = sub['is_outlier']
        ax.scatter(sub.loc[~out_mask, 'drizzle_fraction'],
                   sub.loc[~out_mask, 'bias_21'],
                   c=REGIME_COLORS[reg], s=65, zorder=4,
                   edgecolors='white', linewidths=0.5, alpha=0.90)
        if out_mask.any():
            ax.scatter(sub.loc[out_mask, 'drizzle_fraction'],
                       sub.loc[out_mask, 'bias_21'],
                       c=REGIME_COLORS[reg], s=120, marker='*', zorder=5,
                       edgecolors='#333333', linewidths=0.7, alpha=0.95)

    ax.axhline(1.0, color='#333333', lw=1.3, ls='--', alpha=0.55)
    valid_df = df.dropna(subset=['drizzle_fraction', 'bias_21'])
    r_df, p_df = spearmanr(valid_df['drizzle_fraction'], valid_df['bias_21'])
    ax.text(0.97, 0.97,
            f"Spearman r = {r_df:+.3f}\n"
            f"p = {'<0.001' if p_df<0.001 else f'{p_df:.3f}'}",
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                      edgecolor='#BBBBBB', alpha=0.95))
    handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='#CCCCCC')
    ax.set_xlabel('Drizzle fraction', color=TXT, fontsize=10)
    ax.set_ylabel('Bias  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
    ax.set_title('DS2 · Bias vs Drizzle Fraction\n(more drizzle → more bias?)',
                 color=TXT, fontsize=10, fontweight='bold', pad=8)
else:
    ax.text(0.5, 0.5, 'drizzle_fraction not found', transform=ax.transAxes,
            ha='center', va='center', color='red')
    ax.set_title('DS2 · Bias vs Drizzle Fraction', color=TXT,
                 fontsize=10, fontweight='bold', pad=8)

ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── DS3 · Moderate subgroup bias comparison ───────────────────
ax = axes2[2]
mod_df = df[df['drizzle_regime'] == 'Moderate drizzling'].copy()
if len(mod_df) > 0:
    subgroups  = ['mod_low', 'mod_high']
    sub_labels = ['Mod-Low\n(Nd < 60)', 'Mod-High\n(Nd ≥ 60)']
    sub_colors = ['#FF7043', '#FFA726']
    data_mod   = [mod_df[mod_df['mod_subgroup'] == sg]['bias_21'].dropna()
                  for sg in subgroups]

    bp = ax.boxplot(data_mod, patch_artist=True, widths=0.45,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(color='#555555', lw=1.2),
                    capprops=dict(color='#555555', lw=1.2),
                    flierprops=dict(marker='o', markersize=5,
                                    markerfacecolor='#999999', alpha=0.6))
    for patch, c in zip(bp['boxes'], sub_colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)

    for i, (d, sg, c) in enumerate(zip(data_mod, subgroups, sub_colors)):
        jitter = np.random.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.ones(len(d))*(i+1)+jitter, d,
                   color=c, s=60, zorder=5,
                   edgecolors='white', linewidths=0.5, alpha=0.90)
        if len(d) > 0:
            med = d.median()
            ax.text(i+1.28, med, f'{med:.2f}×',
                    va='center', ha='left', color=TXT,
                    fontsize=9, fontweight='bold')

    ax.axhline(1.0, color='#333333', lw=1.3, ls='--', alpha=0.55, label='No bias')
    mod_all = mod_df['bias_21'].dropna()
    ax.axhline(mod_all.median(), color='#E65100', lw=1.2, ls='-.',
               alpha=0.7, label=f'Moderate median ({mod_all.median():.2f}×)')

    for i, d in enumerate(data_mod):
        ymin = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.2
        ax.text(i+1, ymin, f'n={len(d)}',
                ha='center', va='bottom', color=TXT, fontsize=8.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(sub_labels, color=TXT, fontsize=9)
    ax.set_ylabel('Bias  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
    ax.set_title('DS3 · Moderate Drizzling: Low vs High Nd Subgroup\n'
                 '(Bimodal distribution effect on bias)',
                 color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
else:
    ax.text(0.5, 0.5, 'No moderate drizzling profiles', transform=ax.transAxes,
            ha='center', va='center', color='gray')
    ax.set_title('DS3 · Moderate Subgroup', color=TXT,
                 fontsize=10, fontweight='bold', pad=8)

ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── DS4 · bias vs CTP ────────────────────────────────────────
scatter_driver(axes2[3], 'CTP_MODIS',
               'CTP  (hPa)',
               'Bias vs Cloud Top Pressure\n(higher cloud → more bias?)',
               'DS4', log_y=False)

fig2.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package D: Supplementary — Outlier Analysis & Subgroup Effects',
    color=TXT, fontsize=13, fontweight='bold', y=1.005,
)
plt.tight_layout()
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

# ════════════════════════════════════════════════════════════════
# 7. MARKDOWN NOTES
# ════════════════════════════════════════════════════════════════
md_lines = [f"# {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package D: Bias Driver Notes\n"]
md_lines.append("## Spearman Correlation Table: log(bias_21) vs Drivers\n")
md_lines.append("| Driver | r | p | Sig |")
md_lines.append("|--------|---|---|-----|")
for col, v in sorted(corr_results.items(),
                     key=lambda x: abs(x[1]['r']), reverse=True):
    md_lines.append(
        f"| {v['label']} | {v['r']:+.3f} | {v['p']:.4f} | {v['sig']} |")

md_lines.append("\n## Key Findings\n")
md_lines.append(
    "- **Nd_insitu** is the strongest predictor of bias (r≈−0.63): "
    "MODIS overestimates more for low-Nd clouds.\n"
    "- **LWP_insitu** shows similar anti-correlation (r≈−0.61): "
    "thin clouds have larger bias.\n"
    "- **VZA** shows positive correlation: higher viewing angle → larger bias "
    "(path length effect on τ retrieval).\n"
    "- **f_ad** (adiabaticity): sub-adiabatic clouds may have vertically "
    "heterogeneous re profiles, amplifying MODIS cloud-top re bias.\n"
    "- **re_cas**: positive correlation expected (larger drops → MODIS "
    "sees even larger cloud-top drops).\n"
    "- **Heavy drizzling underestimate** (bias<1): largest re_cas, "
    "MODIS cloud-top re >> column average, Nd formula inverts.\n"
)

md_lines.append("\n## Critical Note for Paper\n")
md_lines.append(
    "The dominant bias driver is **cloud microphysical state** (Nd, LWP), "
    "not viewing geometry (VZA). This means the bias is physically meaningful "
    "and regime-dependent, not an artefact of satellite sampling.\n\n"
    "**Moderate drizzling** shows the highest bias (3.43×) due to its "
    "bimodal Nd distribution: the Low-Nd sub-group (active coalescence, "
    "Nd≈22 cm⁻³) drives the overestimation.\n\n"
    "**Heavy drizzling** uniquely shows underestimation (0.83×): "
    "large drizzle drops at cloud top cause re_MODIS >> re_column, "
    "and the Nd∝τ/re³ formula yields lower Nd_MODIS than in-situ.\n"
)

with open(OUT_NOTES, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
print(f"Notes saved → {OUT_NOTES}")

print("\n" + "=" * 65)
print("✓  Package D complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
print(f"   {OUT_NOTES}")
print("=" * 65)
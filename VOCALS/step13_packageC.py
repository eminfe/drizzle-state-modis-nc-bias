# =============================================================================
# step13_packageC.py — Package C: MODIS vs In-Situ Nd Bias Analysis
# =============================================================================
# Campaign : VOCALS-REx 2008
# Goal     : How well does MODIS retrieve Nd compared to in-situ?
#            What is the magnitude and direction of bias by drizzle regime?
# Data     : VOCALS_MODIS_Matches.csv  ->  matched profiles
# Outputs  : outputs/figures/VOCALS_PackageC_main.png
#            outputs/figures/VOCALS_PackageC_supp.png
#            Terminal statistics table
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import kruskal, pearsonr, spearmanr, wilcoxon
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
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageC_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageC_supp.png'


# ════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════
modis = pd.read_csv(MODIS_CSV)

modis['drizzle_regime'] = (
    modis['drizzle_regime'].astype(str).str.strip()
    .str.replace('_', ' ').str.title()
    .str.replace('Drizzling', 'drizzling')
    .str.replace('Non drizzling', 'Non-drizzling')
)
modis['match_status'] = modis['match_status'].astype(str).str.strip().str.upper()
df = modis[modis['match_status'] == 'MATCHED'].copy()

# Backward-compat aliases: step10 outputs bias_21_calc / Nd_MODIS_21_calc.
# Older Package C code expects bias_21 / Nd_MODIS_21 names.
# Default to *_calc (in-situ measured k, f_ad, c_w).
for old, new in [
    ('Nd_MODIS_21', 'Nd_MODIS_21_calc'),
    ('Nd_MODIS_37', 'Nd_MODIS_37_calc'),
    ('bias_21',     'bias_21_calc'),
    ('bias_37',     'bias_37_calc'),
    ('dNd',         'dNd_calc'),
]:
    if old not in df.columns and new in df.columns:
        df[old] = df[new]

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

# Moderate sub-group flag (bimodal annotation)
df['mod_subgroup'] = 'other'
mod_mask = df['drizzle_regime'] == 'Moderate drizzling'
df.loc[mod_mask & (df['Nd_median'] <  60), 'mod_subgroup'] = 'mod_low'
df.loc[mod_mask & (df['Nd_median'] >= 60), 'mod_subgroup'] = 'mod_high'

print(f"MATCHED profiles : {len(df)}")
print(df['drizzle_regime'].value_counts()[REGIME_ORDER])

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

# ════════════════════════════════════════════════════════════════
# 3. STATISTICS
# ════════════════════════════════════════════════════════════════
np.random.seed(42)

# Drop profiles with Nd_median <= 0 from statistical analysis only.
# log(0) = -inf would break pearsonr/wilcoxon. These profiles still
# appear on scatter plots.
_n_before = len(df)
_zero_nd = df[df['Nd_median'] <= 0]
if len(_zero_nd) > 0:
    print(f"\n[INFO] {len(_zero_nd)} profile(s) excluded from statistics (Nd_median <= 0):")
    for _, row in _zero_nd.iterrows():
        print(f"       {row['cloud_id']}: Nd_median={row['Nd_median']:.2f}, "
              f"Nd_MODIS_21={row['Nd_MODIS_21']:.1f}")
    df_stats = df[df['Nd_median'] > 0].copy()
    print(f"       Profiles used for statistics: {len(df_stats)}/{_n_before}")
else:
    df_stats = df

nd_in  = df_stats['Nd_median'].values
nd_21  = df_stats['Nd_MODIS_21'].values
nd_37  = df_stats['Nd_MODIS_37'].values
b21    = df_stats['bias_21'].values
b37    = df_stats['bias_37'].values

log_b21 = np.log(b21)
log_b37 = np.log(b37)

_, p_wil21 = wilcoxon(nd_21 - nd_in)
_, p_wil37 = wilcoxon(nd_37 - nd_in)
r21, pr21  = pearsonr(np.log(nd_in), np.log(nd_21))
r37, pr37  = pearsonr(np.log(nd_in), np.log(nd_37))

print("\n" + "=" * 65)
print("PACKAGE C — OVERALL STATISTICS")
print("=" * 65)
print(f"\nNd in-situ   : {np.median(nd_in):.1f}  [{np.percentile(nd_in,25):.1f}–{np.percentile(nd_in,75):.1f}]  cm⁻³")
print(f"Nd MODIS 2.1 : {np.median(nd_21):.1f}  [{np.percentile(nd_21,25):.1f}–{np.percentile(nd_21,75):.1f}]  cm⁻³")
print(f"Nd MODIS 3.7 : {np.median(nd_37):.1f}  [{np.percentile(nd_37,25):.1f}–{np.percentile(nd_37,75):.1f}]  cm⁻³")
print(f"\nbias_21 (MODIS/insitu) : median={np.median(b21):.2f}  mean={np.mean(b21):.2f}")
print(f"bias_37 (MODIS/insitu) : median={np.median(b37):.2f}  mean={np.mean(b37):.2f}")
print(f"\nlog_bias_21 : mean={log_b21.mean():.3f}  std={log_b21.std():.3f}")
print(f"log_bias_37 : mean={log_b37.mean():.3f}  std={log_b37.std():.3f}")
print(f"\nWilcoxon (Nd21 vs Nd_in) : p = {p_wil21:.4f}")
print(f"Wilcoxon (Nd37 vs Nd_in) : p = {p_wil37:.4f}")
print(f"\nPearson r (log-log, 2.1µm) : r={r21:.3f}  p={pr21:.4f}")
print(f"Pearson r (log-log, 3.7µm) : r={r37:.3f}  p={pr37:.4f}")

print("\nBias by drizzle regime (2.1 µm):")
for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]['bias_21'].dropna()
    if len(sub) == 0: continue
    print(f"  {reg:22s}: median={sub.median():.2f}  "
          f"[{sub.quantile(0.25):.2f}–{sub.quantile(0.75):.2f}]  (n={len(sub)})")

kw_groups_21 = [df[df['drizzle_regime']==r]['bias_21'].dropna() for r in REGIME_ORDER]
_, p_kw21 = kruskal(*[g for g in kw_groups_21 if len(g)>=2])
print(f"  → K-W p = {p_kw21:.4f}")

print("\nBias by drizzle regime (3.7 µm):")
for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]['bias_37'].dropna()
    if len(sub) == 0: continue
    print(f"  {reg:22s}: median={sub.median():.2f}  "
          f"[{sub.quantile(0.25):.2f}–{sub.quantile(0.75):.2f}]  (n={len(sub)})")

kw_groups_37 = [df[df['drizzle_regime']==r]['bias_37'].dropna() for r in REGIME_ORDER]
_, p_kw37 = kruskal(*[g for g in kw_groups_37 if len(g)>=2])
print(f"  → K-W p = {p_kw37:.4f}")

# ════════════════════════════════════════════════════════════════
# 4. MAIN FIGURE — 2×2
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
fig.patch.set_facecolor('white')
axes = axes.flatten()

# ── C1 · Nd_MODIS_21 vs Nd_insitu scatter ────────────────────
ax = axes[0]
lim_max = max(nd_in.max(), nd_21.max()) * 1.12

for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]
    for _, row in sub.iterrows():
        marker = 'D' if row['mod_subgroup'] == 'mod_low' else \
                 's' if row['mod_subgroup'] == 'mod_high' else 'o'
        ax.scatter(row['Nd_median'], row['Nd_MODIS_21'],
                   c=REGIME_COLORS[reg], s=70, marker=marker,
                   edgecolors='white', linewidths=0.5, zorder=4, alpha=0.90)

ax.plot([0, lim_max], [0, lim_max], 'k--', lw=1.3, alpha=0.45, label='1:1', zorder=2)
ax.plot([0, lim_max], [0, 2*lim_max], color='#999999', lw=0.9,
        ls=':', alpha=0.5, label='2:1 / 0.5:1')
ax.plot([0, lim_max], [0, 0.5*lim_max], color='#999999', lw=0.9, ls=':', alpha=0.5)

mask = (nd_in > 0) & (nd_21 > 0)
slope, intercept, _, _, _ = linregress(np.log10(nd_in[mask]), np.log10(nd_21[mask]))
x_fit = np.logspace(np.log10(nd_in[mask].min()*0.9), np.log10(nd_in[mask].max()*1.1), 100)
y_fit = 10**(intercept + slope * np.log10(x_fit))
ax.plot(x_fit, y_fit, color='#1565C0', lw=1.8, alpha=0.7,
        label=f'Log-log fit (slope={slope:.2f})')

handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
handles += [plt.Line2D([0],[0], color='k', ls='--', lw=1.3, label='1:1'),
            plt.Line2D([0],[0], color='#999999', ls=':', lw=0.9, label='2:1 / 0.5:1'),
            plt.Line2D([0],[0], color='#1565C0', lw=1.8, label=f'Log-log fit')]
ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC', loc='upper left')

ax.text(0.97, 0.05,
        f"n={len(df)}\nMedian bias={np.median(b21):.2f}×\n"
        f"r={r21:.2f}  (log-log)\n"
        f"Wilcoxon p={'<0.001' if p_wil21<0.001 else f'{p_wil21:.3f}'}",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))

ax.set_xlabel('Nd in-situ  (cm⁻³)', color=TXT, fontsize=10)
ax.set_ylabel('Nd MODIS 2.1 µm  (cm⁻³)', color=TXT, fontsize=10)
ax.set_title('C1 · Nd MODIS 2.1 µm vs In-Situ\n(◇ = Mod-Low, □ = Mod-High)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── C2 · bias_21 boxplot by regime ───────────────────────────
ax = axes[1]
data_b21 = [df[df['drizzle_regime']==r]['bias_21'].dropna() for r in REGIME_ORDER]
bp = ax.boxplot(data_b21, patch_artist=True, widths=0.52,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(color='#555555', lw=1.2),
                capprops=dict(color='#555555', lw=1.2),
                flierprops=dict(marker='o', markersize=5,
                                markerfacecolor='#999999', alpha=0.6))
for patch, reg in zip(bp['boxes'], REGIME_ORDER):
    patch.set_facecolor(REGIME_COLORS[reg]); patch.set_alpha(0.82)

for i, (d, reg) in enumerate(zip(data_b21, REGIME_ORDER)):
    jitter = np.random.uniform(-0.18, 0.18, size=len(d))
    ax.scatter(np.ones(len(d))*(i+1)+jitter, d,
               c=REGIME_COLORS[reg], s=45, zorder=5,
               edgecolors='white', linewidths=0.5, alpha=0.85)
    if len(d) > 0:
        med = d.median()
        ax.text(i+1.32, med, f'{med:.2f}×',
                va='center', ha='left', color=TXT,
                fontsize=8, fontweight='bold')

ax.axhline(1.0, color='#333333', lw=1.5, ls='--', alpha=0.6, label='No bias (1:1)')
ax.axhline(np.median(b21), color='#1565C0', lw=1.2, ls='-.',
           alpha=0.6, label=f'Overall median ({np.median(b21):.2f}×)')
ax.set_xticks(range(1,5))
ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
ax.set_ylabel('Bias  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_title('C2 · Nd Bias (2.1 µm) by Drizzle Regime',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
stars = '***' if p_kw21<0.001 else '**' if p_kw21<0.01 else '*' if p_kw21<0.05 else '(ns)'
ax.text(0.98, 0.97, f"K-W p={p_kw21:.3f} {stars}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))
ax.legend(fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── C3 · log(bias) histogram — 2.1 vs 3.7 µm ────────────────
ax = axes[2]
bins = np.linspace(-2.5, 4.5, 28)
ax.hist(log_b21, bins=bins, color='#1565C0', alpha=0.35,
        edgecolor='white', lw=0.6, label='All (2.1 µm)', zorder=2)
ax.hist(log_b37, bins=bins, color='#AD1457', alpha=0.35,
        edgecolor='white', lw=0.6, label='All (3.7 µm)', zorder=2)
ax.axvline(0, color='#333333', lw=1.5, ls='--', alpha=0.7, label='No bias (log=0)')
ax.axvline(log_b21.mean(), color='#1565C0', lw=1.5, ls='-.',
           alpha=0.8, label=f'Mean 2.1µm ({log_b21.mean():.2f})')
ax.axvline(log_b37.mean(), color='#AD1457', lw=1.5, ls='-.',
           alpha=0.8, label=f'Mean 3.7µm ({log_b37.mean():.2f})')
ax.axvspan(0,  5, alpha=0.04, color='#B71C1C', label='Overestimate')
ax.axvspan(-3, 0, alpha=0.04, color='#1565C0', label='Underestimate')
ax.set_xlabel('log(Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_ylabel('Count', color=TXT, fontsize=10)
ax.set_title('C3 · Log-Bias Distribution\n(2.1 µm vs 3.7 µm)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.legend(fontsize=7.8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── C4 · bias_21 vs bias_37 scatter ──────────────────────────
ax = axes[3]
for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]
    ax.scatter(sub['bias_21'], sub['bias_37'],
               c=REGIME_COLORS[reg], s=65, zorder=4,
               edgecolors='white', linewidths=0.5, label=reg, alpha=0.90)

lim_b = max(b21.max(), b37.max()) * 1.1
ax.plot([0, lim_b], [0, lim_b], 'k--', lw=1.3, alpha=0.45, label='1:1')
ax.axhline(1.0, color='#999999', lw=0.9, ls=':', alpha=0.5)
ax.axvline(1.0, color='#999999', lw=0.9, ls=':', alpha=0.5)

r_bias, p_bias = pearsonr(b21, b37)
ax.text(0.97, 0.05,
        f"r = {r_bias:.3f}\np = {'<0.001' if p_bias<0.001 else f'{p_bias:.3f}'}",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))

handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
handles.append(plt.Line2D([0],[0], color='k', ls='--', lw=1.3, label='1:1'))
ax.legend(handles=handles, fontsize=7.8, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC')
ax.set_xlabel('Bias 2.1 µm  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_ylabel('Bias 3.7 µm  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_title('C4 · Bias 2.1 µm vs Bias 3.7 µm\n(How consistent are the two channels?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

fig.suptitle(
    'VOCALS-REx 2008 — Package C: MODIS vs In-Situ Nd Bias Analysis\n'
    'How Well Does MODIS Retrieve Cloud Droplet Number Concentration?',
    color=TXT, fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ════════════════════════════════════════════════════════════════
# 5. SUPPLEMENTARY FIGURE — 2×2
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11))
fig2.patch.set_facecolor('white')
axes2 = axes2.flatten()

# ── S1 · dNd (Nd21 - Nd37) by regime ─────────────────────────
ax = axes2[0]
data_dnd = [df[df['drizzle_regime']==r]['dNd'].dropna() for r in REGIME_ORDER]
bp2 = ax.boxplot(data_dnd, patch_artist=True, widths=0.52,
                 medianprops=dict(color='white', lw=2.5),
                 whiskerprops=dict(color='#555555', lw=1.2),
                 capprops=dict(color='#555555', lw=1.2),
                 flierprops=dict(marker='o', markersize=5,
                                 markerfacecolor='#999999', alpha=0.6))
for patch, reg in zip(bp2['boxes'], REGIME_ORDER):
    patch.set_facecolor(REGIME_COLORS[reg]); patch.set_alpha(0.82)
for i, (d, reg) in enumerate(zip(data_dnd, REGIME_ORDER)):
    jitter = np.random.uniform(-0.18, 0.18, size=len(d))
    ax.scatter(np.ones(len(d))*(i+1)+jitter, d,
               c=REGIME_COLORS[reg], s=45, zorder=5,
               edgecolors='white', linewidths=0.5, alpha=0.85)
    if len(d) > 0:
        med = d.median()
        ax.text(i+1.32, med, f'{med:.1f}',
                va='center', ha='left', color=TXT, fontsize=8, fontweight='bold')
ax.axhline(0, color='#333333', lw=1.3, ls='--', alpha=0.6)
kw_dnd = [g.dropna().values for g in data_dnd if len(g.dropna())>=2]
_, p_dnd = kruskal(*kw_dnd)
stars_dnd = '***' if p_dnd<0.001 else '**' if p_dnd<0.01 else '*' if p_dnd<0.05 else '(ns)'
ax.text(0.98, 0.97, f"K-W p={p_dnd:.3f} {stars_dnd}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))
ax.set_xticks(range(1,5))
ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
ax.set_ylabel('dNd = Nd_21 − Nd_37  (cm⁻³)', color=TXT, fontsize=10)
ax.set_title('S1 · Spectral Nd Difference (dNd)\n(2.1 µm − 3.7 µm)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── S2 · Median Nd: In-Situ vs MODIS bar chart ───────────────
ax = axes2[1]
x = np.arange(len(REGIME_ORDER))
w = 0.28
med_in  = [df[df['drizzle_regime']==r]['Nd_median'].median()   for r in REGIME_ORDER]
med_21  = [df[df['drizzle_regime']==r]['Nd_MODIS_21'].median() for r in REGIME_ORDER]
med_37  = [df[df['drizzle_regime']==r]['Nd_MODIS_37'].median() for r in REGIME_ORDER]

b_in = ax.bar(x - w, med_in,  width=w, color='#37474F', alpha=0.85,
              edgecolor='white', label='In-situ')
b_21 = ax.bar(x,     med_21,  width=w, color='#1565C0', alpha=0.85,
              edgecolor='white', label='MODIS 2.1 µm')
b_37 = ax.bar(x + w, med_37,  width=w, color='#AD1457', alpha=0.85,
              edgecolor='white', label='MODIS 3.7 µm')

for bars in [b_in, b_21, b_37]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f'{h:.0f}', ha='center', va='bottom',
                color=TXT, fontsize=7.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
ax.set_ylabel('Median Nd  (cm⁻³)', color=TXT, fontsize=10)
ax.set_title('S2 · Median Nd: In-Situ vs MODIS\n(by drizzle regime)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.legend(fontsize=8.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── S3 · bias_21 vs Nd_insitu scatter ────────────────────────
ax = axes2[2]
for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]
    ax.scatter(sub['Nd_median'], sub['bias_21'],
               c=REGIME_COLORS[reg], s=65, zorder=4,
               edgecolors='white', linewidths=0.5, label=reg, alpha=0.90)
ax.axhline(1.0, color='#333333', lw=1.3, ls='--', alpha=0.6, label='No bias')
r_nb, p_nb = spearmanr(df['Nd_median'], df['bias_21'])
ax.text(0.97, 0.97,
        f"Spearman r = {r_nb:.3f}\np = {'<0.001' if p_nb<0.001 else f'{p_nb:.3f}'}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))
handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
handles.append(plt.Line2D([0],[0], color='#333333', ls='--', lw=1.3, label='No bias'))
ax.legend(handles=handles, fontsize=7.8, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC')
ax.set_xlabel('Nd in-situ  (cm⁻³)', color=TXT, fontsize=10)
ax.set_ylabel('Bias 2.1 µm  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_title('S3 · Bias vs In-Situ Nd\n(Does MODIS perform worse for low-Nd clouds?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

# ── S4 · bias_21 vs LWP_insitu scatter ───────────────────────
ax = axes2[3]
for reg in REGIME_ORDER:
    sub = df[df['drizzle_regime'] == reg]
    ax.scatter(sub['LWP_insitu'], sub['bias_21'],
               c=REGIME_COLORS[reg], s=65, zorder=4,
               edgecolors='white', linewidths=0.5, label=reg, alpha=0.90)
ax.axhline(1.0, color='#333333', lw=1.3, ls='--', alpha=0.6, label='No bias')

lwp_vals  = df['LWP_insitu'].dropna()
bias_vals = df.loc[lwp_vals.index, 'bias_21'].dropna()
common    = lwp_vals.index.intersection(bias_vals.index)
r_lwp, p_lwp = spearmanr(lwp_vals[common], bias_vals[common])

ax.text(0.97, 0.97,
        f"Spearman r = {r_lwp:.3f}\np = {'<0.001' if p_lwp<0.001 else f'{p_lwp:.3f}'}",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                  edgecolor='#BBBBBB', alpha=0.95))
handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
handles.append(plt.Line2D([0],[0], color='#333333', ls='--', lw=1.3, label='No bias'))
ax.legend(handles=handles, fontsize=7.8, framealpha=0.9,
          facecolor='white', edgecolor='#CCCCCC')
ax.set_xlabel('LWP in-situ  (g m⁻²)', color=TXT, fontsize=10)
ax.set_ylabel('Bias 2.1 µm  (Nd_MODIS / Nd_insitu)', color=TXT, fontsize=10)
ax.set_title('S4 · Bias vs In-Situ LWP\n(Do thicker clouds have smaller bias?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_color('#AAAAAA')
ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
ax.set_axisbelow(True)

fig2.suptitle(
    'VOCALS-REx 2008 — Package C: Supplementary — Nd Bias Deep Dive\n'
    'Spectral Differences, Regime Medians & Bias Drivers',
    color=TXT, fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

print("\n" + "=" * 65)
print("✓  Package C complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
print("=" * 65)
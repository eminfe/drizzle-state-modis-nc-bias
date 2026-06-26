# =============================================================================
# step12_packageB.py — Package B: MODIS Applicability & Match Rate Analysis
# =============================================================================
# Campaign : MASE 2005
# Goal     : Which profiles have valid MODIS retrievals, and why do some fail?
#            What is the spatial/temporal/geometric context of matches?
# Data     : {CAMPAIGN}_MODIS_Matches.csv  ->  modis
# Outputs  : outputs/figures/{CAMPAIGN}_PackageB_main.png
#            outputs/figures/{CAMPAIGN}_PackageB_geo.png
#            Terminal statistics table
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from scipy.stats import kruskal
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
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageB_main.png'
OUT_GEO    = FIG_DIR / f'{NAME}_PackageB_geo.png'



# ════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════
modis = pd.read_csv(MODIS_CSV)

# Normalize regime labels
modis['drizzle_regime'] = (
    modis['drizzle_regime']
    .astype(str).str.strip()
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

STATUS_COLORS = {
    'MATCHED'         : '#1565C0',
    'NO_VALID_PIXELS' : '#F57F17',
    'NO_COVERAGE'     : '#6A1B9A',
    'NO_MATCH'        : '#B71C1C',
}

print("=" * 60)
print("PACKAGE B — MODIS APPLICABILITY")
print("=" * 60)
print(f"\nTotal profiles loaded : {len(modis)}")
print("\nmatch_status distribution:")
print(modis['match_status'].value_counts())
print("\nColumns available:")
print([c for c in modis.columns])

# ════════════════════════════════════════════════════════════════
# 2. STYLE
# ════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.edgecolor'   : '#333333',
    'axes.labelcolor'  : '#111111',
    'xtick.color'      : '#111111',
    'ytick.color'      : '#111111',
    'text.color'       : '#111111',
    'grid.color'       : '#dddddd',
    'grid.linestyle'   : '--',
    'grid.linewidth'   : 0.6,
    'font.size'        : 10,
})
TXT   = '#111111'
ANNOT = '#333333'

# ════════════════════════════════════════════════════════════════
# 3. DERIVED SUBSETS
# ════════════════════════════════════════════════════════════════
modis['match_status'] = modis['match_status'].astype(str).str.strip().str.upper()

matched  = modis[modis['match_status'] == 'MATCHED'].copy()
no_valid = modis[modis['match_status'] == 'NO_VALID_PIXELS'].copy()
no_cov   = modis[modis['match_status'] == 'NO_COVERAGE'].copy()
no_match = modis[modis['match_status'] == 'NO_MATCH'].copy()

n_matched = len(matched)
n_novalid = len(no_valid)
n_nocov   = len(no_cov)
n_nomatch = len(no_match)
n_total   = len(modis)

print(f"\nMATCHED      : {n_matched}  ({100*n_matched/n_total:.1f}%)")
print(f"NO_VALID     : {n_novalid}  ({100*n_novalid/n_total:.1f}%)")
print(f"NO_COVERAGE  : {n_nocov}  ({100*n_nocov/n_total:.1f}%)")
print(f"NO_MATCH     : {n_nomatch}  ({100*n_nomatch/n_total:.1f}%)")

# ════════════════════════════════════════════════════════════════
# 4. MAIN FIGURE  (3 rows × 2 cols)
# ════════════════════════════════════════════════════════════════
np.random.seed(42)
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

# ── B1 · Donut: overall match rate ────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])  # row0, col0
sizes_raw  = [n_matched, n_novalid, n_nocov, n_nomatch]
labels_raw = [f'MATCHED\n(n={n_matched})',
              f'NO_VALID_PIXELS\n(n={n_novalid})',
              f'NO_COVERAGE\n(n={n_nocov})',
              f'NO_MATCH\n(n={n_nomatch})']
colors_raw = [STATUS_COLORS['MATCHED'],
              STATUS_COLORS['NO_VALID_PIXELS'],
              STATUS_COLORS['NO_COVERAGE'],
              STATUS_COLORS['NO_MATCH']]
sizes, labels, colors = zip(*[
    (s, l, c) for s, l, c in zip(sizes_raw, labels_raw, colors_raw) if s > 0
])
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
    textprops=dict(color=TXT, fontsize=9),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(9)
ax1.set_title(f'B1 · Overall MODIS Match Rate\n({len(modis)} profiles)',
              color=TXT, fontsize=10, fontweight='bold', pad=10)

# ── B2 · Match status by drizzle regime (stacked bar) ─────────
ax2 = fig.add_subplot(gs[0, 1])  # row0, col1
regime_status = (modis.groupby(['drizzle_regime', 'match_status'])
                 .size().unstack(fill_value=0))
for s in ['MATCHED', 'NO_VALID_PIXELS', 'NO_COVERAGE', 'NO_MATCH']:
    if s not in regime_status.columns:
        regime_status[s] = 0
regime_status = regime_status.reindex(REGIME_ORDER, fill_value=0)

x      = np.arange(len(REGIME_ORDER))
bottom = np.zeros(len(REGIME_ORDER))
for status in ['MATCHED', 'NO_VALID_PIXELS', 'NO_COVERAGE', 'NO_MATCH']:
    vals = regime_status[status].values
    ax2.bar(x, vals, bottom=bottom,
            color=STATUS_COLORS[status], label=status,
            edgecolor='white', linewidth=0.8, alpha=0.88)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0:
            ax2.text(i, b + v / 2, str(int(v)),
                     ha='center', va='center', color='white',
                     fontsize=9, fontweight='bold')
    bottom += vals

ax2.set_xticks(x)
ax2.set_xticklabels(['Non-\ndrizzle', 'Weak\ndrizzle',
                     'Moderate\ndrizzle', 'Heavy\ndrizzle'],
                    color=TXT, fontsize=8.5)
ax2.set_ylabel('Number of Profiles', color=TXT)
ax2.set_title('B2 · Match Status by Drizzle Regime',
              color=TXT, fontsize=10, fontweight='bold', pad=8)
ax2.legend(
    [Patch(facecolor=STATUS_COLORS['MATCHED'],         label='MATCHED'),
     Patch(facecolor=STATUS_COLORS['NO_VALID_PIXELS'], label='NO_VALID_PIXELS'),
     Patch(facecolor=STATUS_COLORS['NO_COVERAGE'],     label='NO_COVERAGE'),
     Patch(facecolor=STATUS_COLORS['NO_MATCH'],        label='NO_MATCH')],
    ['MATCHED', 'NO_VALID_PIXELS', 'NO_COVERAGE', 'NO_MATCH'],
    fontsize=8.5, framealpha=0.9, edgecolor='#CCCCCC',
    facecolor='white', loc='upper right')
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#AAAAAA')
ax2.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
ax2.set_axisbelow(True)

print("\nMatch status by drizzle regime:")
print(regime_status[['MATCHED', 'NO_VALID_PIXELS', 'NO_COVERAGE', 'NO_MATCH']])

# ── B3 · Match rate % by regime ───────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])  # row0, col2
match_pct = []
for reg in REGIME_ORDER:
    sub = modis[modis['drizzle_regime'] == reg]
    n   = len(sub)
    m   = (sub['match_status'] == 'MATCHED').sum()
    match_pct.append(100 * m / n if n > 0 else 0)

ax3.bar(x, match_pct,
        color=[REGIME_COLORS[r] for r in REGIME_ORDER],
        edgecolor='white', linewidth=0.8, alpha=0.85, width=0.55)
for i, v in enumerate(match_pct):
    ax3.text(i, v + 1.5, f'{v:.0f}%',
             ha='center', va='bottom', color=TXT,
             fontsize=10, fontweight='bold')
ax3.axhline(100 * n_matched / n_total, color='#555555',
            lw=1.5, ls='--', alpha=0.7,
            label=f'Overall {100*n_matched/n_total:.0f}%')
ax3.set_xticks(x)
ax3.set_xticklabels(['Non-\ndrizzle', 'Weak\ndrizzle',
                     'Moderate\ndrizzle', 'Heavy\ndrizzle'],
                    color=TXT, fontsize=8.5)
ax3.set_ylabel('Match Rate  (%)', color=TXT)
ax3.set_ylim(0, 115)
ax3.set_title('B3 · MODIS Match Rate (%) by Regime',
              color=TXT, fontsize=10, fontweight='bold', pad=8)
ax3.legend(fontsize=8.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
ax3.spines[['top', 'right']].set_visible(False)
ax3.spines[['left', 'bottom']].set_color('#AAAAAA')
ax3.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
ax3.set_axisbelow(True)

# ── B4 · Re_MODIS_21 vs Re_insitu ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])  # row1, col0
re_insitu_col = 're_cas_median'
re_modis_col  = 'Re_MODIS_21'
if re_insitu_col in matched.columns and re_modis_col in matched.columns:
    for reg in REGIME_ORDER:
        sub = matched[matched['drizzle_regime'] == reg]
        ax4.scatter(sub[re_insitu_col], sub[re_modis_col],
                    c=REGIME_COLORS[reg], s=60,
                    edgecolors='white', linewidths=0.5,
                    label=reg, zorder=3)
    lim_max = max(matched[re_insitu_col].max(),
                  matched[re_modis_col].max()) * 1.1
    ax4.plot([0, lim_max], [0, lim_max], 'k--', lw=1.2, alpha=0.5, label='1:1')
    ax4.set_xlabel('r$_e$ in-situ CAS  (µm)', color=TXT)
    ax4.set_ylabel('r$_e$ MODIS 2.1 µm  (µm)', color=TXT)
    ax4.set_title('B4 · Re: MODIS 2.1 µm vs In-Situ\n(MATCHED profiles only)',
                  color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax4.legend(fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax4.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax4.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax4.set_axisbelow(True)
else:
    ax4.set_visible(False)
    print(f"B4 skipped — missing columns: {re_insitu_col} or {re_modis_col}")

# ── B5 · tau_MODIS_21 vs tau_insitu ───────────────────────────
ax5 = fig.add_subplot(gs[1, 1])  # row1, col1
tau_insitu_col = 'tau_main'
tau_modis_col  = 'tau_MODIS_21'
if tau_insitu_col in matched.columns and tau_modis_col in matched.columns:
    for reg in REGIME_ORDER:
        sub = matched[matched['drizzle_regime'] == reg]
        ax5.scatter(sub[tau_insitu_col], sub[tau_modis_col],
                    c=REGIME_COLORS[reg], s=60,
                    edgecolors='white', linewidths=0.5,
                    label=reg, zorder=3)
    lim_max = max(matched[tau_insitu_col].max(),
                  matched[tau_modis_col].max()) * 1.1
    ax5.plot([0, lim_max], [0, lim_max], 'k--', lw=1.2, alpha=0.5, label='1:1')
    ax5.set_xlabel('τ in-situ (main)', color=TXT)
    ax5.set_ylabel('τ MODIS 2.1 µm', color=TXT)
    ax5.set_title('B5 · Optical Depth: MODIS vs In-Situ\n(MATCHED profiles only)',
                  color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax5.legend(fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
    ax5.spines[['top', 'right']].set_visible(False)
    ax5.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax5.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax5.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax5.set_axisbelow(True)
else:
    ax5.set_visible(False)
    print(f"B5 skipped — missing columns: {tau_insitu_col} or {tau_modis_col}")

# ── B6 · LWP_MODIS vs LWP_insitu ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])  # row1, col2
lwp_insitu_col = 'LWP_insitu'
lwp_modis_col  = 'LWP_MODIS'
if lwp_insitu_col in matched.columns and lwp_modis_col in matched.columns:
    for reg in REGIME_ORDER:
        sub = matched[matched['drizzle_regime'] == reg]
        ax6.scatter(sub[lwp_insitu_col], sub[lwp_modis_col],
                    c=REGIME_COLORS[reg], s=60,
                    edgecolors='white', linewidths=0.5,
                    label=reg, zorder=3)
    lim_max = max(matched[lwp_insitu_col].max(),
                  matched[lwp_modis_col].max()) * 1.1
    ax6.plot([0, lim_max], [0, lim_max], 'k--', lw=1.2, alpha=0.5, label='1:1')
    ax6.set_xlabel('LWP in-situ  (g m⁻²)', color=TXT)
    ax6.set_ylabel('LWP MODIS  (g m⁻²)', color=TXT)
    ax6.set_title('B6 · LWP: MODIS vs In-Situ\n(MATCHED profiles only)',
                  color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax6.legend(fontsize=7.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
    ax6.spines[['top', 'right']].set_visible(False)
    ax6.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax6.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax6.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax6.set_axisbelow(True)
else:
    ax6.set_visible(False)
    print(f"B6 skipped — missing columns: {lwp_insitu_col} or {lwp_modis_col}")

fig.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package B: MODIS Applicability & Match Rate Analysis\n'
    f'Which Profiles Have Valid MODIS Retrievals?',
    color=TXT, fontsize=14, fontweight='bold', y=1.02,
)
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ════════════════════════════════════════════════════════════════
# 5. GEOMETRY FIGURE  (VZA, SZA, CTT, CTP)
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 11))
fig2.patch.set_facecolor('white')
axes2 = axes2.flatten()

def geo_boxplot(ax, col, ylabel, title, df=None):
    if df is None:
        df = matched
    if col not in df.columns:
        ax.set_visible(False)
        print(f"  geo_boxplot: '{col}' not found — panel hidden")
        return
    data   = [df[df['drizzle_regime'] == r][col].dropna() for r in REGIME_ORDER]
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
        patch.set_facecolor(color); patch.set_alpha(0.80)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.uniform(-0.18, 0.18, size=len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                   color=color, s=30, alpha=0.70, zorder=4,
                   edgecolors='white', linewidths=0.5)
        if len(d) > 0:
            med = np.nanmedian(d)
            ax.text(i + 1.30, med, f'{med:.1f}',
                    va='center', ha='left', color=TXT,
                    fontsize=7.5, fontweight='bold')
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(['Non-\ndrizzle', 'Weak\ndrizzle',
                        'Moderate\ndrizzle', 'Heavy\ndrizzle'],
                       color=TXT, fontsize=8)
    ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    ax.tick_params(colors=TXT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)
    valid = [g.dropna().values for g in data if len(g.dropna()) >= 2]
    if len(valid) >= 2:
        _, p = kruskal(*valid)
        stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
        kw_str = f"K-W p<0.001 {stars}" if p < 0.001 else f"K-W p={p:.3f} {stars}"
        ax.text(0.98, 0.97, kw_str, transform=ax.transAxes,
                ha='right', va='top', color=ANNOT, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#F5F5F5', edgecolor='#BBBBBB', alpha=0.95))

geo_boxplot(axes2[0], 'VZA_mean',  'VZA  (°)',   'G1 · View Zenith Angle')
geo_boxplot(axes2[1], 'SZA_mean',  'SZA  (°)',   'G2 · Solar Zenith Angle')
geo_boxplot(axes2[2], 'CTT_MODIS', 'CTT  (K)',   'G3 · Cloud Top Temperature')
geo_boxplot(axes2[3], 'CTP_MODIS', 'CTP  (hPa)', 'G4 · Cloud Top Pressure')

fig2.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package B: Geometry & Cloud Top Properties  (MATCHED profiles, colour = regime)',
    color=TXT, fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_GEO, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Geometry figure saved → {OUT_GEO}")

# ════════════════════════════════════════════════════════════════
# 6. STATISTICS TABLE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"{'PACKAGE B — MATCHED PROFILES STATISTICS':^72}")
print("=" * 72)

STAT_COLS_B = [
    ('Re_MODIS_21',  'Re MODIS 2.1 µm'),
    ('Re_MODIS_37',  'Re MODIS 3.7 µm'),
    ('tau_MODIS_21', 'tau MODIS 2.1 µm'),
    ('tau_MODIS_37', 'tau MODIS 3.7 µm'),
    ('LWP_MODIS',    'LWP MODIS (g/m²)'),
    ('LWP_insitu',   'LWP in-situ (g/m²)'),
    ('VZA_mean',     'VZA mean (°)'),
    ('SZA_mean',     'SZA mean (°)'),
    ('CTT_MODIS',    'CTT MODIS (K)'),
    ('CTP_MODIS',    'CTP MODIS (hPa)'),
]

for col, label in STAT_COLS_B:
    if col not in matched.columns:
        print(f"\n{label}: COLUMN NOT FOUND")
        continue
    print(f"\n{label}")
    groups = []
    for reg in REGIME_ORDER:
        vals = matched[matched['drizzle_regime'] == reg][col].dropna()
        groups.append(vals)
        if len(vals) == 0:
            print(f"  {reg:22s}: —")
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        print(f"  {reg:22s}: {med:8.2f}  [{q1:.2f} – {q3:.2f}]  (n={len(vals)})")
    valid = [g.dropna().values for g in groups if len(g.dropna()) >= 2]
    if len(valid) >= 2:
        _, p = kruskal(*valid)
        stars  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '(ns)'
        kw_str = f"K-W p<0.001 {stars}" if p < 0.001 else f"K-W p={p:.3f} {stars}"
        print(f"  → {kw_str}")

print("\n" + "=" * 72)
print("✓  Package B complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_GEO}")
print("=" * 72)
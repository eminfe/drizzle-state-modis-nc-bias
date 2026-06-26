# =============================================================================
# step15_packageE.py — Package E: Spectral Sensitivity (2.1 µm vs 3.7 µm)
# =============================================================================
# Campaign : MASE 2005
# Goal     : Do the two MODIS channels tell the same physical story?
#            Re37 vs Re21, Tau37 vs Tau21, dRe and dNd vs drizzle/f_ad/VZA
# Data     : {CAMPAIGN}_MODIS_Matches.csv  ->  MATCHED profiles
# Outputs  : {CAMPAIGN}_PackageE_main.png   (2x3 - Re/Tau scatter + dRe/dNd panels)
#            {CAMPAIGN}_PackageE_supp.png   (2x2 - dNd vs f_ad, VZA, Nd_insitu)
#            Terminal: full statistics table
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageE_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageE_supp.png'


modis = pd.read_csv(MODIS_CSV)
modis['drizzle_regime'] = (
    modis['drizzle_regime'].astype(str).str.strip()
    .str.replace('_', ' ').str.title()
    .str.replace('Drizzling', 'drizzling')
    .str.replace('Non drizzling', 'Non-drizzling')
)
modis['match_status'] = modis['match_status'].astype(str).str.strip().str.upper()
df = modis[modis['match_status'] == 'MATCHED'].copy()
df['drizzle_regime'] = pd.Categorical(
    df['drizzle_regime'],
    categories=['Non-drizzling', 'Weak drizzling', 'Moderate drizzling', 'Heavy drizzling'],
    ordered=True
)

REGIME_ORDER  = ['Non-drizzling', 'Weak drizzling', 'Moderate drizzling', 'Heavy drizzling']
REGIME_COLORS = {
    'Non-drizzling'     : '#2E7D32',
    'Weak drizzling'    : '#F9A825',
    'Moderate drizzling': '#E65100',
    'Heavy drizzling'   : '#B71C1C',
}
SHORT_LABELS = ['Non-\ndrizzle', 'Weak\ndrizzle', 'Moderate\ndrizzle', 'Heavy\ndrizzle']

# ── Derived columns (if not already in CSV) ──────────────────
# Re columns: try both naming conventions
for cand21, cand37 in [('Re_MODIS_21', 'Re_MODIS_37'),
                        ('re_MODIS_21', 're_MODIS_37'),
                        ('Re_21', 'Re_37')]:
    if cand21 in df.columns and cand37 in df.columns:
        df['Re_MODIS_21'] = df[cand21]
        df['Re_MODIS_37'] = df[cand37]
        break

# Tau columns: try both naming conventions
for cand21, cand37 in [('tau_MODIS_21', 'tau_MODIS_37'),
                        ('Tau_MODIS_21', 'Tau_MODIS_37'),
                        ('tau_21', 'tau_37')]:
    if cand21 in df.columns and cand37 in df.columns:
        df['tau_MODIS_21'] = df[cand21]
        df['tau_MODIS_37'] = df[cand37]
        break

# Nd columns: try multiple naming conventions, including step10's *_calc suffix
for cand21, cand37 in [('Nd_MODIS_21', 'Nd_MODIS_37'),
                        ('Nd_MODIS_21_calc', 'Nd_MODIS_37_calc'),
                        ('Nd_21', 'Nd_37')]:
    if cand21 in df.columns and cand37 in df.columns:
        df['Nd_MODIS_21'] = df[cand21]
        df['Nd_MODIS_37'] = df[cand37]
        break

if 'dRe' not in df.columns:
    if 'dRe_37_21' in df.columns:
        # Preferred: paired-pool difference from step09 (Felsefe A, apples-to-apples)
        df['dRe'] = df['dRe_37_21']
    elif 'Re_MODIS_37' in df.columns and 'Re_MODIS_21' in df.columns:
        # Fallback: column subtraction (note: Re_MODIS_21 uses qc_21 pool while
        # Re_MODIS_37 uses qc_both pool, so the difference is NOT strictly
        # apples-to-apples; prefer dRe_37_21 when present)
        df['dRe'] = df['Re_MODIS_37'] - df['Re_MODIS_21']          # µm
if 'dNd' not in df.columns:
    if 'dNd_calc' in df.columns:
        # Preferred: paired-pool difference from step10 (Felsefe A, apples-to-apples)
        df['dNd'] = df['dNd_calc']
    elif 'Nd_MODIS_21' in df.columns and 'Nd_MODIS_37' in df.columns:
        # Fallback: column subtraction (mixed pools; prefer dNd_calc when present)
        df['dNd'] = df['Nd_MODIS_21'] - df['Nd_MODIS_37']          # cm⁻³
if 'dTau' not in df.columns:
    if 'dtau_37_21' in df.columns:
        df['dTau'] = df['dtau_37_21']
    elif 'tau_MODIS_37' in df.columns and 'tau_MODIS_21' in df.columns:
        df['dTau'] = df['tau_MODIS_37'] - df['tau_MODIS_21']

print(f"MATCHED profiles : {len(df)}")
print(df['drizzle_regime'].value_counts()[REGIME_ORDER])
print("\nAvailable columns (first 40):", list(df.columns[:40]))

# ════════════════════════════════════════════════════════════════
# 1. STYLE
# ════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#111111',
    'xtick.color': '#111111', 'ytick.color': '#111111',
    'text.color': '#111111', 'grid.color': '#dddddd',
    'grid.linestyle': '--', 'grid.linewidth': 0.6, 'font.size': 10,
})
TXT = '#111111'
np.random.seed(42)

# ════════════════════════════════════════════════════════════════
# 2. STATISTICS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PACKAGE E — SPECTRAL SENSITIVITY STATISTICS")
print("=" * 65)

# ── Re comparison ────────────────────────────────────────────
re21 = df['Re_MODIS_21'].dropna() if 'Re_MODIS_21' in df.columns else pd.Series(dtype=float)
re37 = df['Re_MODIS_37'].dropna() if 'Re_MODIS_37' in df.columns else pd.Series(dtype=float)

r_re = p_re = slope_re = intercept_re = p_re_wil = None
if 'Re_MODIS_21' in df.columns and 'Re_MODIS_37' in df.columns:
    common_re = df[['Re_MODIS_21', 'Re_MODIS_37']].dropna()
    if len(common_re) >= 5:
        _, p_re_wil = wilcoxon(common_re['Re_MODIS_37'] - common_re['Re_MODIS_21'])
        r_re, p_re  = pearsonr(common_re['Re_MODIS_21'], common_re['Re_MODIS_37'])
        slope_re, intercept_re, _, _, _ = linregress(
            common_re['Re_MODIS_21'], common_re['Re_MODIS_37'])
        print(f"\nRe MODIS 2.1 µm : median={re21.median():.2f}  [{re21.quantile(.25):.2f}–{re21.quantile(.75):.2f}] µm")
        print(f"Re MODIS 3.7 µm : median={re37.median():.2f}  [{re37.quantile(.25):.2f}–{re37.quantile(.75):.2f}] µm")
        if 'dRe' in df.columns:
            print(f"dRe (37-21) median : {df['dRe'].median():.2f} µm  (3.7 µm sees LARGER drops?)")
        print(f"Wilcoxon (Re37 vs Re21) : p = {p_re_wil:.4f}")
        print(f"Pearson r (Re21 vs Re37) : r = {r_re:.3f}  p = {p_re:.4f}")
        print(f"OLS slope (Re37 ~ Re21) : {slope_re:.3f}  intercept: {intercept_re:.3f}")
else:
    print("\nRe_MODIS_21 / Re_MODIS_37 columns not found — skipping Re stats")

# ── Tau comparison ───────────────────────────────────────────
tau21 = df['tau_MODIS_21'].dropna() if 'tau_MODIS_21' in df.columns else pd.Series(dtype=float)
tau37 = df['tau_MODIS_37'].dropna() if 'tau_MODIS_37' in df.columns else pd.Series(dtype=float)

r_tau = p_tau = slope_tau = intercept_tau = p_tau_wil = None
if 'tau_MODIS_21' in df.columns and 'tau_MODIS_37' in df.columns:
    common_tau = df[['tau_MODIS_21', 'tau_MODIS_37']].dropna()
    if len(common_tau) >= 5:
        _, p_tau_wil = wilcoxon(common_tau['tau_MODIS_37'] - common_tau['tau_MODIS_21'])
        r_tau, p_tau = pearsonr(common_tau['tau_MODIS_21'], common_tau['tau_MODIS_37'])
        slope_tau, intercept_tau, _, _, _ = linregress(
            common_tau['tau_MODIS_21'], common_tau['tau_MODIS_37'])
        print(f"\nτ MODIS 2.1 µm : median={tau21.median():.2f}  [{tau21.quantile(.25):.2f}–{tau21.quantile(.75):.2f}]")
        print(f"τ MODIS 3.7 µm : median={tau37.median():.2f}  [{tau37.quantile(.25):.2f}–{tau37.quantile(.75):.2f}]")
        if 'dTau' in df.columns:
            print(f"dTau (37-21) median : {df['dTau'].median():.2f}")
        print(f"Wilcoxon (Tau37 vs Tau21) : p = {p_tau_wil:.4f}")
        print(f"Pearson r (Tau21 vs Tau37) : r = {r_tau:.3f}  p = {p_tau:.4f}")
        print(f"OLS slope (Tau37 ~ Tau21) : {slope_tau:.3f}  intercept: {intercept_tau:.3f}")
else:
    print("\ntau_MODIS_21 / tau_MODIS_37 columns not found — skipping Tau stats")

# ── dRe by regime ────────────────────────────────────────────
kw_dre = []
if 'dRe' in df.columns:
    print("\ndRe (Re37 - Re21) by regime:")
    for reg in REGIME_ORDER:
        sub = df[df['drizzle_regime'] == reg]['dRe'].dropna()
        if len(sub) == 0: continue
        print(f"  {reg:22s}: median={sub.median():.2f}  [{sub.quantile(.25):.2f}–{sub.quantile(.75):.2f}]  (n={len(sub)})")
        if len(sub) >= 2: kw_dre.append(sub)
    if len(kw_dre) >= 2:
        _, p_kw_dre = kruskal(*kw_dre)
        print(f"  → K-W p = {p_kw_dre:.4f}")

# ── dNd by regime ────────────────────────────────────────────
kw_dnd = []
if 'dNd' in df.columns:
    print("\ndNd (Nd21 - Nd37) by regime:")
    for reg in REGIME_ORDER:
        sub = df[df['drizzle_regime'] == reg]['dNd'].dropna()
        if len(sub) == 0: continue
        print(f"  {reg:22s}: median={sub.median():.2f}  [{sub.quantile(.25):.2f}–{sub.quantile(.75):.2f}]  (n={len(sub)})")
        if len(sub) >= 2: kw_dnd.append(sub)
    if len(kw_dnd) >= 2:
        _, p_kw_dnd = kruskal(*kw_dnd)
        print(f"  → K-W p = {p_kw_dnd:.4f}")

# ── dNd correlations ─────────────────────────────────────────
if 'dNd' in df.columns:
    print("\ndNd Spearman correlations:")
    for col, label in [('f_ad_mean', 'f_ad'), ('VZA_mean', 'VZA'),
                       ('Nd_median', 'Nd_insitu'), ('tau_main', 'τ_insitu'),
                       ('re_cas_median', 're_CAS')]:
        if col not in df.columns: continue
        valid = df[[col, 'dNd']].dropna()
        if len(valid) < 5: continue
        r, p = spearmanr(valid[col], valid['dNd'])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  dNd vs {label:<15}: r={r:+.3f}  p={p:.4f}  {sig}")

# ════════════════════════════════════════════════════════════════
# 3. HELPERS
# ════════════════════════════════════════════════════════════════
def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.yaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax.xaxis.grid(True, lw=0.6, color='#DDDDDD', zorder=0)
    ax.set_axisbelow(True)

def regime_scatter(ax, xcol, ycol):
    for reg in REGIME_ORDER:
        sub = df[df['drizzle_regime'] == reg].dropna(subset=[xcol, ycol])
        ax.scatter(sub[xcol], sub[ycol],
                   c=REGIME_COLORS[reg], s=65, zorder=4,
                   edgecolors='white', linewidths=0.5, alpha=0.90, label=reg)
    handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='#CCCCCC', loc='upper left')

def stats_box(ax, txt, loc='lower right'):
    lx = 0.97 if 'right' in loc else 0.03
    ly = 0.05 if 'lower' in loc else 0.97
    ha = 'right' if 'right' in loc else 'left'
    va = 'bottom' if 'lower' in loc else 'top'
    ax.text(lx, ly, txt, transform=ax.transAxes,
            ha=ha, va=va, fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#F5F5F5',
                      edgecolor='#BBBBBB', alpha=0.95))

def col_missing(ax, name):
    ax.text(0.5, 0.5, f'Column\n"{name}"\nnot found in CSV',
            transform=ax.transAxes, ha='center', va='center',
            color='red', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

# ════════════════════════════════════════════════════════════════
# 4. MAIN FIGURE — 2×3
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.patch.set_facecolor('white')
axes = axes.flatten()

# ── E1 · Re_MODIS_37 vs Re_MODIS_21 ─────────────────────────
ax = axes[0]
if 'Re_MODIS_21' in df.columns and 'Re_MODIS_37' in df.columns:
    regime_scatter(ax, 'Re_MODIS_21', 'Re_MODIS_37')
    lim = max(df['Re_MODIS_21'].max(), df['Re_MODIS_37'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1.3, alpha=0.45, label='1:1')
    if slope_re is not None:
        x_fit = np.linspace(df['Re_MODIS_21'].min()*0.9,
                            df['Re_MODIS_21'].max()*1.1, 100)
        ax.plot(x_fit, intercept_re + slope_re*x_fit,
                color='#1565C0', lw=1.8, alpha=0.7,
                label=f'OLS (slope={slope_re:.2f})')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        stats_box(ax,
                  f"r = {r_re:.3f}\nslope = {slope_re:.2f}\n"
                  f"Wilcoxon p = {'<0.001' if p_re_wil<0.001 else f'{p_re_wil:.3f}'}")
    ax.set_xlabel('Re MODIS 2.1 µm  (µm)', color=TXT, fontsize=10)
    ax.set_ylabel('Re MODIS 3.7 µm  (µm)', color=TXT, fontsize=10)
else:
    col_missing(ax, 'Re_MODIS_21 / Re_MODIS_37')
ax.set_title('E1 · Re: 3.7 µm vs 2.1 µm\n(Does 3.7 µm see larger drops?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

# ── E2 · tau_MODIS_37 vs tau_MODIS_21 ───────────────────────
ax = axes[1]
if 'tau_MODIS_21' in df.columns and 'tau_MODIS_37' in df.columns:
    regime_scatter(ax, 'tau_MODIS_21', 'tau_MODIS_37')
    lim_t = max(df['tau_MODIS_21'].max(), df['tau_MODIS_37'].max()) * 1.1
    ax.plot([0, lim_t], [0, lim_t], 'k--', lw=1.3, alpha=0.45)
    if slope_tau is not None:
        x_fit_t = np.linspace(df['tau_MODIS_21'].min()*0.9,
                              df['tau_MODIS_21'].max()*1.1, 100)
        ax.plot(x_fit_t, intercept_tau + slope_tau*x_fit_t,
                color='#1565C0', lw=1.8, alpha=0.7,
                label=f'OLS (slope={slope_tau:.2f})')
        ax.set_xlim(0, lim_t); ax.set_ylim(0, lim_t)
        stats_box(ax,
                  f"r = {r_tau:.3f}\nslope = {slope_tau:.2f}\n"
                  f"Wilcoxon p = {'<0.001' if p_tau_wil<0.001 else f'{p_tau_wil:.3f}'}")
    ax.set_xlabel('τ MODIS 2.1 µm', color=TXT, fontsize=10)
    ax.set_ylabel('τ MODIS 3.7 µm', color=TXT, fontsize=10)
else:
    col_missing(ax, 'tau_MODIS_21 / tau_MODIS_37')
ax.set_title('E2 · τ: 3.7 µm vs 2.1 µm\n(Channels agree on optical depth?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

# ── E3 · dRe by regime boxplot ───────────────────────────────
ax = axes[2]
if 'dRe' in df.columns:
    data_dre = [df[df['drizzle_regime'] == r]['dRe'].dropna() for r in REGIME_ORDER]
    bp = ax.boxplot(data_dre, patch_artist=True, widths=0.52,
                    medianprops=dict(color='white', lw=2.5),
                    whiskerprops=dict(color='#555555', lw=1.2),
                    capprops=dict(color='#555555', lw=1.2),
                    flierprops=dict(marker='o', markersize=5,
                                    markerfacecolor='#999999', alpha=0.6))
    for patch, reg in zip(bp['boxes'], REGIME_ORDER):
        patch.set_facecolor(REGIME_COLORS[reg]); patch.set_alpha(0.82)
    for i, (d, reg) in enumerate(zip(data_dre, REGIME_ORDER)):
        jitter = np.random.uniform(-0.18, 0.18, size=len(d))
        ax.scatter(np.ones(len(d))*(i+1)+jitter, d,
                   c=REGIME_COLORS[reg], s=45, zorder=5,
                   edgecolors='white', linewidths=0.5, alpha=0.85)
        if len(d) > 0:
            med = d.median()
            ax.text(i+1.32, med, f'{med:.1f}',
                    va='center', ha='left', color=TXT,
                    fontsize=8.5, fontweight='bold')
    ax.axhline(0, color='#333333', lw=1.3, ls='--', alpha=0.55, label='No diff (dRe=0)')
    if len(kw_dre) >= 2:
        _, p_kw_dre2 = kruskal(*kw_dre)
        stars = '***' if p_kw_dre2<0.001 else '**' if p_kw_dre2<0.01 else '*' if p_kw_dre2<0.05 else 'ns'
        stats_box(ax, f"K-W p = {p_kw_dre2:.3f} {stars}", loc='upper right')
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
    ax.set_ylabel('dRe = Re_37 − Re_21  (µm)', color=TXT, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
else:
    col_missing(ax, 'dRe')
ax.set_title('E3 · Spectral Re Difference (dRe)\n(3.7 µm − 2.1 µm, by regime)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

# ── E4 · dNd by regime boxplot ───────────────────────────────
ax = axes[3]
if 'dNd' in df.columns:
    data_dnd = [df[df['drizzle_regime'] == r]['dNd'].dropna() for r in REGIME_ORDER]
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
                    va='center', ha='left', color=TXT,
                    fontsize=8.5, fontweight='bold')
    ax.axhline(0, color='#333333', lw=1.3, ls='--', alpha=0.55, label='No diff (dNd=0)')
    if len(kw_dnd) >= 2:
        _, p_kw_dnd2 = kruskal(*kw_dnd)
        stars = '***' if p_kw_dnd2<0.001 else '**' if p_kw_dnd2<0.01 else '*' if p_kw_dnd2<0.05 else 'ns'
        stats_box(ax, f"K-W p = {p_kw_dnd2:.3f} {stars}", loc='upper right')
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
    ax.set_ylabel('dNd = Nd_21 − Nd_37  (cm⁻³)', color=TXT, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
else:
    col_missing(ax, 'dNd')
ax.set_title('E4 · Spectral Nd Difference (dNd)\n(2.1 µm − 3.7 µm, by regime)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

# ── E5 · dRe vs dNd scatter ──────────────────────────────────
ax = axes[4]
if 'dRe' in df.columns and 'dNd' in df.columns:
    valid_e5 = df[['dRe', 'dNd', 'drizzle_regime']].dropna()
    for reg in REGIME_ORDER:
        sub = valid_e5[valid_e5['drizzle_regime'] == reg]
        ax.scatter(sub['dRe'], sub['dNd'],
                   c=REGIME_COLORS[reg], s=65, zorder=4,
                   edgecolors='white', linewidths=0.5, alpha=0.90, label=reg)
    ax.axhline(0, color='#999999', lw=0.9, ls=':', alpha=0.6)
    ax.axvline(0, color='#999999', lw=0.9, ls=':', alpha=0.6)
    r_e5, p_e5 = spearmanr(valid_e5['dRe'], valid_e5['dNd'])
    if len(valid_e5) >= 5:
        sl, ic, _, _, _ = linregress(valid_e5['dRe'], valid_e5['dNd'])
        xf = np.linspace(valid_e5['dRe'].min(), valid_e5['dRe'].max(), 100)
        ax.plot(xf, ic + sl*xf, color='#1565C0', lw=1.6, alpha=0.7)
    handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='#CCCCCC')
    stats_box(ax,
              f"Spearman r = {r_e5:+.3f}\n"
              f"p = {'<0.001' if p_e5<0.001 else f'{p_e5:.3f}'}")
    ax.set_xlabel('dRe = Re_37 − Re_21  (µm)', color=TXT, fontsize=10)
    ax.set_ylabel('dNd = Nd_21 − Nd_37  (cm⁻³)', color=TXT, fontsize=10)
else:
    col_missing(ax, 'dRe / dNd')
ax.set_title('E5 · dRe vs dNd\n(Spectral re difference drives Nd difference?)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

# ── E6 · Regime median summary bar chart ─────────────────────
ax = axes[5]
x = np.arange(len(REGIME_ORDER))
w = 0.25
re_cas_col = 're_cas_median' if 're_cas_median' in df.columns else None
re21_col   = 'Re_MODIS_21'   if 'Re_MODIS_21'  in df.columns else None
re37_col   = 'Re_MODIS_37'   if 'Re_MODIS_37'  in df.columns else None

if re_cas_col and re21_col and re37_col:
    med_recas = [df[df['drizzle_regime'] == r][re_cas_col].median() for r in REGIME_ORDER]
    med_re21  = [df[df['drizzle_regime'] == r][re21_col].median()   for r in REGIME_ORDER]
    med_re37  = [df[df['drizzle_regime'] == r][re37_col].median()   for r in REGIME_ORDER]

    b1 = ax.bar(x - w, med_recas, width=w, color='#37474F', alpha=0.85,
                edgecolor='white', label='re CAS (in-situ)')
    b2 = ax.bar(x,     med_re21,  width=w, color='#1565C0', alpha=0.85,
                edgecolor='white', label='Re MODIS 2.1 µm')
    b3 = ax.bar(x + w, med_re37,  width=w, color='#AD1457', alpha=0.85,
                edgecolor='white', label='Re MODIS 3.7 µm')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if pd.notna(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f'{h:.1f}', ha='center', va='bottom',
                        color=TXT, fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, color=TXT, fontsize=8.5)
    ax.set_ylabel('Median Re  (µm)', color=TXT, fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.9, facecolor='white', edgecolor='#CCCCCC')
else:
    ax.text(0.5, 0.5, 'Re columns not found\n(re_cas_median / Re_MODIS_21 / Re_MODIS_37)',
            transform=ax.transAxes, ha='center', va='center', color='red', fontsize=10)
ax.set_title('E6 · Median Re: In-Situ vs MODIS Channels\n(by drizzle regime)',
             color=TXT, fontsize=10, fontweight='bold', pad=8)
style_ax(ax)

fig.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package E: Spectral Sensitivity (2.1 µm vs 3.7 µm)\n'
    f'Do Both MODIS Channels Tell the Same Physical Story?',
    color=TXT, fontsize=13, fontweight='bold', y=1.005,
)
plt.tight_layout()
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ════════════════════════════════════════════════════════════════
# 5. SUPPLEMENTARY FIGURE — 2×2
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 12))
fig2.patch.set_facecolor('white')
axes2 = axes2.flatten()

def supp_scatter(ax, xcol, ycol, xlabel, ylabel, title,
                 ref_line_y=None, ref_line_x=None):
    if xcol not in df.columns or ycol not in df.columns:
        col_missing(ax, f'{xcol} / {ycol}')
        ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
        return
    valid = df[[xcol, ycol, 'drizzle_regime']].dropna()
    for reg in REGIME_ORDER:
        sub = valid[valid['drizzle_regime'] == reg]
        ax.scatter(sub[xcol], sub[ycol],
                   c=REGIME_COLORS[reg], s=65, zorder=4,
                   edgecolors='white', linewidths=0.5, alpha=0.90)
    if ref_line_y is not None:
        ax.axhline(ref_line_y, color='#333333', lw=1.2, ls='--', alpha=0.5)
    if ref_line_x is not None:
        ax.axvline(ref_line_x, color='#333333', lw=1.2, ls='--', alpha=0.5)
    if len(valid) >= 5:
        sl, ic, _, _, _ = linregress(valid[xcol], valid[ycol])
        xf = np.linspace(valid[xcol].min(), valid[xcol].max(), 100)
        ax.plot(xf, ic + sl*xf, color='#1565C0', lw=1.6, alpha=0.7)
    r, p = spearmanr(valid[xcol], valid[ycol])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    stats_box(ax,
              f"Spearman r = {r:+.3f}\n"
              f"p = {'<0.001' if p<0.001 else f'{p:.3f}'} {sig}")
    handles = [Patch(facecolor=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.9,
              facecolor='white', edgecolor='#CCCCCC', loc='upper left')
    ax.set_xlabel(xlabel, color=TXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TXT, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=10, fontweight='bold', pad=8)
    style_ax(ax)

# ── S1 · dNd vs f_ad ─────────────────────────────────────────
supp_scatter(axes2[0], 'f_ad_mean', 'dNd',
             'f_ad  (adiabaticity)',
             'dNd = Nd_21 − Nd_37  (cm⁻³)',
             'S1 · dNd vs Adiabaticity\n'
             '(thermodynamic control on spectral sensitivity?)',
             ref_line_y=0)

# ── S2 · dNd vs VZA ──────────────────────────────────────────
supp_scatter(axes2[1], 'VZA_mean', 'dNd',
             'VZA  (°)',
             'dNd = Nd_21 − Nd_37  (cm⁻³)',
             'S2 · dNd vs View Zenith Angle\n'
             '(geometry-spectral coupling?)',
             ref_line_y=0)

# ── S3 · dNd vs Nd_insitu ────────────────────────────────────
supp_scatter(axes2[2], 'Nd_median', 'dNd',
             'Nd in-situ  (cm⁻³)',
             'dNd = Nd_21 − Nd_37  (cm⁻³)',
             'S3 · dNd vs In-Situ Nd\n'
             '(spectral divergence in low-Nd clouds?)',
             ref_line_y=0)

# ── S4 · dRe vs tau_main ─────────────────────────────────────
supp_scatter(axes2[3], 'tau_main', 'dRe',
             'τ in-situ  (optical depth)',
             'dRe = Re_37 − Re_21  (µm)',
             'S4 · dRe vs Cloud Optical Depth\n'
             '(thin clouds show larger spectral Re gap?)',
             ref_line_y=0)

fig2.suptitle(
    f'{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package E: Supplementary — Spectral Sensitivity Drivers\n'
    f'What Controls the 2.1 µm vs 3.7 µm Divergence?',
    color=TXT, fontsize=13, fontweight='bold', y=1.005,
)
plt.tight_layout()
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

print("\n" + "=" * 65)
print("✓  Package E complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
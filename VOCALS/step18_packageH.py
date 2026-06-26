# =============================================================================
# step18_packageH.py — Package H: Clean/Polluted Classification and MODIS Bias
# =============================================================================
# Campaign : VOCALS-REx 2008
# Goal     : Is the clean/polluted classification related to MODIS Nd bias?
#
# CLASSIFICATION METHODS (median split):
#   Method A : Nd_median           - high Nd = polluted
#   Method B : k_median            - low k  = polluted (Martin 1994)
#   Method C : Nd_corrected        - Nd_median / f_ad_mean, high = polluted
#
# MAIN FIGURE (2x3):
#   H1: bias_21 boxplot - Clean vs Polluted (3 methods side-by-side)
#   H2: bias_21 vs Nd_median scatter (Method A colors)
#   H3: bias_21 vs k_median scatter  (Method B colors)
#   H4: Clean/Polluted (Method A) x drizzle_regime crosstab heatmap
#   H5: bias_21 by drizzle_regime x clean/polluted (Method A) grouped boxplot
#   H6: Nd_median vs k_median scatter (Method A colors)
#
# SUPPLEMENTARY (2x2):
#   HS1: bias_21 vs Nd_corrected scatter (Method C colors)
#   HS2: bias_21 vs f_ad_mean scatter    (Method A colors)
#   HS3: f_ad_mean vs drizzle_fraction   (Method A colors)
#   HS4: bias_21 distribution KDE - Clean vs Polluted (Method A)
#
# STATISTICS (terminal):
#   - Spearman: bias_21 vs Nd, k, f_ad, Nd_corrected
#   - Mann-Whitney U: Clean vs Polluted bias_21 (3 methods)
#   - Crosstab: Method A x drizzle_regime
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths (config-driven)
# =============================================================================
import config

NAME       = config.CAMPAIGN_NAME
FIG_DIR    = config.FIG_DIR

# Input CSV
MODIS_CSV  = config.STEP09_MODIS_MATCHES_CSV

# Output figure paths
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageH_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageH_supp.png'


# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv(MODIS_CSV)
df['match_status'] = df['match_status'].astype(str).str.strip().str.upper()
df = df[df['match_status'] == 'MATCHED'].copy().reset_index(drop=True)

# Backward-compat aliases: step10 outputs *_calc names; older Package H
# code expects bias_21 / Nd_MODIS_21. Default to *_calc.
for old, new in [
    ('Nd_MODIS_21', 'Nd_MODIS_21_calc'),
    ('Nd_MODIS_37', 'Nd_MODIS_37_calc'),
    ('bias_21',     'bias_21_calc'),
    ('bias_37',     'bias_37_calc'),
    ('dNd',         'dNd_calc'),
]:
    if old not in df.columns and new in df.columns:
        df[old] = df[new]

# Drop profiles with Nd_median <= 0 (bias = Nd_MODIS/0 = inf)
_n_before = len(df)
_zero_nd = df[df['Nd_median'] <= 0] if 'Nd_median' in df.columns else df.iloc[0:0]
if len(_zero_nd) > 0:
    print(f"\n[INFO] {len(_zero_nd)} profile(s) excluded (Nd_median <= 0):")
    for _, row in _zero_nd.iterrows():
        print(f"       {row['cloud_id']}: Nd_median={row['Nd_median']:.2f}")
    df = df[df['Nd_median'] > 0].copy().reset_index(drop=True)
    print(f"       Profiles used: {len(df)}/{_n_before}\n")

# Normalise drizzle_regime to snake_case
df['drizzle_regime'] = (
    df['drizzle_regime'].astype(str).str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('-', '_')
)
print(f"modis_matched: {len(df)} profiles")
print(df['drizzle_regime'].value_counts())

# ─── 2. CLASSIFICATION ────────────────────────────────────────────────────────
# Method A: Nd_median median split
nd_med        = df['Nd_median'].median()
df['class_A'] = np.where(df['Nd_median'] >= nd_med, 'Polluted', 'Clean')

# Method B: k_median median split (low k = polluted, Martin 1994)
k_med         = df['k_median'].median() if 'k_median' in df.columns else np.nan
if 'k_median' in df.columns:
    df['class_B'] = np.where(df['k_median'] < k_med, 'Polluted', 'Clean')
else:
    df['class_B'] = 'Unknown'
    print("WARNING: k_median not found — Method B will be empty")

# Method C: Nd_corrected = Nd_median / f_ad_mean median split
if 'f_ad_mean' in df.columns:
    df['Nd_corrected'] = df['Nd_median'] / df['f_ad_mean']
    ndc_med            = df['Nd_corrected'].median()
    df['class_C']      = np.where(df['Nd_corrected'] >= ndc_med, 'Polluted', 'Clean')
else:
    df['Nd_corrected'] = np.nan
    ndc_med            = np.nan
    df['class_C']      = 'Unknown'
    print("WARNING: f_ad_mean not found — Method C will be empty")

print(f"\nMedian thresholds:")
print(f"  Method A — Nd_median    : {nd_med:.1f} cm⁻³")
print(f"  Method B — k_median     : {k_med:.4f}" if pd.notna(k_med) else "  Method B — k_median : N/A")
print(f"  Method C — Nd_corrected : {ndc_med:.1f} cm⁻³" if pd.notna(ndc_med) else "  Method C — Nd_corrected : N/A")

print(f"\nClass counts:")
for m, col in [('A', 'class_A'), ('B', 'class_B'), ('C', 'class_C')]:
    print(f"  Method {m}: {df[col].value_counts().to_dict()}")

# ─── 3. STYLING ───────────────────────────────────────────────────────────────
CLASS_COLORS = {'Clean': '#1f77b4', 'Polluted': '#d62728'}
CLASS_ORDER  = ['Clean', 'Polluted']
REGIME_ORDER = ['non_drizzling', 'weak_drizzling', 'moderate_drizzling', 'heavy_drizzling']
REGIME_COLORS = {
    'non_drizzling':      '#2ca02c',
    'weak_drizzling':     '#ff7f0e',
    'moderate_drizzling': '#d62728',
    'heavy_drizzling':    '#8B0000',
}
REGIME_LABELS = {
    'non_drizzling':      'Non-drizzling',
    'weak_drizzling':     'Weak drizzling',
    'moderate_drizzling': 'Moderate drizzling',
    'heavy_drizzling':    'Heavy drizzling',
}

patches_class = [mpatches.Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER]

def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.set_facecolor('white')

def col_missing(ax, name):
    ax.text(0.5, 0.5, f'Column\n"{name}"\nnot found in CSV',
            transform=ax.transAxes, ha='center', va='center',
            color='red', fontsize=11)

# ─── 4. STATISTICS ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PAKET H — SPEARMAN CORRELATIONS WITH log(bias_21)")
print("="*65)
mask_b = df['bias_21'].notna() & (df['bias_21'] > 0)
for var, label in [
    ('Nd_median',    'Nd_median'),
    ('k_median',     'k_median'),
    ('f_ad_mean',    'f_ad_mean'),
    ('Nd_corrected', 'Nd_corrected (Nd/f_ad)'),
]:
    if var not in df.columns:
        print(f"  {label:<28} — column not found")
        continue
    mask = mask_b & df[var].notna()
    if mask.sum() < 5:
        print(f"  {label:<28} — n<5, skip")
        continue
    r, p = stats.spearmanr(df.loc[mask, var], np.log10(df.loc[mask, 'bias_21']))
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {label:<28} r={r:+.3f}, p={p:.4f} {stars}  (n={mask.sum()})")

print("\n" + "="*65)
print("PAKET H — MANN-WHITNEY U: Clean vs Polluted bias_21")
print("="*65)
for m, col in [('A (Nd_median)',    'class_A'),
               ('B (k_median)',     'class_B'),
               ('C (Nd_corrected)', 'class_C')]:
    b        = df.loc[mask_b, 'bias_21']
    grp      = df.loc[mask_b, col]
    clean_b    = b[grp == 'Clean'].values
    polluted_b = b[grp == 'Polluted'].values
    if len(clean_b) < 3 or len(polluted_b) < 3:
        print(f"  Method {m}: insufficient data")
        continue
    u, p = stats.mannwhitneyu(clean_b, polluted_b, alternative='two-sided')
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  Method {m}:")
    print(f"    Clean    median bias = {np.median(clean_b):.3f}×  (n={len(clean_b)})")
    print(f"    Polluted median bias = {np.median(polluted_b):.3f}×  (n={len(polluted_b)})")
    print(f"    Mann-Whitney U={u:.0f}, p={p:.4f} {stars}")

print("\n" + "="*65)
print("PAKET H — CROSSTAB: Method A x drizzle_regime")
print("="*65)
ct = pd.crosstab(df['class_A'], df['drizzle_regime'])
print(ct.to_string())

# ─── 5. HELPER: bias scatter coloured by class ────────────────────────────────
def bias_scatter_class(ax, df_in, xcol, class_col, xlabel, title,
                       threshold=None, xlog=False):
    if xcol not in df_in.columns:
        col_missing(ax, xcol)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    mask = df_in[xcol].notna() & df_in['bias_21'].notna() & (df_in['bias_21'] > 0)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, f'n={mask.sum()} — insufficient data',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    for cls in CLASS_ORDER:
        sub = df_in[mask & (df_in[class_col] == cls)]
        ax.scatter(sub[xcol], sub['bias_21'],
                   c=CLASS_COLORS[cls], s=60, alpha=0.85,
                   edgecolors='k', lw=0.4, label=cls, zorder=3)
    ax.axhline(1.0, color='k', ls='--', lw=1.2)
    ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if threshold is not None and pd.notna(threshold):
        ax.axvline(threshold, color='gray', ls=':', lw=1.5,
                   label=f'Median = {threshold:.2g}')
    r, p = stats.spearmanr(df_in.loc[mask, xcol],
                           np.log10(df_in.loc[mask, 'bias_21']))
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.text(0.05, 0.95,
            f'Spearman r = {r:.3f}\np = {p:.4f} {stars}\nn = {mask.sum()}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    style_ax(ax)

# ─── 6. MAIN FIGURE (2×3) ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.patch.set_facecolor('white')
fig.suptitle(
    "VOCALS-REx 2008 — Package H: Clean/Polluted Classification and MODIS Nd Bias\n"
    f"Three parallel classification methods (median split, n={len(df)})",
    fontsize=14, fontweight='bold', y=0.98
)

# ── H1: Boxplot — bias_21 Clean vs Polluted, 3 methods side by side ──────────
ax1 = axes[0, 0]
methods = [('A\n(Nd)', 'class_A'), ('B\n(k)', 'class_B'), ('C\n(Nd/f_ad)', 'class_C')]
x_pos   = np.array([0, 1, 2])
width   = 0.35
for j, cls in enumerate(CLASS_ORDER):
    offsets = x_pos + (j - 0.5) * width
    for i, (mlabel, mcol) in enumerate(methods):
        sub = df[df[mcol] == cls]['bias_21'].dropna()
        sub = sub[sub > 0]
        if len(sub) == 0:
            continue
        ax1.boxplot(sub, positions=[offsets[i]], widths=width * 0.85,
                    patch_artist=True, notch=False,
                    medianprops=dict(color='k', lw=2),
                    boxprops=dict(facecolor=CLASS_COLORS[cls], alpha=0.75),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker='o', ms=4,
                                    markerfacecolor=CLASS_COLORS[cls],
                                    alpha=0.6))
ax1.set_yscale('log')
ax1.axhline(1.0, color='k', ls='--', lw=1.2)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([m[0] for m in methods], fontsize=10)
ax1.set_xlabel('Classification Method', fontsize=10)
ax1.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
ax1.set_title('H1 · Bias: Clean vs Polluted\n(3 methods, median split)',
              fontsize=10, fontweight='bold')
ax1.legend(handles=patches_class, fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')
style_ax(ax1)

# ── H2: bias_21 vs Nd_median (Method A colour) ───────────────────────────────
bias_scatter_class(axes[0, 1], df, 'Nd_median', 'class_A',
                   'Nd_median (cm⁻³)',
                   'H2 · Bias vs Nd_median\n(color = Method A)',
                   threshold=nd_med)

# ── H3: bias_21 vs k_median (Method B colour) ────────────────────────────────
bias_scatter_class(axes[0, 2], df, 'k_median', 'class_B',
                   'k_median',
                   'H3 · Bias vs k_median\n(color = Method B)',
                   threshold=k_med if pd.notna(k_med) else None)

# ── H4: Crosstab heatmap — Method A x drizzle_regime ─────────────────────────
ax4 = axes[1, 0]
ct_norm     = ct.div(ct.sum(axis=1), axis=0)
regime_cols = [r for r in REGIME_ORDER if r in ct_norm.columns]
if len(regime_cols) > 0:
    ct_plot = ct_norm[regime_cols]
    im = ax4.imshow(ct_plot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(regime_cols)))
    ax4.set_xticklabels([REGIME_LABELS[r].replace(' ', '\n') for r in regime_cols],
                        fontsize=8)
    ax4.set_yticks(range(len(CLASS_ORDER)))
    ax4.set_yticklabels(CLASS_ORDER, fontsize=10)
    for i in range(len(CLASS_ORDER)):
        for j_idx, reg in enumerate(regime_cols):
            val_n  = ct_plot.values[i, j_idx]
            val_ct = ct.loc[CLASS_ORDER[i], reg] if reg in ct.columns else 0
            ax4.text(j_idx, i, f'{val_ct}\n({val_n:.0%})',
                     ha='center', va='center', fontsize=9,
                     color='white' if val_n > 0.5 else 'black')
    plt.colorbar(im, ax=ax4, label='Fraction within class')
else:
    ax4.text(0.5, 0.5, 'No matching regime columns', transform=ax4.transAxes,
             ha='center', va='center', color='red')
ax4.set_title('H4 · Clean/Polluted × Drizzle Regime\n(Method A, row-normalized)',
              fontsize=10, fontweight='bold')

# ── H5: Grouped boxplot — bias_21 by regime x clean/polluted (Method A) ──────
ax5 = axes[1, 1]
regimes_present = [r for r in REGIME_ORDER if r in df['drizzle_regime'].values]
x_reg = np.arange(len(regimes_present))
w     = 0.38
for j, cls in enumerate(CLASS_ORDER):
    off     = (j - 0.5) * w
    sub_cls = df[df['class_A'] == cls]
    for i, reg in enumerate(regimes_present):
        vals = sub_cls[sub_cls['drizzle_regime'] == reg]['bias_21'].dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        ax5.boxplot(vals, positions=[x_reg[i] + off], widths=w * 0.85,
                    patch_artist=True, notch=False,
                    medianprops=dict(color='k', lw=2),
                    boxprops=dict(facecolor=CLASS_COLORS[cls], alpha=0.75),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker='o', ms=4,
                                    markerfacecolor=CLASS_COLORS[cls],
                                    alpha=0.6))
ax5.set_yscale('log')
ax5.axhline(1.0, color='k', ls='--', lw=1.2)
ax5.set_xticks(x_reg)
ax5.set_xticklabels([REGIME_LABELS[r].replace(' ', '\n') for r in regimes_present],
                    fontsize=8)
ax5.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
ax5.set_title('H5 · Bias by Drizzle Regime × Clean/Polluted\n(Method A)',
              fontsize=10, fontweight='bold')
ax5.legend(handles=patches_class, fontsize=9, loc='upper right')
ax5.grid(True, alpha=0.3, axis='y')
style_ax(ax5)

# ── H6: Nd_median vs k_median scatter (Method A colour) ──────────────────────
ax6 = axes[1, 2]
if 'k_median' in df.columns:
    for cls in CLASS_ORDER:
        sub = df[df['class_A'] == cls]
        ax6.scatter(sub['k_median'], sub['Nd_median'],
                    c=CLASS_COLORS[cls], s=60, alpha=0.85,
                    edgecolors='k', lw=0.4, label=cls, zorder=3)
    if pd.notna(k_med):
        ax6.axvline(k_med,  color='gray', ls=':', lw=1.5, label=f'k median = {k_med:.3f}')
    ax6.axhline(nd_med, color='gray', ls='--', lw=1.5, label=f'Nd median = {nd_med:.0f}')
    valid_k = df['k_median'].notna()
    r6, p6  = stats.spearmanr(df.loc[valid_k, 'k_median'],
                               df.loc[valid_k, 'Nd_median'])
    stars6  = '***' if p6<0.001 else '**' if p6<0.01 else '*' if p6<0.05 else 'ns'
    ax6.text(0.05, 0.95,
             f'Spearman r = {r6:.3f}\np = {p6:.4f} {stars6}',
             transform=ax6.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax6.legend(fontsize=8, loc='upper right')
else:
    col_missing(ax6, 'k_median')
ax6.set_xlabel('k_median', fontsize=10)
ax6.set_ylabel('Nd_median (cm⁻³)', fontsize=10)
ax6.set_title('H6 · Nd_median vs k_median\n(color = Method A)',
              fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3)
style_ax(ax6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ─── 7. SUPPLEMENTARY FIGURE (2×2) ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.patch.set_facecolor('white')
fig2.suptitle(
    "VOCALS-REx 2008 — Package H: Supplementary — Clean/Polluted Additional Diagnostics",
    fontsize=13, fontweight='bold', y=0.98
)

# ── HS1: bias_21 vs Nd_corrected (Method C colour) ───────────────────────────
bias_scatter_class(axes2[0, 0], df, 'Nd_corrected', 'class_C',
                   'Nd_corrected = Nd_median / f_ad_mean (cm⁻³)',
                   'HS1 · Bias vs Nd_corrected\n(color = Method C)',
                   threshold=ndc_med if pd.notna(ndc_med) else None)

# ── HS2: bias_21 vs f_ad_mean (Method A colour) ──────────────────────────────
bias_scatter_class(axes2[0, 1], df, 'f_ad_mean', 'class_A',
                   'f_ad_mean',
                   'HS2 · Bias vs f_ad_mean\n(color = Method A)')

# ── HS3: f_ad_mean vs drizzle_fraction (Method A colour) ─────────────────────
ax_hs3 = axes2[1, 0]
if 'f_ad_mean' in df.columns and 'drizzle_fraction' in df.columns:
    for cls in CLASS_ORDER:
        sub = df[df['class_A'] == cls]
        ax_hs3.scatter(sub['drizzle_fraction'], sub['f_ad_mean'],
                       c=CLASS_COLORS[cls], s=60, alpha=0.85,
                       edgecolors='k', lw=0.4, label=cls, zorder=3)
    valid_hs3 = df['drizzle_fraction'].notna() & df['f_ad_mean'].notna()
    r_hs3, p_hs3 = stats.spearmanr(df.loc[valid_hs3, 'drizzle_fraction'],
                                    df.loc[valid_hs3, 'f_ad_mean'])
    stars_hs3 = '***' if p_hs3<0.001 else '**' if p_hs3<0.01 else '*' if p_hs3<0.05 else 'ns'
    ax_hs3.text(0.05, 0.95,
                f'Spearman r = {r_hs3:.3f}\np = {p_hs3:.4f} {stars_hs3}',
                transform=ax_hs3.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax_hs3.legend(fontsize=8, loc='upper right')
    ax_hs3.set_xlabel('Drizzle fraction', fontsize=10)
    ax_hs3.set_ylabel('f_ad_mean', fontsize=10)
    ax_hs3.grid(True, alpha=0.3)
else:
    col_missing(ax_hs3, 'f_ad_mean / drizzle_fraction')
ax_hs3.set_title('HS3 · f_ad_mean vs Drizzle Fraction\n(color = Method A)',
                 fontsize=10, fontweight='bold')
style_ax(ax_hs3)

# ── HS4: KDE distribution — bias_21 Clean vs Polluted (Method A) ─────────────
ax_hs4 = axes2[1, 1]
for cls in CLASS_ORDER:
    vals = df[(df['class_A'] == cls) & df['bias_21'].notna() & (df['bias_21'] > 0)]['bias_21']
    if len(vals) < 3:
        continue
    log_vals = np.log10(vals)
    kde      = gaussian_kde(log_vals, bw_method=0.4)
    x_range  = np.linspace(-1, 2, 300)
    ax_hs4.fill_between(x_range, kde(x_range),
                        alpha=0.35, color=CLASS_COLORS[cls],
                        label=f'{cls} (n={len(vals)})')
    ax_hs4.plot(x_range, kde(x_range), color=CLASS_COLORS[cls], lw=2)
    ax_hs4.axvline(np.median(log_vals), color=CLASS_COLORS[cls],
                   ls='--', lw=1.5, alpha=0.8)
ax_hs4.axvline(0, color='k', ls='--', lw=1.2, label='Bias = 1')
ax_hs4.set_xlabel('log₁₀(Bias)', fontsize=10)
ax_hs4.set_ylabel('Density', fontsize=10)
ax_hs4.set_title('HS4 · Bias Distribution: Clean vs Polluted\n(Method A, KDE)',
                 fontsize=10, fontweight='bold')
ax_hs4.legend(fontsize=9)
ax_hs4.grid(True, alpha=0.3)
xt = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
ax_hs4.set_xticks(xt)
ax_hs4.set_xticklabels([f'{10**v:.2g}×' for v in xt], fontsize=8)
style_ax(ax_hs4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

print("\n" + "="*65)
print("✓  Package H complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
print("="*65)
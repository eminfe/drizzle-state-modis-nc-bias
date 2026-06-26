# =============================================================================
# step16_packageF.py — Package F: Assumption Sensitivity Analysis
# =============================================================================
# Campaign : MASE 2005
# Goal     : How sensitive is the MODIS Nd retrieval bias to the assumed
#            values of k, f_ad, and c_w?
#            Compare bias_calc (in-situ measured) vs bias_lit (literature)
#            and isolate which assumption matters most via three scenarios:
#              S1 : k literature   + f_ad/c_w in-situ
#              S2 : f_ad literature + k/c_w in-situ
#              S3 : c_w in-situ only + k/f_ad literature
# Data     : {CAMPAIGN}_MODIS_Matches.csv  ->  MATCHED profiles
# Outputs  : {CAMPAIGN}_PackageF_main.png  (2x3 panels)
#            {CAMPAIGN}_PackageF_supp.png  (2x2 panels)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Paths (config-driven)
# =============================================================================
import config
from utils import calc_nd_grosvenor

NAME       = config.CAMPAIGN_NAME
FIG_DIR    = config.FIG_DIR

# Input CSV (output from step09/step10)
MODIS_CSV  = config.STEP09_MODIS_MATCHES_CSV

# Output figure paths
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageF_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageF_supp.png'


# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(MODIS_CSV)

# Normalise drizzle_regime to snake_case for color lookup
df['drizzle_regime'] = (
    df['drizzle_regime'].astype(str).str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('-', '_')
)

df['match_status'] = df['match_status'].astype(str).str.strip().str.upper()
df = df[df['match_status'] == 'MATCHED'].copy().reset_index(drop=True)

# Backward-compat aliases: step10 outputs *_calc names; older Package F
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
print(f"MATCHED profiles: {len(df)}")

# Drop profiles with Nd_median <= 0 (bias = Nd_MODIS / 0 = inf -> axis errors)
_n_before = len(df)
_zero_nd = df[df['Nd_median'] <= 0]
if len(_zero_nd) > 0:
    print(f"\n[INFO] {len(_zero_nd)} profile(s) excluded (Nd_median <= 0):")
    for _, row in _zero_nd.iterrows():
        print(f"       {row['cloud_id']}: Nd_median={row['Nd_median']:.2f}")
    df = df[df['Nd_median'] > 0].copy().reset_index(drop=True)
    print(f"       Profiles used: {len(df)}/{_n_before}\n")

print(df['drizzle_regime'].value_counts())

# =============================================================================
# 2. Literature defaults (from config.NDLIT)
# =============================================================================
K_LIT    = config.NDLIT["k_lit"]      # 0.67  Martin 1994 marine
F_AD_LIT = config.NDLIT["f_ad_lit"]   # 0.80  operational default
C_W_LIT  = config.NDLIT["c_w_lit"]    # 3.51e-3 g/m^4

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
REGIME_ORDER = ['non_drizzling', 'weak_drizzling', 'moderate_drizzling', 'heavy_drizzling']

# =============================================================================
# 3. bias_calc - already computed by step10
# =============================================================================
df['bias_21'] = df['Nd_MODIS_21'] / df['Nd_median']

# =============================================================================
# 4. SCENARIO CALCULATIONS - use utils.calc_nd_grosvenor (single source of truth)
# =============================================================================
tau_21 = df['tau_MODIS_21'].values
Re_21  = df['Re_MODIS_21'].values
f_ad_v = df['f_ad_mean'].values
k_v    = df['k_median'].values
cw_v   = df['c_w_median'].values

# bias_lit: all parameters from literature
df['Nd_lit_21'] = [calc_nd_grosvenor(Re_21[i], tau_21[i], K_LIT, F_AD_LIT, C_W_LIT)
                   for i in range(len(df))]
df['bias_lit']  = df['Nd_lit_21'] / df['Nd_median']

# S1: only k literature, f_ad and c_w from in-situ
df['Nd_S1_21'] = [calc_nd_grosvenor(Re_21[i], tau_21[i], K_LIT, f_ad_v[i], cw_v[i])
                  for i in range(len(df))]
df['bias_S1']  = df['Nd_S1_21'] / df['Nd_median']

# S2: only f_ad literature, k and c_w from in-situ
df['Nd_S2_21'] = [calc_nd_grosvenor(Re_21[i], tau_21[i], k_v[i], F_AD_LIT, cw_v[i])
                  for i in range(len(df))]
df['bias_S2']  = df['Nd_S2_21'] / df['Nd_median']

# S3: k + f_ad literature, c_w from in-situ
df['Nd_S3_21'] = [calc_nd_grosvenor(Re_21[i], tau_21[i], K_LIT, F_AD_LIT, cw_v[i])
                  for i in range(len(df))]
df['bias_S3']  = df['Nd_S3_21'] / df['Nd_median']

# =============================================================================
# 5. PRINT SUMMARY
# =============================================================================
print("\n" + "="*65)
print("PACKAGE F — BIAS SUMMARY")
print("="*65)
for sc, label in [('bias_21',  'bias_calc (measured k,fad,cw)'),
                  ('bias_lit', 'bias_lit  (fixed k,fad,cw)'),
                  ('bias_S1',  'bias_S1   (fixed k only)'),
                  ('bias_S2',  'bias_S2   (fixed fad only)'),
                  ('bias_S3',  'bias_S3   (fixed k+fad)')]:
    v = df[sc].dropna()
    print(f"  {label:<35} median={v.median():.3f}x  mean={v.mean():.3f}x")

print("\nSpearman correlations with log(bias):")
for param, plabel in [('k_median', 'k_median'),
                      ('f_ad_mean', 'f_ad_mean'),
                      ('c_w_median', 'c_w_median')]:
    if param not in df.columns:
        print(f"  {param} → NOT FOUND in CSV")
        continue
    for sc, slabel in [('bias_21', 'bias_calc'), ('bias_lit', 'bias_lit')]:
        mask = df[param].notna() & df[sc].notna() & (df[sc] > 0)
        if mask.sum() < 5:
            print(f"  log({slabel}) vs {plabel}: n<5, skip")
            continue
        r, p = stats.spearmanr(df.loc[mask, param], np.log10(df.loc[mask, sc]))
        stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  log({slabel}) vs {plabel}: r={r:.3f}, p={p:.4f} {stars}")

# ─── 7. SETUP COLORS ──────────────────────────────────────────────────────────
# Fallback: any unrecognised regime gets gray
colors_pt = [REGIME_COLORS.get(r, '#888888') for r in df['drizzle_regime']]
patches_leg = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r])
               for r in REGIME_ORDER if r in df['drizzle_regime'].values]

# ─── 8. MAIN FIGURE (2×3) ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package F: Assumption Sensitivity Analysis\n"
    f"How Sensitive is MODIS Nd Bias to f_ad, k, c_w Choice?",
    fontsize=14, fontweight='bold', y=0.98
)

def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.set_facecolor('white')

def col_missing(ax, name):
    ax.text(0.5, 0.5, f'Column\n"{name}"\nnot found in CSV',
            transform=ax.transAxes, ha='center', va='center',
            color='red', fontsize=11)

# ── F1: bias_21 vs Nd_insitu ──────────────────────────────────────────────────
ax = axes[0, 0]
mask = (df['Nd_median'] > 0) & (df['bias_21'] > 0)
ax.scatter(df.loc[mask, 'Nd_median'], df.loc[mask, 'bias_21'],
           c=[colors_pt[i] for i in df.index[mask]], s=55, alpha=0.8,
           edgecolors='k', lw=0.4)
ax.axhline(1.0, color='k', ls='--', lw=1.2, label='No bias (=1)')
ax.set_xscale('log'); ax.set_yscale('log')
r1, p1 = stats.spearmanr(np.log10(df.loc[mask, 'Nd_median']),
                          np.log10(df.loc[mask, 'bias_21']))
med1 = df.loc[mask, 'bias_21'].median()
ax.text(0.05, 0.95,
        f'Spearman r = {r1:.3f}\np = {p1:.4f}\nMedian bias = {med1:.2f}×',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax.set_xlabel('Nd in-situ (cm⁻³)', fontsize=10)
ax.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
ax.set_title('F1 · bias_calc vs Nd in-situ\n(measured k, f_ad, c_w)',
             fontsize=10, fontweight='bold')
ax.legend(handles=patches_leg, fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── F2: bias_lit vs Nd_insitu ─────────────────────────────────────────────────
ax = axes[0, 1]
mask2 = (df['Nd_median'] > 0) & (df['bias_lit'] > 0)
ax.scatter(df.loc[mask2, 'Nd_median'], df.loc[mask2, 'bias_lit'],
           c=[colors_pt[i] for i in df.index[mask2]], s=55, alpha=0.8,
           edgecolors='k', lw=0.4)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.set_xscale('log'); ax.set_yscale('log')
r2, p2 = stats.spearmanr(np.log10(df.loc[mask2, 'Nd_median']),
                          np.log10(df.loc[mask2, 'bias_lit']))
med2 = df.loc[mask2, 'bias_lit'].median()
ax.text(0.05, 0.95,
        f'Spearman r = {r2:.3f}\np = {p2:.4f}\nMedian bias = {med2:.2f}×',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax.set_xlabel('Nd in-situ (cm⁻³)', fontsize=10)
ax.set_ylabel('Bias = Nd_MODIS_lit / Nd_insitu', fontsize=10)
ax.set_title('F2 · bias_lit vs Nd in-situ\n(fixed k=0.67, f_ad=0.80, c_w=lit)',
             fontsize=10, fontweight='bold')
ax.legend(handles=patches_leg, fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── F3: bias_21 vs bias_lit scatter ──────────────────────────────────────────
ax = axes[0, 2]
mask3 = (df['bias_21'] > 0) & (df['bias_lit'] > 0)
ax.scatter(df.loc[mask3, 'bias_21'], df.loc[mask3, 'bias_lit'],
           c=[colors_pt[i] for i in df.index[mask3]], s=55, alpha=0.8,
           edgecolors='k', lw=0.4)
lims3 = [min(df.loc[mask3, 'bias_21'].min(), df.loc[mask3, 'bias_lit'].min()) * 0.8,
         max(df.loc[mask3, 'bias_21'].max(), df.loc[mask3, 'bias_lit'].max()) * 1.2]
ax.plot(lims3, lims3, 'k--', lw=1.2, label='1:1 line')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims3); ax.set_ylim(lims3)
r3, p3 = stats.spearmanr(np.log10(df.loc[mask3, 'bias_21']),
                          np.log10(df.loc[mask3, 'bias_lit']))
sl3, ic3, _, _, _ = stats.linregress(np.log10(df.loc[mask3, 'bias_21']),
                                      np.log10(df.loc[mask3, 'bias_lit']))
ax.text(0.05, 0.95,
        f'Spearman r = {r3:.3f}\nOLS slope = {sl3:.3f}\nn = {mask3.sum()}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
ax.set_xlabel('bias_calc (measured k, f_ad, c_w)', fontsize=10)
ax.set_ylabel('bias_lit (fixed k, f_ad, c_w)', fontsize=10)
ax.set_title('F3 · bias_calc vs bias_lit\n(do assumptions change the bias?)',
             fontsize=10, fontweight='bold')
ax.legend(handles=patches_leg + [plt.Line2D([0],[0], color='k', ls='--', label='1:1')],
          fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── F4: bias_21 & bias_lit vs k_median ───────────────────────────────────────
ax = axes[1, 0]
if 'k_median' in df.columns:
    mask4  = df['k_median'].notna() & df['bias_21'].notna() & (df['bias_21'] > 0)
    mask4b = df['k_median'].notna() & df['bias_lit'].notna() & (df['bias_lit'] > 0)
    ax.scatter(df.loc[mask4,  'k_median'], df.loc[mask4,  'bias_21'],
               c=[colors_pt[i] for i in df.index[mask4]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='o', label='bias_calc')
    ax.scatter(df.loc[mask4b, 'k_median'], df.loc[mask4b, 'bias_lit'],
               c=[colors_pt[i] for i in df.index[mask4b]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='^', label='bias_lit')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.axvline(K_LIT, color='steelblue', ls=':', lw=1.5, label=f'k_lit={K_LIT}')
    ax.set_yscale('log')
    r4c, p4c = stats.spearmanr(df.loc[mask4,  'k_median'], np.log10(df.loc[mask4,  'bias_21']))
    r4l, p4l = stats.spearmanr(df.loc[mask4b, 'k_median'], np.log10(df.loc[mask4b, 'bias_lit']))
    st4c = '***' if p4c<0.001 else '**' if p4c<0.01 else '*' if p4c<0.05 else 'ns'
    st4l = '***' if p4l<0.001 else '**' if p4l<0.01 else '*' if p4l<0.05 else 'ns'
    ax.text(0.05, 0.95,
            f'bias_calc:  r={r4c:.3f} {st4c}\nbias_lit:     r={r4l:.3f} {st4l}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel('k_median (profile-measured)', fontsize=10)
    ax.legend(handles=patches_leg + [
        plt.Line2D([0],[0], color='steelblue', ls=':', label=f'k_lit={K_LIT}')],
        fontsize=7.5, loc='upper right')
else:
    col_missing(ax, 'k_median')
ax.set_ylabel('Bias (log scale)', fontsize=10)
ax.set_title('F4 · Bias vs k\n(○ = bias_calc, △ = bias_lit)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── F5: bias_21 & bias_lit vs f_ad_mean ──────────────────────────────────────
ax = axes[1, 1]
if 'f_ad_mean' in df.columns:
    mask5  = df['f_ad_mean'].notna() & df['bias_21'].notna() & (df['bias_21'] > 0)
    mask5b = df['f_ad_mean'].notna() & df['bias_lit'].notna() & (df['bias_lit'] > 0)
    ax.scatter(df.loc[mask5,  'f_ad_mean'], df.loc[mask5,  'bias_21'],
               c=[colors_pt[i] for i in df.index[mask5]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='o')
    ax.scatter(df.loc[mask5b, 'f_ad_mean'], df.loc[mask5b, 'bias_lit'],
               c=[colors_pt[i] for i in df.index[mask5b]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='^')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.axvline(F_AD_LIT, color='steelblue', ls=':', lw=1.5, label=f'f_ad_lit={F_AD_LIT}')
    ax.set_yscale('log')
    r5c, p5c = stats.spearmanr(df.loc[mask5,  'f_ad_mean'], np.log10(df.loc[mask5,  'bias_21']))
    r5l, p5l = stats.spearmanr(df.loc[mask5b, 'f_ad_mean'], np.log10(df.loc[mask5b, 'bias_lit']))
    st5c = '***' if p5c<0.001 else '**' if p5c<0.01 else '*' if p5c<0.05 else 'ns'
    st5l = '***' if p5l<0.001 else '**' if p5l<0.01 else '*' if p5l<0.05 else 'ns'
    ax.text(0.05, 0.95,
            f'bias_calc:  r={r5c:.3f} {st5c}\nbias_lit:     r={r5l:.3f} {st5l}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel('f_ad_mean (profile-measured)', fontsize=10)
    ax.legend(handles=patches_leg + [
        plt.Line2D([0],[0], color='steelblue', ls=':', label=f'f_ad_lit={F_AD_LIT}')],
        fontsize=7.5, loc='upper right')
else:
    col_missing(ax, 'f_ad_mean')
ax.set_ylabel('Bias (log scale)', fontsize=10)
ax.set_title('F5 · Bias vs f_ad\n(○ = bias_calc, △ = bias_lit)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── F6: bias_21 & bias_lit vs c_w_median ─────────────────────────────────────
ax = axes[1, 2]
if 'c_w_median' in df.columns:
    mask6  = df['c_w_median'].notna() & df['bias_21'].notna() & (df['bias_21'] > 0)
    mask6b = df['c_w_median'].notna() & df['bias_lit'].notna() & (df['bias_lit'] > 0)
    ax.scatter(df.loc[mask6,  'c_w_median'], df.loc[mask6,  'bias_21'],
               c=[colors_pt[i] for i in df.index[mask6]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='o')
    ax.scatter(df.loc[mask6b, 'c_w_median'], df.loc[mask6b, 'bias_lit'],
               c=[colors_pt[i] for i in df.index[mask6b]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='^')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.axvline(C_W_LIT, color='gray', ls=':', lw=1.5, label=f'c_w_lit={C_W_LIT:.4f}')
    ax.set_yscale('log')
    r6c, p6c = stats.spearmanr(df.loc[mask6,  'c_w_median'], np.log10(df.loc[mask6,  'bias_21']))
    r6l, p6l = stats.spearmanr(df.loc[mask6b, 'c_w_median'], np.log10(df.loc[mask6b, 'bias_lit']))
    st6c = '***' if p6c<0.001 else '**' if p6c<0.01 else '*' if p6c<0.05 else 'ns'
    st6l = '***' if p6l<0.001 else '**' if p6l<0.01 else '*' if p6l<0.05 else 'ns'
    ax.text(0.05, 0.95,
            f'bias_calc:  r={r6c:.3f} {st6c}\nbias_lit:     r={r6l:.3f} {st6l}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel('c_w_median (profile-measured, g m⁻⁴)', fontsize=10)
    ax.legend(handles=patches_leg + [
        plt.Line2D([0],[0], color='gray', ls=':', label=f'c_w_lit={C_W_LIT:.4f}')],
        fontsize=7.5, loc='upper right')
else:
    col_missing(ax, 'c_w_median')
ax.set_ylabel('Bias (log scale)', fontsize=10)
ax.set_title('F6 · Bias vs c_w\n(○ = bias_calc, △ = bias_lit)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
style_ax(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ─── 9. SUPPLEMENTARY FIGURE (2×2) ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.patch.set_facecolor('white')
fig2.suptitle(
    f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package F: Supplementary — Scenario Bias Breakdown\n"
    f"Isolating the Effect of Each Parameter",
    fontsize=13, fontweight='bold', y=0.98
)

sc_list   = ['bias_21', 'bias_lit', 'bias_S1', 'bias_S2', 'bias_S3']
sc_labels = ['bias_calc\n(meas. k,fad,cw)',
             'bias_lit\n(fixed k,fad,cw)',
             'S1\n(fixed k only)',
             'S2\n(fixed fad only)',
             'S3\n(fixed k+fad)']
sc_colors = ['#4878CF', '#D65F5F', '#6ACC65', '#B47CC7', '#F0A500']

# ── FS1: Bias boxplot — tum senaryolar ───────────────────────────────────────
ax = axes2[0, 0]
bias_data = [df[sc].dropna().values for sc in sc_list]
bp = ax.boxplot(bias_data, patch_artist=True, notch=False,
                medianprops=dict(color='white', lw=2.5),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker='o', ms=4, alpha=0.5))
for patch, col in zip(bp['boxes'], sc_colors):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.set_xticks(range(1, len(sc_list)+1))
ax.set_xticklabels(sc_labels, fontsize=8.5)
ax.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
ax.set_title('FS1 · Bias Distribution — All Scenarios\n(2.1 µm)',
             fontsize=10, fontweight='bold')
for i, sc in enumerate(sc_list):
    med = df[sc].median()
    ax.text(i+1, med + 0.05, f'{med:.2f}×', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
style_ax(ax)

# ── FS2: Bias by drizzle regime — bias_calc vs bias_lit ──────────────────────
ax = axes2[0, 1]
x = np.arange(len(REGIME_ORDER))
w = 0.35
for j, (sc, col, lbl) in enumerate([('bias_21',  '#4878CF', 'bias_calc'),
                                      ('bias_lit', '#D65F5F', 'bias_lit')]):
    meds = [df.loc[df['drizzle_regime'] == r, sc].median() for r in REGIME_ORDER]
    ax.bar(x + j*w - w/2, meds, w, color=col, alpha=0.8,
           label=lbl, edgecolor='k', lw=0.5)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([REGIME_LABELS[r].replace(' ', '\n') for r in REGIME_ORDER], fontsize=9)
ax.set_ylabel('Median Bias', fontsize=10)
ax.set_title('FS2 · Median Bias by Drizzle Regime\nbias_calc vs bias_lit',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
style_ax(ax)

# ── FS3: Effect of k — bias_calc vs bias_S1 ──────────────────────────────────
ax = axes2[1, 0]
if 'k_median' in df.columns:
    mask_s1 = df['k_median'].notna() & df['bias_21'].notna() & df['bias_S1'].notna()
    ax.scatter(df.loc[mask_s1, 'k_median'], df.loc[mask_s1, 'bias_21'],
               c=[colors_pt[i] for i in df.index[mask_s1]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='o',
               label='bias_calc (meas. k)')
    ax.scatter(df.loc[mask_s1, 'k_median'], df.loc[mask_s1, 'bias_S1'],
               c=[colors_pt[i] for i in df.index[mask_s1]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='^',
               label='S1 (fixed k=0.67)')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.axvline(K_LIT, color='steelblue', ls=':', lw=1.5)
    ax.set_yscale('log')
    r_s1, p_s1 = stats.spearmanr(df.loc[mask_s1, 'k_median'],
                                   np.log10(df.loc[mask_s1, 'bias_21']))
    ax.text(0.05, 0.95, f'bias_calc vs k: r={r_s1:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel('k_median (profile-measured)', fontsize=10)
    ax.legend(handles=patches_leg + [
        plt.Line2D([0],[0], marker='o', color='gray', ls='None', label='bias_calc'),
        plt.Line2D([0],[0], marker='^', color='gray', ls='None', label='S1 (k fixed)')],
        fontsize=7.5, loc='upper right')
else:
    col_missing(ax, 'k_median')
ax.set_ylabel('Bias (log scale)', fontsize=10)
ax.set_title('FS3 · Effect of k\n(bias_calc vs S1: only k fixed)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
style_ax(ax)

# ── FS4: Effect of f_ad — bias_calc vs bias_S2 ───────────────────────────────
ax = axes2[1, 1]
if 'f_ad_mean' in df.columns:
    mask_s2 = df['f_ad_mean'].notna() & df['bias_21'].notna() & df['bias_S2'].notna()
    ax.scatter(df.loc[mask_s2, 'f_ad_mean'], df.loc[mask_s2, 'bias_21'],
               c=[colors_pt[i] for i in df.index[mask_s2]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='o',
               label='bias_calc (meas. f_ad)')
    ax.scatter(df.loc[mask_s2, 'f_ad_mean'], df.loc[mask_s2, 'bias_S2'],
               c=[colors_pt[i] for i in df.index[mask_s2]],
               s=55, alpha=0.85, edgecolors='k', lw=0.4, marker='^',
               label='S2 (fixed f_ad=0.80)')
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.axvline(F_AD_LIT, color='steelblue', ls=':', lw=1.5)
    ax.set_yscale('log')
    r_s2, p_s2 = stats.spearmanr(df.loc[mask_s2, 'f_ad_mean'],
                                   np.log10(df.loc[mask_s2, 'bias_21']))
    ax.text(0.05, 0.95, f'bias_calc vs f_ad: r={r_s2:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel('f_ad_mean (profile-measured)', fontsize=10)
    ax.legend(handles=patches_leg + [
        plt.Line2D([0],[0], marker='o', color='gray', ls='None', label='bias_calc'),
        plt.Line2D([0],[0], marker='^', color='gray', ls='None', label='S2 (f_ad fixed)')],
        fontsize=7.5, loc='upper right')
else:
    col_missing(ax, 'f_ad_mean')
ax.set_ylabel('Bias (log scale)', fontsize=10)
ax.set_title('FS4 · Effect of f_ad\n(bias_calc vs S2: only f_ad fixed)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
style_ax(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

print("\n" + "="*65)
print("✓  Package F complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
print("="*65)
# =============================================================================
# step17_packageG.py — Package G: Vertical Structure and Altitude
# =============================================================================
# Campaign : MASE 2005
# Goal     : Are cloud altitude and vertical structure related to MODIS bias?
#
# DATA:
#   modis_matched (MATCHED profiles) -> bias_21, z_base_m, z_top_m, cloud_depth,
#                                       mean_z_drizzle (0-1 normalized),
#                                       min_z_drizzle, drizzle_fraction
#   gm  (per-second in-situ)         -> z_norm, Nc_CAS, re_cas, f_ad,
#                                       drizzle_flag, drizzle_regime, cloud_id
#
# MAIN FIGURE (2x3):
#   G1: bias_21 vs z_top_m
#   G2: bias_21 vs cloud_depth
#   G3: bias_21 vs mean_z_drizzle  (0=cloud-base, 1=cloud-top)
#   G4: vertical profile z_norm vs Nd (Nc_CAS)  - by regime
#   G5: vertical profile z_norm vs re_cas        - by regime
#   G6: vertical profile z_norm vs f_ad          - by regime
#
# SUPPLEMENTARY (2x2):
#   GS1: bias_21 vs z_base_m
#   GS2: bias_21 vs min_z_drizzle  (normalized 0-1)
#   GS3: bias_21 vs drizzle_fraction
#   GS4: Vertical profile z_norm vs drizzle_flag - by regime
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
from pathlib import Path

NAME       = config.CAMPAIGN_NAME
FIG_DIR    = config.FIG_DIR

# Input CSVs
MODIS_CSV  = config.STEP09_MODIS_MATCHES_CSV
GM_CSV     = config.STEP04_GOLDEN_MICRO_CSV   # per-second in-situ data

# Output figure paths
OUT_MAIN   = FIG_DIR / f'{NAME}_PackageG_main.png'
OUT_SUPP   = FIG_DIR / f'{NAME}_PackageG_supp.png'


# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv(MODIS_CSV)
df['match_status'] = df['match_status'].astype(str).str.strip().str.upper()
df = df[df['match_status'] == 'MATCHED'].copy().reset_index(drop=True)

# Backward-compat aliases: step10 outputs *_calc names; older Package G
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

# gm — per-second in-situ data (vertical profiles)
if not Path(GM_CSV).exists():
    raise FileNotFoundError(
        f"gm CSV not found: {GM_CSV}\n"
        f"Run step04 first to generate {config.STEP04_GOLDEN_MICRO_CSV.name}"
    )
gm = pd.read_csv(GM_CSV)
print(f"gm loaded from: {GM_CSV.name}  shape={gm.shape}")

# ─── 2. CONSTANTS & STYLING ───────────────────────────────────────────────────
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

colors_pt   = [REGIME_COLORS.get(r, '#888888') for r in df['drizzle_regime']]
patches_leg = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r])
               for r in REGIME_ORDER if r in df['drizzle_regime'].values]

# ─── 3. DRIZZLE HEIGHT CHECK ──────────────────────────────────────────────────
for col in ['mean_z_drizzle', 'min_z_drizzle']:
    if col in df.columns:
        print(f"{col} range: {df[col].min():.3f} – {df[col].max():.3f}")
    else:
        print(f"WARNING: {col} not found in MODIS CSV")

if 'mean_z_drizzle' in df.columns:
    print("\nmean_z_drizzle by regime (median):")
    print(df.groupby('drizzle_regime')['mean_z_drizzle'].median())

# ─── 4. SPEARMAN SUMMARY ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("PAKET G — SPEARMAN CORRELATIONS WITH log(bias_21)")
print("="*60)
mask_b = df['bias_21'].notna() & (df['bias_21'] > 0)
for var, label in [
    ('z_top_m',          'z_top_m'),
    ('z_base_m',         'z_base_m'),
    ('cloud_depth',      'cloud_depth'),
    ('mean_z_drizzle',   'mean_z_drizzle (norm 0-1)'),
    ('min_z_drizzle',    'min_z_drizzle  (norm 0-1)'),
    ('max_z_drizzle',    'max_z_drizzle  (norm 0-1)'),
    ('drizzle_fraction', 'drizzle_fraction'),
]:
    if var not in df.columns:
        print(f"  {label:<30} — column not found")
        continue
    mask = mask_b & df[var].notna()
    if mask.sum() < 5:
        print(f"  {label:<30} — insufficient data (n={mask.sum()})")
        continue
    r, p = stats.spearmanr(df.loc[mask, var], np.log10(df.loc[mask, 'bias_21']))
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {label:<30} r={r:+.3f}, p={p:.4f} {stars}  (n={mask.sum()})")

# ─── 5. GM MATCH — _merged SUFFIX STRIP ──────────────────────────────────────
if 'cloud_id' in gm.columns:
    gm['cloud_id_clean'] = gm['cloud_id'].astype(str).str.replace('_merged', '', regex=False)
else:
    # fallback: try profile_id
    id_col = next((c for c in ['profile_id', 'cloud_id', 'flight_id'] if c in gm.columns), None)
    if id_col:
        gm['cloud_id_clean'] = gm[id_col].astype(str).str.replace('_merged', '', regex=False)
    else:
        gm['cloud_id_clean'] = 'unknown'

matched_cloud_ids = set(df['cloud_id'].astype(str))
gm_m = gm[gm['cloud_id_clean'].isin(matched_cloud_ids)].copy()
n_prof = gm_m['cloud_id_clean'].nunique()
print(f"\ngm matched (after _merged strip): {len(gm_m)} rows, {n_prof} profiles")

# drizzle_regime from modis match
regime_map = df.set_index('cloud_id')['drizzle_regime'].to_dict()
gm_m['drizzle_regime'] = gm_m['cloud_id_clean'].map(regime_map)
print("gm_m drizzle_regime counts:")
print(gm_m['drizzle_regime'].value_counts())

# z_norm bins: 0–1, 20 bins
Z_BINS = np.linspace(0, 1, 21)
Z_MIDS = 0.5 * (Z_BINS[:-1] + Z_BINS[1:])

if 'z_norm' in gm_m.columns:
    gm_m['z_bin'] = pd.cut(gm_m['z_norm'], bins=Z_BINS, labels=False, include_lowest=True)
else:
    print("WARNING: z_norm not found in gm — vertical profiles will be empty")
    gm_m['z_norm'] = np.nan
    gm_m['z_bin']  = np.nan

# ─── 6. HELPERS ───────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAAAAA')
    ax.set_facecolor('white')

def col_missing(ax, name):
    ax.text(0.5, 0.5, f'Column\n"{name}"\nnot found in CSV',
            transform=ax.transAxes, ha='center', va='center',
            color='red', fontsize=11)

def vertical_profile(gm_data, var, regime_order, regime_colors, regime_labels,
                     ax, xlabel, title, vline=None):
    """Per-regime z_norm vs var median profile."""
    if var not in gm_data.columns:
        col_missing(ax, var)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    for reg in regime_order:
        sub = gm_data[gm_data['drizzle_regime'] == reg]
        if len(sub) == 0:
            continue
        prof   = sub.groupby('z_bin')[var].median()
        z_vals = Z_MIDS[prof.index.astype(int)]
        v_vals = prof.values
        valid  = ~np.isnan(v_vals)
        if valid.sum() == 0:
            continue
        ax.plot(v_vals[valid], z_vals[valid],
                color=regime_colors[reg], lw=2.0,
                label=regime_labels[reg], marker='o', ms=3)
    if vline is not None:
        ax.axvline(vline, color='k', ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Normalized height (z_norm)', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    style_ax(ax)


def bias_scatter(ax, df_in, xcol, xlabel, title, col_pts, leg_patches,
                 xlog=False, ref_line=None):
    if xcol not in df_in.columns:
        col_missing(ax, xcol)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    mask = df_in[xcol].notna() & df_in['bias_21'].notna() & (df_in['bias_21'] > 0)
    if mask.sum() < 5:
        ax.text(0.5, 0.5, f'n={mask.sum()} — insufficient data',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        return
    x = df_in.loc[mask, xcol]
    y = df_in.loc[mask, 'bias_21']
    c = [col_pts[i] for i in df_in.index[mask]]
    ax.scatter(x, y, c=c, s=55, alpha=0.85, edgecolors='k', lw=0.4)
    ax.axhline(1.0, color='k', ls='--', lw=1.2)
    ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if ref_line is not None:
        ax.axvline(ref_line, color='gray', ls=':', lw=1.2)
    r, p = stats.spearmanr(x, np.log10(y))
    stars = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.text(0.05, 0.95,
            f'Spearman r = {r:.3f}\np = {p:.4f} {stars}\nn = {mask.sum()}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(handles=leg_patches, fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    style_ax(ax)


# ─── 7. MAIN FIGURE (2×3) ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.patch.set_facecolor('white')
fig.suptitle(
    f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package G: Vertical Structure and Altitude\n"
    f"Is MODIS Nd Bias Related to Cloud Height and Vertical Structure?",
    fontsize=14, fontweight='bold', y=0.98
)

# G1: bias_21 vs z_top_m
bias_scatter(axes[0, 0], df, 'z_top_m',
             'Cloud-top altitude (m)',
             'G1 · Bias vs Cloud-top Altitude',
             colors_pt, patches_leg)

# G2: bias_21 vs cloud_depth
bias_scatter(axes[0, 1], df, 'cloud_depth',
             'Cloud depth (m)',
             'G2 · Bias vs Cloud Depth',
             colors_pt, patches_leg)

# G3: bias_21 vs mean_z_drizzle (already 0-1 normalized)
ax3 = axes[0, 2]
if 'mean_z_drizzle' in df.columns:
    mask3 = df['mean_z_drizzle'].notna() & df['bias_21'].notna() & (df['bias_21'] > 0)
    if mask3.sum() >= 5:
        x3 = df.loc[mask3, 'mean_z_drizzle']
        y3 = df.loc[mask3, 'bias_21']
        c3 = [colors_pt[i] for i in df.index[mask3]]
        ax3.scatter(x3, y3, c=c3, s=55, alpha=0.85, edgecolors='k', lw=0.4)
        ax3.axhline(1.0, color='k', ls='--', lw=1.2)
        ax3.set_yscale('log')
        r3, p3 = stats.spearmanr(x3, np.log10(y3))
        stars3 = '***' if p3<0.001 else '**' if p3<0.01 else '*' if p3<0.05 else 'ns'
        ax3.text(0.05, 0.95,
                 f'Spearman r = {r3:.3f}\np = {p3:.4f} {stars3}\nn = {mask3.sum()}\n'
                 f'0 = cloud-base  |  1 = cloud-top',
                 transform=ax3.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        ax3.set_xlabel('Mean drizzle height (normalized)\n(0 = cloud-base, 1 = cloud-top)',
                       fontsize=10)
        ax3.set_ylabel('Bias = Nd_MODIS / Nd_insitu', fontsize=10)
        ax3.legend(handles=patches_leg, fontsize=7.5, loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, f'n={mask3.sum()} — insufficient data',
                 transform=ax3.transAxes, ha='center', va='center', color='gray')
else:
    col_missing(ax3, 'mean_z_drizzle')
ax3.set_title('G3 · Bias vs Mean Drizzle Height in Cloud\n(0 = base, 1 = top)',
              fontsize=10, fontweight='bold')
style_ax(ax3)

# G4: vertical profile — z_norm vs Nc_CAS
vertical_profile(gm_m, 'Nc_CAS', REGIME_ORDER, REGIME_COLORS, REGIME_LABELS,
                 axes[1, 0],
                 xlabel='Nd (Nc_CAS, cm⁻³)',
                 title='G4 · Vertical Profile: Nd\n(median by z_norm bin, matched profiles)')
axes[1, 0].legend(fontsize=8, loc='upper right')

# G5: vertical profile — z_norm vs re_cas
vertical_profile(gm_m, 're_cas', REGIME_ORDER, REGIME_COLORS, REGIME_LABELS,
                 axes[1, 1],
                 xlabel='Effective radius re_cas (µm)',
                 title='G5 · Vertical Profile: re\n(median by z_norm bin, matched profiles)')
axes[1, 1].legend(fontsize=8, loc='upper right')

# G6: vertical profile — z_norm vs f_ad
vertical_profile(gm_m, 'f_ad', REGIME_ORDER, REGIME_COLORS, REGIME_LABELS,
                 axes[1, 2],
                 xlabel='Adiabatic fraction f_ad',
                 title='G6 · Vertical Profile: f_ad\n(median by z_norm bin, matched profiles)',
                 vline=0.8)
axes[1, 2].legend(fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_MAIN, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nMain figure saved → {OUT_MAIN}")

# ─── 8. SUPPLEMENTARY FIGURE (2×2) ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.patch.set_facecolor('white')
fig2.suptitle(
    f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} — Package G: Supplementary — Additional Vertical Structure Diagnostics",
    fontsize=13, fontweight='bold', y=0.98
)

# GS1: bias_21 vs z_base_m
bias_scatter(axes2[0, 0], df, 'z_base_m',
             'Cloud-base altitude (m)',
             'GS1 · Bias vs Cloud-base Altitude',
             colors_pt, patches_leg)

# GS2: bias_21 vs min_z_drizzle (normalized 0-1)
bias_scatter(axes2[0, 1], df, 'min_z_drizzle',
             'Min drizzle height (normalized, 0–1)',
             'GS2 · Bias vs Min Drizzle Height\n(0 = cloud-base, 1 = cloud-top)',
             colors_pt, patches_leg)

# GS3: bias_21 vs drizzle_fraction
bias_scatter(axes2[1, 0], df, 'drizzle_fraction',
             'Drizzle fraction',
             'GS3 · Bias vs Drizzle Fraction',
             colors_pt, patches_leg)

# GS4: Drizzle fraction by height & regime (horizontal bar chart)
ax_gs4 = axes2[1, 1]
bar_data = {}
drizzle_col = next((c for c in ['drizzle_flag', 'is_drizzle', 'drizzle'] if c in gm_m.columns), None)
if drizzle_col and 'z_bin' in gm_m.columns:
    for reg in REGIME_ORDER:
        sub = gm_m[gm_m['drizzle_regime'] == reg]
        if len(sub) == 0:
            bar_data[reg] = np.zeros(len(Z_MIDS))
            continue
        frac = sub.groupby('z_bin')[drizzle_col].mean()
        arr  = np.zeros(len(Z_MIDS))
        for idx, val in frac.items():
            if pd.notna(val):
                arr[int(idx)] = val
        bar_data[reg] = arr

    bar_h    = Z_BINS[1] - Z_BINS[0]
    n_reg    = len(REGIME_ORDER)
    offsets  = np.linspace(-(n_reg-1)/2, (n_reg-1)/2, n_reg) * bar_h * 0.22

    for i, reg in enumerate(REGIME_ORDER):
        arr       = bar_data[reg]
        y_centers = Z_MIDS + offsets[i]
        ax_gs4.barh(y_centers, arr,
                    height=bar_h * 0.22,
                    color=REGIME_COLORS[reg],
                    alpha=0.85,
                    label=REGIME_LABELS[reg],
                    edgecolor='white', linewidth=0.3)

    ax_gs4.set_xlim(0, 1.05)
    ax_gs4.set_ylim(0, 1)
    ax_gs4.set_xlabel(f'Drizzle fraction (mean {drizzle_col} per bin)', fontsize=10)
    ax_gs4.set_ylabel('Normalized height (z_norm)', fontsize=10)
    ax_gs4.axvline(0.5, color='k', ls='--', lw=1.0, alpha=0.5)
    ax_gs4.legend(fontsize=8, loc='upper right')
    ax_gs4.grid(True, alpha=0.3, axis='x')
else:
    ax_gs4.text(0.5, 0.5,
                f'drizzle_flag column not found\n(tried: drizzle_flag, is_drizzle, drizzle)',
                transform=ax_gs4.transAxes, ha='center', va='center',
                color='red', fontsize=10)
ax_gs4.set_title('GS4 · Drizzle Fraction by Height & Regime\n'
                 '(mean drizzle_flag per z_norm bin)',
                 fontsize=10, fontweight='bold')
style_ax(ax_gs4)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_SUPP, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Supplementary figure saved → {OUT_SUPP}")

print("\n" + "="*60)
print("✓  Package G v2 complete.")
print(f"   {OUT_MAIN}")
print(f"   {OUT_SUPP}")
print("="*60)
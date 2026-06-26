"""
cc_11_mechanism_tests.py

Test 5 candidate mechanisms for MODIS Nd under-estimation in Heavy state.

Mechanisms:
  M1. Vertical weighting (cloud-top re vs profile re)
  M2. Photon penetration depth (cloud_depth × bias)
  M3. Drizzle tail Re contribution (Δre × bias)
  M4. Subpixel heterogeneity (match_rate × state)
  M5. Nd inversion sensitivity (f_ad × inflation; matrix sensitivity)
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


# Load main matched dataset
m = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')
m['delta_re'] = m['re_full_median'] - m['re_cas_median']

# Match-rate test needs ALL profiles (not just MATCHED)
all_master = pd.read_csv(OUT_DIR / 'cc_master_all.csv')

print(f"\nMATCHED profiles: {len(m)}")
print(f"ALL profiles:     {len(all_master)}")

# State-color mapping
STATE_COLORS = {'Non': '#3366cc', 'Transition': '#ff8c00', 'Heavy': '#cc3333'}

# ==========================================================================
# MECHANISM 1: Vertical weighting
#  Hypothesis: MODIS Re_2.1 (cloud-top dominated) > re_CAS_in_situ (profile median)
#              The gap should be largest in Heavy state because drizzle drops
#              accumulate near cloud top.
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM 1: Vertical Weighting")
print("=" * 80)
print("Hypothesis: MODIS Re_2.1 (cloud-top weighted) > in-situ re_CAS (profile median)")
print("Test: ΔRe_vw = Re_MODIS_21 - re_cas_median, by state")
print()

m['delta_re_vw'] = m['Re_MODIS_21'] - m['re_cas_median']

print(f"{'State':<12} {'n':>4} {'median ΔRe':>12} {'IQR':>20} {'p (Wilcoxon vs 0)':>20}")
print("-" * 80)
for state in ['Non', 'Transition', 'Heavy']:
    sub = m[m['drizzle_state'] == state]
    d = sub['delta_re_vw'].dropna()
    if len(d) >= 3:
        med = d.median()
        q25, q75 = d.quantile([.25, .75])
        try:
            stat, p = stats.wilcoxon(d.values)
            sig = ('***' if p < 0.001 else '**' if p < 0.01
                    else '*' if p < 0.05 else 'ns')
        except Exception:
            p = np.nan
            sig = '?'
        print(f"{state:<12} {len(d):>4} {med:>+11.2f}  [{q25:>+5.2f}, {q75:>+5.2f}]  p={p:.3f} {sig}")

# K-W across states
groups = [m[m['drizzle_state'] == s]['delta_re_vw'].dropna()
          for s in ['Non', 'Transition', 'Heavy']]
groups = [g for g in groups if len(g) >= 3]
if len(groups) >= 2:
    kw, p = stats.kruskal(*groups)
    print(f"\nK-W test across states: p = {p:.4f}")

# ==========================================================================
# MECHANISM 2: Photon penetration depth
#  Hypothesis: deeper clouds → MODIS sees less of the column → bias larger
#  Test: Spearman correlation of cloud_depth × bias_calc within each state
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM 2: Photon Penetration Depth")
print("=" * 80)
print("Hypothesis: Deeper clouds amplify MODIS under-estimation")
print("Test: Spearman r of cloud_depth × bias_21_calc")
print()

print(f"{'Group':<22} {'n':>4} {'Spearman r':>12} {'p':>10}")
print("-" * 60)
for state in ['Non', 'Transition', 'Heavy', 'POOLED']:
    if state == 'POOLED':
        sub = m
    else:
        sub = m[m['drizzle_state'] == state]
    d = sub.dropna(subset=['cloud_depth', 'bias_21_calc'])
    if len(d) >= 4:
        r, p = stats.spearmanr(d['cloud_depth'], d['bias_21_calc'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{state:<22} {len(d):>4} {r:>+11.3f}  p={p:.4f} {sig}")
    else:
        print(f"{state:<22} {len(d):>4}  (n too small)")

# Median cloud depth per state
print(f"\nCloud depth median by state:")
for state in ['Non', 'Transition', 'Heavy']:
    d = m[m['drizzle_state'] == state]['cloud_depth'].dropna()
    if len(d) > 0:
        print(f"  {state:<12}: {d.median():.0f} m  (IQR: {d.quantile(.25):.0f}-{d.quantile(.75):.0f})")

# ==========================================================================
# MECHANISM 3: Drizzle tail contribution to Re
#  Hypothesis: Δre (re_full - re_CAS) measures drizzle's contribution to
#              effective radius. Higher Δre → larger MODIS Re → lower Nd.
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM 3: Drizzle Tail Contribution to Re")
print("=" * 80)
print("Hypothesis: Δre = re_full - re_CAS amplifies MODIS Re cloud-top retrieval")
print("Test: Spearman r of Δre × bias_21_calc")
print()

print(f"{'Group':<22} {'n':>4} {'median Δre (µm)':>16} {'Spearman r':>12} {'p':>8}")
print("-" * 70)
for state in ['Non', 'Transition', 'Heavy', 'POOLED']:
    if state == 'POOLED':
        sub = m
    else:
        sub = m[m['drizzle_state'] == state]
    d = sub.dropna(subset=['delta_re', 'bias_21_calc'])
    med_dre = d['delta_re'].median() if len(d) > 0 else np.nan
    if len(d) >= 4:
        r, p = stats.spearmanr(d['delta_re'], d['bias_21_calc'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"{state:<22} {len(d):>4} {med_dre:>15.2f}    {r:>+11.3f}  p={p:.3f} {sig}")
    else:
        print(f"{state:<22} {len(d):>4} {med_dre:>15.2f}    (n too small)")

# Direct test: Δre vs (Re_MODIS - re_cas_in_situ)
print(f"\nKey check: Does Δre (in-situ measure) correlate with Re_MODIS - re_cas (satellite vs in-situ gap)?")
d = m.dropna(subset=['delta_re', 'delta_re_vw'])
if len(d) >= 4:
    r, p = stats.spearmanr(d['delta_re'], d['delta_re_vw'])
    print(f"  Spearman r = {r:+.3f}, p = {p:.4f} (n={len(d)})")
    print(f"  Interpretation: if r>0, large drizzle tail correlates with large MODIS-insitu Re gap → vertical weighting via drizzle")

# ==========================================================================
# MECHANISM 4: Subpixel heterogeneity (3D effects)
#  Hypothesis: Heavy state has more broken/cellular cloud structure →
#              MODIS plane-parallel retrieval fails more often.
#  Proxy: match_rate (MATCHED / total) by state
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM 4: Subpixel Heterogeneity (3D Effects)")
print("=" * 80)
print("Hypothesis: Heavy clouds have more broken structure → lower MODIS QC pass rate")
print("Test: Match-rate by state (using all_master)")
print()

# all_master: all 130 profiles (with match_status)
# Build state column for all_master
def regime_to_state(reg):
    if reg == 'non_drizzling':
        return 'Non'
    elif reg in ('weak_drizzling', 'moderate_drizzling'):
        return 'Transition'
    elif reg == 'heavy_drizzling':
        return 'Heavy'
    return 'Unknown'
all_master['drizzle_state'] = all_master['drizzle_regime_clean'].map(regime_to_state)

print(f"{'State':<12} {'Total':>7} {'MATCHED':>9} {'NO_VALID':>10} {'NO_COV':>8} {'Match%':>8}")
print("-" * 70)
for state in ['Non', 'Transition', 'Heavy']:
    sub = all_master[all_master['drizzle_state'] == state]
    n_total = len(sub)
    n_matched = (sub['match_status'] == 'MATCHED').sum()
    n_novalid = (sub['match_status'] == 'NO_VALID_PIXELS').sum()
    n_nocov = (sub['match_status'] == 'NO_COVERAGE').sum()
    match_rate = n_matched / n_total * 100 if n_total else 0
    print(f"{state:<12} {n_total:>7} {n_matched:>9} {n_novalid:>10} {n_nocov:>8} {match_rate:>7.1f}%")

# By state x campaign
print(f"\nMatch rate by state × campaign:")
for state in ['Non', 'Transition', 'Heavy']:
    print(f"\n  {state}:")
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub = all_master[(all_master['drizzle_state'] == state) &
                          (all_master['campaign'] == camp)]
        if len(sub) == 0:
            continue
        n_total = len(sub)
        n_matched = (sub['match_status'] == 'MATCHED').sum()
        rate = n_matched / n_total * 100 if n_total else 0
        print(f"    {camp:<8}: {n_matched}/{n_total} = {rate:.0f}%")

# ==========================================================================
# MECHANISM 5: Nd inversion sensitivity (matematical chain)
#  Hypothesis: ∂Nd/Nd ~ -2.5 ∂re/re. Re error has 5x weight than τ.
#  Plus: f_ad gap (lit-insitu) drives inflation factor.
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM 5: Nd Inversion Sensitivity")
print("=" * 80)
print("Hypothesis: Combined re + f_ad errors compound non-linearly")
print()

# 5a: f_ad gap correlation with inflation
print("5a. f_ad gap × inflation_21 correlation:")
m['f_ad_gap'] = 0.80 - m['f_ad_mean']  # lit (0.80) - in-situ
print(f"  Pooled (n={len(m)}):")
d = m.dropna(subset=['f_ad_gap', 'inflation_21'])
r, p = stats.spearmanr(d['f_ad_gap'], d['inflation_21'])
sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
print(f"    Spearman r = {r:+.3f}  p = {p:.4f} {sig}")

# 5b: per state
print(f"\n  Per state:")
for state in ['Non', 'Transition', 'Heavy']:
    sub = m[m['drizzle_state'] == state].dropna(subset=['f_ad_gap', 'inflation_21'])
    if len(sub) >= 4:
        r, p = stats.spearmanr(sub['f_ad_gap'], sub['inflation_21'])
        med_gap = sub['f_ad_gap'].median()
        med_infl = sub['inflation_21'].median()
        print(f"    {state:<12}: gap={med_gap:.2f}, infl={med_infl:.2f}, r={r:+.3f}, p={p:.3f}")

# 5c: Theoretical sensitivity check
print(f"\n5c. Theoretical: Nd ∝ re^-2.5")
print(f"    1% increase in MODIS re → 2.5% decrease in inferred Nd")
print(f"    From mechanism 1: median Heavy Re_MODIS - re_cas = {m[m['drizzle_state']=='Heavy']['delta_re_vw'].median():.2f} µm")
print(f"    Heavy state median re_cas = {m[m['drizzle_state']=='Heavy']['re_cas_median'].median():.2f} µm")
print(f"    Predicted bias contribution from Re error alone:")
re_med = m[m['drizzle_state']=='Heavy']['re_cas_median'].median()
delta_re_vw = m[m['drizzle_state']=='Heavy']['delta_re_vw'].median()
re_ratio = (re_med + delta_re_vw) / re_med
predicted_nd_ratio = re_ratio ** (-2.5)
print(f"    re_ratio = {re_ratio:.3f}, predicted Nd_ratio = {predicted_nd_ratio:.3f}")
observed_bias = m[m['drizzle_state']=='Heavy']['bias_21_calc'].median()
print(f"    Observed bias_calc median (Heavy) = {observed_bias:.3f}")
print(f"    Re-error alone explains: {predicted_nd_ratio:.2f} of observed {observed_bias:.2f}")
explained_pct = (1 - predicted_nd_ratio) / (1 - observed_bias) * 100
print(f"    Re-error contribution: {explained_pct:.0f}% of observed under-estimation")

# ==========================================================================
# Save summary as CSV
# ==========================================================================
print("\n" + "=" * 80)
print("MECHANISM EVIDENCE SUMMARY")
print("=" * 80)
summary_lines = [
    ['M1', 'Vertical weighting',       'Re_MODIS_21 - re_cas',  'Direct'],
    ['M2', 'Photon penetration',       'cloud_depth × bias',     'Indirect'],
    ['M3', 'Drizzle tail Re',          'Δre × bias',             'Direct'],
    ['M4', 'Subpixel heterogeneity',   'match_rate × state',     'Indirect'],
    ['M5', 'Nd inversion sensitivity', 'f_ad_gap × inflation',   'Direct (+ theoretical)'],
]
print(f"\n{'M':<3} {'Mechanism':<25} {'Proxy':<25} {'Evidence Type'}")
print("-" * 80)
for row in summary_lines:
    print(f"{row[0]:<3} {row[1]:<25} {row[2]:<25} {row[3]}")

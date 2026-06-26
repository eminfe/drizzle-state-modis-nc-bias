"""
cc_02_bootstrap_ci.py  -  Bootstrap CI for Cross-Campaign Metrics

Computes 10,000-iteration bootstrap 95% CI for:
  - bias_21_calc median
  - bias_21_lit  median
  - inflation factor median
  - dNd_calc median

For each campaign and each drizzle regime.

OUTPUT:
  cc_bootstrap_ci.csv  -- All CI values in tabular form
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===

N_BOOT = 10_000
CI_LEVEL = 95
np.random.seed(42)


def bootstrap_median_ci(values, n_boot=N_BOOT, ci=CI_LEVEL):
    """Bootstrap median + 95% CI."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 3:
        return np.nan, np.nan, np.nan, n

    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        medians[i] = np.median(sample)

    lower = np.percentile(medians, (100 - ci) / 2)
    upper = np.percentile(medians, 100 - (100 - ci) / 2)
    return float(np.median(values)), float(lower), float(upper), n


def bootstrap_mean_ci(values, n_boot=N_BOOT, ci=CI_LEVEL):
    """Bootstrap mean + 95% CI."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 3:
        return np.nan, np.nan, np.nan, n

    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lower), float(upper), n


def main():
    df = pd.read_csv(OUT_DIR / 'cc_master_matched.csv')
    print(f"Loaded {len(df)} MATCHED profiles")

    metrics = ['bias_21_calc', 'bias_21_lit', 'inflation_21',
               'bias_37_calc', 'bias_37_lit',
               'dNd_calc', 'dNd_lit',
               'dRe_37_21']

    results = []

    # === BY CAMPAIGN ===
    print(f"\n{'='*80}")
    print("BOOTSTRAP CI BY CAMPAIGN")
    print(f"{'='*80}")
    for camp in df['campaign'].unique():
        sub = df[df['campaign'] == camp]
        print(f"\n--- {camp} (n={len(sub)} matched) ---")
        for m in metrics:
            if m not in sub.columns:
                continue
            med, lo, hi, n = bootstrap_median_ci(sub[m].values)
            print(f"  {m:18s}: median={med:7.3f}  95% CI=[{lo:7.3f}, {hi:7.3f}]  n={n}")
            results.append({
                'group_type': 'campaign',
                'group':      camp,
                'subgroup':   '',
                'metric':     m,
                'stat':       'median',
                'value':      med,
                'ci_lower':   lo,
                'ci_upper':   hi,
                'n':          n,
            })

    # === BY CAMPAIGN x DRIZZLE REGIME ===
    print(f"\n{'='*80}")
    print("BOOTSTRAP CI BY CAMPAIGN x DRIZZLE REGIME")
    print(f"{'='*80}")
    for camp in df['campaign'].unique():
        for regime in ['non_drizzling', 'weak_drizzling',
                       'moderate_drizzling', 'heavy_drizzling']:
            sub = df[(df['campaign'] == camp) &
                     (df['drizzle_regime_clean'] == regime)]
            if len(sub) == 0:
                continue
            print(f"\n--- {camp} x {regime} (n={len(sub)}) ---")
            for m in ['bias_21_calc', 'bias_21_lit', 'inflation_21']:
                if m not in sub.columns:
                    continue
                med, lo, hi, n = bootstrap_median_ci(sub[m].values)
                if n >= 3:
                    print(f"  {m:18s}: median={med:7.3f}  95% CI=[{lo:7.3f}, {hi:7.3f}]  n={n}")
                else:
                    print(f"  {m:18s}: n={n} (too small for CI)")
                results.append({
                    'group_type': 'campaign_regime',
                    'group':      camp,
                    'subgroup':   regime,
                    'metric':     m,
                    'stat':       'median',
                    'value':      med,
                    'ci_lower':   lo,
                    'ci_upper':   hi,
                    'n':          n,
                })

    # === ALL CAMPAIGNS POOLED ===
    print(f"\n{'='*80}")
    print("BOOTSTRAP CI - ALL CAMPAIGNS POOLED (n=52)")
    print(f"{'='*80}")
    for m in metrics:
        if m not in df.columns:
            continue
        med, lo, hi, n = bootstrap_median_ci(df[m].values)
        print(f"  {m:18s}: median={med:7.3f}  95% CI=[{lo:7.3f}, {hi:7.3f}]  n={n}")
        results.append({
            'group_type': 'pooled',
            'group':      'all_campaigns',
            'subgroup':   '',
            'metric':     m,
            'stat':       'median',
            'value':      med,
            'ci_lower':   lo,
            'ci_upper':   hi,
            'n':          n,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_DIR / 'cc_bootstrap_ci.csv', index=False)
    print(f"\n[SAVE] cc_bootstrap_ci.csv  ({df_out.shape})")

    # === KEY FINDING: Do CIs overlap between campaigns? ===
    print(f"\n{'='*80}")
    print("CRITICAL ANALYSIS: Do campaign CIs overlap?")
    print(f"{'='*80}")
    for m in ['bias_21_calc', 'bias_21_lit', 'inflation_21']:
        camp_ci = df_out[(df_out['group_type'] == 'campaign') &
                          (df_out['metric'] == m)]
        print(f"\n{m}:")
        for _, r in camp_ci.iterrows():
            print(f"  {r['group']:8s}: {r['value']:.3f}  CI=[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]  n={r['n']}")
        # Check overlap
        rows = camp_ci.to_dict('records')
        for i in range(len(rows)):
            for j in range(i+1, len(rows)):
                r1, r2 = rows[i], rows[j]
                # Overlap if max(low) < min(high)
                low_max = max(r1['ci_lower'], r2['ci_lower'])
                high_min = min(r1['ci_upper'], r2['ci_upper'])
                overlaps = low_max < high_min
                tag = "OVERLAP" if overlaps else "DISTINCT"
                print(f"    {r1['group']} vs {r2['group']}: {tag}")


if __name__ == '__main__':
    main()

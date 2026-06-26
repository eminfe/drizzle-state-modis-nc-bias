"""
cc_06_3state_bootstrap.py

Bootstrap 95% CI for 3-state grouping.
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
np.random.seed(42)


def bootstrap_median_ci(values, n_boot=N_BOOT, ci=95):
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


master = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')
print(f"Loaded {len(master)} matched profiles")

results = []
metrics = ['bias_21_calc', 'bias_21_lit', 'inflation_21',
           'dNd_calc', 'dRe_37_21']

# === BY 3-STATE (POOLED) ===
print("\n" + "=" * 80)
print("BOOTSTRAP CI BY 3-STATE (pooled across campaigns)")
print("=" * 80)
for state in ['Non', 'Transition', 'Heavy']:
    sub = master[master['drizzle_state'] == state]
    print(f"\n--- {state} (n={len(sub)}) ---")
    for m in metrics:
        if m not in sub.columns:
            continue
        med, lo, hi, n = bootstrap_median_ci(sub[m].values)
        if n >= 3:
            print(f"  {m:18s}: median={med:7.3f}  95% CI=[{lo:7.3f}, {hi:7.3f}]  n={n}")
        results.append(dict(group_type='state', group=state, subgroup='',
                             metric=m, value=med, ci_lower=lo, ci_upper=hi, n=n))

# === BY 3-STATE x CAMPAIGN ===
print("\n" + "=" * 80)
print("BOOTSTRAP CI BY 3-STATE x CAMPAIGN")
print("=" * 80)
for state in ['Non', 'Transition', 'Heavy']:
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub = master[(master['drizzle_state'] == state) &
                      (master['campaign'] == camp)]
        if len(sub) == 0:
            continue
        print(f"\n--- {state} x {camp} (n={len(sub)}) ---")
        for m in ['bias_21_calc', 'bias_21_lit', 'inflation_21']:
            if m not in sub.columns:
                continue
            med, lo, hi, n = bootstrap_median_ci(sub[m].values)
            if n >= 3:
                print(f"  {m:18s}: median={med:7.3f}  95% CI=[{lo:7.3f}, {hi:7.3f}]  n={n}")
            else:
                print(f"  {m:18s}: median={med:7.3f}  (n={n} too small for CI)")
            results.append(dict(group_type='state_campaign', group=state,
                                 subgroup=camp, metric=m, value=med,
                                 ci_lower=lo, ci_upper=hi, n=n))

# Save
df_out = pd.DataFrame(results)
df_out.to_csv(OUT_DIR / 'cc_bootstrap_3state.csv', index=False)
print(f"\n[SAVE] cc_bootstrap_3state.csv  ({df_out.shape})")

# === KEY ANALYSIS: Do state CIs overlap? ===
print("\n" + "=" * 80)
print("CRITICAL ANALYSIS: Do 3-state CIs overlap?")
print("=" * 80)
for m in ['bias_21_calc', 'bias_21_lit', 'inflation_21']:
    state_ci = df_out[(df_out['group_type'] == 'state') &
                       (df_out['metric'] == m)]
    print(f"\n{m}:")
    rows = state_ci.to_dict('records')
    for r in rows:
        print(f"  {r['group']:<12}: {r['value']:.3f}  CI=[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]  n={r['n']}")
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            r1, r2 = rows[i], rows[j]
            low_max = max(r1['ci_lower'], r2['ci_lower'])
            high_min = min(r1['ci_upper'], r2['ci_upper'])
            overlaps = low_max < high_min
            tag = "OVERLAP" if overlaps else "DISTINCT"
            print(f"    {r1['group']} vs {r2['group']}: {tag}")

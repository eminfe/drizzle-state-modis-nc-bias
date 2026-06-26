"""
cc_05_3state_master.py

Add drizzle_state column to master dataset:
  Non         <- non_drizzling
  Transition  <- weak_drizzling + moderate_drizzling
  Heavy       <- heavy_drizzling

Output: cc_master_3state.csv
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

# Load master
master = pd.read_csv(OUT_DIR / 'cc_master_matched.csv')
print(f"Loaded {len(master)} matched profiles")

# Map 4-regime -> 3-state
def regime_to_state(reg):
    if reg == 'non_drizzling':
        return 'Non'
    elif reg in ('weak_drizzling', 'moderate_drizzling'):
        return 'Transition'
    elif reg == 'heavy_drizzling':
        return 'Heavy'
    else:
        return 'Unknown'

master['drizzle_state'] = master['drizzle_regime_clean'].map(regime_to_state)

# Save
master.to_csv(OUT_DIR / 'cc_master_3state.csv', index=False)
print(f"\n[SAVE] cc_master_3state.csv")

# Summary
print("\n" + "=" * 70)
print("3-state distribution by campaign:")
print("=" * 70)
xtab = pd.crosstab(master['drizzle_state'], master['campaign'], margins=True)
xtab = xtab.reindex(['Non', 'Transition', 'Heavy', 'All'])
print(xtab.to_string())

print("\n" + "=" * 70)
print("Bias by 3-state (pooled across campaigns):")
print("=" * 70)
for state in ['Non', 'Transition', 'Heavy']:
    sub = master[master['drizzle_state'] == state]
    bias_calc = sub['bias_21_calc'].dropna()
    bias_lit  = sub['bias_21_lit'].dropna()
    inflation = sub['inflation_21'].dropna()
    print(f"\n--- {state} (n={len(sub)}) ---")
    print(f"  bias_calc  : median={bias_calc.median():.3f}  IQR=[{bias_calc.quantile(.25):.2f}, {bias_calc.quantile(.75):.2f}]")
    print(f"  bias_lit   : median={bias_lit.median():.3f}   IQR=[{bias_lit.quantile(.25):.2f}, {bias_lit.quantile(.75):.2f}]")
    print(f"  inflation  : median={inflation.median():.3f}  IQR=[{inflation.quantile(.25):.2f}, {inflation.quantile(.75):.2f}]")
    p_under = (bias_calc < 1.0).sum() / len(bias_calc)
    p_over = (bias_calc > 1.0).sum() / len(bias_calc)
    print(f"  Direction  : {p_over:.0%} over, {p_under:.0%} under")
    cv = (bias_calc.std() / bias_calc.mean()) * 100
    print(f"  CV         : {cv:.0f}%")

print("\n" + "=" * 70)
print("Bias by 3-state x campaign:")
print("=" * 70)
print(f"\n{'State':<12} {'Campaign':<10} {'n':>3} {'bias_calc':>11} {'bias_lit':>11} {'inflation':>11}")
print("-" * 70)
for state in ['Non', 'Transition', 'Heavy']:
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub = master[(master['drizzle_state'] == state) &
                      (master['campaign'] == camp)]
        if len(sub) == 0:
            print(f"{state:<12} {camp:<10} {0:>3}   —")
            continue
        n = len(sub)
        bc = sub['bias_21_calc'].median()
        bl = sub['bias_21_lit'].median()
        infl = sub['inflation_21'].median()
        print(f"{state:<12} {camp:<10} {n:>3} {bc:>11.3f} {bl:>11.3f} {infl:>11.3f}")

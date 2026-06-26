"""
cc_01_build_master.py  -  Cross-Campaign Master Dataset Builder

Loads POST + MASE + VOCALS golden_case.csv and MODIS_Matches.csv files,
harmonizes column names, and builds a single concatenated DataFrame
with a 'campaign' identifier column.

OUTPUT:
  cc_master_all.csv      -- All profiles (130 rows)
  cc_master_matched.csv  -- MATCHED profiles only (52 rows, for bias analysis)
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

# Columns we want in the master dataset
KEEP_COLS_GC = [
    'cloud_id', 'flight_id', 'duration_s', 'H_m', 'cloud_depth',
    'lat_median', 'lon_median',
    'Nd_median', 're_cas_median', 're_full_median',
    'tau_main', 'tau_full', 'tau_diff',
    'f_ad_mean', 'f_ad_median', 'k_median', 'c_w_median',
    'LWP_insitu', 'drizzle_regime', 'drizzle_fraction',
    'dominant_mixing_regime',
    'mean_z_drizzle', 'min_z_drizzle', 'max_z_drizzle',
]

# Columns from MODIS_Matches we want
KEEP_COLS_MM = [
    'cloud_id', 'match_status', 'modis_file', 'delta_t_min',
    'n_pixels_valid_21', 'n_pixels_valid_37',
    'Re_MODIS_21', 'Re_MODIS_37', 'tau_MODIS_21', 'tau_MODIS_37',
    'Re_MODIS_21_paired37', 'tau_MODIS_21_paired37',
    'dRe_37_21', 'dtau_37_21',
    'SZA_mean', 'VZA_mean',
    'LWP_MODIS', 'CTT_MODIS', 'CTP_MODIS',
    'Nd_MODIS_21_calc', 'Nd_MODIS_37_calc', 'Nd_MODIS_21_paired37_calc',
    'Nd_MODIS_21_lit',  'Nd_MODIS_37_lit',  'Nd_MODIS_21_paired37_lit',
    'bias_21_calc', 'bias_37_calc', 'dNd_calc',
    'bias_21_lit',  'bias_37_lit',  'dNd_lit',
]


def load_campaign(camp_name):
    """Load gc + matches for one campaign and merge."""
    print(f"\n=== Loading {camp_name} ===")
    gc = pd.read_csv(DATA_DIR / camp_name / f'{camp_name}_golden_case.csv')
    mm = pd.read_csv(DATA_DIR / camp_name / f'{camp_name}_MODIS_Matches.csv')

    print(f"  gc:      {gc.shape}  cols={gc.shape[1]}")
    print(f"  matches: {mm.shape}  cols={mm.shape[1]}")

    # Keep only available columns
    gc_cols = [c for c in KEEP_COLS_GC if c in gc.columns]
    gc = gc[gc_cols].copy()
    mm_cols = [c for c in KEEP_COLS_MM if c in mm.columns]
    mm = mm[mm_cols].copy()

    # Merge on cloud_id
    df = gc.merge(mm, on='cloud_id', how='left', suffixes=('', '_dup'))
    # Drop any duplicate columns
    df = df[[c for c in df.columns if not c.endswith('_dup')]]

    # Add campaign tag
    df['campaign'] = camp_name

    # Inflation factor (per-profile)
    if 'bias_21_calc' in df.columns and 'bias_21_lit' in df.columns:
        df['inflation_21'] = df['bias_21_lit'] / df['bias_21_calc']

    print(f"  merged: {df.shape}  matched={(df['match_status']=='MATCHED').sum()}")
    return df


def harmonize_drizzle_regime(df):
    """Standardize drizzle_regime values across campaigns."""
    if 'drizzle_regime' in df.columns:
        # Different campaigns may use different label conventions
        df['drizzle_regime'] = df['drizzle_regime'].astype(str).str.strip()
        # Normalize to lowercase
        df['drizzle_regime_clean'] = df['drizzle_regime'].str.lower()

        # Map all variants
        mapping = {
            'non-drizzling':       'non_drizzling',
            'weak drizzling':      'weak_drizzling',
            'moderate drizzling':  'moderate_drizzling',
            'heavy drizzling':     'heavy_drizzling',
        }
        df['drizzle_regime_clean'] = df['drizzle_regime_clean'].replace(mapping)
    return df


def main():
    dfs = []
    for camp in CAMPAIGNS:
        df = load_campaign(camp)
        df = harmonize_drizzle_regime(df)
        dfs.append(df)

    master = pd.concat(dfs, ignore_index=True)

    # Reorder columns: campaign first
    cols = ['campaign'] + [c for c in master.columns if c != 'campaign']
    master = master[cols]

    print(f"\n{'='*70}")
    print(f"=== Master dataset ===")
    print(f"{'='*70}")
    print(f"Total shape: {master.shape}")
    print(f"\nProfiles per campaign:")
    print(master['campaign'].value_counts().to_string())
    print(f"\nMATCHED per campaign:")
    matched = master[master['match_status'] == 'MATCHED']
    print(matched['campaign'].value_counts().to_string())

    print(f"\nBias summary (MATCHED only):")
    summary = matched.groupby('campaign')[
        ['bias_21_calc', 'bias_21_lit', 'inflation_21', 'dNd_calc']
    ].agg(['median', 'mean', 'std', 'count']).round(3)
    print(summary.to_string())

    # Save
    master.to_csv(OUT_DIR / 'cc_master_all.csv', index=False)
    matched.to_csv(OUT_DIR / 'cc_master_matched.csv', index=False)
    print(f"\n[SAVE] cc_master_all.csv      ({master.shape})")
    print(f"[SAVE] cc_master_matched.csv  ({matched.shape})")


if __name__ == '__main__':
    main()

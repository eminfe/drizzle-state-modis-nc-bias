# =============================================================================
# step02_qc_filtering.py  -  MASE 2005
# =============================================================================
# Purpose:
#   - Apply physical-range QC masks (LWC, AT, PS, GALT) -> out-of-range = NaN
#   - Clean CAS / CIP bin columns: inf -> NaN, negative -> 0
#   - Zero out INACTIVE bins (NaN midpoint in config) — instrument
#     overflow/noise channels that are not real measurements
#   - Zero out CIP overlap-zone bins (D <= CIP_CUTOFF_UM) to prevent
#     double-counting of cloud droplets that are already measured by CAS
#   - Compute number concentrations: Nc_CAS, Nc_CIP, Nc_Total
#   - Save as STEP02_QC_PARQUET for downstream steps
#
# Does NOT:
#   - Drop any rows (only flags bad values as NaN, preserving row count)
#   - Detect cloud profiles (handled by step03)
#
# Inactive-bin handling (MASE-relevant):
#   MASE CAS Bin 1 and CIP Bin 1 have undefined lower bounds and are
#   flagged in config with D_mid = NaN. CAS Bin 1 in MASE is empty
#   (instrument did not report counts), but CIP Bin 1 contains
#   instrument noise / overflow counts of ~0.03 cm^-3 mean — an order
#   of magnitude larger than CIP Bin 2. If left untreated these counts
#   silently inflate Nc_CIP and Nc_Total. zero_inactive_bins() removes
#   all inactive bin contributions so downstream Nc/LWC sums reflect
#   real droplet measurements only.
#
# CIP overlap handling (Wood 2012):
#   CIP bins with midpoint <= CIP_CUTOFF_UM (50 um) overlap with CAS bins
#   in the same size range. CAS is the precision instrument for cloud
#   droplets in this overlap zone. To prevent double-counting in:
#     - LWC_CIP (step04 drizzle classification)
#     - Nc_CIP and Nc_Total (cloud-core detection)
#     - re_full composite spectrum (step05)
#   the overlap-zone CIP bins are zeroed out at this stage. NaN-flagged
#   placeholder bins (e.g., MASE Bin 1 with NaN lower bound) are preserved
#   as NaN; only bins with valid midpoint <= cutoff get zeroed.
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import (
    get_cas_columns, get_cip_columns,
    clip_lower, mask_outside_range,
)


# Local helpers defined below: apply_physical_qc, clean_bin_columns,
# zero_cip_overlap_zone, zero_inactive_bins, compute_number_concentrations


# =============================================================================
# Helpers
# =============================================================================
def apply_physical_qc(df, qc=config.QC, var_map=config.VAR_MAP):
    """
    Replace out-of-physical-range values with NaN. Row count is preserved.

    Variable-by-variable:
      LWC < 0                       -> NaN  (instrument noise / baseline drift)
      AT  outside [at_min, at_max]  -> NaN  (sensor failure or ground sample)
      PS  outside [ps_min, ps_max]  -> NaN  (sensor failure)
      GALT < 0                      -> NaN  (impossible altitude)
    """
    counts = {}
    col_lwc = var_map["lwc"]
    col_at  = var_map["temperature"]
    col_ps  = var_map["pressure"]
    col_alt = var_map["altitude"]

    before = df[col_lwc].notna().sum()
    df.loc[df[col_lwc] < qc["lwc_min"], col_lwc] = np.nan
    counts["lwc"] = before - df[col_lwc].notna().sum()

    before = df[col_at].notna().sum()
    df[col_at] = mask_outside_range(df[col_at], qc["at_min"], qc["at_max"])
    counts["temperature"] = before - df[col_at].notna().sum()

    before = df[col_ps].notna().sum()
    df[col_ps] = mask_outside_range(df[col_ps], qc["ps_min"], qc["ps_max"])
    counts["pressure"] = before - df[col_ps].notna().sum()

    before = df[col_alt].notna().sum()
    df.loc[df[col_alt] < qc["alt_min"], col_alt] = np.nan
    counts["altitude"] = before - df[col_alt].notna().sum()

    return df, counts


def clean_bin_columns(df, cas_cols, cip_cols):
    """
    Clean CAS/CIP bin columns:
      inf      -> NaN
      negative -> 0    (instrument noise floor)
      NaN      -> preserved
    """
    bin_cols = cas_cols + cip_cols
    df[bin_cols] = df[bin_cols].replace([np.inf, -np.inf], np.nan)
    for col in bin_cols:
        df[col] = clip_lower(df[col], lower=0.0)
    return df


def zero_cip_overlap_zone(df, cip_cols, cip_d_mid_um, cutoff_um):
    """
    Zero out CIP bins that fall in the CAS-CIP overlap region.

    CIP bins with midpoint <= cutoff_um (e.g., 50 um) measure droplets that
    are already measured precisely by CAS. Including them in CIP-derived
    quantities (LWC_CIP, Nc_CIP) causes double-counting and inflates
    drizzle metrics by orders of magnitude.

    NaN-flagged bins (placeholder) are preserved as NaN.
    """
    d = np.asarray(cip_d_mid_um, dtype=float)
    if len(d) != len(cip_cols):
        n = min(len(d), len(cip_cols))
        d = d[:n]
        cip_cols = cip_cols[:n]

    overlap_mask = (~np.isnan(d)) & (d <= cutoff_um)
    overlap_bin_names = [c for c, m in zip(cip_cols, overlap_mask) if m]

    for col in overlap_bin_names:
        df[col] = 0.0

    return df, len(overlap_bin_names), overlap_bin_names


def zero_inactive_bins(df, cols, d_mid_um, label=""):
    """
    Zero out instrument bins flagged as INACTIVE (NaN midpoint in config).

    Some campaigns include bin columns whose lower bound is undefined
    (e.g., MASE 'CIP Bin 1' with bounds NaN-25 um). These bins may
    contain instrument noise or overflow counts that are NOT physical
    droplet measurements. Including them in Nc/LWC sums silently
    inflates downstream quantities.

    This function zeros any column whose corresponding d_mid is NaN.

    Parameters
    ----------
    df : DataFrame
    cols : list of str  (bin column names, aligned with d_mid_um)
    d_mid_um : array-like
    label : str  (e.g., 'CAS' or 'CIP', for the report only)

    Returns
    -------
    df : DataFrame  (modified in-place)
    n_zeroed : int
    zeroed_names : list of str
    """
    d = np.asarray(d_mid_um, dtype=float)
    if len(d) != len(cols):
        n = min(len(d), len(cols))
        d = d[:n]
        cols = cols[:n]

    inactive_mask = np.isnan(d)
    inactive_names = [c for c, m in zip(cols, inactive_mask) if m]

    for col in inactive_names:
        df[col] = 0.0

    return df, len(inactive_names), inactive_names


def compute_number_concentrations(df, cas_cols, cip_cols):
    """
    Compute total droplet number concentration per row:

        Nc_CAS   = sum(CAS_bin_*)
        Nc_CIP   = sum(CIP_bin_*)   [overlap zone already zeroed]
        Nc_Total = Nc_CAS + Nc_CIP

    The min_count=1 argument ensures rows with all bins NaN return NaN,
    rather than zero (which would falsely indicate "cloud-free").
    """
    def _safe_sum(cols):
        if not cols:
            return pd.Series(np.nan, index=df.index)
        return df[cols].clip(lower=0.0).sum(axis=1, min_count=1)

    df["Nc_CAS"]   = _safe_sum(cas_cols)
    df["Nc_CIP"]   = _safe_sum(cip_cols)
    df["Nc_Total"] = df[["Nc_CAS", "Nc_CIP"]].sum(axis=1, min_count=1)
    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(f"  step02 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}")
    print("=" * 65)

    if not config.STEP01_PARQUET.exists():
        print(f"\n  [FAIL] step01 parquet not found: {config.STEP01_PARQUET}")
        print(f"  Run step01_load_data.py first.")
        sys.exit(1)

    print(f"\n  Loading {config.STEP01_PARQUET.name}...")
    df = pd.read_parquet(config.STEP01_PARQUET)
    print(f"  [OK] {len(df):,} rows x {len(df.columns)} columns")

    cas_cols = get_cas_columns(df, config.VAR_MAP["cas_prefix"])
    cip_cols = get_cip_columns(df, config.VAR_MAP["cip_prefix"])
    print(f"  CAS bin count : {len(cas_cols)}")
    print(f"  CIP bin count : {len(cip_cols)}")

    # ----------- Physical-range QC -----------
    print(f"\n  Applying physical QC masks...")
    df, counts = apply_physical_qc(df)
    n = len(df)
    for var, cnt in counts.items():
        pct = 100.0 * cnt / n
        tag = 'OK' if cnt == 0 else 'MASK'
        print(f"  [{tag:>4}] {var:<12}: "
              f"{cnt:>10,} -> NaN  ({pct:.2f}%)")

    # ----------- Bin cleanup -----------
    print(f"\n  Bin cleanup (inf -> NaN, negative -> 0)...")
    n_inf = np.isinf(df[cas_cols + cip_cols].values).sum()
    print(f"  inf count (will become NaN): {n_inf}")
    df = clean_bin_columns(df, cas_cols, cip_cols)

    # ----------- Inactive-bin zeroing (NaN midpoint -> not a real bin) -----------
    print(f"\n  Inactive-bin zeroing (D_mid = NaN -> not a real measurement bin)...")
    df, n_cas_inactive, cas_inactive_names = zero_inactive_bins(
        df, cas_cols, config.CAS_D_MID_ALL, label="CAS"
    )
    print(f"  CAS inactive bins zeroed: {n_cas_inactive}"
          f"{' -> ' + ', '.join(cas_inactive_names) if cas_inactive_names else ''}")
    df, n_cip_inactive, cip_inactive_names = zero_inactive_bins(
        df, cip_cols, config.CIP_D_MID_UM, label="CIP"
    )
    print(f"  CIP inactive bins zeroed: {n_cip_inactive}"
          f"{' -> ' + ', '.join(cip_inactive_names) if cip_inactive_names else ''}")

    # ----------- CIP overlap zone fix -----------
    print(f"\n  CIP overlap fix (Wood 2012, cutoff = {config.CIP_CUTOFF_UM:.1f} um)")
    df, n_overlap, overlap_names = zero_cip_overlap_zone(
        df, cip_cols, config.CIP_D_MID_UM, config.CIP_CUTOFF_UM
    )
    print(f"  Bins zeroed (D <= {config.CIP_CUTOFF_UM:.0f} um, CAS overlap): {n_overlap}")
    if overlap_names:
        first_idx = cip_cols.index(overlap_names[0])
        last_idx  = cip_cols.index(overlap_names[-1])
        d_first = config.CIP_D_MID_UM[first_idx]
        d_last  = config.CIP_D_MID_UM[last_idx]
        print(f"    First : {overlap_names[0]}  (D = {d_first:.1f} um)")
        print(f"    Last  : {overlap_names[-1]}  (D = {d_last:.1f} um)")
        if last_idx + 1 < len(cip_cols):
            print(f"    First retained CIP bin: {cip_cols[last_idx+1]}  "
                  f"(D = {config.CIP_D_MID_UM[last_idx+1]:.1f} um)")

    # ----------- Number concentrations -----------
    print(f"\n  Computing Nc_CAS, Nc_CIP, Nc_Total...")
    df = compute_number_concentrations(df, cas_cols, cip_cols)
    for col in ["Nc_CAS", "Nc_CIP", "Nc_Total"]:
        s = df[col].dropna()
        nan_pct = 100.0 * df[col].isna().sum() / n
        print(f"  [OK] {col:<10}  median={s.median():.2f}  max={s.max():.2f}  "
              f"NaN={nan_pct:.1f}%")

    # ----------- Save -----------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.STEP02_QC_PARQUET, index=False)
    print(f"\n  [OK] Saved -> {config.STEP02_QC_PARQUET.name}")
    print(f"        Size: {config.STEP02_QC_PARQUET.stat().st_size / 1e6:.1f} MB")
    print("\n  step02 COMPLETE - ready for step03.\n")
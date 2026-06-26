# =============================================================================
# step01_load_data.py  -  MASE 2005
# =============================================================================
# Purpose:
#   - Read MASE_10s_merged_v2.parquet
#   - Compute GWIU (vertical velocity) using actual dt per flight
#       (MASE raw data has no vertical velocity column; must compute)
#   - Validate datetime column and required-column presence
#   - Print health-check report (flights, variable ranges, NaN counts)
#   - Save as STEP01_PARQUET for downstream steps
#
# Does NOT do:
#   - QC masking (handled by step02)
#   - Profile detection (handled by step03)
#
# Note on vertical velocity computation (Bug #16 fix):
#   The earlier MASE pipeline used `Alt.diff() / config.DT` which assumes a
#   uniform 10 s sampling rate. If any flight has gaps (sensor dropout, time
#   stamp irregularity), the computed Vz becomes wrong wherever dt != 10 s.
#   This step now uses the actual time delta per row:
#       dt_actual[i] = (time[i] - time[i-1]).total_seconds()
#       Vz[i]        = (alt[i] - alt[i-1]) / dt_actual[i]
#   This matches utils.compute_vertical_velocity() and is correct under
#   any sampling regularity, including the 10 s MASE cadence.
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import get_cas_columns, get_cip_columns, check_required_columns


# =============================================================================
# Settings
# =============================================================================
DATA_FILE     = config.PARQUET_FILE
CAMPAIGN_NAME = config.CAMPAIGN_NAME
VAR_MAP       = config.VAR_MAP
DT            = config.DT                # 10.0 s nominal sampling interval

CAS_PREFIX    = VAR_MAP["cas_prefix"]
CIP_PREFIX    = VAR_MAP["cip_prefix"]
GWIU_COL      = VAR_MAP["aircraft_vert_vel_raw"]   # "GWIU"
TIME_COL      = VAR_MAP["time"]                     # "Datetime"

# Required columns - GWIU is intentionally NOT in this list because it is
# always computed (MASE raw data does not contain it).
REQUIRED_KEYS = [
    "flight_id", "time", "altitude", "latitude",
    "longitude", "temperature", "pressure", "lwc",
]


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(f"  step01 - {CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  ({config.CAMPAIGN_REGION})")
    print("=" * 65)

    if not DATA_FILE.exists():
        print(f"\n  [FAIL] Data file not found:\n    {DATA_FILE}")
        print(f"\n  Hint: place '{DATA_FILE.name}' under")
        print(f"        {DATA_FILE.parent}")
        sys.exit(1)

    print(f"\n  [OK] Data file located:\n    {DATA_FILE}")
    print(f"        Size: {DATA_FILE.stat().st_size / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # Load (parquet, csv.gz, or csv - based on file suffix)
    # ------------------------------------------------------------------
    print(f"\n  Loading data...")
    suffix = DATA_FILE.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(DATA_FILE)
    elif suffix == ".gz":
        df = pd.read_csv(DATA_FILE, compression="gzip", parse_dates=[TIME_COL])
    elif suffix == ".csv":
        df = pd.read_csv(DATA_FILE, parse_dates=[TIME_COL])
    else:
        print(f"  [FAIL] Unsupported format: {suffix}")
        sys.exit(1)
    print(f"  [OK] Loaded - {len(df):,} rows x {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Required columns check
    # ------------------------------------------------------------------
    print(f"\n  Checking required columns...")
    required_cols = [VAR_MAP[k] for k in REQUIRED_KEYS]
    try:
        check_required_columns(df, required_cols)
        print(f"  [OK] All {len(required_cols)} required columns present")
    except ValueError as e:
        print(f"\n  [FAIL] {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Datetime check (MASE uses 'Datetime' as timestamp column)
    # ------------------------------------------------------------------
    print(f"\n  Checking '{TIME_COL}' is datetime64...")
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        try:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
            print(f"  [OK] Converted to datetime")
        except Exception as e:
            print(f"\n  [FAIL] Cannot convert: {e}")
            sys.exit(1)
    else:
        print(f"  [OK] Already datetime64")

    print(f"        Range: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")

    # ------------------------------------------------------------------
    # GWIU - always computed for MASE using ACTUAL dt per flight (Bug #16
    # fix). If GWIU happens to exist already (e.g., re-running this step),
    # we recompute to ensure consistency.
    # ------------------------------------------------------------------
    alt_col = VAR_MAP["altitude"]
    flt_col = VAR_MAP["flight_id"]

    print(f"\n  Computing '{GWIU_COL}' from altitude derivative (per flight, actual dt)...")
    df = df.sort_values([flt_col, TIME_COL]).reset_index(drop=True)

    # Per-flight altitude diff (NaN at the first row of each flight)
    alt_diff = df.groupby(flt_col, sort=False)[alt_col].diff()

    # Per-flight time diff in seconds (NaN at the first row of each flight)
    dt_actual = (
        df.groupby(flt_col, sort=False)[TIME_COL]
          .diff()
          .dt.total_seconds()
    )

    # Compute Vz = dAlt / dt_actual. Where dt_actual is 0 or NaN, set NaN
    # (avoids division-by-zero and propagates "no info" cleanly).
    with np.errstate(divide="ignore", invalid="ignore"):
        df[GWIU_COL] = np.where(
            (dt_actual > 0) & dt_actual.notna(),
            alt_diff / dt_actual,
            np.nan,
        )

    # Health report on dt regularity (good way to spot data dropouts)
    dt_valid = dt_actual.dropna()
    if len(dt_valid) > 0:
        dt_median = dt_valid.median()
        dt_irregular = ((dt_valid < DT * 0.95) | (dt_valid > DT * 1.05)).sum()
        print(f"  [OK] Time-step regularity:")
        print(f"        Median dt   : {dt_median:.2f} s  (nominal: {DT:.1f} s)")
        print(f"        Irregular dt: {dt_irregular:,} rows ({100*dt_irregular/len(dt_valid):.2f}%)")

    n_valid = df[GWIU_COL].notna().sum()
    print(f"  [OK] '{GWIU_COL}' computed - {n_valid:,} valid values")
    print(f"        Range: [{df[GWIU_COL].min():.3f}, {df[GWIU_COL].max():.3f}] m/s")
    print(f"        Mean : {df[GWIU_COL].mean():.3f} m/s")

    # ------------------------------------------------------------------
    # CAS / CIP bin column detection
    # ------------------------------------------------------------------
    print(f"\n  Detecting size-bin columns...")
    cas_cols = get_cas_columns(df, CAS_PREFIX)
    cip_cols = get_cip_columns(df, CIP_PREFIX)
    print(f"  [OK] CAS bins : {len(cas_cols)} (first: {cas_cols[0]!r}, last: {cas_cols[-1]!r})")
    print(f"  [OK] CIP bins : {len(cip_cols)} (first: {cip_cols[0]!r}, last: {cip_cols[-1]!r})")

    # Sanity check against config
    if len(cas_cols) != config.CAS_N_BINS_NOMINAL:
        print(f"  [WARN] Expected {config.CAS_N_BINS_NOMINAL} CAS bins, found {len(cas_cols)}")
    if len(cip_cols) != config.CIP_N_BINS_NOMINAL:
        print(f"  [WARN] Expected {config.CIP_N_BINS_NOMINAL} CIP bins, found {len(cip_cols)}")

    # ------------------------------------------------------------------
    # Per-flight summary
    # ------------------------------------------------------------------
    flights = sorted(df[flt_col].dropna().unique())
    print(f"\n  Flight summary  ({len(flights)} flights):")
    print(f"  {'Flight':<10} {'Rows':>8} {'Duration (min)':>15}")
    for flt in flights:
        sub = df[df[flt_col] == flt]
        dur_min = len(sub) * DT / 60.0
        print(f"  {flt:<10} {len(sub):>8,} {dur_min:>15.1f}")

    # ------------------------------------------------------------------
    # Variable range summary
    # ------------------------------------------------------------------
    report_vars = {
        "altitude"   : VAR_MAP["altitude"],
        "temperature": VAR_MAP["temperature"],
        "pressure"   : VAR_MAP["pressure"],
        "lwc"        : VAR_MAP["lwc"],
        "GWIU (Vz)"  : GWIU_COL,
    }
    print(f"\n  Variable range summary")
    print(f"  {'Variable':<14} {'Min':>12} {'Max':>12} {'Mean':>12}")
    print(f"  {'-'*52}")
    for label, col in report_vars.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            print(f"  {label:<14} {'(all NaN)':>38}")
        else:
            print(f"  {label:<14} {s.min():>12.3f} {s.max():>12.3f} {s.mean():>12.3f}")

    # ------------------------------------------------------------------
    # NaN summary (top 15 columns with most NaN)
    # ------------------------------------------------------------------
    print(f"\n  NaN summary (top 15 columns with most NaN):")
    print(f"  {'Column':<25} {'NaN count':>10} {'NaN %':>8}")
    print(f"  {'-'*45}")
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False).head(15)
    for col, cnt in nan_counts.items():
        pct = 100.0 * cnt / len(df)
        print(f"  {col:<25} {cnt:>10,} {pct:>7.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_parquet(config.STEP01_PARQUET, index=False)
    print(f"\n  [SAVE] {config.STEP01_PARQUET.name}")
    print(f"         Path: {config.STEP01_PARQUET}")
    print(f"         Size: {config.STEP01_PARQUET.stat().st_size / 1e6:.1f} MB")
    print("\n" + "=" * 65)
    print(f"  step01 PASSED - {CAMPAIGN_NAME} data is structurally valid.")
    print("=" * 65)

# =============================================================================
# step01_load_data.py  -  VOCALS-REx 2008
# =============================================================================
# Purpose:
#   - Read vocals_clean_export.parquet
#   - Use GWIU (vertical velocity) from parquet if present; else compute as
#     Alt.diff() / DT per flight
#   - Validate datetime column and required-column presence
#   - Print health-check report (flights, variable ranges, NaN counts)
#   - Save as STEP01_PARQUET for downstream steps
#
# Does NOT do:
#   - QC masking (handled by step02)
#   - Profile detection (handled by step03)
# =============================================================================

import sys
import pandas as pd

import config
from utils import get_cas_columns, get_cip_columns, check_required_columns


# =============================================================================
# Settings
# =============================================================================
DATA_FILE     = config.PARQUET_FILE
CAMPAIGN_NAME = config.CAMPAIGN_NAME
VAR_MAP       = config.VAR_MAP
DT            = config.DT                # 1.0 s sampling interval

CAS_PREFIX    = VAR_MAP["cas_prefix"]
CIP_PREFIX    = VAR_MAP["cip_prefix"]
GWIU_COL      = VAR_MAP["aircraft_vert_vel_raw"]   # "GWIU"

# Required columns - GWIU is intentionally NOT in this list because it can
# be computed if missing.
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
        df = pd.read_csv(DATA_FILE, compression="gzip", parse_dates=["UTC"])
    elif suffix == ".csv":
        df = pd.read_csv(DATA_FILE, parse_dates=["UTC"])
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
    # Datetime check (VOCALS uses 'UTC' as timestamp column)
    # ------------------------------------------------------------------
    time_col = VAR_MAP["time"]
    print(f"\n  Checking '{time_col}' is datetime64...")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            print(f"  [OK] Converted to datetime")
        except Exception as e:
            print(f"\n  [FAIL] Cannot convert: {e}")
            sys.exit(1)
    else:
        print(f"  [OK] Already datetime64")

    print(f"        Range: {df[time_col].min()} -> {df[time_col].max()}")

    # ------------------------------------------------------------------
    # GWIU - use existing values from parquet if present, else compute
    # from per-flight altitude derivative.
    # ------------------------------------------------------------------
    alt_col = VAR_MAP["altitude"]
    flt_col = VAR_MAP["flight_id"]

    if GWIU_COL in df.columns and df[GWIU_COL].notna().sum() > 0:
        print(f"\n  '{GWIU_COL}' already present in parquet - using existing values")
        n_valid = df[GWIU_COL].notna().sum()
    else:
        print(f"\n  '{GWIU_COL}' missing - computing as Alt.diff()/{DT:.0f}s per flight...")
        df[GWIU_COL] = (
            df.sort_values([flt_col, time_col])
              .groupby(flt_col, sort=False)[alt_col].diff() / DT
        )
        n_valid = df[GWIU_COL].notna().sum()
        print(f"  [OK] '{GWIU_COL}' computed - {n_valid:,} valid values")

    print(f"        Valid: {n_valid:,}  |  Range: "
          f"[{df[GWIU_COL].min():.3f}, {df[GWIU_COL].max():.3f}] m/s")
    print(f"        Mean : {df[GWIU_COL].mean():.3f} m/s")

    # ------------------------------------------------------------------
    # CAS / CIP bin column detection
    # ------------------------------------------------------------------
    print(f"\n  Detecting size-bin columns...")
    cas_cols = get_cas_columns(df, CAS_PREFIX)
    cip_cols = get_cip_columns(df, CIP_PREFIX)
    print(f"  [OK] CAS bins : {len(cas_cols)} (first: {cas_cols[0]!r}, last: {cas_cols[-1]!r})")
    print(f"  [OK] CIP bins : {len(cip_cols)} (first: {cip_cols[0]!r}, last: {cip_cols[-1]!r})")

    # ------------------------------------------------------------------
    # Per-flight summary
    # ------------------------------------------------------------------
    flights = sorted(df[flt_col].dropna().unique())
    print(f"\n  Flight summary  ({len(flights)} flights):")
    print(f"  {'Flight':<14} {'Rows':>8} {'Duration (min)':>15}")
    for flt in flights:
        sub = df[df[flt_col] == flt]
        dur_min = len(sub) * DT / 60.0
        print(f"  {flt:<14} {len(sub):>8,} {dur_min:>15.1f}")

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
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.STEP01_PARQUET, index=False)
    print(f"\n  [SAVE] {config.STEP01_PARQUET.name}")
    print(f"         Size: {config.STEP01_PARQUET.stat().st_size / 1e6:.1f} MB")
    print("\n" + "=" * 65)
    print(f"  step01 PASSED - {CAMPAIGN_NAME} data is structurally valid.")
    print("=" * 65)
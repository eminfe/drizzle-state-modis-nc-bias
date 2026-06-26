# =============================================================================
# step01_load_data.py  -  POST 2008
# =============================================================================
# Purpose:
#   - Read per-flight NetCDF files from LOC_DIR (RFxx_YYYYMMDD_*.nc)
#   - Merge into a single DataFrame with VAR_MAP-aligned column names
#   - Use GWIU (vertical velocity) from NetCDF if present; else compute as
#     Alt.diff() / dt_actual per flight (Bug #16 fix - actual dt, not assumed)
#   - Validate datetime column and required-column presence
#   - Print health-check report (flights, variable ranges, NaN counts)
#   - Save merged DataFrame to STEP01_PARQUET for downstream steps
#
# Does NOT do:
#   - QC masking (handled by step02)
#   - Profile detection (handled by step03)
#
# POST-specific:
#   - Raw data is per-flight NetCDF (not single Parquet like VOCALS/MASE)
#   - NetCDF time variable is SSM (seconds since midnight) or `time`
#   - Flight date encoded in filename: RF01_20080726_*.nc -> 2008-07-26
#   - CCAPS_CAS / CCAPS_CIP are 2D arrays (n_time x n_bins) in the NetCDF
# =============================================================================

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc

import config
from utils import get_cas_columns, get_cip_columns, check_required_columns


# =============================================================================
# Settings
# =============================================================================
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
# NetCDF helpers
# =============================================================================
def _safe_var(ds, var_name, n):
    """Return float64 1D array; if variable missing, return NaN-filled."""
    if var_name in ds.variables:
        arr = np.ma.filled(ds.variables[var_name][:], np.nan).astype(np.float64).flatten()
        return arr
    return np.full(n, np.nan)


def _read_bins(ds, var_name, expected_bins, n):
    """Read 2D bin matrix (n x bins). If missing, return NaN-filled."""
    if var_name in ds.variables:
        data = np.ma.filled(ds.variables[var_name][:], np.nan).astype(np.float64)
        return data.reshape(n, -1)
    return np.full((n, expected_bins), np.nan)


def read_one_flight(nc_path):
    """
    Read one POST per-flight NetCDF into a DataFrame.

    Filename pattern: RF01_20080726_LOCATION.nc  ->  flight_id="RF01", date 2008-07-26.

    Returns DataFrame with columns aligned to config.VAR_MAP and CAS/CIP bin
    columns (CAS_bin_00..CAS_bin_19, CIP_bin_00..CIP_bin_62).
    """
    nc_path = Path(nc_path)
    parts = nc_path.stem.split("_")
    flight_id = parts[0]                            # "RF01"
    raw_date  = parts[1]                            # "20080726"
    formatted = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

    ds = nc.Dataset(str(nc_path))

    # Time variable: POST NetCDFs use either SSM (Seconds Since Midnight) or `time`
    time_var = "SSM" if "SSM" in ds.variables else "time"
    if time_var not in ds.variables:
        ds.close()
        raise ValueError(f"{nc_path.name}: no time variable (SSM or time) found")

    ssm = np.ma.filled(ds.variables[time_var][:], np.nan).astype(np.float64).flatten()
    n   = len(ssm)

    # 1D scalar variables
    galt = _safe_var(ds, "GALT",       n)
    glat = _safe_var(ds, "GLAT",       n)
    glon = _safe_var(ds, "GLON",       n)
    at   = _safe_var(ds, "AT",         n)
    ps   = _safe_var(ds, "PS",         n)
    gwiu = _safe_var(ds, "GWIU",       n)
    lwc  = _safe_var(ds, "LWC_Gerber", n)

    # 2D bin matrices
    cas = _read_bins(ds, "CCAPS_CAS", config.CAS_N_BINS_NOMINAL, n)
    cip = _read_bins(ds, "CCAPS_CIP", config.CIP_N_BINS_NOMINAL, n)
    ds.close()

    # Construct timestamps (datetime64)
    dt = pd.to_datetime(formatted) + pd.to_timedelta(ssm, unit="s")

    df = pd.DataFrame({
        "flight_id" : flight_id,
        "dt"        : dt,
        "GALT"      : galt,
        "GLAT"      : glat,
        "GLON"      : glon,
        "AT"        : at,
        "PS"        : ps,
        "GWIU"      : gwiu,
        "LWC_Gerber": lwc,
    })

    for i in range(cas.shape[1]):
        df[f"CAS_bin_{i:02d}"] = cas[:, i]
    for i in range(cip.shape[1]):
        df[f"CIP_bin_{i:02d}"] = cip[:, i]

    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(f"  step01 - {CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  ({config.CAMPAIGN_REGION})")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Locate per-flight NetCDF files
    # ------------------------------------------------------------------
    if not config.LOC_DIR.exists():
        print(f"\n  [FAIL] LOC_DIR not found:\n    {config.LOC_DIR}")
        sys.exit(1)

    files = sorted(glob.glob(str(config.LOC_DIR / "RF*.nc")))
    if not files:
        files = sorted(glob.glob(str(config.LOC_DIR / "RF*.cdf")))
    if not files:
        print(f"\n  [FAIL] No RF*.nc / RF*.cdf files in {config.LOC_DIR}")
        sys.exit(1)

    print(f"\n  [OK] {len(files)} NetCDF files found in")
    print(f"       {config.LOC_DIR}")

    # ------------------------------------------------------------------
    # Read each flight, append to list
    # ------------------------------------------------------------------
    print(f"\n  Reading flights...")
    frames = []
    for f in files:
        try:
            df_f = read_one_flight(f)
            frames.append(df_f)
            print(f"    [OK] {Path(f).name:<40} {len(df_f):>8,} rows")
        except Exception as e:
            print(f"    [FAIL] {Path(f).name}: {e}")

    if not frames:
        print("\n  [FAIL] No flights could be read.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["flight_id", "dt"]).reset_index(drop=True)
    print(f"\n  [OK] Merged - {len(df):,} rows x {len(df.columns)} columns")

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
    # Datetime check (POST uses 'dt' as timestamp column)
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
    # GWIU - use NetCDF values if present, else compute from per-flight
    # altitude derivative using ACTUAL dt (Bug #16 fix).
    # ------------------------------------------------------------------
    alt_col = VAR_MAP["altitude"]
    flt_col = VAR_MAP["flight_id"]

    if GWIU_COL in df.columns and df[GWIU_COL].notna().sum() > 0:
        print(f"\n  '{GWIU_COL}' already present - using NetCDF values")
        n_valid = df[GWIU_COL].notna().sum()
    else:
        print(f"\n  '{GWIU_COL}' missing - computing per-flight from Alt.diff() / dt_actual...")
        df = df.sort_values([flt_col, time_col]).reset_index(drop=True)
        # Compute actual dt per flight (Bug #16 fix - do not assume DT)
        dt_actual = df.groupby(flt_col, sort=False)[time_col].diff().dt.total_seconds()
        d_alt     = df.groupby(flt_col, sort=False)[alt_col].diff()
        df[GWIU_COL] = d_alt / dt_actual
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

    if len(cas_cols) != config.CAS_N_BINS_NOMINAL:
        print(f"  [WARN] Expected {config.CAS_N_BINS_NOMINAL} CAS bins, found {len(cas_cols)}")
    if len(cip_cols) != config.CIP_N_BINS_NOMINAL:
        print(f"  [WARN] Expected {config.CIP_N_BINS_NOMINAL} CIP bins, found {len(cip_cols)}")

    # ------------------------------------------------------------------
    # Per-flight summary
    # ------------------------------------------------------------------
    flights = sorted(df[flt_col].dropna().unique())
    print(f"\n  Flight summary  ({len(flights)} flights):")
    print(f"  {'Flight':<14} {'Rows':>8} {'Duration (min)':>15}")
    for flt in flights:
        sub = df[df[flt_col] == flt]
        dur_min = (sub[time_col].max() - sub[time_col].min()).total_seconds() / 60.0
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
    df.to_parquet(config.STEP01_PARQUET, index=False)
    print(f"\n  [SAVE] {config.STEP01_PARQUET.name}")
    print(f"         Path: {config.STEP01_PARQUET}")
    print(f"         Size: {config.STEP01_PARQUET.stat().st_size / 1e6:.1f} MB")
    print("\n" + "=" * 65)
    print(f"  step01 PASSED - {CAMPAIGN_NAME} data is structurally valid.")
    print(f"  Next: python step02_qc_filtering.py")
    print("=" * 65)

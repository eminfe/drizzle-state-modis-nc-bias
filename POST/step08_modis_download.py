# =============================================================================
# step08_modis_download.py  -  POST 2008
# =============================================================================
# Search and download MODIS L2 cloud product granules (MOD06_L2 from Terra,
# MYD06_L2 from Aqua) covering the time and space of the surviving golden
# profiles after step07 filtering.
#
# WHAT THIS STEP DOES:
#   - Read the filtered golden_case.csv (filtered profile set after step07)
#   - Group profiles by flight_id
#   - For each flight, query NASA Earthdata for MODIS granules with:
#       - Temporal range: profile time window +/- DT_PAD_MIN (typ. 30 min)
#       - Bounding box: profile bbox + BBOX_PAD_DEG (typ. 0.5 deg)
#   - Download HDF files to MODIS_DIR
#   - Save granule log with metadata
#
# WHAT THIS STEP DOES NOT DO:
#   - VZA/SZA filtering (deferred to step09 pixel-level co-location)
#   - Cloud mask validation (step09)
#   - Pixel-aircraft matching (step09)
#
# RATIONALE:
#   step08 casts a wide spatial/temporal net to ensure no candidate granule
#   is missed. step09 then applies strict per-pixel filters:
#       VZA <= 55 deg  (Maddux et al. 2010, Painemal & Zuidema 2011)
#       SZA <= 65 deg  (Grosvenor & Wood 2014)
#       Cloud mask = "confident clear -> confident cloudy"
#       Phase = liquid
#   These are not applied here because:
#     1. We need to download granules first to access pixel-level metadata
#     2. Wide search reduces re-download risk for marginal cases
#     3. step09 sensitivity analysis can vary thresholds without re-downloading
#
# PREREQUISITES:
#   - earthaccess Python package installed
#   - NASA Earthdata account with valid credentials
#   - ~/.netrc configured or manual login at runtime
#
# OUTPUTS:
#   - MODIS HDF files in config.MODIS_DIR
#   - Granule log: config.GRANULE_LOG (CSV)
# =============================================================================

import sys
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

import config


def main():
    print("=" * 70)
    print(f"  step08 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(MODIS Granule Download)")
    print("=" * 70)

    # ----------------------------------------------------------
    # Input check
    # ----------------------------------------------------------
    if not config.STEP04_GOLDEN_CASE_CSV.exists():
        print(f"\n  [FAIL] {config.STEP04_GOLDEN_CASE_CSV.name} not found.")
        print(f"         Run step07 first.")
        sys.exit(1)

    try:
        import earthaccess
    except ImportError:
        print("\n  [FAIL] earthaccess package not installed.")
        print("         pip install earthaccess")
        sys.exit(1)

    # ----------------------------------------------------------
    # Earthdata login
    # ----------------------------------------------------------
    print("\n  Earthdata login...")
    auth = earthaccess.login()
    if not auth.authenticated:
        print("  [FAIL] Earthdata authentication failed.")
        print("         Check ~/.netrc or run earthaccess.login(strategy='interactive')")
        sys.exit(1)
    print("  [OK] Authenticated")

    # ----------------------------------------------------------
    # Load filtered golden_case
    # ----------------------------------------------------------
    df = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    # errors='coerce' makes conversion robust to mixed format strings
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors='coerce')
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True, errors='coerce')

    if df["start_time"].isna().any() or df["end_time"].isna().any():
        n_bad = df["start_time"].isna().sum() + df["end_time"].isna().sum()
        print(f"\n  [WARN] {n_bad} time values failed to parse - check input file")

    print(f"\n  Loaded golden_case: {len(df)} profiles")
    print(f"  Flights           : {sorted(df['flight_id'].unique())}")

    # Required columns from step07 lat/lon merge
    required = ["lat_min", "lat_max", "lon_min", "lon_max"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  [FAIL] Missing columns: {missing}")
        print(f"         Re-run step07 (lat/lon merge).")
        sys.exit(1)

    # Spatial extent
    print(f"  Spatial extent    : "
          f"lat=[{df['lat_min'].min():.2f}, {df['lat_max'].max():.2f}], "
          f"lon=[{df['lon_min'].min():.2f}, {df['lon_max'].max():.2f}]")

    # ----------------------------------------------------------
    # Output dirs
    # ----------------------------------------------------------
    config.MODIS_DIR.mkdir(parents=True, exist_ok=True)
    config.GRANULE_LOG.parent.mkdir(parents=True, exist_ok=True)

    DT_PAD_MIN   = config.MODIS["dt_pad_min"]
    BBOX_PAD_DEG = config.MODIS["bbox_pad_deg"]
    PRODUCTS     = ["MOD06_L2", "MYD06_L2"]   # Terra + Aqua

    print(f"\n  Search parameters:")
    print(f"    Products      : {PRODUCTS}")
    print(f"    Time padding  : +/-{DT_PAD_MIN} minutes")
    print(f"    Bbox padding  : +/-{BBOX_PAD_DEG} degrees")
    print(f"    MODIS_DIR     : {config.MODIS_DIR}")

    # ----------------------------------------------------------
    # Per-flight loop
    # ----------------------------------------------------------
    all_rows = []
    n_flights = df['flight_id'].nunique()

    for i, (flight_id, fdata) in enumerate(df.groupby("flight_id"), start=1):
        print(f"\n  {'='*60}")
        print(f"  [{i}/{n_flights}] Flight {flight_id}  ({len(fdata)} profiles)")

        t_min = fdata["start_time"].min() - pd.Timedelta(minutes=DT_PAD_MIN)
        t_max = fdata["end_time"].max()   + pd.Timedelta(minutes=DT_PAD_MIN)
        bbox = (
            float(fdata["lon_min"].min() - BBOX_PAD_DEG),
            float(fdata["lat_min"].min() - BBOX_PAD_DEG),
            float(fdata["lon_max"].max() + BBOX_PAD_DEG),
            float(fdata["lat_max"].max() + BBOX_PAD_DEG),
        )
        print(f"      Time : {t_min:%Y-%m-%d %H:%M} -> {t_max:%H:%M} UTC")
        print(f"      Bbox : lon=[{bbox[0]:.2f}, {bbox[2]:.2f}]  "
              f"lat=[{bbox[1]:.2f}, {bbox[3]:.2f}]")

        # ----- Search -----
        try:
            granules = earthaccess.search_data(
                short_name=PRODUCTS,
                temporal=(t_min, t_max),
                bounding_box=bbox,
            )
            print(f"      Granules found: {len(granules)}")
        except Exception as e:
            print(f"      [WARN] search error: {e}")
            continue

        if not granules:
            print(f"      No granules - skipping.")
            continue

        # ----- Log granule metadata -----
        for g in granules:
            try:
                t_str  = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
                g_time = pd.to_datetime(t_str.replace("Z", "+00:00"))
            except Exception:
                g_time = None
            try:
                links = g.data_links()
                url   = links[0] if links else ""
                fname = url.split("/")[-1]
            except Exception:
                fname = g["meta"]["native-id"]
                url   = ""
            satellite = "Terra" if "MOD06" in fname else "Aqua"

            all_rows.append({
                "flight_id"   : flight_id,
                "granule_id"  : g["meta"]["native-id"],
                "filename"    : fname,
                "url"         : url,
                "satellite"   : satellite,
                "granule_time": g_time,
            })
            print(f"        [{satellite:5s}] {fname}  {g_time}")

        # ----- Download -----
        print(f"      Downloading {len(granules)} granules...")
        try:
            downloaded = earthaccess.download(granules, str(config.MODIS_DIR))
            print(f"      [OK] {len(downloaded)} files downloaded")
        except Exception as e:
            print(f"      [WARN] download error: {e}")

    # ----------------------------------------------------------
    # Save granule log
    # ----------------------------------------------------------
    df_log = pd.DataFrame(all_rows)
    df_log.to_csv(config.GRANULE_LOG, index=False)

    print(f"\n  {'='*60}")
    print(f"  SUMMARY")
    print(f"  {'='*60}")
    print(f"  Granule log : {config.GRANULE_LOG.name}")
    print(f"  Total rows  : {len(df_log)}")
    if not df_log.empty:
        n_terra = (df_log['satellite'] == 'Terra').sum()
        n_aqua  = (df_log['satellite'] == 'Aqua').sum()
        print(f"    Terra (MOD06) : {n_terra}")
        print(f"    Aqua  (MYD06) : {n_aqua}")

    hdf_files = sorted([f for f in os.listdir(config.MODIS_DIR) if f.endswith(".hdf")])
    total_mb = sum(os.path.getsize(config.MODIS_DIR / f) for f in hdf_files) / 1e6
    print(f"\n  HDF files on disk : {len(hdf_files)}")
    print(f"  Total size        : {total_mb:.0f} MB")

    print("\n" + "=" * 70)
    print(f"  step08 COMPLETE - proceed to step09 (co-location + VZA/SZA filter)")
    print("=" * 70)


if __name__ == "__main__":
    main()
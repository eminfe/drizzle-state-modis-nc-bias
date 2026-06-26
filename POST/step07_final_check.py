# =============================================================================
# step07_final_check.py  -  POST 2008
# =============================================================================
# Final dataset preparation for MODIS comparison. Combines former step07
# (cleanup/ordering) and step07b (filtering) into a single orchestrator.
#
# THREE PHASES:
#
#   Phase 1: CLEANUP
#     - Type fixes (drizzle_flag -> bool, drizzle_regime -> category)
#     - Merge lat/lon statistics into gc
#     - Recompute tau_diff (idempotent)
#     - Order columns logically
#
#   Phase 2: PHYSICAL FILTERING
#     - Drop profiles with f_ad > 1.0 (physically impossible)
#         Painemal & Zuidema (2011) formulation enforces f_ad <= 1
#         by construction; values exceeding unity indicate incomplete
#         profile sampling or cloud-base/top altitude misidentification.
#
#     - Drop profiles with Nd_median <= ND_MIN (sub-detection limit)
#         Grosvenor et al. (2018), Bennartz (2007). MODIS Nd retrieval
#         is reliable for Nd >~ 10 cm^-3. Below ~5 cm^-3, both MODIS
#         retrieval and CAS counting statistics become unreliable, and
#         bias estimates explode (0/0 limit).
#
#   Phase 3: SAVE
#     - Updated gc/gm overwrite step04 outputs (single source of truth)
#     - Filter report saved separately for traceability
#
# REPRODUCIBILITY:
#   For sensitivity analysis with the unfiltered set, re-run from step04
#   onwards (step04 outputs the unfiltered set).
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config


# =============================================================================
# Filter thresholds (Bug #18 fix: read from config for cross-campaign tunability)
# =============================================================================
F_AD_MAX = config.FILTERS["f_ad_max"]    # Painemal & Zuidema (2011) physical constraint
ND_MIN   = config.FILTERS["nd_min"]      # cm^-3 - sub-detection threshold (Bennartz 2007)


# =============================================================================
# Phase 1: Cleanup helpers
# =============================================================================
def fix_types(gm, gc):
    """Convert flags and regime labels to appropriate types."""
    for col in ["drizzle_flag"]:
        if col in gm.columns:
            gm[col] = gm[col].astype(bool)
    for col in ["drizzle_regime", "mixing_regime"]:
        if col in gm.columns:
            gm[col] = gm[col].astype("category")
    for col in ["drizzle_regime", "dominant_mixing_regime"]:
        if col in gc.columns:
            gc[col] = gc[col].astype("category")
    return gm, gc


def add_lat_lon(gm, gc, var_map=config.VAR_MAP):
    """Merge per-profile lat/lon statistics (median, mean, min, max) into gc."""
    col_lat = var_map["latitude"]
    col_lon = var_map["longitude"]
    if col_lat not in gm.columns or col_lon not in gm.columns:
        print(f"  [WARN] {col_lat}/{col_lon} not in gm; lat/lon merge skipped.")
        return gc

    coord = (
        gm.groupby("cloud_id")
          .agg(
              lat_median=(col_lat, "median"),
              lon_median=(col_lon, "median"),
              lat_mean  =(col_lat, "mean"),
              lon_mean  =(col_lon, "mean"),
              lat_min   =(col_lat, "min"),
              lat_max   =(col_lat, "max"),
              lon_min   =(col_lon, "min"),
              lon_max   =(col_lon, "max"),
          )
          .reset_index()
    )
    drop_cols = [c for c in coord.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(coord, on="cloud_id", how="left")
    return gc


def order_columns(gm, gc):
    """Order gc/gm columns logically for readability."""
    gc_order = [
        # ID + time
        "flight_id", "cloud_id", "Selected",
        "start_time", "end_time", "duration_s",
        # GEO
        "lat_median", "lon_median", "lat_mean", "lon_mean",
        "lat_min", "lat_max", "lon_min", "lon_max",
        # Geometry
        "z_base_m", "z_top_m", "H_m",
        # LWC variants
        "Nc_Total_med", "LWC_med",
        "LWC_med_max", "LWC_med_true",
        "Nc_Total_med_notebook", "Nc_Total_med_true",
        # Microphysics
        "Nd_mean", "Nd_median", "Nd_std",
        "re_cas_mean", "re_cas_median", "re_cas_std",
        "re_full_mean", "re_full_median", "re_full_std",
        # Optical
        "tau_main", "tau_full", "tau_diff", "n_tau_points",
        # Drizzle
        "drizzle_fraction", "drizzle_regime",
        "n_total", "n_drizzle", "n_drizzle_points",
        "mean_N_large", "max_N_large",
        "mean_LWC_ratio", "max_LWC_ratio",
        # Vertical drizzle
        "min_z_drizzle", "mean_z_drizzle", "median_z_drizzle",
        "max_z_drizzle", "std_z_drizzle",
        # Mixing
        "f_ad_mean", "f_ad_median", "f_ad_std", "f_ad_min", "f_ad_max",
        "dominant_mixing_regime",
        # Auxiliary
        "k_median", "k_mean", "k_std",
        "c_w_median", "c_w_mean", "c_w_std",
        "LWP_insitu", "cloud_depth",
        # Integrity
        "max_gap_sec", "gap_ratio",
    ]
    gc_order = [c for c in gc_order if c in gc.columns]
    gc_rest  = [c for c in gc.columns if c not in gc_order]
    gc = gc[gc_order + gc_rest]

    gm_order = [
        "flight_id", "cloud_id",
        config.VAR_MAP["time"],
        config.VAR_MAP["altitude"],
        config.VAR_MAP["latitude"],
        config.VAR_MAP["longitude"],
        config.VAR_MAP["temperature"],
        config.VAR_MAP["pressure"],
        config.VAR_MAP["aircraft_vert_vel_raw"],
        config.VAR_MAP["lwc"],
        "Nc_CAS", "Nc_CIP", "Nc_Total",
        "LWC_CIP", "LWC_total", "drizzle_ratio", "N_large", "drizzle_flag",
        "z_norm", "drizzle_regime",
        "LWC_ad", "f_ad", "mixing_regime",
        "k_martin", "c_w",
        "re_cas", "re_full",
        "dz", "tau_layer_main", "tau_layer_full",
    ]
    gm_order = [c for c in gm_order if c in gm.columns]
    gm_rest  = [c for c in gm.columns if c not in gm_order]
    gm = gm[gm_order + gm_rest]

    return gm, gc


# =============================================================================
# Phase 2: Filtering
# =============================================================================
def apply_physical_filters(gc, gm, f_ad_max=F_AD_MAX, nd_min=ND_MIN):
    """
    Apply physical/retrieval filters and return cleaned datasets plus a
    filter report listing every removed profile and the reason.
    """
    n0 = len(gc)
    report_rows = []

    # --- f_ad filter ---
    f_ad_mask = gc["f_ad_mean"] > f_ad_max
    n_f_ad = int(f_ad_mask.sum())
    for _, r in gc[f_ad_mask].iterrows():
        report_rows.append({
            "cloud_id"     : r["cloud_id"],
            "flight_id"    : r["flight_id"],
            "reason"       : "f_ad > 1.0 (physically impossible)",
            "f_ad_mean"    : r["f_ad_mean"],
            "Nd_median"    : r.get("Nd_median", np.nan),
            "H_m"          : r.get("H_m", np.nan),
            "drizzle_regime": r.get("drizzle_regime", np.nan),
        })

    # --- Nd filter (only on profiles that pass f_ad) ---
    nd_mask = (~f_ad_mask) & (gc["Nd_median"] <= nd_min)
    n_nd = int(nd_mask.sum())
    for _, r in gc[nd_mask].iterrows():
        report_rows.append({
            "cloud_id"     : r["cloud_id"],
            "flight_id"    : r["flight_id"],
            "reason"       : f"Nd <= {nd_min:.0f} cm^-3 (sub-detection)",
            "f_ad_mean"    : r["f_ad_mean"],
            "Nd_median"    : r.get("Nd_median", np.nan),
            "H_m"          : r.get("H_m", np.nan),
            "drizzle_regime": r.get("drizzle_regime", np.nan),
        })

    # Combined keep mask
    keep_mask = ~(f_ad_mask | nd_mask)
    keep_ids = gc.loc[keep_mask, "cloud_id"].unique()

    gc_f = gc[keep_mask].copy().reset_index(drop=True)
    gm_f = gm[gm["cloud_id"].isin(keep_ids)].copy().reset_index(drop=True)

    report = pd.DataFrame(report_rows)
    return gc_f, gm_f, report, n_f_ad, n_nd


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step07 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(Final check + filtering)")
    print("=" * 70)

    if not config.STEP04_GOLDEN_CASE_CSV.exists() or not config.STEP04_GOLDEN_MICRO_CSV.exists():
        print(f"\n  [FAIL] Run step05 first.")
        sys.exit(1)

    gc = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    gm = pd.read_csv(config.STEP04_GOLDEN_MICRO_CSV)
    gc["start_time"] = pd.to_datetime(gc["start_time"])
    gc["end_time"]   = pd.to_datetime(gc["end_time"])
    gm[config.VAR_MAP["time"]] = pd.to_datetime(gm[config.VAR_MAP["time"]])
    print(f"\n  [OK] gc: {gc.shape}  |  gm: {gm.shape}")

    # =========================================================================
    # PHASE 1 - Cleanup
    # =========================================================================
    print(f"\n  [PHASE 1] Cleanup")
    print(f"  ---------------------------------------------")

    print(f"\n  [1.1] Type fixes...")
    gm, gc = fix_types(gm, gc)
    print(f"        Types fixed (drizzle_flag bool, regimes category)")

    print(f"\n  [1.2] lat/lon merge...")
    gc = add_lat_lon(gm, gc)
    print(f"        Lat range : [{gc['lat_median'].min():.2f}, {gc['lat_median'].max():.2f}]")
    print(f"        Lon range : [{gc['lon_median'].min():.2f}, {gc['lon_median'].max():.2f}]")

    if "tau_main" in gc.columns and "tau_full" in gc.columns:
        gc["tau_diff"] = gc["tau_main"] - gc["tau_full"]
        print(f"\n  [1.3] tau_diff recomputed (median = {gc['tau_diff'].median():.3f})")

    print(f"\n  [1.4] Column ordering...")
    gm, gc = order_columns(gm, gc)
    print(f"        gc: {len(gc.columns)} cols, gm: {len(gm.columns)} cols")

    # =========================================================================
    # PHASE 2 - Physical filtering
    # =========================================================================
    print(f"\n  [PHASE 2] Physical filtering")
    print(f"  ---------------------------------------------")
    print(f"  Filters applied:")
    print(f"    f_ad > {F_AD_MAX:.1f} : excluded (Painemal 2011 physical constraint)")
    print(f"    Nd  <= {ND_MIN:.1f} cm^-3 : excluded (Bennartz 2007 sub-detection)")

    gc_filt, gm_filt, report, n_f_ad, n_nd = apply_physical_filters(gc, gm)

    n0 = len(gc)
    n1 = len(gc_filt)
    n_removed = n0 - n1
    print(f"\n  Filter results:")
    print(f"    Before        : {n0} profiles")
    print(f"    After         : {n1} profiles")
    print(f"    Removed (f_ad): {n_f_ad}")
    print(f"    Removed (Nd)  : {n_nd}")

    if n_removed > 0:
        print(f"\n  Removed profiles:")
        print(f"    {'cloud_id':<25} {'reason':<40} "
              f"{'f_ad':>5} {'Nd':>6} {'H_m':>5}")
        print(f"    {'-' * 25} {'-' * 40} {'-' * 5} {'-' * 6} {'-' * 5}")
        for _, r in report.iterrows():
            cid = str(r['cloud_id'])[:24]
            reason = str(r['reason'])[:39]
            f_ad = r['f_ad_mean']
            nd = r['Nd_median'] if pd.notna(r['Nd_median']) else 0.0
            h = r['H_m'] if pd.notna(r['H_m']) else 0.0
            print(f"    {cid:<25} {reason:<40} "
                  f"{f_ad:>5.2f} {nd:>6.1f} {h:>5.0f}")

    # Regime distribution after filtering
    print(f"\n  Regime distribution (after filtering):")
    for line in gc_filt["drizzle_regime"].value_counts(dropna=False).to_string().split("\n"):
        print(f"    {line}")

    # Sanity report
    key_cols = ["Nd_median", "re_cas_median", "re_full_median",
                "tau_main", "tau_full", "tau_diff", "f_ad_mean",
                "k_median", "c_w_median", "LWP_insitu", "cloud_depth"]
    key_cols = [c for c in key_cols if c in gc_filt.columns]
    if key_cols:
        print(f"\n  Key variables summary (filtered set):")
        for line in gc_filt[key_cols].describe().round(3).to_string().split("\n"):
            print(f"    {line}")

    # =========================================================================
    # PHASE 3 - Save
    # =========================================================================
    print(f"\n  [PHASE 3] Save")
    print(f"  ---------------------------------------------")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Overwrite the two main analysis files with filtered set.
    # No separate filter-report file: removed profiles are listed in the
    # terminal output above. For traceability across sessions, re-run
    # step07 and inspect the printed log.
    gc_filt.to_csv(config.STEP04_GOLDEN_CASE_CSV,  index=False)
    gm_filt.to_csv(config.STEP04_GOLDEN_MICRO_CSV, index=False)
    print(f"\n  [SAVE] {config.STEP04_GOLDEN_CASE_CSV.name}   ({gc_filt.shape})")
    print(f"  [SAVE] {config.STEP04_GOLDEN_MICRO_CSV.name}  ({gm_filt.shape})")

    print("\n" + "=" * 70)
    print(f"  step07 COMPLETE - Ready for MODIS matching (step08+)")
    print(f"  Final n_{config.CAMPAIGN_NAME} = {len(gc_filt)} profiles")
    print("=" * 70)
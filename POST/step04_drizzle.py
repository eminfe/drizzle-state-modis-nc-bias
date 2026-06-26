# =============================================================================
# step04_drizzle.py  -  POST 2008
# =============================================================================
# Combines former step04a (build base files) and step04 (drizzle classification)
# into a single step. Reads step03 golden profiles and step02 QC parquet,
# produces the two main analysis files used by step05+.
#
# THREE PHASES:
#
#   Phase 1: BUILD BASE FILES
#     - Read step03 golden profiles -> golden_case (gc), one row per profile
#     - Slice 1 Hz QC data per profile time window -> golden_microphysics (gm)
#     - Strip '_merged' suffix from cloud_id for downstream consistency
#
#   Phase 2: DRIZZLE CLASSIFICATION
#     2a) POINT-LEVEL (gm):
#           LWC_CIP       = sum(N_i * vol_i * rho_w)        [g/m^3]
#           LWC_total     = LWC_Gerber + LWC_CIP            [g/m^3]
#           drizzle_ratio = LWC_CIP / LWC_total
#           N_large       = sum(CIP bins where D > d_large_um) * 1000  [L^-1]
#           drizzle_flag  = (drizzle_ratio > 0.10) AND (N_large > 10)
#
#     2b) PROFILE-LEVEL (gc):
#           drizzle_fraction  = n_drizzle_points / n_total_points
#           drizzle_regime    = classify(drizzle_fraction)
#                                non / weak / moderate / heavy
#
#     2c) VERTICAL DRIZZLE METRICS:
#           z_norm           = (z - z_base) / (z_top - z_base)
#                              0 = cloud base, 1 = cloud top
#           min/mean/median/max/std_z_drizzle = z_norm of drizzle points
#
#   Phase 3: MERGE METRICS BACK INTO gc/gm AND SAVE
#
# OUTPUTS:
#   - {CAMPAIGN}_golden_case.csv         (gc with drizzle metrics)
#   - {CAMPAIGN}_golden_microphysics.csv (gm with drizzle_flag, z_norm)
#   - {CAMPAIGN}_profile_drizzle.csv     (drizzle classification summary)
#   - {CAMPAIGN}_profile_vertical.csv    (vertical drizzle metrics summary)
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import get_cas_columns, get_cip_columns


# =============================================================================
# Phase 2 helpers
# =============================================================================
def _cip_bin_volume_m3(d_mid_um):
    """
    Single-droplet volume per CIP bin (m^3) from midpoint diameter (um).
        V = (4/3) * pi * (d/2)^3
    """
    d_mid_m = d_mid_um * 1e-6
    return (4.0 / 3.0) * np.pi * (d_mid_m / 2.0) ** 3


def _large_bin_start_index(d_large_um, d_mid_um):
    """
    Find the first CIP bin index whose midpoint exceeds d_large_um.

    Used to define 'large drops' for drizzle detection. The first CIP bin
    whose midpoint exceeds d_large_um (typically 100 um) marks the start
    of the large-droplet/drizzle regime; all bins from that index onward
    are summed to give N_large.
    """
    return int(np.argmax(d_mid_um > d_large_um))


def compute_point_level_drizzle(gm, drizzle_cfg=config.DRIZZLE,
                                 var_map=config.VAR_MAP,
                                 cip_d_mid_um=config.CIP_D_MID_UM):
    """
    Per-point drizzle indicators.

    LWC_CIP    = sum over CIP bins of (N_i * V_i * rho_w)
                 In CGS-mixed units: N (cm^-3) * V (m^3) * 1e12 = LWC (g/m^3)
                 The 1e12 factor is (cm^-3 -> m^-3) * rho_w (g/m^3) = 1e6 * 1e6.
    drizzle_ratio = LWC_CIP / (LWC_Gerber + LWC_CIP)
    N_large    = sum of CIP bins with D > d_large_um, in L^-1
    drizzle_flag = both ratio AND N_large above threshold.
    """
    col_lwc = var_map["lwc"]
    cip_prefix = var_map["cip_prefix"]

    cip_vol_m3 = _cip_bin_volume_m3(cip_d_mid_um)

    cip_cols = get_cip_columns(gm, cip_prefix)
    if len(cip_cols) != len(cip_d_mid_um):
        print(f"  [WARN] CIP bin count mismatch: gm={len(cip_cols)}, "
              f"config={len(cip_d_mid_um)}. Using first {len(cip_cols)}.")
        cip_d_mid_um = cip_d_mid_um[:len(cip_cols)]
        cip_vol_m3   = cip_vol_m3[:len(cip_cols)]

    # NaN bins (e.g., CIP_bin_62 placeholder) contribute zero
    cip_data = gm[cip_cols].fillna(0).values
    vol_safe = np.where(np.isnan(cip_vol_m3), 0.0, cip_vol_m3)
    gm["LWC_CIP"] = (cip_data * vol_safe * 1e12).sum(axis=1)

    gm["LWC_total"] = gm[col_lwc].fillna(0) + gm["LWC_CIP"]
    gm["drizzle_ratio"] = gm["LWC_CIP"] / (gm["LWC_total"] + 1e-10)

    # Large-drop concentration: only valid (non-NaN) bins above threshold
    valid_d = ~np.isnan(cip_d_mid_um)
    valid_large = valid_d & (cip_d_mid_um > drizzle_cfg["d_large_um"])
    large_cols = [c for c, v in zip(cip_cols, valid_large) if v]
    if large_cols:
        first_large_d = cip_d_mid_um[valid_large][0]
        print(f"      N_large bin start: {large_cols[0]}  "
              f"(D={first_large_d:.1f} um > {drizzle_cfg['d_large_um']:.0f} um)")
    else:
        print(f"      [WARN] No CIP bins above {drizzle_cfg['d_large_um']:.0f} um")
    gm["N_large"] = gm[large_cols].fillna(0).sum(axis=1) * 1000.0  # cm^-3 -> L^-1

    cond_ratio  = gm["drizzle_ratio"] > drizzle_cfg["lwc_ratio_thresh"]
    cond_nlarge = gm["N_large"]       > drizzle_cfg["n_large_thresh"]
    gm["drizzle_flag"] = cond_ratio & cond_nlarge
    return gm


def classify_regime(frac, thresholds=config.DRIZZLE_REGIME_THRESHOLDS):
    """Classify a profile by its drizzle fraction."""
    if pd.isna(frac):
        return np.nan
    if frac < thresholds["non_to_weak"]:
        return "non_drizzling"
    if frac < thresholds["weak_to_mod"]:
        return "weak_drizzling"
    if frac < thresholds["mod_to_heavy"]:
        return "moderate_drizzling"
    return "heavy_drizzling"


def compute_profile_drizzle(gm):
    """Per-profile drizzle metrics: counts, means, fraction, regime."""
    p = (
        gm.groupby("cloud_id")
          .agg(
              n_total        =("drizzle_flag", "count"),
              n_drizzle      =("drizzle_flag", "sum"),
              mean_N_large   =("N_large", "mean"),
              max_N_large    =("N_large", "max"),
              mean_LWC_ratio =("drizzle_ratio", "mean"),
              max_LWC_ratio  =("drizzle_ratio", "max"),
          )
          .reset_index()
    )
    p["drizzle_fraction"] = p["n_drizzle"] / p["n_total"]
    p["drizzle_regime"]   = p["drizzle_fraction"].apply(classify_regime)
    return p


def compute_z_norm(gm, var_map=config.VAR_MAP):
    """
    Normalized in-cloud altitude per profile:
        z_norm = (z - z_base) / (z_top - z_base)
    where z_base, z_top are the per-profile altitude min/max.
    Range: 0 (cloud base) to 1 (cloud top).
    """
    col_alt = var_map["altitude"]
    z_min = gm.groupby("cloud_id")[col_alt].transform("min")
    z_max = gm.groupby("cloud_id")[col_alt].transform("max")
    z_range = z_max - z_min
    gm["z_norm"] = np.where(z_range > 0, (gm[col_alt] - z_min) / z_range, np.nan)
    return gm


def compute_vertical_drizzle(gm):
    """
    z_norm statistics for drizzle-flagged points only.
    Reveals whether drizzle concentrates near cloud base, mid, or top.
    """
    drizzle_pts = gm[gm["drizzle_flag"]].copy()
    if drizzle_pts.empty:
        return pd.DataFrame(columns=[
            "cloud_id", "n_drizzle_points",
            "min_z_drizzle", "mean_z_drizzle", "median_z_drizzle",
            "max_z_drizzle", "std_z_drizzle",
        ])
    pv = (
        drizzle_pts.groupby("cloud_id")
          .agg(
              n_drizzle_points =("z_norm", "count"),
              min_z_drizzle    =("z_norm", "min"),
              mean_z_drizzle   =("z_norm", "mean"),
              median_z_drizzle =("z_norm", "median"),
              max_z_drizzle    =("z_norm", "max"),
              std_z_drizzle    =("z_norm", "std"),
          )
          .reset_index()
    )
    return pv


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step04 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(Build gc/gm + Drizzle classification)")
    print("=" * 70)

    if not config.STEP03_GOLDEN_CSV.exists():
        print(f"\n  [FAIL] {config.STEP03_GOLDEN_CSV} not found.")
        print(f"         Run step03 first.")
        sys.exit(1)

    if not config.STEP02_QC_PARQUET.exists():
        print(f"\n  [FAIL] {config.STEP02_QC_PARQUET} not found.")
        print(f"         Run step02 first.")
        sys.exit(1)

    # =========================================================================
    # PHASE 1 - Build base files (gc + gm)
    # =========================================================================
    print(f"\n  [PHASE 1] Build base files (gc + gm)")
    print(f"  ---------------------------------------------")

    print(f"\n  [1.1] Loading golden profiles from {config.STEP03_GOLDEN_CSV.name}...")
    golden = pd.read_csv(config.STEP03_GOLDEN_CSV)
    golden["start_time"] = pd.to_datetime(golden["start_time"])
    golden["end_time"]   = pd.to_datetime(golden["end_time"])

    # Strip '_merged' suffix from cloud_id (consistency with step05+ joins)
    golden["cloud_id"] = (
        golden["cloud_id"].astype(str)
        .str.replace(r"_merged$", "", regex=True)
    )

    # gc: profile-level summary columns
    base_cols = ["flight_id", "cloud_id", "Selected",
                 "start_time", "end_time", "duration_s",
                 "z_base_m", "z_top_m", "H_m",
                 "Nc_Total_med", "LWC_med"]
    extra_cols = [c for c in ["LWC_med_true", "LWC_med_max",
                              "Nc_Total_med_true", "Nc_Total_med_notebook",
                              "max_gap_sec", "gap_ratio"]
                  if c in golden.columns]
    gc = golden[base_cols + extra_cols].copy().reset_index(drop=True)
    print(f"        gc shape  : {gc.shape}")
    print(f"        Flights   : {sorted(gc['flight_id'].unique())}")

    print(f"\n  [1.2] Loading {config.STEP02_QC_PARQUET.name}...")
    df_all = pd.read_parquet(config.STEP02_QC_PARQUET)
    df_all[config.VAR_MAP["time"]] = pd.to_datetime(df_all[config.VAR_MAP["time"]])
    print(f"        {len(df_all):,} rows x {df_all.shape[1]} columns")

    cas_cols = get_cas_columns(df_all, config.VAR_MAP["cas_prefix"])
    cip_cols = get_cip_columns(df_all, config.VAR_MAP["cip_prefix"])

    print(f"\n  [1.3] Slicing point data per profile...")
    col_t   = config.VAR_MAP["time"]
    col_flt = config.VAR_MAP["flight_id"]

    essential = [
        "DATE", "UTC", col_t, col_flt,
        config.VAR_MAP["altitude"],
        config.VAR_MAP["latitude"],
        config.VAR_MAP["longitude"],
        config.VAR_MAP["temperature"],
        config.VAR_MAP["pressure"],
        config.VAR_MAP["aircraft_vert_vel_raw"],
        config.VAR_MAP["lwc"],
        "Nc_CAS", "Nc_CIP", "Nc_Total",
    ]
    essential = [c for c in essential if c in df_all.columns]
    keep_cols = essential + cas_cols + cip_cols

    segments = []
    for _, prof in gc.iterrows():
        seg = df_all[
            (df_all[col_flt] == prof["flight_id"]) &
            (df_all[col_t]   >= prof["start_time"]) &
            (df_all[col_t]   <= prof["end_time"])
        ][keep_cols].copy()
        seg["cloud_id"] = prof["cloud_id"]
        segments.append(seg)

    if not segments:
        print("\n  [FAIL] No segments could be sliced.")
        sys.exit(1)

    gm = pd.concat(segments, ignore_index=True)
    print(f"        gm shape       : {gm.shape}")
    print(f"        Profile count  : {gm['cloud_id'].nunique()}")
    pts_per_profile = gm.groupby('cloud_id').size().describe().round(1)
    print(f"        Points/profile : "
          f"min={pts_per_profile['min']:.0f}  "
          f"median={pts_per_profile['50%']:.0f}  "
          f"max={pts_per_profile['max']:.0f}")

    # =========================================================================
    # PHASE 2 - Drizzle classification
    # =========================================================================
    print(f"\n  [PHASE 2] Drizzle classification")
    print(f"  ---------------------------------------------")

    print(f"\n  [2.1] Point-level drizzle flag")
    print(f"        Thresholds:")
    print(f"          d_large_um       = {config.DRIZZLE['d_large_um']:.0f} um")
    print(f"          n_large_thresh   = {config.DRIZZLE['n_large_thresh']:.1f} L^-1")
    print(f"          lwc_ratio_thresh = {config.DRIZZLE['lwc_ratio_thresh']:.2f}")
    gm = compute_point_level_drizzle(gm)
    n_pts = len(gm)
    n_dr  = int(gm["drizzle_flag"].sum())
    print(f"        Total points     : {n_pts:,}")
    print(f"        Drizzle points   : {n_dr:,}  ({100*n_dr/n_pts:.1f}%)")

    print(f"\n  [2.2] Profile-level drizzle metrics + regime classification")
    profile_drizzle = compute_profile_drizzle(gm)
    profile_drizzle.to_csv(config.STEP04_PROFILE_DRIZZLE_CSV, index=False)
    print(f"        Regime distribution:")
    for line in profile_drizzle["drizzle_regime"].value_counts().to_string().split("\n"):
        print(f"          {line}")
    print(f"        [SAVE] {config.STEP04_PROFILE_DRIZZLE_CSV.name}")

    print(f"\n  [2.3] z_norm + vertical drizzle metrics")
    gm = compute_z_norm(gm)
    profile_vertical = compute_vertical_drizzle(gm)
    profile_vertical = profile_vertical.merge(
        profile_drizzle[["cloud_id", "drizzle_regime"]],
        on="cloud_id", how="left"
    )
    profile_vertical.to_csv(config.STEP04_PROFILE_VERTICAL_CSV, index=False)
    if not profile_vertical.empty:
        print(f"        z_drizzle by regime (mean of min/mean/max):")
        summ = (profile_vertical
                .groupby("drizzle_regime", observed=True)
                [["min_z_drizzle", "mean_z_drizzle", "max_z_drizzle"]]
                .mean().round(2))
        for line in summ.to_string().split("\n"):
            print(f"          {line}")
    else:
        print(f"        No drizzle points - vertical metrics empty.")
    print(f"        [SAVE] {config.STEP04_PROFILE_VERTICAL_CSV.name}")

    # =========================================================================
    # PHASE 3 - Merge metrics back into gc/gm and save
    # =========================================================================
    print(f"\n  [PHASE 3] Merge metrics and save")
    print(f"  ---------------------------------------------")

    gm = gm.merge(profile_drizzle[["cloud_id", "drizzle_regime"]],
                  on="cloud_id", how="left")

    drizzle_cols = ["cloud_id", "n_total", "n_drizzle",
                    "mean_N_large", "max_N_large",
                    "mean_LWC_ratio", "max_LWC_ratio",
                    "drizzle_fraction", "drizzle_regime"]
    vertical_cols = ["cloud_id", "n_drizzle_points",
                     "min_z_drizzle", "mean_z_drizzle", "median_z_drizzle",
                     "max_z_drizzle", "std_z_drizzle"]

    # idempotent merge: drop existing columns first
    drop_from_gc = (
        [c for c in drizzle_cols  if c != "cloud_id" and c in gc.columns] +
        [c for c in vertical_cols if c != "cloud_id" and c in gc.columns]
    )
    if drop_from_gc:
        gc = gc.drop(columns=drop_from_gc)

    gc = gc.merge(profile_drizzle [drizzle_cols],  on="cloud_id", how="left")
    gc = gc.merge(profile_vertical[vertical_cols], on="cloud_id", how="left")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gc.to_csv(config.STEP04_GOLDEN_CASE_CSV,   index=False)
    gm.to_csv(config.STEP04_GOLDEN_MICRO_CSV,  index=False)

    print(f"  [SAVE] {config.STEP04_GOLDEN_CASE_CSV.name}   ({gc.shape})")
    print(f"  [SAVE] {config.STEP04_GOLDEN_MICRO_CSV.name}  ({gm.shape})")
    
    print("\n" + "=" * 70)
    print(f"  step04 COMPLETE")
    print(f"    gc: {gc.shape}  |  gm: {gm.shape}")
    print(f"    {len(profile_drizzle)} profiles classified into drizzle regimes")
    print("=" * 70)
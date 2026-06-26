# =============================================================================
# step03_vertical_profiles.py  -  VOCALS-REx 2008
# =============================================================================
# *** PIPELINE'S MOST CRITICAL STEP - Vertical Cloud Profile Identification ***
#
# Goal: Identify vertical penetrations through marine stratocumulus clouds
# from continuous 1 Hz aircraft data, distinguishing them from horizontal
# in-cloud transects and isolating high-quality profiles for analysis.
#
# FOUR-STAGE FILTER:
#
#   1. CLOUD-CORE POINT IDENTIFICATION  (Section A)
#      Per flight:
#        a) GALT.rolling(5, center=True).mean() = Alt_Smooth
#        b) Vz = Alt_Smooth.diff() / dt_actual    [m/s]
#        c) is_moving = |Vz| > 0.5 m/s            (excludes horizontal flight)
#        d) Assign block_id to consecutive is_moving=True regions
#        e) Within each block, find cloud-core points
#                                  (LWC > 0.05 AND Nc_Total > 10)
#        f) Keep blocks with >= 10 cloud-core points
#
#   2. SEGMENT SUMMARIZATION  (Section B)
#      Within Section A output:
#        a) New profile_group_id when time_diff > 120 s
#        b) Per segment: start/end, z_base/z_top, H, duration,
#                         Nc_Total_med, LWC_med
#        c) cloud_id = "{flight_id}_P{NN}" (in-flight sequence)
#        d) Selected = (H >= final_min_depth_m) AND (duration >= 60 s)
#                        intermediate filter; raised to 120 s in Section C
#                        for VOCALS, both stages use 60 s (broken cloud regime)
#
#   3. SUPER PROFILE MERGE  (Section C)
#      Treat nearby segments as a single cloud body:
#        a) Merge two segments if same flight + gap <= stitch_gap_seconds
#        b) z_base = min, z_top = max, H = z_top - z_base
#        c) duration = end - start
#        d) LWC_med computed two ways:
#             - LWC_med_max  : max(s1, s2) - reproduces notebook style
#             - LWC_med_true : true median across all raw cloud-core points
#                              (statistically correct)
#           Same applies to Nc_Total (_notebook vs _true)
#        e) Final QC: H >= final_min_depth_m
#                     AND duration >= final_min_duration_s
#                     AND LWC_med_true > final_min_lwc
#                     (falls back to LWC_med_max if LWC_med_true is NaN)
#
#   4. INTEGRITY CHECK  (Section D)
#      1 Hz data may have sensor dropouts within a single profile.
#        a) Extract raw points covered by the profile time window
#        b) max_gap = largest consecutive time gap (seconds)
#        c) gap_ratio = max_gap / total_span
#        d) gap_ratio <= 0.10 AND n_points >= 15 -> PASS
#
# OUTPUTS:
#   - STEP03_PROFILE_POINTS_PARQUET : all cloud-core points
#   - STEP03_PROFILE_SUMMARY_CSV    : all candidate segments (Section B)
#   - STEP03_SUPER_PROFILES_CSV     : merged super profiles (Section C)
#   - STEP03_GOLDEN_CSV             : Selected=True + integrity OK -> GOLDEN
#
# All numerical thresholds come from config.PROFILE; no magic numbers.
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import (
    compute_vertical_velocity,
    compute_max_gap_ratio,
    compute_span_score,
)


# =============================================================================
# Section A - Cloud-core point identification
# =============================================================================
def identify_cloud_core_points(df, params=config.PROFILE, var_map=config.VAR_MAP):
    """
    Find cloud-core points within ascending/descending segments per flight.

    Logic:
      1. Compute Vz per flight using rolling-mean smoothing.
      2. is_moving = |Vz| > vz_threshold.
      3. Within each consecutive is_moving block, select cloud-core points
         (LWC > lwc_threshold AND Nc_Total > nc_threshold).
      4. Keep blocks containing >= min_profile_pts cloud-core points.

    Returns
    -------
    df_core : DataFrame
        Cloud-core points with added 'profile_tag' column ('{flight}_B{block}').
    """
    col_alt = var_map["altitude"]
    col_t   = var_map["time"]
    col_lwc = var_map["lwc"]
    col_flt = var_map["flight_id"]

    lwc_thr = params["lwc_threshold"]
    nc_thr  = params["nc_threshold"]
    vz_thr  = params["vz_threshold"]
    win     = params["alt_smooth_win"]
    minpts  = params["min_profile_pts"]

    out_blocks = []

    for fid, df_f in df.groupby(col_flt, sort=False):
        df_f = df_f.sort_values(col_t).copy()

        # 1. Vz computation (smoothed altitude derivative)
        df_f = compute_vertical_velocity(
            df_f, alt_col=col_alt, time_col=col_t, smooth_win=win
        )

        # 2. is_moving (NaN -> False, protects against edge effects)
        df_f["is_moving"] = (df_f["Vz"].abs() > vz_thr).fillna(False)

        # 3. Assign block_id to consecutive is_moving regions
        df_f["block_id"] = (df_f["is_moving"] != df_f["is_moving"].shift()).cumsum()

        # 4. Cloud-core points within each moving block
        for bid, block in df_f.groupby("block_id"):
            if not block["is_moving"].iloc[0]:
                continue   # horizontal-flight block

            mask_core = (block[col_lwc] > lwc_thr) & (block["Nc_Total"] > nc_thr)
            core = block[mask_core].copy()
            if len(core) < minpts:
                continue

            core["profile_tag"] = f"{fid}_B{bid}"
            out_blocks.append(core)

    if not out_blocks:
        print("  [WARN] No cloud-core profile candidates found.")
        return pd.DataFrame()

    df_core = pd.concat(out_blocks, ignore_index=True)
    return df_core


# =============================================================================
# Section B - Segment summarization
# =============================================================================
def create_profile_summary(df_core, params=config.PROFILE, var_map=config.VAR_MAP):
    """
    Split cloud-core points into segments by time gap, summarize each segment.

    Logic:
      time_diff > summary_gap_seconds -> new segment.
      Each segment: z_base, z_top, H, duration, medians.
      cloud_id = '{flight_id}_P{NN}'.

    Returns
    -------
    df_summary : DataFrame
    """
    col_alt = var_map["altitude"]
    col_t   = var_map["time"]
    col_lwc = var_map["lwc"]
    col_flt = var_map["flight_id"]
    nc_col  = "Nc_Total"

    gap_s   = params["summary_gap_seconds"]
    H_min   = params["final_min_depth_m"]
    # Section B intermediate filter (loose pre-merge screen).
    # Falls back to final_min_duration_s if summary_min_duration_s is not
    # specified, preserving backward compatibility with single-stage configs.
    dur_min = params.get("summary_min_duration_s", params["final_min_duration_s"])

    df = df_core.sort_values([col_flt, col_t]).copy()
    df["time_diff"] = df.groupby(col_flt)[col_t].diff().dt.total_seconds().fillna(0)

    cond_new = (df[col_flt] != df[col_flt].shift()) | (df["time_diff"] > gap_s)
    df["profile_group_id"] = cond_new.cumsum()

    rows = []
    for gid, g in df.groupby("profile_group_id"):
        if g.empty:
            continue
        start_t = g[col_t].iloc[0]
        end_t   = g[col_t].iloc[-1]
        dur     = (end_t - start_t).total_seconds()
        z_min   = g[col_alt].min()
        z_max   = g[col_alt].max()
        H       = z_max - z_min

        rows.append({
            "flight_id"   : g[col_flt].iloc[0],
            "Selected"    : (H >= H_min) and (dur >= dur_min),
            "start_time"  : start_t,
            "end_time"    : end_t,
            "duration_s"  : dur,
            "z_base_m"    : z_min,
            "z_top_m"     : z_max,
            "H_m"         : H,
            "Nc_Total_med": g[nc_col].median(),
            "LWC_med"     : g[col_lwc].median(),
        })

    df_summary = pd.DataFrame(rows)
    if df_summary.empty:
        return df_summary

    # cloud_id: per-flight sequence
    df_summary["cloud_id"] = (
        df_summary.groupby("flight_id").cumcount() + 1
    )
    df_summary["cloud_id"] = (
        df_summary["flight_id"] + "_P" +
        df_summary["cloud_id"].astype(str).str.zfill(2)
    )

    cols = ["flight_id", "cloud_id", "Selected", "start_time", "end_time",
            "duration_s", "z_base_m", "z_top_m", "H_m",
            "Nc_Total_med", "LWC_med"]
    return df_summary[cols]


# =============================================================================
# Section C - Super profile merge
# =============================================================================
def merge_to_super_profiles(df_summary, df_qc_raw=None,
                             params=config.PROFILE, var_map=config.VAR_MAP):
    """
    Merge nearby segments within a flight into super profiles.

    Two segments are merged when:
      - same flight_id AND
      - gap between end_time(s1) and start_time(s2) <= stitch_gap_seconds

    LWC_med is computed two ways:
      - LWC_med_max  : max(s1, s2). Notebook-style (reproducibility).
      - LWC_med_true : true median across all raw cloud-core points within
                       the merged time window. Statistically correct.

    Same logic applied to Nc_Total (_notebook vs _true variants).

    Final 'Selected' filter uses LWC_med_true if available; falls back to
    LWC_med_max otherwise.

    Parameters
    ----------
    df_summary : DataFrame
        Output of Section B.
    df_qc_raw : DataFrame, optional
        Step02 QC parquet (raw 1 Hz data). If not provided, only _max
        columns are populated and _true columns remain NaN.
    """
    if df_summary.empty:
        return df_summary

    stitch_s = params["stitch_gap_seconds"]
    H_min    = params["final_min_depth_m"]
    dur_min  = params["final_min_duration_s"]
    lwc_min  = params["final_min_lwc"]
    lwc_thr  = params["lwc_threshold"]
    nc_thr   = params["nc_threshold"]
    # span_score filter (sawtooth-friendly vertical-penetration quality).
    # Default 0.0 = disabled (VOCALS/POST behavior); MASE config sets ~0.4.
    min_span = params.get("min_span_score", 0.0)

    col_t   = var_map["time"]
    col_flt = var_map["flight_id"]
    col_lwc = var_map["lwc"]
    col_alt = var_map["altitude"]
    col_nc  = "Nc_Total"

    df = df_summary.sort_values(["flight_id", "start_time"]).reset_index(drop=True)
    merged = []
    cur = df.iloc[0].to_dict()
    cur["LWC_med_max"]            = cur["LWC_med"]
    cur["Nc_Total_med_notebook"]  = cur["Nc_Total_med"]

    for i in range(1, len(df)):
        nxt = df.iloc[i]
        same_flt = (cur["flight_id"] == nxt["flight_id"])
        gap_s    = (nxt["start_time"] - cur["end_time"]).total_seconds()

        if same_flt and (gap_s <= stitch_s):
            # MERGE - update geometry consistently
            cur["end_time"]      = nxt["end_time"]
            cur["z_base_m"]      = min(cur["z_base_m"], nxt["z_base_m"])
            cur["z_top_m"]       = max(cur["z_top_m"], nxt["z_top_m"])
            cur["H_m"]           = cur["z_top_m"] - cur["z_base_m"]
            cur["duration_s"]    = (cur["end_time"] - cur["start_time"]).total_seconds()
            # Notebook-style aggregates (reproducibility):
            cur["LWC_med_max"]            = float(np.nanmax([cur["LWC_med_max"], nxt["LWC_med"]]))
            cur["Nc_Total_med_notebook"]  = float(np.nanmedian([cur["Nc_Total_med_notebook"], nxt["Nc_Total_med"]]))
            if "_merged" not in str(cur["cloud_id"]):
                cur["cloud_id"]  = str(cur["cloud_id"]) + "_merged"
        else:
            merged.append(cur)
            cur = nxt.to_dict()
            cur["LWC_med_max"]            = cur["LWC_med"]
            cur["Nc_Total_med_notebook"]  = cur["Nc_Total_med"]
    merged.append(cur)

    df_m = pd.DataFrame(merged)

    # True medians and span_score from raw cloud-core points.
    # span_score = (z_max - z_min) / sum(|d_alt|) — sawtooth-friendly
    # vertical-penetration quality metric (see utils.compute_span_score).
    if df_qc_raw is not None:
        true_lwc    = []
        true_nc     = []
        span_scores = []
        for _, prof in df_m.iterrows():
            seg = df_qc_raw[
                (df_qc_raw[col_flt] == prof["flight_id"]) &
                (df_qc_raw[col_t]   >= prof["start_time"]) &
                (df_qc_raw[col_t]   <= prof["end_time"])
            ]
            # Cloud-core points (same definition as Section A)
            core = seg[(seg[col_lwc] > lwc_thr) & (seg[col_nc] > nc_thr)]
            if len(core) > 0:
                core_sorted = core.sort_values(col_t)
                true_lwc.append(core_sorted[col_lwc].median())
                true_nc.append(core_sorted[col_nc].median())
                span_scores.append(compute_span_score(core_sorted[col_alt].values))
            else:
                true_lwc.append(np.nan)
                true_nc.append(np.nan)
                span_scores.append(np.nan)
        df_m["LWC_med_true"]      = true_lwc
        df_m["Nc_Total_med_true"] = true_nc
        df_m["span_score"]        = span_scores
    else:
        df_m["LWC_med_true"]      = np.nan
        df_m["Nc_Total_med_true"] = np.nan
        df_m["span_score"]        = np.nan

    # Backward-compatibility aliases (= notebook style, same as _max / _notebook)
    df_m["LWC_med"]      = df_m["LWC_med_max"]
    df_m["Nc_Total_med"] = df_m["Nc_Total_med_notebook"]

    # Final QC: prefer LWC_med_true; fall back to LWC_med_max if NaN.
    # min_span_score guards against level-leg transects that pass H/dur
    # filters but are not real vertical penetrations. Default 0.0 in
    # config disables the filter (VOCALS/POST behavior).
    lwc_for_filter = df_m["LWC_med_true"].fillna(df_m["LWC_med_max"])
    span_for_filter = df_m["span_score"].fillna(1.0)  # NaN -> permit
    df_m["Selected"] = (
        (df_m["H_m"] >= H_min) &
        (df_m["duration_s"] >= dur_min) &
        (lwc_for_filter > lwc_min) &
        (span_for_filter >= min_span)
    )

    # Column ordering - place _true and _max side by side
    cols_pref = [
        "flight_id", "cloud_id", "Selected",
        "start_time", "end_time", "duration_s",
        "z_base_m", "z_top_m", "H_m",
        "LWC_med", "LWC_med_max", "LWC_med_true",
        "Nc_Total_med", "Nc_Total_med_notebook", "Nc_Total_med_true",
        "span_score",
    ]
    cols_pref = [c for c in cols_pref if c in df_m.columns]
    rest      = [c for c in df_m.columns if c not in cols_pref]
    return df_m[cols_pref + rest]


# =============================================================================
# Section D - Profile integrity check
# =============================================================================
def check_profile_integrity(df_super, df_qc_raw,
                             params=config.PROFILE, var_map=config.VAR_MAP):
    """
    Verify that the profile time window contains continuous 1 Hz data with
    no large sensor dropouts.

    Logic:
      Extract raw points within the profile time window. Compute the largest
      consecutive time gap. gap_ratio = max_gap / total_span <= 0.10 -> PASS.

    Parameters
    ----------
    df_super : DataFrame
        Super profile candidates (Section C output, Selected=True).
    df_qc_raw : DataFrame
        Step02 QC parquet (raw 1 Hz data after physical QC masking).
    """
    if df_super.empty:
        return df_super

    col_t   = var_map["time"]
    col_flt = var_map["flight_id"]

    min_pts   = params["integrity_min_pts"]
    max_gap_r = params["integrity_max_gap_ratio"]

    rows = []
    for _, prof in df_super.iterrows():
        seg = df_qc_raw[
            (df_qc_raw[col_flt] == prof["flight_id"]) &
            (df_qc_raw[col_t]   >= prof["start_time"]) &
            (df_qc_raw[col_t]   <= prof["end_time"])
        ].sort_values(col_t)

        if len(seg) < min_pts:
            continue

        max_gap_sec, gap_ratio = compute_max_gap_ratio(seg[col_t])
        if gap_ratio > max_gap_r:
            continue

        d = prof.to_dict()
        d["max_gap_sec"] = round(max_gap_sec, 2)
        d["gap_ratio"]   = round(gap_ratio, 4)
        rows.append(d)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step03 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(Vertical Cloud Profile Identification)")
    print("=" * 70)

    if not config.STEP02_QC_PARQUET.exists():
        print(f"\n  [FAIL] step02 parquet not found: {config.STEP02_QC_PARQUET}")
        print(f"         Run step02_qc_filtering.py first.")
        sys.exit(1)

    print(f"\n  Loading {config.STEP02_QC_PARQUET.name}...")
    df_qc = pd.read_parquet(config.STEP02_QC_PARQUET)
    df_qc[config.VAR_MAP["time"]] = pd.to_datetime(df_qc[config.VAR_MAP["time"]])
    print(f"  [OK] {len(df_qc):,} rows x {len(df_qc.columns)} columns")

    # ----------------------------------------------------------
    # SECTION A - Cloud-core points
    # ----------------------------------------------------------
    print(f"\n  [A] Cloud-core point identification "
          f"(|Vz|>{config.PROFILE['vz_threshold']} m/s, "
          f"LWC>{config.PROFILE['lwc_threshold']}, "
          f"Nc>{config.PROFILE['nc_threshold']})")
    df_core = identify_cloud_core_points(df_qc)
    if df_core.empty:
        print("  [FAIL] No cloud-core points found.")
        sys.exit(1)
    n_blocks = df_core["profile_tag"].nunique()
    print(f"      Cloud-core points        : {len(df_core):,}")
    print(f"      Candidate blocks         : {n_blocks}")

    # Save cloud-core points as parquet
    df_core.to_parquet(config.STEP03_PROFILE_POINTS_PARQUET, index=False)
    print(f"      [SAVE] {config.STEP03_PROFILE_POINTS_PARQUET.name}")

    # ----------------------------------------------------------
    # SECTION B - Segment summary
    # ----------------------------------------------------------
    print(f"\n  [B] Segment summarization "
          f"(gap > {config.PROFILE['summary_gap_seconds']}s -> new segment)")
    df_summary = create_profile_summary(df_core)
    _dur_b = config.PROFILE.get("summary_min_duration_s",
                                config.PROFILE["final_min_duration_s"])
    print(f"      Total candidate segments : {len(df_summary)}")
    print(f"      H >= {config.PROFILE['final_min_depth_m']:.0f} m & "
          f"dur >= {_dur_b:.0f} s "
          f"(Section B Selected): {df_summary['Selected'].sum()}")
    df_summary.to_csv(config.STEP03_PROFILE_SUMMARY_CSV, index=False)
    print(f"      [SAVE] {config.STEP03_PROFILE_SUMMARY_CSV.name}")

    # ----------------------------------------------------------
    # SECTION C - Super profile merge (LWC_med_true from raw points)
    # ----------------------------------------------------------
    print(f"\n  [C] Super profile merge "
          f"(gap <= {config.PROFILE['stitch_gap_seconds']}s -> merge)")
    df_super = merge_to_super_profiles(df_summary, df_qc_raw=df_qc)
    n_merged_only = (df_super["cloud_id"].astype(str).str.contains("_merged")).sum()
    _min_span = config.PROFILE.get("min_span_score", 0.0)
    print(f"      Super profiles            : {len(df_super)}")
    print(f"      Merged (_merged)          : {n_merged_only}")
    print(f"      Final QC passed "
          f"(H>={config.PROFILE['final_min_depth_m']:.0f}, "
          f"dur>={config.PROFILE['final_min_duration_s']:.0f}, "
          f"LWC>{config.PROFILE['final_min_lwc']:.2f}, "
          f"span>={_min_span:.2f}): "
          f"{df_super['Selected'].sum()}")

    # span_score breakdown (vertical-penetration quality filter diagnostics)
    if _min_span > 0 and "span_score" in df_super.columns:
        span_passed_geom = (
            (df_super["H_m"] >= config.PROFILE["final_min_depth_m"]) &
            (df_super["duration_s"] >= config.PROFILE["final_min_duration_s"])
        )
        n_geom = int(span_passed_geom.sum())
        n_span_reject = int(
            (span_passed_geom & (df_super["span_score"].fillna(1.0) < _min_span)).sum()
        )
        if n_geom > 0:
            print(f"      span_score filter         : "
                  f"rejected {n_span_reject}/{n_geom} geom-passing profiles "
                  f"(< {_min_span:.2f})")

    # LWC_med_max vs LWC_med_true comparison (only differs in merged profiles)
    merged_mask = df_super["cloud_id"].astype(str).str.contains("_merged")
    if merged_mask.any():
        cmp = df_super.loc[merged_mask, ["cloud_id", "LWC_med_max", "LWC_med_true"]]
        cmp["abs_diff"] = (cmp["LWC_med_max"] - cmp["LWC_med_true"]).abs()
        n_diverge = (cmp["abs_diff"] > 0.005).sum()  # > 0.005 g/m^3 = meaningful
        print(f"      LWC_med_max vs _true diverging (>0.005): "
              f"{n_diverge}/{merged_mask.sum()}")

    df_super.to_csv(config.STEP03_SUPER_PROFILES_CSV, index=False)
    print(f"      [SAVE] {config.STEP03_SUPER_PROFILES_CSV.name}")

    # ----------------------------------------------------------
    # SECTION D - Integrity check (only on Selected=True profiles)
    #
    # Toggled by config.PROFILE["integrity_check_enabled"]:
    #   - True  (VOCALS, POST 1 Hz)  : run gap-ratio check, drop gappy profiles
    #   - False (MASE 10 s)          : skip — Section C output IS the golden set
    #
    # The integrity check is meaningful only when the sampling rate is fast
    # enough that intra-profile sensor dropouts can be detected. At 10 s
    # cadence the analysis becomes noise-dominated and uninformative.
    # ----------------------------------------------------------
    candidates = df_super[df_super["Selected"]].copy()
    if config.PROFILE.get("integrity_check_enabled", True):
        print(f"\n  [D] Profile integrity check "
              f"(gap_ratio <= {config.PROFILE['integrity_max_gap_ratio']*100:.0f}%)")
        print(f"      QC candidates             : {len(candidates)}")
        df_golden = check_profile_integrity(candidates, df_qc)
        print(f"      Integrity passed          : {len(df_golden)}  <- GOLDEN")
        print(f"      Eliminated (gappy)        : {len(candidates) - len(df_golden)}")
    else:
        print(f"\n  [D] Profile integrity check  -  SKIPPED ({config.CAMPAIGN_NAME} "
              f"{config.DT:.0f} s sampling)")
        print(f"      Section C Selected=True profiles are taken as the golden set.")
        df_golden = candidates.copy()
        print(f"      Golden profiles            : {len(df_golden)}  <- GOLDEN")

    if df_golden.empty:
        print("\n  [FAIL] No golden profiles found.")
        sys.exit(1)

    df_golden.to_csv(config.STEP03_GOLDEN_CSV, index=False)
    print(f"      [SAVE] {config.STEP03_GOLDEN_CSV.name}")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  step03 COMPLETE - {len(df_golden)} golden profiles ready.")
    print("=" * 70)
    print(f"\n  Flight distribution:")
    print(df_golden["flight_id"].value_counts().sort_index().to_string())
    print(f"\n  H_m statistics:")
    print(df_golden["H_m"].describe().round(1).to_string())
    print(f"\n  duration_s statistics:")
    print(df_golden["duration_s"].describe().round(1).to_string())
    print()
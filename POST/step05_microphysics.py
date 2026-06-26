# =============================================================================
# step05_microphysics.py  -  POST 2008
# =============================================================================
# Compute in-situ microphysics for each golden profile:
#
#   [1] f_ad + mixing regime    (Painemal & Zuidema 2011 method)
#   [2] re_cas + re_full        (effective radius from CAS only and CAS+CIP)
#   [3] Profile Nd, re_cas, re_full statistics
#   [4] tau_main + tau_full     (optical thickness from re_cas and re_full)
#   [5] k_martin                (Martin et al. 1994 spectral broadening)
#   [6] c_w                     (adiabatic LWC lapse rate, Bolton/moist adiabatic)
#   [7] LWP_insitu              (trapezoid vertical integration)
#
# All physics is in utils.py. This file is a pure orchestrator.
#
# CIP overlap handling:
#   step02 already zeroed CIP bins with D <= CIP_CUTOFF_UM (50 um). Therefore
#   compute_re_full uses ALL non-NaN CIP bins; the overlap-zone bins
#   contribute zero to N and effectively drop out of re_full.
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import (
    calc_re_effective, calc_k_martin,
    calc_cw, calc_lwp_trapezoid,
    calc_lwc_ad_max, classify_mixing_regime,
    select_active_bins,
)


# =============================================================================
# Helpers
# =============================================================================
def compute_fad(gm, gc, var_map=config.VAR_MAP):
    """
    Painemal & Zuidema (2011) profile-level f_ad:

        gamma_ad   = c_w(T_base, P_base)               [g/m^4]
        LWC_ad_max = gamma_ad * H                      [g/m^3]
        LWC_obs_max = profile 95th-percentile LWC      [g/m^3]
        f_ad       = LWC_obs_max / LWC_ad_max          [dimensionless, <= 1]

    The 95th percentile (instead of max) is robust to single-point spikes.
    Cloud base T, P used for c_w because f_ad is defined relative to the
    parcel's history from cloud base.

    Why this matters: an earlier point-by-point implementation
    (dz = z - z_base) systematically inflated f_ad by ~2x because z_base is
    the cloud-core minimum (50-150 m above the true thermodynamic base).
    """
    col_alt = var_map["altitude"]
    col_t   = var_map["temperature"]
    col_p   = var_map["pressure"]
    col_lwc = var_map["lwc"]

    gm["LWC_ad"]        = np.nan
    gm["f_ad"]          = np.nan
    gm["mixing_regime"] = pd.Series([None] * len(gm), index=gm.index, dtype="object")

    profile_fad = []

    for _, prof in gc.iterrows():
        cid  = prof["cloud_id"]
        mask = gm["cloud_id"] == cid
        if not mask.any():
            continue
        seg = gm[mask].sort_values(col_alt).copy()
        if len(seg) < 5:
            continue

        T_base = seg[col_t].iloc[0]
        P_base = seg[col_p].iloc[0]
        if pd.isna(T_base) or pd.isna(P_base):
            continue

        z_base = prof["z_base_m"]
        z_top  = prof["z_top_m"]
        H      = z_top - z_base
        if H <= 0:
            continue

        LWC_ad_max  = calc_lwc_ad_max(T_base, P_base, H)
        LWC_obs_max = seg[col_lwc].quantile(0.95)
        if LWC_ad_max <= 0:
            continue

        f_ad_profile = LWC_obs_max / LWC_ad_max

        profile_fad.append({
            "cloud_id"   : cid,
            "f_ad_mean"  : f_ad_profile,
            "f_ad_median": f_ad_profile,
            "f_ad_std"   : np.nan,
            "f_ad_min"   : f_ad_profile,
            "f_ad_max"   : f_ad_profile,
        })

        idx = seg.index
        gm.loc[idx, "LWC_ad"] = LWC_ad_max
        gm.loc[idx, "f_ad"]   = seg[col_lwc].values / LWC_ad_max
        gm.loc[idx, "mixing_regime"] = classify_mixing_regime(
            f_ad_profile,
            adiabatic_min=config.MIXING_REGIME_THRESHOLDS["adiabatic_min"],
            sub_adiabatic_min=config.MIXING_REGIME_THRESHOLDS["sub_adiabatic_min"],
        )

    if profile_fad:
        fad_stats = pd.DataFrame(profile_fad)
    else:
        fad_stats = pd.DataFrame(columns=[
            "cloud_id", "f_ad_mean", "f_ad_median", "f_ad_std",
            "f_ad_min", "f_ad_max"
        ])

    # Dominant regime per profile (mode of point-level mixing_regime)
    dom = (
        gm.dropna(subset=["mixing_regime"])
          .groupby("cloud_id")["mixing_regime"]
          .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
          .reset_index()
          .rename(columns={"mixing_regime": "dominant_mixing_regime"})
    )

    new_cols = list(fad_stats.columns) + list(dom.columns)
    drop_cols = [c for c in new_cols if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(fad_stats, on="cloud_id", how="left")
    gc = gc.merge(dom,       on="cloud_id", how="left")
    return gm, gc


def compute_re_cas(gm, var_map=config.VAR_MAP):
    """
    Effective radius from CAS only.
    Uses config.CAS_ACTIVE_BIN_INDICES (small-droplet bins, excluding the
    NaN placeholder if any).
    """
    cas_prefix = var_map["cas_prefix"]
    active_idx = config.CAS_ACTIVE_BIN_INDICES
    cas_cols = [f"{cas_prefix}{i:02d}" for i in active_idx]
    cas_cols = [c for c in cas_cols if c in gm.columns]

    D_cas_all = np.array(config.CAS_D_MID_ALL[active_idx[:len(cas_cols)]], dtype=float)
    D_cas, cas_cols = select_active_bins(D_cas_all, cas_cols, cutoff_um=None)

    r_cas = D_cas / 2.0
    N = gm[cas_cols].values
    gm["re_cas"] = calc_re_effective(N, r_cas)
    return gm, cas_cols, r_cas


def compute_re_full(gm, cas_cols, r_cas, var_map=config.VAR_MAP,
                     cip_d_mid_um=config.CIP_D_MID_UM):
    """
    Effective radius from combined CAS + CIP spectrum.

    CIP overlap zone (D <= CIP_CUTOFF_UM = 50 um) was already zeroed in
    step02, so those bins contribute 0 to the moments. We just need to
    drop NaN-midpoint placeholder bins to avoid contaminating the
    diameter array.
    """
    cip_prefix = var_map["cip_prefix"]
    n_cip = len(cip_d_mid_um)
    cip_cols = [f"{cip_prefix}{i:02d}" for i in range(n_cip)]
    cip_cols = [c for c in cip_cols if c in gm.columns]
    if len(cip_cols) == 0:
        gm["re_full"] = gm["re_cas"]
        return gm

    D_cip_all = np.array(cip_d_mid_um[:len(cip_cols)], dtype=float)
    D_cip, cip_cols = select_active_bins(D_cip_all, cip_cols, cutoff_um=None)
    if len(cip_cols) == 0:
        gm["re_full"] = gm["re_cas"]
        return gm

    r_cip = D_cip / 2.0

    N_cas = gm[cas_cols].fillna(0).values
    N_cip = gm[cip_cols].fillna(0).values
    N_full = np.concatenate([N_cas, N_cip], axis=1)
    r_full = np.concatenate([r_cas, r_cip])
    gm["re_full"] = calc_re_effective(N_full, r_full)
    return gm


def compute_profile_re_nd(gm, gc):
    """Aggregate Nd, re_cas, re_full to profile level."""
    nd_p = (
        gm.groupby("cloud_id")["Nc_CAS"]
          .agg(Nd_mean="mean", Nd_median="median", Nd_std="std")
          .reset_index()
    )
    re_cas_p = (
        gm.groupby("cloud_id")["re_cas"]
          .agg(re_cas_mean="mean", re_cas_median="median", re_cas_std="std")
          .reset_index()
    )
    re_full_p = (
        gm.groupby("cloud_id")["re_full"]
          .agg(re_full_mean="mean", re_full_median="median", re_full_std="std")
          .reset_index()
    )
    drop_cols = []
    for df in (nd_p, re_cas_p, re_full_p):
        drop_cols += [c for c in df.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=list(set(drop_cols)))
    gc = gc.merge(nd_p,      on="cloud_id", how="left")
    gc = gc.merge(re_cas_p,  on="cloud_id", how="left")
    gc = gc.merge(re_full_p, on="cloud_id", how="left")
    return gc


def compute_tau(gm, gc, var_map=config.VAR_MAP, params=config.PROFILE,
                phys=config.PHYS):
    """
    Per-layer and per-profile optical thickness:
        tau_layer = (3 * LWC * dz) / (2 * rho_w * re)
        tau_profile = sum(tau_layer)

    Computed twice:
      tau_main : uses re_cas (CAS only) - what MODIS-equivalent retrieval sees
      tau_full : uses re_full (CAS+CIP) - true geometric optical thickness

    The difference tau_diff = tau_main - tau_full quantifies the optical
    contribution of drizzle-sized droplets that MODIS does not see.
    """
    col_alt = var_map["altitude"]
    col_lwc = var_map["lwc"]
    lwc_thr = params["lwc_threshold"]
    nc_thr  = params["nc_threshold"]
    rho_w_g_m3 = phys["rho_w_gm3"]

    gm = gm.sort_values(["cloud_id", col_alt]).reset_index(drop=True)

    mask = (
        (gm[col_lwc] > lwc_thr) &
        (gm["Nc_CAS"] > nc_thr) &
        gm["re_cas"].notna() &
        gm["re_full"].notna()
    )

    gm["dz"] = gm.groupby("cloud_id")[col_alt].diff().abs().fillna(0)

    re_cas_m  = gm["re_cas"]  * 1e-6
    re_full_m = gm["re_full"] * 1e-6

    gm["tau_layer_main"] = np.where(
        mask, (3.0 * gm[col_lwc]) / (2.0 * rho_w_g_m3 * re_cas_m) * gm["dz"], np.nan)
    gm["tau_layer_full"] = np.where(
        mask, (3.0 * gm[col_lwc]) / (2.0 * rho_w_g_m3 * re_full_m) * gm["dz"], np.nan)

    profile_tau = (
        gm.groupby("cloud_id")
          .agg(tau_main =("tau_layer_main", "sum"),
               tau_full =("tau_layer_full", "sum"),
               n_tau_points=("tau_layer_main", "count"))
          .reset_index()
    )
    profile_tau["tau_diff"] = profile_tau["tau_main"] - profile_tau["tau_full"]

    drop_cols = [c for c in profile_tau.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(profile_tau, on="cloud_id", how="left")
    return gm, gc


def compute_k_martin_profile(gm, gc, var_map=config.VAR_MAP):
    """
    Point-level Martin et al. (1994) spectral broadening parameter.
    Aggregated to profile median over in-cloud points only (LWC > threshold).
    """
    cas_prefix = var_map["cas_prefix"]
    active_idx = config.CAS_ACTIVE_BIN_INDICES
    cas_cols = [f"{cas_prefix}{i:02d}" for i in active_idx]
    cas_cols = [c for c in cas_cols if c in gm.columns]

    D_cas_all = np.array(config.CAS_D_MID_ALL[active_idx[:len(cas_cols)]], dtype=float)
    D_cas, cas_cols = select_active_bins(D_cas_all, cas_cols, cutoff_um=None)

    r_cas = D_cas / 2.0
    N = gm[cas_cols].values
    gm["k_martin"] = calc_k_martin(N, r_cas)

    cloud_pts = gm[gm[var_map["lwc"]] > config.PROFILE["lwc_threshold"]]
    k_p = (
        cloud_pts.groupby("cloud_id")["k_martin"]
          .agg(k_median="median", k_mean="mean", k_std="std")
          .reset_index()
    )
    drop_cols = [c for c in k_p.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(k_p, on="cloud_id", how="left")
    return gm, gc


def compute_cw_profile(gm, gc, var_map=config.VAR_MAP):
    """
    Point-level adiabatic LWC lapse rate, then median over base-of-cloud
    points (z_norm <= 0.05). c_w varies slowly within profile but
    cloud-base value is the conventional reference.
    """
    gm["c_w"] = calc_cw(gm[var_map["temperature"]], gm[var_map["pressure"]])

    base_pts = gm[gm["z_norm"] <= 0.05]
    cw_p = (
        base_pts.groupby("cloud_id")["c_w"]
          .agg(c_w_median="median", c_w_mean="mean", c_w_std="std")
          .reset_index()
    )
    drop_cols = [c for c in cw_p.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(cw_p, on="cloud_id", how="left")
    return gm, gc


def compute_lwp_insitu(gm, gc, var_map=config.VAR_MAP):
    """
    Per-profile LWP by trapezoidal vertical integration of LWC vs altitude.
    Cloud depth is the altitude span of in-cloud points (LWC > 0.01).
    """
    col_alt = var_map["altitude"]
    col_lwc = var_map["lwc"]

    rows = []
    for cid, g in gm.groupby("cloud_id"):
        lwp, npts = calc_lwp_trapezoid(g[col_alt].values, g[col_lwc].values, lwc_min=0.01)
        if pd.isna(lwp):
            rows.append({"cloud_id": cid, "LWP_insitu": np.nan, "cloud_depth": np.nan})
            continue
        sub = g[g[col_lwc] > 0.01]
        rows.append({
            "cloud_id"   : cid,
            "LWP_insitu" : lwp,
            "cloud_depth": float(sub[col_alt].max() - sub[col_alt].min()),
        })
    lwp_df = pd.DataFrame(rows)
    drop_cols = [c for c in lwp_df.columns if c != "cloud_id" and c in gc.columns]
    if drop_cols:
        gc = gc.drop(columns=drop_cols)
    gc = gc.merge(lwp_df, on="cloud_id", how="left")
    return gc


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step05 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(In-situ microphysics)")
    print("=" * 70)

    if not config.STEP04_GOLDEN_CASE_CSV.exists() or not config.STEP04_GOLDEN_MICRO_CSV.exists():
        print(f"\n  [FAIL] Run step04 first.")
        sys.exit(1)

    gc = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    gm = pd.read_csv(config.STEP04_GOLDEN_MICRO_CSV)
    gc["start_time"] = pd.to_datetime(gc["start_time"])
    gc["end_time"]   = pd.to_datetime(gc["end_time"])
    gm[config.VAR_MAP["time"]] = pd.to_datetime(gm[config.VAR_MAP["time"]])
    print(f"\n  [OK] gc: {gc.shape}  |  gm: {gm.shape}")

    # ----------------------------------------------------------
    # 1. f_ad + mixing regime
    # ----------------------------------------------------------
    print(f"\n  [1] f_ad (Painemal 2011 method) + mixing regime")
    gm, gc = compute_fad(gm, gc)
    f_med_profile = gc["f_ad_mean"].median()
    n_valid = gc['f_ad_mean'].notna().sum()
    n_super = (gc['f_ad_mean'] > 1.0).sum()
    print(f"      f_ad median (profile-level) : {f_med_profile:.3f}")
    print(f"      Profiles with valid f_ad    : {n_valid}/{len(gc)}")
    print(f"      Super-adiabatic (f_ad > 1)  : {n_super}/{n_valid}")
    print(f"      Dominant regime distribution:")
    for line in gc["dominant_mixing_regime"].value_counts(dropna=False).to_string().split("\n"):
        print(f"        {line}")

    # ----------------------------------------------------------
    # 2. re_cas + re_full
    # ----------------------------------------------------------
    print(f"\n  [2] Effective radius (CAS only + CAS+CIP)")
    gm, cas_cols, r_cas = compute_re_cas(gm)
    print(f"      Active CAS bins      : {len(cas_cols)}")
    print(f"      re_cas  median       : {gm['re_cas'].median():.2f} um")
    gm = compute_re_full(gm, cas_cols, r_cas)
    print(f"      re_full median       : {gm['re_full'].median():.2f} um")

    # ----------------------------------------------------------
    # 2b. POSTPROC: re_eff sensor-artifact filter
    # ----------------------------------------------------------
    # Single-frame CIP sensor artifacts (e.g., spurious counts across all
    # bins) can produce non-physical Re >> 30 um. Bennartz (2007) marine
    # Sc upper bound is ~30 um. We flag these points as instrument
    # artifacts and set to NaN, preserving the rest of the profile data
    # and downstream median statistics. POST 1 Hz data rarely shows this
    # issue (sub-second integration), but the filter is enabled for
    # cross-campaign code symmetry.
    re_max = config.POSTPROC.get("re_max_physical_um", 30.0)
    n_re_cas_bad  = int((gm["re_cas"]  > re_max).sum())
    n_re_full_bad = int((gm["re_full"] > re_max).sum())
    if n_re_cas_bad + n_re_full_bad > 0:
        bad_mask = (gm["re_cas"] > re_max) | (gm["re_full"] > re_max)
        bad_profiles = gm.loc[bad_mask, "cloud_id"].value_counts()
        print(f"      [QC] re_eff > {re_max:.0f} um (sensor artifacts):")
        print(f"            re_cas  : {n_re_cas_bad} points -> NaN")
        print(f"            re_full : {n_re_full_bad} points -> NaN")
        for cid, n in bad_profiles.items():
            n_total = int((gm["cloud_id"] == cid).sum())
            print(f"              {cid}: {n}/{n_total} points affected")
        gm.loc[gm["re_cas"]  > re_max, "re_cas"]  = np.nan
        gm.loc[gm["re_full"] > re_max, "re_full"] = np.nan
    else:
        print(f"      [QC] re_eff sanity check passed (no points > {re_max:.0f} um)")

    # ----------------------------------------------------------
    # 3. Profile-level Nd, re_cas, re_full statistics
    # ----------------------------------------------------------
    print(f"\n  [3] Profile-level Nd, re_cas, re_full")
    gc = compute_profile_re_nd(gm, gc)

    # ----------------------------------------------------------
    # 4. Optical thickness (tau_main + tau_full)
    # ----------------------------------------------------------
    print(f"\n  [4] Optical thickness (tau_main + tau_full)")
    gm, gc = compute_tau(gm, gc)
    print(f"      tau_main median     : {gc['tau_main'].median():.2f}")
    print(f"      tau_full median     : {gc['tau_full'].median():.2f}")
    print(f"      tau_diff median     : {gc['tau_diff'].median():.2f}")

    # ----------------------------------------------------------
    # 5. k_martin
    # ----------------------------------------------------------
    print(f"\n  [5] k_martin (Martin et al. 1994 spectral broadening)")
    gm, gc = compute_k_martin_profile(gm, gc)
    print(f"      k_median average     : {gc['k_median'].mean():.3f}")
    print(f"      k_median range       : [{gc['k_median'].min():.3f}, {gc['k_median'].max():.3f}]")

    # ----------------------------------------------------------
    # 6. c_w (adiabatic LWC lapse rate)
    # ----------------------------------------------------------
    print(f"\n  [6] c_w (adiabatic LWC lapse rate, Bolton/moist adiabatic)")
    gm, gc = compute_cw_profile(gm, gc)
    print(f"      c_w_median average   : {gc['c_w_median'].mean():.5f} g/m^4")

    # ----------------------------------------------------------
    # 7. LWP_insitu
    # ----------------------------------------------------------
    print(f"\n  [7] LWP_insitu (trapezoid integration)")
    gc = compute_lwp_insitu(gm, gc)
    print(f"      LWP_insitu mean      : {gc['LWP_insitu'].mean():.1f} g/m^2")
    print(f"      cloud_depth mean     : {gc['cloud_depth'].mean():.1f} m")

    # ----------------------------------------------------------
    # Save: in-place update of the two main datasets (gc + gm).
    # gc and golden_microphysics are the single source of truth -
    # downstream steps (step07, step09, step10) read these directly.
    # ----------------------------------------------------------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gm.to_csv(config.STEP04_GOLDEN_MICRO_CSV, index=False)
    gc.to_csv(config.STEP04_GOLDEN_CASE_CSV,  index=False)

    print(f"\n  [SAVE] {config.STEP04_GOLDEN_MICRO_CSV.name}  ({gm.shape})")
    print(f"  [SAVE] {config.STEP04_GOLDEN_CASE_CSV.name}   ({gc.shape})")

    print("\n" + "=" * 70)
    print(f"  step05 COMPLETE")
    print("=" * 70)
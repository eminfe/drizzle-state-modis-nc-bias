# =============================================================================
# step09_modis_colocation.py  -  VOCALS-REx 2008
# =============================================================================
# Spatial + temporal co-location of MODIS L2 cloud retrievals (MOD06_L2 /
# MYD06_L2) with each filtered golden profile. For each profile:
#
#   1. Find granules within +/- max_diff_min minutes (per flight)
#   2. Crop pixels in PPA_BOX_DEG square around profile lat_mean/lon_mean
#   3. QC mask (per pixel):
#        - Phase = liquid (= 2)
#        - SZA   <  sza_max
#        - VZA   <  vza_max  (relaxed=60 deg or strict=40 deg)
#        - tau   >  tau_min
#        - re_min < Re < re_max
#   4. Average passing pixels for: Re_21, Re_37, tau_21, tau_37,
#                                  LWP, CTT, CTP, SZA, VZA
#   5. Spectral differences:
#        dRe_37_21  = Re_3.7  - Re_2.1   (vertical heterogeneity signal)
#        dtau_37_21 = tau_3.7 - tau_2.1
#
# When multiple granules cover the same profile, the granule yielding the
# largest n_pixels_valid is selected (better than "first valid wins").
#
# Usage:
#   python step09_modis_colocation.py            # relaxed VZA
#   python step09_modis_colocation.py strict     # strict VZA (40 deg)
# =============================================================================

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config


# =============================================================================
# HDF helpers
# =============================================================================
def read_scaled(hdf, var_name, band_idx=None):
    """
    Read an HDF SDS variable and apply scale/offset/fill cleanup.
        data = (raw - add_offset) * scale_factor

    band_idx is only used for 3-D arrays. Pass None for 2-D variables
    such as Cloud_Effective_Radius and Cloud_Optical_Thickness.
    """
    try:
        sds = hdf.select(var_name)
        data = sds[:].astype(np.float32)
        attrs = sds.attributes()
        scale  = float(attrs.get("scale_factor", 1.0))
        offset = float(attrs.get("add_offset",   0.0))
        fill   = attrs.get("_FillValue", -9999)
        vrange = attrs.get("valid_range", [None, None])
        data[data == fill] = np.nan
        if vrange[0] is not None:
            data[data < vrange[0]] = np.nan
        if vrange[1] is not None:
            data[data > vrange[1]] = np.nan
        data = (data - offset) * scale
        if band_idx is not None and data.ndim == 3:
            data = data[band_idx, :, :]
        return data
    except Exception:
        return None


def match_and_crop(data, target_shape):
    """Crop a 2-D array to the target shape (truncate, never pad)."""
    if data is None:
        return None
    h, w = data.shape
    th, tw = target_shape
    return data[:min(h, th), :min(w, tw)]


def granule_covers_box(hdf_path, lat_c, lon_c, ppa_box_deg):
    """Quick bbox pre-check: does this granule overlap with the profile box?"""
    try:
        from pyhdf.SD import SD, SDC
        hdf = SD(str(hdf_path), SDC.READ)
        lat_5km = hdf.select("Latitude")[:]
        lon_5km = hdf.select("Longitude")[:]
        hdf.end()
    except Exception:
        return False
    return (
        lat_5km.min() <= (lat_c + ppa_box_deg) and
        lat_5km.max() >= (lat_c - ppa_box_deg) and
        lon_5km.min() <= (lon_c + ppa_box_deg) and
        lon_5km.max() >= (lon_c - ppa_box_deg)
    )


def extract_ppa(hdf_path, lat_c, lon_c, vza_max, modis_cfg=config.MODIS):
    """
    Extract pixel-patch averages around (lat_c, lon_c) within PPA_BOX_DEG.

    Returns
    -------
    dict | None
        Pixel-averaged Re_21, Re_37, tau_21, tau_37, LWP, CTT, CTP,
        SZA_mean, VZA_mean, dRe_37_21, dtau_37_21,
        plus n_pixels_box, n_pixels_valid, status.
    """
    try:
        from pyhdf.SD import SD, SDC
        from scipy.ndimage import zoom
    except ImportError:
        print("[FAIL] pyhdf and scipy are required (pip install pyhdf scipy).")
        sys.exit(1)

    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception:
        return None

    try:
        # 1 km cloud properties (2-D - band_idx=None)
        Re_21  = read_scaled(hdf, "Cloud_Effective_Radius")
        Re_37  = read_scaled(hdf, "Cloud_Effective_Radius_37")
        tau_21 = read_scaled(hdf, "Cloud_Optical_Thickness")
        tau_37 = read_scaled(hdf, "Cloud_Optical_Thickness_37")

        if Re_21 is None or tau_21 is None:
            hdf.end()
            return {"status": "READ_FAIL"}

        target_shape = Re_21.shape

        # Phase
        try:
            phase = hdf.select("Cloud_Phase_Optical_Properties")[:].astype(np.float32)
        except Exception:
            hdf.end()
            return {"status": "PHASE_READ_FAIL"}

        # Auxiliary
        lwp  = read_scaled(hdf, "Cloud_Water_Path")
        ctt  = read_scaled(hdf, "cloud_top_temperature_1km")
        ctp  = read_scaled(hdf, "cloud_top_pressure_1km")

        # Geolocation + geometry are 5 km, must be upscaled to 1 km
        lat_5km = hdf.select("Latitude")[:]
        lon_5km = hdf.select("Longitude")[:]
        sza_5km = read_scaled(hdf, "Solar_Zenith")
        vza_5km = read_scaled(hdf, "Sensor_Zenith")
        z_h = target_shape[0] / lat_5km.shape[0]
        z_w = target_shape[1] / lat_5km.shape[1]

        lat = zoom(lat_5km, (z_h, z_w), order=1)
        lon = zoom(lon_5km, (z_h, z_w), order=1)
        sza = zoom(sza_5km, (z_h, z_w), order=1) if sza_5km is not None else None
        vza = zoom(vza_5km, (z_h, z_w), order=1) if vza_5km is not None else None

        # Crop everything to target_shape
        Re_21  = match_and_crop(Re_21,  target_shape)
        Re_37  = match_and_crop(Re_37,  target_shape) if Re_37  is not None else None
        tau_21 = match_and_crop(tau_21, target_shape)
        tau_37 = match_and_crop(tau_37, target_shape) if tau_37 is not None else None
        lwp    = match_and_crop(lwp,    target_shape) if lwp    is not None else None
        ctt    = match_and_crop(ctt,    target_shape) if ctt    is not None else None
        ctp    = match_and_crop(ctp,    target_shape) if ctp    is not None else None
        lat    = match_and_crop(lat,    target_shape)
        lon    = match_and_crop(lon,    target_shape)
        sza    = match_and_crop(sza,    target_shape) if sza is not None else None
        vza    = match_and_crop(vza,    target_shape) if vza is not None else None
        phase  = match_and_crop(phase,  target_shape)

        hdf.end()
    except Exception:
        try:
            hdf.end()
        except Exception:
            pass
        return None

    # Spatial bounding box around profile center
    box = config.MODIS["ppa_box_deg"]
    f_box = (
        (lat >= lat_c - box) & (lat <= lat_c + box) &
        (lon >= lon_c - box) & (lon <= lon_c + box)
    )
    n_box = int(f_box.sum())
    if n_box == 0:
        return {"status": "NO_COVERAGE", "n_pixels_box": 0, "n_pixels_valid": 0}

    # ----- Geometric / phase / SZA-VZA filter (shared by both channels) -----
    qc_geom = f_box & (phase == modis_cfg["phase_liquid"])
    if sza is not None:
        qc_geom &= (sza < modis_cfg["sza_max"])
    if vza is not None:
        qc_geom &= (vza < vza_max)

    # ----- 2.1 um channel QC -----
    qc_21 = qc_geom & ~np.isnan(Re_21) & ~np.isnan(tau_21)
    qc_21 &= (tau_21 > modis_cfg["tau_min"])
    qc_21 &= (Re_21  > modis_cfg["re_min_um"]) & (Re_21 < modis_cfg["re_max_um"])

    n_valid_21 = int(qc_21.sum())
    if n_valid_21 == 0:
        return {"status": "NO_VALID_PIXELS",
                "n_pixels_box": n_box, "n_pixels_valid": 0,
                "n_pixels_valid_21": 0, "n_pixels_valid_37": 0}

    # ----- 3.7 um channel QC (independent gates, then intersect with 2.1) -----
    # Felsefe A (paired-pool): dRe / dtau / dNd are computed only on pixels
    # that pass BOTH channel QC, ensuring apples-to-apples spectral comparison.
    if Re_37 is not None and tau_37 is not None:
        qc_37 = qc_geom & ~np.isnan(Re_37) & ~np.isnan(tau_37)
        qc_37 &= (tau_37 > modis_cfg["tau_min"])
        qc_37 &= (Re_37  > modis_cfg["re_min_um"]) & (Re_37 < modis_cfg["re_max_um"])
        qc_both = qc_21 & qc_37
        n_valid_37 = int(qc_both.sum())
    else:
        qc_both = None
        n_valid_37 = 0

    # Single-channel pool (used by Package B/C/D/G/H stats and bias_calc)
    qc = qc_21
    n_valid = n_valid_21

    out = {
        "status"            : "MATCHED",
        "n_pixels_box"      : n_box,
        "n_pixels_valid"    : n_valid,        # legacy alias = n_pixels_valid_21
        "n_pixels_valid_21" : n_valid_21,
        "n_pixels_valid_37" : n_valid_37,
        "Re_MODIS_21"       : float(np.nanmean(Re_21[qc])),
        "Re_MODIS_21_std"   : float(np.nanstd(Re_21[qc])),
        "tau_MODIS_21"      : float(np.nanmean(tau_21[qc])),
        "tau_MODIS_21_std"  : float(np.nanstd(tau_21[qc])),
        "SZA_mean"          : float(np.nanmean(sza[qc])) if sza is not None else np.nan,
        "VZA_mean"          : float(np.nanmean(vza[qc])) if vza is not None else np.nan,
        "LWP_MODIS"         : float(np.nanmean(lwp[qc])) if lwp is not None else np.nan,
        "CTT_MODIS"         : float(np.nanmean(ctt[qc])) if ctt is not None else np.nan,
        "CTP_MODIS"         : float(np.nanmean(ctp[qc])) if ctp is not None else np.nan,
    }

    # 3.7 um spectral channel (paired pool for spectral differences)
    if qc_both is not None and n_valid_37 > 0:
        # 3.7 means on the paired pool
        out["Re_MODIS_37"]      = float(np.nanmean(Re_37[qc_both]))
        out["Re_MODIS_37_std"]  = float(np.nanstd(Re_37[qc_both]))
        out["tau_MODIS_37"]     = float(np.nanmean(tau_37[qc_both]))
        out["tau_MODIS_37_std"] = float(np.nanstd(tau_37[qc_both]))
        # 2.1 means on the SAME paired pool (transparency: separate columns
        # so spectral differences are apples-to-apples without changing the
        # primary Re_MODIS_21 / tau_MODIS_21 baseline used by Package C/D)
        Re_21_paired  = float(np.nanmean(Re_21[qc_both]))
        tau_21_paired = float(np.nanmean(tau_21[qc_both]))
        out["Re_MODIS_21_paired37"]  = Re_21_paired
        out["tau_MODIS_21_paired37"] = tau_21_paired
        # Spectral differences (paired pool only)
        out["dRe_37_21"]  = out["Re_MODIS_37"]  - Re_21_paired
        out["dtau_37_21"] = out["tau_MODIS_37"] - tau_21_paired
    else:
        out.update({k: np.nan for k in [
            "Re_MODIS_37", "Re_MODIS_37_std",
            "tau_MODIS_37", "tau_MODIS_37_std",
            "Re_MODIS_21_paired37", "tau_MODIS_21_paired37",
            "dRe_37_21", "dtau_37_21",
        ]})
    return out


# =============================================================================
# Main
# =============================================================================
def main(vza_mode="relaxed"):
    """
    vza_mode = 'relaxed' (60 deg) or 'strict' (40 deg)
    """
    print("=" * 70)
    print(f"  step09 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(MODIS Co-location, VZA {vza_mode})")
    print("=" * 70)

    if not config.STEP04_GOLDEN_CASE_CSV.exists():
        print(f"\n  [FAIL] {config.STEP04_GOLDEN_CASE_CSV.name} not found.")
        print(f"         Run step07 first.")
        sys.exit(1)
    if not config.GRANULE_LOG.exists():
        print(f"\n  [FAIL] {config.GRANULE_LOG.name} not found.")
        print(f"         Run step08 first.")
        sys.exit(1)

    df = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    df_log = pd.read_csv(config.GRANULE_LOG)
    for col in ["start_time", "end_time"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    df_log["granule_time"] = pd.to_datetime(df_log["granule_time"], utc=True)

    vza_max = (config.MODIS["vza_max_strict"] if vza_mode == "strict"
               else config.MODIS["vza_max_relaxed"])
    dt_max  = config.MODIS["max_diff_min"]
    box     = config.MODIS["ppa_box_deg"]

    print(f"\n  Profiles : {len(df)}   |   Granules : {len(df_log)}")
    print(f"  VZA_MAX  = {vza_max} deg   "
          f"DT_MAX = {dt_max} min   PPA_BOX = +/-{box} deg")

    results = []
    for _, prof in df.iterrows():
        cid     = prof["cloud_id"]
        lat_c   = prof["lat_mean"]
        lon_c   = prof["lon_mean"]
        p_time  = prof["start_time"] + (prof["end_time"] - prof["start_time"]) / 2

        # Closest granules per flight
        gset = df_log[df_log["flight_id"] == prof["flight_id"]].copy()
        if gset.empty:
            row = prof.to_dict()
            row.update({"match_status": "NO_GRANULE",
                        "modis_file": "", "delta_t_min": np.nan,
                        "n_pixels_box": 0, "n_pixels_valid": 0})
            results.append(row)
            continue
        gset["dt_sec"] = (gset["granule_time"] - p_time).dt.total_seconds().abs()
        gset = gset.sort_values("dt_sec")

        best_modis = None
        best_g     = None

        # Walk granules in order of temporal proximity, keep best n_pixels_valid
        for _, g in gset.iterrows():
            if (g["dt_sec"] / 60.0) > dt_max:
                break  # gset is sorted; no closer granule remains
            hdf_path = config.MODIS_DIR / g["filename"]
            if not hdf_path.exists():
                continue
            if not granule_covers_box(hdf_path, lat_c, lon_c, box):
                continue
            modis = extract_ppa(hdf_path, lat_c, lon_c, vza_max)
            if modis is None or modis.get("status") in (None, "READ_FAIL"):
                continue
            n_valid = modis.get("n_pixels_valid", 0)
            if best_modis is None or n_valid > best_modis.get("n_pixels_valid", 0):
                best_modis = modis
                best_g     = g

        row = prof.to_dict()
        if best_modis is None:
            row.update({"match_status": "NO_VALID_PIXELS",
                        "modis_file": "", "delta_t_min": np.nan,
                        "n_pixels_box": 0, "n_pixels_valid": 0})
        else:
            row.update(best_modis)
            row["match_status"] = best_modis["status"]
            row["modis_file"]   = best_g["filename"]
            row["delta_t_min"]  = round(best_g["dt_sec"] / 60.0, 1)

        results.append(row)
        print(f"    {cid:<28}  {row['match_status']:<18}  "
              f"n_valid={row.get('n_pixels_valid', 0)}  "
              f"dt={row.get('delta_t_min', '?')} min")

    out = pd.DataFrame(results)
    out.to_csv(config.STEP09_MODIS_MATCHES_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"  Status counts:")
    print("  " + out["match_status"].value_counts().to_string().replace("\n", "\n  "))
    print(f"\n  [SAVE] {config.STEP09_MODIS_MATCHES_CSV.name}")
    print("=" * 70)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "relaxed"
    if mode not in ("relaxed", "strict"):
        print("Usage: python step09_modis_colocation.py [relaxed|strict]")
        sys.exit(1)
    main(vza_mode=mode)
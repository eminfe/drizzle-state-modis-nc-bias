# =============================================================================
# config.py - POST 2008 Campaign Configuration
# =============================================================================
# Single source of truth for all campaign-specific constants:
#   - File paths and output locations
#   - Column-name mapping for raw data
#   - Instrument metadata (bin bounds, midpoints)
#   - QC thresholds and profile-detection parameters
#   - Drizzle and mixing-regime classification thresholds
#   - Physical constants
#   - MODIS co-location settings
#   - Literature Nd-retrieval defaults (NDLIT)
#   - Final-pipeline filters (FILTERS) and post-processing (POSTPROC)
#
# Step files import from this module; they do not hardcode any values.
#
# POST vs MASE / VOCALS:
#   - POST: 17 flights (RF01..RF17), 1 Hz sampling
#   - POST: |Vz| > 0.5 m/s threshold (zigzag vertical sampling pattern)
#   - POST: 20 CAS bins + 63 CIP bins (~25 um spacing, last NaN)
#   - POST: integrity check ENABLED (gap_ratio < 0.10 at 1 Hz)
#   - POST: raw data is per-flight NetCDF (LOC_DIR), not a single Parquet
# =============================================================================

from pathlib import Path
import numpy as np

# =============================================================================
# 1. Campaign identity
# =============================================================================
CAMPAIGN_NAME   = "POST"
CAMPAIGN_YEAR   = 2008
CAMPAIGN_REGION = "NE Pacific (California coast)"
PLATFORM        = "CIRPAS Twin Otter"
FLIGHT_PREFIX   = "RF"             # POST flight IDs: RF01..RF17
N_FLIGHTS       = 17

# =============================================================================
# 2. Paths
# =============================================================================
# All pipeline code and outputs live under BASE_DIR.
# POST raw data lives outside BASE_DIR (per-flight NetCDF in LOC_DIR).
#
# Output organization:
#   OUTPUT_DIR/                     <- final, user-facing analysis files
#     |- POST_golden_case.csv
#     |- POST_golden_microphysics.csv
#     |- POST_MODIS_Matches.csv
#     |- figures/                   <- paper figures + QC plots
#     |- intermediate/              <- debug/inspection files
#         |- POST_step01_clean.parquet
#         |- POST_step02_qc.parquet
#         |- POST_step03_*.csv|.parquet
BASE_DIR         = Path(__file__).parent.resolve()  # set to your POST checkout
DATA_DIR         = BASE_DIR
OUTPUT_DIR       = BASE_DIR / "outputs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FIG_DIR          = OUTPUT_DIR / "figures"
MODIS_DIR        = BASE_DIR / "modis_data"
GRANULE_LOG      = BASE_DIR / "modis_granule_log.csv"

# POST-specific: per-flight raw NetCDF input directory (one RFxx.nc per flight)
LOC_DIR          = BASE_DIR / "raw" / "LOCATION"  # set to your POST raw per-flight NetCDF directory

# Step01 produces a single merged Parquet from the per-flight NetCDF set
PARQUET_FILE     = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_clean_export.parquet"

# =============================================================================
# 3. Variable mapping - POST column names in raw NetCDF
# =============================================================================
# Centralizes the campaign-specific column names so step files can use
# generic accessors (e.g., var_map["altitude"]) instead of hardcoding.
VAR_MAP = {
    "flight_id"            : "flight_id",
    "time"                 : "dt",          # POST datetime column
    "altitude"             : "GALT",        # GPS altitude (m)
    "latitude"             : "GLAT",        # GPS latitude (deg)
    "longitude"            : "GLON",        # GPS longitude (deg)
    "temperature"          : "AT",          # ambient temperature (C)
    "pressure"             : "PS",          # static pressure (hPa)
    "aircraft_vert_vel_raw": "GWIU",        # raw aircraft vertical velocity (m/s)
    "lwc"                  : "LWC_Gerber",  # Gerber probe LWC (g/m^3)
    "cas_prefix"           : "CAS_bin_",    # CAS bin column prefix
    "cip_prefix"           : "CIP_bin_",    # CIP bin column prefix
}

# =============================================================================
# 4. Instrument metadata
# =============================================================================
# Source: POST 2008 NetCDF headers (CIRPAS Twin Otter)
#   - CAS bin bounds: from CCAPS_CAS.CellSizes attribute
#   - CIP bin bounds: from CCAPS_CIP.CellSizes attribute
#
# CAS - Cloud and Aerosol Spectrometer (DMT)
# 20 logarithmically-spaced bins covering D = 0.62-50.90 um.
CAS_N_BINS_NOMINAL = 20
CAS_D_MID_ALL = np.array([
    0.62,    # Bin 1
    0.67,    # Bin 2
    0.74,    # Bin 3
    0.81,    # Bin 4
    0.90,    # Bin 5
    1.00,    # Bin 6
    1.12,    # Bin 7
    1.20,    # Bin 8
    1.58,    # Bin 9
    2.07,    # Bin 10
    2.84,    # Bin 11
    4.17,    # Bin 12
    7.40,    # Bin 13
    10.80,   # Bin 14
    13.70,   # Bin 15
    18.20,   # Bin 16
    23.70,   # Bin 17
    30.70,   # Bin 18
    40.20,   # Bin 19
    50.90,   # Bin 20
], dtype=float)
CAS_ACTIVE_BIN_INDICES = list(range(0, 20))  # all 20 active
CAS_D_MIN_UM = 0.62
CAS_D_MAX_UM = 50.90

# CIP - Cloud Imaging Probe (DMT)
# 63 columns; first 62 valid (variable spacing ~25 um), last is NaN placeholder.
# (FirstBin=1, LastBin=62 in NetCDF metadata.)
CIP_N_BINS_NOMINAL = 63
CIP_D_MID_UM = np.array([
    15.45,    40.45,    64.04,    88.53,   113.28,   138.12,   163.02,
   187.95,   212.89,   237.85,   262.81,   287.78,   312.76,   337.74,
   362.72,   387.71,   412.70,   437.68,   462.67,   487.66,   512.66,
   537.65,   562.64,   587.64,   612.63,   637.63,   662.62,   687.62,
   712.61,   737.61,   762.60,   787.60,   812.60,   837.59,   862.59,
   887.59,   912.59,   937.58,   962.58,   987.58,  1012.58,  1037.58,
  1062.57,  1087.57,  1112.57,  1137.57,  1162.57,  1187.57,  1212.57,
  1237.56,  1262.56,  1287.56,  1312.56,  1337.56,  1362.56,  1387.56,
  1412.56,  1437.55,  1462.55,  1487.55,  1512.55,  1537.55,  1562.55,
   np.nan,  # Bin 63 - structural placeholder, filtered automatically
], dtype=float)
CIP_D_MIN_UM = 15.45
CIP_D_MAX_UM = 1562.55

# CAS-CIP overlap cutoff (Wood 2012 marine Sc drizzle threshold).
# Used by step05 compute_re_full: CIP bins with midpoint <= this value are
# excluded from the composite spectrum to prevent double-counting in the
# overlap region where CAS provides the dominant measurement.
CIP_CUTOFF_UM = 50.0

# =============================================================================
# 5. QC thresholds
# =============================================================================
# Physical-range checks applied in step02 to remove sensor artifacts.
QC = {
    "lwc_min" : 0.0,      # g/m^3 - clip negative LWC to zero
    "at_min"  : -100.0,   # C - minimum plausible ambient temperature
    "at_max"  : 50.0,     # C - maximum plausible ambient temperature
    "ps_min"  : 500.0,    # hPa - minimum plausible static pressure
    "ps_max"  : 1100.0,   # hPa - maximum plausible static pressure
    "alt_min" : 0.0,      # m  - reject negative altitude
}

# =============================================================================
# 6. Profile-detection thresholds
# =============================================================================
# POST 1 Hz data, NE Pacific stratocumulus.
# Standard depth/duration thresholds (200 m / 120 s) - thicker than VOCALS
# (100 m / 60 s relaxed).
DT = 1.0   # s - POST native sampling rate

PROFILE = {
    # Cloud-core thresholds (per-point)
    "lwc_threshold"        : 0.05,   # g/m^3 - minimum LWC for in-cloud point
    "nc_threshold"         : 10.0,   # cm^-3 - minimum droplet count
    "vz_threshold"         : 0.5,    # m/s   - POST: 0.5 (zigzag sampling),
                                     #         MASE: 0.1 (level legs)

    # Vertical velocity smoothing (rolling mean window)
    "alt_smooth_win"       : 5,      # 5 points = 5 seconds at 1 Hz

    # Minimum profile size
    "min_profile_pts"      : 10,     # 10 points = 10 seconds

    # Segmentation / merging
    "summary_gap_seconds"  : 120,    # split if gap exceeds this
    "stitch_gap_seconds"   : 300,    # merge nearby profiles into super-profiles

    # Final geometric QC (standard for NE Pacific Sc)
    "final_min_depth_m"    : 200.0,  # m - minimum cloud depth
    "final_min_duration_s" : 120.0,  # s - minimum profile duration
    "final_min_lwc"        : 0.05,   # g/m^3 - minimum profile peak LWC

    # Profile span score (sawtooth-aware vertical-penetration filter).
    # Defined as (z_max - z_min) / sum(|d_alt|).
    # Values near 1: monotonic ascent/descent. <0.4: level-leg wandering.
    # Set to 0.0 to disable. POST 1 Hz with |Vz|>0.5 m/s rarely hits this.
    "min_span_score"       : 0.0,

    # Time-series integrity (sensor dropout detection - POST-specific).
    # Enabled at 1 Hz. MASE 10 s data disables this (gap_ratio test n/a).
    "integrity_check_enabled": True,
    "integrity_min_pts"      : 15,
    "integrity_max_gap_ratio": 0.10,
}

# =============================================================================
# 7. Drizzle classification thresholds
# =============================================================================
# Drizzle indicators based on large-droplet concentration and LWC ratio.
# POST 1 Hz: standard 10 L^-1 threshold (Rossiter 2012, King et al. 2013).
# MASE 10 s averaging dilutes large-drop concentration -> uses 1 L^-1.
DRIZZLE = {
    "n_large_thresh"  : 10.0,    # L^-1 - drizzle drop concentration threshold
    "lwc_ratio_thresh": 0.10,    # LWC_drizzle / LWC_total ratio
    "d_large_um"      : 100.0,   # um - diameter cutoff for "large" drops
}

# Profile-level drizzle regime classification by drizzle fraction.
DRIZZLE_REGIME_THRESHOLDS = {
    "non_to_weak"  : 0.011,
    "weak_to_mod"  : 0.045,
    "mod_to_heavy" : 0.125,
}

# =============================================================================
# 8. Mixing regime thresholds
# =============================================================================
# Adiabatic fraction thresholds for cloud mixing classification.
# References: Painemal and Zuidema (2011); Albrecht et al. (1985);
# Brenguier et al. (2000).
MIXING_REGIME_THRESHOLDS = {
    "adiabatic_min"     : 0.8,    # f_ad >= 0.8 -> adiabatic
    "sub_adiabatic_min" : 0.4,    # 0.4 <= f_ad < 0.8 -> sub-adiabatic
                                  # f_ad < 0.4 -> strongly mixed
}

# =============================================================================
# 9. Physical constants
# =============================================================================
PHYS = {
    "rho_w_gm3" : 1.0e6,    # liquid water density (g/m^3)
    "rho_w_gcm3": 1.0,      # liquid water density (g/cm^3)
    "rho_w_si"  : 1000.0,   # liquid water density (kg/m^3)
    "Rv"        : 461.5,    # water vapor gas constant (J/kg/K)
    "Rd"        : 287.05,   # dry air gas constant (J/kg/K)
    "Lv"        : 2.5e6,    # latent heat of vaporization (J/kg)
    "cp"        : 1005.0,   # specific heat of dry air (J/kg/K)
    "g"         : 9.81,     # gravitational acceleration (m/s^2)
    "Q_ext"     : 2.0,      # extinction efficiency (geometric optics)
}

# =============================================================================
# 10. MODIS co-location settings
# =============================================================================
# POST: standard relaxed 60 deg VZA (Ma et al. 2017) and 0.5 deg bbox padding.
MODIS = {
    # MODIS overpass search window
    "max_diff_min"    : 90,        # +/- minutes around aircraft profile
    "bbox_pad_deg"    : 0.50,      # spatial padding around profile bbox
    "dt_pad_min"      : 110,       # download time padding

    # Pixel co-location
    "ppa_box_deg"     : 0.10,      # spatial averaging box (deg lat/lon)
    "sza_max"         : 65.0,      # max solar zenith angle
    "vza_max_relaxed" : 60.0,      # relaxed VZA cutoff (Ma et al. 2017)
    "vza_max_strict"  : 40.0,      # strict VZA cutoff (Painemal & Zuidema 2011)
    "tau_min"         : 4.0,       # min optical thickness (Bennartz 2007)
    "re_min_um"       : 4.0,       # min effective radius (retrieval validity)
    "re_max_um"       : 30.0,      # max effective radius
    "phase_liquid"    : 2,         # MODIS phase flag value for liquid
}

# Literature defaults used by operational satellite Nd retrievals.
# Two parallel bias calculations:
#   bias_calc : Grosvenor with in-situ measured k, f_ad, c_w
#   bias_lit  : Grosvenor with these literature defaults
# The difference quantifies the assumption sensitivity (Package F, paper Section 4.7).
NDLIT = {
    "k_lit"    : 0.8,     # Martin et al. (1994) marine stratocumulus
    "f_ad_lit" : 0.80,     # adiabatic fraction (operational default, Quaas 2006)
    "c_w_lit"  : 2.3e-3,  # g/m^4 - marine Sc baseline (T=12C, P=925 hPa)
}

# =============================================================================
# 10b. Final-pipeline filters (Bug #18 fix)
# =============================================================================
# Used by step07 to filter golden profiles to physically valid cases.
# Previously hardcoded; now config-driven for cross-campaign consistency.
FILTERS = {
    "f_ad_max" : 1.0,     # Painemal & Zuidema (2011) physical constraint
    "nd_min"   : 5.0,     # cm^-3 - sub-detection threshold (Bennartz 2007;
                          # Grosvenor et al. 2018: MODIS Nd reliable for >~10).
                          # We use 5.0 (most permissive) to retain borderline
                          # profiles; >5 only excludes true sub-detection cases.
}

# =============================================================================
# 10c. Post-processing filters (sensor artifact removal)
# =============================================================================
# Applied in step05 after re_eff calculation. Filters non-physical effective
# radius values that arise from CIP sensor artifacts (e.g., spurious counts
# in tail bins producing Re > 100 um). The threshold is set above the
# physiological drizzle Re tail to preserve real heavy-drizzle measurements.
#
# Threshold rationale:
#   - Marine Sc cloud droplet:    Re <= 15 um  (typical)
#   - Marine Sc with drizzle:     Re tail to 25-40 um  (physiological)
#   - Heavy precip in stratocum.: Re up to 50 um  (rare but real)
#   - Re > 50 um in 1 Hz CIP:     dominated by sensor artifacts (Wood 2012)
#
# This is an IN-SITU measurement filter. The MODIS-comparable upper limit
# is 30 um (Bennartz 2007), enforced separately in step09 MODIS QC.
POSTPROC = {
    "re_max_physical_um": 50.0,    # Wood (2012) marine Sc in-situ upper bound;
                                    # preserves drizzle tail, removes sensor artifacts
}

# =============================================================================
# 11. Output file paths
# =============================================================================
# Final user-facing outputs live directly under OUTPUT_DIR.
# Intermediate (debug/inspection) files live under INTERMEDIATE_DIR.

# --- Intermediate (debug/inspection) ---
STEP01_PARQUET                = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step01_clean.parquet"
STEP02_QC_PARQUET             = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step02_qc.parquet"

STEP03_PROFILE_POINTS_PARQUET = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step03_profile_points.parquet"
STEP03_PROFILE_SUMMARY_CSV    = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step03_profile_summary.csv"
STEP03_SUPER_PROFILES_CSV     = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step03_super_profiles.csv"
STEP03_GOLDEN_CSV             = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_step03_golden_profiles.csv"

# step04 debug summaries (also stored as columns in golden_case.csv)
STEP04_PROFILE_DRIZZLE_CSV    = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_profile_drizzle.csv"
STEP04_PROFILE_VERTICAL_CSV   = INTERMEDIATE_DIR / f"{CAMPAIGN_NAME}_profile_vertical.csv"

# --- Final user-facing analysis outputs ---
# step04 -> step10 update these in place
STEP04_GOLDEN_CASE_CSV        = OUTPUT_DIR / f"{CAMPAIGN_NAME}_golden_case.csv"
STEP04_GOLDEN_MICRO_CSV       = OUTPUT_DIR / f"{CAMPAIGN_NAME}_golden_microphysics.csv"

# step09 MODIS matching
STEP09_MODIS_MATCHES_CSV      = OUTPUT_DIR / f"{CAMPAIGN_NAME}_MODIS_Matches.csv"

# =============================================================================
# 12. Figure settings
# =============================================================================
FIG = {
    "dpi": 150,
    "fmt": "png",
}

# =============================================================================
# 13. Auto-create directory structure on import
# =============================================================================
# Step files do not need to call mkdir themselves; importing config.py
# guarantees that all output directories exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODIS_DIR.mkdir(parents=True, exist_ok=True)

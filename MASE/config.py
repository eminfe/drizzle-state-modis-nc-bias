# =============================================================================
# config.py - MASE 2005 Campaign Configuration
# =============================================================================
# Single source of truth for all campaign-specific constants:
#   - File paths and output locations
#   - Column-name mapping for raw data
#   - Instrument metadata (bin bounds, midpoints)
#   - QC thresholds and profile-detection parameters
#   - Drizzle and mixing-regime classification thresholds
#   - Physical constants
#   - MODIS co-location settings
#   - Final filter thresholds (cross-campaign tunable)
#
# Step files import from this module; they do not hardcode any values.
# =============================================================================

from pathlib import Path
import numpy as np

# =============================================================================
# 1. Campaign identity
# =============================================================================
CAMPAIGN_NAME   = "MASE"
CAMPAIGN_YEAR   = 2005
CAMPAIGN_REGION = "NE Pacific (California coast)"
PLATFORM        = "CIRPAS Twin Otter"
FLIGHT_PREFIX   = "RF"            # MASE flight IDs are RF01..RF13
N_FLIGHTS       = 13

# =============================================================================
# 2. Paths
# =============================================================================
# All pipeline code and outputs live under BASE_DIR. The raw data file
# (single Parquet covering all flights) sits at BASE_DIR root.
#
# Output organization:
#   OUTPUT_DIR/                     <- final, user-facing analysis files
#     |- MASE_golden_case.csv
#     |- MASE_golden_microphysics.csv
#     |- MASE_MODIS_Matches.csv
#     |- figures/                   <- paper figures + QC plots
#     |- intermediate/              <- debug/inspection intermediate files
#         |- MASE_step01_clean.parquet
#         |- MASE_step02_qc.parquet
#         |- MASE_step03_*.csv|.parquet
BASE_DIR         = Path(__file__).parent.resolve()  # set to your MASE checkout
DATA_DIR         = BASE_DIR
OUTPUT_DIR       = BASE_DIR / "outputs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FIG_DIR          = OUTPUT_DIR / "figures"
MODIS_DIR        = BASE_DIR / "modis_data"
GRANULE_LOG      = BASE_DIR / "modis_granule_log.csv"

# Raw 10 s data file (all flights merged into a single Parquet)
PARQUET_FILE = DATA_DIR / "MASE_10s_merged_v2.parquet"

# =============================================================================
# 3. Variable mapping - MASE column names in raw data
# =============================================================================
# Centralizes the campaign-specific column names so step files can use
# generic accessors (e.g., var_map["altitude"]) instead of hardcoding.
#
# Note: MASE bin column names contain a SPACE ("CAS Bin 1") and are
# 1-indexed, unlike VOCALS ("CAS_bin_00", 0-indexed). The string sort
# would order "CAS Bin 10" before "CAS Bin 2"; utils.get_cas_columns
# applies a numeric sort to handle this.
VAR_MAP = {
    "flight_id"            : "flight_id",
    "time"                 : "Datetime",
    "altitude"             : "Alt",
    "latitude"             : "Lat",
    "longitude"            : "Long",
    "temperature"          : "T amb",        # ambient temperature (C)
    "pressure"             : "Static P",     # static pressure (hPa)
    "aircraft_vert_vel_raw": "GWIU",         # NOT in raw data; computed in step01
    "lwc"                  : "LWC Gerber",   # Gerber probe LWC
    "cas_prefix"           : "CAS Bin ",     # space-separated, 1-indexed
    "cip_prefix"           : "CIP Bin ",     # space-separated, 1-indexed
}

# =============================================================================
# 4. Instrument metadata
# =============================================================================
# Source: MASE 2005 raw data file headers (CIRPAS, instrument metadata).
#
# CAS - Cloud and Aerosol Spectrometer (DMT)
# 20 logarithmically-spaced bins covering D = 0.703-54.0 um.
# For logarithmic bin spacing, the geometric mean is the correct midpoint:
#     D_mid = sqrt(D_lower * D_upper)
#
# Note: MASE Bin 1 has an unspecified lower bound (instrument noise band)
# and is treated as INACTIVE (D_mid = NaN). Active bins are 2..20.
# This contrasts with VOCALS where bin_00..bin_19 are active and bin_20
# is structurally empty.
CAS_N_BINS_NOMINAL = 20
CAS_BIN_BOUNDS = np.array([
    [np.nan,  0.703],   # Bin 1  (INACTIVE - lower bound undefined)
    [0.703,   0.730],   # Bin 2
    [0.730,   0.770],   # Bin 3
    [0.770,   0.817],   # Bin 4
    [0.817,   0.882],   # Bin 5
    [0.882,   0.986],   # Bin 6
    [0.986,   1.063],   # Bin 7
    [1.063,   1.350],   # Bin 8
    [1.350,   1.600],   # Bin 9
    [1.600,   2.100],   # Bin 10
    [2.100,   2.800],   # Bin 11
    [2.800,   4.000],   # Bin 12
    [4.000,   6.900],   # Bin 13
    [6.900,   8.900],   # Bin 14
    [8.900,  12.500],   # Bin 15
    [12.500, 17.000],   # Bin 16
    [17.000, 23.500],   # Bin 17
    [23.500, 31.000],   # Bin 18
    [31.000, 41.000],   # Bin 19
    [41.000, 54.000],   # Bin 20
])
# Geometric mean midpoints (NaN propagates through Bin 1)
CAS_D_MID_ALL = np.sqrt(CAS_BIN_BOUNDS[:, 0] * CAS_BIN_BOUNDS[:, 1])
# Active bin indices: 1..19 (Bin 2..Bin 20 in 1-indexed naming)
CAS_ACTIVE_BIN_INDICES = list(range(1, 20))
CAS_D_MIN_UM = 0.703
CAS_D_MAX_UM = 54.0

# CIP - Cloud Imaging Probe (DMT)
# 60 linearly-spaced bins covering D = 25-1500 um (25 um step).
# For linear spacing, the arithmetic mean is the correct midpoint:
#     D_mid = (D_lower + D_upper) / 2
#
# Note: MASE Bin 1 has an unspecified lower bound and is treated as
# INACTIVE (D_mid = NaN). Active bins are 2..60 (D_mid = 37.5..1487.5 um).
CIP_N_BINS_NOMINAL = 60
CIP_D_MID_UM = np.concatenate([
    [np.nan],                                      # Bin 1 (INACTIVE)
    np.arange(37.5, 37.5 + 59 * 25.0, 25.0),       # Bin 2..60
])
CIP_D_MIN_UM = 25.0
CIP_D_MAX_UM = 1500.0

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
# MASE 10 s data with thresholds tuned for NE Pacific stratocumulus and
# the Twin Otter level-leg flight pattern (mostly horizontal, not zigzag).
DT = 10.0   # s - MASE native sampling rate

PROFILE = {
    # Cloud-core thresholds (per-point)
    "lwc_threshold"        : 0.05,   # g/m^3 - minimum LWC for in-cloud point
    "nc_threshold"         : 10.0,   # cm^-3 - minimum droplet count
    "vz_threshold"         : 0.1,    # m/s   - relaxed for level-leg flights
                                     # (VOCALS/POST: 0.5 for zigzag profiles)

    # Vertical velocity smoothing (rolling mean window, in points)
    # MASE: 3 points = 30 s window; VOCALS: 5 points = 5 s window
    "alt_smooth_win"       : 3,

    # Minimum profile size (in points)
    # MASE: 3 points = 30 s; VOCALS: 10 points = 10 s
    "min_profile_pts"      : 3,

    # Segmentation / merging (in seconds, time-based)
    "summary_gap_seconds"  : 120,    # split if gap exceeds this
    "stitch_gap_seconds"   : 300,    # merge nearby profiles into super-profiles

    # Final QC (NE Pacific stratocumulus is thicker than VOCALS SE Pacific)
    "final_min_depth_m"    : 200.0,  # m - minimum cloud depth
    # Section B intermediate filter (looser; pre-merge segment screen)
    # In MASE this is set to 60 s, but the final Section C cut is 120 s.
    # In VOCALS both are 60 s (broken-cloud regime, no two-stage filter).
    "summary_min_duration_s" : 60.0,
    "final_min_duration_s" : 120.0,  # s - minimum profile duration
    "final_min_lwc"        : 0.05,   # g/m^3 - minimum profile peak LWC
    
    # Vertical-penetration quality (sawtooth-friendly).
        # span_score = (z_max - z_min) / sum(|d_alt|)
        #   1.0   = monotonic profile
        #   ~0.6  = stepped/sawtooth descent (acceptable)
        #   <0.4  = level-leg / wandering (reject)
        # MASE 10 s level-leg flights need this; VOCALS/POST 1 Hz zigzag
        # passes by default with span >= 0.7.
        "min_span_score" : 0.4,

    # Time-series integrity (sensor dropout detection)
    # MASE 10 s data is already coarse; integrity check is much less
    # discriminating than at 1 Hz. Kept for API compatibility but
    # parameters are loosened.
    
    # Section D toggle: integrity check is meaningful only at sub-second
    # sampling. MASE 10 s data is already coarse; gap-ratio analysis is
    # not informative, so the check is disabled here.
    "integrity_check_enabled" : False,
    "integrity_min_pts"      : 5,    # 5 points = 50 s
    "integrity_max_gap_ratio": 0.30, # 30% (loosened from VOCALS 10%)
}

# =============================================================================
# 6b. Post-processing QC (instrument artifact rejection)
# =============================================================================
# Single-frame sensor artifacts can produce non-physical effective radii
# even after step02 QC. The composite re_full is more vulnerable than
# re_cas because a transient spike across CIP bins (which span 25-1500 um)
# can dominate the moment integral. step05 rejects per-point re values
# above this physical limit. Marine stratocumulus cloud-droplet effective
# radii are bounded by ~30 um in observations (Bennartz 2007 review).
POSTPROC = {
    "re_max_physical_um" : 30.0,    # NaN above this (instrument artifact)
}


# =============================================================================
# 7. Drizzle classification thresholds
# =============================================================================
# Drizzle indicators based on large-droplet concentration and LWC ratio.
# MASE n_large threshold is loosened to 1 L^-1 because 10 s averaging
# washes out short bursts that the VOCALS 1 Hz pipeline catches at 10 L^-1.
DRIZZLE = {
    "n_large_thresh"  : 1.0,     # L^-1 - drizzle drop concentration threshold
                                 # (VOCALS/POST: 10 L^-1)
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
# MASE-specific: time window slightly relaxed (120 min vs VOCALS 90 min)
# because Twin Otter flight tracks are shorter than G-1 / C-130.
MODIS = {
    # MODIS overpass search window
    "max_diff_min"    : 120,       # +/- minutes around aircraft profile
                                   # (VOCALS: 90)
    "bbox_pad_deg"    : 0.50,      # spatial padding around profile bbox
    "dt_pad_min"      : 180,       # download time padding

    # Pixel co-location
    "ppa_box_deg"     : 0.10,      # spatial averaging box (deg lat/lon)
    "sza_max"         : 65.0,      # max solar zenith angle
    "vza_max_relaxed" : 60.0,      # relaxed VZA cutoff
    "vza_max_strict"  : 40.0,      # strict VZA cutoff (Ma et al. 2017)
    "tau_min"         : 4.0,       # min optical thickness (Bennartz 2007)
    "re_min_um"       : 4.0,       # min effective radius (retrieval validity)
    "re_max_um"       : 30.0,      # max effective radius
    "phase_liquid"    : 2,         # MODIS phase flag value for liquid
}

# Literature defaults used by operational satellite Nd retrievals.
# Two parallel bias calculations:
#   bias_calc : Grosvenor with in-situ measured k, f_ad, c_w
#   bias_lit  : Grosvenor with these literature defaults
# The difference quantifies the assumption sensitivity (Package F).
NDLIT = {
    "k_lit"    : 0.80,     # Martin et al. (1994) marine stratocumulus
    "f_ad_lit" : 0.80,     # adiabatic fraction (operational default, Quaas 2006)
    "c_w_lit"  : 2.3e-3,  # g/m^4 - marine Sc baseline (T=12C, P=925 hPa)
}

# =============================================================================
# 11. Final filter thresholds
# =============================================================================
# Bug #18 fix: thresholds previously hard-coded in step07_final_check.py
# moved here for cross-campaign tunability and consistency.
FILTERS = {
    "f_ad_max" : 1.0,     # Painemal & Zuidema (2011) physical constraint
    "nd_min"   : 5.0,     # cm^-3 - sub-detection threshold (Bennartz 2007)
}

# =============================================================================
# 12. Output file paths
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
# 13. Figure settings
# =============================================================================
FIG = {
    "dpi": 150,
    "fmt": "png",
}

# =============================================================================
# 14. Auto-create directory structure on import
# =============================================================================
# Step files do not need to call mkdir themselves; importing config.py
# guarantees that all output directories exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODIS_DIR.mkdir(parents=True, exist_ok=True)

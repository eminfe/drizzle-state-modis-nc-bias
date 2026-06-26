# =============================================================================
# config.py - VOCALS-REx 2008 Campaign Configuration
# =============================================================================
# Single source of truth for all campaign-specific constants:
#   - File paths and output locations
#   - Column-name mapping for raw data
#   - Instrument metadata (bin bounds, midpoints)
#   - QC thresholds and profile-detection parameters
#   - Drizzle and mixing-regime classification thresholds
#   - Physical constants
#   - MODIS co-location settings
#
# Step files import from this module; they do not hardcode any values.
# =============================================================================

from pathlib import Path
import numpy as np

# =============================================================================
# 1. Campaign identity
# =============================================================================
CAMPAIGN_NAME   = "VOCALS"
CAMPAIGN_YEAR   = 2008
CAMPAIGN_REGION = "SE Pacific (Chilean coast)"
PLATFORM        = "BNL G-1 Research Aircraft"
FLIGHT_PREFIX   = ""              # VOCALS flight IDs are date-based (e.g., 081014a)
N_FLIGHTS       = 18

# =============================================================================
# 2. Paths
# =============================================================================
# All pipeline code and outputs live under BASE_DIR. The raw data file
# (single Parquet covering all flights) sits at BASE_DIR root.
#
# Output organization:
#   OUTPUT_DIR/                     <- final, user-facing analysis files
#     |- VOCALS_golden_case.csv
#     |- VOCALS_golden_microphysics.csv
#     |- VOCALS_MODIS_Matches.csv
#     |- figures/                   <- paper figures + QC plots
#     |- intermediate/              <- debug/inspection ara dosyalar
#         |- VOCALS_step01_clean.parquet
#         |- VOCALS_step02_qc.parquet
#         |- VOCALS_step03_*.csv|.parquet
BASE_DIR         = Path(__file__).parent.resolve()  # set to your VOCALS checkout
DATA_DIR         = BASE_DIR
OUTPUT_DIR       = BASE_DIR / "outputs"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FIG_DIR          = OUTPUT_DIR / "figures"
MODIS_DIR        = BASE_DIR / "modis_data"
GRANULE_LOG      = BASE_DIR / "modis_granule_log.csv"

# Raw 1 Hz data file (all flights merged into a single Parquet)
PARQUET_FILE = DATA_DIR / "vocals_clean_export.parquet"

# =============================================================================
# 3. Variable mapping - VOCALS column names in raw data
# =============================================================================
# Centralizes the campaign-specific column names so step files can use
# generic accessors (e.g., var_map["altitude"]) instead of hardcoding.
VAR_MAP = {
    "flight_id"            : "flight_id",
    "time"                 : "UTC",
    "altitude"             : "GALT",
    "latitude"             : "GLAT",
    "longitude"            : "GLON",
    "temperature"          : "AT",          # ambient temperature
    "pressure"             : "PS",          # static pressure
    "aircraft_vert_vel_raw": "GWIU",        # raw aircraft vertical velocity
    "lwc"                  : "LWC_Gerber",  # Gerber probe LWC
    "cas_prefix"           : "CAS_bin_",    # CAS bin column prefix
    "cip_prefix"           : "CIP_bin_",    # CIP bin column prefix
}

# =============================================================================
# 4. Instrument metadata
# =============================================================================
# Source: VOCALS 2008 G-1 raw data file headers (BNL, Senum/Springston).
#   - CAS bin bounds: extracted from 081018a CAS file header
#   - CIP bin bounds: extracted from 081022a CIP file header
#
# CAS - Cloud and Aerosol Spectrometer (DMT)
# 20 logarithmically-spaced bins covering D = 0.6-56.3 um.
# For logarithmic bin spacing, the geometric mean is the correct midpoint:
#     D_mid = sqrt(D_lower * D_upper)
#
# Note: VOCALS data files have 21 CAS columns (CAS_bin_00..CAS_bin_20).
# Only the first 20 represent valid bins; CAS_bin_20 is a structural
# placeholder (NaN/all-zero in raw data). The pipeline filters this
# automatically via select_active_bins() in utils.py.
CAS_N_BINS_NOMINAL = 20
CAS_BIN_BOUNDS = np.array([
    [0.600, 0.679],   # Bin 1
    [0.679, 0.712],   # Bin 2
    [0.712, 0.758],   # Bin 3
    [0.758, 0.817],   # Bin 4
    [0.817, 0.890],   # Bin 5
    [0.890, 0.978],   # Bin 6
    [0.978, 1.010],   # Bin 7
    [1.010, 1.380],   # Bin 8
    [1.380, 1.660],   # Bin 9
    [1.660, 2.140],   # Bin 10
    [2.140, 2.930],   # Bin 11
    [2.930, 4.270],   # Bin 12
    [4.270, 6.520],   # Bin 13
    [6.520, 10.300],  # Bin 14
    [10.300, 13.600], # Bin 15
    [13.600, 18.100], # Bin 16
    [18.100, 24.100], # Bin 17
    [24.100, 32.100], # Bin 18
    [32.100, 42.500], # Bin 19
    [42.500, 56.300], # Bin 20
])
# Geometric mean midpoints
CAS_D_MID_ALL = np.sqrt(CAS_BIN_BOUNDS[:, 0] * CAS_BIN_BOUNDS[:, 1])
# Append NaN for the 21st column (placeholder); pipeline filters it
CAS_D_MID_ALL = np.append(CAS_D_MID_ALL, np.nan)
CAS_ACTIVE_BIN_INDICES = list(range(0, 20))  # bin_00..bin_19 (bin_20 is NaN)
CAS_D_MIN_UM = 0.600
CAS_D_MAX_UM = 56.300

# CIP - Cloud Imaging Probe (DMT)
# 62 linearly-spaced bins covering D = 7.5-937.5 um (15 um step).
# For linear spacing, the arithmetic mean is the correct midpoint:
#     D_mid = (D_lower + D_upper) / 2
#
# Note: VOCALS data files have 63 CIP columns (CIP_bin_00..CIP_bin_62).
# Only the first 62 represent valid bins; the 63rd is a placeholder.
CIP_N_BINS_NOMINAL = 62
CIP_D_MID_UM = np.arange(15.0, 15.0 + 62 * 15.0, 15.0, dtype=float)
# Append NaN for the 63rd column; pipeline filters it
CIP_D_MID_UM = np.append(CIP_D_MID_UM, np.nan)
CIP_D_MIN_UM = 7.5
CIP_D_MAX_UM = 937.5

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
# VOCALS 1 Hz data with thresholds tuned for SE Pacific stratocumulus, which
# tends to be thinner and more broken than POST/MASE NE Pacific stratocumulus.
# Final depth and duration thresholds are relaxed accordingly.
DT = 1.0   # s - VOCALS native sampling rate

PROFILE = {
    # Cloud-core thresholds (per-point)
    "lwc_threshold"        : 0.05,   # g/m^3 - minimum LWC for in-cloud point
    "nc_threshold"         : 10.0,   # cm^-3 - minimum droplet count
    "vz_threshold"         : 0.5,    # m/s   - minimum |Vz| for ascent/descent

    # Vertical velocity smoothing (rolling mean window)
    "alt_smooth_win"       : 5,      # 5 points = 5 seconds at 1 Hz

    # Minimum profile size
    "min_profile_pts"      : 10,     # 10 points = 10 seconds

    # Segmentation / merging
    "summary_gap_seconds"  : 120,    # split if gap exceeds this
    "stitch_gap_seconds"   : 300,    # merge nearby profiles into super-profiles

    # Final QC (relaxed for VOCALS broken cloud)
    "final_min_depth_m"    : 100.0,  # m - minimum cloud depth
    "final_min_duration_s" : 60.0,   # s - minimum profile duration
    "final_min_lwc"        : 0.05,   # g/m^3 - minimum profile peak LWC

    # Time-series integrity (sensor dropout detection)
    "integrity_min_pts"      : 15,
    "integrity_max_gap_ratio": 0.10,
    # Toggle: VOCALS 1 Hz -> integrity check meaningful (enabled).
    # Set to False for slower-cadence campaigns (e.g., MASE 10 s averaging)
    # where intra-profile dropout detection becomes noise-dominated.
    "integrity_check_enabled": True,

    # Profile span score (sawtooth-aware vertical-penetration filter).
    # Defined as (z_max - z_min) / sum(|d_alt|).
    # Values near 1: monotonic ascent/descent. <0.4: level-leg wandering.
    # Set to 0.0 to disable. VOCALS broken cloud regime: 0.0 (rarely needed).
    "min_span_score"         : 0.0,
}

# =============================================================================
# 7. Drizzle classification thresholds
# =============================================================================
# Drizzle indicators based on large-droplet concentration and LWC ratio.
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
# VOCALS-specific: strict VZA limit (40 deg, following Ma et al. 2017) and
# tight bbox padding (0.10 deg) consistent with the broken-cloud nature of
# SE Pacific stratocumulus.
MODIS = {
    # MODIS overpass search window
    "max_diff_min"    : 90,        # +/- minutes around aircraft profile
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
# The difference quantifies the assumption sensitivity (Package F, paper Section 4.7).
NDLIT = {
    "k_lit"    : 0.80,     # Martin et al. (1994) marine stratocumulus
    "f_ad_lit" : 0.80,     # adiabatic fraction (operational default, Quaas 2006)
    "c_w_lit"  : 2.3e-3,  # g/m^4 - marine Sc baseline (T=12C, P=925 hPa)
}

# =============================================================================
# 10c. Post-processing filters (sensor artifact removal)
# =============================================================================
# Applied in step05 after re_eff calculation. Filters non-physical effective
# radius values that arise from CIP sensor artifacts. The threshold is set
# above the physiological drizzle Re tail to preserve real heavy-drizzle
# measurements.
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
    "re_max_physical_um": 50.0,   # Wood (2012) marine Sc in-situ upper bound
}

# =============================================================================
# 10d. Final-check physical filters (Bug #18: config-driven)
# =============================================================================
# Applied in step07 after microphysics. Eliminates profiles with
# non-physical f_ad or sub-detection Nd.
FILTERS = {
    "f_ad_max" : 1.0,    # Painemal & Zuidema (2011) physical limit
    "nd_min"   : 5.0,    # cm^-3 - Bennartz (2007) sub-detection threshold
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
# =============================================================================
# utils.py - Shared helpers for MASE 2005 pipeline
# =============================================================================
# All physical and mathematical computations live in this module. Step files
# act as orchestrators (read input -> call utils -> write output); they do
# not contain physics or math logic.
#
# This separation ensures:
#   - Each utility function is testable in isolation
#   - Scientific formulas have a single source of truth
#   - Step files stay readable as workflow descriptions
#
# This module is campaign-agnostic. Campaign-specific values (bin midpoints,
# column names, thresholds) come from config.py. The same utils.py works
# across POST, MASE, and VOCALS pipelines without modification.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# numpy 2.0+ compatibility: trapezoid was renamed from trapz
try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz


# =============================================================================
# Column helpers
# =============================================================================
def get_cas_columns(df, prefix):
    """
    Return CAS bin column names in numeric order.

    MASE bin column names contain a space and a 1-based index:
        "CAS Bin 1", "CAS Bin 2", ..., "CAS Bin 20"

    A naive string sort would order "CAS Bin 10" before "CAS Bin 2".
    This function extracts the trailing integer and sorts numerically.

    Works with VOCALS-style names ("CAS_bin_00") too: those sort
    correctly under either string or numeric ordering.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]

    def _bin_index(name):
        # Strip prefix, then strip any leading underscores/spaces.
        tail = name[len(prefix):].strip().lstrip("_")
        try:
            return int(tail)
        except ValueError:
            return 9999  # unparseable -> push to end, deterministic

    return sorted(cols, key=_bin_index)


def get_cip_columns(df, prefix):
    """
    Return CIP bin column names in numeric order.

    MASE: "CIP Bin 1", "CIP Bin 2", ..., "CIP Bin 60"
    See get_cas_columns for sorting rationale.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]

    def _bin_index(name):
        tail = name[len(prefix):].strip().lstrip("_")
        try:
            return int(tail)
        except ValueError:
            return 9999

    return sorted(cols, key=_bin_index)


def check_required_columns(df, required_cols):
    """
    Validate that required columns exist in the DataFrame.
    Raises ValueError listing all missing columns.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns ({len(missing)}):\n  " +
            "\n  ".join(missing)
        )


def select_active_bins(d_mid_um, columns, cutoff_um=None):
    """
    Filter bins by NaN midpoints and optional minimum-diameter cutoff.

    Parameters
    ----------
    d_mid_um : array-like
        Bin midpoint diameters (um). May contain NaN for invalid bins.
    columns : list of str
        Column names corresponding to the bins (same length as d_mid_um).
    cutoff_um : float, optional
        If given, only bins with midpoint > cutoff_um are kept. If None,
        only NaN-flagged bins are filtered.

    Returns
    -------
    d_valid : np.ndarray
        Filtered midpoint diameters (um).
    cols_valid : list of str
        Filtered column names, aligned with d_valid.

    Notes
    -----
    Used by step05 for:
      - CAS bin selection: drop NaN-flagged placeholder bins (e.g., CAS_bin_20
        in VOCALS, which is structurally NaN in raw data files).
      - CIP bin selection with CIP_CUTOFF_UM = 50 um (Wood 2012 marine Sc
        drizzle threshold) to prevent double-counting in the CAS-CIP overlap
        region where CAS dominates by factors of 2.6x to 361x across campaigns.
    """
    d = np.asarray(d_mid_um, dtype=float)
    valid = ~np.isnan(d)
    if cutoff_um is not None:
        valid = valid & (d > cutoff_um)
    d_valid = d[valid]
    cols_valid = [c for c, v in zip(columns, valid) if v]
    return d_valid, cols_valid


# =============================================================================
# QC helpers
# =============================================================================
def clip_lower(df_or_series, lower=0.0):
    """Clip negative values to a lower bound (default 0)."""
    return df_or_series.clip(lower=lower)


def mask_outside_range(series, vmin, vmax):
    """Replace values outside [vmin, vmax] with NaN."""
    return series.where((series >= vmin) & (series <= vmax), other=np.nan)


# =============================================================================
# Vertical velocity helpers (VOCALS 1 Hz time series)
# =============================================================================
def compute_vertical_velocity(df, alt_col, time_col, smooth_win=5, group_col=None):
    """
    Compute aircraft vertical velocity from smoothed altitude derivatives.

    Method:
        Alt_Smooth = Alt.rolling(window=smooth_win, center=True).mean()
        dt_actual  = time.diff().dt.total_seconds()
        Vz         = Alt_Smooth.diff() / dt_actual

    If group_col is provided, computation is performed per group (e.g., per
    flight) to avoid spurious values at flight boundaries.

    Returns
    -------
    df : DataFrame
        Input DataFrame with added 'Alt_Smooth' and 'Vz' columns.
    """
    if group_col is None:
        df = df.sort_values(time_col).copy()
        df["Alt_Smooth"] = (
            df[alt_col].rolling(window=smooth_win, center=True, min_periods=1).mean()
        )
        dt_actual = df[time_col].diff().dt.total_seconds().fillna(1.0)
        df["Vz"] = df["Alt_Smooth"].diff().fillna(0.0) / dt_actual
        return df

    out = []
    for _, df_g in df.groupby(group_col, sort=False):
        df_g = df_g.sort_values(time_col).copy()
        df_g["Alt_Smooth"] = (
            df_g[alt_col].rolling(window=smooth_win, center=True, min_periods=1).mean()
        )
        dt_actual = df_g[time_col].diff().dt.total_seconds().fillna(1.0)
        df_g["Vz"] = df_g["Alt_Smooth"].diff().fillna(0.0) / dt_actual
        out.append(df_g)
    return pd.concat(out, ignore_index=False).sort_index()


def make_contiguous_blocks(mask):
    """
    Assign block IDs to contiguous True regions in a boolean mask.
    False positions are NaN. Used for cloud segment detection.
    """
    mask = pd.Series(mask).fillna(False).astype(bool)
    block = (mask != mask.shift()).cumsum()
    return block.where(mask, other=np.nan)


def compute_max_gap_ratio(time_series):
    """
    Time-series integrity check.

    Returns
    -------
    max_gap_sec : float
        Largest gap between consecutive timestamps (seconds).
    gap_ratio : float in [0, 1]
        max_gap / total_span. Used to reject profiles with sampling gaps.
    """
    ts = pd.Series(time_series).sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return 0.0, 0.0
    diffs = ts.diff().dt.total_seconds().dropna()
    max_gap_sec = float(diffs.max())
    total_span = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())
    if total_span <= 0:
        return max_gap_sec, 0.0
    return max_gap_sec, max_gap_sec / total_span


def compute_span_score(altitudes):
    """
    Vertical-penetration quality metric (sawtooth-friendly).

    Computes the ratio of the profile's altitude range to the total
    altitude path length:
        span_score = (z_max - z_min) / sum(|d_alt|)

    Interpretation:
      ~1.0  : monotonic ascent or descent (clean vertical profile)
      ~0.6  : stepped or sawtooth descent (multiple short level segments
              connected by genuine vertical motion - acceptable)
      <0.4  : level-leg / wandering (instrument samples a narrow
              altitude band over a long time - reject)

    This metric was designed to handle MASE 10 s level-leg flight patterns
    where the aircraft samples cloud at constant altitude for extended
    periods. A pure monotonicity score (net displacement / path) would
    incorrectly reject sawtooth profiles that do contain real vertical
    penetration. span_score uses the altitude RANGE rather than net
    displacement, so it gives credit to profiles that go down-then-up or
    descend in steps.

    For VOCALS / POST 1 Hz zigzag flights, span_score is typically >= 0.7,
    so a threshold of 0.4 leaves them unaffected.

    Parameters
    ----------
    altitudes : array-like
        Altitude values along the profile, time-ordered.

    Returns
    -------
    float in [0, 1]  (or NaN if too few points or zero path)
    """
    alt = np.asarray(altitudes, dtype=float)
    alt = alt[~np.isnan(alt)]
    if len(alt) < 2:
        return np.nan
    H    = float(alt.max() - alt.min())
    path = float(np.sum(np.abs(np.diff(alt))))
    return H / path if path > 0 else np.nan


# =============================================================================
# Effective radius and spectral broadening
# =============================================================================
def calc_re_effective(N_matrix, r_um):
    """
    Effective radius - third-to-second moment ratio of the size distribution.

        re = sum(N_i * r_i^3) / sum(N_i * r_i^2)

    Parameters
    ----------
    N_matrix : array (n_points, n_bins)
        Droplet number concentration per bin (cm^-3). NaN/negative -> 0.
    r_um : array (n_bins,)
        Bin midpoint radii (um). NaN entries are filtered out automatically.

    Returns
    -------
    re : array (n_points,)
        Effective radius per point (um).

    References
    ----------
    Hansen and Travis (1974). re is the radius to which satellite optical
    retrievals are sensitive (third-to-second moment ratio of the DSD).
    Nakajima and King (1990) bispectral retrieval recovers this quantity
    directly from MODIS-channel reflectances.
    """
    r_um = np.asarray(r_um, dtype=float)
    valid = ~np.isnan(r_um)
    if not valid.any():
        return np.full(N_matrix.shape[0], np.nan)

    N = np.where(np.isfinite(N_matrix) & (N_matrix > 0), N_matrix, 0.0)
    N_valid = N[:, valid]
    r_valid = r_um[valid]

    num = (N_valid * r_valid**3).sum(axis=1)
    den = (N_valid * r_valid**2).sum(axis=1)
    return np.where(den > 0, num / den, np.nan)


def calc_k_martin(N_matrix, r_um):
    """
    Martin et al. (1994) spectral broadening parameter.

        k = M2^3 / (M3^2 * M0)
        where M_n = sum(N_i * r_i^n)

    Equivalent to (r_v / r_e)^3, with r_v the volume-mean radius. A narrower
    distribution yields k -> 1; a broader distribution gives smaller k.
    Typical marine stratocumulus: k = 0.6 - 0.9 (Martin 1994; Pawlowska 2006).

    Returns NaN where M0 < 1 or M3 <= 0 (no valid droplets).
    """
    r_um = np.asarray(r_um, dtype=float)
    valid = ~np.isnan(r_um)
    if not valid.any():
        return np.full(N_matrix.shape[0], np.nan)

    N = np.where(np.isfinite(N_matrix) & (N_matrix > 0), N_matrix, 0.0)
    N_valid = N[:, valid]
    r_valid = r_um[valid]

    M0 = N_valid.sum(axis=1)
    M2 = (N_valid * r_valid**2).sum(axis=1)
    M3 = (N_valid * r_valid**3).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((M0 >= 1) & (M3 > 0), M2**3 / (M3**2 * M0), np.nan)


# =============================================================================
# Thermodynamics
# =============================================================================
def calc_cw(T_C, P_hPa,
             Lv=2.5e6, Rv=461.5, Rd=287.05, cp=1005.0, g=9.81):
    """
    Adiabatic LWC lapse rate ('condensation rate') from temperature and pressure.

    Combines the Bolton (1980) saturation vapor pressure formula with the
    moist adiabatic lapse rate to give dLWC/dz under saturated ascent.

        es      = 611.2 * exp(17.67 * T_C / (T_C + 243.5))   [Pa]
        qs      = (Rd/Rv) * es / (P - es)                     [kg/kg]
        dqs/dT  = Lv * qs / (Rv * T_K^2)
        Gamma_m = g * (1 + Lv*qs/(Rd*T_K)) /
                  (cp + Lv^2*qs/(Rv*T_K^2))
        c_w     = rho * dqs/dT * Gamma_m                      [g/m^3/m]

    A defensive lower floor of 1e-4 is applied to prevent division-by-zero
    in downstream Grosvenor calculations under pathological inputs. For
    typical marine Sc (T = 10-20 C, P = 950-1010 hPa), c_w = 1.5-2.5 g/m^4
    and the floor is never reached.

    Used by:
      - step05 compute_fad: cloud-top adiabatic LWC for f_ad
      - step05 compute_cw_profile: c_w input to Grosvenor Nd formula

    References
    ----------
    Bolton (1980), MWR. Brenguier et al. (2000), J. Atmos. Sci.
    Painemal and Zuidema (2011), JGR.
    """
    T_K  = T_C + 273.15
    P_Pa = P_hPa * 100.0
    eps     = Rd / Rv
    rho     = P_Pa / (Rd * T_K)
    es      = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    qs      = eps * es / (P_Pa - es)
    dqs_dT  = Lv * qs / (Rv * T_K**2)
    gamma_m = g * (1 + Lv*qs/(Rd*T_K)) / (cp + Lv**2*qs/(Rv*T_K**2))
    cw = rho * dqs_dT * gamma_m * 1000.0   # g/m^4
    return np.maximum(cw, 1e-4)


def calc_lwc_ad_max(T_base_C, P_base_hPa, H_m):
    """
    Adiabatic liquid water content at cloud top.

        LWC_ad_max = c_w(T_base, P_base) * H

    Used in the Painemal and Zuidema (2011) f_ad calculation:
        f_ad = LWC_obs_max / LWC_ad_max

    where LWC_obs_max is the 95th-percentile observed LWC in the profile
    (robust against single-point outliers near cloud top).

    Parameters
    ----------
    T_base_C : float
        Cloud-base temperature (C).
    P_base_hPa : float
        Cloud-base pressure (hPa).
    H_m : float
        Cloud depth (m), z_top - z_base.

    Returns
    -------
    LWC_ad_max : float
        Maximum adiabatic LWC at cloud top (g/m^3).

    References
    ----------
    Painemal and Zuidema (2011), JGR, Eq. 4. This profile-based formulation
    avoids systematic positive bias in f_ad that arises when cloud-base
    altitude is taken as the within-profile minimum (typically 50-150 m
    above the true cloud base).
    """
    return calc_cw(T_base_C, P_base_hPa) * H_m


# =============================================================================
# Cloud optical thickness (tau)
# =============================================================================
def calc_tau_layer(lwc_gm3, re_um, dz_m, rho_w_gm3=1e6):
    """
    Per-layer optical thickness from LWC and effective radius.

        tau_layer = (3 * LWC * dz) / (2 * rho_w * re)

    Profile-integrated tau is obtained by summing layer values:
        tau = sum(tau_layer)

    Parameters
    ----------
    lwc_gm3 : array
        Liquid water content per layer (g/m^3).
    re_um : array
        Effective radius per layer (um).
    dz_m : array
        Layer thickness (m).
    rho_w_gm3 : float
        Liquid water density (default 1e6 g/m^3 = 1 g/cm^3).

    Returns
    -------
    tau_layer : array
        Per-layer optical thickness (dimensionless).

    References
    ----------
    Stephens (1978), J. Atmos. Sci. Standard relation between cloud LWP,
    re, and tau under geometric-optics extinction (Q_ext = 2):
        tau = (3/2) * LWP / (rho_w * re)
    Differentiated layer-by-layer for vertical integration through aircraft
    profiles.
    """
    re_m = re_um * 1e-6
    return (3.0 * lwc_gm3) / (2.0 * rho_w_gm3 * re_m) * dz_m


# =============================================================================
# Mixing regime classification
# =============================================================================
def classify_mixing_regime(f_ad, adiabatic_min=0.80, sub_adiabatic_min=0.40):
    """
    Classify cloud mixing regime by adiabatic fraction f_ad.

        f_ad >= adiabatic_min      -> "adiabatic"
        f_ad >= sub_adiabatic_min  -> "sub_adiabatic"
        f_ad <  sub_adiabatic_min  -> "strongly_mixed"
        NaN                        -> NaN

    References
    ----------
    Painemal and Zuidema (2011), JGR: typical marine Sc f_ad = 0.5-0.9.
    Boers et al. (1998): broken-cloud edges f_ad < 0.3.
    Albrecht et al. (1985) and Brenguier et al. (2000) discuss the
    adiabatic-to-strongly-mixed continuum.
    """
    if pd.isna(f_ad):
        return np.nan
    if f_ad >= adiabatic_min:
        return "adiabatic"
    if f_ad >= sub_adiabatic_min:
        return "sub_adiabatic"
    return "strongly_mixed"


# =============================================================================
# Grosvenor cloud droplet number concentration
# =============================================================================
def calc_nd_grosvenor(re_um, tau, k, f_ad, c_w_gm4,
                       Q_ext=2.0, rho_w_gcm3=1.0):
    """
    Grosvenor et al. (2018) cloud droplet number concentration retrieval.

        Nd [cm^-3] = (sqrt(5) / (2 * pi * k)) *
                     sqrt(f_ad * c_w * tau / (Q_ext * rho_w * re^5))

    Unit conversions inside the function:
        re_um   [um]   -> re_cm  = re_um * 1e-4   [cm]
        c_w_gm4 [g/m^4] -> c_w_cgs = c_w_gm4 * 1e-8 [g/cm^4]
        rho_w  = 1 g/cm^3, Q_ext = 2 (geometric optics)

    Parameters
    ----------
    re_um : float
        Effective radius (um).
    tau : float
        Cloud optical thickness (dimensionless).
    k : float
        Spectral broadening parameter (Martin 1994).
    f_ad : float
        Adiabatic fraction (Painemal and Zuidema 2011).
    c_w_gm4 : float
        Adiabatic LWC lapse rate (g/m^4).

    Returns
    -------
    Nd : float
        Cloud droplet number concentration (cm^-3). NaN if any input is
        invalid (NaN, zero, or negative).

    References
    ----------
    Grosvenor et al. (2018), Reviews of Geophysics, Eq. 19. Originally
    derived from the adiabatic stratocumulus assumption in Brenguier
    et al. (2000) and Bennartz (2007). Standard satellite retrieval
    assumes k = 0.8 and f_ad = 0.8 (Quaas et al. 2006); this work uses
    the in-situ measured values of k, f_ad, and c_w when applying the
    formula to MODIS-retrieved tau and re.
    """
    if any(pd.isna([re_um, tau, k, f_ad, c_w_gm4])):
        return np.nan
    if re_um <= 0 or tau <= 0 or k <= 0 or f_ad <= 0 or c_w_gm4 <= 0:
        return np.nan

    re_cm   = re_um * 1e-4
    c_w_cgs = c_w_gm4 * 1e-8
    numer = np.sqrt(5) * np.sqrt(f_ad * c_w_cgs * tau)
    denom = 2 * np.pi * k * np.sqrt(Q_ext * rho_w_gcm3 * re_cm**5)
    return numer / denom        # cm^-3


# =============================================================================
# LWP integration
# =============================================================================
def calc_lwp_trapezoid(altitudes_m, lwc_gm3, lwc_min=0.01):
    """
    Liquid water path by trapezoid integration over altitude.

        LWP = integral(LWC, dz)  [g/m^2]

    Points below lwc_min are excluded (treated as below-threshold or
    out-of-cloud). Integration uses the absolute value to handle either
    ascent or descent profiles.

    Parameters
    ----------
    altitudes_m : array
        Altitude per point (m).
    lwc_gm3 : array
        LWC per point (g/m^3).
    lwc_min : float
        LWC threshold for valid in-cloud points (default 0.01 g/m^3).

    Returns
    -------
    LWP : float
        Integrated LWP (g/m^2). NaN if fewer than 2 valid points.
    n_valid : int
        Number of points used in the integration.
    """
    arr_alt = np.asarray(altitudes_m, dtype=float)
    arr_lwc = np.asarray(lwc_gm3, dtype=float)
    mask = (arr_lwc > lwc_min) & np.isfinite(arr_alt)
    if mask.sum() < 2:
        return np.nan, mask.sum()
    order = np.argsort(arr_alt[mask])
    return abs(_trapz(arr_lwc[mask][order], arr_alt[mask][order])), mask.sum()


# =============================================================================
# Figure helper
# =============================================================================
def save_figure(fig, path, dpi=150):
    """Save figure to disk, close it, and log the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path}")
# =============================================================================
# step10_nd_calculation.py  -  POST 2008
# =============================================================================
# Final pipeline step: compute MODIS-retrieved Nd from Re/tau, compare
# against in-situ Nd from CAS, and quantify sensitivity to assumed
# parameters (k, f_ad, c_w).
#
# TWO BIAS CALCULATIONS (paper Section 4.7):
#
#   bias_calc : Grosvenor (2018) with IN-SITU measured k, f_ad, c_w
#   bias_lit  : Grosvenor (2018) with LITERATURE defaults
#               (k=0.80, f_ad=0.80, c_w=2e-3 g/m^4)
#
# The difference between bias_calc and bias_lit quantifies the bias that
# operational satellite Nd retrievals carry by assuming generic marine Sc
# parameters instead of measured ones.
#
# Grosvenor formula (utils.calc_nd_grosvenor):
#   Nd [cm^-3] = (sqrt(5) / (2*pi*k)) *
#                sqrt(f_ad * c_w * tau / (Q_ext * rho_w * re^5))
#
# Both bias calculations also computed for the 3.7 um channel (Re_MODIS_37,
# tau_MODIS_37) to detect vertical heterogeneity (dNd = Nd_2.1 - Nd_3.7).
#
# OUTPUT:
#   STEP09_MODIS_MATCHES_CSV is overwritten in-place with the new columns:
#     Nd_MODIS_21_calc, Nd_MODIS_37_calc, bias_21_calc, bias_37_calc
#     Nd_MODIS_21_lit,  Nd_MODIS_37_lit,  bias_21_lit,  bias_37_lit
#     dNd_calc, dNd_lit
# =============================================================================

import sys
import numpy as np
import pandas as pd

import config
from utils import calc_nd_grosvenor


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print(f"  step10 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  "
          f"(MODIS Nd Calculation - Grosvenor 2018)")
    print("=" * 70)

    if not config.STEP09_MODIS_MATCHES_CSV.exists():
        print(f"\n  [FAIL] {config.STEP09_MODIS_MATCHES_CSV.name} not found.")
        print(f"         Run step09 first.")
        sys.exit(1)
    if not config.STEP04_GOLDEN_CASE_CSV.exists():
        print(f"\n  [FAIL] {config.STEP04_GOLDEN_CASE_CSV.name} not found.")
        print(f"         Run step07 first.")
        sys.exit(1)

    print(f"\n  Loading files...")
    modis = pd.read_csv(config.STEP09_MODIS_MATCHES_CSV)
    gc    = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    print(f"  [OK] modis: {modis.shape}    gc: {gc.shape}")

    # ----------------------------------------------------------
    # 1. Drop pre-existing computed columns (idempotency)
    # ----------------------------------------------------------
    drop_cols = [
        # bias_calc columns
        "Nd_MODIS_21_calc", "Nd_MODIS_37_calc",
        "bias_21_calc", "bias_37_calc", "dNd_calc",
        # bias_lit columns
        "Nd_MODIS_21_lit", "Nd_MODIS_37_lit",
        "bias_21_lit", "bias_37_lit", "dNd_lit",
        # legacy column names from earlier versions
        "Nd_MODIS_21", "Nd_MODIS_37", "bias_21", "bias_37", "dNd",
        # in-situ merge columns (will be re-merged below)
        "Nd_median", "re_cas_median", "tau_main",
        "f_ad_mean", "k_median", "c_w_median",
        "drizzle_regime", "LWP_insitu", "cloud_depth",
    ]
    modis = modis.drop(columns=[c for c in drop_cols if c in modis.columns])

    # ----------------------------------------------------------
    # 2. Merge in-situ parameters from gc
    # ----------------------------------------------------------
    insitu_cols = [
        "cloud_id",
        "Nd_median", "re_cas_median", "tau_main",
        "f_ad_mean", "k_median", "c_w_median",
        "drizzle_regime", "LWP_insitu", "cloud_depth",
    ]
    insitu_cols = [c for c in insitu_cols if c in gc.columns]
    modis = modis.merge(gc[insitu_cols], on="cloud_id", how="left")
    n_with_insitu = modis["Nd_median"].notna().sum()
    print(f"\n  In-situ merge : {n_with_insitu}/{len(modis)} profiles matched")

    # ----------------------------------------------------------
    # 3a. bias_calc - Grosvenor with IN-SITU measured k, f_ad, c_w
    # ----------------------------------------------------------
    print(f"\n  Computing Nd_MODIS via Grosvenor (2018)...")
    print(f"    bias_calc : in-situ measured k, f_ad, c_w")
    modis["Nd_MODIS_21_calc"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_21"),  r.get("tau_MODIS_21"),
            r.get("k_median"),     r.get("f_ad_mean"),
            r.get("c_w_median")
        ),
        axis=1,
    )
    modis["Nd_MODIS_37_calc"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_37"),  r.get("tau_MODIS_37"),
            r.get("k_median"),     r.get("f_ad_mean"),
            r.get("c_w_median")
        ),
        axis=1,
    )
    # Paired-pool 2.1 um Nd (Felsefe A): used ONLY for dNd_calc spectral
    # difference, ensuring apples-to-apples comparison with Nd_MODIS_37_calc.
    # The primary Nd_MODIS_21_calc above (qc_21 pool) is unchanged and continues
    # to drive bias_21_calc / Package C-D-G-H statistics.
    modis["Nd_MODIS_21_paired37_calc"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_21_paired37"),  r.get("tau_MODIS_21_paired37"),
            r.get("k_median"),              r.get("f_ad_mean"),
            r.get("c_w_median")
        ),
        axis=1,
    )

    # ----------------------------------------------------------
    # 3b. bias_lit - Grosvenor with LITERATURE defaults
    # ----------------------------------------------------------
    k_lit    = config.NDLIT["k_lit"]
    f_ad_lit = config.NDLIT["f_ad_lit"]
    c_w_lit  = config.NDLIT["c_w_lit"]
    print(f"    bias_lit  : k={k_lit}, f_ad={f_ad_lit}, c_w={c_w_lit:.2e}")
    modis["Nd_MODIS_21_lit"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_21"),  r.get("tau_MODIS_21"),
            k_lit, f_ad_lit, c_w_lit
        ),
        axis=1,
    )
    modis["Nd_MODIS_37_lit"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_37"),  r.get("tau_MODIS_37"),
            k_lit, f_ad_lit, c_w_lit
        ),
        axis=1,
    )
    # Paired-pool literature 2.1 um Nd (for dNd_lit apples-to-apples)
    modis["Nd_MODIS_21_paired37_lit"] = modis.apply(
        lambda r: calc_nd_grosvenor(
            r.get("Re_MODIS_21_paired37"),  r.get("tau_MODIS_21_paired37"),
            k_lit, f_ad_lit, c_w_lit
        ),
        axis=1,
    )

    n_calc = modis["Nd_MODIS_21_calc"].notna().sum()
    n_lit  = modis["Nd_MODIS_21_lit"].notna().sum()
    print(f"  [OK] Nd_MODIS_21_calc : {n_calc}/{len(modis)} profiles")
    print(f"  [OK] Nd_MODIS_21_lit  : {n_lit}/{len(modis)} profiles")

    # ----------------------------------------------------------
    # 4. Bias and dNd (both versions)
    # ----------------------------------------------------------
    modis["bias_21_calc"] = modis["Nd_MODIS_21_calc"] / modis["Nd_median"]
    modis["bias_37_calc"] = modis["Nd_MODIS_37_calc"] / modis["Nd_median"]
    # dNd uses paired-pool 21 to match the pool used for 37 (Felsefe A)
    modis["dNd_calc"]     = modis["Nd_MODIS_21_paired37_calc"] - modis["Nd_MODIS_37_calc"]

    modis["bias_21_lit"]  = modis["Nd_MODIS_21_lit"]  / modis["Nd_median"]
    modis["bias_37_lit"]  = modis["Nd_MODIS_37_lit"]  / modis["Nd_median"]
    modis["dNd_lit"]      = modis["Nd_MODIS_21_paired37_lit"]  - modis["Nd_MODIS_37_lit"]

    # ----------------------------------------------------------
    # 5. Save
    # ----------------------------------------------------------
    modis.to_csv(config.STEP09_MODIS_MATCHES_CSV, index=False)
    print(f"\n  [SAVE] {config.STEP09_MODIS_MATCHES_CSV.name}  ({modis.shape})")

    # ----------------------------------------------------------
    # 6. Summary
    # ----------------------------------------------------------
    matched = modis[modis["match_status"] == "MATCHED"].copy()
    print(f"\n  Match status:")
    print("  " + modis["match_status"].value_counts().to_string().replace("\n", "\n  "))
    print(f"\n  MATCHED profiles: {len(matched)}")

    if len(matched) > 0:
        # bias_calc summary
        print(f"\n  --- bias_calc (in-situ assumptions) ---")
        cols_calc = ["Nd_median", "Nd_MODIS_21_calc", "Nd_MODIS_37_calc",
                     "bias_21_calc", "bias_37_calc", "dNd_calc"]
        cols_calc = [c for c in cols_calc if c in matched.columns]
        if cols_calc:
            print("  " + matched[cols_calc].describe().round(2)
                                  .to_string().replace("\n", "\n  "))

        # bias_lit summary
        print(f"\n  --- bias_lit (literature defaults) ---")
        cols_lit = ["Nd_MODIS_21_lit", "Nd_MODIS_37_lit",
                    "bias_21_lit", "bias_37_lit", "dNd_lit"]
        cols_lit = [c for c in cols_lit if c in matched.columns]
        if cols_lit:
            print("  " + matched[cols_lit].describe().round(2)
                                  .to_string().replace("\n", "\n  "))

        # Side-by-side bias comparison
        print(f"\n  --- bias_calc vs bias_lit (median, MATCHED) ---")
        m_calc = matched["bias_21_calc"].median()
        m_lit  = matched["bias_21_lit"].median()
        print(f"    bias_21_calc median : {m_calc:.3f}x")
        print(f"    bias_21_lit  median : {m_lit:.3f}x")
        print(f"    Difference          : {m_lit - m_calc:+.3f}  "
              f"(positive -> literature inflates bias)")

        # Per-regime
        if "drizzle_regime" in matched.columns:
            print(f"\n  --- Median bias by drizzle regime ---")
            stat_cols = [c for c in ["bias_21_calc", "bias_21_lit"]
                         if c in matched.columns]
            if stat_cols:
                grp = (matched.groupby("drizzle_regime", observed=True)[stat_cols]
                              .median().round(3))
                print("  " + grp.to_string().replace("\n", "\n  "))

    print("\n" + "=" * 70)
    print(f"  step10 COMPLETE - Pipeline finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
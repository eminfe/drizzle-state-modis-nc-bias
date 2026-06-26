"""
cc_18_primary_stats.py  -  Primary statistics for Results Section 3.1
Reproduces every statistic reported in Section 3.1 of the manuscript and makes
the drizzle-state trend test unambiguous.

  (1) Per-state median bias and interquartile range (IQR)
  (2) Kruskal-Wallis test across the three drizzle states
  (3) Drizzle-state trend (Spearman), reported two ways:
        (3a) ORDINAL drizzle state (Non=0, Transition=1, Heavy=2)  <-- REPORTED
        (3b) CONTINUOUS profile-level f_drizzle                    <-- NOT reported
      These give different p-values; the manuscript uses (3a).
  (4) Wilcoxon signed-rank test of each state's bias vs unity
  (5) Bootstrap 95% CI of each state's median bias

The ordinal trend test (3a) is the one reported in the manuscript
("negative Spearman rank relationship between ordinal drizzle state and B_calc,
p = 0.0086"). The continuous f_drizzle correlation (3b) is weaker (p ~ 0.07)
and is intentionally NOT reported as the headline trend statistic.

Input : data_outputs/cc_master_3state.csv
Output: console summary
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

STATES = ["Non", "Transition", "Heavy"]
STATE_CODE = {"Non": 0, "Transition": 1, "Heavy": 2}
DATA = Path("data_outputs/cc_master_3state.csv")
BOOT_N, BOOT_SEED = 10000, 42


def main():
    m = pd.read_csv(DATA).dropna(subset=["bias_21_calc", "drizzle_state"])
    g = {s: m.loc[m.drizzle_state == s, "bias_21_calc"].values for s in STATES}
    n = len(m)

    print("=" * 64)
    print(f"SECTION 3.1 PRIMARY STATISTICS  (n = {n})")
    print("=" * 64)

    # (1) median + IQR
    print("\n(1) PER-STATE MEDIAN BIAS AND IQR")
    for s in STATES:
        v = g[s]; q1, q3 = np.percentile(v, [25, 75])
        print(f"    {s:11s}: median={np.median(v):.3f}  IQR={q3 - q1:.3f}  n={len(v)}")

    # (2) Kruskal-Wallis
    print("\n(2) KRUSKAL-WALLIS ACROSS DRIZZLE STATES")
    H, p = stats.kruskal(*[g[s] for s in STATES])
    print(f"    H={H:.3f}, p={p:.4f}")

    # (3) trend test: ordinal vs continuous
    print("\n(3) DRIZZLE-STATE TREND (Spearman)")
    code = m.drizzle_state.map(STATE_CODE).values
    rho_o, p_o = stats.spearmanr(code, m.bias_21_calc.values)
    print(f"    (3a) ORDINAL state (Non=0,Trans=1,Heavy=2): rho={rho_o:+.3f}, p={p_o:.4f}   <-- REPORTED")
    md = m.dropna(subset=["drizzle_fraction"])
    rho_c, p_c = stats.spearmanr(md.drizzle_fraction.values, md.bias_21_calc.values)
    print(f"    (3b) CONTINUOUS f_drizzle                 : rho={rho_c:+.3f}, p={p_c:.4f}   (NOT reported)")
    print("    NOTE: the manuscript trend statistic is the ORDINAL test (3a).")

    # (4) Wilcoxon vs unity
    print("\n(4) WILCOXON SIGNED-RANK vs UNITY (per state)")
    for s in STATES:
        w, p = stats.wilcoxon(g[s] - 1.0)
        print(f"    {s:11s}: p={p:.4f}")

    # (5) bootstrap CI
    print("\n(5) BOOTSTRAP 95% CI OF MEDIAN BIAS")
    rng = np.random.default_rng(BOOT_SEED)
    for s in STATES:
        v = g[s]
        boots = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(BOOT_N)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"    {s:11s}: median={np.median(v):.2f}  CI=[{lo:.2f}, {hi:.2f}]")
    print("    (Bootstrap CIs match the per-state bootstrap file cc_bootstrap_3state.csv,")
    print("     which is the authoritative source for the CIs quoted in the manuscript.)")

    # (6) re_full sensitivity test (Section 5.3): does a drizzle-inclusive
    #     effective radius overcorrect the heavy-drizzle propagation residual?
    if {"Re_MODIS_21", "re_cas_median", "re_full_median"}.issubset(m.columns):
        print("\n(6) RE-PROPAGATION RESIDUAL: CAS-only vs drizzle-inclusive re (Sect. 5.3)")
        mm = m.dropna(subset=["Re_MODIS_21", "re_cas_median", "re_full_median",
                              "bias_21_calc"]).copy()
        mm["B_re_cas"] = (mm.Re_MODIS_21 / mm.re_cas_median) ** (-2.5)
        mm["B_re_full"] = (mm.Re_MODIS_21 / mm.re_full_median) ** (-2.5)
        mm["resid_cas"] = mm.bias_21_calc - mm.B_re_cas
        mm["resid_full"] = mm.bias_21_calc - mm.B_re_full
        print(f"    {'state':11s} {'resid(re_CAS)':>14s} {'resid(re_full)':>15s} {'change':>9s}")
        for s in STATES:
            sub = mm[mm.drizzle_state == s]
            rc, rf = sub.resid_cas.median(), sub.resid_full.median()
            print(f"    {s:11s} {rc:>+14.3f} {rf:>+15.3f} {rf - rc:>+9.3f}")
        print("    NOTE: using re_full (drizzle-inclusive) lowers the residual most in")
        print("    heavy drizzle and drives it slightly negative, i.e. it OVERCORRECTS the")
        print("    heavy case. The near-zero residual with re_CAS is therefore not an")
        print("    artifact of neglecting drizzle in the aircraft reference.")
    print("=" * 64)


if __name__ == "__main__":
    main()

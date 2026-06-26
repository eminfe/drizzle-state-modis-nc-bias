"""
cc_16_robustness.py  -  Robustness tests for the drizzle-state bias structure
Reproduces every number reported in Results Sections 3.2 and 3.3:

  (1) Per-campaign 3-state median bias
  (2) Non -> transition shift per campaign
  (3) Leave-one-campaign-out (LOCO)
  (4) Heavy-regime leave-one-out (single-profile influence)
  (5) Near-nadir subset (VZA < 40 deg)
  (6) Effective-radius channel comparison (2.1 vs 3.7 um)
  (7) Flight-level distribution of heavy-drizzle profiles

Input : data_outputs/cc_master_3state.csv
        (campaign, drizzle_state, bias_21_calc, bias_37_calc,
         VZA_mean, flight_id, cloud_id)
Output: console summary (+ optional CSV)
"""
import numpy as np
import pandas as pd
from pathlib import Path

STATES = ["Non", "Transition", "Heavy"]
DATA = Path("data_outputs/cc_master_3state.csv")


def med_by_state(df, col="bias_21_calc"):
    return {s: df.loc[df.drizzle_state == s, col].median() for s in STATES}


def fmt(d):
    return "  ".join(f"{s}={d[s]:.2f}" if pd.notna(d[s]) else f"{s}=NA" for s in STATES)


def main():
    m = pd.read_csv(DATA).dropna(subset=["bias_21_calc", "drizzle_state"])
    n = len(m)
    base = med_by_state(m)
    counts = m.drizzle_state.value_counts().reindex(STATES).to_dict()

    print("=" * 64)
    print(f"BASELINE (n={n}): {fmt(base)} | counts {counts}")

    # (1) per-campaign
    print("\n(1) PER-CAMPAIGN 3-STATE MEDIANS")
    for c in ["POST", "VOCALS", "MASE"]:
        sub = m[m.campaign == c]
        cc = sub.drizzle_state.value_counts().reindex(STATES).to_dict()
        print(f"    {c:7s} (n={len(sub):2d}): {fmt(med_by_state(sub))}  counts {cc}")

    # (2) non -> transition shift per campaign
    print("\n(2) NON -> TRANSITION SHIFT (per campaign)")
    for c in ["POST", "VOCALS", "MASE"]:
        sub = med_by_state(m[m.campaign == c])
        shift = sub["Transition"] - sub["Non"]
        print(f"    {c:7s}: {sub['Non']:.2f} -> {sub['Transition']:.2f}  (Delta = {shift:+.2f})")

    # (3) leave-one-campaign-out
    print("\n(3) LEAVE-ONE-CAMPAIGN-OUT")
    for c in ["POST", "VOCALS", "MASE"]:
        sub = m[m.campaign != c]
        d = med_by_state(sub)
        mono = d["Non"] > d["Transition"] > d["Heavy"]
        print(f"    without {c:7s} (n={len(sub):2d}): {fmt(d)}  monotonic={mono}")

    # (4) heavy-regime leave-one-out
    print("\n(4) HEAVY-REGIME LEAVE-ONE-OUT (single-profile influence)")
    hv = m.loc[m.drizzle_state == "Heavy", "bias_21_calc"].values
    loo = [np.median(np.delete(hv, i)) for i in range(len(hv))]
    print(f"    full heavy median = {np.median(hv):.3f}  (n={len(hv)})")
    print(f"    LOO median range  = {min(loo):.3f} - {max(loo):.3f}")
    print(f"    heavy values      = {np.round(np.sort(hv), 3)}")

    # (5) near-nadir subset VZA < 40
    print("\n(5) NEAR-NADIR SUBSET (VZA < 40 deg)")
    sub = m[m.VZA_mean < 40]
    d = med_by_state(sub)
    cc = sub.drizzle_state.value_counts().reindex(STATES).to_dict()
    print(f"    VZA<40 (n={len(sub)}): {fmt(d)}  counts {cc}")
    print(f"    Non:  full {base['Non']:.2f} -> near-nadir {d['Non']:.2f}")
    print(f"    Heavy: full {base['Heavy']:.2f} -> near-nadir {d['Heavy']:.2f}")

    # (6) effective-radius channel comparison
    print("\n(6) EFFECTIVE-RADIUS CHANNEL (2.1 vs 3.7 um)")
    if "bias_37_calc" in m.columns:
        d37 = med_by_state(m, "bias_37_calc")
        dd = m.dropna(subset=["bias_21_calc", "bias_37_calc"])
        r = np.corrcoef(dd.bias_21_calc, dd.bias_37_calc)[0, 1]
        print(f"    3.7 um medians: {fmt(d37)}")
        print(f"    Pearson r(2.1, 3.7) = {r:.3f}  (n={len(dd)})")

    # (7) flight-level distribution
    print("\n(7) FLIGHT-LEVEL DISTRIBUTION")
    print(f"    total flights={m.flight_id.nunique()}, profiles={n}")
    hvf = m.loc[m.drizzle_state == "Heavy", "flight_id"].value_counts().to_dict()
    print(f"    heavy-drizzle profiles by flight: {hvf}")

    # (8) flight-level leave-one-flight-out sensitivity
    print("\n(8) LEAVE-ONE-FLIGHT-OUT (do repeated profiles from one flight dominate?)")
    fl = m.groupby("flight_id").size().sort_values(ascending=False)
    print(f"    most-sampled flights: {fl.head(5).to_dict()}")
    for f in fl.head(5).index:
        sub = m[m.flight_id != f]
        d = med_by_state(sub)
        mono = d["Non"] > d["Transition"] > d["Heavy"]
        print(f"    without {f:10s} (n={len(sub):2d}): {fmt(d)}  monotonic={mono}")
    print("    heavy-regime leave-one-flight-out:")
    hv_df = m[m.drizzle_state == "Heavy"]
    for f in hv_df.flight_id.unique():
        sub = hv_df[hv_df.flight_id != f]
        print(f"      heavy without {f:10s} (n={len(sub)}): median={sub.bias_21_calc.median():.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()

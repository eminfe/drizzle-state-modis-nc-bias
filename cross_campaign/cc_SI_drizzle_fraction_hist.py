"""
Supporting Information figure (two panels):
(a) POST matched profile-level drizzle-fraction distribution — the campaign from
    which the Table 3 regime boundaries were selected (natural breaks).
(b) Pooled matched distribution (all three campaigns) — showing that the
    heavy-drizzle threshold remains separated from lower-drizzle profiles by an
    empirical gap (no profiles between f_drizzle ~ 0.12 and ~ 0.18).

Input : data_outputs/cc_master_matched.csv
        (columns: campaign, drizzle_fraction, drizzle_state)
Output: figures/cc_SI_drizzle_fraction_hist.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
STATE_COLORS = {"Non": "#3366cc", "Transition": "#ff8c00", "Heavy": "#cc3333"}

T_NW, T_WM, T_MH = 0.011, 0.045, 0.125
FLOOR = 0.0018

m = pd.read_csv("data_outputs/cc_master_matched.csv")
m = m.dropna(subset=["drizzle_fraction"])
# 4-class regime -> 3-state grouping (weak + moderate = Transition)
_map = {"non_drizzling": "Non", "weak_drizzling": "Transition",
        "moderate_drizzling": "Transition", "heavy_drizzling": "Heavy"}
m["drizzle_state"] = m["drizzle_regime"].map(_map)
post = m[m["campaign"] == "POST"]
subsets = [
    ("(a) POST matched profiles", post["drizzle_fraction"].values, post["drizzle_state"].values),
    ("(b) All campaigns (pooled matched)", m["drizzle_fraction"].values, m["drizzle_state"].values),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
xmax = m["drizzle_fraction"].max() * 1.2
xmin = FLOOR * 0.85
ymax = 13

for ax, (title, f, state) in zip(axes, subsets):
    n = len(f)
    fd = np.where(f <= 0, FLOOR, f)
    bins = np.logspace(np.log10(FLOOR), np.log10(xmax), 24)

    ax.axvspan(xmin, T_NW, color=STATE_COLORS["Non"],        alpha=0.07, zorder=0)
    ax.axvspan(T_NW, T_MH, color=STATE_COLORS["Transition"], alpha=0.07, zorder=0)
    ax.axvspan(T_MH, xmax, color=STATE_COLORS["Heavy"],      alpha=0.07, zorder=0)

    ax.hist(fd, bins=bins, color="#5a5a5a", edgecolor="white", linewidth=0.6, zorder=2)

    lo = f[f < T_MH].max() if (f < T_MH).any() else T_MH
    hi = f[f >= T_MH].min() if (f >= T_MH).any() else T_MH
    if hi > lo:
        ax.axvspan(lo, hi, facecolor="0.5", alpha=0.16, hatch="//",
                   edgecolor="0.45", linewidth=0.0, zorder=1)

    for x, lab in [(T_NW, "0.011"), (T_WM, "0.045"), (T_MH, "0.125")]:
        ax.axvline(x, color="k", lw=1.4, ls="--", zorder=3)
        ax.text(x, ymax * 0.97, f" {lab}", rotation=90, va="top", ha="left",
                fontsize=11.5, fontweight="bold")

    y_rug = -0.85
    for fi, st in zip(fd, state):
        ax.plot([fi, fi], [y_rug - 0.42, y_rug + 0.42], color=STATE_COLORS.get(st, "k"),
                lw=1.5, alpha=0.9, solid_capstyle="round", zorder=4)

    if hi > lo:
        ax.annotate("empirical gap", xy=(np.sqrt(lo * hi), ymax * 0.5), ha="center",
                    va="center", fontsize=11.5, fontweight="bold", color="0.25")

    nz = int((f <= 0).sum())
    if nz:
        ax.annotate(f"{nz} profiles\nat $f_{{drizzle}}=0$", xy=(FLOOR, ymax*0.985),
                    ha="center", va="top", fontsize=10.5, color=STATE_COLORS["Non"])

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.5, ymax)
    ax.set_xlabel(r"Profile drizzle fraction  $f_{drizzle}$  (log scale)", fontsize=15)
    ax.set_title(f"{title}   (n = {n})", fontsize=15, fontweight="bold", loc="left")
    ax.tick_params(axis="both", labelsize=12)

axes[0].set_ylabel("Number of profiles", fontsize=15)
handles = [Line2D([0], [0], color=STATE_COLORS[s], lw=3,
                  label={"Non": "Non-drizzling", "Transition": "Transition-drizzle",
                         "Heavy": "Heavy-drizzle"}[s]) for s in ["Non", "Transition", "Heavy"]]
axes[1].legend(handles=handles, loc="upper right", fontsize=11.5, frameon=False)

fig.tight_layout()
out = "figures/cc_SI_drizzle_fraction_hist.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("[SAVE]", out)

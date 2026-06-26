"""
cc_17_robustness_figure.py  -  Robustness figure for Section 3.2 / 3.3

(a) Per-campaign median-bias trajectories across the three drizzle states,
    with the pooled result overlaid. Shows that the non-drizzling -> transition
    decrease occurs in every campaign, the full monotonic decrease is clearest
    in POST, and MASE does not extend monotonically into the (single-profile)
    heavy regime.

(b) Leave-one-campaign-out medians per state, showing that removing VOCALS-REx
    or MASE preserves the monotonic decrease, whereas removing POST weakens it.

Input : data_outputs/cc_master_3state.csv
Output: figures/cc_17_robustness.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
STATE_COLORS = {"Non": "#3366cc", "Transition": "#ff8c00", "Heavy": "#cc3333"}
STATES = ["Non", "Transition", "Heavy"]
XLAB = ["Non-\ndrizzling", "Transition-\ndrizzle", "Heavy-\ndrizzle"]

m = pd.read_csv("data_outputs/cc_master_3state.csv").dropna(subset=["bias_21_calc", "drizzle_state"])

def med(df):
    return [df.loc[df.drizzle_state == s, "bias_21_calc"].median() for s in STATES]
def cnt(df):
    return [int((df.drizzle_state == s).sum()) for s in STATES]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
x = np.arange(3)

# ---------------- Panel (a): per-campaign trajectories ----------------
ax = axes[0]
ax.axhline(1.0, color="0.6", lw=1.2, ls=":", zorder=1)
camp_style = {"POST": ("#222222", "o", 30), "VOCALS": ("#1f78b4", "s", 12), "MASE": ("#33a02c", "^", 10)}
for c, (col, mk, _) in camp_style.items():
    sub = m[m.campaign == c]
    y = med(sub); nn = cnt(sub)
    ax.plot(x, y, color=col, marker=mk, ms=9, lw=2.0, label=f"{c} (n={len(sub)})", zorder=3)
# pooled (thick)
yp = med(m)
ax.plot(x, yp, color="#cc3333", marker="D", ms=11, lw=3.2, label=f"Pooled (n={len(m)})",
        zorder=4, alpha=0.9)
# regime color ticks
ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=12.5)
for tick, s in zip(ax.get_xticklabels(), STATES):
    tick.set_color(STATE_COLORS[s]); tick.set_fontweight("bold")
ax.set_ylabel(r"Median $B_{calc}$  ($N_{c}^{MODIS}/N_{c}^{CAS}$)", fontsize=14)
ax.set_title("(a) Per-campaign drizzle-state trajectories", fontsize=15, fontweight="bold", loc="left")
ax.legend(fontsize=11.5, frameon=False, loc="upper right")
ax.tick_params(labelsize=12)
ax.text(0.02, 1.0, "unbiased", transform=ax.get_yaxis_transform(), fontsize=10,
        color="0.45", va="bottom")

# ---------------- Panel (b): leave-one-campaign-out ----------------
ax = axes[1]
ax.axhline(1.0, color="0.6", lw=1.2, ls=":", zorder=1)
variants = [
    ("Pooled",        m,                       "#000000", "D", 12),
    ("\u2212 POST",   m[m.campaign != "POST"],   "#e31a1c", "o", 9),
    ("\u2212 VOCALS", m[m.campaign != "VOCALS"], "#6a3d9a", "s", 9),
    ("\u2212 MASE",   m[m.campaign != "MASE"],   "#ff7f00", "^", 9),
]
offsets = np.linspace(-0.22, 0.22, len(variants))
for (lab, df, col, mk, ms_), off in zip(variants, offsets):
    y = med(df)
    lw = 3.0 if lab == "Pooled" else 1.8
    ax.plot(x + off, y, color=col, marker=mk, ms=ms_, lw=lw,
            label=f"{lab} (n={len(df)})", zorder=4 if lab == "Pooled" else 3,
            alpha=0.95)
ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=12.5)
for tick, s in zip(ax.get_xticklabels(), STATES):
    tick.set_color(STATE_COLORS[s]); tick.set_fontweight("bold")
ax.set_ylabel(r"Median $B_{calc}$", fontsize=14)
ax.set_title("(b) Leave-one-campaign-out", fontsize=15, fontweight="bold", loc="left")
ax.legend(fontsize=11.5, frameon=False, loc="upper right")
ax.tick_params(labelsize=12)
# annotate the POST-removal weakening at heavy
yb = med(m[m.campaign != "POST"])
ax.annotate("removing POST\nweakens ordering", xy=(2 + offsets[1], yb[2]),
            xytext=(1.15, 1.18), fontsize=10.5, color="#e31a1c",
            arrowprops=dict(arrowstyle="->", color="#e31a1c", lw=1.3))

# shared y-limits
ylo = min(min(med(m)), 0.6) - 0.06
yhi = 1.75
for a in axes:
    a.set_ylim(ylo, yhi)

fig.tight_layout()
out = "figures/cc_17_robustness.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("[SAVE]", out)

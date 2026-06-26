"""
cc_15_independence.py  —  Figure for Results 3.3
Independent contributions of drizzle fraction and aircraft N_c to MODIS-aircraft N_c bias.

Panel (a): raw vs partial Spearman correlations (drizzle fraction, N_c)
Panel (b): standardized regression coefficients, outcome = standardized log(B_calc)

Reads the three campaign *_MODIS_Matches.csv files and writes
cc_fig8_independence.png to FIG_DIR.

Usage (in cross-campaign scripts folder):
    python cc_15_independence.py
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---------- paths (cc package config, else local fallback) ----------
try:
    from config import DATA_DIR, FIG_DIR
    MATCH_GLOB = str(DATA_DIR / "*" / "*_MODIS_Matches.csv")
    OUTPNG = FIG_DIR / "cc_fig8_independence.png"
except Exception:
    from pathlib import Path
    MATCH_GLOB = "**/*_MODIS_Matches.csv"
    OUTPNG = Path("cc_fig8_independence.png")

# ---------- load ----------
files = glob.glob(MATCH_GLOB, recursive=True)
dfs = []
for f in files:
    if any(c in f for c in ["POST", "MASE", "VOCALS"]):
        m = pd.read_csv(f)
        m = m[m["bias_21_calc"].notna()].drop_duplicates("cloud_id")
        dfs.append(m)
A = pd.concat(dfs, ignore_index=True)
d = A.dropna(subset=["bias_21_calc", "Nd_median", "drizzle_fraction"]).copy()
d["logbias"] = np.log10(d["bias_21_calc"])
d["logNd"]   = np.log10(d["Nd_median"])
N = len(d)

dz, nd, yb = d["drizzle_fraction"].values, d["logNd"].values, d["logbias"].values

# ---------- partial Spearman ----------
def partial_spearman(x, y, z):
    R = stats.rankdata
    Z = np.column_stack([np.ones(len(x)), R(z)])
    res = lambda v: R(v) - Z @ np.linalg.lstsq(Z, R(v), rcond=None)[0]
    return stats.pearsonr(res(x), res(y))

raw_dz, p_raw_dz = stats.spearmanr(dz, yb)
raw_nd, p_raw_nd = stats.spearmanr(nd, yb)
par_dz, p_par_dz = partial_spearman(dz, yb, nd)
par_nd, p_par_nd = partial_spearman(nd, yb, dz)

# ---------- standardized regression: z(log bias) ~ z(drizzle) + z(logNd) ----------
z = lambda v: (v - v.mean()) / v.std(ddof=1)
X = np.column_stack([np.ones(N), z(dz), z(nd)])
beta, *_ = np.linalg.lstsq(X, z(yb), rcond=None)
resid = z(yb) - X @ beta
s2 = resid @ resid / (N - 3)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
pval = 2 * (1 - stats.t.cdf(np.abs(beta / se), N - 3))
R2 = 1 - (resid @ resid) / np.sum((z(yb) - z(yb).mean()) ** 2)
ci = 1.96 * se

def star(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

# ---------------- FIGURE ----------------
plt.rcParams.update({"font.size": 13, "font.family": "sans-serif", "xtick.labelsize": 13, "ytick.labelsize": 13})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
C_DZ, C_ND = "#7B5AA6", "#2A9D8F"  # mor=drizzle, teal=Nc (rejim paletinden ayri)

# ----- Panel (a) -----
xpos = np.array([0, 1]); w = 0.34
raw_vals = [raw_dz, raw_nd]; par_vals = [par_dz, par_nd]
ax1.bar(xpos - w/2, raw_vals, w, color=[C_DZ, C_ND], alpha=0.40,
        edgecolor="k", linewidth=0.6, label="Raw")
ax1.bar(xpos + w/2, par_vals, w, color=[C_DZ, C_ND], alpha=1.0,
        edgecolor="k", linewidth=0.6, label="Partial")
ax1.axhline(0, color="k", lw=0.8)
for x, v, p in zip(xpos - w/2, raw_vals, [p_raw_dz, p_raw_nd]):
    ax1.annotate(f"{v:+.2f}\n{star(p)}", (x, v), ha="center", va="top",
                 xytext=(0, -3), textcoords="offset points", fontsize=11)
for x, v, p in zip(xpos + w/2, par_vals, [p_par_dz, p_par_nd]):
    ax1.annotate(f"{v:+.2f}\n{star(p)}", (x, v), ha="center", va="top",
                 xytext=(0, -3), textcoords="offset points", fontsize=11, fontweight="bold")
# strengthening arrow for drizzle (placed to the LEFT, clear of bars)
ax1.annotate("", xy=(-w/2 - 0.05, par_dz), xytext=(-w/2 - 0.05, raw_dz),
             arrowprops=dict(arrowstyle="-|>", color=C_DZ, lw=1.4, alpha=0.8))
ax1.set_xticks(xpos)
ax1.set_xticklabels(["Drizzle\nfraction", "Aircraft\n$N_{c,\\mathrm{CAS}}$"])
ax1.set_ylabel("Spearman correlation with $B_{calc}$", fontsize=15)
ax1.set_ylim(-0.72, 0.12); ax1.set_xlim(-0.78, 1.6)
ax1.set_title("(a) Raw vs. partial correlations", fontweight="bold", loc="left", fontsize=16)
ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10.5)
ax1.grid(axis="y", alpha=0.25)

# ----- Panel (b) -----
ypos = np.array([1, 0]); bvals = [beta[1], beta[2]]; bcis = [ci[1], ci[2]]
ax2.barh(ypos, bvals, xerr=bcis, color=[C_DZ, C_ND], alpha=0.9, edgecolor="k",
         linewidth=0.6, height=0.5, error_kw=dict(ecolor="k", capsize=4, lw=1.2))
ax2.axvline(0, color="k", lw=0.8)
for yp, bv, pv in zip(ypos, bvals, [pval[1], pval[2]]):
    ax2.annotate(f"  {bv:+.2f} {star(pv)}", (0, yp), va="center", ha="left",
                 fontsize=13, fontweight="bold")
ax2.set_yticks(ypos)
ax2.set_yticklabels(["Drizzle fraction", "$\\log\\,N_{c,\\mathrm{CAS}}$"])
ax2.set_xlabel("Standardized $\\beta$  (outcome: standardized $\\log B_{calc}$)", fontsize=15)
ax2.set_xlim(-0.98, 0.42); ax2.set_ylim(-0.6, 1.6)
ax2.set_title("(b) Standardized regression", fontweight="bold", loc="left", fontsize=16)
ax2.text(0.97, 0.06, f"$R^2$ = {R2:.2f}   (n = {N})", transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=12.5,
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
ax2.grid(axis="x", alpha=0.25)

# AGU: no in-figure title (moved to caption)
# fig.suptitle("Statistical separation of drizzle-fraction and aircraft-$N_c$ effects", ...)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print(f"n={N}")
print(f"(a) raw drizzle {raw_dz:+.3f}(p{p_raw_dz:.3f}) -> partial {par_dz:+.3f}(p{p_par_dz:.4f})")
print(f"(a) raw Nc      {raw_nd:+.3f}(p{p_raw_nd:.3f}) -> partial {par_nd:+.3f}(p{p_par_nd:.4f})")
print(f"(b) beta drizzle={beta[1]:+.3f}(p{pval[1]:.4f}) logNc={beta[2]:+.3f}(p{pval[2]:.4f}) R2={R2:.3f}")
print(f"[saved] {OUTPNG}")

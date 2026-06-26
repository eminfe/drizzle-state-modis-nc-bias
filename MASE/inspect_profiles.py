# =============================================================================
# inspect_profiles.py  -  visual sanity check for step03 golden profiles
# =============================================================================
# Plots Alt vs Time per flight, with golden profile time spans shaded.
# Each profile is annotated with its monotonicity score and duration.
#
# Usage:
#   cd MASE
#   python inspect_profiles.py
#
# Output: outputs/figures/MASE_step03_profile_inspection.png
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
df_qc    = pd.read_parquet(config.STEP02_QC_PARQUET)
df_super = pd.read_csv(config.STEP03_SUPER_PROFILES_CSV)
df_pts   = pd.read_parquet(config.STEP03_PROFILE_POINTS_PARQUET)

# Ensure datetime types
df_qc["Datetime"]    = pd.to_datetime(df_qc["Datetime"])
df_pts["Datetime"]   = pd.to_datetime(df_pts["Datetime"])
df_super["start_time"] = pd.to_datetime(df_super["start_time"])
df_super["end_time"]   = pd.to_datetime(df_super["end_time"])

golden = df_super[df_super["Selected"]].copy()
print(f"Loaded: {len(df_qc):,} QC rows, {len(golden)} golden profiles, "
      f"{df_qc['flight_id'].nunique()} flights")

# -----------------------------------------------------------------------------
# Compute monotonicity for each golden profile
# -----------------------------------------------------------------------------
def monotonicity(alt):
    if len(alt) < 2:
        return np.nan
    net_disp = abs(alt[-1] - alt[0])
    total    = np.sum(np.abs(np.diff(alt)))
    return net_disp / total if total > 0 else np.nan

mono_scores = []
for _, row in golden.iterrows():
    mask = ((df_pts["flight_id"] == row["flight_id"]) &
            (df_pts["Datetime"] >= row["start_time"]) &
            (df_pts["Datetime"] <= row["end_time"]))
    alt = df_pts.loc[mask].sort_values("Datetime")["Alt"].values
    mono_scores.append(monotonicity(alt))
golden["monotonicity"] = mono_scores

# -----------------------------------------------------------------------------
# Color mapping by monotonicity (red = wandering, green = clean profile)
# -----------------------------------------------------------------------------
def mono_color(m):
    if pd.isna(m):
        return "gray"
    if m < 0.3:
        return "#d62728"   # red    -> wandering, suspicious
    if m < 0.6:
        return "#ff7f0e"   # orange -> partial
    return "#2ca02c"       # green  -> clean monotonic

# -----------------------------------------------------------------------------
# Figure: one subplot per flight that has golden profiles
# -----------------------------------------------------------------------------
flights_with_golden = sorted(golden["flight_id"].unique())
n = len(flights_with_golden)
ncols = 2
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=False)
axes = np.atleast_1d(axes).flatten()

for ax, flt in zip(axes, flights_with_golden):
    sub_qc = df_qc[df_qc["flight_id"] == flt].sort_values("Datetime")

    # Background: full flight altitude track (gray)
    ax.plot(sub_qc["Datetime"], sub_qc["Alt"], color="lightgray", lw=0.6,
            label="_nolegend_", zorder=1)

    # In-cloud points (LWC > 0.05 AND Nc > 10) overlaid
    in_cloud = (sub_qc["LWC Gerber"] > 0.05) & (sub_qc["Nc_Total"] > 10)
    ax.scatter(sub_qc.loc[in_cloud, "Datetime"], sub_qc.loc[in_cloud, "Alt"],
               s=2, color="steelblue", alpha=0.4, label="in-cloud points",
               zorder=2)

    # Golden profile spans
    flt_golden = golden[golden["flight_id"] == flt].sort_values("start_time").reset_index(drop=True)
    for i, p in flt_golden.iterrows():
        color = mono_color(p["monotonicity"])
        ax.axvspan(p["start_time"], p["end_time"], color=color, alpha=0.20, zorder=0)

        # Annotation: duration + monotonicity
        mid_t = p["start_time"] + (p["end_time"] - p["start_time"]) / 2
        y_top = sub_qc["Alt"].max() * 0.95
        ax.annotate(
            f"P{i+1}\n{p['duration_s']:.0f}s\nm={p['monotonicity']:.2f}",
            xy=(mid_t, y_top),
            ha="center", va="top",
            fontsize=7,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, alpha=0.85),
        )

    ax.set_title(f"{flt}  ({len(flt_golden)} golden profile(s))",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

# Hide unused axes
for ax in axes[len(flights_with_golden):]:
    ax.set_visible(False)

# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(color="#2ca02c", alpha=0.4, label="Monotonicity >= 0.6 (clean)"),
    mpatches.Patch(color="#ff7f0e", alpha=0.4, label="Monotonicity 0.3-0.6 (partial)"),
    mpatches.Patch(color="#d62728", alpha=0.4, label="Monotonicity < 0.3 (wandering)"),
    plt.Line2D([0], [0], color="lightgray", lw=1.5, label="Full flight track"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=4, label="In-cloud points (LWC>0.05, Nc>10)"),
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.0 - 0.005),
           ncol=5, fontsize=9, frameon=True)

fig.suptitle(
    f"MASE 2005 - step03 Golden Profile Inspection  "
    f"({len(golden)} profiles across {n} flights)",
    fontsize=13, fontweight="bold", y=1.0,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
out_path = config.FIG_DIR / "MASE_step03_profile_inspection.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\n[OK] Saved: {out_path}")

# Show summary table
print(f"\n{'Flight':<8} {'#Prof':>5}  Profiles (sorted by start time):")
print("-" * 75)
for flt in flights_with_golden:
    sub = golden[golden["flight_id"] == flt].sort_values("start_time")
    parts = []
    for i, (_, p) in enumerate(sub.iterrows()):
        parts.append(f"P{i+1}(d={p['duration_s']:.0f}s, m={p['monotonicity']:.2f})")
    print(f"{flt:<8} {len(sub):>5}  " + ", ".join(parts))

plt.show()

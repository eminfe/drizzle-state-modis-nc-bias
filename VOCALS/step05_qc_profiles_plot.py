# =============================================================================
# step05_qc_profiles_plot.py  -  VOCALS-REx 2008
# =============================================================================
# Per-profile QC visualization. Plots all 40 golden profiles on a single
# multi-panel figure for visual inspection of:
#
#   - LWC vertical structure (adiabatic-like vs drizzle-flattened)
#   - Drizzle regime spatial pattern (heavy clusters on Nov 1)
#   - f_ad super-adiabatic profiles (which ones, why)
#   - Profile shape outliers
#
# OUTPUT:
#   outputs/figures/step05_qc/all_profiles_LWC.png
#   outputs/figures/step05_qc/all_profiles_re.png
#
# NOTE: This is a debug/inspection plot, not a paper figure. Paper-quality
# regime-aggregated figures are produced by step06.
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


# Color per drizzle regime
REGIME_COLORS = {
    "non_drizzling"      : "#4393c3",   # blue
    "weak_drizzling"     : "#92c5de",   # light blue
    "moderate_drizzling" : "#f4a582",   # orange
    "heavy_drizzling"    : "#d6604d",   # red
}
REGIME_SHORT = {
    "non_drizzling"      : "non",
    "weak_drizzling"     : "weak",
    "moderate_drizzling" : "mod",
    "heavy_drizzling"    : "HEAVY",
}


def plot_grid(gm, gc, x_col, x_label, title, out_path,
              n_cols=5, panel_size=(2.2, 2.0)):
    """
    Plot all profiles on one figure: x = metric, y = z_norm.
    Each panel is one profile, colored by drizzle regime.
    """
    profiles = gc.sort_values(["flight_id", "cloud_id"]).reset_index(drop=True)
    n = len(profiles)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} - {title}\n"
        f"All {n} golden profiles, colored by drizzle regime",
        fontsize=12, fontweight="bold", y=0.998,
    )

    for i, (_, prof) in enumerate(profiles.iterrows()):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        cid = prof["cloud_id"]
        seg = gm[gm["cloud_id"] == cid].sort_values("z_norm")
        if seg.empty or x_col not in seg.columns:
            ax.set_visible(False)
            continue

        regime = prof.get("drizzle_regime", "non_drizzling")
        color = REGIME_COLORS.get(regime, "gray")
        regime_short = REGIME_SHORT.get(regime, "?")

        # Drizzle points highlighted
        if "drizzle_flag" in seg.columns:
            non_dz = seg[~seg["drizzle_flag"].astype(bool)]
            dz     = seg[seg["drizzle_flag"].astype(bool)]
            ax.scatter(non_dz[x_col], non_dz["z_norm"],
                       s=4, color=color, alpha=0.5)
            if len(dz) > 0:
                ax.scatter(dz[x_col], dz["z_norm"],
                           s=8, color="black", alpha=0.6, marker="x")
        else:
            ax.scatter(seg[x_col], seg["z_norm"],
                       s=4, color=color, alpha=0.6)

        # Title: cloud_id + regime + key metric
        f_ad = prof.get("f_ad_mean", np.nan)
        f_str = f"f={f_ad:.2f}" if not pd.isna(f_ad) else "f=?"
        if not pd.isna(f_ad) and f_ad > 1.0:
            f_str = f"f={f_ad:.2f}*"   # asterisk marks super-adiabatic

        title_str = f"{cid}\n{regime_short}, {f_str}"
        ax.set_title(title_str, fontsize=8, fontweight="bold", color=color)

        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel("z_norm", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused panels
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].set_visible(False)

    # Legend on top
    legend_elements = [
        plt.scatter([], [], s=20, color=REGIME_COLORS["non_drizzling"], label="non"),
        plt.scatter([], [], s=20, color=REGIME_COLORS["weak_drizzling"], label="weak"),
        plt.scatter([], [], s=20, color=REGIME_COLORS["moderate_drizzling"], label="moderate"),
        plt.scatter([], [], s=20, color=REGIME_COLORS["heavy_drizzling"], label="HEAVY"),
        plt.scatter([], [], s=20, color="black", marker="x", label="drizzle point"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=5, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out_path.name}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step05 QC plotting - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}")
    print("=" * 70)

    if not config.STEP04_GOLDEN_CASE_CSV.exists() or not config.STEP04_GOLDEN_MICRO_CSV.exists():
        print(f"\n  [FAIL] Run step05 first.")
        sys.exit(1)

    gc = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    gm = pd.read_csv(config.STEP04_GOLDEN_MICRO_CSV)
    print(f"\n  [OK] gc: {gc.shape}  |  gm: {gm.shape}")

    # Output directory
    fig_dir = config.OUTPUT_DIR / "figures" / "step05_qc"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output dir: {fig_dir}")

    # Quick regime distribution recap
    print(f"\n  Regime distribution:")
    for regime, count in gc["drizzle_regime"].value_counts().items():
        print(f"    {REGIME_SHORT.get(regime, regime):<8} : {count:3d}")

    # Super-adiabatic profiles
    super_ad = gc[gc["f_ad_mean"] > 1.0]
    if len(super_ad) > 0:
        print(f"\n  Super-adiabatic profiles (f_ad > 1, marked with *):")
        for _, r in super_ad.iterrows():
            print(f"    {r['cloud_id']:<25}  f_ad={r['f_ad_mean']:.2f}  "
                  f"H={r['H_m']:.0f}m  regime={r.get('drizzle_regime', '?')}")

    # Plots
    print(f"\n  Plotting...")
    plot_grid(
        gm, gc,
        x_col=config.VAR_MAP["lwc"],
        x_label="LWC (g/m^3)",
        title="LWC vertical structure",
        out_path=fig_dir / f"{config.CAMPAIGN_NAME}_qc_LWC.png",
    )
    plot_grid(
        gm, gc,
        x_col="re_cas",
        x_label="re_cas (um)",
        title="Effective radius (CAS only)",
        out_path=fig_dir / f"{config.CAMPAIGN_NAME}_qc_re_cas.png",
    )
    plot_grid(
        gm, gc,
        x_col="re_full",
        x_label="re_full (um)",
        title="Effective radius (CAS+CIP)",
        out_path=fig_dir / f"{config.CAMPAIGN_NAME}_qc_re_full.png",
    )
    plot_grid(
        gm, gc,
        x_col="Nc_CAS",
        x_label="Nc_CAS (cm^-3)",
        title="Cloud droplet concentration",
        out_path=fig_dir / f"{config.CAMPAIGN_NAME}_qc_Nc.png",
    )

    print("\n" + "=" * 70)
    print(f"  step05 QC plotting COMPLETE - 4 figures saved")
    print(f"  Path: {fig_dir}")
    print("=" * 70)
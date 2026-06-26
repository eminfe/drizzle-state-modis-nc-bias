# =============================================================================
# step06_figures.py  -  MASE 2005
# =============================================================================
# Generate three diagnostic figures from the in-situ analysis:
#
#   Figure 1 - 2x3 boxplots: Nd, re_cas, re_full, tau_main, tau_diff, f_ad
#              by drizzle regime (profile-level, gc)
#
#   Figure 2 - 2x2 vertical profiles: LWC, Nc_CAS, re_cas, f_ad vs z_norm
#              by drizzle regime (point-level, gm)
#
#   Figure 3 - 2x2 scatters: Nd vs tau, re vs tau, f_ad vs Nd, re_cas vs re_full
#
# Used both for QC inspection of the pipeline output and as paper figures.
# Inputs: gc (golden_case), gm (golden_microphysics).
# =============================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


# =============================================================================
# Styling
# =============================================================================
REGIME_ORDER = [
    "non_drizzling",
    "weak_drizzling",
    "moderate_drizzling",
    "heavy_drizzling",
]
REGIME_COLORS = {
    "non_drizzling"      : "#4393c3",
    "weak_drizzling"     : "#92c5de",
    "moderate_drizzling" : "#f4a582",
    "heavy_drizzling"    : "#d6604d",
}
REGIME_LABELS = {
    "non_drizzling"      : "Non-drizzling",
    "weak_drizzling"     : "Weak",
    "moderate_drizzling" : "Moderate",
    "heavy_drizzling"    : "Heavy",
}


# =============================================================================
# Figure 1: Box plots (profile-level)
# =============================================================================
def plot_figure1(gc, out_path):
    print("  Figure 1 - Boxplots by drizzle regime...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} - In-Situ Cloud "
        f"Properties by Drizzle Regime\n(Profile-level, n={len(gc)})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    metrics = [
        ("Nd_median",      r"$N_d$ (cm$^{-3}$)",                     "(a) Cloud Droplet Concentration"),
        ("re_cas_median",  r"$r_{e,CAS}$ (um)",                      "(b) Effective Radius - CAS only"),
        ("re_full_median", r"$r_{e,full}$ (um)",                     "(c) Effective Radius - CAS+CIP"),
        ("tau_main",       r"$\tau_{main}$",                          "(d) Optical Thickness - Main"),
        ("tau_diff",       r"$\Delta\tau = \tau_{main}-\tau_{full}$", "(e) Drizzle Contribution to tau"),
        ("f_ad_mean",      r"$f_{ad}$",                               "(f) Adiabatic Fraction"),
    ]

    order   = [r for r in REGIME_ORDER if r in gc["drizzle_regime"].values]
    palette = {r: REGIME_COLORS[r] for r in order}

    for ax, (col, ylabel, title) in zip(axes.flatten(), metrics):
        if col not in gc.columns:
            ax.text(0.5, 0.5, f"{col}\nnot available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        sns.boxplot(
            data=gc, x="drizzle_regime", y=col,
            hue="drizzle_regime", order=order, palette=palette,
            legend=False, ax=ax, width=0.55, linewidth=1.2,
            flierprops=dict(marker="o", markersize=4, alpha=0.6),
        )

        ylim = ax.get_ylim()
        for i, regime in enumerate(order):
            n = (gc["drizzle_regime"] == regime).sum()
            ax.text(i, ylim[1] * 0.97, f"n={n}",
                    ha="center", va="top", fontsize=8.5, color="gray")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([REGIME_LABELS[r] for r in order], fontsize=9, rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == "f_ad_mean":
            ax.axhline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.7)
            ax.axhline(0.4, color="gray", linestyle=":",  lw=0.8, alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=config.FIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVE] {out_path.name}")


# =============================================================================
# Figure 2: Mean vertical profiles by regime
# =============================================================================
def plot_figure2(gm, gc, out_path):
    print("  Figure 2 - Mean vertical profiles by drizzle regime...")

    if "drizzle_regime" not in gm.columns:
        gm = gm.merge(gc[["cloud_id", "drizzle_regime"]], on="cloud_id", how="left")

    gm = gm.copy()
    z_bins    = np.linspace(0, 1, 16)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    gm["z_bin"] = pd.cut(
        gm["z_norm"], bins=z_bins, labels=z_centers, include_lowest=True
    ).astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    fig.suptitle(
        f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} - Mean Vertical "
        f"Profiles by Drizzle Regime",
        fontsize=14, fontweight="bold", y=1.01,
    )

    metrics = [
        (config.VAR_MAP["lwc"], r"LWC$_{Gerber}$ (g m$^{-3}$)", "(a) Liquid Water Content"),
        ("Nc_CAS",              r"$N_{c,CAS}$ (cm$^{-3}$)",     "(b) Cloud Droplet Concentration"),
        ("re_cas",              r"$r_{e,CAS}$ (um)",            "(c) Effective Radius - CAS"),
        ("f_ad",                r"$f_{ad}$",                     "(d) Adiabatic Fraction"),
    ]

    order = [r for r in REGIME_ORDER if r in gm["drizzle_regime"].values]

    for ax, (col, xlabel, title) in zip(axes.flatten(), metrics):
        if col not in gm.columns:
            ax.text(0.5, 0.5, f"{col}\nnot available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        for regime in order:
            sub = gm[gm["drizzle_regime"] == regime]
            if sub.empty:
                continue
            profile     = sub.groupby("z_bin")[col].mean().reindex(z_centers)
            profile_std = sub.groupby("z_bin")[col].std().reindex(z_centers)
            ax.plot(profile.values, profile.index,
                    color=REGIME_COLORS[regime], linewidth=2.5,
                    label=REGIME_LABELS[regime])
            ax.fill_betweenx(profile.index,
                             (profile - profile_std).values,
                             (profile + profile_std).values,
                             color=REGIME_COLORS[regime], alpha=0.12)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(r"$z_{norm}$ (0=base, 1=top)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        ax.axhline(0.5, color="gray", linestyle="--", lw=0.7, alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == "f_ad":
            ax.axvline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.6)
            ax.axvline(0.4, color="gray", linestyle=":",  lw=0.8, alpha=0.6)
            ax.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=config.FIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVE] {out_path.name}")


# =============================================================================
# Figure 3: Scatters
# =============================================================================
def plot_figure3(gc, out_path):
    print("  Figure 3 - Scatter relationships...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} - Microphysical "
        f"Relationships by Drizzle Regime",
        fontsize=14, fontweight="bold", y=1.01,
    )

    order = [r for r in REGIME_ORDER if r in gc["drizzle_regime"].values]
    sk = dict(s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    # (a) Nd vs tau_main
    ax = axes[0, 0]
    for regime in order:
        sub = gc[gc["drizzle_regime"] == regime]
        ax.scatter(sub["tau_main"], sub["Nd_median"],
                   color=REGIME_COLORS[regime], label=REGIME_LABELS[regime], **sk)
    ax.set_xlabel(r"$\tau_{main}$", fontsize=11)
    ax.set_ylabel(r"$N_d$ (cm$^{-3}$)", fontsize=11)
    ax.set_title(r"(a) $N_d$ vs Optical Thickness", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (b) re_cas vs tau_main
    ax = axes[0, 1]
    for regime in order:
        sub = gc[gc["drizzle_regime"] == regime]
        ax.scatter(sub["tau_main"], sub["re_cas_median"],
                   color=REGIME_COLORS[regime], label=REGIME_LABELS[regime], **sk)
    ax.set_xlabel(r"$\tau_{main}$", fontsize=11)
    ax.set_ylabel(r"$r_{e,CAS}$ (um)", fontsize=11)
    ax.set_title(r"(b) $r_{e,CAS}$ vs Optical Thickness", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (c) f_ad vs Nd
    ax = axes[1, 0]
    for regime in order:
        sub = gc[gc["drizzle_regime"] == regime]
        ax.scatter(sub["Nd_median"], sub["f_ad_mean"],
                   color=REGIME_COLORS[regime], label=REGIME_LABELS[regime], **sk)
    ax.set_xlabel(r"$N_d$ (cm$^{-3}$)", fontsize=11)
    ax.set_ylabel(r"$f_{ad}$", fontsize=11)
    ax.set_title(r"(c) Adiabatic Fraction vs $N_d$", fontsize=11, fontweight="bold")
    ax.axhline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.axhline(0.4, color="gray", linestyle=":",  lw=0.8, alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # (d) re_cas vs re_full + 1:1
    ax = axes[1, 1]
    for regime in order:
        sub = gc[gc["drizzle_regime"] == regime]
        ax.scatter(sub["re_cas_median"], sub["re_full_median"],
                   color=REGIME_COLORS[regime], label=REGIME_LABELS[regime], **sk)
    all_re = pd.concat([gc["re_cas_median"], gc["re_full_median"]]).dropna()
    if len(all_re) > 0:
        lims = [all_re.min() - 0.5, all_re.max() + 0.5]
        ax.plot(lims, lims, "k--", lw=1.2, alpha=0.5, label="1:1 line")
        ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"$r_{e,CAS}$ (um)", fontsize=11)
    ax.set_ylabel(r"$r_{e,full}$ (um)", fontsize=11)
    ax.set_title(r"(d) $r_{e,full}$ vs $r_{e,CAS}$ - Drizzle Sensitivity",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=config.FIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVE] {out_path.name}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"  step06 - {config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  (Figures)")
    print("=" * 70)

    if not config.STEP04_GOLDEN_CASE_CSV.exists() or not config.STEP04_GOLDEN_MICRO_CSV.exists():
        print(f"\n  [FAIL] Run step05 first.")
        sys.exit(1)

    gc = pd.read_csv(config.STEP04_GOLDEN_CASE_CSV)
    gm = pd.read_csv(config.STEP04_GOLDEN_MICRO_CSV)
    print(f"\n  [OK] gc:{gc.shape}  gm:{gm.shape}")

    # Compute tau_diff if missing (idempotent)
    if "tau_full" in gc.columns and "tau_main" in gc.columns and "tau_diff" not in gc.columns:
        gc["tau_diff"] = gc["tau_main"] - gc["tau_full"]

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = config.FIG_DIR
    ext = config.FIG["fmt"]
    name = config.CAMPAIGN_NAME

    print()
    plot_figure1(gc,        fig_dir / f"{name}_Figure1_Boxplots_Regime.{ext}")
    plot_figure2(gm, gc,    fig_dir / f"{name}_Figure2_VerticalProfiles.{ext}")
    plot_figure3(gc,        fig_dir / f"{name}_Figure3_Scatters.{ext}")

    print("\n" + "=" * 70)
    print(f"  step06 COMPLETE - 3 figures saved")
    print("=" * 70)
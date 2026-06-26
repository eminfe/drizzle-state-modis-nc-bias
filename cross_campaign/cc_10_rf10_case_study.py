"""
cc_10_rf10_case_study.py  v2

Final case-study figure for POST RF10 — improved layout.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===
warnings.filterwarnings('ignore')

CAS_D_MID = np.array([
    0.62, 0.67, 0.74, 0.81, 0.90, 1.00, 1.12, 1.20, 1.58, 2.07,
    2.84, 4.17, 7.40, 10.80, 13.70, 18.20, 23.70, 30.70, 40.20, 50.90
])
CIP_D_MID = np.array([
    15.45, 40.45, 64.04, 88.53, 113.28, 138.12, 163.02, 187.95, 212.89,
    237.85, 262.81, 287.78, 312.76, 337.74, 362.72, 387.71, 412.70, 437.68,
    462.67, 487.66, 512.66, 537.65, 562.64, 587.64, 612.63, 637.63, 662.62,
    687.62, 712.61, 737.61, 762.60, 787.60, 812.60, 837.59, 862.59, 887.59,
    912.59, 937.58, 962.58, 987.58, 1012.58, 1037.58, 1062.57, 1087.57,
    1112.57, 1137.57, 1162.57, 1187.57, 1212.57, 1237.56, 1262.56, 1287.56,
    1312.56, 1337.56, 1362.56, 1387.56, 1412.56, 1437.55, 1462.55, 1487.55,
    1512.55, 1537.55, 1562.55,
])
CIP_OVERLAP_MASK = CIP_D_MID > 50.0


gm = pd.read_csv(str(DATA_DIR / 'POST' / 'POST_golden_microphysics.csv'))
master = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')

CASES = [
    ('RF10_P14', 'Non',        '#3366cc'),
    ('RF10_P07', 'Transition', '#ff8c00'),
    ('RF10_P01', 'Heavy',      '#cc3333'),
]


def get_spectrum(pdata):
    cas_cols = [f'CAS_bin_{i:02d}' for i in range(20)]
    cas_means = np.array([pdata[c].mean() for c in cas_cols if c in pdata.columns])
    cip_cols = [f'CIP_bin_{i:02d}' for i in range(62)]
    cip_means = np.array([pdata[c].mean() for c in cip_cols if c in pdata.columns])
    cip_masked = cip_means.copy()
    cip_masked[~CIP_OVERLAP_MASK[:len(cip_means)]] = np.nan
    return CAS_D_MID, cas_means, CIP_D_MID[:len(cip_means)], cip_masked


# Pre-bin profiles
all_profs = {}
for pid, _, _ in CASES:
    sub = gm[gm['cloud_id'] == pid].sort_values('z_norm').copy()
    z_bins = np.linspace(0, 1, 21)
    sub['z_bin'] = pd.cut(sub['z_norm'], z_bins,
                          labels=(z_bins[:-1] + z_bins[1:]) / 2)
    all_profs[pid] = sub

LWC_MAX = 0.65
NC_MAX = 350
RE_MAX = 14

fig, axes = plt.subplots(3, 3, figsize=(14, 14))
plt.subplots_adjust(hspace=0.55, wspace=0.32,
                     top=0.83, bottom=0.06, left=0.07, right=0.96)

# ---------- ROW 1: LWC + Nc ----------
for col, (pid, state, color) in enumerate(CASES):
    ax = axes[0, col]
    sub = all_profs[pid]
    info = master[master['cloud_id'] == pid].iloc[0]

    lwc_b = sub.groupby('z_bin', observed=True)['LWC_total'].agg(
        ['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    lwc_b.columns = ['med', 'q25', 'q75']
    lwc_b = lwc_b.dropna()
    z_lwc = lwc_b.index.astype(float)
    nc_b = sub.groupby('z_bin', observed=True)['Nc_Total'].median().dropna()
    z_nc = nc_b.index.astype(float)

    ax.fill_betweenx(z_lwc, lwc_b['q25'], lwc_b['q75'],
                      color='lightblue', alpha=0.5)
    ax.plot(lwc_b['med'], z_lwc, color='steelblue', lw=2.5,
            marker='o', markersize=5, markerfacecolor='steelblue',
            markeredgecolor='black', markeredgewidth=0.5,
            label=r'LWC$_{total}$')
    ax.set_xlim(0, LWC_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'LWC$_{total}$ (g/m³)', fontsize=13, color='steelblue', labelpad=2)
    ax.tick_params(axis='x', labelcolor='steelblue', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.plot(nc_b.values, z_nc, color='darkgreen', lw=2,
             linestyle='--', marker='s', markersize=4,
             markerfacecolor='darkgreen', markeredgecolor='black',
             markeredgewidth=0.5, label=r'$N_c$')
    ax2.set_xlim(0, NC_MAX)
    ax2.set_xlabel(r'$N_c$ (cm$^{-3}$)', fontsize=13, color='darkgreen', labelpad=2)
    ax2.tick_params(axis='x', labelcolor='darkgreen', labelsize=11)
    if col == 0:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=13, frameon=False)

    pid_label = pid.replace('RF10_', '')
    duration = info['duration_s']
    panel_letter = chr(ord('a') + col)
    ax2.set_title(f'({panel_letter}) {state} — {pid_label}',
                   fontsize=15, fontweight='bold', color=color, pad=42, loc='left')

    if col == 0:
        ax.set_ylabel(r'$z_{norm}$ (0=base, 1=top)', fontsize=13)


# ---------- ROW 2: re + f_ad ----------
for col, (pid, state, color) in enumerate(CASES):
    ax = axes[1, col]
    sub = all_profs[pid]

    re_b = sub.groupby('z_bin', observed=True)['re_cas'].agg(
        ['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    re_b.columns = ['med', 'q25', 'q75']
    re_b = re_b.dropna()
    z_re = re_b.index.astype(float)
    refull_b = sub.groupby('z_bin', observed=True)['re_full'].median().dropna()
    z_refull = refull_b.index.astype(float)
    fad_b = sub.groupby('z_bin', observed=True)['f_ad'].median().dropna()
    z_fad = fad_b.index.astype(float)

    ax.fill_betweenx(z_re, re_b['q25'], re_b['q75'],
                      color='plum', alpha=0.4)
    ax.plot(re_b['med'], z_re, color='purple', lw=2.5,
            marker='o', markersize=5, markerfacecolor='purple',
            markeredgecolor='black', markeredgewidth=0.5,
            label=r'$r_{e,CAS}$')
    ax.plot(refull_b.values, z_refull, color='magenta', lw=1.5,
            linestyle=':', marker='^', markersize=4,
            markerfacecolor='magenta', markeredgecolor='black',
            markeredgewidth=0.5, label=r'$r_{e,full}$')

    ax.set_xlim(0, RE_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$r_e$ (µm)', fontsize=13, color='purple', labelpad=2)
    ax.tick_params(axis='x', labelcolor='purple', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(loc='upper right', fontsize=13, frameon=False)

    ax2 = ax.twiny()
    ax2.plot(fad_b.values, z_fad, color='saddlebrown', lw=2,
             linestyle='--', marker='s', markersize=4,
             markerfacecolor='saddlebrown', markeredgecolor='black',
             markeredgewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r'$f_{ad}$', fontsize=13, color='saddlebrown', labelpad=2)
    ax2.tick_params(axis='x', labelcolor='saddlebrown', labelsize=11)

    if col == 0:
        ax.set_ylabel(r'$z_{norm}$', fontsize=13)


# ---------- ROW 3: Size spectrum + metrics ----------
for col, (pid, state, color) in enumerate(CASES):
    ax = axes[2, col]
    sub = all_profs[pid]
    info = master[master['cloud_id'] == pid].iloc[0]

    cas_d, cas_n, cip_d, cip_n = get_spectrum(sub)

    ax.plot(cas_d, cas_n, color='steelblue', lw=2.2,
            marker='o', markersize=5, markerfacecolor='steelblue',
            markeredgecolor='black', markeredgewidth=0.5,
            label='CAS (cloud)', zorder=3)

    valid_cip = ~np.isnan(cip_n) & (cip_n > 0)
    if valid_cip.any():
        ax.plot(cip_d[valid_cip], cip_n[valid_cip], color='crimson',
                lw=2.2, marker='s', markersize=5,
                markerfacecolor='crimson', markeredgecolor='black',
                markeredgewidth=0.5, label='CIP (drizzle)', zorder=3)

    ax.axvspan(0.5, 50, alpha=0.06, color='gray', zorder=0)
    ax.axvline(50, color='black', linestyle='--', alpha=0.4, lw=1)
    ax.axvline(100, color='red', linestyle=':', alpha=0.5, lw=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.5, 2000)
    ax.set_ylim(1e-6, 100)
    ax.set_xlabel('Droplet diameter D (µm)', fontsize=13)
    ax.tick_params(labelsize=11)
    if col == 0:
        ax.set_ylabel(r'$N$ (cm$^{-3}$ bin$^{-1}$)', fontsize=13)
    if col == 0:
        ax.legend(loc='upper right', fontsize=13, frameon=False)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Profile-averaged size spectrum',
                  fontsize=13, color=color, fontweight='bold', pad=4)

    bias = info['bias_21_calc']
    if bias > 1.05:
        bias_dir = 'OVER'
        bias_color = '#cc4444'
    elif bias < 0.95:
        bias_dir = 'UNDER'
        bias_color = '#4444cc'
    else:
        bias_dir = 'near-unity'
        bias_color = '#666666'

    metrics_text = (
        f"$N_c$ = {info['Nd_median']:.0f} cm⁻³\n"
        f"$r_{{e,CAS}}$ = {info['re_cas_median']:.2f} µm\n"
        f"Δ$r_e$ = {info['re_full_median']-info['re_cas_median']:.2f} µm\n"
        f"drz frac = {info['drizzle_fraction']:.3f}\n"
        f"$f_{{ad}}$ = {info['f_ad_mean']:.2f}\n"
        f"H = {info['cloud_depth']:.0f} m"
    )
    ax.text(0.04, 0.04, metrics_text, transform=ax.transAxes,
             va='bottom', ha='left', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, lw=1.2, alpha=0.9))

    bias_text = f'MODIS bias = {bias:.2f}\n({bias_dir})'
    ax.text(0.5, 1.10, bias_text, transform=ax.transAxes,
             ha='center', va='bottom', fontsize=13, fontweight='bold',
             color=bias_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor=bias_color, lw=1.5))

# AGU: no in-figure title (moved to caption)
# fig.suptitle(...)

out = FIG_DIR / 'cc_rf10_case_study.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"[SAVE] {out.name}")

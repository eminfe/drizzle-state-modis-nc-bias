"""
cc_09_rf10_qc_diagnostic.py

Show vertical profiles for all 8 POST RF10 MATCHED profiles to QC-screen
candidates for case study selection.

Each profile gets a 4-panel column:
  Row 1: LWC vs z_norm
  Row 2: Nc_Total vs z_norm
  Row 3: re_cas vs z_norm
  Row 4: f_ad vs z_norm

8 columns x 4 rows = 32 panels total.

QC criteria:
  - LWC profile monotonic-like (no multi-layer artifact)
  - Nc profile follows LWC pattern
  - re profile increases with altitude (typical Sc)
  - f_ad reasonable (>0.2)
  - No sensor dropouts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===


# Load
gm = pd.read_csv(str(DATA_DIR / 'POST' / 'POST_golden_microphysics.csv'))
master = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')

RF10_IDS = ['RF10_P01', 'RF10_P04', 'RF10_P05', 'RF10_P07',
             'RF10_P08', 'RF10_P14', 'RF10_P16', 'RF10_P20']

# Per profile state
prof_state = master[master['cloud_id'].isin(RF10_IDS)].set_index('cloud_id')[
    ['drizzle_state', 'bias_21_calc', 're_cas_median',
      'drizzle_fraction', 'cloud_depth']]

STATE_COLORS = {'Non': '#3366cc', 'Transition': '#ff8c00', 'Heavy': '#cc3333'}

# Sort by state then by ID
def sort_key(pid):
    state = prof_state.loc[pid, 'drizzle_state']
    state_order = {'Non': 0, 'Transition': 1, 'Heavy': 2}
    return (state_order.get(state, 99), pid)

ids_sorted = sorted(RF10_IDS, key=sort_key)

# Figure: 4 rows x 8 cols
fig, axes = plt.subplots(4, 8, figsize=(24, 14), sharey=True)

# Variables to plot
panels = [
    ('LWC_total',  'LWC (g/m³)',           (0, 0.6),  'lightblue'),
    ('Nc_Total',   r'$N_c$ (cm$^{-3}$)',   (0, 400),  'tab:green'),
    ('re_cas',     r'$r_e$ CAS (µm)',       (0, 16),   'tab:purple'),
    ('f_ad',       r'$f_{ad}$',              (0, 1.0),  'tab:brown'),
]

for col_idx, pid in enumerate(ids_sorted):
    sub = gm[gm['cloud_id'] == pid].copy()
    sub = sub.sort_values('z_norm')
    if len(sub) == 0:
        continue

    state = prof_state.loc[pid, 'drizzle_state']
    bias = prof_state.loc[pid, 'bias_21_calc']
    re_med = prof_state.loc[pid, 're_cas_median']
    drz_f = prof_state.loc[pid, 'drizzle_fraction']
    depth = prof_state.loc[pid, 'cloud_depth']
    color = STATE_COLORS[state]

    # Top label
    pid_label = pid.replace('RF10_', '')
    title = (f"{pid_label}\n{state}\n"
              f"re={re_med:.1f}, drz={drz_f:.3f}\n"
              f"bias={bias:.2f}, H={depth:.0f}m")

    for row_idx, (var, xlabel, xlim, fillcolor) in enumerate(panels):
        ax = axes[row_idx, col_idx]
        if var not in sub.columns:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        d = sub[[var, 'z_norm']].dropna()
        if len(d) == 0:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        # Bin by z_norm
        z_bins = np.linspace(0, 1, 21)
        d['z_bin'] = pd.cut(d['z_norm'], z_bins,
                              labels=(z_bins[:-1]+z_bins[1:])/2)
        binned = d.groupby('z_bin', observed=True)[var].agg(
            ['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
        binned.columns = ['med', 'q25', 'q75']
        binned = binned.dropna()
        if len(binned) == 0:
            continue
        z_centers = binned.index.astype(float)
        # Fill IQR
        ax.fill_betweenx(z_centers, binned['q25'], binned['q75'],
                          color=fillcolor, alpha=0.3)
        # Median line
        ax.plot(binned['med'], z_centers, color=color, lw=2.5, marker='o',
                 markersize=4, markerfacecolor=color, markeredgecolor='black',
                 markeredgewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if row_idx == 0:
            ax.set_title(title, fontsize=9, fontweight='bold', color=color,
                         loc='center')
        if col_idx == 0:
            ax.set_ylabel(f'z_norm\n({xlabel})', fontsize=10)
        if row_idx == 3:
            ax.set_xlabel(xlabel, fontsize=10)

# Add row labels on left
for row_idx, (var, xlabel, _, _) in enumerate(panels):
    axes[row_idx, 0].set_ylabel(xlabel + '\nz_norm', fontsize=11, fontweight='bold')

fig.suptitle('POST RF10 — Vertical Profile QC Diagnostic (n=8 MATCHED profiles, all 3 states)\n'
              'Sorted left-to-right: Non (blue) → Transition (orange) → Heavy (red)',
              fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
out = FIG_DIR / 'cc_rf10_qc_vertical.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"[SAVE] {out.name}")

# Summary: which profile is QC-cleanest?
print("\n" + "=" * 80)
print("QC FLAGS PER PROFILE")
print("=" * 80)
for pid in ids_sorted:
    sub = gm[gm['cloud_id'] == pid].copy()
    state = prof_state.loc[pid, 'drizzle_state']
    n = len(sub)
    n_lwc_valid = sub['LWC_total'].notna().sum()
    n_re_valid = sub['re_cas'].notna().sum()
    pct_lwc = n_lwc_valid / n * 100
    pct_re = n_re_valid / n * 100

    # Multi-layer test: LWC drop above z=0.6?
    if 'z_norm' in sub.columns:
        upper = sub[sub['z_norm'] >= 0.7]
        lower = sub[(sub['z_norm'] >= 0.3) & (sub['z_norm'] <= 0.7)]
        if len(upper) > 0 and len(lower) > 0:
            lwc_upper = upper['LWC_total'].median()
            lwc_lower = lower['LWC_total'].median()
            lwc_ratio = lwc_upper / lwc_lower if lwc_lower > 0 else np.nan
            multi_layer = "MULTI-LAYER?" if lwc_ratio < 0.3 else "OK"
        else:
            multi_layer = "?"

    # Nc profile sanity
    n_nc_low = (sub['Nc_Total'] < 5).sum() / n if n > 0 else 0

    # Print
    flag = ""
    if pct_re < 70:
        flag += " ⚠️ low re coverage"
    if multi_layer == "MULTI-LAYER?":
        flag += " ⚠️ multi-layer"
    if n < 200:
        flag += f" ⚠️ short ({n} pts)"

    print(f"  {pid:12s} ({state:<12}): "
          f"n={n:>5}  re_cov={pct_re:>5.1f}%  multi={multi_layer:<13}{flag}")

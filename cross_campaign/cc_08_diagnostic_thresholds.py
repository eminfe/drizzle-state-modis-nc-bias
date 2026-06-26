"""
cc_08_diagnostic_thresholds.py

Visualize re_CAS, drizzle_fraction, and Δre = re_full - re_CAS distributions
to find natural break points for State A/B/C definition.

Output: cc_diag_thresholds.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR

CAMPAIGN_COLORS = {'POST': '#1f77b4', 'MASE': '#d62728', 'VOCALS': '#2ca02c'}

master = pd.read_csv(OUT_DIR / 'cc_master_3state.csv')
print(f"Loaded {len(master)} MATCHED profiles")

# Compute Δre = re_full - re_CAS
master['delta_re'] = master['re_full_median'] - master['re_cas_median']

# ---- Show distributions across campaigns ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# ===== ROW 1: Marginal distributions (KDE + tertile lines) =====
metrics_row1 = [
    ('re_cas_median',    r'$r_e$ CAS median (µm)',  False),
    ('drizzle_fraction', 'Drizzle fraction (log scale)', True),
    ('delta_re',         r'$\Delta r_e = r_{e,full} - r_{e,CAS}$ (µm)', False),
]

for ax, (col, xlabel, log_x) in zip(axes[0], metrics_row1):
    # All-campaign KDE
    all_vals = master[col].dropna().values
    if log_x:
        # Use log-space for drizzle_fraction
        vals_log = np.log10(all_vals + 1e-4)
        kde = stats.gaussian_kde(vals_log)
        x = np.linspace(vals_log.min(), vals_log.max(), 200)
        ax.fill_between(10**x, 0, kde(x), color='gray', alpha=0.25,
                        label='All (n=52)')
        ax.plot(10**x, kde(x), color='black', lw=2)
        ax.set_xscale('log')
    else:
        kde = stats.gaussian_kde(all_vals)
        x = np.linspace(all_vals.min(), all_vals.max(), 200)
        ax.fill_between(x, 0, kde(x), color='gray', alpha=0.25,
                        label='All (n=52)')
        ax.plot(x, kde(x), color='black', lw=2)

    # Per-campaign histograms underneath
    for camp in ['POST', 'MASE', 'VOCALS']:
        sub = master[master['campaign'] == camp][col].dropna().values
        if log_x:
            sub_plot = sub + 1e-4
        else:
            sub_plot = sub
        ax.scatter(sub_plot, np.full_like(sub_plot, -kde(x).max() * 0.05 *
                                         (1 + ['POST','MASE','VOCALS'].index(camp))),
                    color=CAMPAIGN_COLORS[camp], s=35, alpha=0.7,
                    edgecolors='black', linewidth=0.4,
                    label=f"{camp} (n={len(sub)})")

    # Tertile thresholds
    t33 = np.percentile(all_vals, 33)
    t67 = np.percentile(all_vals, 67)
    for t, lbl in [(t33, '33%'), (t67, '67%')]:
        ax.axvline(t, color='red', linestyle='--', lw=1.5, alpha=0.7)
        ax.text(t, kde(x).max() * 0.95, f'{lbl}\n{t:.2f}',
                ha='center', va='top', fontsize=9, color='red',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{col}\nmedian={np.median(all_vals):.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

# ===== ROW 2: 2D phase space colored by bias =====
# Panel 1: re_CAS vs drizzle_fraction
ax = axes[1, 0]
sc = ax.scatter(master['re_cas_median'],
                 master['drizzle_fraction'] + 1e-4,
                 c=np.log10(master['bias_21_calc']),
                 cmap='RdBu_r', s=80, alpha=0.85,
                 edgecolors='black', linewidth=0.6,
                 vmin=-0.4, vmax=0.4)
plt.colorbar(sc, ax=ax, label=r'log$_{10}$(bias$_{calc}$)')
# Tertile lines
re_t33 = np.percentile(master['re_cas_median'], 33)
re_t67 = np.percentile(master['re_cas_median'], 67)
drz_t33 = np.percentile(master['drizzle_fraction'], 33)
drz_t67 = np.percentile(master['drizzle_fraction'], 67)
ax.axvline(re_t33, color='red', linestyle='--', alpha=0.6)
ax.axvline(re_t67, color='red', linestyle='--', alpha=0.6)
ax.axhline(drz_t33 + 1e-4, color='blue', linestyle='--', alpha=0.6)
ax.axhline(drz_t67 + 1e-4, color='blue', linestyle='--', alpha=0.6)
ax.set_yscale('log')
ax.set_xlabel(r'$r_e$ CAS median (µm)', fontsize=11)
ax.set_ylabel('Drizzle fraction (+ 1e-4)', fontsize=11)
ax.set_title('(a) Phase Space: re_CAS vs drizzle_frac\n'
              'Color = bias direction', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Panel 2: re_CAS vs Δre
ax = axes[1, 1]
sc = ax.scatter(master['re_cas_median'],
                 master['delta_re'],
                 c=np.log10(master['bias_21_calc']),
                 cmap='RdBu_r', s=80, alpha=0.85,
                 edgecolors='black', linewidth=0.6,
                 vmin=-0.4, vmax=0.4)
plt.colorbar(sc, ax=ax, label=r'log$_{10}$(bias$_{calc}$)')
ax.axvline(re_t33, color='red', linestyle='--', alpha=0.6, label=f'33%: {re_t33:.1f}')
ax.axvline(re_t67, color='red', linestyle='--', alpha=0.6, label=f'67%: {re_t67:.1f}')
ax.axhline(0, color='black', linestyle=':', alpha=0.5)
dre_t33 = np.percentile(master['delta_re'], 33)
dre_t67 = np.percentile(master['delta_re'], 67)
ax.axhline(dre_t33, color='purple', linestyle='--', alpha=0.6,
           label=f'Δre 33%: {dre_t33:.2f}')
ax.axhline(dre_t67, color='purple', linestyle='--', alpha=0.6,
           label=f'Δre 67%: {dre_t67:.2f}')
ax.set_xlabel(r'$r_e$ CAS median (µm)', fontsize=11)
ax.set_ylabel(r'$\Delta r_e = r_{e,full} - r_{e,CAS}$ (µm)', fontsize=11)
ax.set_title('(b) Phase Space: re_CAS vs Δre\n'
             'Color = bias direction', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Δre vs drizzle_fraction
ax = axes[1, 2]
sc = ax.scatter(master['delta_re'],
                 master['drizzle_fraction'] + 1e-4,
                 c=np.log10(master['bias_21_calc']),
                 cmap='RdBu_r', s=80, alpha=0.85,
                 edgecolors='black', linewidth=0.6,
                 vmin=-0.4, vmax=0.4)
plt.colorbar(sc, ax=ax, label=r'log$_{10}$(bias$_{calc}$)')
ax.set_yscale('log')
ax.set_xlabel(r'$\Delta r_e$ (µm)', fontsize=11)
ax.set_ylabel('Drizzle fraction (+1e-4)', fontsize=11)
ax.set_title('(c) Δre vs drizzle_fraction\n'
             'Are they redundant?', fontsize=11, fontweight='bold')
# Spearman
from scipy.stats import spearmanr

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===
rho, p = spearmanr(master['delta_re'].dropna(),
                    master['drizzle_fraction'].dropna())
ax.text(0.05, 0.95, f'Spearman r = {rho:.2f}\np = {p:.3g}',
         transform=ax.transAxes, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.grid(True, alpha=0.3, which='both')

fig.suptitle('Diagnostic: Distributions for State A/B/C threshold definition',
              fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
out = FIG_DIR / 'cc_diag_thresholds.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[SAVE] {out.name}")

# ---- Print summary stats ----
print("\n" + "=" * 80)
print("THRESHOLD CANDIDATES")
print("=" * 80)

print("\n1. re_CAS (µm):")
for q in [25, 33, 50, 67, 75]:
    v = np.percentile(master['re_cas_median'], q)
    print(f"   {q:>3}th percentile: {v:.2f}")
print("   POST    median:", master[master['campaign']=='POST']['re_cas_median'].median())
print("   MASE    median:", master[master['campaign']=='MASE']['re_cas_median'].median())
print("   VOCALS  median:", master[master['campaign']=='VOCALS']['re_cas_median'].median())

print("\n2. drizzle_fraction:")
for q in [25, 33, 50, 67, 75]:
    v = np.percentile(master['drizzle_fraction'], q)
    print(f"   {q:>3}th percentile: {v:.4f}")

print("\n3. Δre = re_full - re_CAS (µm):")
for q in [25, 33, 50, 67, 75]:
    v = np.percentile(master['delta_re'], q)
    print(f"   {q:>3}th percentile: {v:.3f}")

print("\nCorrelation matrix (Spearman):")
metrics = ['re_cas_median', 'drizzle_fraction', 'delta_re', 'bias_21_calc']
sub = master[metrics].dropna()
for m1 in metrics:
    for m2 in metrics:
        if m1 < m2:
            r, p = spearmanr(sub[m1], sub[m2])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {m1:18s} vs {m2:18s}: r={r:+.3f}  p={p:.4f} {sig}")

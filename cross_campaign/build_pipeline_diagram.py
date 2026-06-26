"""
build_pipeline_diagram.py  v3 - clean rewrite with proper section spacing.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# === Path config ===
import sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from config import DATA_DIR, OUT_DIR, FIG_DIR, CAMPAIGNS, BOOTSTRAP_N_ITER, BOOTSTRAP_SEED
# === End path config ===

OUT = FIG_DIR

# Colors
C_INPUT, C_PROCESS = '#5b8def', '#7cb342'
C_QC, C_FORMULA, C_OUTPUT = '#e58e26', '#9c27b0', '#cc4444'

# =============================================================================
# Vertical layout sections (top → bottom)
# =============================================================================
# Section 1: Title + goal      (y range 175-190)
# Section 2: Two pipelines     (y range 50-170)  — ~120 units of pipeline space
# Section 3: Convergence       (y range 30-48)
# Section 4: Step 17 + 18      (y range 5-28)
# Section 5: KEY FINDINGS      (y range -10 to 0)
# Section 6: KEY FORMULAS      (y range -25 to -12)
# Section 7: Legend            (y range -32 to -28)

fig = plt.figure(figsize=(15, 32))
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(-20, 195)
ax.axis('off')

STEP_W, STEP_H = 38, 7.5
LEFT_X, RIGHT_X = 5, 57

# Pipeline section: top of step 1 = 168, going down by STEP_DY each step
PIPELINE_TOP = 168
STEP_DY = 9.8

# =============================================================================
# Helpers
# =============================================================================
def step_box(num, side, idx, title, body, color, body_size=8.5):
    x = LEFT_X if side == 'L' else RIGHT_X
    y = PIPELINE_TOP - idx * STEP_DY
    box = FancyBboxPatch((x, y), STEP_W, STEP_H,
                          boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='black',
                          linewidth=0.8, alpha=0.88, zorder=3)
    ax.add_patch(box)
    cx, cy = x + 2, y + STEP_H - 1.2
    ax.add_patch(mpatches.Circle((cx, cy), 1.0,
                                  facecolor='white', edgecolor='black',
                                  linewidth=1, zorder=4))
    ax.text(cx, cy, str(num), ha='center', va='center',
            fontsize=10, fontweight='bold', zorder=5)
    ax.text(x + 4.5, cy, title, ha='left', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=5)
    ax.text(x + 1, cy - 1.5, body, ha='left', va='top',
            fontsize=body_size, color='white', zorder=5)
    return x, y


def arrow_steps(side, idx_top, idx_bot, label=None):
    x = (LEFT_X if side == 'L' else RIGHT_X) + STEP_W / 2
    y_top = PIPELINE_TOP - idx_top * STEP_DY
    y_bot = PIPELINE_TOP - idx_bot * STEP_DY + STEP_H
    arr = FancyArrowPatch((x, y_top), (x, y_bot),
                           arrowstyle='->', mutation_scale=15,
                           color='#666', linewidth=1.5, zorder=2)
    ax.add_patch(arr)
    if label:
        y_mid = (y_top + y_bot) / 2
        w = 32
        box = FancyBboxPatch((x - w/2, y_mid - 0.7), w, 1.4,
                              boxstyle='round,pad=0.1',
                              facecolor='#fff3e0', edgecolor=C_QC,
                              linewidth=1, zorder=4)
        ax.add_patch(box)
        ax.text(x, y_mid, '! ' + label,
                ha='center', va='center', fontsize=8, color='#7d4c0a',
                fontweight='bold', zorder=5)


def output_box(side, idx, text_main, text_sub):
    x = LEFT_X if side == 'L' else RIGHT_X
    y = PIPELINE_TOP - idx * STEP_DY
    box = FancyBboxPatch((x, y + 0.5), STEP_W, STEP_H - 0.5,
                          boxstyle='round,pad=0.15',
                          facecolor='#ffe8e8', edgecolor=C_OUTPUT,
                          linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(x + STEP_W/2, y + STEP_H - 1.2, text_main,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=C_OUTPUT)
    ax.text(x + STEP_W/2, y + 2, text_sub,
            ha='center', va='center', fontsize=9, color='#444')

# =============================================================================
# 1. Title block
# =============================================================================
ax.text(50, 188, 'MODIS N$_d$ Bias Validation Pipeline',
        ha='center', va='center', fontsize=22, fontweight='bold')
ax.text(50, 184, 'POST 2008  ·  MASE 2005  ·  VOCALS-REx 2008',
        ha='center', va='center', fontsize=13, style='italic', color='#444')

goal = FancyBboxPatch((10, 178), 80, 4,
                       boxstyle='round,pad=0.3',
                       facecolor='#fff8dc', edgecolor='#888',
                       linewidth=1.2, zorder=2)
ax.add_patch(goal)
ax.text(50, 180,
         'GOAL: Quantify MODIS N$_d$ retrieval bias and identify '
         'cloud microphysical state-dependence',
         ha='center', va='center', fontsize=11, fontweight='bold')

# Column headers
ax.text(LEFT_X + STEP_W/2, 173, 'IN-SITU PIPELINE',
        ha='center', fontsize=13, fontweight='bold', color='#1a4d8f')
ax.text(RIGHT_X + STEP_W/2, 173, 'MODIS PIPELINE',
        ha='center', fontsize=13, fontweight='bold', color='#1a4d8f')

# =============================================================================
# 2. LEFT pipeline (steps 1-9)
# =============================================================================
step_box(1, 'L', 1, 'Raw flight data (NetCDF)',
          'CAS, CIP, FSSP, 260X, 2DC, 2DP\nT, P, LWC, position\nNative rate: 1 Hz / 10 s avg',
          C_INPUT)
arrow_steps('L', 1, 2, 'physical-range QC')

step_box(2, 'L', 2, 'QC + standardize',
          'Drop sensor artifacts\nUnify column names\nTime-align all probes',
          C_PROCESS)
arrow_steps('L', 2, 3)

step_box(3, 'L', 3, 'Cloud-core point ID',
          'LWC > LWC$_{thr}$ (8 thresholds tested)\nN$_c$ ≥ 5 cm⁻³\nFinal: 0.05 g/m³ (Wood 2012)',
          C_QC)
arrow_steps('L', 3, 4)

step_box(4, 'L', 4, 'Cloud segmentation',
          'Group consecutive cloud points\nMin duration: 10 s\nGap tolerance: ≤ 5 s',
          C_PROCESS)
arrow_steps('L', 4, 5)

step_box(5, 'L', 5, 'Build super-profiles',
          'Merge segments at similar location/altitude\nz_norm = (z − z_base)/(z_top − z_base)\nVertical profile per cloud_id',
          C_PROCESS)
arrow_steps('L', 5, 6)

step_box(6, 'L', 6, 'Compute microphysics',
          'r$_e$ from CAS (cloud) and CAS+CIP (full)\n'
          'r$_{e,full}$: CIP > 50 µm only (Wood 2012)\n'
          'LWC$_{ad}$(z) = c$_w$·(z − z$_{base}$)\n'
          'f$_{ad}$ = LWC$_{measured}$/LWC$_{ad}$',
          C_FORMULA, body_size=8)
arrow_steps('L', 6, 7)

step_box(7, 'L', 7, 'Drizzle classification',
          'drizzle_ratio = LWC$_{CIP}$ / LWC$_{total}$\n'
          'N_large = N(D > 100 µm)\n'
          'flag=1 if drz>0.10 AND N_large≥10/L\n'
          'Regimes: non/weak/moderate/heavy',
          C_FORMULA)
arrow_steps('L', 7, 8)

step_box(8, 'L', 8, 'In-situ N$_d$ (Grosvenor 2018)',
          r'$N_d = \frac{\sqrt{5}}{2\pi} k^{-1/2} (f_{ad} c_w \tau)^{1/2} r_e^{-2.5}$' +
          '\nTwo scenarios:\n'
          '• calc: in-situ k, f$_{ad}$, c$_w$\n'
          '• lit:  defaults k=0.67, f$_{ad}$=0.80',
          C_FORMULA, body_size=8)
arrow_steps('L', 8, 9, 'f$_{ad}$ ≤ 1.0, N$_d$ ≥ 5 cm⁻³')

step_box(9, 'L', 9, 'Profile integrity QC',
          'span_score: cloud-base to cloud-top coverage\nMin altitude span: 100 m\nduration ≤ 500 s → single-penetration',
          C_QC)

output_box('L', 10, '★ Per-profile in-situ N$_d$, r$_e$, drizzle state, f$_{ad}$',
           'POST: 88  ·  MASE: 12  ·  VOCALS: 30  candidate profiles')
arrow_steps('L', 9, 10)

# =============================================================================
# 3. RIGHT pipeline (steps 10-16)
# =============================================================================
step_box(10, 'R', 1, 'MODIS L2 download',
          'MOD06_L2 / MYD06_L2 (1 km)\nEarthdata LAADS DAAC\nVars: τ, r$_{e,2.1}$, r$_{e,3.7}$, LWP',
          C_INPUT)
arrow_steps('R', 1, 2)

step_box(11, 'R', 2, 'Spatial-temporal collocation',
          '±0.5° bbox around aircraft track\n±90 min time window\nAqua + Terra both used',
          C_PROCESS)
arrow_steps('R', 2, 3, 'phase=liquid, SZA<65°, VZA<60°')

step_box(12, 'R', 3, 'Pixel-level QC',
          'Drop ice/uncertain phase\nDrop high SZA/VZA\nApply MODIS internal QA flags',
          C_QC)
arrow_steps('R', 3, 4)

step_box(13, 'R', 4, 'Channel pairing',
          'Use only pixels valid in BOTH r$_e^{2.1}$ AND r$_e^{3.7}$\n(paired-pool QC; avoids pool mismatch)',
          C_QC)
arrow_steps('R', 4, 5)

step_box(14, 'R', 5, 'Compute MODIS N$_d$',
          'Same Grosvenor formula:\n'
          r'$N_d^{MODIS} = \frac{\sqrt{5}}{2\pi} k^{-1/2} (f_{ad} c_w \tau)^{1/2} r_e^{-2.5}$' +
          '\nLiterature defaults: k=0.67, f$_{ad}$=0.80',
          C_FORMULA, body_size=8)
arrow_steps('R', 5, 6)

step_box(15, 'R', 6, 'Pixel aggregation per profile',
          'Median of valid pixels in bbox\nTrack n_pixels_valid\nSeparately for r$_e^{2.1}$ and r$_e^{3.7}$',
          C_PROCESS)
arrow_steps('R', 6, 7, 'min n_pixels_valid ≥ 25')

step_box(16, 'R', 7, 'Match status assignment',
          'MATCHED / NO_VALID_PIXELS / NO_COVERAGE\nPer profile, both channels',
          C_QC)

output_box('R', 8, '★ Per-profile MODIS N$_d$, r$_e$, τ, LWP',
           'MATCHED only used for bias analysis')
arrow_steps('R', 7, 8)

# =============================================================================
# 4. CONVERGENCE: arrows from both columns to step 17
# =============================================================================
# Step 17 box position - leave clear gap from pipelines
y_step17 = 32  # absolute y position

# Arrows from output boxes converging
arr_l = FancyArrowPatch(
    (LEFT_X + STEP_W/2, PIPELINE_TOP - 10 * STEP_DY),
    (40, y_step17 + STEP_H),
    arrowstyle='->', mutation_scale=18,
    color='#444', linewidth=2, zorder=2,
    connectionstyle='arc3,rad=-0.1')
ax.add_patch(arr_l)

arr_r = FancyArrowPatch(
    (RIGHT_X + STEP_W/2, PIPELINE_TOP - 9 * STEP_DY),
    (60, y_step17 + STEP_H),
    arrowstyle='->', mutation_scale=18,
    color='#444', linewidth=2, zorder=2,
    connectionstyle='arc3,rad=0.1')
ax.add_patch(arr_r)

# Step 17 - bias computation (full width)
box17 = FancyBboxPatch((10, y_step17), 85, STEP_H + 1,
                        boxstyle='round,pad=0.15',
                        facecolor=C_FORMULA, edgecolor='black',
                        linewidth=0.8, alpha=0.88, zorder=3)
ax.add_patch(box17)
ax.add_patch(mpatches.Circle((12, y_step17 + STEP_H - 0.2), 1.0,
                              facecolor='white', edgecolor='black',
                              linewidth=1, zorder=4))
ax.text(12, y_step17 + STEP_H - 0.2, '17', ha='center', va='center',
        fontsize=10, fontweight='bold', zorder=5)
ax.text(15, y_step17 + STEP_H - 0.2, 'Compute bias (per matched profile)',
        ha='left', va='center', fontsize=11, fontweight='bold',
        color='white', zorder=5)
ax.text(11, y_step17 + STEP_H - 1.8,
         'bias$_{calc}$ = $N_d^{MODIS}$ / $N_d^{insitu,calc}$  '
         '(uses in-situ k, f$_{ad}$, c$_w$)\n'
         'bias$_{lit}$  = $N_d^{MODIS}$ / $N_d^{insitu,lit}$   '
         '(uses literature defaults)\n'
         'inflation = bias$_{lit}$ / bias$_{calc}$        '
         'Computed for both r$_e^{2.1}$ and r$_e^{3.7}$',
         ha='left', va='top', fontsize=8.5, color='white', zorder=5)

# Arrow 17 → 18
y_step18 = 19
arr_btw = FancyArrowPatch((50, y_step17), (50, y_step18 + STEP_H + 1),
                           arrowstyle='->', mutation_scale=15,
                           color='#666', linewidth=1.5, zorder=2)
ax.add_patch(arr_btw)

# Step 18
box18 = FancyBboxPatch((10, y_step18), 85, STEP_H + 1,
                        boxstyle='round,pad=0.15',
                        facecolor=C_PROCESS, edgecolor='black',
                        linewidth=0.8, alpha=0.88, zorder=3)
ax.add_patch(box18)
ax.add_patch(mpatches.Circle((12, y_step18 + STEP_H - 0.2), 1.0,
                              facecolor='white', edgecolor='black',
                              linewidth=1, zorder=4))
ax.text(12, y_step18 + STEP_H - 0.2, '18', ha='center', va='center',
        fontsize=10, fontweight='bold', zorder=5)
ax.text(15, y_step18 + STEP_H - 0.2,
        'State classification + cross-campaign statistics',
        ha='left', va='center', fontsize=11, fontweight='bold',
        color='white', zorder=5)
ax.text(11, y_step18 + STEP_H - 1.8,
         'State assignment: drizzle_regime → Non / Transition / Heavy\n'
         'Bootstrap 95% CI (10,000 iterations) per state\n'
         'Spearman correlations: bias × (re, depth, drz_frac, CTT, ...)\n'
         'Re-error propagation: predicted = (R$_e^{MODIS}$/r$_e^{insitu}$)$^{-2.5}$',
         ha='left', va='top', fontsize=8.5, color='white', zorder=5)

# =============================================================================
# 5. KEY FINDINGS box
# =============================================================================
y_find = 5.5
arr_find = FancyArrowPatch((50, y_step18), (50, y_find + 8),
                            arrowstyle='->', mutation_scale=15,
                            color='#666', linewidth=2, zorder=2)
ax.add_patch(arr_find)

box_find = FancyBboxPatch((5, y_find), 90, 7,
                           boxstyle='round,pad=0.2',
                           facecolor='#ffd1d1', edgecolor=C_OUTPUT,
                           linewidth=2.5, zorder=3)
ax.add_patch(box_find)
ax.text(50, y_find + 5.5, '★  KEY FINDINGS',
         ha='center', va='center', fontsize=13, fontweight='bold',
         color=C_OUTPUT)
ax.text(50, y_find + 3.3,
         'Non state  bias = 1.18 [0.97, 1.61]      ·      '
         'Transition  0.90 [0.77, 1.42]      ·      '
         'Heavy  0.74 [0.47, 0.89]',
         ha='center', va='center', fontsize=10.5)
ax.text(50, y_find + 1.3,
         'Constant Re overestimate ≈ +2.7 µm    ·    '
         'Heavy state Re-propagation gap p=0.46 ns    ·    '
         'Inflation factor 1.54×',
         ha='center', va='center', fontsize=10, style='italic')

# =============================================================================
# 6. KEY FORMULAS REFERENCE box (bottom, separate from KEY FINDINGS)
# =============================================================================
y_form = -10
form = FancyBboxPatch((2, y_form), 96, 12,
                       boxstyle='round,pad=0.3',
                       facecolor='#f0f4ff', edgecolor='#1a4d8f',
                       linewidth=1.2, zorder=2)
ax.add_patch(form)
ax.text(50, y_form + 11, 'KEY FORMULAS REFERENCE',
        ha='center', va='top', fontsize=11, fontweight='bold',
        color='#1a4d8f')

# 4 formulas in a 2x2 layout
ax.text(5, y_form + 7.5,
         r'$N_d = \frac{\sqrt{5}}{2\pi} \cdot k^{-1/2} \cdot '
         r'(f_{ad} \cdot c_w \cdot \tau)^{1/2} \cdot r_e^{-5/2}$',
         ha='left', va='center', fontsize=12)
ax.text(5, y_form + 5.5,
         '(Grosvenor et al. 2018: droplet number from optical depth + effective radius)',
         ha='left', va='center', fontsize=8, style='italic', color='#555')

ax.text(55, y_form + 7.5,
         r'$f_{ad} = \dfrac{LWC_{measured}}{LWC_{adiabatic}}$,    '
         r'$LWC_{ad}(z) = c_w (z - z_{base})$',
         ha='left', va='center', fontsize=12)
ax.text(55, y_form + 5.5,
         '(Painemal & Zuidema 2011: sub-adiabaticity factor)',
         ha='left', va='center', fontsize=8, style='italic', color='#555')

ax.text(5, y_form + 2.5,
         r'drizzle_flag = 1 if  $\dfrac{LWC_{CIP}}{LWC_{total}} > 0.10$  AND  '
         r'$N(D>100\mu m) \geq 10\,/L$',
         ha='left', va='center', fontsize=10.5)

ax.text(55, y_form + 2.5,
         r'bias = $N_d^{MODIS} / N_d^{insitu}$,     '
         r'inflation = bias$_{lit}$ / bias$_{calc}$',
         ha='left', va='center', fontsize=10.5)

# =============================================================================
# 7. Legend (bottom-most)
# =============================================================================
legends = [
    (C_INPUT, 'Data input'),
    (C_PROCESS, 'Processing'),
    (C_QC, 'QC / filter'),
    (C_FORMULA, 'Formula'),
    (C_OUTPUT, 'Key output'),
]
y_leg = -16
x_start = 5
for i, (c, lbl) in enumerate(legends):
    x = x_start + i * 18
    ax.add_patch(mpatches.Rectangle((x, y_leg), 1.5, 1.5,
                                     facecolor=c, edgecolor='black',
                                     linewidth=0.6))
    ax.text(x + 2.2, y_leg + 0.7, lbl, ha='left', va='center',
            fontsize=9)

# Save
out_svg = OUT / 'pipeline_flow_chart.svg'
fig.savefig(out_svg, format='svg', bbox_inches='tight')
out_png = OUT / 'pipeline_flow_chart.png'
fig.savefig(out_png, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"[SAVE] {out_svg.name}")
print(f"[SAVE] {out_png.name}")

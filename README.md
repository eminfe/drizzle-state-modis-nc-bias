# Drizzle-State Dependence of MODIS Cloud Droplet Number Retrieval Bias in Marine Stratocumulus

Processing pipeline and analysis code for evaluating whether the bias between
MODIS-derived cloud droplet number concentration (*N*c) and an aircraft
cloud-droplet reference is organized by drizzle state, using three marine
stratocumulus field campaigns flown by the CIRPAS Twin Otter: POST, MASE, and
VOCALS-REx.

This repository accompanies the manuscript:

> Emin, F., & Hudson, J. G. (2026). *Drizzle-State Dependence of MODIS Cloud
> Droplet Number Retrieval Bias in Marine Stratocumulus.* Journal of Geophysical
> Research: Atmospheres (submitted).

---

## Overview

The analysis proceeds in two stages:

1. **Per-campaign pipelines** (`POST/`, `MASE/`, `VOCALS/`) process the raw
   aircraft data and MODIS granules for each campaign independently, producing a
   harmonized per-profile "golden case" table and a MODIS-match table.
2. **Cross-campaign analysis** (`cross_campaign/`) concatenates the three
   campaign outputs into a single matched dataset and runs all statistics,
   robustness tests, and figures reported in the manuscript.

The aircraft *N*c reference is defined from CAS cloud droplets only; drizzle
state is classified independently from CIP measurements; and MODIS *N*c is
recomputed for each profile using its aircraft-derived spectral-shape parameter
(*k*), adiabatic fraction (*f*ad), and adiabatic liquid-water lapse rate (*c*w),
rather than fixed retrieval assumptions.

---

## Repository structure

```
.
├── POST/                     # Per-campaign pipeline (step01–step18)
├── MASE/                     # Per-campaign pipeline (step01–step18)
├── VOCALS/                   # Per-campaign pipeline (step01–step18)
├── cross_campaign/           # Cross-campaign analysis (cc_01–cc_18)
│   ├── config.py             # Single-point path and parameter configuration
│   ├── run_all.py            # Sequential runner for the full cross-campaign analysis
│   ├── cc_01_build_master.py … cc_18_primary_stats.py
│   └── data_outputs/         # Derived datasets (see below)
└── README.md
```

### Per-campaign pipeline steps (step01–step18)

Each campaign pipeline follows the same 18-step sequence:

| Steps | Purpose |
|-------|---------|
| step01–step02 | Load raw aircraft data; quality-control filtering |
| step03 | Construct vertical cloud profiles (penetrations) |
| step04 | Drizzle classification from CIP large-drop spectra |
| step05 | Microphysics (CAS *N*c, effective radius, *f*ad, *k*, *c*w) |
| step06–step07 | Per-campaign figures and final consistency checks |
| step08–step09 | MODIS granule download and aircraft–MODIS colocation |
| step10 | MODIS-derived *N*c calculation |
| step11–step18 | Output "packages" (matched per-profile tables) |

Each campaign pipeline writes `{CAMPAIGN}_golden_case.csv` and
`{CAMPAIGN}_MODIS_Matches.csv`, which are the inputs to the cross-campaign stage.

### Cross-campaign analysis steps (cc_01–cc_18)

| Script | Purpose |
|--------|---------|
| `cc_01_build_master.py` | Concatenate the three campaigns → 130 candidate profiles, 52 matched |
| `cc_02_bootstrap_ci.py` | Bootstrap confidence intervals for pooled statistics |
| `cc_05_3state_master.py` | Build the three-state drizzle classification (Non / Transition / Heavy) |
| `cc_06_3state_bootstrap.py` | Per-state bootstrap confidence intervals |
| `cc_07_3state_figures.py` | Three-state figures (forest, trajectory, signature, density) |
| `cc_08_diagnostic_thresholds.py` | Drizzle-fraction threshold diagnostics (Figure S1) |
| `cc_10_rf10_case_study.py` | POST RF10 single-flight case study (Figure 5) |
| `cc_11_mechanism_tests.py` | Effective-radius mechanism analysis |
| `cc_12_mechanism_synthesis.py` | Mechanism synthesis figure (Figure 4) |
| `cc_13_two_regime_test_v2.py` | Effective-radius propagation residual test |
| `cc_15_independence.py` | Partial-correlation / joint-regression independence analysis |
| `cc_16_robustness.py` | Robustness suite (per-campaign, leave-one-out, VZA, 3.7 µm, flight-level) — Tables S1–S2 |
| `cc_18_primary_stats.py` | Primary §3.1 statistics and the re,full sensitivity test — Table S3 |

Run the full cross-campaign analysis with:

```bash
cd cross_campaign
python run_all.py            # full pipeline
python run_all.py --start 5  # resume from step 5
python run_all.py --only 16  # run a single step
```

---

## Derived data products

The following derived tables (in `cross_campaign/data_outputs/`) are the single
source of truth for all numbers in the manuscript:

| File | Contents |
|------|----------|
| `cc_master_all.csv` | All 130 candidate profiles with match status |
| `cc_master_3state.csv` | 52 matched profiles with three-state classification, bias, microphysics |
| `cc_bootstrap_3state.csv` | Per-state bootstrap confidence intervals (authoritative CIs) |

These derived products are included so that the analysis and figures can be
reproduced without re-running the per-campaign MODIS download and colocation
steps.

---

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scipy`, `matplotlib`

Install with:

```bash
pip install numpy pandas scipy matplotlib
```

---

## Raw data availability

The raw aircraft and satellite data are **not** redistributed here; they are
available from their primary public archives:

- **POST** (CIRPAS Twin Otter flight-level data): NSF NCAR Earth Observing
  Laboratory (EOL), Jonsson (2009), https://doi.org/10.26023/3W1T-VT0T-JK0B
- **VOCALS-REx** (CIRPAS Twin Otter cloud-probe / state data): NSF NCAR/EOL,
  Albrecht (2011), https://doi.org/10.26023/RR7J-XCQ5-NJ05 (and EOL datasets
  89.157, 89.132)
- **MASE** (CIRPAS Twin Otter cloud / precipitation probe data): U.S. DOE ARM
  data archive, https://www.arm.gov/research/campaigns/osc2005mase
- **MODIS Level-2 cloud product** (Collection 6.1): NASA LAADS DAAC,
  Terra MOD06_L2 (https://doi.org/10.5067/MODIS/MOD06_L2.061) and
  Aqua MYD06_L2 (https://doi.org/10.5067/MODIS/MYD06_L2.061)

To reproduce the per-campaign stage from scratch, set the campaign data paths in
each pipeline's `config.py` to the locations of the downloaded raw data.

---

## Configuration

All paths and analysis parameters for the cross-campaign stage are set in
`cross_campaign/config.py`. Before running, set `PROJECT_ROOT` to your local
checkout and point `POST_OUTPUTS`, `MASE_OUTPUTS`, and `VOCALS_OUTPUTS` to the
per-campaign output folders.

---

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code or the derived datasets, please cite both the manuscript
above and this archived release:

> Emin, F. (2026). *Cross-campaign MODIS–aircraft Nc drizzle-state analysis
> pipeline* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20945708

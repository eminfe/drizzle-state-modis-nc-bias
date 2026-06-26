# POST 2008 Pipeline

MODIS Nd retrieval bias validation pipeline for the POST 2008 campaign.
Modular config-driven structure. Runs end-to-end with `run_pipeline.py`.

---

## Campaign

- **Region:** NE Pacific (off the California coast)
- **Year:** 2008 (July)
- **Aircraft:** CIRPAS Twin Otter (10 s averaged)
- **Data:** 10 s, single parquet file (`POST_10s_merged_v2.parquet`)
- **Flights:** 13 (RF01..RF13)
- **Coordinates:** lat/lon POSITIVE (lat ~ 36-38 deg, lon ~ -124 to -122 deg)
- **Date range:** 2008-07-05 to 2008-07-27

---

## Folder Layout

```
POSTV1/                             <- BASE_DIR
├── POST_10s_merged_v2.parquet      <- raw data (10 s, all flights, located one level up)
├── config.py
├── utils.py
├── run_pipeline.py
├── step01..step18 (Python files)
├── README.md
│
├── outputs/                        <- created automatically
│   ├── POST_golden_case.csv             (final, 12 profiles, ~64 cols)
│   ├── POST_golden_microphysics.csv     (final, ~936 rows of per-second data)
│   ├── POST_MODIS_Matches.csv           (final, with bias columns, ~101 cols)
│   ├── figures/
│   │   ├── POST_Figure1-3_*.png         (paper main figures)
│   │   ├── POST_PackageA-H_*.png        (analysis package figures)
│   │   └── stepXX_qc/                   (debug/QC plots)
│   └── intermediate/                    (intermediate parquet/csv files)
│
└── modis_data/                     <- MODIS HDF files (step08 downloads them here)
```

---

## POST vs VOCALS / POST Differences

| Parameter | POST | VOCALS | POST |
|---|---|---|---|
| Data time step | **10 sec** | 1 s | 1 s |
| Flights | 13 | 18 | 17 |
| Region | **NE Pacific (CA)** | SE Pacific (Chile) | NE Pacific (CA) |
| Lat/Lon sign | Positive | **Negative** | Positive |
| Data format | Single parquet | Single parquet | NetCDF/flight |
| Flight ID format | `RF01..RF13` | `081014a` (date-coded) | `RF01..RF17` |
| Datetime column | `Datetime` | `UTC` | `dt` |
| CAS bins | 20 (Bin 1 NaN) | 21 | 20 |
| CIP bins | 60 (Bin 1 NaN) | 63 | 63 |
| `final_min_depth_m` | 200 | 100 (relaxed) | 200 |
| `final_min_duration_s` | 120 | 60 (relaxed) | 120 |
| `min_span_score` | **0.4 (sawtooth-aware)** | 0.0 (n/a) | 0.0 (n/a) |
| `summary_gap_seconds` | 120 | 60 | 60 |
| `stitch_gap_seconds` | 300 | 180 | 180 |
| MODIS bbox padding | 0.5 deg | 0.10 deg (tight) | 0.5 |
| MODIS time window | **±120 min** | ±90 min | ±90 min |
| Drizzle `n_large_thresh` | **1 L⁻¹** (10 s averaging) | 10 L⁻¹ | 10 L⁻¹ |
| `integrity_check_enabled` | **False** (10 s) | True | True |

---

## How to Run

### Full pipeline (recommended)

```bash
cd POST
python run_pipeline.py
```

18 steps run in order:
- step01-07: in-situ pipeline (~3-5 min)
- step08-10: MODIS pipeline (~10-30 min, requires internet + Earthdata)
- step11-18: analysis packages (~2-3 min)

### Filter options

```bash
python run_pipeline.py --dry-run                    # just show the step list
python run_pipeline.py --skip-modis                 # skip MODIS steps
python run_pipeline.py --only-core                  # only step01-07
python run_pipeline.py --only-modis                 # only step08-10
python run_pipeline.py --only-packages              # only step11-18
python run_pipeline.py --from step03 --to step07    # specific range
python run_pipeline.py --steps step09               # single step
python run_pipeline.py --continue-on-error          # keep going on failure
```

### MODIS VZA strict mode

step09 default uses `relaxed` (VZA <= 60 deg). For comparison with stricter geometry:

```bash
python step09_modis_colocation.py strict
```

---

## Pipeline Steps

### Core (step01..step07) - in-situ processing

| Step | Script | Output |
|------|--------|--------|
| 01 | `step01_load_data.py`        | Clean parquet, GWIU computed |
| 02 | `step02_qc_filtering.py`     | QC mask, zero overlap+inactive bins, Nc_CAS, Nc_CIP, Nc_Total |
| 03 | `step03_vertical_profiles.py`| 12 golden profiles (4-section logic + span_score) |
| 04 | `step04_drizzle.py`          | Drizzle flag + regime + z_norm |
| 05 | `step05_microphysics.py`     | f_ad, Re, tau, k, c_w, LWP per profile + Re outlier filter |
| 06 | `step06_figures.py`          | 3 main paper figures |
| 07 | `step07_final_check.py`      | Final filter (12 -> 12 profiles for POST) |

### MODIS (step08..step10) - satellite retrieval

| Step | Script | Output |
|------|--------|--------|
| 08 | `step08_modis_download.py`     | 23 MODIS granules (12 Terra + 11 Aqua) |
| 09 | `step09_modis_colocation.py`   | 10 MATCHED profiles (83% match rate) |
| 10 | `step10_nd_calculation.py`     | Nd_MODIS via Grosvenor (2018), bias_calc + bias_lit |

### Packages (step11..step18) - analysis

| Step | Script | Question |
|------|--------|----------|
| 11 | `step11_packageA.py` | Do drizzle regimes alter cloud physics? |
| 12 | `step12_packageB.py` | Which profiles have valid MODIS retrievals? |
| 13 | `step13_packageC.py` | How well does MODIS retrieve Nd? |
| 14 | `step14_packageD.py` | What drives the MODIS Nd bias? |
| 15 | `step15_packageE.py` | Do 2.1 and 3.7 um channels agree? |
| 16 | `step16_packageF.py` | How sensitive is bias to k, f_ad, c_w assumptions? |
| 17 | `step17_packageG.py` | Is bias related to cloud altitude? |
| 18 | `step18_packageH.py` | Does clean/polluted classification matter? |

### Optional QC plots (debug)

| Script | Purpose |
|--------|---------|
| `step05_qc_profiles_plot.py` | Per-profile LWC / Re / Nc grids (12 panels) |
| `inspect_profiles.py`        | Per-flight altitude time-series with profile bands |

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy pyarrow
pip install pyhdf earthaccess           # for MODIS download
```

The file `POST_10s_merged_v2.parquet` must exist at `..\POST_10s_merged_v2.parquet`
relative to BASE_DIR (one folder up from POSTV1).

---

## Key Findings (10 MATCHED profiles)

| Metric | Value | Notes |
|---|---|---|
| bias_calc median | **0.69x** | MODIS underestimates in-situ Nd by 31% |
| bias_lit median | **1.00x** | Coincidentally near-unity (compensating biases) |
| Literature inflation | +45% | (vs +58% in VOCALS) |
| Re_MODIS / Re_in-situ | 1.49-1.72x | Systematic across all regimes |
| LWP_MODIS / LWP_in-situ | up to 2x in drizzle | MODIS overestimates LWP |
| Pearson r (log-log Nd) | r=0.80, p<0.01 | Strong correlation despite bias |
| Wilcoxon (3.7 um) | p=0.049 | MODIS Nd_3.7 != Nd_in-situ |
| z_top vs bias | r=-0.66, p=0.04 | Higher clouds -> stronger underestimate |
| f_ad as dominant driver | Δbias = +0.39 (S2) | f_ad gap largest in POST |
| Drizzle fraction (KW) | p=0.009 ** | Robust drizzle classification |
| Cloud depth (KW) | p=0.026 * | Wood 2008 confirmed |

See `outputs/figures/POST_PackageA-H_*.png` for full diagnostic plots.

---

## POST-Specific Notes

1. **GWIU computed:** Raw POST data does not include GWIU; step01 computes it
   from `Alt` differences using actual dt (Bug #16 fix).

2. **CAS Bin 1 / CIP Bin 1 NaN:** Inactive instrument bins zeroed in step02.
   Critical for POST -- CIP Bin 1 raw data had non-zero values that would
   inflate Nc_Total without zeroing.

3. **CIP Bin 2 (D=37.5 um) zeroed:** Overlap with CAS upper range; standard
   `CIP_CUTOFF_UM = 50` filter handles this.

4. **Span_score filter (`min_span_score=0.4`):** POST 10 s flights include
   level-leg transects that pass H/duration thresholds but are not real
   vertical penetrations. The span_score = (z_max - z_min) / sum(|d_alt|)
   filter rejects these. 4/16 geometric-pass profiles rejected for POST.

5. **No integrity check (`integrity_check_enabled=False`):** 10 s sampling
   doesn't support sub-minute gap checks (max_gap_ratio test designed for 1 Hz).

6. **Re_eff outlier filter (POSTPROC):** Single-frame CIP sensor artifacts
   can produce non-physical Re. step05 rejects per-point Re > 30 um
   (Bennartz 2007 marine Sc upper bound). Filter caught 1 point in POST
   (RF08_P06 idx 556, re_full=451.75 um sensor artifact).

7. **Drizzle threshold n_large=1 L⁻¹:** Lower than POST/VOCALS (10 L⁻¹)
   because 10 s averaging dilutes large-drop concentrations; POST drizzle
   detection requires looser threshold.

8. **MODIS time window 120 min:** Wider than VOCALS (90 min) because RF08
   flights had limited Aqua/Terra overlap. Even at 120 min, RF08_P01 missed
   by 11 minutes (Δt=131 min).

9. **2 NO_VALID matches:** RF04_P11 (granule co-located but all pixels failed
   QC, likely sun-glint or thin cloud) and RF08_P01 (no granule within
   ±120 min). These two profiles inform in-situ analyses (Package A, G)
   but are excluded from MODIS bias comparisons (Packages C-H).

10. **Heavy regime n=1 (RF08_P17):** Single heavy-drizzling profile. Heavy
    statistics in all package figures should be interpreted as a single
    case rather than a sample distribution.

11. **Bimodal analysis skipped:** Package A bimodal analysis is auto-skipped
    when moderate-drizzling profiles all fall in a single Nd cluster
    (POST: all moderate Nd > 100 cm⁻³, no Mod-Low cluster). This contrasts
    with VOCALS where bimodality reflects clean ocean patches.

---

## Bug-Fix Log (vs Original Pipeline)

| Bug | Description | Fix Location |
|-----|-------------|--------------|
| #1 | `classify_mixing_regime` thresholds not config-driven | step05 (config.MIXING_REGIME_THRESHOLDS) |
| #6 | 2.1/3.7 um spectral comparison used mixed pools (apples-to-oranges) | step09/step10 (paired qc_both pool, dRe_37_21 + dNd_calc) |
| #16 | GWIU computed with assumed dt=10s instead of actual | step01 (groupby diff dt_actual) |
| #18 | step07 hardcoded F_AD_MAX=1.0 / ND_MIN=5.0 | step07 (config.FILTERS) |
| - | POST Re sensor artifacts > 30 um | step05 (POSTPROC.re_max_physical_um) |
| - | POST level-leg false positives in profile detection | step03 (compute_span_score) |
| - | Bimodal analysis on single-cluster regimes produces KDE artifacts | step11 (dual gate: N_MOD>=4 AND both Mod-Low/Mod-High populated) |

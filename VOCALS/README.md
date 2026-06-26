# VOCALS-REx 2008 Pipeline

MODIS Nd retrieval bias validation pipeline for the VOCALS-REx 2008 campaign.
Modular config-driven structure. Runs end-to-end with `run_pipeline.py`.

---

## Campaign

- **Region:** SE Pacific (off the coast of Chile)
- **Year:** 2008 (October-November)
- **Aircraft:** CIRPAS Twin Otter
- **Data:** 1 Hz, single parquet file (`vocals_clean_export.parquet`)
- **Flights:** 18 (date-coded, e.g. `081014a`, `081017a`, ..., `081113a`)
- **Coordinates:** lat/lon NEGATIVE (lat ~ -20 deg, lon ~ -75 deg)

---

## Folder Layout

```
VOCALSV1/                           <- BASE_DIR
├── vocals_clean_export.parquet     <- raw data (1 Hz, all flights)
├── config.py
├── utils.py
├── run_pipeline.py
├── step01..step18 (Python files)
├── README.md
│
├── outputs/                        <- created automatically
│   ├── VOCALS_golden_case.csv             (final, 30 profiles)
│   ├── VOCALS_golden_microphysics.csv     (final, per-second data)
│   ├── VOCALS_MODIS_Matches.csv           (final, with bias columns)
│   ├── figures/
│   │   ├── VOCALS_Figure1-3_*.png         (paper main figures)
│   │   ├── VOCALS_PackageA-H_*.png        (analysis package figures)
│   │   └── stepXX_qc/                     (QC plots)
│   └── intermediate/                       (debug parquet/csv files)
│
└── modis_data/                     <- MODIS HDF files (step08 downloads them)
```

---

## VOCALS vs POST/MASE Differences

| Parameter | VOCALS | POST | MASE |
|---|---|---|---|
| Data time step | **1 sec** | 1 s | 10 s |
| Flights | 18 | 17 | 13 |
| Region | SE Pacific | NE Pacific | NE Pacific |
| Lat/Lon sign | **Negative** | Positive | Positive |
| Data format | **Single parquet** | NetCDF/flight | Single parquet |
| Flight ID format | **Date-coded (081014a)** | RF01..17 | RF01..13 |
| Datetime column | `UTC` | `dt` | `Datetime` |
| CAS bins | 21 (idx 0-20, 20 empty) | 20 | 20 |
| CIP bins | 63 (62 empty) | 63 | 60 |
| `final_min_depth_m` | **100** (relaxed) | 200 | 200 |
| `final_min_duration_s` | **60** (relaxed) | 120 | 120 |
| MODIS bbox padding | **0.10 deg** (tight) | 0.5 | 0.5 |
| MODIS VZA default | 60 deg (relaxed) | 60 | 60 |
| MODIS VZA optional | **40 deg (strict)** | 40 | 40 |
| Drizzle n_large | 10 L^-1 | 10 | 1 |

---

## How to Run

### Full pipeline (recommended)

```bash
cd VOCALS
python run_pipeline.py
```

18 steps run in order:
- step01-07: in-situ pipeline (~5-10 min)
- step08-10: MODIS pipeline (~30-60 min, requires internet + Earthdata)
- step11-18: analysis packages (~3-5 min)

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

### MODIS VZA strict mode (for comparison with older VOCALS pipelines)

step09 default uses `relaxed` (60 deg). For one-to-one comparison with the
older VOCALS work:

```bash
python step09_modis_colocation.py strict
```

---

## Pipeline Steps

### Core (step01..step07) - in-situ processing

| Step | Script | Output |
|------|--------|--------|
| 01 | `step01_load_data.py`        | Clean parquet, GWIU computed |
| 02 | `step02_qc_filtering.py`     | QC mask, Nc_CAS, Nc_CIP, Nc_Total |
| 03 | `step03_detect_profiles.py`  | 40 golden profiles (4-section logic) |
| 04 | `step04_drizzle.py`          | Drizzle flag + regime (CIP-based) |
| 05 | `step05_microphysics.py`     | f_ad, Re, tau, k, c_w, LWP per profile |
| 06 | `step06_figures.py`          | 3 main paper figures |
| 07 | `step07_final_check.py`      | Final filter (40 -> 30 profiles) |

### MODIS (step08..step10) - satellite retrieval

| Step | Script | Output |
|------|--------|--------|
| 08 | `step08_modis_download.py`     | 22 MODIS granules (Terra + Aqua) |
| 09 | `step09_modis_colocation.py`   | 12 MATCHED profiles (40% match rate) |
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

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy pyarrow
pip install pyhdf earthaccess           # for MODIS download
```

The file `vocals_clean_export.parquet` must exist inside `BASE_DIR`.

---

## Key Findings (12 MATCHED profiles)

| Metric | Value |
|---|---|
| bias_calc (median) | 0.92x |
| bias_lit (median)  | 1.46x |
| Literature inflation | +58% |
| dNd vs f_ad correlation | r=+0.79, p=0.002 |
| CTP as bias driver | r=+0.71, p=0.010 |

See package outputs for full statistics.

---

## VOCALS-Specific Notes

1. **GWIU automatic:** Used from parquet if present, otherwise step01 computes it.
2. **CAS_bin_20 + CIP_bin_62 empty:** dropped automatically by step02 sparse cleanup.
3. **Negative lat/lon:** SE Pacific bbox operations work correctly.
4. **Date-coded flight_id:** `'081014a'` produces cloud_id like `081014a_P01`.
5. **Relaxed thresholds:** 100 m / 60 s - SE Pacific stratocumulus is thin and short-lived.
6. **Reference comparison:** ~33 golden profiles expected from older VOCALS work (we get 40 -> 30 after final filter).
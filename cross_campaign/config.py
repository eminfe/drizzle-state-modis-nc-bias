"""
config.py  -  Cross-Campaign Analysis Configuration

Path'ler senin gerçek Windows yapına göre ayarlandı.
"""

from pathlib import Path

# =============================================================================
# 1. PROJECT ROOT (CROSS_CAMPAIGN klasörü)
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()  # repo: cross_campaign/

# =============================================================================
# 2. ALT KLASÖRLER (CROSS_CAMPAIGN içinde — outputs için)
# =============================================================================
DATA_DIR     = PROJECT_ROOT / "data"   # kullanılmıyor ama eski script'ler için
OUT_DIR      = PROJECT_ROOT / "data_outputs"
FIG_DIR      = PROJECT_ROOT / "figures"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"

# Klasörleri oluştur
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. KAMPANYA VERİ KLASÖRLERİ (senin gerçek path'lerin)
# =============================================================================
POST_OUTPUTS   = PROJECT_ROOT.parent / "POST" / "outputs"      # set to your POST pipeline outputs
MASE_OUTPUTS   = PROJECT_ROOT.parent / "MASE" / "outputs"      # set to your MASE pipeline outputs
VOCALS_OUTPUTS = PROJECT_ROOT.parent / "VOCALS" / "outputs"  # set to your VOCALS pipeline outputs

# =============================================================================
# 4. INPUT DOSYA YOLLARI
# =============================================================================
CAMPAIGN_DATA_PATHS = {
    "POST": {
        "golden":        POST_OUTPUTS / "POST_golden_case.csv",
        "modis_matches": POST_OUTPUTS / "POST_MODIS_Matches.csv",
    },
    "MASE": {
        "golden":        MASE_OUTPUTS / "MASE_golden_case.csv",
        "modis_matches": MASE_OUTPUTS / "MASE_MODIS_Matches.csv",
    },
    "VOCALS": {
        "golden":        VOCALS_OUTPUTS / "VOCALS_golden_case.csv",
        "modis_matches": VOCALS_OUTPUTS / "VOCALS_MODIS_Matches.csv",
    },
}

# RF10 case study için ekstra dosya
POST_MICROPHYSICS_FILE = POST_OUTPUTS / "POST_golden_microphysics.csv"

# =============================================================================
# 5. ANALYSIS PARAMETERS
# =============================================================================
CAMPAIGNS = ["POST", "MASE", "VOCALS"]

BOOTSTRAP_N_ITER = 10_000
BOOTSTRAP_SEED   = 42

# Drizzle state thresholds
DRIZZLE_RATIO_NON   = 0.05
DRIZZLE_RATIO_HEAVY = 0.20
N_LARGE_NON         = 5
N_LARGE_HEAVY       = 30

# Bias QC
F_AD_MAX = 1.0
ND_MIN   = 5.0

# Literature defaults
K_LITERATURE    = 0.67
F_AD_LITERATURE = 0.80
C_W_LITERATURE  = 2.0e-6

# =============================================================================
# 6. PLOT COLORS
# =============================================================================
STATE_COLORS = {
    "Non":         "#3366CC",
    "Transition":  "#FF8C00",
    "Heavy":       "#CC3333",
}

CAMPAIGN_COLORS = {
    "POST":   "#1f77b4",
    "MASE":   "#2ca02c",
    "VOCALS": "#d62728",
}

FIG_DPI = 150

# =============================================================================
# 7. PATH KONTROLÜ (python config.py çalıştırınca gösterir)
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CROSS-CAMPAIGN ANALYSIS CONFIG")
    print("=" * 70)
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"OUT_DIR:         {OUT_DIR}")
    print(f"FIG_DIR:         {FIG_DIR}")
    print()
    print("KAMPANYA OUTPUT KLASÖRLERİ:")
    print(f"  POST:    {POST_OUTPUTS}     ({'VAR' if POST_OUTPUTS.exists() else 'YOK'})")
    print(f"  MASE:    {MASE_OUTPUTS}     ({'VAR' if MASE_OUTPUTS.exists() else 'YOK'})")
    print(f"  VOCALS:  {VOCALS_OUTPUTS}   ({'VAR' if VOCALS_OUTPUTS.exists() else 'YOK'})")
    print()
    print("INPUT DOSYALARI:")
    for camp, paths in CAMPAIGN_DATA_PATHS.items():
        print(f"\n  {camp}:")
        for kind, path in paths.items():
            mark = "VAR " if path.exists() else "YOK!"
            print(f"    [{mark}] {kind:15s}  {path.name}")
    print()
    print(f"  POST_MICROPHYSICS:")
    mark = "VAR " if POST_MICROPHYSICS_FILE.exists() else "YOK!"
    print(f"    [{mark}] {POST_MICROPHYSICS_FILE.name}")
    print()
    print("=" * 70)
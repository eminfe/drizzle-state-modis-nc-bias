"""
run_all.py  -  Cross-Campaign Analysis Master Runner

Sıralı şekilde tüm cross-campaign analiz pipeline'ını çalıştırır.
Adımlar arasında bağımlılık vardır — sırayla çalışmalı.

KULLANIM:
    cd cross_campaign
    python run_all.py

    # Veya belirli bir adımdan başlamak için:
    python run_all.py --start 5    # 5. adımdan başla
    python run_all.py --only 11    # sadece 11'i çalıştır

ÖN KOŞULLAR:
    1. config.py'deki PROJECT_ROOT senin Windows yoluna ayarlı olmalı
    2. data/ klasöründe POST, MASE, VOCALS alt klasörleri olmalı
    3. Her kampanya için golden_case.csv ve MODIS_Matches.csv olmalı
    4. POST için: POST_golden_microphysics.csv da olmalı (RF10 case study için)
"""

import subprocess
import sys
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# Pipeline sırası — her adım sırayla çalışır
PIPELINE = [
    ("01", "cc_01_build_master.py",       "Build master dataset (3 campaigns → 130 profiles, 52 matched)"),
    ("02", "cc_02_bootstrap_ci.py",       "Bootstrap CI for pooled bias statistics"),
    ("03", "cc_03_figures.py",            "Initial figures: forest, regime, inflation, direction, drivers"),
    ("04", "cc_04_state_hypothesis.py",   "Initial state hypothesis exploration"),
    ("05", "cc_05_3state_master.py",      "Build 3-state classification (Non/Trans/Heavy)"),
    ("06", "cc_06_3state_bootstrap.py",   "Per-state bootstrap CI"),
    ("07", "cc_07_3state_figures.py",     "3-state figures: forest, trajectory, signature, density"),
    ("08", "cc_08_diagnostic_thresholds.py", "Diagnostic threshold tests"),
    ("09", "cc_09_rf10_qc_diagnostic.py", "RF10 QC diagnostic (POST flight only)"),
    ("10", "cc_10_rf10_case_study.py",    "RF10 single-flight case study figure"),
    ("11", "cc_11_mechanism_tests.py",    "5-mechanism analysis (M1-M5)"),
    ("12", "cc_12_mechanism_synthesis.py", "Mechanism synthesis figure (4-panel)"),
    ("13", "cc_13_two_regime_test_v2.py", "Re-error propagation test (final v2)"),
]


def run_step(num, script, desc):
    print(f"\n{'='*70}")
    print(f"  STEP {num}: {script}")
    print(f"  {desc}")
    print(f"{'='*70}")
    
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        print(f"  ✗ Script bulunamadı: {script_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPT_DIR
    )
    
    if result.returncode != 0:
        print(f"  ✗ STEP {num} BAŞARISIZ (return code: {result.returncode})")
        return False
    
    print(f"  ✓ STEP {num} tamamlandı.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Cross-campaign analysis pipeline runner")
    parser.add_argument('--start', type=int, default=1,
                        help='Start at step number (1-13, default 1)')
    parser.add_argument('--end', type=int, default=13,
                        help='End at step number (default 13)')
    parser.add_argument('--only', type=int, default=None,
                        help='Run ONLY this step number')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  CROSS-CAMPAIGN ANALYSIS PIPELINE")
    print("  POST 2008 · MASE 2005 · VOCALS-REx 2008")
    print("="*70)
    
    # Verify config first
    print("\nÖn kontrol: config.py path'leri test ediliyor...")
    config_check = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "config.py")],
        cwd=SCRIPT_DIR
    )
    if config_check.returncode != 0:
        print("✗ config.py problemi var. Önce path'leri düzelt.")
        return 1
    
    # Run pipeline
    steps_to_run = PIPELINE
    if args.only is not None:
        steps_to_run = [s for s in PIPELINE if int(s[0]) == args.only]
    else:
        steps_to_run = [s for s in PIPELINE if args.start <= int(s[0]) <= args.end]
    
    if not steps_to_run:
        print("✗ Çalıştırılacak step yok.")
        return 1
    
    print(f"\nÇalıştırılacak {len(steps_to_run)} step:")
    for num, script, desc in steps_to_run:
        print(f"  {num}. {script}")
    
    failed = []
    for num, script, desc in steps_to_run:
        ok = run_step(num, script, desc)
        if not ok:
            failed.append(num)
            response = input("\n  Başarısız oldu. Devam edeyim mi? [y/N]: ")
            if response.lower() != 'y':
                break
    
    print("\n" + "="*70)
    if failed:
        print(f"  TAMAMLANDI — {len(failed)} step başarısız: {failed}")
    else:
        print(f"  ✓ TÜM {len(steps_to_run)} STEP BAŞARILI")
    print("="*70 + "\n")
    
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

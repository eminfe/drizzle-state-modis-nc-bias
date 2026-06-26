# =============================================================================
# run_pipeline.py  —  MASE 2005 Pipeline Runner
# =============================================================================
# Tüm step dosyalarını sırayla çalıştırır.
#
# Kullanım:
#   python run_pipeline.py                  # tümünü çalıştır
#   python run_pipeline.py --skip-modis     # MODIS adımlarını (08,09,10) atla
#   python run_pipeline.py --skip-packages  # paketleri (11..18) atla
#   python run_pipeline.py --only-modis     # sadece MODIS adımları
#   python run_pipeline.py --only-packages  # sadece paketler
#   python run_pipeline.py --only-core      # sadece core (step01..step07)
#   python run_pipeline.py --from step03    # belirli bir adımdan başla
#   python run_pipeline.py --to step07      # belirli bir adıma kadar
#   python run_pipeline.py --steps step10,step13,step16  # sadece belirli adımlar
#   python run_pipeline.py --continue-on-error  # hata olursa devam et
#   python run_pipeline.py --dry-run         # sadece sırayı göster, çalıştırma
# =============================================================================

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config


# =============================================================================
# Pipeline tanımı  (MASE: step03a / step04a YOK)
# =============================================================================
# (step_id, dosya_adı, açıklama, kategori)
#   kategori: 'core', 'modis', 'package'
PIPELINE = [
    # ─── CORE (in-situ) ─────────────────────────────────────────────────
    ("step01",  "step01_load_data.py",          "parquet -> clean parquet",            "core"),
    ("step02",  "step02_qc_filtering.py",       "QC mask + Nc hesabı",                 "core"),
    ("step03",  "step03_vertical_profiles.py",  "Vertical cloud profile",              "core"),
    ("inspect", "inspect_profiles.py",          "Profil görsel kontrolü (Alt vs Time)", "core"),
    ("step04",  "step04_drizzle.py",            "Drizzle flag + regime + z_norm",      "core"),
    ("step05",  "step05_microphysics.py",       "f_ad, Re, tau, k, c_w, LWP",          "core"),
    ("step06",  "step06_figures.py",            "3 ana figür",                          "core"),
    ("step07",  "step07_final_check.py",        "Final check + final dosyalar",        "core"),

    # ─── MODIS ──────────────────────────────────────────────────────────
    ("step08",  "step08_modis_download.py",     "MODIS HDF download",                  "modis"),
    ("step09",  "step09_modis_colocation.py",   "MODIS-aircraft co-location",          "modis"),
    ("step10",  "step10_nd_calculation.py",     "Grosvenor 2018 Nd_MODIS + bias",      "modis"),

    # ─── PACKAGES (görselleştirme + analiz) ─────────────────────────────
    ("step11",  "step11_packageA.py",           "Paket A — In-Situ Cloud Physics",     "package"),
    ("step12",  "step12_packageB.py",           "Paket B — MODIS Applicability",       "package"),
    ("step13",  "step13_packageC.py",           "Paket C — Nd Bias",                    "package"),
    ("step14",  "step14_packageD.py",           "Paket D — Bias Drivers",              "package"),
    ("step15",  "step15_packageE.py",           "Paket E — Spectral Sensitivity",      "package"),
    ("step16",  "step16_packageF.py",           "Paket F — Assumption Sensitivity",    "package"),
    ("step17",  "step17_packageG.py",           "Paket G — Vertical Structure",        "package"),
    ("step18",  "step18_packageH.py",           "Paket H — Clean/Polluted",            "package"),
]


# =============================================================================
# Helpers
# =============================================================================
def fmt_duration(sec):
    """Saniyeyi okunaklı formatta döndür."""
    if sec < 60:
        return f"{sec:.1f}s"
    elif sec < 3600:
        return f"{sec//60:.0f}m {sec%60:.1f}s"
    return f"{sec//3600:.0f}h {(sec%3600)//60:.0f}m"


def print_header(title, char="=", width=70):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def run_step(step_id, filename, description, base_dir):
    """
    Bir step'i çalıştırır. Konsol çıktısı canlı akar.

    Returns
    -------
    status : str   ('ok' | 'fail' | 'skip')
    duration : float (saniye)
    """
    script_path = base_dir / filename
    if not script_path.exists():
        print(f"  [SKIP] {filename} bulunamadı (bu kampanyada yok) — atlanıyor.")
        return "skip", 0.0

    print()
    print("-" * 70)
    print(f"  >  {step_id.upper()}  —  {description}")
    print(f"     ({filename})")
    print("-" * 70)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            check=False,
        )
        duration = time.time() - start
        return ("ok" if result.returncode == 0 else "fail"), duration
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {step_id} kullanıcı tarafından durduruldu.")
        raise
    except Exception as e:
        duration = time.time() - start
        print(f"  [ERROR] {step_id} çalıştırılamadı: {e}")
        return "fail", duration


def filter_pipeline(args):
    """CLI argümanlarına göre çalıştırılacak step listesini filtrele."""
    pipeline = list(PIPELINE)

    if args.steps:
        wanted = {s.strip() for s in args.steps.split(",")}
        return [p for p in pipeline if p[0] in wanted]

    if args.only_modis:
        pipeline = [p for p in pipeline if p[3] == "modis"]
    elif args.only_packages:
        pipeline = [p for p in pipeline if p[3] == "package"]
    elif args.only_core:
        pipeline = [p for p in pipeline if p[3] == "core"]

    if args.skip_modis:
        pipeline = [p for p in pipeline if p[3] != "modis"]
    if args.skip_packages:
        pipeline = [p for p in pipeline if p[3] != "package"]

    step_ids = [p[0] for p in pipeline]
    start_idx, end_idx = 0, len(pipeline)
    if args.from_step:
        if args.from_step in step_ids:
            start_idx = step_ids.index(args.from_step)
        else:
            print(f"[WARN] --from {args.from_step} pipeline'da yok.")
    if args.to_step:
        if args.to_step in step_ids:
            end_idx = step_ids.index(args.to_step) + 1
        else:
            print(f"[WARN] --to {args.to_step} pipeline'da yok.")

    return pipeline[start_idx:end_idx]


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-modis",    action="store_true", help="MODIS adımlarını atla (08,09,10)")
    parser.add_argument("--skip-packages", action="store_true", help="Paketleri atla (11..18)")
    parser.add_argument("--only-modis",    action="store_true", help="Sadece MODIS adımları")
    parser.add_argument("--only-packages", action="store_true", help="Sadece paketler")
    parser.add_argument("--only-core",     action="store_true", help="Sadece core (01..07)")
    parser.add_argument("--from", dest="from_step", metavar="STEP", help="Belirli adımdan başla")
    parser.add_argument("--to",   dest="to_step",   metavar="STEP", help="Belirli adıma kadar")
    parser.add_argument("--steps", metavar="LIST", help="Sadece belirli adımlar (virgülle)")
    parser.add_argument("--continue-on-error", action="store_true", help="Hata olursa devam et")
    parser.add_argument("--dry-run",       action="store_true", help="Sadece listeyi göster")
    args = parser.parse_args()

    selected = filter_pipeline(args)
    if not selected:
        print("[FAIL] Çalıştırılacak step kalmadı. Filtreleri kontrol et.")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent

    print_header(f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  —  Pipeline Runner", "=", 70)
    print(f"  BASE_DIR    : {config.BASE_DIR}")
    print(f"  OUTPUT_DIR  : {config.OUTPUT_DIR}")
    print(f"  Çalışacak adım sayısı : {len(selected)}")
    print()
    print(f"  {'#':<3} {'Step':<8} {'Kategori':<10} Açıklama")
    print(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*40}")
    for i, (sid, fname, desc, cat) in enumerate(selected, 1):
        print(f"  {i:<3} {sid:<8} {cat:<10} {desc}")

    if args.dry_run:
        print("\n[DRY-RUN] Hiçbir step çalıştırılmadı.")
        return

    if len(selected) > 10 and sys.stdin.isatty():
        try:
            ans = input(f"\n  {len(selected)} adım çalıştırılacak. Devam? [Y/n]: ").strip().lower()
            if ans not in ("", "y", "yes", "evet"):
                print("İptal edildi.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nİptal edildi.")
            return

    overall_start = time.time()
    results = []   # (step_id, status, duration)

    for sid, fname, desc, cat in selected:
        try:
            status, duration = run_step(sid, fname, desc, base_dir)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Pipeline durduruldu.")
            results.append((sid, "fail", 0.0))
            break

        icon = {"ok": "[OK]", "fail": "[FAIL]", "skip": "[SKIP]"}[status]
        print(f"\n  {icon}  {sid} → {status.upper()}  ({fmt_duration(duration)})")
        results.append((sid, status, duration))

        # SADECE gerçek 'fail' HALT eder; 'skip' (dosya yok) pipeline'ı durdurmaz.
        if status == "fail" and not args.continue_on_error:
            print(f"\n  [HALT] {sid} başarısız oldu. --continue-on-error ile devam edebilirsin.")
            break

    overall_duration = time.time() - overall_start

    print_header("PIPELINE ÖZETİ", "=", 70)
    n_ok   = sum(1 for _, s, _ in results if s == "ok")
    n_fail = sum(1 for _, s, _ in results if s == "fail")
    n_skip = sum(1 for _, s, _ in results if s == "skip")
    n_notrun = len(selected) - len(results)

    print(f"\n  Çalışan adım : {len(results)} / {len(selected)}")
    print(f"  Başarılı     : {n_ok}")
    print(f"  Başarısız    : {n_fail}")
    print(f"  Atlandı (yok): {n_skip}")
    if n_notrun > 0:
        print(f"  Çalıştırılmadı: {n_notrun}")
    print(f"  Toplam süre  : {fmt_duration(overall_duration)}")

    print(f"\n  {'#':<3} {'Step':<8} {'Durum':<10} Süre")
    print(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*15}")
    for i, (sid, s, dur) in enumerate(results, 1):
        icon = {"ok": "[OK]", "fail": "[FAIL]", "skip": "[SKIP]"}[s]
        print(f"  {i:<3} {sid:<8} {icon} {s.upper():<7} {fmt_duration(dur)}")

    run_ids = {r[0] for r in results}
    not_run = [s[0] for s in selected if s[0] not in run_ids]
    if not_run:
        print(f"\n  Atlanan adımlar (HALT sonrası): {', '.join(not_run)}")

    if config.OUTPUT_DIR.exists():
        print(f"\n  Çıktılar : {config.OUTPUT_DIR}")
        for f in sorted(config.OUTPUT_DIR.glob("*.csv")):
            print(f"    {f.name:<55} {f.stat().st_size/1024:>8.1f} KB")
        if config.FIG_DIR.exists():
            n_fig = len(list(config.FIG_DIR.glob("*.png")))
            print(f"\n  Figürler : {config.FIG_DIR}  ({n_fig} PNG)")

    print()
    print("=" * 70)
    if n_fail == 0:
        print("  [OK]  PIPELINE BAŞARIYLA TAMAMLANDI")
    else:
        print(f"  [!]   PIPELINE TAMAMLANDI — {n_fail} adım başarısız")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

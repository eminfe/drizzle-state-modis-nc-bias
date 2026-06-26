# =============================================================================
# run_pipeline.py  -  POST 2008 Pipeline Runner
# =============================================================================
# Runs all step files in order.
#
# Usage:
#   python run_pipeline.py                  # run all
#   python run_pipeline.py --skip-modis     # skip MODIS steps (08, 09, 10)
#   python run_pipeline.py --skip-packages  # skip packages (11..18)
#   python run_pipeline.py --only-modis     # only MODIS steps
#   python run_pipeline.py --only-packages  # only packages
#   python run_pipeline.py --only-core      # only core (step01..step07)
#   python run_pipeline.py --from step03    # start from a specific step
#   python run_pipeline.py --to step07      # run up to a specific step
#   python run_pipeline.py --steps step03,step05,step10  # only specific steps
#   python run_pipeline.py --continue-on-error  # continue on failure
#   python run_pipeline.py --dry-run         # just show the order, do not run
# =============================================================================

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config


# =============================================================================
# Pipeline definition
# =============================================================================
# (step_id, filename, description, category)
#   category: 'core', 'modis', 'package'
#
# 'core'    -> in-situ pipeline (step01..step07)
# 'modis'   -> MODIS pipeline (step08..step10) - requires internet + Earthdata
# 'package' -> visualization / analysis packages (step11..step18)
PIPELINE = [
    # --- CORE (in-situ) -------------------------------------------------
    ("step01",  "step01_load_data.py",          "Load parquet -> clean parquet",        "core"),
    ("step02",  "step02_qc_filtering.py",       "QC mask + Nc computation",             "core"),
    ("step03",  "step03_vertical_profiles.py",  "Vertical cloud profile detection",     "core"),
    ("inspect", "inspect_profiles.py",          "Profile inspection (Alt vs Time)",     "core"),
    ("step04",  "step04_drizzle.py",            "Drizzle flag + regime + z_norm",       "core"),
    ("step05",  "step05_microphysics.py",       "f_ad, Re, tau, k, c_w, LWP",           "core"),
    ("step06",  "step06_figures.py",            "3 main paper figures",                 "core"),
    ("step07",  "step07_final_check.py",        "Final filter + final files",           "core"),

    # --- MODIS ----------------------------------------------------------
    ("step08",  "step08_modis_download.py",     "MODIS HDF download",                   "modis"),
    ("step09",  "step09_modis_colocation.py",   "MODIS-aircraft co-location",           "modis"),
    ("step10",  "step10_nd_calculation.py",     "Grosvenor 2018 Nd_MODIS + bias",       "modis"),

    # --- PACKAGES (visualization + analysis) ----------------------------
    ("step11",  "step11_packageA.py",           "Package A - In-Situ Cloud Physics",    "package"),
    ("step12",  "step12_packageB.py",           "Package B - MODIS Applicability",      "package"),
    ("step13",  "step13_packageC.py",           "Package C - Nd Bias",                  "package"),
    ("step14",  "step14_packageD.py",           "Package D - Bias Drivers",             "package"),
    ("step15",  "step15_packageE.py",           "Package E - Spectral Sensitivity",     "package"),
    ("step16",  "step16_packageF.py",           "Package F - Assumption Sensitivity",   "package"),
    ("step17",  "step17_packageG.py",           "Package G - Vertical Structure",       "package"),
    ("step18",  "step18_packageH.py",           "Package H - Clean/Polluted",           "package"),
]


# =============================================================================
# Helpers
# =============================================================================
def fmt_duration(sec):
    """Format seconds as a human-readable duration."""
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
    Run a single step. Console output streams live.

    Returns
    -------
    success : bool
    duration : float (seconds)
    """
    script_path = base_dir / filename
    if not script_path.exists():
        print(f"  [SKIP] {filename} not found.")
        return False, 0.0

    print()
    print("-" * 70)
    print(f"  >  {step_id.upper()}  -  {description}")
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
        success = (result.returncode == 0)
        return success, duration
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {step_id} stopped by user.")
        raise
    except Exception as e:
        duration = time.time() - start
        print(f"  [ERROR] {step_id} could not run: {e}")
        return False, duration


def filter_pipeline(args):
    """Filter the pipeline based on CLI arguments."""
    pipeline = list(PIPELINE)

    # --steps explicit list
    if args.steps:
        wanted = {s.strip() for s in args.steps.split(",")}
        pipeline = [p for p in pipeline if p[0] in wanted]
        return pipeline

    # --only-* flags (mutually exclusive)
    if args.only_modis:
        pipeline = [p for p in pipeline if p[3] == "modis"]
    elif args.only_packages:
        pipeline = [p for p in pipeline if p[3] == "package"]
    elif args.only_core:
        pipeline = [p for p in pipeline if p[3] == "core"]

    # --skip-* flags (additive)
    if args.skip_modis:
        pipeline = [p for p in pipeline if p[3] != "modis"]
    if args.skip_packages:
        pipeline = [p for p in pipeline if p[3] != "package"]

    # --from / --to range filtering
    step_ids = [p[0] for p in pipeline]
    start_idx = 0
    end_idx = len(pipeline)
    if args.from_step:
        if args.from_step in step_ids:
            start_idx = step_ids.index(args.from_step)
        else:
            print(f"[WARN] --from {args.from_step} not in pipeline.")
    if args.to_step:
        if args.to_step in step_ids:
            end_idx = step_ids.index(args.to_step) + 1
        else:
            print(f"[WARN] --to {args.to_step} not in pipeline.")

    return pipeline[start_idx:end_idx]


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR} pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Filtering
    parser.add_argument("--skip-modis",    action="store_true",
                        help="Skip MODIS steps (step08, step09, step10)")
    parser.add_argument("--skip-packages", action="store_true",
                        help="Skip packages (step11..step18)")
    parser.add_argument("--only-modis",    action="store_true",
                        help="Only MODIS steps")
    parser.add_argument("--only-packages", action="store_true",
                        help="Only packages")
    parser.add_argument("--only-core",     action="store_true",
                        help="Only core (step01..step07)")
    parser.add_argument("--from", dest="from_step", metavar="STEP",
                        help="Start from a specific step (e.g. step03)")
    parser.add_argument("--to",   dest="to_step",   metavar="STEP",
                        help="Run up to a specific step (e.g. step07)")
    parser.add_argument("--steps", metavar="LIST",
                        help="Run only specific steps (comma-separated, e.g. step03,step05)")
    # Behavior
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue if a step fails")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Just show the step list, do not run")

    args = parser.parse_args()

    # Filter pipeline
    selected = filter_pipeline(args)

    if not selected:
        print("[FAIL] No steps left to run. Check filters.")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent

    # Banner
    print_header(
        f"{config.CAMPAIGN_NAME} {config.CAMPAIGN_YEAR}  -  Pipeline Runner",
        "=", 70,
    )
    print(f"  BASE_DIR    : {config.BASE_DIR}")
    print(f"  OUTPUT_DIR  : {config.OUTPUT_DIR}")
    print(f"  Steps to run: {len(selected)}")
    print()
    print(f"  {'#':<3} {'Step':<8} {'Category':<10} Description")
    print(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*40}")
    for i, (sid, fname, desc, cat) in enumerate(selected, 1):
        print(f"  {i:<3} {sid:<8} {cat:<10} {desc}")

    if args.dry_run:
        print("\n[DRY-RUN] No steps were run.")
        return

    # Confirm if many steps
    if len(selected) > 10 and sys.stdin.isatty():
        try:
            ans = input(f"\n  About to run {len(selected)} steps. Continue? [Y/n]: ").strip().lower()
            if ans not in ("", "y", "yes"):
                print("Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return

    # --- Run --------------------------------------------------------------
    overall_start = time.time()
    results = []   # (step_id, success, duration)

    for sid, fname, desc, cat in selected:
        try:
            success, duration = run_step(sid, fname, desc, base_dir)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Pipeline stopped.")
            results.append((sid, False, 0.0))
            break

        status_icon = "[OK]" if success else "[FAIL]"
        print(f"\n  {status_icon}  {sid} -> {'OK' if success else 'FAILED'}  ({fmt_duration(duration)})")
        results.append((sid, success, duration))

        if not success and not args.continue_on_error:
            print(f"\n  [HALT] {sid} failed. Use --continue-on-error to keep going.")
            break

    overall_duration = time.time() - overall_start

    # --- Final report -----------------------------------------------------
    print_header("PIPELINE SUMMARY", "=", 70)
    n_ok   = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    n_skip = len(selected) - len(results)

    print(f"\n  Steps run     : {len(results)} / {len(selected)}")
    print(f"  Succeeded     : {n_ok}")
    print(f"  Failed        : {n_fail}")
    if n_skip > 0:
        print(f"  Not run       : {n_skip}")
    print(f"  Total time    : {fmt_duration(overall_duration)}")

    print(f"\n  {'#':<3} {'Step':<8} {'Status':<10} Duration")
    print(f"  {'-'*3} {'-'*8} {'-'*10} {'-'*15}")
    for i, (sid, ok, dur) in enumerate(results, 1):
        icon = "[OK]" if ok else "[FAIL]"
        status = "OK" if ok else "FAILED"
        print(f"  {i:<3} {sid:<8} {icon} {status:<7} {fmt_duration(dur)}")

    # Show steps that didn't run
    run_ids = {r[0] for r in results}
    not_run = [s[0] for s in selected if s[0] not in run_ids]
    if not_run:
        print(f"\n  Skipped after HALT: {', '.join(not_run)}")

    # Output files
    if config.OUTPUT_DIR.exists():
        print(f"\n  Outputs : {config.OUTPUT_DIR}")
        for f in sorted(config.OUTPUT_DIR.glob("*.csv")):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<55} {size_kb:>8.1f} KB")
        if config.FIG_DIR.exists():
            n_fig = len(list(config.FIG_DIR.glob("*.png")))
            print(f"\n  Figures : {config.FIG_DIR}  ({n_fig} PNG)")

    print()
    print("=" * 70)
    if n_fail == 0:
        print("  PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print(f"  PIPELINE COMPLETED - {n_fail} step(s) failed")
    print("=" * 70)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
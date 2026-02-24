#!/usr/bin/env python3
"""
run_all.py — Master script: one command reproduces everything.

Usage:
    python src/run_all.py              # Full pipeline (requires data download)
    python src/run_all.py --skip-download  # Skip download, process existing data
    python src/run_all.py --from-step 4    # Start from step 4 (merge)

Steps:
    1. Download Platsbanken data (~5.4 GB)
    2. Process JSONL → SSYK4 × month aggregates
    3. Fetch auxiliary data (OMXS30, Riksbanken, DAIOE)
    4. Merge postings with DAIOE, classify by exposure quartile
    5. DiD regression analysis
    6. Generate figures and tables
    7. Robustness checks

The full pipeline takes ~20–30 minutes on first run (dominated by download).
Subsequent runs with --skip-download take ~5–10 minutes.

All intermediate outputs are saved to data/processed/ and can be inspected.
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure src/ is on the path
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))


def run_step(step_num: int, module_name: str, description: str):
    """Import and run a pipeline step, with timing."""
    print(f"\n{'=' * 70}")
    print(f"  STEP {step_num}: {description}")
    print(f"{'=' * 70}\n")

    t0 = time.time()

    try:
        # Dynamic import
        mod = __import__(module_name)
        mod.main()
    except Exception as e:
        print(f"\n  ERROR in step {step_num} ({module_name}): {e}")
        import traceback
        traceback.print_exc()
        print(f"\n  Continuing to next step...")
        return False

    elapsed = time.time() - t0
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Master replication script for 'Two Economies?' paper"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the data download step (use existing files)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use 1%% sample data for pipeline testing (~100x smaller)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from this step number (1–7)",
    )
    args = parser.parse_args()

    # Pass --sample to step 1 via sys.argv manipulation
    if args.sample:
        sys.argv = [sys.argv[0], "--sample"]
    else:
        sys.argv = [sys.argv[0]]

    mode = "SAMPLE (1%)" if args.sample else "FULL"
    print("=" * 70)
    print("  REPLICATION PIPELINE")
    print("  'Two Economies? Stock Markets, Job Postings,")
    print("   and AI Exposure in Sweden'")
    print(f"  Mode: {mode}")
    print("=" * 70)
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Starting from step: {args.from_step}")
    print(f"  Skip download: {args.skip_download}")

    t_start = time.time()

    steps = [
        (1, "01_download_platsbanken", "Download Platsbanken data"),
        (2, "02_process_platsbanken", "Process JSONL → aggregates"),
        (3, "03_fetch_auxiliary", "Fetch auxiliary data"),
        (4, "04_merge_and_classify", "Merge with DAIOE + classify"),
        (5, "05_analysis", "DiD regression analysis"),
        (6, "06_figures_tables", "Generate figures and tables"),
        (7, "07_robustness", "Robustness checks"),
    ]

    results = {}
    for step_num, module, desc in steps:
        if step_num < args.from_step:
            print(f"\n  Skipping step {step_num}: {desc}")
            continue
        if step_num == 1 and args.skip_download:
            print(f"\n  Skipping step 1: Download (--skip-download)")
            continue

        success = run_step(step_num, module, desc)
        results[step_num] = success

    # Summary
    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total/60:.1f} minutes")
    print(f"{'=' * 70}")

    for step_num, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  Step {step_num}: {status}")

    print(f"\n  Output directories:")
    print(f"    data/processed/  — intermediate CSV files")
    print(f"    figures/         — publication figures (300 dpi PNG)")
    print(f"    tables/          — regression tables (CSV + LaTeX)")


if __name__ == "__main__":
    main()

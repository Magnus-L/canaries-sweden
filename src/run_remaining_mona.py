#!/usr/bin/env python3
"""
run_remaining_mona.py — Master script to run all remaining MONA robustness scripts.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
======================================================================

WHAT IT DOES:
  Runs scripts 23-30 sequentially, logging progress and any errors.
  Script 31 is excluded (it runs locally, no MONA access needed).

  Script 23 (Rambachan-Roth) is R; the rest are Python.
  All scripts are independent — no ordering constraints.
  Script 23 runs first because it is the gatekeeper for submission.

PREREQUISITES:
  - output_18/corrected_es_all_ref2022H1.csv must exist (from script 18)
  - daioe_quartiles.csv must be on the MONA network share
  - For script 26: dingel_neiman_ssyk4.csv on MONA share (optional; script
    will attempt to build from raw crosswalks if missing)
  - For script 29: eloundou_ssyk4.csv on MONA share (optional; script
    will attempt to build from raw crosswalks if missing)
  - R with HonestDiD package installed (for script 23)

USAGE:
  python run_remaining_mona.py

  Or to skip a script that already ran successfully:
  python run_remaining_mona.py --skip 24 --skip 27

OUTPUT:
  - Each script writes to its own output_NN/ directory
  - This master script writes run_remaining_log.txt with timestamps,
    pass/fail status, and any error messages
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------- Configuration ----------

# Directory where all scripts live (same as this file)
SCRIPT_DIR = Path(__file__).parent

# Scripts to run, in order. Script 23 first (gatekeeper).
SCRIPTS = [
    {"num": 23, "file": "23_mona_rambachan_roth.R",     "lang": "R",      "desc": "Rambachan-Roth sensitivity (GATEKEEPER)"},
    {"num": 24, "file": "24_mona_summary_stats.py",      "lang": "python", "desc": "Employment summary stats + zero-cell diagnostics"},
    {"num": 25, "file": "25_mona_placebo_dates.py",      "lang": "python", "desc": "Placebo treatment dates (Nov 2021, Jul 2022)"},
    {"num": 26, "file": "26_mona_telework_split.py",     "lang": "python", "desc": "Teleworkability split (Dingel-Neiman)"},
    {"num": 27, "file": "27_mona_alt_se_clustering.py",  "lang": "python", "desc": "Alternative SE clustering"},
    {"num": 28, "file": "28_mona_ssyk_attrition.py",     "lang": "python", "desc": "SSYK attrition robustness (2019-2023 only)"},
    {"num": 29, "file": "29_mona_eloundou_measure.py",   "lang": "python", "desc": "Eloundou GPT exposure measure"},
    {"num": 30, "file": "30_mona_entrant_composition.py", "lang": "python", "desc": "Entrant composition (Q4 share over time)"},
]

LOG_FILE = SCRIPT_DIR.parent / "run_remaining_log.txt"


def run_script(script_info):
    """Run a single script and return (success, duration_seconds, error_message)."""
    filepath = SCRIPT_DIR / script_info["file"]

    if not filepath.exists():
        return False, 0, f"File not found: {filepath}"

    if script_info["lang"] == "R":
        cmd = ["Rscript", str(filepath)]
    else:
        cmd = [sys.executable, str(filepath)]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max per script
        )
        duration = time.time() - start

        if result.returncode != 0:
            # Capture last 20 lines of stderr for the log
            err_lines = result.stderr.strip().split("\n")[-20:]
            return False, duration, "\n".join(err_lines)

        return True, duration, None

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, duration, "TIMEOUT after 2 hours"
    except Exception as e:
        duration = time.time() - start
        return False, duration, str(e)


def main():
    # Parse --skip arguments
    skip_nums = set()
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--skip" and i + 1 < len(sys.argv):
            skip_nums.add(int(sys.argv[i + 1]))
            i += 2
        else:
            i += 1

    # Open log file
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        header = f"MONA robustness pipeline — started {datetime.now():%Y-%m-%d %H:%M}"
        log.write(header + "\n")
        log.write("=" * len(header) + "\n\n")
        print(header)

        results = []

        for s in SCRIPTS:
            if s["num"] in skip_nums:
                msg = f"  [{s['num']}] SKIPPED (--skip flag)"
                log.write(msg + "\n")
                print(msg)
                results.append((s["num"], "SKIPPED", 0))
                continue

            print(f"\n  [{s['num']}] {s['desc']}...")
            log.write(f"\n[{s['num']}] {s['file']} — {s['desc']}\n")
            log.write(f"  Started: {datetime.now():%H:%M:%S}\n")
            log.flush()

            success, duration, error = run_script(s)
            mins = duration / 60

            if success:
                msg = f"  [{s['num']}] PASSED  ({mins:.1f} min)"
                log.write(f"  PASSED in {mins:.1f} min\n")
            else:
                msg = f"  [{s['num']}] FAILED  ({mins:.1f} min)"
                log.write(f"  FAILED in {mins:.1f} min\n")
                log.write(f"  Error:\n{error}\n")
                print(f"  Error: {error[:200]}")

                # If the gatekeeper fails, warn but continue
                if s["num"] == 23:
                    log.write("  *** GATEKEEPER FAILED — check output before proceeding ***\n")
                    print("  *** GATEKEEPER (Rambachan-Roth) FAILED ***")

            print(msg)
            log.flush()
            results.append((s["num"], "PASSED" if success else "FAILED", mins))

        # Summary
        log.write("\n\n" + "=" * 50 + "\n")
        log.write(f"SUMMARY — finished {datetime.now():%Y-%m-%d %H:%M}\n\n")
        print(f"\n{'=' * 50}")
        print("SUMMARY\n")

        for num, status, mins in results:
            line = f"  Script {num}: {status}" + (f"  ({mins:.1f} min)" if status != "SKIPPED" else "")
            log.write(line + "\n")
            print(line)

        n_passed = sum(1 for _, s, _ in results if s == "PASSED")
        n_failed = sum(1 for _, s, _ in results if s == "FAILED")
        n_skipped = sum(1 for _, s, _ in results if s == "SKIPPED")
        total_mins = sum(m for _, s, m in results if s != "SKIPPED")

        footer = f"\n  {n_passed} passed, {n_failed} failed, {n_skipped} skipped. Total: {total_mins:.0f} min."
        log.write(footer + "\n")
        print(footer)

    print(f"\nLog written to: {LOG_FILE}")


if __name__ == "__main__":
    main()

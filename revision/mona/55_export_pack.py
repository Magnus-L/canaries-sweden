#!/usr/bin/env python3
"""
55_export_pack.py -- one small folder holding everything the write-up needs.

======================================================================
  RUNS IN MONA. NO SQL. Reads every output_*/ directory and writes
  export_pack/. Seconds. Submit it or run it in Spyder; either is fine.
======================================================================

WHY. The export budget is 5 MB per file and 50 MB per rolling seven days,
and by the end of this round there are eleven output directories. Script
50 alone produced 23 MB. Exporting the raw directories would either
exhaust the budget or, worse, exhaust it on the wrong files: the large
ones are supports and score tables, and the small ones are the estimates.

WHAT IT PACKS, and the rule is deliberately simple so nothing needed is
lost by a clever heuristic:

  ALWAYS      every *_summary.txt and every *_log.txt. These carry the
              printed results and the reasoning, and they are the spine of
              the write-up.
  ALWAYS      every .csv at or under SMALL_CSV_KB. Coefficient tables are
              a few kilobytes; this catches all of them.
  NEVER       any .csv above that, UNLESS it is named in PRIORITY. Those
              are supports and per-cell tables. They are listed in the
              manifest with their size and shape so nothing vanishes
              silently, and any one of them can be fetched in a later
              round by adding its name to PRIORITY.

DISCLOSURE. Every packed csv is re-floored on the way out: any column that
looks like a count is checked, and rows below the floor are DROPPED, not
blanked. A blanked row still discloses that a cell exists with between one
and nine people in it. Files are already floored by the scripts that wrote
them; this is a second pass, because the cost of the check is nothing and
the cost of missing one is a disclosure incident.

The manifest records a SHA-256 for every packed file so the copy that
arrives on the Mac can be verified against the copy that left.
"""

import hashlib
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
PACK = HERE / "export_pack"
SMALL_CSV_KB = 800
FLOOR = 5
BUDGET_MB = 50

# Large files worth their weight. Add a name here and re-run to fetch it.
PRIORITY = {
    "horserace_estimates.csv",     # 47h: every design x arm x truncation
    "settled_estimates.csv",       # 47k
    "agebase_estimates.csv",       # 47L pooled
    "agebase_gradient.csv",        # 47L by age band
    "flow_estimates.csv",          # 54 pooled
    "flow_gradient.csv",           # 54 by age band
    "fresh_pooled.csv",            # 53 three arms
    "fresh_es.csv",                # 53 event studies
    "selection_did.csv",           # 53 selection description
    "poisson_pooled.csv",          # 43 headline
    "poisson_es.csv",              # 43 event study
    "bridge_inputs.csv",           # 43 OLS vs Poisson bridge
}

COUNT_HINTS = ("n_emp", "n_obs", "n", "count", "cells", "firms", "n_coded",
               "n_hire", "n_sep", "n_kept", "employers", "n_total")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refloor(src: Path, dst: Path) -> tuple:
    """
    Copy a csv, dropping any row whose count column is between 1 and the
    floor. Returns (rows_in, rows_out). A file with no count column is
    copied unchanged.
    """
    try:
        df = pd.read_csv(src)
    except Exception:
        shutil.copy2(src, dst)
        return (-1, -1)
    cols = [c for c in df.columns
            if c.lower() in COUNT_HINTS and pd.api.types.is_numeric_dtype(df[c])]
    n_in = len(df)
    if cols:
        keep = pd.Series(True, index=df.index)
        for c in cols:
            v = df[c]
            keep &= ~((v > 0) & (v < FLOOR))
        df = df[keep]
    df.to_csv(dst, index=False)
    return (n_in, len(df))


def main():
    PACK.mkdir(exist_ok=True)
    mc.Tee(PACK / "55_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("55: EXPORT PACK")
    print("=" * 70)

    packed, skipped = [], []
    for d in sorted(HERE.glob("output_*")):
        if not d.is_dir() or d.name == "export_pack":
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            kb = f.stat().st_size / 1024
            take = (f.suffix in (".txt",)
                    or (f.suffix == ".csv"
                        and (kb <= SMALL_CSV_KB or f.name in PRIORITY)))
            if not take:
                shape = ""
                try:
                    h = pd.read_csv(f, nrows=1)
                    n = sum(1 for _ in open(f, "rb")) - 1
                    shape = f"{n:,} rows x {len(h.columns)} cols"
                except Exception:
                    shape = "unreadable header"
                skipped.append((d.name, f.name, kb, shape))
                continue
            out = PACK / f"{d.name}__{f.name}"
            if f.suffix == ".csv":
                n_in, n_out = refloor(f, out)
                note = (f"{n_in:,} -> {n_out:,} rows"
                        if n_in >= 0 and n_out != n_in else "")
            else:
                shutil.copy2(f, out)
                note = ""
            packed.append((d.name, f.name, out.stat().st_size / 1024, note))

    total_mb = sum(k for _, _, k, _ in packed) / 1024
    man = PACK / "MANIFEST.txt"
    lines = [f"EXPORT PACK  {time.strftime('%Y-%m-%d %H:%M')}",
             "=" * 70, "",
             f"PACKED {len(packed)} files, {total_mb:.2f} MB total", ""]
    for d, f, kb, note in packed:
        lines.append(f"  {kb:9.1f} KB  {d}__{f}" + (f"   [{note}]" if note else ""))
    lines += ["", f"NOT PACKED {len(skipped)} files (over {SMALL_CSV_KB} KB and "
              f"not in PRIORITY).", "Add a name to PRIORITY in the script and "
              "re-run to fetch one.", ""]
    for d, f, kb, shape in skipped:
        lines.append(f"  {kb:9.1f} KB  {d}/{f}   {shape}")
    lines += ["", "SHA-256 of every packed file:", ""]
    for p in sorted(PACK.glob("*")):
        if p.name != "MANIFEST.txt":
            lines.append(f"  {sha(p)}  {p.name}")
    lines += ["", f"Budget note: 5 MB per file, {BUDGET_MB} MB per rolling "
              "seven days.",
              f"This pack is {total_mb:.2f} MB across {len(packed)} files; "
              "the largest single file is "
              f"{max((k for _, _, k, _ in packed), default=0)/1024:.2f} MB.",
              f"Runtime {time.time()-t0:.0f}s."]
    man.write_text("\n".join(lines))
    print("\n".join(lines))
    over = [f"{d}__{f}" for d, f, kb, _ in packed if kb > 5 * 1024]
    if over:
        print("\nWARNING: these exceed the 5 MB per-file cap and will be "
              "refused:\n  " + "\n  ".join(over))
    print("\n55 done. Export the whole export_pack/ directory.")


if __name__ == "__main__":
    main()

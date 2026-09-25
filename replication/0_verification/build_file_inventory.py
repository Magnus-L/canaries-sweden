#!/usr/bin/env python3
"""
build_file_inventory.py: write FILES.csv, the inventory of the replication package.

Walks the package root and lists every file with its size in bytes and its SHA-256
digest, one row per file, with paths relative to the package root and forward
slashes, sorted. Three kinds of file are left out because they are not part of the
package as deposited: FILES.csv itself (it cannot list its own digest), editor and
interpreter caches (.DS_Store, __pycache__), and what a run writes (output/,
data/processed/, and the Platsbanken archives under data/raw/platsbanken/, which
are downloaded by 1_data_public/ and verified there against their own digests). The
inputs that are fetched or built by the replicator rather than shipped are also left
out: the three Yahoo Finance daily series fetched by 1_data_public/03, and the two
inputs of Part V (the employer cube and the business-register bulk file).

Usage:  python 0_verification/build_file_inventory.py [package_root]
Re-run after any change to the package; the inventory is compared with the archive
that is deposited.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

EXCLUDED_DIRS = {"__pycache__", "output", ".ipynb_checkpoints"}
EXCLUDED_PREFIXES = ("data/processed/", "data/raw/platsbanken/")
EXCLUDED_NAMES = {".DS_Store", "FILES.csv"}
NOT_SHIPPED = {"data/raw/omxs30_daily.csv", "data/raw/omxspi_daily.csv",
               "data/raw/sp500_daily.csv", "data/raw/firm_month_v2.csv.gz",
               "data/raw/scb_bulkfil.zip"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path(__file__).resolve().parents[1]).resolve()
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in EXCLUDED_NAMES:
            continue
        rel = p.relative_to(root).as_posix()
        if set(p.relative_to(root).parts[:-1]) & EXCLUDED_DIRS:
            continue
        if rel.startswith(EXCLUDED_PREFIXES) or rel in NOT_SHIPPED:
            continue
        rows.append((rel, p.stat().st_size, sha256(p)))
    with (root / "FILES.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)
    print(f"FILES.csv: {len(rows)} files, {sum(r[1] for r in rows):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

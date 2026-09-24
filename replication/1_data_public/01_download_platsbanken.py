#!/usr/bin/env python3
"""
01_download_platsbanken.py: download the Platsbanken advertisement archives.

WHAT IT DOES
Downloads JobTech's historical archives of Platsbanken, the Swedish Public
Employment Service's job board, into config.JOBADS_DIR: one annual file for
each year 2020 to 2025 and the two closed-quarter files for the first half of
2026 (2026-Q1, 2026-Q2). Each file is a zip holding one JSON object per
advertisement. Files already present are not downloaded again. With --sample,
the 1 per cent sample files are fetched instead, for testing the chain.

The live feed (JobStream) is not used anywhere in the package: it returns
only the advertisements published on the day of extraction, so recent months
are under-counted. The closed-quarter files are complete.

SOURCE AND LICENCE
https://data.jobtechdev.se/annonser/historiska/ (Arbetsförmedlingen, CC0).
The archives are republished when JobTech revises them, so a download made
today need not be byte-identical to the one the paper used; the SHA-256 of
every archive the paper used is listed in data/DATA-MANIFEST.csv and printed
by --verify.

RUNTIME AND SIZE
About 6 GB in total (0.5 to 1.4 GB per annual file); the download time is set
by the connection.

    python 1_data_public/01_download_platsbanken.py            # full archives
    python 1_data_public/01_download_platsbanken.py --sample   # 1 per cent samples
    python 1_data_public/01_download_platsbanken.py --verify   # hashes only
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

BASE = "https://data.jobtechdev.se/annonser/historiska"
SAMPLE_OVERRIDES = {2025: "2025_Q3_1_percent_jsonl.zip"}
MANIFEST = config.DATA / "DATA-MANIFEST.csv"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Stream a file to disk with a progress bar; the archives exceed 1 GB."""
    if dest.exists():
        print(f"  already present: {dest.name}")
        return dest
    print(f"  downloading {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                      desc=dest.name, disable=total == 0) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> int:
    """Compare every archive present with the hash of the one the paper used."""
    expected = {}
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["file"].endswith(".jsonl.zip"):
                expected[row["file"]] = row["sha256"]
    bad = 0
    for name, digest in sorted(expected.items()):
        p = config.JOBADS_DIR / name
        if not p.exists():
            print(f"  MISSING   {name}")
            bad += 1
            continue
        got = sha256(p)
        ok = got == digest
        bad += not ok
        print(f"  {'SAME     ' if ok else 'DIFFERENT'} {name}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", action="store_true", help="1%% sample files")
    ap.add_argument("--verify", action="store_true", help="hash check only")
    args = ap.parse_args()
    if args.verify:
        return verify()

    config.JOBADS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Platsbanken archives into {config.JOBADS_DIR}")
    for year in config.PLATSBANKEN_YEARS:
        if args.sample:
            name = SAMPLE_OVERRIDES.get(year, f"{year}_beta1_1_percent_jsonl.zip")
            url = f"{BASE}/berikade/exempel/{name}"
            dest = config.JOBADS_DIR / f"{year}_sample.jsonl.zip"
        else:
            url = f"{BASE}/{year}.jsonl.zip"
            dest = config.platsbanken_zip(year)
        try:
            download_file(url, dest)
        except requests.HTTPError as e:
            print(f"  WARNING: {year} failed ({e})")
    if not args.sample:
        for stem in config.PLATSBANKEN_QUARTERS:
            try:
                download_file(f"{BASE}/{stem}.jsonl.zip", config.platsbanken_zip(stem))
            except requests.HTTPError as e:
                print(f"  WARNING: {stem} failed ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

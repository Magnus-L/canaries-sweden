#!/usr/bin/env python3
"""
01_download_platsbanken.py — Download Platsbanken historical job posting data.

Downloads JSONL zip files from data.jobtechdev.se for 2020–2025.
Uses streaming downloads with progress tracking to handle large files
(individual files range 527 MB to 1.34 GB compressed).

Modes:
  --sample   Download 1% enriched sample files (~8–15 MB each) for testing
  (default)  Download full raw JSONL files (~500 MB–1.3 GB each)

Also fetches the most recent ads via the JobStream API to supplement
the historical bulk download with the latest data.

Data source: Arbetsförmedlingen, CC0 license.
URL: https://data.jobtechdev.se/annonser/historiska/index.html
"""

import sys
import argparse
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PLATSBANKEN_YEARS, platsbanken_url,
    platsbanken_sample_url, JOBSTREAM_BASE,
)

import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """
    Stream-download a file with progress bar.

    Why streaming: Platsbanken files can be >1 GB. Loading the entire
    response into memory would be wasteful. Streaming writes chunks
    directly to disk.
    """
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return dest

    print(f"  Downloading {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dest.name,
        disable=total == 0,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    size_mb = dest.stat().st_size / 1e6
    print(f"  Saved: {dest.name} ({size_mb:.0f} MB)")
    return dest


def download_historical(years: list[int] = None, sample: bool = False) -> list[Path]:
    """
    Download historical Platsbanken JSONL zip files for specified years.

    If sample=True, downloads the 1% enriched sample files instead (~100x smaller).
    These are pseudo-random samples that preserve all fields — ideal for
    testing the pipeline before committing to the full 5.4 GB download.
    """
    if years is None:
        years = PLATSBANKEN_YEARS

    mode = "1% SAMPLE" if sample else "FULL"
    print(f"Downloading Platsbanken {mode} data for {years[0]}–{years[-1]}...")
    downloaded = []

    for year in years:
        if sample:
            url = platsbanken_sample_url(year)
            dest = RAW / f"{year}_sample.jsonl.zip"
        else:
            url = platsbanken_url(year)
            dest = RAW / f"{year}.jsonl.zip"

        try:
            download_file(url, dest)
            downloaded.append(dest)
        except requests.HTTPError as e:
            print(f"  WARNING: {year} download failed ({e}) — trying next year")

    return downloaded


def fetch_jobstream_snapshot() -> Path:
    """
    Fetch all currently published ads from the JobStream API.

    The snapshot endpoint returns every ad currently live on Platsbanken.
    This supplements the historical bulk download with the very latest
    postings that may not yet appear in the annual files.

    No authentication required (despite plan notes about API keys —
    the docs confirm open access).
    """
    print("Fetching JobStream snapshot (current live ads)...")

    dest = RAW / "jobstream_snapshot.jsonl"
    if dest.exists():
        print(f"  Already fetched: {dest.name}")
        print("  (Delete to re-fetch)")
        return dest

    url = f"{JOBSTREAM_BASE}/v2/snapshot"
    resp = requests.get(url, stream=True, timeout=120, headers={"Accept": "application/jsonl"})
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc="jobstream_snapshot",
        disable=total == 0,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    size_mb = dest.stat().st_size / 1e6
    print(f"  Saved: {dest.name} ({size_mb:.0f} MB)")
    return dest


def main():
    """
    Download all Platsbanken data needed for the analysis.

    --sample mode: 1% files (~70 MB total) for pipeline testing.
    Full mode: raw JSONL files (~5.4 GB total) for production.
    """
    parser = argparse.ArgumentParser(description="Download Platsbanken data")
    parser.add_argument(
        "--sample", action="store_true",
        help="Download 1%% sample files instead of full data (~100x smaller)",
    )
    parser.add_argument(
        "--no-jobstream", action="store_true",
        help="Skip the JobStream snapshot download",
    )
    args = parser.parse_args()

    print("=" * 70)
    mode = "1% SAMPLE (pipeline testing)" if args.sample else "FULL"
    print(f"STEP 1: Download Platsbanken data [{mode}]")
    print("=" * 70)

    files = download_historical(sample=args.sample)
    print(f"\nDownloaded {len(files)} files.")

    total_mb = sum(f.stat().st_size for f in files if f.exists()) / 1e6
    print(f"Total size: {total_mb:.0f} MB")

    if not args.no_jobstream and not args.sample:
        print("\n" + "=" * 70)
        print("STEP 2: Fetch JobStream snapshot (current live ads)")
        print("=" * 70)

        try:
            fetch_jobstream_snapshot()
        except Exception as e:
            print(f"  WARNING: JobStream fetch failed: {e}")
            print("  (This is optional — historical data is sufficient for the paper)")
    elif args.sample:
        print("\n  Skipping JobStream snapshot in sample mode.")

    print("\nDone. Run 02_process_platsbanken.py next.")


if __name__ == "__main__":
    main()

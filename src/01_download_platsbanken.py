#!/usr/bin/env python3
"""
01_download_platsbanken.py — Download Platsbanken historical job posting data.

Downloads JSONL zip files from data.jobtechdev.se for 2020–2025.
Uses streaming downloads with progress tracking to handle large files
(individual files range 527 MB to 1.34 GB compressed).

Also fetches the most recent ads via the JobStream API to supplement
the historical bulk download with the latest data.

Data source: Arbetsförmedlingen, CC0 license.
URL: https://data.jobtechdev.se/annonser/historiska/index.html
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW, PLATSBANKEN_YEARS, platsbanken_url, JOBSTREAM_BASE

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


def download_historical(years: list[int] = None) -> list[Path]:
    """
    Download historical Platsbanken JSONL zip files for specified years.

    Each year's file contains all job ads published that year, one JSON
    object per line. These are the raw (non-enriched) files, which have
    a consistent schema across all years.
    """
    if years is None:
        years = PLATSBANKEN_YEARS

    print(f"Downloading Platsbanken historical data for {years[0]}–{years[-1]}...")
    downloaded = []

    for year in years:
        url = platsbanken_url(year)
        dest = RAW / f"{year}.jsonl.zip"
        download_file(url, dest)
        downloaded.append(dest)

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

    Step 1: Historical JSONL files (2020–2025), ~5.4 GB total compressed.
    Step 2: JobStream snapshot for the very latest ads.
    """
    print("=" * 70)
    print("STEP 1: Download Platsbanken historical data")
    print("=" * 70)

    files = download_historical()
    print(f"\nDownloaded {len(files)} files.")

    total_mb = sum(f.stat().st_size for f in files if f.exists()) / 1e6
    print(f"Total size: {total_mb:.0f} MB")

    print("\n" + "=" * 70)
    print("STEP 2: Fetch JobStream snapshot (current live ads)")
    print("=" * 70)

    try:
        fetch_jobstream_snapshot()
    except Exception as e:
        print(f"  WARNING: JobStream fetch failed: {e}")
        print("  (This is optional — historical data is sufficient for the paper)")

    print("\nDone. Run 02_process_platsbanken.py next.")


if __name__ == "__main__":
    main()

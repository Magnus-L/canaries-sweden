#!/usr/bin/env python3
"""
03_fetch_auxiliary.py — Fetch auxiliary data: OMXS30, Riksbanken, Indeed US, DAIOE.

Reuses patterns from the SvD project for consistency and comparability.
All data are publicly available and replicable.
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, DAIOE_PATH, DAIOE_XLSX, SVD_PROJECT,
    BASE_MONTH,
)

import pandas as pd
import numpy as np
import requests


# ── OMXS30 ────────────────────────────────────────────────────────────────────

def fetch_omxs30() -> pd.DataFrame:
    """
    Download OMXS30 daily close prices from Yahoo Finance, Jan 2020 – Feb 2026.

    OMXS30 is the Stockholm stock exchange headline index (30 largest caps).
    We use it as the "stock market" series in the Swedish scary chart,
    analogous to the S&P 500 in the US version.
    """
    print("Fetching OMXS30...")

    import yfinance as yf

    ticker = yf.Ticker("^OMX")
    df = ticker.history(start="2020-01-01", end="2026-03-01")
    if df.empty:
        raise ValueError("yfinance returned no data for ^OMX")

    df.index = df.index.tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "omxs30_close"})

    out = RAW / "omxs30_daily.csv"
    df.to_csv(out)
    print(f"  Saved {len(df)} daily prices → {out.name}")
    return df


def process_omxs30() -> pd.DataFrame:
    """
    Resample OMXS30 to monthly averages, index to 100 at base month.

    Why monthly averages (not end-of-month): reduces noise from daily
    volatility, giving a smoother comparison with monthly postings data.
    """
    print("Processing OMXS30 to monthly index...")

    df = pd.read_csv(RAW / "omxs30_daily.csv", index_col=0, parse_dates=True)
    monthly = df.resample("MS").mean()
    monthly.columns = ["omxs30"]

    base = monthly.loc[BASE_MONTH, "omxs30"]
    monthly["omxs30_idx"] = (monthly["omxs30"] / base) * 100

    out = PROCESSED / "omxs30_monthly.csv"
    monthly.to_csv(out)
    print(f"  {len(monthly)} months, base (Feb 2020) = {base:.1f}")
    return monthly


# ── OMXSPI (All-Share) ────────────────────────────────────────────────────────

def fetch_omxspi() -> pd.DataFrame:
    """
    Download OMXSPI (OMX Stockholm All-Share Price Index) from Yahoo Finance.

    Covers all companies listed on Nasdaq Stockholm, not just the 30 largest.
    Used as a robustness check on the scary chart — ensures the divergence
    isn't driven by the composition of the OMXS30 (heavy in banks/industrials).
    """
    print("Fetching OMXSPI (All-Share)...")

    import yfinance as yf

    ticker = yf.Ticker("^OMXSPI")
    df = ticker.history(start="2020-01-01", end="2026-03-01")
    if df.empty:
        raise ValueError("yfinance returned no data for ^OMXSPI")

    df.index = df.index.tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "omxspi_close"})

    out = RAW / "omxspi_daily.csv"
    df.to_csv(out)
    print(f"  Saved {len(df)} daily prices → {out.name}")
    return df


def process_omxspi() -> pd.DataFrame:
    """Resample OMXSPI to monthly averages, index to 100 at base month."""
    print("Processing OMXSPI to monthly index...")

    df = pd.read_csv(RAW / "omxspi_daily.csv", index_col=0, parse_dates=True)
    monthly = df.resample("MS").mean()
    monthly.columns = ["omxspi"]

    base = monthly.loc[BASE_MONTH, "omxspi"]
    monthly["omxspi_idx"] = (monthly["omxspi"] / base) * 100

    out = PROCESSED / "omxspi_monthly.csv"
    monthly.to_csv(out)
    print(f"  {len(monthly)} months, base (Feb 2020) = {base:.1f}")
    return monthly


# ── Riksbanken policy rate ────────────────────────────────────────────────────

def create_riksbank_rate() -> pd.DataFrame:
    """
    Riksbanken policy rate changes, manually verified from riksbank.se.

    The timing of the first hike (April 2022) is central to our identification:
    postings began declining 7 months before ChatGPT launched, coinciding
    with the start of the tightening cycle.
    """
    print("Creating Riksbanken policy rate timeline...")

    changes = [
        ("2020-01-01", 0.00),
        ("2022-04-28", 0.25),   # first hike — key date for our analysis
        ("2022-06-30", 0.75),
        ("2022-09-20", 1.75),
        ("2022-11-24", 2.50),
        ("2023-02-09", 3.00),
        ("2023-04-26", 3.50),
        ("2023-06-29", 3.75),
        ("2023-09-21", 4.00),   # peak
        ("2024-05-08", 3.75),   # first cut
        ("2024-06-27", 3.75),
        ("2024-08-20", 3.50),
        ("2024-09-25", 3.25),
        ("2024-11-07", 2.75),
        ("2024-12-19", 2.50),
        ("2025-01-30", 2.25),   # current as of Feb 2026
    ]

    df = pd.DataFrame(changes, columns=["date", "rate_pct"])
    df["date"] = pd.to_datetime(df["date"])

    out = RAW / "riksbank_rate.csv"
    df.to_csv(out, index=False)
    print(f"  {len(df)} rate changes → {out.name}")

    # Forward-fill to monthly
    monthly_dates = pd.date_range("2020-01-01", "2026-02-01", freq="MS")
    monthly = df.set_index("date").reindex(monthly_dates).ffill()
    monthly.index.name = "date"

    out2 = PROCESSED / "riksbank_monthly.csv"
    monthly.to_csv(out2)
    print(f"  Monthly series: {len(monthly)} months → {out2.name}")

    return monthly


# ── Indeed US job postings ────────────────────────────────────────────────────

def fetch_indeed_us() -> pd.DataFrame:
    """
    Download US aggregate job postings index from Indeed Hiring Lab.

    Used in the online appendix to compare Swedish and US trends.
    Indeed publishes daily data indexed to Feb 1, 2020 = 100.
    """
    print("Fetching Indeed US job postings index...")

    url = (
        "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/"
        "master/US/aggregate_job_postings_US.csv"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    out = RAW / "indeed_us_aggregate.csv"
    out.write_text(resp.text, encoding="utf-8")

    df = pd.read_csv(out)
    print(f"  {len(df)} rows → {out.name}")
    return df


# ── S&P 500 (for US comparison) ──────────────────────────────────────────────

def fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 daily prices for the US scary chart comparison."""
    print("Fetching S&P 500...")

    import yfinance as yf

    ticker = yf.Ticker("^GSPC")
    df = ticker.history(start="2020-01-01", end="2026-03-01")
    if df.empty:
        raise ValueError("yfinance returned no data for ^GSPC")

    df.index = df.index.tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "sp500_close"})

    out = RAW / "sp500_daily.csv"
    df.to_csv(out)
    print(f"  {len(df)} daily prices → {out.name}")
    return df


# ── DAIOE genAI exposure ─────────────────────────────────────────────────────

def copy_daioe() -> pd.DataFrame:
    """
    Copy DAIOE data to project raw/ directory.

    DAIOE (Dynamic AI Occupational Exposure) is our measure of how
    exposed each SSYK 4-digit occupation is to generative AI. It is
    publicly available and provides a percentile ranking (pctl_rank_genai)
    on a 0–100 scale.

    The ssyk2012_4 column contains codes like "0110 Officerare" — we
    extract the first 4 characters as the numeric SSYK code.
    """
    print("Copying DAIOE data...")

    dest = RAW / "daioe_ssyk2012.csv"

    if DAIOE_PATH.exists():
        shutil.copy2(DAIOE_PATH, dest)
        print(f"  Copied from {DAIOE_PATH.name}")
    elif DAIOE_XLSX.exists():
        print(f"  CSV not found; reading from Excel: {DAIOE_XLSX.name}")
        df = pd.read_excel(DAIOE_XLSX)
        df.to_csv(dest, sep="\t", index=False)
    else:
        raise FileNotFoundError(
            f"DAIOE not found at {DAIOE_PATH} or {DAIOE_XLSX}.\n"
            "Download from the DAIOE public repository."
        )

    # Verify the data
    df = pd.read_csv(dest, sep="\t")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    n_occ = df["ssyk2012_4"].str[:4].nunique()
    print(f"  Unique SSYK4 codes: {n_occ}")

    return df


# ── Validation data from SvD project ─────────────────────────────────────────

def copy_validation_data():
    """
    Copy AF published statistics from the SvD project for validation.

    We compare our Platsbanken microdata aggregates against AF's official
    published figures to verify data quality.
    """
    print("Copying validation data from SvD project...")

    # Monthly index (AF aggregates)
    svd_monthly = SVD_PROJECT / "data" / "processed" / "monthly_index.csv"
    if svd_monthly.exists():
        dest = RAW / "af_monthly_index_svd.csv"
        shutil.copy2(svd_monthly, dest)
        print(f"  Copied AF monthly index → {dest.name}")
    else:
        print(f"  WARNING: SvD monthly index not found at {svd_monthly}")

    # AF occupation data
    svd_occ = SVD_PROJECT / "data" / "processed" / "af_occupation_changes.csv"
    if svd_occ.exists():
        dest = RAW / "af_occupation_changes_svd.csv"
        shutil.copy2(svd_occ, dest)
        print(f"  Copied AF occupation data → {dest.name}")
    else:
        print(f"  WARNING: SvD occupation data not found at {svd_occ}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 3: Fetch auxiliary data")
    print("=" * 70)

    # OMXS30
    fetch_omxs30()
    process_omxs30()

    # OMXSPI (All-Share, for robustness)
    try:
        fetch_omxspi()
        process_omxspi()
    except Exception as e:
        print(f"  WARNING: OMXSPI fetch failed: {e}")

    # Riksbanken
    create_riksbank_rate()

    # Indeed US (for appendix comparison)
    try:
        fetch_indeed_us()
    except Exception as e:
        print(f"  WARNING: Indeed US fetch failed: {e}")

    # S&P 500 (for appendix comparison)
    try:
        fetch_sp500()
    except Exception as e:
        print(f"  WARNING: S&P 500 fetch failed: {e}")

    # DAIOE
    copy_daioe()

    # Validation data
    copy_validation_data()

    print("\nDone. Run 04_merge_and_classify.py next.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
l11_refresh_stock.py -- refresh the Stockholm indices to the present.

The submitted figure's stock series was fetched on 24 February 2026 and
hard-stops there (src/03_fetch_auxiliary.py has end="2026-03-01" written
into it). The posting series now runs to June 2026, so the two halves of
Figure 1 were seven months apart. This refetches both indices and rewrites
the same four files, so every downstream script picks the new data up
without changing a path.

    python l11_refresh_stock.py

Writes data/raw/omxs30_daily.csv, data/processed/omxs30_monthly.csv and the
OMXSPI pair. Schema is unchanged: Date, <index>_close for the daily files,
Date, <index>, <index>_idx for the monthly ones, indexed to 100 at the
February 2020 mean, which is the base the paper states.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BASE_MONTH = "2020-02"
TICKERS = {"omxs30": "^OMX", "omxspi": "^OMXSPI"}


def refresh(name: str, ticker: str) -> pd.DataFrame:
    print(f"Fetching {name} ({ticker})...")
    df = yf.Ticker(ticker).history(start="2020-01-01", auto_adjust=False)
    if df.empty:
        raise SystemExit(f"yfinance returned nothing for {ticker}")
    df = df[["Close"]].rename(columns={"Close": f"{name}_close"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df.to_csv(RAW / f"{name}_daily.csv")

    monthly = df.resample("MS").mean()
    base = monthly.loc[f"{BASE_MONTH}-01", f"{name}_close"]
    out = pd.DataFrame({name: monthly[f"{name}_close"],
                        f"{name}_idx": 100 * monthly[f"{name}_close"] / base})
    # Drop the running month: a part-month average is not comparable with
    # the full months beside it, and it is the same artefact we cut the
    # posting series for.
    today = pd.Timestamp.today().normalize()
    out = out[out.index < today.replace(day=1)]
    out.index.name = "Date"
    out.to_csv(PROCESSED / f"{name}_monthly.csv")
    last = out.index[-1].strftime("%Y-%m")
    print(f"  {name}: daily to {df.index[-1].date()}, monthly to {last}, "
          f"index {out[f'{name}_idx'].iloc[-1]:.1f} "
          f"(peak {out[f'{name}_idx'].max():.1f} in "
          f"{out[f'{name}_idx'].idxmax():%Y-%m})")
    return out


if __name__ == "__main__":
    for n, t in TICKERS.items():
        refresh(n, t)

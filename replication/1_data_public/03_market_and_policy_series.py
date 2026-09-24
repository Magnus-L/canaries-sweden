#!/usr/bin/env python3
"""
03_market_and_policy_series.py: the stock indices and the policy rate.

WHAT IT BUILDS
  Stock indices. The OMX Stockholm 30 (^OMX) and the OMX Stockholm All-Share
  (^OMXSPI), daily closes from Yahoo Finance, averaged by calendar month and
  indexed to 100 at the February 2020 mean. The month in which the daily
  series was fetched is dropped, since a part-month average is not
  comparable with the full months beside it. The package ships the daily
  closes fetched on 18 September 2026 (data/raw/omxs30_daily.csv,
  omxspi_daily.csv), so the monthly files run to August 2026; --refresh
  fetches them again and then runs to the last complete month.
  Riksbank policy rate. The decisions from January 2020 to January 2025, as
  announced at riksbank.se, as a monthly series of the rate in force at the
  end of each month, January 2020 to February 2026.
  The S&P 500 daily closes (data/raw/sp500_daily.csv, fetched 24 February
  2026) and the Indeed Hiring Lab US postings index
  (data/raw/indeed_us_aggregate.csv, same date) are read as shipped by the
  figure of Online Appendix II.1; --refresh fetches them again.

OUTPUTS  data/processed/omxs30_monthly.csv, omxspi_monthly.csv,
         riksbank_rate.csv, riksbank_monthly.csv
SERVES   Figure 1 (upper panel) and Online Appendix Figure A1, panels (a),
         (b) and (d)
RUNTIME  seconds (plus the fetch with --refresh)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import pandas as pd  # noqa: E402

BASE_MONTH = "2020-02"
TICKERS = {"omxs30": "^OMX", "omxspi": "^OMXSPI"}
# The shipped daily files were fetched on this date; the month it falls in is
# incomplete and is dropped from the monthly series.
FETCHED = pd.Timestamp("2026-09-18")


def fetch_index(name: str, ticker: str) -> pd.Timestamp:
    """Refetch one index from Yahoo Finance into data/raw/<name>_daily.csv."""
    import yfinance as yf
    print(f"  fetching {name} ({ticker})")
    df = yf.Ticker(ticker).history(start="2020-01-01", auto_adjust=False)
    if df.empty:
        raise SystemExit(f"yfinance returned nothing for {ticker}")
    df = df[["Close"]].rename(columns={"Close": f"{name}_close"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df.to_csv(config.RAW / f"{name}_daily.csv")
    return pd.Timestamp.today().normalize()


def monthly_index(name: str, fetched: pd.Timestamp) -> pd.DataFrame:
    """Monthly mean of daily closes, indexed to the February 2020 mean."""
    df = pd.read_csv(config.RAW / f"{name}_daily.csv", index_col=0, parse_dates=True)
    monthly = df.resample("MS").mean()
    base = monthly.loc[f"{BASE_MONTH}-01", f"{name}_close"]
    out = pd.DataFrame({name: monthly[f"{name}_close"],
                        f"{name}_idx": 100 * monthly[f"{name}_close"] / base})
    out = out[out.index < fetched.replace(day=1)]
    out.index.name = "Date"
    out.to_csv(config.PROCESSED / f"{name}_monthly.csv")
    print(f"  {name}: monthly to {out.index[-1]:%Y-%m}, "
          f"last index {out[f'{name}_idx'].iloc[-1]:.1f}")
    return out


def riksbank_rate() -> None:
    """Policy-rate decisions, as announced at riksbank.se, to a monthly series."""
    changes = [
        ("2020-01-01", 0.00),
        ("2022-04-28", 0.25),   # the first rise, the date the design turns on
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
        ("2025-01-30", 2.25),
    ]
    df = pd.DataFrame(changes, columns=["date", "rate_pct"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(config.PROCESSED / "riksbank_rate.csv", index=False)
    # The rate in force on the last day of each month: a decision taken on
    # 28 April 2022 sets April's value.
    daily = (df.set_index("date")
             .reindex(pd.date_range("2020-01-01", "2026-02-28", freq="D")).ffill())
    monthly = daily.resample("MS").last()
    monthly.index.name = "date"
    monthly.to_csv(config.PROCESSED / "riksbank_monthly.csv")
    print(f"  policy rate: {len(df)} decisions, {len(monthly)} months")


def refresh_us() -> None:
    """Refetch the two US series of Figure A1, panel (a)."""
    import requests
    import yfinance as yf
    url = ("https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/"
           "master/US/aggregate_job_postings_US.csv")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    (config.RAW / "indeed_us_aggregate.csv").write_text(r.text, encoding="utf-8")
    df = yf.Ticker("^GSPC").history(start="2020-01-01", end="2026-03-01")
    df.index = df.index.tz_localize(None)
    df[["Close"]].rename(columns={"Close": "sp500_close"}).to_csv(
        config.RAW / "sp500_daily.csv")
    print("  refreshed indeed_us_aggregate.csv and sp500_daily.csv")


def main():
    refresh = "--refresh" in sys.argv
    print("Stock indices and the policy rate")
    for name, ticker in TICKERS.items():
        fetched = fetch_index(name, ticker) if refresh else FETCHED
        monthly_index(name, fetched)
    riksbank_rate()
    if refresh:
        refresh_us()


if __name__ == "__main__":
    main()

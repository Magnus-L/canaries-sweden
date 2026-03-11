#!/usr/bin/env python3
"""
15_stock_market_comparison.py — Compare OMXS30 with Nasdaq and S&P 500.

Motivation: The US "scary chart" implies AI drives the stock-posting divergence.
But the Swedish OMXS30 is dominated by industrials and banks, not AI firms.
This figure shows OMXS30 substantially underperforms the Nasdaq Composite
(tech/AI-heavy), confirming that the Swedish stock rise reflects different
forces than US AI-driven equity revaluation.

Output: figures/figA12_stock_market_comparison.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, FIGDIR, BASE_MONTH,
    DARK_BLUE, ORANGE, TEAL, GRAY, DARK_TEXT,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
    set_rcparams,
)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

set_rcparams()


# ── Fetch / load helpers ─────────────────────────────────────────────────────

def fetch_index(ticker_symbol: str, name: str) -> pd.DataFrame:
    """
    Download daily close prices from Yahoo Finance and save to raw/.

    Uses the same pattern as 03_fetch_auxiliary.py for consistency.
    yfinance returns daily OHLCV data; we keep only Close.
    """
    import yfinance as yf

    print(f"Fetching {name} ({ticker_symbol})...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start="2020-01-01", end="2026-03-01")
    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker_symbol}")

    # yfinance returns timezone-aware index; strip for consistency
    df.index = df.index.tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": f"{name}_close"})

    out = RAW / f"{name}_daily.csv"
    df.to_csv(out)
    print(f"  Saved {len(df)} daily prices → {out.name}")
    return df


def to_monthly_index(daily_csv: Path, col_name: str) -> pd.Series:
    """
    Resample daily prices to monthly averages, index to 100 at BASE_MONTH.

    Monthly averages (not end-of-month) reduce daily volatility noise,
    giving a smoother comparison that matches the monthly posting data.
    """
    df = pd.read_csv(daily_csv, index_col=0, parse_dates=True)
    col = df.columns[0]
    monthly = df[col].resample("MS").mean()
    base = monthly.loc[BASE_MONTH]
    indexed = (monthly / base) * 100
    indexed.name = col_name
    return indexed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 15: Stock market comparison (OMXS30 vs Nasdaq vs S&P 500)")
    print("=" * 70)

    # Fetch Nasdaq Composite if not already present
    nasdaq_raw = RAW / "nasdaq_daily.csv"
    if not nasdaq_raw.exists():
        fetch_index("^IXIC", "nasdaq")
    else:
        print(f"Using cached {nasdaq_raw.name}")

    # Fetch S&P 500 if not already present
    sp500_raw = RAW / "sp500_daily.csv"
    if not sp500_raw.exists():
        fetch_index("^GSPC", "sp500")
    else:
        print(f"Using cached {sp500_raw.name}")

    # OMXS30 should already exist from 03_fetch_auxiliary.py
    omxs30_raw = RAW / "omxs30_daily.csv"
    if not omxs30_raw.exists():
        fetch_index("^OMX", "omxs30")

    # Build monthly indexed series
    omxs30 = to_monthly_index(omxs30_raw, "OMXS30")
    sp500 = to_monthly_index(sp500_raw, "S&P 500")
    nasdaq = to_monthly_index(nasdaq_raw, "Nasdaq Composite")

    # Merge into a single DataFrame for easy plotting
    combined = pd.concat([omxs30, sp500, nasdaq], axis=1).dropna()
    print(f"\n  Monthly data: {combined.index[0].strftime('%b %Y')} – "
          f"{combined.index[-1].strftime('%b %Y')} ({len(combined)} months)")
    print(f"  Latest values: OMXS30={combined['OMXS30'].iloc[-1]:.0f}, "
          f"S&P 500={combined['S&P 500'].iloc[-1]:.0f}, "
          f"Nasdaq={combined['Nasdaq Composite'].iloc[-1]:.0f}")

    # Save processed data
    out_csv = PROCESSED / "stock_comparison_monthly.csv"
    combined.to_csv(out_csv)
    print(f"  Saved → {out_csv.name}")

    # ── Plot ──────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Nasdaq: the AI-heavy benchmark (most prominent)
    ax.plot(combined.index, combined["Nasdaq Composite"],
            color=ORANGE, linewidth=2.2, alpha=0.9,
            label="Nasdaq Composite (US, tech-heavy)")

    # S&P 500: broad US market (middle ground)
    ax.plot(combined.index, combined["S&P 500"],
            color=TEAL, linewidth=2.0, alpha=0.85,
            label="S&P 500 (US, broad market)")

    # OMXS30: Swedish headline index
    ax.plot(combined.index, combined["OMXS30"],
            color=DARK_BLUE, linewidth=2.2, alpha=0.9,
            label="OMXS30 (Sweden, industrials/banks)")

    # Vertical event markers (same style as other figures)
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)

    ax.axvline(rb_date, color=TEAL, linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(gpt_date, color=GRAY, linestyle=":", linewidth=1, alpha=0.6)

    # Labels for vertical lines (positioned just above the data range)
    ymax = combined.max().max()
    ax.text(rb_date, ymax * 1.02, "Riksbank\nrate hike",
            ha="center", va="bottom", fontsize=8, color=TEAL, alpha=0.8)
    ax.text(gpt_date, ymax * 1.02, "ChatGPT\nlaunch",
            ha="center", va="bottom", fontsize=8, color=GRAY, alpha=0.8)

    # Reference line at 100
    ax.axhline(100, color=GRAY, linewidth=0.5, alpha=0.4)

    # Formatting
    ax.set_ylabel("Index (Feb 2020 = 100)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax.legend(loc="upper left", frameon=False)

    # Annotate the gap at the end of the series
    last = combined.iloc[-1]
    gap = last["Nasdaq Composite"] - last["OMXS30"]
    ax.annotate(
        f"{gap:.0f} pp gap",
        xy=(combined.index[-1], (last["Nasdaq Composite"] + last["OMXS30"]) / 2),
        xytext=(15, 0), textcoords="offset points",
        fontsize=9, color=DARK_TEXT, alpha=0.7,
        arrowprops=dict(arrowstyle="-", color=GRAY, alpha=0.4),
    )

    # Source note (consistent with other figures)
    ax.text(0.01, -0.08,
            "Source: Yahoo Finance. Monthly averages, indexed to Feb 2020 = 100.",
            transform=ax.transAxes, fontsize=7, fontstyle="italic", color=GRAY)

    # Save
    outpath = FIGDIR / "figA12_stock_market_comparison.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"\n  Figure saved → {outpath.name}")
    print("Done.")


if __name__ == "__main__":
    main()

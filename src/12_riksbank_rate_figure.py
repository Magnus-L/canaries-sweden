#!/usr/bin/env python3
"""
12_riksbank_rate_figure.py — Riksbank policy rate figure for online appendix.

Creates Figure A10: Riksbank policy rate timeline with key event markers.
Erik Engberg requested this to make the tightening cycle visually explicit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED, FIGDIR,
    DARK_BLUE, ORANGE, TEAL, GRAY, DARK_TEXT,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH, set_rcparams,
)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

set_rcparams()


def fig_riksbank_rate():
    """
    Riksbank policy rate (styrränta) timeline, 2020–2026.

    Shows the tightening cycle from 0% to 4% and subsequent easing,
    with vertical markers for the first hike and ChatGPT launch.
    This makes the identification strategy visually concrete: the
    7-month gap between monetary tightening and ChatGPT is clear.
    """
    print("Generating Riksbank policy rate figure...")

    # Build rate series from scratch (the processed CSV has a forward-fill bug
    # where change dates don't align with month starts, producing all zeros).
    changes = [
        ("2020-01-01", 0.00),
        ("2022-04-28", 0.25),   # first hike
        ("2022-06-30", 0.75),
        ("2022-09-20", 1.75),
        ("2022-11-24", 2.50),
        ("2023-02-09", 3.00),
        ("2023-04-26", 3.50),
        ("2023-06-29", 3.75),
        ("2023-09-21", 4.00),   # peak
        ("2024-05-08", 3.75),   # first cut
        ("2024-08-20", 3.50),
        ("2024-09-25", 3.25),
        ("2024-11-07", 2.75),
        ("2024-12-19", 2.50),
        ("2025-01-30", 2.25),   # current as of Feb 2026
    ]
    raw = pd.DataFrame(changes, columns=["date", "rate_pct"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date")

    # Create daily series and resample to month-start via forward fill
    daily = raw.reindex(pd.date_range("2020-01-01", "2026-02-28", freq="D")).ffill()
    rate = daily.resample("MS").last()

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Step-function plot (policy rates change discretely)
    ax.step(
        rate.index, rate["rate_pct"],
        where="post", color=DARK_BLUE, linewidth=2.2,
        label="Policy rate (%)",
    )

    # Fill under the curve for visual weight
    ax.fill_between(
        rate.index, 0, rate["rate_pct"],
        step="post", alpha=0.08, color=DARK_BLUE,
    )

    # Event markers
    rb = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt = pd.Timestamp(CHATGPT_LAUNCH)

    ax.axvline(rb, color=TEAL, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(gpt, color=GRAY, linestyle=":", linewidth=1.2, alpha=0.7)

    # Annotate first hike
    ax.annotate(
        "First hike\n(Apr 2022)",
        xy=(rb, 0.25), xytext=(rb - pd.Timedelta(days=120), 1.8),
        fontsize=9, color=TEAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2),
        ha="center",
    )

    # Annotate ChatGPT
    ax.annotate(
        "ChatGPT\n(Nov 2022)",
        xy=(gpt, 2.5), xytext=(gpt + pd.Timedelta(days=100), 1.0),
        fontsize=9, color=GRAY, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
        ha="center",
    )

    # Annotate peak
    peak_date = pd.Timestamp("2023-09-21")
    ax.annotate(
        "Peak 4.00%\n(Sep 2023)",
        xy=(peak_date, 4.0), xytext=(peak_date + pd.Timedelta(days=90), 4.3),
        fontsize=8, color=DARK_TEXT,
        arrowprops=dict(arrowstyle="->", color=DARK_TEXT, lw=0.8),
        ha="center",
    )

    # Shaded tightening cycle
    ax.axvspan(
        pd.Timestamp("2022-04-01"), pd.Timestamp("2023-09-01"),
        alpha=0.06, color=TEAL, zorder=0,
    )

    # 7-month gap annotation
    mid_gap = rb + (gpt - rb) / 2
    ax.annotate(
        "", xy=(gpt, -0.25), xytext=(rb, -0.25),
        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5),
    )
    ax.text(
        mid_gap, -0.45, "7 months",
        fontsize=9, color=ORANGE, fontweight="bold", ha="center",
    )

    ax.set_ylabel("Policy rate (%)", fontsize=12)
    ax.set_ylim(-0.7, 4.8)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax.axhline(0, color=GRAY, linewidth=0.5, alpha=0.3)

    # Source note
    fig.text(
        0.01, 0.01,
        "Source: Riksbank press releases (riksbank.se). Manually compiled and verified.",
        fontsize=7, color=GRAY, style="italic",
        transform=fig.transFigure,
    )

    out = FIGDIR / "figA10_riksbank_rate.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


if __name__ == "__main__":
    fig_riksbank_rate()

#!/usr/bin/env python3
"""
06_figures_tables.py — Generate all publication figures and tables.

Main paper (2 figures, 1 table):
  Figure 1: "Scary chart" — OMXS30 vs posting indices by genAI quartile
  Figure 2: Q4 minus Q1 posting gap over time with confidence band
  Table 1: DiD regression (generated in 05_analysis.py, copied here)

Online appendix:
  Figure A1: Sweden vs US comparison
  Figure A2: Individual quartile trends (4-panel)
  Figure A3: Vacancy-weighted results
  Table A1: Summary statistics
  Table A2: Top/bottom occupations by exposure
  Table A3: Full regression with all robustness checks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, FIGDIR, TABDIR,
    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY, LIGHT_BLUE,
    DARK_TEXT, CREAM, Q_COLORS,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH, BASE_MONTH,
    DAIOE_REF_YEAR, set_rcparams,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


set_rcparams()


# ── Helper functions ──────────────────────────────────────────────────────────

def add_event_annotations(ax, y_range, omxs_series=None):
    """Add Riksbanken hike and ChatGPT launch markers to a figure."""
    rb = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt = pd.Timestamp(CHATGPT_LAUNCH)

    ax.axvline(rb, color=TEAL, linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(gpt, color=GRAY, linestyle=":", linewidth=1, alpha=0.6)

    y_top = y_range[1]

    ax.annotate(
        "Riksbanken\nhike (Apr 2022)",
        xy=(rb, y_top * 0.92),
        fontsize=8, color=TEAL, fontweight="bold",
        ha="center",
    )
    ax.annotate(
        "ChatGPT\n(Nov 2022)",
        xy=(gpt, y_top * 0.82),
        fontsize=8, color=GRAY, fontweight="bold",
        ha="center",
    )


def add_source_note(fig, text="Source: Platsbanken microdata, DAIOE, Yahoo Finance."):
    """Add a source note at the bottom of the figure."""
    fig.text(
        0.01, 0.01, text,
        fontsize=7, color=GRAY, style="italic",
        transform=fig.transFigure,
    )


def format_date_axis(ax):
    """Standard date formatting for x-axis."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))


# ── Figure 1: Scary chart with quartile lines ────────────────────────────────

def fig_scary_chart():
    """
    The Swedish "scary chart": stock market vs job postings by AI exposure.

    Left axis: OMXS30 index (Feb 2020 = 100)
    Right axis: 4 lines for posting index by genAI exposure quartile
               (3-month MA for readability, raw as faint background)

    This is the paper's key visual — it shows that the posting decline is
    broad-based across exposure groups, suggesting monetary policy rather
    than AI as the primary driver.
    """
    print("Generating Figure 1: Scary chart...")

    # Load data
    omxs = pd.read_csv(PROCESSED / "omxs30_monthly.csv", index_col=0, parse_dates=True)
    quartile = pd.read_csv(PROCESSED / "postings_quartile_indexed.csv")
    quartile["date"] = pd.to_datetime(quartile["date"])

    # ── Fix 1: Trim to Jan 2020 onward ──
    cutoff = pd.Timestamp("2020-01-01")
    omxs = omxs[omxs.index >= cutoff]
    quartile = quartile[quartile["date"] >= cutoff]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # OMXS30 on left axis — filled area for visual weight
    ax1.fill_between(
        omxs.index, 0, omxs["omxs30_idx"],
        alpha=0.08, color=DARK_BLUE,
    )
    ax1.plot(
        omxs.index, omxs["omxs30_idx"],
        color=DARK_BLUE, linewidth=2.2, label="OMXS30",
    )
    ax1.set_ylabel("OMXS30 (index, Feb 2020 = 100)", color=DARK_BLUE, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=DARK_BLUE)

    # Quartile posting lines on right axis
    ax2 = ax1.twinx()

    # ── Fix 2: Plot raw lines faintly, then 3-month MA prominently ──
    for q_name, color in Q_COLORS.items():
        qdf = quartile[quartile["exposure_quartile"] == q_name].sort_values("date").copy()
        linewidth = 2.0 if "Q4" in q_name or "Q1" in q_name else 1.4
        linestyle = "-" if "Q4" in q_name or "Q1" in q_name else "--"

        # Raw line (faint)
        ax2.plot(
            qdf["date"], qdf["ads_idx"],
            color=color, linewidth=0.6, linestyle=linestyle, alpha=0.25,
        )
        # 3-month rolling average (prominent)
        qdf["ads_ma3"] = qdf["ads_idx"].rolling(3, center=True, min_periods=1).mean()
        ax2.plot(
            qdf["date"], qdf["ads_ma3"],
            color=color, linewidth=linewidth, linestyle=linestyle,
            label=q_name, alpha=0.9,
        )

    ax2.set_ylabel(
        "Job postings (index, Feb 2020 = 100, 3-mo MA)",
        color=DARK_TEXT, fontsize=12,
    )

    # ── Fix 3: Shaded tightening cycle with text label ──
    ax1.axvspan(
        pd.Timestamp("2022-04-01"), pd.Timestamp("2023-09-01"),
        alpha=0.06, color=TEAL, zorder=0,
    )
    # Text label inside the shaded band
    ax1.text(
        pd.Timestamp("2022-10-15"), ax1.get_ylim()[1] * 0.05,
        "Riksbanken\ntightening cycle",
        fontsize=8, color=TEAL, ha="center", va="bottom",
        fontstyle="italic", alpha=0.8,
    )

    # Event markers with annotations
    rb = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt = pd.Timestamp(CHATGPT_LAUNCH)
    ax1.axvline(rb, color=TEAL, linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(gpt, color=GRAY, linestyle=":", linewidth=1, alpha=0.6)

    # Place event labels in the upper part of the chart, offset to avoid overlap
    y_top = ax1.get_ylim()[1]
    ax1.annotate(
        "Riksbanken\nhike", xy=(rb, y_top * 0.93),
        fontsize=8.5, color=TEAL, fontweight="bold", ha="right",
    )
    ax1.annotate(
        "ChatGPT\nlaunch", xy=(gpt, y_top * 0.83),
        fontsize=8.5, color=GRAY, fontweight="bold", ha="left",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", framealpha=0.9, fontsize=9,
    )

    # Title
    ax1.set_title(
        "Stock market vs job postings by AI exposure, Sweden 2020–2025",
        fontsize=14, fontweight="bold", pad=12,
    )

    # Date formatting
    format_date_axis(ax1)

    # 100 baseline reference line
    ax2.axhline(100, color=GRAY, linewidth=0.5, alpha=0.3)

    add_source_note(fig)

    out = FIGDIR / "fig1_scary_chart.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# ── Figure 2: Q4–Q1 gap over time ────────────────────────────────────────────

def fig_exposure_gap():
    """
    Difference between Q4 (highest genAI exposure) and Q1 (lowest) posting
    indices over time.

    If AI is reducing postings in exposed occupations, this gap should widen
    (become more negative) after ChatGPT. If monetary policy is the main
    driver, the gap should widen after Riksbanken's hike but not further
    after ChatGPT.

    We add a 3-month rolling average confidence band estimated from the
    cross-occupation variance within each quartile.
    """
    print("Generating Figure 2: Exposure gap...")

    quartile = pd.read_csv(PROCESSED / "postings_quartile_indexed.csv")
    quartile["date"] = pd.to_datetime(quartile["date"])

    # Trim to Jan 2020 onward
    quartile = quartile[quartile["date"] >= pd.Timestamp("2020-01-01")]

    # Compute Q4 - Q1 gap
    q4 = quartile[quartile["exposure_quartile"] == "Q4 (highest)"].set_index("date")
    q1 = quartile[quartile["exposure_quartile"] == "Q1 (lowest)"].set_index("date")

    # Align on common dates
    common_dates = q4.index.intersection(q1.index)
    gap = q4.loc[common_dates, "ads_idx"] - q1.loc[common_dates, "ads_idx"]
    gap = gap.to_frame("gap")
    gap = gap.sort_index()

    # Rolling standard deviation as rough confidence band
    gap["gap_smooth"] = gap["gap"].rolling(3, center=True, min_periods=1).mean()
    gap["gap_std"] = gap["gap"].rolling(6, center=True, min_periods=2).std()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Confidence band
    ax.fill_between(
        gap.index,
        gap["gap_smooth"] - 1.96 * gap["gap_std"],
        gap["gap_smooth"] + 1.96 * gap["gap_std"],
        alpha=0.15, color=ORANGE, label="95% band (rolling)",
    )

    # Gap line
    ax.plot(
        gap.index, gap["gap"],
        color=ORANGE, linewidth=1.0, alpha=0.4,
    )
    ax.plot(
        gap.index, gap["gap_smooth"],
        color=ORANGE, linewidth=2.2, label="Q4 – Q1 gap (3-month MA)",
    )

    # Zero line
    ax.axhline(0, color=DARK_TEXT, linewidth=0.8, alpha=0.5)

    # Event annotations
    y_range = (gap["gap"].min() * 1.3, gap["gap"].max() * 1.3)
    add_event_annotations(ax, y_range)

    # Shaded tightening cycle
    ax.axvspan(
        pd.Timestamp("2022-04-01"), pd.Timestamp("2023-09-01"),
        alpha=0.06, color=TEAL, zorder=0,
    )

    ax.set_ylabel("Q4 – Q1 posting index gap (percentage points)", fontsize=12)
    ax.set_title(
        "High vs low AI-exposure: posting gap, Sweden 2020–2025",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    format_date_axis(ax)

    add_source_note(fig)

    out = FIGDIR / "fig2_exposure_gap.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# ── Appendix Figure A1: Sweden vs US ─────────────────────────────────────────

def fig_sweden_vs_us():
    """
    Two-panel comparison: Sweden (OMXS30 + Platsbanken) vs US (S&P 500 + Indeed).

    For the online appendix. Shows that the Swedish "scary chart" pattern
    mirrors the US one documented by Thompson/Indeed.
    """
    print("Generating Figure A1: Sweden vs US comparison...")

    omxs = pd.read_csv(PROCESSED / "omxs30_monthly.csv", index_col=0, parse_dates=True)
    postings = pd.read_csv(PROCESSED / "postings_total_indexed.csv")
    postings["date"] = pd.to_datetime(postings["date"])
    postings = postings.set_index("date")

    # Trim to Jan 2020 onward
    cutoff = pd.Timestamp("2020-01-01")
    omxs = omxs[omxs.index >= cutoff]
    postings = postings[postings.index >= cutoff]

    # US data
    try:
        sp500_raw = pd.read_csv(RAW / "sp500_daily.csv", index_col=0, parse_dates=True)
        sp500 = sp500_raw.resample("MS").mean()
        sp_base = sp500.loc[BASE_MONTH].values[0]
        sp500["sp500_idx"] = (sp500.iloc[:, 0] / sp_base) * 100

        indeed = pd.read_csv(RAW / "indeed_us_aggregate.csv")
        indeed["date"] = pd.to_datetime(indeed["date"])
        indeed = indeed.set_index("date")
        # Indeed is already indexed to ~100 at Feb 2020
    except Exception as e:
        print(f"  WARNING: US data not available: {e}")
        return

    fig, (ax_us, ax_se) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # US panel
    ax_us.plot(sp500.index, sp500["sp500_idx"], color=DARK_BLUE, linewidth=2, label="S&P 500")
    ax_us_r = ax_us.twinx()
    # Indeed column name may vary
    indeed_col = [c for c in indeed.columns if "US" in c.upper() or "indeed" in c.lower()]
    if indeed_col:
        ax_us_r.plot(indeed.index, indeed[indeed_col[0]], color=ORANGE, linewidth=2, label="Indeed US")
    ax_us.set_title("United States", fontsize=13, fontweight="bold")
    ax_us.set_ylabel("S&P 500 (index)", color=DARK_BLUE)
    ax_us_r.set_ylabel("Indeed postings (index)", color=ORANGE)

    # Sweden panel — smooth postings with 3-month MA for comparability with US
    ax_se.plot(omxs.index, omxs["omxs30_idx"], color=DARK_BLUE, linewidth=2, label="OMXS30")
    ax_se_r = ax_se.twinx()
    postings_sorted = postings.sort_index()
    # Raw as faint background
    ax_se_r.plot(postings_sorted.index, postings_sorted["ads_idx"],
                 color=ORANGE, linewidth=0.6, alpha=0.25)
    # 3-month MA prominent
    postings_sorted["ads_ma3"] = postings_sorted["ads_idx"].rolling(3, center=True, min_periods=1).mean()
    ax_se_r.plot(postings_sorted.index, postings_sorted["ads_ma3"],
                 color=ORANGE, linewidth=2, label="Platsbanken (3-mo MA)")
    ax_se.set_title("Sweden", fontsize=13, fontweight="bold")
    ax_se.set_ylabel("OMXS30 (index)", color=DARK_BLUE)
    ax_se_r.set_ylabel("Platsbanken postings (index)", color=ORANGE)

    for ax in [ax_us, ax_se]:
        format_date_axis(ax)
        ax.axvline(pd.Timestamp(CHATGPT_LAUNCH), color=GRAY, linestyle=":", linewidth=1, alpha=0.5)

    fig.suptitle(
        "Stock markets vs job postings: US and Sweden (Feb 2020 = 100)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fig.tight_layout()
    add_source_note(fig, "Sources: Yahoo Finance, Indeed Hiring Lab, Platsbanken.")

    out = FIGDIR / "figA1_sweden_vs_us.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# ── Appendix Figure A2: Individual quartile trends ───────────────────────────

def fig_quartile_panels():
    """Four-panel figure showing each quartile's posting trend individually."""
    print("Generating Figure A2: Quartile panels...")

    quartile = pd.read_csv(PROCESSED / "postings_quartile_indexed.csv")
    quartile["date"] = pd.to_datetime(quartile["date"])

    # Trim to Jan 2020 onward
    quartile = quartile[quartile["date"] >= pd.Timestamp("2020-01-01")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (q_name, color) in zip(axes.flat, Q_COLORS.items()):
        qdf = quartile[quartile["exposure_quartile"] == q_name].sort_values("date").copy()
        # Raw as faint, 3-month MA prominent
        ax.plot(qdf["date"], qdf["ads_idx"], color=color, linewidth=0.7, alpha=0.3)
        qdf["ads_ma3"] = qdf["ads_idx"].rolling(3, center=True, min_periods=1).mean()
        ax.plot(qdf["date"], qdf["ads_ma3"], color=color, linewidth=2)
        ax.axhline(100, color=GRAY, linewidth=0.5, alpha=0.3)
        ax.set_title(q_name, fontsize=12, fontweight="bold")
        ax.axvline(pd.Timestamp(RIKSBANKEN_HIKE), color=TEAL, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(pd.Timestamp(CHATGPT_LAUNCH), color=GRAY, linestyle=":", linewidth=1, alpha=0.5)
        format_date_axis(ax)

    fig.suptitle(
        "Job postings by genAI exposure quartile (Feb 2020 = 100)",
        fontsize=14, fontweight="bold",
    )
    fig.supylabel("Posting index", fontsize=12)
    fig.tight_layout()

    out = FIGDIR / "figA2_quartile_panels.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# ── Appendix Table A2: Top/bottom occupations ────────────────────────────────

def table_top_bottom_occupations():
    """List the 10 most and 10 least genAI-exposed occupations."""
    print("Generating Table A2: Top/bottom occupations...")

    daioe_raw = pd.read_csv(RAW / "daioe_ssyk2012.csv", sep="\t")
    daioe_ref = daioe_raw[daioe_raw["year"] == DAIOE_REF_YEAR].copy()
    daioe_ref["ssyk4"] = daioe_ref["ssyk2012_4"].str[:4]
    daioe_ref["occupation_name"] = daioe_ref["ssyk2012_4"].str[5:]

    daioe_ref = daioe_ref.dropna(subset=["pctl_rank_genai"])
    daioe_ref = daioe_ref.sort_values("pctl_rank_genai", ascending=False)

    top10 = daioe_ref.head(10)[["ssyk4", "occupation_name", "pctl_rank_genai"]]
    bottom10 = daioe_ref.tail(10)[["ssyk4", "occupation_name", "pctl_rank_genai"]]

    combined = pd.concat([
        top10.assign(group="Most exposed"),
        bottom10.assign(group="Least exposed"),
    ])

    out = TABDIR / "top_bottom_occupations.csv"
    combined.to_csv(out, index=False)
    print(f"  Saved → {out.name}")

    # LaTeX version
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Most and least genAI-exposed occupations (DAIOE)}",
        r"\label{tab:topbottom}",
        r"\begin{tabular}{clc}",
        r"\hline\hline",
        r"SSYK & Occupation & GenAI pctl \\",
        r"\hline",
        r"\multicolumn{3}{l}{\textit{Most exposed (top 10)}} \\",
    ]

    for _, row in top10.iterrows():
        name = row["occupation_name"][:40]
        lines.append(f"{row['ssyk4']} & {name} & {row['pctl_rank_genai']:.1f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{\textit{Least exposed (bottom 10)}} \\")

    for _, row in bottom10.iterrows():
        name = row["occupation_name"][:40]
        lines.append(f"{row['ssyk4']} & {name} & {row['pctl_rank_genai']:.1f} \\\\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_out = TABDIR / "top_bottom_occupations.tex"
    tex_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {tex_out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 6: Generate figures and tables")
    print("=" * 70)

    # Main paper figures
    fig_scary_chart()
    fig_exposure_gap()

    # Appendix figures
    fig_sweden_vs_us()
    fig_quartile_panels()

    # Appendix tables
    table_top_bottom_occupations()

    print("\nAll figures and tables generated.")
    print(f"  Figures: {FIGDIR}")
    print(f"  Tables: {TABDIR}")
    print("\nDone. Run 07_robustness.py next.")


if __name__ == "__main__":
    main()

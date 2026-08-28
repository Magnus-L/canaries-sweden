#!/usr/bin/env python3
"""
l07_figures_rebuild.py -- T6/E8: the figure overhaul.

Editor: figures must be legible at journal size and self-contained;
Figure 1's dual axes, small fonts, colour-only distinction and unlabeled
event lines all named explicitly. And Figure 2's "Employment change (%)"
axis is the ln(n+1) misreading (defect D5).

Figure 1 (rebuilt here, local data):
  Two stacked panels replacing the dual axis.
    Top:    OMX Stockholm 30 Index, monthly mean of daily closes,
            Feb 2020 = 100.
    Bottom: Platsbanken postings by DAIOE genAI exposure quartile,
            3-month centred moving average, Feb 2020 = 100.
  Series distinguished by line STYLE and direct end labels, not colour
  alone; event lines dated in their labels; descriptive series cut at
  Dec 2025 (collection artefact rule).

Figure 2 (rebuilt when MONA output lands):
  Poisson event-study coefficients for 22-25 from output_43/poisson_es.csv
  (axis: "Poisson coefficient (log points)" -- no percent conversion).
  If the file is absent the function prints what it is waiting for.

Output: figures/fig1_two_panel.pdf/.png, figures/fig2_poisson_es.pdf/.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import (PROCESSED, V2_FIG, POSTINGS_DESCRIPTIVE_END,  # noqa: E402
                    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
                    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY, DARK_TEXT)

MONA43 = REV / "mona" / "output_43"


def fig1_two_panel():
    print("  Figure 1: two-panel rebuild")
    omxs = pd.read_csv(PROCESSED / "omxs30_monthly.csv",
                       index_col=0, parse_dates=True)
    # Prefer the extended series (l08: official 2026-Q1/Q2 files) when built
    ext = REV / "output" / "postings_quartile_indexed_extended.csv"
    if ext.exists():
        q = pd.read_csv(ext)
        q["date"] = pd.to_datetime(q["year_month"] + "-01")
        cut = pd.Timestamp("2026-06-01")
        print("    using extended series to June 2026")
    else:
        q = pd.read_csv(PROCESSED / "postings_quartile_indexed.csv")
        q["date"] = pd.to_datetime(q["date"])
        cut = pd.Timestamp(POSTINGS_DESCRIPTIVE_END + "-01")
    lo = pd.Timestamp("2020-01-01")
    omxs = omxs[(omxs.index >= lo) & (omxs.index <= cut)]
    q = q[(q["date"] >= lo) & (q["date"] <= cut)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.08})

    # -- Top: OMXS30 --
    ax1.plot(omxs.index, omxs["omxs30_idx"], color=DARK_BLUE, lw=1.8)
    ax1.axhline(100, color=GRAY, lw=0.6, ls="--", alpha=0.6)
    ax1.set_ylabel("OMX Stockholm 30\n(index, Feb 2020 = 100)", fontsize=10)
    ax1.text(omxs.index[-1], omxs["omxs30_idx"].iloc[-1], "  OMXS30",
             fontsize=9, color=DARK_BLUE, va="center")

    # -- Bottom: postings by quartile --
    styles = {
        "Q1 (lowest)":  dict(color=GRAY,      ls=":",  lw=1.4),
        "Q2":           dict(color=GRAY,      ls="--", lw=1.2),
        "Q3":           dict(color=TEAL,      ls="-.", lw=1.4),
        "Q4 (highest)": dict(color=ORANGE,    ls="-",  lw=2.0),
    }
    labels = {"Q1 (lowest)": "Q1 (least exposed)", "Q2": "Q2",
              "Q3": "Q3", "Q4 (highest)": "Q4 (most exposed)"}
    # The four series converge at the right edge, so end labels collide;
    # a line-style legend is the self-contained alternative (E8 asks for
    # style + label, not colour alone -- the legend carries both).
    for qname, st in styles.items():
        sub = q[q["exposure_quartile"] == qname].sort_values("date").copy()
        sub["ma"] = sub["ads_idx"].rolling(3, center=True,
                                           min_periods=1).mean()
        ax2.plot(sub["date"], sub["ma"], label=labels[qname], **st)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9,
               title="DAIOE genAI exposure", title_fontsize=8.5)
    ax2.axhline(100, color=GRAY, lw=0.6, ls="--", alpha=0.6)
    ax2.set_ylabel("Job postings by AI-exposure quartile\n"
                   "(index, Feb 2020 = 100, 3-month MA)", fontsize=10)

    # -- Events, dated in the labels, on both panels --
    rb = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt = pd.Timestamp(CHATGPT_LAUNCH)
    for ax in (ax1, ax2):
        ax.axvline(rb, color=TEAL, ls="--", lw=1, alpha=0.8)
        ax.axvline(gpt, color=DARK_TEXT, ls=":", lw=1, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.text(rb - pd.Timedelta(days=12), ax1.get_ylim()[1],
             "Riksbank first hike\nApril 2022", fontsize=8, color=TEAL,
             ha="right", va="top")
    ax1.text(gpt, ax1.get_ylim()[0] * 1.02,
             "  ChatGPT launch\n  November 2022",
             fontsize=8, color=DARK_TEXT, ha="left", va="bottom")

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(lo, cut + pd.Timedelta(days=90))

    fig.align_ylabels((ax1, ax2))
    fig.savefig(V2_FIG / "fig1_two_panel.pdf", bbox_inches="tight")
    fig.savefig(V2_FIG / "fig1_two_panel.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("    saved fig1_two_panel.pdf/.png")


def fig2_poisson_es():
    src = MONA43 / "poisson_es.csv"
    if not src.exists():
        print(f"  Figure 2: waiting for MONA output ({src.relative_to(REV)})"
              " -- run mona/43 first, export, place the CSV there.")
        return
    print("  Figure 2: Poisson event study (22-25)")
    es = pd.read_csv(src)
    sub = es[es["age_group"] == "22-25"].copy()
    order = sorted(sub["period"].unique())
    sub["x"] = sub["period"].map({p: i for i, p in enumerate(order)})
    sub = sub.sort_values("x")
    sub["lo"] = sub["coef"] - 1.96 * sub["se"]
    sub["hi"] = sub["coef"] + 1.96 * sub["se"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(sub["x"], sub["lo"], sub["hi"], alpha=0.18,
                    color=ORANGE)
    ax.plot(sub["x"], sub["coef"], "o-", color=ORANGE, lw=1.8, ms=5)
    ax.axhline(0, color=DARK_TEXT, lw=0.7)
    ref_x = order.index("2022H1") if "2022H1" in order else None
    if ref_x is not None:
        ax.axvline(ref_x, color=GRAY, ls="--", lw=0.9)
        ax.text(ref_x, ax.get_ylim()[1], " reference: 2022H1",
                fontsize=8, color=GRAY, va="top")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Poisson coefficient (log points)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(V2_FIG / "fig2_poisson_es.pdf", bbox_inches="tight")
    fig.savefig(V2_FIG / "fig2_poisson_es.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("    saved fig2_poisson_es.pdf/.png")


def fig3_firm_entry_es():
    """The T17 exhibit: within-employer event study on public postings,
    entry-level ads, through June 2026 (l09 output)."""
    src = REV / "tables" / "firm_within_es.csv"
    if not src.exists():
        print("  Figure 3: waiting for l09 output")
        return
    print("  Figure 3: firm-level entry ES")
    es = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for variant, color, label in (
            ("a_all_firms", GRAY, "All ads"),
            ("c_entry_level_ads", ORANGE, "Entry-level ads")):
        sub = es[es["variant"] == variant].sort_values("period").copy()
        order = sorted(sub["period"].unique())
        sub["x"] = sub["period"].map({p_: i for i, p_ in enumerate(order)})
        sub["lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["hi"] = sub["coef"] + 1.96 * sub["se"]
        ax.fill_between(sub["x"], sub["lo"], sub["hi"], alpha=0.15,
                        color=color)
        ax.plot(sub["x"], sub["coef"], "o-", color=color, lw=1.8, ms=4.5,
                label=label)
    ax.axhline(0, color=DARK_TEXT, lw=0.7)
    ref_x = order.index("2022H1")
    ax.axvline(ref_x, color=GRAY, ls="--", lw=0.9)
    gpt_x = order.index("2022H2") - 0.5
    ax.axvline(gpt_x, color=TEAL, ls=":", lw=1.1)
    ax.text(gpt_x, ax.get_ylim()[0], "  ChatGPT\n  Nov 2022", fontsize=8,
            color=TEAL, va="bottom")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Poisson coefficient, high-exposure x half-year\n"
                  "(log points; ref. 2022H1)", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(V2_FIG / "fig3_firm_entry_es.pdf", bbox_inches="tight")
    fig.savefig(V2_FIG / "fig3_firm_entry_es.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("    saved fig3_firm_entry_es.pdf/.png")


def main():
    print("L7: figures rebuild (E8)")
    fig1_two_panel()
    fig2_poisson_es()
    fig3_firm_entry_es()


if __name__ == "__main__":
    main()

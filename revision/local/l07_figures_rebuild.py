#!/usr/bin/env python3
"""
l07_figures_rebuild.py: Figure 1 of the paper and the within-employer
posting event study of Online Appendix V.

WHAT IT DRAWS
Figure 1 (fig1_two_panel): two stacked panels replacing a dual axis. The
upper panel is the OMX Stockholm 30 index, the monthly mean of daily
closes indexed to 100 at February 2020. The lower panel is Platsbanken
postings by DAIOE generative-AI exposure quartile on the same base, as a
three-month centred moving average, drawn from the series extended to
June 2026 when script l08 has produced it and from the submitted series
cut at December 2025 otherwise. Series are distinguished by line style
and a legend rather than by colour alone, and the two event lines (the
Riksbank's first rate rise, April 2022; the first full month after the
ChatGPT launch, December 2022, which is where CHATGPT_LAUNCH puts the
line) carry their dates. Until 26 Sep 2026 the label read "ChatGPT
launch, November 2022" while the line stood at 1 December, the
mismatch the letter's figure check is there to catch.

Figure fig:firm_entry_es (fig3_firm_entry_es): the half-year Poisson
event-study coefficients of the within-employer posting design of script
l09, all advertisements and entry-level advertisements, reference the
first half of 2022, with 95 per cent intervals.

Figure figA_posting_context: the descriptive context for the posting
margin, four exhibits of the submitted online appendix (the Sweden
against United States comparison, the OMXSPI alternative to the OMXS30,
the individual exposure-quartile trends and the Riksbank policy rate)
drawn as one figure. Panel (a) is the United States, the S&P 500 against
the Indeed Hiring Lab total-postings index; panel (b) is Sweden, the
OMXS30 and the OMXSPI all-share against Platsbanken postings on one
indexed axis, so that the two stock indices can be read against each
other; panel (c) is the four exposure quartiles drawn individually; panel
(d) is the policy rate, its tightening cycle shaded and the seven-month
gap to the ChatGPT launch marked. The posting series run to June 2026
from the closed-quarter files, so the harvesting artefact of the last two
collection months, which the submitted version's figures still showed, is
gone.

A third function draws the Poisson event study of the withdrawn
occupation design from mona/output_43/poisson_es.csv when that file is
present; it belongs to the submitted version and produces nothing for the
current manuscript.

INPUTS AND OUTPUTS
Reads data/processed/omxs30_monthly.csv,
revision/output/postings_quartile_indexed_extended.csv (or
data/processed/postings_quartile_indexed.csv) and
revision/tables/firm_within_es.csv. figA_posting_context additionally
reads data/processed/omxspi_monthly.csv, data/processed/
riksbank_monthly.csv, data/raw/sp500_daily.csv and
data/raw/indeed_us_aggregate.csv. Writes revision/figures/
fig1_two_panel.pdf and .png, fig3_firm_entry_es.pdf and .png and
figA_posting_context.pdf and .png, the last copied to the paper repo.

IN THE PAPER
Figure 1 (Section 3), Online Appendix II.1 (figA_posting_context) and
Online Appendix V, Figure fig:firm_entry_es.
"""

import shutil
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsafe import save  # noqa: E402
from config import (PROCESSED, RAW, V2_FIG, POSTINGS_DESCRIPTIVE_END,  # noqa: E402
                    POSTINGS_REGRESSION_END, BASE_MONTH,
                    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
                    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY, DARK_TEXT)

MONA43 = REV / "mona" / "output_43"
PAPER = REV.parent.parent / "canaries-sweden-paper"


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
    # a line-style legend is the self-contained alternative (style and
    # label, not colour alone; the legend carries both).
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
             "  First full month after\n  ChatGPT launch, Dec 2022",
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
    """The within-employer posting event study of Online Appendix V,
    entry-level advertisements, through June 2026 (script l09's export)."""
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


def _quartile_series():
    """Postings by exposure quartile, extended to June 2026 when l08 has run.

    The submitted series carries the collection artefact of its final two
    months, so it is cut at POSTINGS_DESCRIPTIVE_END when it is the only
    one available.
    """
    ext = REV / "output" / "postings_quartile_indexed_extended.csv"
    if ext.exists():
        q = pd.read_csv(ext)
        q["date"] = pd.to_datetime(q["year_month"] + "-01")
        cut = pd.Timestamp(POSTINGS_REGRESSION_END + "-01")
    else:
        q = pd.read_csv(PROCESSED / "postings_quartile_indexed.csv")
        q["date"] = pd.to_datetime(q["date"])
        cut = pd.Timestamp(POSTINGS_DESCRIPTIVE_END + "-01")
    lo = pd.Timestamp("2020-01-01")
    return q[(q["date"] >= lo) & (q["date"] <= cut)].copy(), lo, cut


def figA_posting_context():
    """Online Appendix II.1: the four descriptive posting exhibits as one.

    Everything here is context for the "scary chart" rather than an
    estimate: whether the Swedish divergence also appears in the United
    States, whether it survives a broader stock index, whether the
    exposure quartiles move together, and what the monetary cycle behind
    them looked like. Four separate figures said this in the submitted
    version; one figure with four panels says it in the same detail.
    """
    print("  Figure A (II.1): posting context, four panels")
    q, lo, cut = _quartile_series()

    # Total postings are the sum over the four quartiles, so the panel
    # and the quartile panels are the same sample by construction.
    tot = (q.groupby("date", as_index=False)["n_ads"].sum()
             .sort_values("date"))
    base = tot.loc[tot["date"] == pd.Timestamp(BASE_MONTH), "n_ads"]
    tot["idx"] = tot["n_ads"] / float(base.iloc[0]) * 100
    tot["ma"] = tot["idx"].rolling(3, center=True, min_periods=1).mean()

    omxs = pd.read_csv(PROCESSED / "omxs30_monthly.csv",
                       index_col=0, parse_dates=True)
    omxspi = pd.read_csv(PROCESSED / "omxspi_monthly.csv",
                         index_col=0, parse_dates=True)
    rate = pd.read_csv(PROCESSED / "riksbank_monthly.csv",
                       parse_dates=["date"])
    omxs = omxs[(omxs.index >= lo) & (omxs.index <= cut)]
    omxspi = omxspi[(omxspi.index >= lo) & (omxspi.index <= cut)]
    rate = rate[(rate["date"] >= lo) & (rate["date"] <= cut)]

    # United States. The S&P 500 is indexed here on the same base month as
    # everything else; the Indeed index is published on that base already.
    sp = (pd.read_csv(RAW / "sp500_daily.csv", index_col=0, parse_dates=True)
            .resample("MS").mean())
    sp["idx"] = sp.iloc[:, 0] / sp.loc[BASE_MONTH].iloc[0] * 100
    sp = sp[(sp.index >= lo) & (sp.index <= cut)]
    ind = pd.read_csv(RAW / "indeed_us_aggregate.csv", parse_dates=["date"])
    ind = ind[ind["variable"] == "total postings"].set_index("date")
    ind = ind.resample("MS")["indeed_job_postings_index_SA"].mean()
    ind = ind[(ind.index >= lo) & (ind.index <= cut)]

    rb, gpt = pd.Timestamp(RIKSBANKEN_HIKE), pd.Timestamp(CHATGPT_LAUNCH)

    fig = plt.figure(figsize=(7.6, 6.9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.30, 0.95, 0.95],
                          hspace=0.75, wspace=0.18)
    ax_us = fig.add_subplot(gs[0, 0:2])
    ax_se = fig.add_subplot(gs[0, 2:4], sharey=ax_us)
    axq = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for a in axq[1:]:
        a.sharey(axq[0])
    ax_rate = fig.add_subplot(gs[2, 0:4])

    # -- (a) United States --
    ax_us.plot(sp.index, sp["idx"], color=DARK_BLUE, lw=1.6,
               label="S&P 500")
    ax_us.plot(ind.index, ind.values, color=ORANGE, lw=1.6, ls="-",
               label="Indeed postings")
    ax_us.set_title("(a) United States", fontsize=9.5, loc="left")
    ax_us.set_ylabel("Index, Feb 2020 = 100", fontsize=8.5)
    ax_us.legend(fontsize=7.5, frameon=False, loc="upper left")

    # -- (b) Sweden, both stock indices against postings --
    ax_se.plot(omxs.index, omxs["omxs30_idx"], color=DARK_BLUE, lw=1.6,
               label="OMXS30")
    ax_se.plot(omxspi.index, omxspi["omxspi_idx"], color=TEAL, lw=1.4,
               ls="--", label="OMXSPI (all-share)")
    ax_se.plot(tot["date"], tot["ma"], color=ORANGE, lw=1.6,
               label="Platsbanken postings")
    ax_se.set_title("(b) Sweden", fontsize=9.5, loc="left")
    ax_se.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_se.tick_params(labelleft=False)

    # -- (c) the four exposure quartiles, drawn one by one --
    names = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    titles = ["Q1, least exposed", "Q2", "Q3", "Q4, most exposed"]
    for a, name, title in zip(axq, names, titles):
        sub = q[q["exposure_quartile"] == name].sort_values("date").copy()
        sub["ma"] = sub["ads_idx"].rolling(3, center=True,
                                           min_periods=1).mean()
        a.plot(sub["date"], sub["ads_idx"], color=LIGHT_GRAY, lw=0.6)
        a.plot(sub["date"], sub["ma"], color=DARK_BLUE, lw=1.4)
        a.axhline(100, color=GRAY, lw=0.5, ls="--", alpha=0.6)
        a.set_title(title, fontsize=8, loc="left")
    axq[0].set_ylabel("Postings index", fontsize=8.5)
    for a in axq[1:]:
        a.tick_params(labelleft=False)
    axq[0].text(0.0, 1.30, "(c) Job postings by generative-AI exposure "
                "quartile", transform=axq[0].transAxes, fontsize=9.5,
                ha="left", va="bottom")

    # -- (d) the policy rate behind the cycle --
    ax_rate.step(rate["date"], rate["rate_pct"], where="post",
                 color=DARK_BLUE, lw=1.6)
    ax_rate.fill_between(rate["date"], 0, rate["rate_pct"], step="post",
                         color=DARK_BLUE, alpha=0.08)
    ax_rate.axvspan(rb, pd.Timestamp("2023-09-01"), color=TEAL, alpha=0.07,
                    zorder=0)
    ax_rate.annotate("Peak 4.00 per cent,\nSeptember 2023",
                     xy=(pd.Timestamp("2023-09-01"), 4.0),
                     xytext=(pd.Timestamp("2024-02-01"), 4.15),
                     fontsize=7.5, color=DARK_TEXT,
                     arrowprops=dict(arrowstyle="->", lw=0.7,
                                     color=DARK_TEXT))
    ax_rate.annotate("", xy=(gpt, -0.30), xytext=(rb, -0.30),
                     arrowprops=dict(arrowstyle="<->", lw=1.1,
                                     color=ORANGE))
    ax_rate.text(rb + (gpt - rb) / 2, -0.78, "7 months", fontsize=7.5,
                 color=ORANGE, ha="center", va="top")
    ax_rate.set_ylim(-1.25, 4.8)
    ax_rate.set_ylabel("Policy rate, per cent", fontsize=8.5)
    ax_rate.set_title("(d) Riksbank policy rate", fontsize=9.5, loc="left")

    # -- shared cosmetics: the two event dates on every panel --
    for a in [ax_us, ax_se] + axq + [ax_rate]:
        a.axvline(rb, color=TEAL, ls="--", lw=0.9, alpha=0.8)
        a.axvline(gpt, color=DARK_TEXT, ls=":", lw=0.9, alpha=0.8)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(lo, cut)
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.xaxis.set_minor_locator(mdates.YearLocator())
        a.tick_params(labelsize=7.5)
    for a in (ax_us, ax_se, ax_rate):
        a.xaxis.set_major_locator(mdates.YearLocator())
    # The two event lines are named in the caption rather than on the
    # panels: at this size an in-panel label lands on the legend.

    save(fig, "figA_posting_context", __file__)
    plt.close(fig)
    for ext in (".pdf", ".png"):
        shutil.copy(V2_FIG / f"figA_posting_context{ext}",
                    PAPER / "figures" / f"figA_posting_context{ext}")
    print(f"    saved figA_posting_context.pdf/.png, postings to "
          f"{cut:%Y-%m}, and copied to the paper repo")


def main():
    print("L7: figures rebuild (E8)")
    fig1_two_panel()
    fig2_poisson_es()
    fig3_firm_entry_es()
    figA_posting_context()


if __name__ == "__main__":
    main()

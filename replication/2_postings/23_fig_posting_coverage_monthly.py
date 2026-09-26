#!/usr/bin/env python3
"""
23_fig_posting_coverage_monthly.py: Online Appendix Figure A3 (Section II.5),
posting coverage by month, January 2020 to June 2026.

WHAT IT DRAWS
Three panels, with the rate rise (April 2022) and the first full month after
the launch (December 2022) marked: (a) the share of advertisements with a
valid SSYK 2012 code, by month and delivery channel (source_type), from the
monthly coverage file of 17; (b) the number of four-digit occupations with
at least one advertisement in the month, from the occupation-by-month
counts of 03; (c) the share of zero cells on the balanced occupation-by-
month grid of DAIOE-scored occupations, for all scored occupations and for
the top exposure quartile. Duplicates are included in (a), since coverage is
a property of the data. Channels with fewer than 1,000 advertisements over
the window are not drawn.

INPUTS   output/results/postings_coverage_monthly_extended.csv (17),
         postings_ssyk4_monthly_extended.csv (03);
         data/processed/daioe_quartiles.csv (1_data_public/04)
OUTPUTS  output/figures/fig_posting_coverage_monthly.pdf and .png
SERVES   Online Appendix II.5, Figure A3 (the 374 to 396 active occupations
         and the one per cent of zero cells in the text)
RUNTIME  seconds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS, PROCESSED, FIGURES = config.RESULTS, config.PROCESSED, config.FIGURES
COVER = RESULTS / "postings_coverage_monthly_extended.csv"
POST = RESULTS / "postings_ssyk4_monthly_extended.csv"
QUART = PROCESSED / "daioe_quartiles.csv"
START, END = "2020-01", "2026-06"
RB, GPT = "2022-04", "2022-12"
DARK, MID, LIGHT = "#222222", "#666666", "#aaaaaa"
STYLES = ["-", "--", "-.", ":"]


def months(lo: str, hi: str) -> list:
    return [str(p) for p in pd.period_range(lo, hi, freq="M")]


def main() -> int:
    for f in (COVER, POST, QUART):
        if not f.exists():
            raise SystemExit(f"  missing input: {f}")
    ms = months(START, END)
    x = list(range(len(ms)))
    pos = {m: i for i, m in enumerate(ms)}

    # (a) valid-code share by channel
    c = pd.read_csv(COVER, dtype={"year_month": str})
    c = c[c["year_month"].isin(ms)]
    channels = (c.groupby("source_type")["n_ads"].sum()
                .sort_values(ascending=False))
    channels = [s for s in channels.index if channels[s] >= 1000]

    # (b) active occupations; (c) zero cells on the scored grid
    p = pd.read_csv(POST, dtype={"ssyk4": str, "year_month": str})
    p = p[p["year_month"].isin(ms)]
    active = p[p["n_ads"] > 0].groupby("year_month")["ssyk4"].nunique()
    q = pd.read_csv(QUART, dtype={"ssyk4": str})
    q["ssyk4"] = q["ssyk4"].str.zfill(4)
    top = set(q.loc[q["exposure_quartile"].astype(str).str.startswith("Q4")
                    | (q.get("high_exposure", 0) == 1), "ssyk4"])
    scored = set(q["ssyk4"]) & set(p["ssyk4"])
    grid = pd.MultiIndex.from_product([sorted(scored), ms],
                                      names=["ssyk4", "year_month"])
    full = (p.set_index(["ssyk4", "year_month"])["n_ads"]
            .reindex(grid, fill_value=0).reset_index())
    full["zero"] = (full["n_ads"] == 0).astype(int)
    zero_all = full.groupby("year_month")["zero"].mean()
    zero_top = full[full["ssyk4"].isin(top)].groupby("year_month")["zero"].mean()

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.6), sharex=True)
    ax = axes[0]
    for k, ch in enumerate(channels):
        s = c[c["source_type"] == ch].set_index("year_month")["valid_share"]
        s = s.reindex(ms)
        ax.plot(x, 100 * s.to_numpy(), STYLES[k % 4], color=DARK if k == 0 else MID,
                lw=1.3, label="no source" if ch == "(none)" else ch)
    ax.set_ylabel("Valid SSYK code,\nper cent of advertisements")
    ax.set_ylim(90, 100.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    ax.set_title("(a) Valid-code share by month and delivery channel", loc="left",
                 fontsize=9)

    ax = axes[1]
    ax.plot(x, active.reindex(ms).to_numpy(), "-", color=DARK, lw=1.3)
    ax.set_ylabel("Occupations with at\nleast one advertisement")
    ax.set_title("(b) Active four-digit occupations", loc="left", fontsize=9)

    ax = axes[2]
    ax.plot(x, 100 * zero_all.reindex(ms).to_numpy(), "-", color=MID, lw=1.3,
            label=f"All scored occupations ({len(scored)})")
    ax.plot(x, 100 * zero_top.reindex(ms).to_numpy(), "-", color=DARK, lw=1.6,
            label=f"Top exposure quartile ({len(top & scored)})")
    ax.set_ylabel("Zero cells, per cent of\noccupation-months")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(c) Share of occupation-months with no advertisement",
                 loc="left", fontsize=9)

    for ax in axes:
        for m in (RB, GPT):
            ax.axvline(pos[m], color=LIGHT, lw=0.9, ls="--", zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].text(pos[RB] - 0.6, 90.4, "rate rise", ha="right", va="bottom",
                 fontsize=7.5, color=MID)
    axes[0].text(pos[GPT] + 0.6, 90.4, "launch", ha="left", va="bottom",
                 fontsize=7.5, color=MID)
    ticks = [pos[m] for m in ms if m.endswith("-01")]
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([m[:4] for m in ms if m.endswith("-01")])
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        kw = {"dpi": 300} if ext == ".png" else {}
        fig.savefig(FIGURES / f"fig_posting_coverage_monthly{ext}", bbox_inches="tight", **kw)
    print(f"  channels drawn: {channels}")
    print(f"  active occupations: {int(active.min())}-{int(active.max())} a month")
    print(f"  zero-cell share, all scored: {100 * zero_all.mean():.1f} per cent mean; "
          f"top quartile: {100 * zero_top.mean():.1f}")
    print("  wrote output/figures/fig_posting_coverage_monthly.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

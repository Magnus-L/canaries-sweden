#!/usr/bin/env python3
"""
l14_fig_spreading.py: Figure 3 of the paper, the quarterly path of young
employment relative to older colleagues, and its monthly diagnostic.

WHAT IT DRAWS
The quarterly path for 22-25 and for 26-30 on one pair of axes, from the
seasonal path export of script 68: each point is the coefficient on that
quarter's interaction with High x Young, a step from the level of the
tightening months (the Riksbank term stays in the fit), with the calendar
cycle removed by the quarter-of-year terms, and 95 per cent intervals
clustered by employer. The point for 2022Q4 is December 2022 alone, the
first full month after the launch, and is drawn hollow. Both series come
from the same specification, so the comparison is between two seasonally
adjusted paths. A lower panel gives Statistics Sweden's published share of
enterprises with ten or more employees using AI (10, 25 and 35 per cent
for the survey years 2023, 2024 and 2025), which is the series the timing
is read against; it is published, not estimated here. The last points
rest on six months of 2025 rather than twelve; the employer declarations
are not revised after delivery, so they are not preliminary.

With --monthly the same design is drawn month by month. That chart is a
diagnostic and not an estimate: the specification controls quarter of
year, so a monthly coefficient retains whatever separates the month from
its own quarter's mean, and that residual is large at 22-25. It shows how
much the 2025 endpoint moves month to month and is used in the online
appendix for that purpose only.

    python3 revision/local/l14_fig_spreading.py [export_dir]
    python3 revision/local/l14_fig_spreading.py --monthly [export_dir]

INPUTS AND OUTPUTS
Reads seasonal_path.csv from the export directory the final-code manifest
names (or from a directory given on the command line). Writes
revision/figures/fig2_spreading.pdf and .png, and with --monthly
fig2_spreading_monthly.pdf and .png, through _figsafe.save; the copies in
the manuscript repository's figures/ folder are placed there by hand.

IN THE PAPER
Figure 3 (label fig:age_gradient, file fig2_spreading.pdf) in Section 3,
and the monthly diagnostic figure of Online Appendix III.2.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsafe import save  # noqa: E402
from config import V2_FIG, DARK_BLUE, ORANGE, GRAY, LIGHT_GRAY, DARK_TEXT

# Statistics Sweden's firm AI-use series, the thing the timing is read
# against: the share of enterprises with ten or more employees that use
# AI, from the annual survey on ICT usage in enterprises (Företagens
# användning av IT), survey years 2023, 2024 and 2025. The paper quotes
# the first two in Section 3 and in Online Appendix III.2. The series is
# published, not estimated here, and is not in any export in this tree.
SCB_ADOPTION = {"2023": 10, "2024": 25, "2025": 35}
LAUNCH_Q = "2022Q4"

# The script 68 export the final-code manifest names. Pass a directory on
# the command line to read seasonal_path.csv from there instead.
LANE14 = REV / "output" / "round3_20260921-2152-lane14-seasonal-complete"


def find_path_csv(argv) -> Path | None:
    d = Path(argv[1]) if len(argv) > 1 else LANE14
    p = d / "seasonal_path.csv"
    return p if p.exists() else None


def _to_date(period: str):
    """'2025-01' and '2025Q1' onto one axis; a quarter sits at its middle."""
    p = str(period)
    if "Q" in p:
        y, qq = p.split("Q")
        return pd.Timestamp(int(y), 3 * int(qq) - 1, 15)
    return pd.Timestamp(int(p[:4]), int(p[5:7]), 15)


def monthly_figure(d: pd.DataFrame) -> int:
    m = d[(d["shape"] == "month") & (d.get("status", "ok") == "ok")].copy()
    q = d[(d["shape"] == "quarter") & (d.get("status", "ok") == "ok")].copy()
    if m.empty:
        print("  no monthly rows in this export")
        return 1
    for f in (m, q):
        f["t"] = f["period"].map(_to_date)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    drawn = {}
    for band, colour, mk in (("22-25", ORANGE, "o"),
                             ("26-30", DARK_BLUE, "s")):
        mb = m[m["young_band"] == band].sort_values("t")
        if mb.empty:
            # If a band has no monthly path, fall back to its quarterly
            # series and say so in the legend rather than silently mixing
            # frequencies.
            qb = q[q["young_band"] == band].sort_values("t")
            if qb.empty:
                continue
            ax.plot(qb["t"], qb["coef"], "--", color=colour, lw=1.6,
                    marker=mk, ms=4.5,
                    label=f"{band}, QUARTERLY (no monthly path)",
                    zorder=3)
            drawn[band] = "quarterly"
            continue
        ax.fill_between(mb["t"], mb["coef"] - 1.96 * mb["se"],
                        mb["coef"] + 1.96 * mb["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(mb["t"], mb["coef"], "-", color=colour, lw=1.5,
                marker=mk, ms=3.4, label=f"{band}, monthly", zorder=3)
        drawn[band] = "monthly"

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    ax.axvline(pd.Timestamp(2022, 11, 30), color=GRAY, ls="--", lw=0.9)
    ax.text(pd.Timestamp(2022, 12, 5), ax.get_ylim()[1], " ChatGPT",
            fontsize=8, color=GRAY, va="top")

    # No shading on 2025: the employer declarations are not revised after
    # delivery, so 2025 is definitive. That the endpoint rests on six
    # months rather than twelve belongs in the caption, not in a grey box
    # that reads as a health warning.

    ax.set_ylabel("Employment, log points, cycle removed", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.tick_params(labelsize=8.5)
    fig.autofmt_xdate(rotation=45, ha="right")
    save(fig, "fig2_spreading_monthly", __file__)
    plt.close(fig)
    print(f"    saved fig2_spreading_monthly.pdf/.png  {drawn}")
    if any(v == "quarterly" for v in drawn.values()):
        print("    WARNING: a band fell back to quarterly; the chart "
              "mixes frequencies")
    return 0


def main() -> int:
    monthly = "--monthly" in sys.argv
    if monthly:
        sys.argv.remove("--monthly")
    src = find_path_csv(sys.argv)
    if src is None:
        print("  no seasonal_path.csv found; run lane 14 and export it")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src)
    if monthly:
        return monthly_figure(d)
    q = d[(d["shape"] == "quarter") & (d.get("status", "ok") == "ok")].copy()
    if q.empty:
        print("  seasonal_path.csv has no quarterly rows")
        return 1

    order = sorted(q["period"].unique())
    q["x"] = q["period"].map({p: i for i, p in enumerate(order)})

    # SCB adoption goes in its OWN panel. Drawn behind the estimates on a
    # twin axis it read as part of the result: grey bars rising from the
    # bottom of a chart whose left axis is negative look like a series.
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.12})

    styles = {"22-25": (ORANGE, "o", "22-25"),
              "26-30": (DARK_BLUE, "s", "26-30")}
    for band, (colour, marker, label) in styles.items():
        b = q[q["young_band"] == band].sort_values("x")
        if b.empty:
            continue
        ax.fill_between(b["x"], b["coef"] - 1.96 * b["se"],
                        b["coef"] + 1.96 * b["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(b["x"], b["coef"], "-", color=colour, lw=1.9,
                label=label, zorder=3)
        post = b[b["period"] != LAUNCH_Q]
        pre = b[b["period"] == LAUNCH_Q]
        ax.plot(post["x"], post["coef"], marker, color=colour, ms=5,
                zorder=4)
        ax.plot(pre["x"], pre["coef"], marker, mfc="white", mec=colour,
                mew=1.4, ms=5, zorder=4)

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    if LAUNCH_Q in order:
        xl = order.index(LAUNCH_Q)
        for a in (ax, axb):
            a.axvline(xl, color=GRAY, ls="--", lw=0.9, zorder=1)
        ax.annotate("ChatGPT", xy=(xl, ax.get_ylim()[1]),
                    xytext=(xl + 0.15, ax.get_ylim()[1]),
                    fontsize=8, color=GRAY, va="top")
    ax.set_ylabel("Employment, log points\ncycle removed", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(axis="y", labelsize=8.5)

    for yr, pct in SCB_ADOPTION.items():
        xs = [i for i, p in enumerate(order) if p.startswith(yr)]
        if xs:
            axb.bar(xs, [pct] * len(xs), width=0.94, color=LIGHT_GRAY,
                    align="center")
            axb.text(sum(xs) / len(xs), pct + 3, f"{pct:.0f}%",
                     ha="center", fontsize=8, color=GRAY)
    axb.set_ylim(0, 52)
    axb.set_ylabel("SCB: firms\nusing AI", fontsize=8.5, color=GRAY)
    axb.tick_params(axis="y", labelsize=8, colors=GRAY)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels(order, rotation=45, ha="right", fontsize=8.5)

    save(fig, "fig2_spreading", __file__)
    plt.close(fig)
    print(f"    saved fig2_spreading.pdf/.png "
          f"({len(order)} quarters, bands "
          f"{sorted(q['young_band'].unique())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

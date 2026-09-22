#!/usr/bin/env python3
"""
l14_fig_spreading.py -- the revision's headline exhibit, built from the
exported lane 14 path rather than by hand.

WHY THIS EXISTS. `figures/fig2_quarterly_path.pdf` was produced ad hoc on
21 September: no PNG twin, no script in the tree, no way to tell which
export it came from. That is the wrong footing for the figure the paper
leads on, and the change list has since respecified it anyway, as two
series rather than one.

WHAT IT SHOWS. The quarterly path for 22-25 and for 26-30 on the same
axes, with SCB's measured firm AI adoption behind them. The claim is the
second line peeling away from zero about a year after the first: the
effect reaches the youngest band first and the next one later. Both
series have the calendar cycle removed, which matters -- the 22-25 and
26-30 paths must come from the SAME specification or the comparison is
between a seasonally adjusted series and a raw one.

READ RULES CARRIED INTO THE FIGURE.
  * 2022Q4 straddles the ChatGPT launch and is neither pre nor post. It
    is drawn hollow and excluded from any statement about the pre-period.
  * 2025 is half a year of AGI data, so the last points rest on fewer
    months than the rest. They are NOT preliminary; see below.
  * No yearly series is plotted, because the quarterly one is strictly
    finer. The 22-25 yearly path did land in the end (-0.0178, -0.0414,
    -0.0403); the note that it "was never estimated" described the
    13:23 export, whose fit had crashed, and is withdrawn.

    python3 revision/local/l14_fig_spreading.py [export_dir]
    python3 revision/local/l14_fig_spreading.py --monthly [export_dir]

THE MONTHLY VARIANT IS A DIAGNOSTIC, NOT AN ESTIMATE, and must not be
the paper's figure. The design controls quarter-of-year (q1/q2/q3 x
high x young), so a quarterly coefficient is measured against a control
at its own frequency and a monthly one is not: each month keeps
whatever separates it from its own quarter's mean. That residual is
large in 22-25 and small in 26-30. The within-Q1 spread of the monthly
coefficients runs 0.0152, 0.0112 and 0.0404 across 2023-25 for 22-25,
the last of these wider than the 2025Q1 coefficient itself, against
0.0087, 0.0093 and 0.0104 for 26-30. So February 2025 at +0.0208 and
June at -0.1003 are the artefact, not the signal, and they are not a
reason to prefer monthly. What the chart is good for is showing how
much the 2025 endpoint moves month to month. Both bands have a monthly
path since the 21:52 export of 21 September; before that only 22-25
did, and the chart mixed frequencies.

2025 IS NOT PRELIMINARY. SCB confirmed to ML on 21 September 2026 that
the AGI monthly figures are not revised after delivery. Earlier notes
in this project hedged the 2025 endpoint as preliminary; that hedge is
withdrawn. The endpoint still rests on six months rather than twelve,
which is a different and much weaker caveat.
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
SCB_ADOPTION = {"2023": 10.4, "2024": 25.2, "2025": 35.0}
LAUNCH_Q = "2022Q4"

# The lane 14 export the final-code manifest names. Pass a directory on
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
            # Before the 21:52 export of 21 Sep only 22-25 had a monthly
            # path. Fall back to the quarterly series and SAY SO in the
            # legend rather than silently mixing frequencies.
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

    # NO SHADING ON 2025. SCB confirmed to ML that the AGI months are not
    # revised after delivery, so 2025 is definitive and not preliminary.
    # What remains true is only that it is half a year, which is a
    # statement about how many months the endpoint rests on, not about
    # whether those months will change. That belongs in the caption, not
    # in a grey box that reads as a health warning.

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

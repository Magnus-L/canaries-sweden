#!/usr/bin/env python3
"""
l28_fig_prepath.py: the online-appendix figure of the pre-period, the
quarterly path of young employment relative to older colleagues on the
plain specification.

WHAT IT DRAWS AND WHY IT IS DRAWN THIS WAY
The path for workers aged 22 to 25, from the pre-period export of script
78 (lane 25, part A(i)): each point is the coefficient on that quarter's
interaction with High x Young, measured against 2022Q1, which is omitted.
The specification is the plain one, with employer-by-month,
employer-by-age and month-by-age effects but NO calendar-quarter terms,
so nothing removes the seasonal cycle. That is the point of the exhibit:
the reader sees the raw object the paper's calendar terms are there to
handle, and can judge the pre-period for herself rather than take the
drift test on trust.

Drawn quarter by quarter as one line, the cycle dominates and the chart
is illegible: the fourth quarter sits far above the first in every year,
so a sawtooth swamps any movement between years. The figure therefore
draws FOUR series, one per calendar quarter, against the year. Reading
along a series holds the quarter fixed and shows the movement the paper
cares about; reading down a year shows the cycle itself. Whiskers are 95
per cent intervals clustered by employer.

The panel behind this figure reaches back to 2019 and so is larger than
the headline panel, which starts in January 2021; the cell count printed
on the chart is this panel's. Nothing here is a pooled estimate. The
verdict on the pre-period is the linear drift test of part A(ii),
reported in Table 1 of the paper.

    python3 revision/local/l28_fig_prepath.py [export_dir]

INPUTS AND OUTPUTS
Reads prepath_plain.csv from the script 78 export directory pinned
below (or from a directory given on the command line). Writes
revision/figures/fig_prepath.pdf and .png through _figsafe.save and
copies both to canaries-sweden-paper/figures/.

IN THE PAPER
Online Appendix III.2, label fig:prepath, file fig_prepath.pdf. The main
text points at it where it reports the pre-launch drift.
"""
import shutil
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
from config import DARK_BLUE, ORANGE, TEAL, GRAY, DARK_TEXT  # noqa: E402

PAPER = REV.parent.parent / "canaries-sweden-paper"
# The script 78 export (lane 25a) the final-code manifest names.
LANE25A = REV / "output" / "round3_20260922-1333-lane25a-ADG"

BAND = "22-25"
REFERENCE = "2022Q1"
# One style per calendar quarter. Distinct markers and dashes as well as
# colours, so the figure survives printing in grey.
STYLE = {
    1: (DARK_BLUE, "o", "-", "Q1"),
    2: (ORANGE, "s", "--", "Q2"),
    3: (TEAL, "^", "-.", "Q3"),
    4: (GRAY, "D", ":", "Q4"),
}


def main(export_dir: Path) -> int:
    src = export_dir / "prepath_plain.csv"
    if not src.exists():
        raise SystemExit(f"  missing input: {src}")
    d = pd.read_csv(src)
    d = d[d["young_band"] == BAND].copy()
    if d.empty:
        raise SystemExit(f"  no rows for band {BAND} in {src.name}")
    if REFERENCE not in set(d["quarter"]):
        raise SystemExit(f"  {REFERENCE} is not in the export; the omitted "
                         f"quarter has to be drawn, not assumed")

    d["year"] = d["quarter"].str.slice(0, 4).astype(int)
    d["q"] = d["quarter"].str.slice(5, 6).astype(int)
    cells = int(d["n_obs"].max())

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    for q in (1, 2, 3, 4):
        colour, marker, dash, label = STYLE[q]
        s = d[d["q"] == q].sort_values("year")
        if s.empty:
            continue
        ax.plot(s["year"], s["coef"], dash, color=colour, lw=1.5, zorder=3)
        for _, r in s.iterrows():
            if r["se"] > 0:
                ax.plot([r["year"], r["year"]],
                        [r["coef"] - 1.96 * r["se"], r["coef"] + 1.96 * r["se"]],
                        color=colour, lw=1.0, alpha=0.7, zorder=2)
        post = s[s["quarter"] != REFERENCE]
        ax.plot(post["year"], post["coef"], marker, ms=5, mfc=colour,
                mec=colour, zorder=4, label=label)
        ref = s[s["quarter"] == REFERENCE]
        if not ref.empty:
            # The omitted quarter is a zero by construction, not an
            # estimate: hollow, and named on the chart.
            ax.plot(ref["year"], ref["coef"], marker, ms=6.5, mfc="white",
                    mec=colour, mew=1.4, zorder=5)
            ax.annotate(f"{REFERENCE} omitted",
                        xy=(float(ref["year"].iloc[0]), 0.0),
                        xytext=(float(ref["year"].iloc[0]) + 0.10, -0.040),
                        fontsize=8, color=DARK_TEXT,
                        arrowprops=dict(arrowstyle="-", lw=0.7,
                                        color=DARK_TEXT))

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    ax.set_xticks(sorted(d["year"].unique()))
    ax.set_xlabel("Year", fontsize=9.5)
    ax.set_ylabel("Employment of 22–25 relative to older colleagues,\n"
                  "log points, no calendar terms", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper right",
              title="Calendar quarter", title_fontsize=8.5)
    ax.text(0.01, 0.02, f"{cells:,} employer × age × month cells",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=GRAY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()

    save(fig, "fig_prepath", __file__)
    plt.close(fig)
    for ext in (".pdf", ".png"):
        shutil.copy(REV / "figures" / f"fig_prepath{ext}",
                    PAPER / "figures" / f"fig_prepath{ext}")
    print(f"    saved fig_prepath.pdf/.png "
          f"({len(d)} quarters, band {BAND}) and copied to the paper repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else LANE25A))

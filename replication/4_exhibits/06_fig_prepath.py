#!/usr/bin/env python3
"""
06_fig_prepath.py: Online Appendix Figure A7 (Section III.2), the quarterly
path of young employment relative to older colleagues on the plain
specification, 2019Q1 to 2025Q2.

Each point is the coefficient on one quarter's interaction with High x Young
for workers aged 22 to 25, against 2022Q1, which is omitted, with
employer-by-month, employer-by-age and month-by-age effects and no calendar
terms, so the seasonal cycle is visible. The path is drawn as four series, one
per calendar quarter, against the year; whiskers are 95 per cent intervals
clustered by employer. The panel reaches back to 2019 and is larger than the
headline panel. The verdict on the pre-period is the linear drift test of
Table 1 and Table A23.

Export read: 3_register_mona/exports/2026-09-23_1234_s86/occ_route_prepath.csv
(script 86); script 78's export on the education-based score,
2026-09-22_1333_s78/prepath_plain.csv, is read if its folder is given.
Output: output/figures/fig_prepath.pdf and .png

    python 4_exhibits/06_fig_prepath.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save  # noqa: E402
from config import EXPORTS, DARK_BLUE, ORANGE, TEAL, GRAY, DARK_TEXT  # noqa: E402

# Script 86 ran script 78's part A on the occupation-mix score the paper
# reports. Script 78's own export is on the education-based score and is
# read if its folder is given on the command line; the two name the same
# frame differently, so both names are tried.
S86 = EXPORTS / "2026-09-23_1234_s86"
S78 = EXPORTS / "2026-09-22_1333_s78"
NAMES = ("occ_route_prepath.csv", "prepath_plain.csv")

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
    src = next((export_dir / n for n in NAMES if (export_dir / n).exists()),
               None)
    if src is None:
        raise SystemExit(f"  missing input: none of {NAMES} in {export_dir}")
    print(f"  reading {src}")
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
    print(f"    saved fig_prepath.pdf/.png "
          f"({len(d)} quarters, band {BAND})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else S86))

#!/usr/bin/env python3
"""
09_honestdid_figure.py: panel (b) of Online Appendix Figure A2, drawn from the
exact Rambachan and Roth (2023) relative-magnitudes bounds of 08.

WHAT IS DRAWN
The reported variant (reference November 2022, 34 pre-periods, the 43
post-launch months averaged): the 95 per cent robust confidence interval for
each Mbar on 08's grid, the point estimate with its interval under exact
parallel trends, and the breakdown value. Grid points whose interval reaches
the edge of 08's test grid [-2, 2] are dropped, since their end points are the
grid's and not the interval's. The axis stops at Mbar = 0.05, three times the
breakdown; the full grid is in the CSV.

INPUTS   output/results/posting_rr_honestdid_v3.csv (08)
OUTPUTS  output/figures/figA6_rambachan_roth_v3.pdf and .png
SERVES   Online Appendix II.2, Figure A2, panel (b)
RUNTIME  seconds
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, set_rcparams, save_pdf_png, DARK_BLUE, ORANGE, GRAY  # noqa: E402

set_rcparams()

VARIANT = "A_pre_launch"


def main():
    d = pd.read_csv(config.RESULTS / "posting_rr_honestdid_v3.csv")
    d = d[d["variant"] == VARIANT]
    orig = d[d["method"] == "original"].iloc[0]
    g = d[(d["method"] != "original") & ~d["at_grid_edge"]
          & (d["Mbar"] <= 0.05 + 1e-9)].sort_values("Mbar")
    theta, bd = orig["theta_hat"], orig["breakdown_mbar"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(g["Mbar"], g["lb"], g["ub"], alpha=0.2, color=DARK_BLUE,
                    label="95% robust CI (HonestDiD)")
    ax.plot(g["Mbar"], g["lb"], color=DARK_BLUE, linewidth=0.8)
    ax.plot(g["Mbar"], g["ub"], color=DARK_BLUE, linewidth=0.8)
    ax.errorbar([0], [theta], yerr=[[theta - orig["lb"]], [orig["ub"] - theta]],
                fmt="o", color=DARK_BLUE, capsize=4,
                label=f"$\\hat{{\\theta}}$ = {theta:.3f}, original 95% CI")
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.axvline(bd, color=ORANGE, linewidth=1.5, linestyle=":",
               label=f"Breakdown $\\bar{{M}}$ = {bd:.3f}")
    ax.set_xlabel("$\\bar{M}$ (relative magnitudes)")
    ax.set_ylabel("Average post-ChatGPT effect")
    ax.set_title("Rambachan-Roth sensitivity: average post-ChatGPT effect\n"
                 "on high vs low genAI exposure occupations")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    save_pdf_png(fig, "figA6_rambachan_roth_v3")
    plt.close(fig)
    print(f"  figA6 drawn from the exact bounds ({len(g)} grid points, "
          f"breakdown {bd:.3f})")


if __name__ == "__main__":
    main()

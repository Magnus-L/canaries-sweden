#!/usr/bin/env python3
"""
l47b_fig_honestdid_v3.py: panel (b) of Online Appendix II.2, drawn from
the exact Rambachan and Roth (2023) relative-magnitudes bounds of l47.

Until 24 September 2026 panel (b) was drawn from the simplified interval
theta +/- (1.96 SE + Mbar x Dmax) of src/07_robustness.py (kept by l46 as
figA6_rambachan_roth_simplified_v3). That interval is narrower than the
exact one, so this figure replaces it in the paper.

WHAT IS DRAWN
The reported variant (A: reference November 2022, 34 pre-periods, the 43
post-launch months averaged): the 95 per cent robust confidence interval
for each Mbar on l47's grid, the point estimate, the original interval
under exact parallel trends, and the breakdown value. Grid points whose
interval reaches the edge of l47's test grid [-2, 2] are dropped, since
their end points are the grid's and not the interval's. The axis stops
at Mbar = 0.05, three times the breakdown; the wider grid is in the CSV.

    python3 revision/local/l47b_fig_honestdid_v3.py

INPUT   revision/tables/posting_rr_honestdid_v3.csv (l47)
OUTPUT  revision/figures/figA6_rambachan_roth_v3.{pdf,png}; the png is
        copied to canaries-sweden-paper/figures/.
IN THE PAPER Online Appendix II.2, Figure fig:posting_design, panel (b).
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "local"))
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
from _figsafe import save  # noqa: E402

_src_spec = importlib.util.spec_from_file_location(
    "srcconfig", REV.parent / "src" / "config.py")
_src = importlib.util.module_from_spec(_src_spec)
_src_spec.loader.exec_module(_src)
DARK_BLUE, ORANGE, GRAY = _src.DARK_BLUE, _src.ORANGE, _src.GRAY
_src.set_rcparams()

PAPER_FIG = REV.parents[1] / "canaries-sweden-paper" / "figures"
VARIANT = "A_pre_launch"


def main():
    d = pd.read_csv(_cfg.V2_TAB / "posting_rr_honestdid_v3.csv")
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
    save(fig, "figA6_rambachan_roth_v3", __file__, _cfg.V2_FIG)
    plt.close(fig)
    shutil.copy(_cfg.V2_FIG / "figA6_rambachan_roth_v3.png",
                PAPER_FIG / "figA6_rambachan_roth_v3.png")
    print(f"  figA6 drawn from the exact bounds ({len(g)} grid points, "
          f"breakdown {bd:.3f}); png copied to the paper repo")


if __name__ == "__main__":
    main()

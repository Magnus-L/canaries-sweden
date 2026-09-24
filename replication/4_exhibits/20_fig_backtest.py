#!/usr/bin/env python3
"""
20_fig_backtest.py: Online Appendix Figure A8 (Section IV.3), the as-of
backtest.

For each truncation of the occupation register (2021 and 2022), the
coefficient the submitted design returns with the codes as they eventually
read and with the codes the coding cascade would have produced had the
register stopped at that year (script 45), with 95 per cent intervals. The gap
between the two, as-of minus true, is written on the figure as the artefact.

Export read: 3_register_mona/exports/2026-09-18_1736_s45/asof_estimates.csv
Output: output/figures/figA1_asof_backtest.pdf and .png

    python 4_exhibits/20_fig_backtest.py [export_dir]
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
from config import EXPORTS, DARK_BLUE, ORANGE, GRAY, DARK_TEXT  # noqa: E402

# Script 45's export. Pass a directory on the command line to read from
# there instead.
S45 = EXPORTS / "2026-09-18_1736_s45"


def find(argv, name):
    d = Path(argv[1]) if len(argv) > 1 else S45
    p = d / name
    return p if p.exists() else None


def main() -> int:
    src = find(sys.argv, "asof_estimates.csv")
    if src is None:
        print("  no asof_estimates.csv found")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src).sort_values(["trunc", "assignment"])

    truncs = sorted(d["trunc"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style = {"true": (DARK_BLUE, "o", "codes as they eventually read"),
             "asof": (ORANGE, "s", "codes as they read at T")}

    for i, t in enumerate(truncs):
        sub = d[d["trunc"] == t]
        for arm, (colour, marker, label) in style.items():
            r = sub[sub["assignment"] == arm]
            if r.empty:
                continue
            x = i + (-0.11 if arm == "true" else 0.11)
            g, se = float(r["gamma2"].iloc[0]), float(r["se2"].iloc[0])
            ax.errorbar(x, g, yerr=1.96 * se, fmt=marker, color=colour,
                        ms=7, capsize=4, lw=1.6,
                        label=label if i == 0 else None)
        # the gap IS the artefact: name it on the figure
        tr = sub[sub["assignment"] == "true"]
        af = sub[sub["assignment"] == "asof"]
        if not tr.empty and not af.empty:
            a, b = float(tr["gamma2"].iloc[0]), float(af["gamma2"].iloc[0])
            ax.annotate("", xy=(i, b), xytext=(i, a),
                        arrowprops=dict(arrowstyle="<->", color=GRAY,
                                        lw=1.1))
            ax.text(i + 0.055, (a + b) / 2, f"artefact {b - a:+.3f}",
                    fontsize=8.5, color=DARK_TEXT, va="center")

    ax.axhline(0, color=DARK_TEXT, lw=0.8)
    ax.set_xticks(range(len(truncs)))
    ax.set_xticklabels([f"truncated at {t}" for t in truncs], fontsize=9.5)
    ax.set_xlim(-0.5, len(truncs) - 0.35)
    ax.set_ylabel("Age gradient, log points", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # upper left: lower left collides with the 2021 as-of marker, which
    # sits near -0.29 and is the point the figure exists to show
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(0.005, 0.72))

    save(fig, "figA1_asof_backtest", __file__)
    plt.close(fig)
    print(f"    saved figA1_asof_backtest.pdf/.png ({len(truncs)} truncations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

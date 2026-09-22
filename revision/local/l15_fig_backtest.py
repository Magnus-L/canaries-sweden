#!/usr/bin/env python3
"""
l15_fig_backtest.py: Figure A1, the as-of backtest.

WHAT IT DRAWS
For each truncation of the occupation register (2021 and 2022), the
coefficient the submitted design returns with the true contemporaneous
codes and with the codes the coding cascade would have produced had the
register stopped at that year, from the estimates of script 45, with 95
per cent intervals. The distance between the two markers is the artefact
the lag alone manufactures, and it is written on the figure as the as-of
coefficient minus the true one.

    python3 revision/local/l15_fig_backtest.py [export_dir]

INPUTS AND OUTPUTS
Reads asof_estimates.csv from the script 45 export directory the
final-code manifest names (or from a directory given on the command
line). Writes revision/figures/figA1_asof_backtest.pdf and .png through
_figsafe.save.

IN THE PAPER
Online Appendix IV.3, Figure fig:asof_backtest, beside Table IV.3.
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
from config import V2_FIG, DARK_BLUE, ORANGE, GRAY, DARK_TEXT

# The script 45 export the final-code manifest names. Pass a directory
# on the command line to read from there instead.
B45 = REV / "output" / "round2_20260918-1736-script45"


def find(argv, name):
    d = Path(argv[1]) if len(argv) > 1 else B45
    p = d / name
    return p if p.exists() else None


def main() -> int:
    src = find(sys.argv, "asof_estimates.csv")
    if src is None:
        print("  no asof_estimates.csv found; run script 45 and export it")
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

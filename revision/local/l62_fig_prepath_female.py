#!/usr/bin/env python3
"""
l62_fig_prepath_female.py: the online-appendix figure of the female
differential's pre-launch path, raw, from the lane 37b export.

WHAT IT DRAWS AND WHY
The quarterly path of the young x exposed x female term, January 2021 to
December 2022, each quarter against 2022Q1 (omitted), on the plain
specification: sex-specific employer-by-age, employer-by-month and
month-by-age effects and NO calendar-quarter terms, so the seasonal cycle
is visible. Two sets of whiskers: 95 per cent intervals clustered by
employer (dark) and by three-digit industry (light). The reader sees the
recurring third-quarter dip that the paper's calendar terms absorb, and
can judge the pre-period against the drift test (Table tab:final_checks,
Panel E) rather than take it on trust.

Why raw and not on the paper's own specification: once every pre-launch
quarter carries its own term, the calendar-quarter terms are identified
from the eight tightening months alone and the first-quarter term drops
out, so a seasonally adjusted pre-period path on the headline
specification does not exist. The drift test is the adjusted diagnostic.

INPUTS AND OUTPUTS
Reads headline_checks.csv (part P, specs path and path_indcl, terms
pq_YYYYQn_x_hyf) from the lane 37b export, or a directory given on the
command line. Writes revision/figures/fig_prepath_female.pdf and .png
through _figsafe.save and copies both to canaries-sweden-paper/figures/.

    python3 revision/local/l62_fig_prepath_female.py [export_dir]

IN THE PAPER
Online Appendix III.2, label fig:prepath_female.
"""
import re
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

REPO = REV.parent
PAPER = REPO.parent / "canaries-sweden-paper"
DEFAULT = REV / "output" / "round3_20260925-2250-lane37b-97"

DARK = "#222222"
MID = "#888888"
REFERENCE = "2022Q1"
QUARTERS = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2",
            "2022Q3", "2022Q4"]


def path(d: pd.DataFrame, spec: str) -> pd.DataFrame:
    p = d[(d["part"] == "P") & (d["spec"] == spec)
          & d["term"].str.contains(r"_x_hyf$|_reference$")].copy()
    p["quarter"] = p["term"].str.extract(r"pq_(\d{4}Q\d)")[0]
    p.loc[p["term"].str.endswith("_reference"), "quarter"] = REFERENCE
    return p.set_index("quarter")


def main(export_dir: Path) -> int:
    src = export_dir / "headline_checks.csv"
    if not src.exists():
        raise SystemExit(f"  missing input: {src}")
    d = pd.read_csv(src)
    emp, ind = path(d, "path"), path(d, "path_indcl")
    for name, p in (("employer", emp), ("industry", ind)):
        missing = [q for q in QUARTERS if q not in p.index]
        if missing:
            raise SystemExit(f"  the {name}-clustered path is missing {missing}")
        if REFERENCE not in p.index:
            raise SystemExit("  the omitted quarter must be in the export")
    n_firms = int(emp["n_firms"].max())

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(QUARTERS)))
    for i, q in enumerate(QUARTERS):
        c = float(emp.loc[q, "coef"])
        se_e, se_i = float(emp.loc[q, "se"]), float(ind.loc[q, "se"])
        if q == REFERENCE:
            ax.plot(xs[i], 0, "o", ms=6.5, mfc="white", mec=DARK, mew=1.4,
                    zorder=5)
            continue
        ax.plot([xs[i], xs[i]], [c - 1.96 * se_i, c + 1.96 * se_i],
                color=MID, lw=2.6, alpha=0.6, zorder=2,
                label="95 per cent, industry-clustered" if i == 0 else None)
        ax.plot([xs[i], xs[i]], [c - 1.96 * se_e, c + 1.96 * se_e],
                color=DARK, lw=1.2, zorder=3,
                label="95 per cent, employer-clustered" if i == 0 else None)
        ax.plot(xs[i], c, "o", ms=6, mfc=DARK, mec=DARK, zorder=4)
    ax.plot(xs, [float(emp.loc[q, "coef"]) for q in QUARTERS], "-",
            color=DARK, lw=1.0, zorder=1)
    ax.axhline(0, color=DARK, lw=0.8, zorder=1)
    # the launch: December 2022 is the last month of 2022Q4
    ax.axvline(xs[-1] + 0.35, color=MID, lw=0.8, ls="--", zorder=1)
    ax.text(xs[-1] + 0.32, ax.get_ylim()[1] * 0.92, "launch", ha="right",
            va="top", fontsize=8, color=MID)
    ax.set_xticks(xs)
    ax.set_xticklabels([q.replace("Q", "\nQ") for q in QUARTERS], fontsize=9)
    ax.set_ylabel("Young women minus young men, exposed employers,\n"
                  "log points relative to 2022Q1", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(0.99, 0.02, f"{n_firms:,} employers; no calendar terms",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_prepath_female", __file__)
    for ext in (".pdf", ".png"):
        shutil.copy(REV / "figures" / f"fig_prepath_female{ext}",
                    PAPER / "figures" / f"fig_prepath_female{ext}")
    print("wrote fig_prepath_female.pdf/.png to revision/figures and the "
          "paper repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

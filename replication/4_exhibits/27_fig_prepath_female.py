#!/usr/bin/env python3
"""
27_fig_prepath_female.py: Online Appendix Figure A6 (Section III.2), the
female differential before the launch, raw.

The quarterly coefficient on young x exposed x female, January 2021 to
December 2022, each quarter against 2022Q1 (omitted), on the sex
specification without calendar-quarter terms (sex-specific employer-by-age,
employer-by-month and month-by-age effects), so that the seasonal cycle is
visible; script 97, part P, specs `path` (clustered by employer) and
`path_indcl` (clustered by three-digit industry). Whiskers are 95 per cent
intervals under each clustering. The drift test with the calendar terms on
is Table A21, Panel E.

Nothing is drawn unless both clusterings carry every quarter and the
omitted reference, the two clusterings agree in every coefficient, and every
coefficient and employer-clustered standard error agrees, to four decimals,
with the run's own summary.

Export read: 3_register_mona/exports/2026-09-25_2250_s97/
  headline_checks.csv, 97_summary.txt
Output: output/figures/fig_prepath_female.pdf and .png

    python 4_exhibits/27_fig_prepath_female.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS  # noqa: E402

DEFAULT = EXPORTS / "2026-09-25_2250_s97"

DARK = "#222222"
MID = "#888888"
REFERENCE = "2022Q1"
QUARTERS = ["2021Q1", "2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2",
            "2022Q3", "2022Q4"]
SUMMARY = re.compile(r"^\s*(\d{4}Q\d) ([-+][0-9.]+) \(([0-9.]+)\)")


def path(d: pd.DataFrame, spec: str) -> pd.DataFrame:
    p = d[(d["part"] == "P") & (d["spec"] == spec)
          & d["term"].str.contains(r"_x_hyf$|_reference$")].copy()
    p["quarter"] = p["term"].str.extract(r"pq_(\d{4}Q\d)")[0]
    p.loc[p["term"].str.endswith("_reference"), "quarter"] = REFERENCE
    return p.set_index("quarter")


def summary_path(text: str) -> dict:
    said, in_p = {}, False
    for line in text.splitlines():
        if line.startswith("P. THE FEMALE DIFFERENTIAL BEFORE THE LAUNCH"):
            in_p = True
            continue
        if in_p and (line.startswith("Q.") or line.startswith("VERDICTS")):
            break
        m = SUMMARY.match(line) if in_p else None
        if m:
            said[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return said


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
    said = summary_path((export_dir / "97_summary.txt").read_text(encoding="utf-8",
                                                                  errors="replace"))
    for q in QUARTERS:
        if q == REFERENCE:
            continue
        c, se = float(emp.loc[q, "coef"]), float(emp.loc[q, "se"])
        if abs(c - float(ind.loc[q, "coef"])) > 1e-9:
            raise SystemExit(f"  {q}: the two clusterings differ in the coefficient")
        if q not in said or round(c, 4) != said[q][0] or round(se, 4) != said[q][1]:
            raise SystemExit(f"  {q}: the export gives {c:+.4f} ({se:.4f}) and "
                             f"97_summary.txt {said.get(q)}; nothing is drawn")
    n_firms = set(int(x) for x in emp["n_firms"])
    if len(n_firms) != 1:
        raise SystemExit(f"  one employer count expected, {n_firms}")
    n_firms = n_firms.pop()
    print(f"  seven quarters reproduce 97_summary.txt; {n_firms:,} employers")

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
    print("  wrote output/figures/fig_prepath_female.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

#!/usr/bin/env python3
"""
l45_tab_fixed_contrasts.py: Online Appendix III.2, the headline read off
the flexible quarterly path instead of the period partition.

WHY THIS EXISTS
The paper's headline compares January 2024 to June 2025 with December
2022 to December 2023. A referee can object to that partition on three
grounds at once, and this table answers all three without a new fit:

  1. THE BOUNDARY. The decline is gradual and is already distinguishable
     from the tightening level in 2023Q4, so January 2024 does not sit on
     a break. Any contrast defined by a boundary inherits that choice.
  2. THE CALENDAR COMPOSITION. The two periods are thirteen and eighteen
     months long, and the longer one contains January to June twice but
     July to December once. Quarter-of-year terms remove a common
     seasonal cycle; they do not make two periods of unequal calendar
     composition comparable.
  3. THE LAUNCH QUARTER. The comparison period opens in December 2022,
     inside 2022Q4, whose other two months belong to the tightening
     benchmark.

A contrast between calendar 2024 and calendar 2023 is free of all three:
equal length, identical quarter composition, and no dependence on the
launch quarter or on the January boundary.

WHAT IT REPORTS
Three linear combinations of the quarterly path coefficients, each with a
standard error from the path fit's own clustered covariance, w'Vw:

  2024 against 2023        the four 2024 quarters less the four 2023
                           quarters, equally weighted. This is the
                           boundary-free counterpart of the headline.
  2024H2 against 2023H2    the same on the second halves alone, which is
                           where the movement is.
  2024Q1 against 2023Q4    the two quarters either side of the boundary.
                           It disciplines the language: if this is
                           indistinguishable from zero, nothing
                           discontinuous happens at January 2024 and the
                           paper must not write as though it does.

THESE ARE NOT THE HEADLINE RE-ESTIMATED. The pooled Poisson coefficient
weights employer-months; these weight calendar quarters equally. The two
answer the same question with different weights and they are reported
side by side for that reason, not because either corrects the other.

INPUTS AND OUTPUTS
Reads occ_route_path.csv and vcov_s84_path_quarter_<band>.csv from the
lane 30 export (script 84). Writes revision/tables/tableA_fixed_contrasts.tex
and copies it to canaries-sweden-paper/tables/.

    python3 revision/local/l45_tab_fixed_contrasts.py [export_dir]

IN THE PAPER
Online Appendix III.2, Table tab:fixed_contrasts.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE30 = OUT / "round3_20260923-0917-lane30-path"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"
BANDS = ["22-25", "26-30"]
TERM = "pq_{}_x_high_x_young"

CONTRASTS = [
    ("Calendar 2024 against calendar 2023",
     [f"2024Q{i}" for i in (1, 2, 3, 4)], [f"2023Q{i}" for i in (1, 2, 3, 4)]),
    ("Second half of 2024 against second half of 2023",
     ["2024Q3", "2024Q4"], ["2023Q3", "2023Q4"]),
    ("2024Q1 against 2023Q4, the quarters either side of the boundary",
     ["2024Q1"], ["2023Q4"]),
]


def source(name: str) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE30
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def main() -> int:
    rows = {}
    for band in BANDS:
        path = pd.read_csv(source("occ_route_path.csv"))
        g = path[(path.young_band == band) & (path["shape"] == "quarter")]
        if g.empty:
            raise SystemExit(f"  no quarterly path for {band}")
        bad = g[g.status != "ok"]
        if len(bad):
            raise SystemExit(f"  {band}: {len(bad)} quarters are not 'ok'; "
                             f"a contrast must not average a failed fit")
        g = g.set_index("period")
        v = pd.read_csv(source(f"vcov_s84_path_quarter_"
                               f"{band.replace('-', '_')}.csv"), index_col=0)

        # The covariance is the second record: every quarter's own
        # standard error must be the square root of its diagonal, or the
        # matrix is not the one that fit produced and no w'Vw from it is
        # trustworthy.
        for q in g.index:
            t = TERM.format(q)
            if t not in v.index:
                raise SystemExit(f"  {band}: {t} is not in the covariance")
            said, got = float(np.sqrt(float(v.loc[t, t]))), float(g.loc[q, "se"])
            if round(said, 6) != round(got, 6):
                raise SystemExit(
                    f"  {band} {q}: the export reports {got:.6f} and its own "
                    f"covariance {said:.6f}; the table is not written")
        print(f"  {band}: the covariance reproduces every quarterly standard "
              f"error to 6 decimals")

        out = []
        for label, pos, neg in CONTRASTS:
            w = pd.Series(0.0, index=v.index)
            for q in pos:
                w[TERM.format(q)] += 1.0 / len(pos)
            for q in neg:
                w[TERM.format(q)] -= 1.0 / len(neg)
            c = sum(float(w[TERM.format(q)]) * float(g.loc[q, "coef"])
                    for q in set(pos) | set(neg))
            var = float(w.values @ v.values @ w.values)
            if var <= 0:
                raise SystemExit(f"  {band} {label}: non-positive variance")
            se = float(np.sqrt(var))
            out.append((label, c, se))
            print(f"    {label[:46]:46s} {c:+.4f} ({se:.4f})  "
                  f"t {c/se:+.2f}  {100*(np.exp(c)-1):+.2f} per cent")
        rows[band] = out

    def cell(c, se):
        star = "^{*}" if abs(c) > 1.96 * se else ""
        return f"${c:+.4f}{star}$ ({se:.4f})"

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The decline read off the quarterly path, on contrasts "
           r"that do not depend on where the adoption window is judged to "
           r"begin.}",
           r"\label{tab:fixed_contrasts}", r"\footnotesize",
           r"\begin{tabular}{lcc}", r"\toprule",
           r"Contrast & Ages 22--25 & Ages 26--30 \\", r"\midrule"]
    for i, (label, _, _) in enumerate(CONTRASTS):
        tex.append(f"{label} & "
                   + " & ".join(cell(rows[b][i][1], rows[b][i][2])
                                for b in BANDS) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}"
            r"Linear combinations of the quarterly coefficients of "
            r"Figure~3 of the paper, with standard errors $\sqrt{w'Vw}$ from "
            r"that fit's own clustered covariance. Each row compares periods "
            r"of equal length and identical calendar-quarter composition, so "
            r"none of them depends on the January 2024 boundary, on the "
            r"unequal lengths of the periods Equation~(2) partitions, or on "
            r"the launch quarter. The first row is the boundary-free "
            r"counterpart of the headline; it is not the headline "
            r"re-estimated, since the pooled Poisson coefficient weights "
            r"employer-months where these weight quarters equally. The last "
            r"row is the pair of quarters either side of the boundary. "
            r"$^{*}$ $p<0.05$. Source: script 84.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_fixed_contrasts.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV.parent)}")
    if PAPER_TAB.exists():
        shutil.copy2(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

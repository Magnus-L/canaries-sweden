#!/usr/bin/env python3
"""
13_tab_fixed_contrasts.py: Online Appendix Table A12 (Section III.2), the
decline read off the quarterly path on contrasts that do not depend on where
the adoption window begins.

Three linear combinations of the quarterly coefficients of Figure 3 (script
84), each with a standard error sqrt(w'Vw) from the path fit's own clustered
covariance: calendar 2024 against calendar 2023, the second half of 2024
against the second half of 2023, and 2024Q1 against 2023Q4. Each compares
periods of equal length and identical calendar-quarter composition, so none
depends on the January 2024 boundary, on the unequal lengths of the periods of
Equation (2), or on the launch quarter. They weight quarters equally where the
pooled Poisson coefficient weights employer-months, so they are not the
headline re-estimated. Every quarterly standard error must equal the square
root of its own covariance diagonal to six decimals.

Exports read: 3_register_mona/exports/2026-09-23_0917_s84/
  occ_route_path.csv, vcov_s84_path_quarter_<band>.csv
Output: output/tables/tableA_fixed_contrasts.tex

    python 4_exhibits/13_tab_fixed_contrasts.py [export_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S84 = EXPORTS / "2026-09-23_0917_s84"
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
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else S84
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
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}Linear combinations of the quarterly coefficients of Figure~3 of the paper, with standard errors from that fit's clustered covariance. The first two rows compare periods of equal length and identical calendar-quarter composition without using the January 2024 boundary; the third compares the quarters either side of that boundary, net of the estimated calendar terms. The first row is the boundary-free counterpart of the headline, weighting quarters equally where the pooled coefficient weights employer-months. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_fixed_contrasts.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

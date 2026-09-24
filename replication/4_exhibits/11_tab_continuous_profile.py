#!/usr/bin/env python3
"""
11_tab_continuous_profile.py: Online Appendix Table A14 (Section III.2), the
age profile on continuous occupation-scaled exposure measures.

Six age bands on three continuous measures (DAIOE; the model-graded beta
rating of Eloundou et al., 2024; the teleworkability classification of Dingel
and Neiman, 2020), each crosswalked from O*NET SOC 2010 to ISCO-08 to SSYK
2012 on unweighted means at each step. Poisson with employer-by-month,
employer-by-age and month-by-age effects, treatment January 2024, clustered by
employer; a coefficient is per standard deviation of the 2019 firm-age
baseline on that measure's own scale (script 63, adoption-dated rows). A
different estimand from the paper's, reported as a check on the measure.
Stars use the exported p-value.

Export read: 3_register_mona/exports/2026-09-20_2148_s61-s63/output_63__robustness_gradient.csv
Output: output/tables/tableA_age_profile.tex

    python 4_exhibits/11_tab_continuous_profile.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S63 = EXPORTS / "2026-09-20_2148_s61-s63"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]


def source(default_dir: Path, name: str) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def fmt(c, se, p=None):
    """Estimate with a star at five per cent, from the exported p-value
    where the export carries one and from the normal threshold otherwise."""
    sig = (p < 0.05) if p is not None else (abs(c) > 1.96 * se)
    star = "^{*}" if sig else ""
    return f"${c:+.4f}{star}$ ({se:.4f})"


def panel_b():
    """Six bands, continuous occupation-scaled measures."""
    d = pd.read_csv(source(S63, "output_63__robustness_gradient.csv"))
    d = d[d.dating == "adoption"]
    out = []
    for b in BANDS:
        cells = []
        for m in ("daioe", "eloundou", "telework"):
            r = d[(d.age_group == b) & (d.measure == m) & (d.outcome == "stock")]
            cells.append("--" if r.empty else
                         fmt(float(r.coef.iloc[0]), float(r.se.iloc[0]),
                             float(r.pvalue.iloc[0])))
        out.append((b, cells))
    return out


def main() -> int:
    B = panel_b()

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The age profile on a continuous occupation-scaled "
           r"exposure measure.}",
           r"\label{tab:age_profile_rebuilt}", r"\footnotesize",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Age band & DAIOE & Eloundou & Teleworkable \\", r"\midrule"]
    for b, cells in B:
        tex.append(f"{b} & " + " & ".join(cells) + r" \\")
        print(f"  {b:6s} " + "  ".join(c.replace('$', '') for c in cells))
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson with employer-by-month, employer-by-age and month-by-age effects, treatment January 2024, clustered by employer; no calendar terms. Occupations are scored continuously, so exposure varies within an employer-month, and each coefficient is per standard deviation of that measure. The Eloundou column is the $\beta$ rating of \citet{eloundou2024gpts}; the teleworkable column is the classification of \citet{dingel2020many}; both reach SSYK by the crosswalk chain of Section~\ref{sec:datasources}. A check on the measure, not a second estimate of the step. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]
    out = TABLES / "tableA_age_profile.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

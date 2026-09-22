#!/usr/bin/env python3
"""
l25_tab_industry_credit.py -- the appendix table that tests two rival
explanations for the decline of the young inside exposed employers.

THE QUESTION. Could the within-employer age pattern be an industry-specific
age shock (a sector that stopped hiring the young for reasons unrelated to
AI), or the credit channel of the rate cycle (leveraged employers cutting
their youngest staff as borrowing costs rose)?

WHAT IS ESTIMATED. Poisson pseudo-maximum likelihood on employer x age x
month counts with employer-by-month, employer-by-age and month-by-age
effects, exposure frozen at the employer's 2019 education mix, treatment
dated January 2024; the calendar cycle is not removed here, so every row
is read against the pre-cycle baseline in the same panel and never against
Table 1 of the paper.

  Panel A  The adoption step with and without three-digit industry (2019)
           interacted with age band and month. "Retained" is the
           industry-controlled step over the baseline; the gate fixed before
           the run was 50 per cent.
  Panel B  On the employers with a 2019 balance sheet, the adoption step
           interacted with an indicator for above-median leverage (1 minus
           equity over assets), beside a term for leverage x young that is
           common to all employers in the sample. The step averaged over the
           two halves is the exposure step plus half the additional term,
           with its standard error from the exported covariance, and it is
           the number to compare with the baseline on the same sample. Read
           rule fixed before the run: MONETARY if the averaged step loses
           more than half of that baseline, AI SURVIVES otherwise.

INPUTS, pinned to the export directories the final-code manifest names.

  lane 19  industry_fe.csv                  Panel A (script 73, Part A)
  lane 24  credit_test.csv, vcov_r73_lev_*  Panel B (script 73, Part B,
                                            re-run on the 2019 close)

OUTPUT. revision/tables/tableA_industry_credit.tex, copied to the manuscript
repository's tables/ folder.

    python3 revision/local/l25_tab_industry_credit.py
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
LANE19 = OUT / "round3_20260921-lane19-final"
LANE24 = OUT / "round3_20260922-0949-lane24-credit"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30"]
EXPO = "post_x_high_x_young"
TRIPLE = "post_x_high_x_young_x_lev"
LEV = "post_x_young_x_lev"
SAME = "baseline_on_balance_sheet_sample"

# Stated in the lane 24 summary (73_summary.txt); quoted in the note.
SPLIT = {"22-25": 0.691, "26-30": 0.688}
COVERAGE = {"22-25": 82.7, "26-30": 80.1}


def fmt(c, se):
    star = "^{*}" if abs(c) > 1.96 * se else ""
    return f"${c:+.4f}{star}$ ({se:.4f})"


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def main() -> int:
    ind = pd.read_csv(need(LANE19 / "industry_fe.csv"))
    cred = pd.read_csv(need(LANE24 / "credit_test.csv"))
    a, b = {}, {}
    for band in BANDS:
        i = ind[ind.band == band].set_index("spec")
        base, withind = i.loc["baseline"], i.loc["industry_age_t"]
        a[band] = {"base": (base.coef, base.se),
                   "ind": (withind.coef, withind.se),
                   "kept": 100 * withind.coef / base.coef}
        c = cred[cred.band == band].set_index("term")
        v = pd.read_csv(need(LANE24 / f"vcov_r73_lev_{band}.csv"), index_col=0)
        avg = c.loc[EXPO, "coef"] + 0.5 * c.loc[TRIPLE, "coef"]
        avg_se = np.sqrt(v.loc[EXPO, EXPO] + 0.25 * v.loc[TRIPLE, TRIPLE]
                         + v.loc[EXPO, TRIPLE])
        b[band] = {"same": (c.loc[SAME, "coef"], c.loc[SAME, "se"]),
                   "expo": (c.loc[EXPO, "coef"], c.loc[EXPO, "se"]),
                   "triple": (c.loc[TRIPLE, "coef"], c.loc[TRIPLE, "se"]),
                   "avg": (avg, avg_se),
                   "lev": (c.loc[LEV, "coef"], c.loc[LEV, "se"]),
                   "kept": 100 * avg / c.loc[SAME, "coef"]}
    verdict = {band: ("AI SURVIVES" if b[band]["kept"] >= 50 else "MONETARY")
               for band in BANDS}
    for band in BANDS:
        print(f"  {band}: industry retains {a[band]['kept']:.0f}%; credit: averaged "
              f"step {b[band]['avg'][0]:+.4f} ({b[band]['avg'][1]:.4f}) = "
              f"{b[band]['kept']:.0f}% of the same-sample baseline; {verdict[band]}")

    def row(label, key, src):
        return f"{label} & " + " & ".join(fmt(*src[band][key]) for band in BANDS) + r" \\"

    tex = [
        r"\begin{table}[ht!]", r"\centering",
        r"\caption{An industry-specific age shock and the credit channel as rival "
        r"explanations for the decline of the young inside exposed employers.}",
        r"\label{tab:industry_credit}", r"\footnotesize",
        r"\begin{tabular}{lcc}", r"\toprule",
        r" & 22--25 & 26--30 \\", r"\midrule",
        r"\multicolumn{3}{l}{\textit{Panel A. An industry-specific age shock}} \\",
        row("Baseline", "base", a),
        row(r"With industry $\times$ age $\times$ month", "ind", a),
        "Retained (per cent) & " + " & ".join(f"{a[band]['kept']:.0f}" for band in BANDS) + r" \\",
        r"\addlinespace[4pt]",
        r"\multicolumn{3}{l}{\textit{Panel B. The credit channel, employers with a 2019 balance sheet}} \\",
        row("Baseline on this sample", "same", b),
        row("Adoption step, less leveraged half", "expo", b),
        row("Additional step, more leveraged half", "triple", b),
        row("Adoption step averaged over the halves", "avg", b),
        row(r"Leverage $\times$ young, all employers in the sample", "lev", b),
        "Retained, averaged step over baseline (per cent) & "
        + " & ".join(f"{b[band]['kept']:.0f}" for band in BANDS) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.9\textwidth}\footnotesize\vspace{4pt}",
        r"Poisson pseudo-maximum likelihood, employer-by-month, employer-by-age and "
        r"month-by-age effects, exposure frozen at the employer's 2019 education mix, "
        r"treatment dated January 2024, standard errors clustered by employer; the "
        r"calendar cycle is not removed in these specifications, so every row is read "
        r"against the baseline in its own panel, never against Table~1 of the paper. "
        r"Industry is the employer's three-digit NACE in 2019 (265 groups, carried by "
        r"95 per cent of the 22--25 panel and 93 per cent of the 26--30 panel), "
        r"interacted with age band and month; retained is the industry-controlled step "
        r"over the baseline, against a gate of 50 per cent fixed before the run. "
        r"Leverage is one minus equity over assets from the employer's latest balance "
        r"sheet closing in 2019 (Serrano), carried by "
        f"{COVERAGE['22-25']:.1f} and {COVERAGE['26-30']:.1f} per cent of the two panels; "
        f"the split is at the sample median, {SPLIT['22-25']:.3f} and {SPLIT['26-30']:.3f}. "
        r"The adoption step is interacted with the above-median indicator, so the first "
        r"credit row is the step among the less leveraged exposed employers and the second "
        r"the additional step among the more leveraged; the averaged step is the first "
        r"plus half the second, with its standard error from the estimated covariance, "
        r"and it is compared with the baseline estimated on the same employers. The "
        r"leverage $\times$ young row is the credit channel common to every employer in "
        r"the sample. Read rule fixed before the run: the exposure step is judged monetary "
        r"if the averaged step loses more than half of the same-sample baseline; it "
        f"retains {b['22-25']['kept']:.0f} and {b['26-30']['kept']:.0f} per cent. "
        r"$^{*}$ $p<0.05$. Source: script 73 (\texttt{industry\_fe.csv}, lane 19; "
        r"\texttt{credit\_test.csv} and \texttt{vcov\_r73\_lev\_*.csv}, lane 24).",
        r"\end{minipage}", r"\end{table}"]
    out = V2_TAB / "tableA_industry_credit.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l21_tab_partIV.py: the three tables of Online Appendix Part IV, on the
coverage of the occupation register.

WHAT THE TABLES REPORT
  tableIV1_coverage.tex   Panel A, the share of employment excluded for
                          want of any code, by year and age band; Panel B,
                          the age of the code in 2023, 2024 and 2025 as a
                          share of employment; Panel C, the match rate by
                          worker group (incumbent, recent hire, entrant)
                          pooled over 2020 to 2023. Source: script 40.
  tableIV2_vintage.tex    the half-year event study of the submitted
                          design estimated separately on workers coded
                          from the 2021, 2022 and 2023 registers, ages
                          22-25. Source: script 41.
  tableIV3_backtest.tex   the as-of backtest: the true and as-of
                          coefficients at the two truncations and the
                          artefact, the as-of coefficient minus the true
                          one. Source: script 45. The note also quotes the
                          vintage re-scoring of the reported design (the
                          as-of arm of script 68), which is the
                          sensitivity that applies to it.

The artefact is the gap between the two columns and not the as-of
coefficient; the tables print all three so the distinction cannot be
lost.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names:
excluded_counts.csv, vintage_composition.csv and
entrant_split_coverage.csv (script 40); output_41__vintage_es.csv and
output_41__margin_pair_counts.csv (script 41); asof_estimates.csv (script
45); seasonal_pooled.csv (script 68). Writes the three .tex files to
revision/tables/ and copies them to canaries-sweden-paper/tables/.

    python3 revision/local/l21_tab_partIV.py

IN THE PAPER
Online Appendix IV.1 (tab:iv_coverage), IV.2 (tab:iv_vintage) and IV.3
(tab:iv_backtest).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
B40 = OUT / "round2_20260918-1640"
B45 = OUT / "round2_20260918-1736-script45"
B41 = OUT / "round2_20260920-exportpack"
LANE14 = OUT / "round3_20260921-2152-lane14-seasonal-complete"
# Script 82 part C, the occupation route's own vintage arms.
LANE28C = OUT / "round3_20260922-2327-lane28c"
# The manuscript repository is a sibling of this one; the appendix
# \input{}s the tables from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"
BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]


def t1():
    """Coverage: complete by construction after 2019, and the cost is staleness."""
    ex = pd.read_csv(B40 / "excluded_counts.csv")
    ex["year"] = ex["year_month"].str[:4]
    exs = ex.groupby(["year", "age_group"]).excluded_share.mean().unstack() * 100
    vc = pd.read_csv(B40 / "vintage_composition.csv")
    vc["year"] = vc["year_month"].str[:4]
    esc = pd.read_csv(B40 / "entrant_split_coverage.csv")
    grp = esc[esc.year.between(2020, 2023)].groupby("group")[["n_coded", "n_pairs"]].sum()
    grp["rate"] = 100 * grp.n_coded / grp.n_pairs

    # age of the code, in years, for each panel year
    agepct = {}
    for y in ("2023", "2024", "2025"):
        d = vc[vc.year == y]
        if d.empty:
            continue
        tot = d.n_emp.sum()
        agepct[y] = {int(y) - int(v): 100 * g.n_emp.sum() / tot
                     for v, g in d.groupby("vintage")}

    L = [r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
         r"\caption{Occupation-code coverage: complete after 2019, and the "
         r"cost is the age of the code.}", r"\label{tab:iv_coverage}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"\multicolumn{5}{l}{\textit{Panel A. Share of employment excluded "
         r"for want of any code, by age band}} \\", r"\addlinespace[2pt]",
         r"Year & 22--25 & 26--30 & 41--49 & 50+ \\", r"\midrule"]
    for y in ("2019", "2021", "2023", "2025"):
        if y not in exs.index:
            continue
        L.append(f"{y} & " + " & ".join(
            f"{exs.loc[y, b]:.2f}\\%" for b in ("22-25", "26-30", "41-49", "50+")) + r" \\")
    L += [r"\addlinespace[6pt]",
          r"\multicolumn{5}{l}{\textit{Panel B. Age of the occupation code, "
          r"share of all employment}} \\", r"\addlinespace[2pt]",
          r"Year & Current & One year old & Two years old & Older \\",
          r"\midrule"]
    for y in ("2023", "2024", "2025"):
        if y not in agepct:
            continue
        a = agepct[y]
        cells = [a.get(0, 0.0), a.get(1, 0.0), a.get(2, 0.0),
                 sum(v for k, v in a.items() if k >= 3)]
        L.append(f"{y} & " + " & ".join(f"{c:.1f}\\%" for c in cells) + r" \\")
        print(f"   {y} code age:", {k: round(v, 1) for k, v in sorted(a.items())})
    L += [r"\addlinespace[6pt]",
          r"\multicolumn{5}{l}{\textit{Panel C. Match rate by worker group, "
          r"2020--2023 pooled}} \\", r"\addlinespace[2pt]",
          r"Group & \multicolumn{4}{c}{Share carrying a code} \\", r"\midrule"]
    for g, lab in (("incumbent", "Incumbent"), ("recent_hire", "Recent hire"),
                   ("entrant", "Entrant, first panel year")):
        if g in grp.index:
            L.append(f"{lab} & \\multicolumn{{4}}{{c}}"
                     f"{{{grp.loc[g, 'rate']:.1f}\\%}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}"
          r"Panel A: the share of employed workers for whom no occupation code "
          r"can be assigned from any register vintage. It is non-zero only in "
          r"2019, the first panel year, where a worker's employment spell is "
          r"left-censored; from 2020 it is zero to two decimal places. "
          r"\textbf{Coverage is therefore not the problem.} The cascade "
          r"assigns every worker a code by carrying the most recent one "
          r"forward, so nobody drops out; what changes is how old that code "
          r"is. Panel B: by 2025 no worker carries a code recorded in the "
          r"current year, because the register is published with a two-year "
          r"lag, and essentially all of them carry the 2023 code. Panel C: "
          r"entrants are the worst-covered group, which matters because young "
          r"workers are disproportionately entrants. What the lag does to an "
          r"estimate is measured in Section~\ref{sec:asof}. Source: script 40."
          r"\end{minipage}", r"\end{table}"]
    (V2_TAB / "tableIV1_coverage.tex").write_text("\n".join(L) + "\n")
    print("  tableIV1_coverage.tex")
    print("   match rates:", grp["rate"].round(1).to_dict())


def t2():
    """Event studies run separately on each code vintage."""
    ve = pd.read_csv(B41 / "output_41__vintage_es.csv")
    mp = pd.read_csv(B41 / "output_41__margin_pair_counts.csv")
    L = [r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
         r"\caption{The half-year event study estimated separately on workers "
         r"coded from each register vintage, ages 22--25.}",
         r"\label{tab:iv_vintage}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Period & Coded from 2021 & Coded from 2022 & Coded from 2023 \\",
         r"\midrule"]
    per = sorted(ve.period.unique())
    for p in per:
        row = [p]
        for v in ("V2021_22-25", "V2022_22-25", "V2023_22-25"):
            r = ve[(ve.period == p) & (ve.variant == v)]
            row.append("--" if r.empty else
                       f"${float(r.iloc[0]['coef']):+.3f}$ ({float(r.iloc[0]['se']):.3f})")
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.90\textwidth}\footnotesize\vspace{4pt}"
          r"Each column restricts to workers whose occupation code comes from "
          r"the stated register year, and runs the submitted employer-level "
          r"event study on that subsample. \textbf{The columns are not "
          r"comparable and none of them estimates a treatment effect.} "
          r"Conditioning on code vintage conditions on how recently a worker's "
          r"occupation was observed, which is itself a function of tenure, "
          r"entry and mobility; the subsamples shrink sharply in the later "
          r"periods, which is why the standard errors explode and the point "
          r"estimates reach implausible magnitudes. The exercise is reported "
          r"because the editor asked for it and because the instability is "
          r"the answer: a design whose estimate depends this strongly on which "
          r"register year supplied the code is not identified. Source: "
          r"script 41."
          r"\end{minipage}", r"\end{table}"]
    (V2_TAB / "tableIV2_vintage.tex").write_text("\n".join(L) + "\n")
    print("  tableIV2_vintage.tex  (pairs 2022: "
          f"{mp[mp.year == 2022].n_pairs.sum():,})")


def rescoring_artefact() -> tuple[float, float]:
    """The vintage-sensitivity check on the reported design, script 82's
    cascade arm: the change in its adoption step when the score is
    restricted to the codes observed in 2019, rather than carried back
    from earlier files, and the share of employers that keep their
    quartile under that restriction.

    THIS IS NOT A LATER-VINTAGE RE-SCORING AND MUST NOT BE WRITTEN AS
    ONE. The arm that re-scores the 2019 incumbents from the 2021
    register scored zero employers on this route and produced no fit, so
    the occupation route has no later-vintage check to quote. The
    education route had one, +0.0113, and it is not this design's.
    """
    d = pd.read_csv(LANE28C / "occ_route_vintage.csv")
    fit = d[(d.block == "fit") & (d.arm == "cascade_back_2019")
            & (d.item == "post_x_high_x_young")]
    only = d[(d.block == "fit") & (d.arm == "code_2019_only")
             & (d.item == "post_x_high_x_young")]
    keep = d[(d.block == "stability")
             & (d.item == "share_keeping_quartile")]
    if len(fit) != 1 or len(only) != 1 or len(keep) != 1:
        raise SystemExit("  occ_route_vintage.csv: expected one cascade "
                         "fit, one 2019-only fit and one quartile share")
    return (float(only.coef.iloc[0]) - float(fit.coef.iloc[0]),
            float(keep.value.iloc[0]))
    return asof - true


def t3():
    """The as-of backtest. Artefact = as-of minus true."""
    a = pd.read_csv(B45 / "asof_estimates.csv")
    rescored, keep_share = rescoring_artefact()
    L = [r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
         r"\caption{The as-of backtest: what the register's lag alone "
         r"produces in years where the true age gap is observable.}",
         r"\label{tab:iv_backtest}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Register truncated at & True codes & As-of codes & Artefact \\",
         r"\midrule"]
    for t in sorted(a.trunc.unique()):
        tr = a[(a.trunc == t) & (a.assignment == "true")].iloc[0]
        af = a[(a.trunc == t) & (a.assignment == "asof")].iloc[0]
        L.append(f"{t} & ${tr.gamma2:+.4f}$ ({tr.se2:.4f}) & "
                 f"${af.gamma2:+.4f}$ ({af.se2:.4f}) & "
                 f"$\\mathbf{{{af.gamma2 - tr.gamma2:+.4f}}}$ \\\\")
        print(f"   T={t}: true {tr.gamma2:+.4f}  asof {af.gamma2:+.4f}  "
              f"artefact {af.gamma2 - tr.gamma2:+.4f}")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.90\textwidth}\footnotesize\vspace{4pt}"
          r"The coefficient is the submitted design's $\hat\gamma_2$ on "
          r"2019--2023, years in which every worker's true contemporaneous "
          r"occupation is observed. The as-of column re-runs it having "
          r"truncated the register at the stated year and rebuilt the coding "
          r"cascade as it would have been, which imposes on those years "
          r"exactly the staleness that 2024--25 inherit. \textbf{The artefact "
          r"is the gap between the two columns, not the as-of coefficient.} "
          r"At the 2021 truncation the lag moves an estimate of $+0.019$, "
          r"indistinguishable from zero, to $-0.288$; the artefact is "
          r"$-0.307$. A design of that kind returns about $-0.17$ on these data, "
          r"so the lag alone can produce more than the whole of it. "
          r"The design the paper "
          r"reports admits no occupation code recorded after 2019, so this "
          r"test does not apply to it. Its own 2019 codes are not all fresh: "
          r"restricting the score to those observed in 2019, rather than "
          r"carried back from earlier files, moves its adoption step by "
          f"${rescored:+.3f}$, and {100 * keep_share:.1f} per cent of "
          r"employers keep their quartile. Source: scripts 45, 68 and 82."
          r"\end{minipage}", r"\end{table}"]
    (V2_TAB / "tableIV3_backtest.tex").write_text("\n".join(L) + "\n")
    print(f"  tableIV3_backtest.tex  (vintage restriction {rescored:+.4f}, "
          f"{100 * keep_share:.1f} per cent keep their quartile)")


def main() -> int:
    t1(); t2(); t3()
    print(f"\n  wrote 3 tables to {V2_TAB.relative_to(REV)}")
    if PAPER_TAB.exists():
        for name in ("tableIV1_coverage.tex", "tableIV2_vintage.tex",
                     "tableIV3_backtest.tex"):
            shutil.copy(V2_TAB / name, PAPER_TAB / name)
        print(f"  copied to {PAPER_TAB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

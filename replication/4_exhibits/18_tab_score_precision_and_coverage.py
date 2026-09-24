#!/usr/bin/env python3
"""
18_tab_score_precision_and_coverage.py: Online Appendix Table A22 (Section
III.3, the precision of the firm score) and Table A28 (Section IV.4, the codes
the exposure score is built from).

tableA_occ_coverage.tex: for incumbents aged 31 to 69 on the 2019 payroll, by
age band, the share coded from the 2019 register alone, the share coded with
the carry-back to 2018-2015, and the share the DAIOE index scores; the
employers scored at floors of one, three and five person-months; and the share
of employers that change quartile, with the rank correlation of the firm
scores, when the three-digit book is replaced by four digits where usable or by
four digits only (script 82, part A). The rows are read from the backward
(reported) arm of occ_route_coverage.csv, not from the forward arm the
appendix export labels "used".

tableA_size_reliability.tex: Panel A, the reliability of the firm score at
quantiles of the incumbent count, by employers and by incumbent employment, and
the shares below a reliability of one half; Panel B, the adoption step at the
reported floor, at a floor of sixty person-months and in two firm-size groups
(1 to 3 and 4 or more incumbents; the intended terciles degenerate), with the
national quartile carried in unchanged (script 83, parts C and D).

Two numbers in the coverage note are typed, not read: the 2.5 per cent of
resolved codes taken from a year before 2019, and the 7.4 per cent of the
average employer's codes taken from those years (script 82's summary).

Exports read (3_register_mona/exports/):
  2026-09-22_2232_s82-partA/  occ_route_coverage.csv,
      occ_route_appendix_coverage.csv
  2026-09-23_0655_s82-partB_s83-partsBCD/  occ_rest_reliability.csv,
      occ_rest_size.csv
Outputs: output/tables/tableA_occ_coverage.tex, tableA_size_reliability.tex

    python 4_exhibits/18_tab_score_precision_and_coverage.py
"""
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S82A = EXPORTS / "2026-09-22_2232_s82-partA"
S83 = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
TERM = "post_x_high_x_young"


def write(name: str, lines: list) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {p.relative_to(PACKAGE)}")


def n(v) -> str:
    return "" if v != v else f"{int(round(float(v))):,}"


def sh(v) -> str:
    return "" if v != v else f"{float(v) * 100:.1f}"


def coverage() -> None:
    """
    The coverage table, built from occ_route_coverage.csv and NOT from the
    "used" rows of occ_route_appendix_coverage.csv.

    THE TRAP, and why this function reads the other file. In the appendix
    export the year labelled "used" is the FORWARD arm, which adds 2020 and
    2021: its resolved share is 96.3 per cent against the reported score's
    94.4. Building the table from those rows would print the coverage of an
    arm the paper does not report and overstate the reported one by two
    points. The coverage block of occ_route_coverage.csv carries all three
    arms explicitly, so the table is built from there and the backward arm
    is named in the column head.
    """
    c = pd.read_csv(S82A / "occ_route_coverage.csv")
    cov = c[c.block == "coverage"].set_index(["group", "item"])
    bands = [("31-34", "31--34"), ("35-40", "35--40"), ("41-49", "41--49"),
             ("50+", "50 and over"),
             ("31-69 incumbents", "All incumbents, 31--69")]
    # The employer rows come from the floor and loss blocks of the same
    # file, so every count in the table is on the reported backward arm.
    fl = c[(c.block == "floor") & (c.item == "n_employers")] \
        .set_index("group")["n_employers"]
    ls = c[(c.block == "loss") & (c.panel == "all employers")] \
        .set_index("item" if "all" not in set(c["group"]) else "group")
    arm = pd.read_csv(S82A / "occ_route_appendix_coverage.csv")
    arm = arm[(arm.year == "used") & (arm.unit == "employers")
              & (arm.age_band == "total")].set_index("metric")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{What the exposure score is built from: the 2019 "
           r"occupation codes of the employer's own incumbents}",
           r"\label{tab:occ_coverage}", r"\scriptsize",
           r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r" & Incumbents & Coded from & Coded, with & Scored \\",
           r" & & 2019 alone (\%) & carry-back (\%) & (\%) \\",
           r"\midrule"]
    for key, label in bands:
        if (key, "coded_share_backward") not in cov.index:
            continue
        r = cov.loc[key]
        if label.startswith("All"):
            tex.append(r"\midrule")
        tex.append(f"{label} & {n(r.loc['coded_share_backward', 'n_obs'])} & "
                   f"{sh(r.loc['coded_share_2019_only', 'share'])} & "
                   f"{sh(r.loc['coded_share_backward', 'share'])} & "
                   f"{sh(r.loc['scored_share_backward', 'share'])} \\\\")
    tex += [r"\addlinespace[3pt]", r"\midrule",
            r"\multicolumn{5}{l}{\textit{Employers, and the floor}} \\",
            f"With an incumbent aged 31 to 69 & "
            f"{n(ls.loc['all', 'n_employers'])} & & & \\\\"]
    for f_, lab in ((1, "one person-month"), (3, "three person-months"),
                    (5, "five person-months")):
        star = " (reported)" if f_ == 5 else ""
        tex.append(f"\\quad scored at a floor of {lab}{star} & "
                   f"{n(fl.loc[f'floor_{f_}'])} & & & \\\\")
    tex += [r"\addlinespace[3pt]", r"\midrule",
            r"\multicolumn{5}{l}{\textit{What the scoring level changes}} "
            r"\\"]
    for m, lab in (("quartile_changed_vs_mixed43",
                    r"Employers moving quartile against "
                    r"four-digits-where-usable (\%)"),
                   ("spearman_uniform3_mixed43",
                    r"\quad rank correlation of the two firm scores"),
                   ("quartile_changed_vs_four_only",
                    r"Employers moving quartile against four-digits-only "
                    r"(\%)"),
                   ("spearman_uniform3_four_only",
                    r"\quad rank correlation of the two firm scores")):
        if m not in arm.index:
            continue
        r = arm.loc[m]
        v = (f"{float(r['share']) * 100:.1f}" if "quartile" in m
             else f"{float(r['value']):+.3f}")
        tex.append(f"\\multicolumn{{4}}{{l}}{{{lab}}} & {v} \\\\")
    note = r"Incumbents are the employees aged 31 to 69 on the employer's 2019 payroll, the workers the score averages over. A code is taken from the 2019 register and otherwise from 2018 back to 2015, the carry-back column; 2.5 per cent of resolved codes come from those earlier years, and none from a later file. The average employer draws 7.4 per cent of its codes from them, because small employers depend more on the older files. Scored is the share the DAIOE index can price. Every incumbent is scored from the three-digit book, and the last block reports what scoring at four digits changes."
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    write("tableA_occ_coverage.tex", tex)


def size_reliability() -> None:
    rel = pd.read_csv(S83 / "occ_rest_reliability.csv")
    size = pd.read_csv(S83 / "occ_rest_size.csv")
    q = rel[rel.block == "reliability_by_firm_quantile"]
    e = rel[rel.block == "reliability_by_employment_quantile"]
    thin = rel[rel.block == "thin"].iloc[0]
    v = rel[rel.block == "variance"].set_index("item")["value"]
    p = size[size.term == TERM]

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The precision of a firm score averaged over its "
           r"incumbents, and what it does to the adoption step}",
           r"\label{tab:size_reliability}", r"\footnotesize",
           r"\begin{tabular}{lrr}", r"\toprule",
           r"\multicolumn{3}{l}{\textit{Panel A. Reliability of the firm "
           r"score}} \\",
           r" & Incumbents & Reliability \\", r"\midrule"]
    for _, r in q.iterrows():
        pc = r["item"].split("_")[0][1:]
        tex.append(f"{pc}th percentile of employers & "
                   f"{int(r.n_incumbents):,} & {float(r.value):.2f} \\\\")
    for _, r in e.iterrows():
        pc = r["item"].split("_")[0][1:]
        tex.append(f"{pc}th percentile of incumbent employment & "
                   f"{int(r.n_incumbents):,} & {float(r.value):.3f} \\\\")
    tex.append(f"\\addlinespace[2pt]\nBelow a reliability of one half & "
               f"{float(thin.share_firms) * 100:.1f}\\% of employers & "
               f"{float(thin.share_employment) * 100:.1f}\\% of employment "
               f"\\\\")
    tex += [r"\addlinespace[4pt]", r"\midrule",
            r"\multicolumn{3}{l}{\textit{Panel B. The adoption step by the "
            r"number of incumbents behind the score}} \\",
            r" & Ages 22--25 & Ages 26--30 \\", r"\midrule"]
    lab = {"floor_5": "Reported floor, five incumbent person-months",
           "floor_60": "Floor of sixty person-months",
           "tercile_2": "Employers with 1 to 3 incumbents",
           "tercile_3": "Employers with 4 or more incumbents"}
    for spec in ("floor_5", "floor_60", "tercile_2", "tercile_3"):
        cells = []
        for band in ("22-25", "26-30"):
            r = p[(p.spec == spec) & (p.young_band == band)]
            cells.append(f"${float(r.coef.iloc[0]):+.4f}$ "
                         f"({float(r.se.iloc[0]):.4f})" if len(r) else "")
        tex.append(f"{lab[spec]} & {cells[0]} & {cells[1]} \\\\")
        cells = []
        for band in ("22-25", "26-30"):
            r = p[(p.spec == spec) & (p.young_band == band)]
            cells.append(f"{int(r.n_firms.iloc[0]):,}" if len(r) else "")
        tex.append(f"\\quad employers in the panel & {cells[0]} & "
                   f"{cells[1]} \\\\")
    note = r"Panel~A: reliability is the share of the variance in the firm score that is signal rather than sampling noise, which rises with the number of incumbents behind the score; it is reported and not used to correct any estimate, since the measurement error is not classical. Panel~B: the adoption step of Equation~(2), clustered by employer, with the national exposure quartile carried into every arm unchanged. The size groups split at four incumbents, because the incumbent count is too skewed for thirds."
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    write("tableA_size_reliability.tex", tex)


def main() -> int:
    coverage()
    size_reliability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

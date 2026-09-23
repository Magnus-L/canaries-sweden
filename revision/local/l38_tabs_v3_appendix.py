#!/usr/bin/env python3
"""
l38_tabs_v3_appendix.py: the two online-appendix tables the occupation
route owes, built from the lane 28 and lane 29 exports.

WHAT THEY REPORT

tableA_occ_coverage.tex, for Online Appendix IV.4. What the exposure
score is built from: for the year each incumbent's code was taken from,
by age band, how many incumbents the cascade resolves, how many are
scored at four digits and how many at three, and how many no level can
score; the same for employers; and what the choice of scoring level
changes, as the share of employers that move quartile between the
reported arm and each alternative and the rank correlation of the three
firm scores. Script 82 part A built this export for this table.

tableA_size_reliability.tex, for Online Appendix III.3. Whether a score
averaged over few incumbents distorts the treatment group. Panel A is the
reliability of the firm score at the quantiles of the incumbent-count
distribution, by employers and by incumbent employment, with the share of
each sitting below a reliability of one half. Panel B is the adoption step
at the reported floor, at a floor twelve times larger, and in each firm-size
group, with the national quartile carried in unchanged in every arm.

THE SIZE GROUPS ARE NOT TERCILES, AND THE TABLE SAYS SO. Part D cut the
scored employers at the thirds of the incumbent count. The count is so
skewed that the lower cut point falls at one incumbent, so the first group
is empty and what the export holds is a two-way split, 1 to 3 incumbents
against 4 or more. The table prints the ranges rather than the word
tercile, and the note says the cut degenerated.

INPUTS AND OUTPUTS
Reads occ_route_appendix_coverage.csv (lane 28a) and
occ_rest_reliability.csv with occ_rest_size.csv (lane 29d). Writes both
tables to revision/tables/ and copies them to canaries-sweden-paper/tables/.

    python3 revision/local/l38_tabs_v3_appendix.py

IN THE PAPER
Online Appendix III.3 and IV.4 of the v3 manuscript.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE28A = OUT / "round3_20260922-2232-lane28a"
OCC = OUT / "round3_20260923-0655-lanes28b-29bcd"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
TERM = "post_x_high_x_young"


def write(name: str, lines: list) -> None:
    V2_TAB.mkdir(parents=True, exist_ok=True)
    p = V2_TAB / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {p.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(p, PAPER_TAB / name)
        print(f"  copied to {PAPER_TAB / name}")


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
    c = pd.read_csv(LANE28A / "occ_route_coverage.csv")
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
    arm = pd.read_csv(LANE28A / "occ_route_appendix_coverage.csv")
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
    note = (
        r"Incumbents are the employees aged 31 to 69 on the employer's 2019 "
        r"payroll, the workers the score averages over; no younger worker is "
        r"scored and no code dated after the 2019 register file enters. A "
        r"code is taken from the 2019 register where it holds one and from "
        r"2018, 2017, 2016 or 2015 otherwise, which is the carry-back "
        r"column; 2.5 per cent of the resolved codes come from one of those "
        r"earlier years. Scored is the share the DAIOE index can price after "
        r"the merge. The reported arm scores every resolved incumbent from "
        r"the three-digit book: in this delivery every 2019 code carries "
        r"four digits, so the coarsening is a choice and not a gap, and the "
        r"last block reports what the alternatives change. A forward arm "
        r"that also reads 2020 and 2021 resolves 96.3 per cent of "
        r"incumbents; it is reported in the replication export and is never "
        r"the score."
    )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    write("tableA_occ_coverage.tex", tex)


def size_reliability() -> None:
    rel = pd.read_csv(OCC / "occ_rest_reliability.csv")
    size = pd.read_csv(OCC / "occ_rest_size.csv")
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
    note = (
        r"Panel A. The firm score is the employment-weighted mean of its "
        r"incumbents' occupation percentiles, so its sampling variance falls "
        r"with the number of incumbents behind it. Reliability is "
        r"$\sigma^2_b/(\sigma^2_b+\sigma^2_w/n)$ with the between-firm "
        r"component net of the sampling noise it contains: "
        rf"{float(v['variance_between_firms_net']):.0f} against a within-firm "
        rf"{float(v['variance_within_firm']):.0f}. Reliability is reported and "
        r"is not used to correct any estimate, since the measurement error "
        r"here is not classical and the standard attenuation correction does "
        r"not apply. Panel B. Poisson on the specification of Equation~(2), "
        r"clustered by employer. The national exposure quartile is carried "
        r"into every arm unchanged and is never recut inside a subsample. The "
        r"size groups are a two-way split at four incumbents: the intended "
        r"cut at the thirds of the incumbent count degenerates, because the "
        r"count is skewed enough that the lower cut point falls at one "
        r"incumbent and leaves the first group empty."
    )
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

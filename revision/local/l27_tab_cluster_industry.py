#!/usr/bin/env python3
"""
l27_tab_cluster_industry.py: Online Appendix Table (tab:cluster_industry),
every term of the headline design under two clusterings, on the
OCCUPATION route.

WHY THIS SCRIPT CHANGED. Until 23 September 2026 it drew the table from
four education-route lanes: lane 27's cluster_industry_v2.csv for the
coefficients and the industry standard errors, lane 21/22's window
specification for the employer-clustered standard error of the step, and
lane 25's sex split for Panel C's employer column. The v3 paper scores an
employer from the 2019 three-digit occupations of its own incumbents aged
31 to 69, and script 80 part B refitted the whole exercise on that score,
exporting both clusterings of every fit and the covariance of each. The
table is therefore built from one lane and one fit per panel, and nothing
is borrowed from a run the table does not report. The education-route
version of this builder is in the history of this repository.

THE QUESTION
Exposure is a firm-level score built from the occupations of a workforce,
and those are correlated within industry, so employers in the same sector
are not independent draws. Clustering on the employer treats them as if
they were. The table asks how much of the paper's inference survives when
the standard errors instead allow a disturbance common to every employer
in a three-digit industry.

WHAT IS ESTIMATED
The headline specification of Equation (2) is estimated once and its
variance computed twice, on employer clusters and on the employer's
three-digit NACE group, the code completed from a cascade across register
years rather than read from 2019 alone. The employers the cascade cannot
resolve share one residual group instead of each holding a cluster of its
own, which is why there are some 260 groups and not several thousand. The
point estimates are the same object under both clusterings, so the
exercise bears on inference alone: the coefficient is printed once, in its
own column, and the two standard errors beside it.

  Panel A  Ages 22 to 25 against the older bands pooled, 260 industry
           clusters.
  Panel B  Ages 26 to 30 on the same design, 263 clusters.
  Panel C  The sex split at ages 22 to 25, every treatment term
           interacted with female and the effects sex-specific, on the
           260 clusters of the younger panel. It is the panel the paper
           reads the hardest, because the female differential is a
           contrast drawn inside the employer and inside one age band, so
           a disturbance common to an industry has less of it to move.

The step from the 2023 level is the adoption term minus the interim term,
gamma_2 minus gamma_0 of Equation (2), and the step for young women is the
male step plus the female differential. Each derived row takes both of its
standard errors from the covariance of the two terms in the clustering
that column reports, Var(a-b) = Vaa + Vbb - 2Vab and Var(a+b) = Vaa + Vbb
+ 2Vab.

INPUTS AND OUTPUTS
Reads, from the lane pinned below (or one directory given on the command
line): occ_rest_cluster.csv, whose spec "pooled" holds Panels A and B and
whose spec "gender" holds Panel C, with se_employer and
se_industry_complete; and the four covariance exports of the same fits,
vcov_s80_clemp_<band>.csv and vcov_s80_clind2_<band>.csv for the pooled
panels and vcov_s80_gender_clemp_22_25.csv and
vcov_s80_gender_clind2_22_25.csv for the sexes. Nothing is typed in.
Writes revision/tables/tableA_cluster_industry.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l27_tab_cluster_industry.py [export_dir]

THE GATE
Part B is an inference exercise and nothing else: if a coefficient moved,
the panel is not the one the paper reports and no standard error from this
run may be quoted. Every coef_match_4dp in the export must be True and
each coefficient must equal its own employer-clustered run to four
decimals. Each standard error must be the square root of its own diagonal
in the covariance the same column comes from, so the derived rows are
built from the matrix that produced the rows above them. The
industry-cluster count of each panel must be the one recorded here when
the script was written. Every printed number is read back from the string
that goes into the table and compared with the export it came from. Any
disagreement stops the script and nothing is written.

IN THE PAPER
Online Appendix III.2, the paragraph on the same estimates under industry
clustering, Table tab:cluster_industry. The industry column of Table 1 of
the paper comes from the same export.
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
# Lanes 28b and 29b-d, the occupation route: script 80 part B refitted
# both clusterings of every panel on that score.
LANE = OUT / "round3_20260923-0655-lanes28b-29bcd"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

POST = "post_x_high_x_young"
INTER = "interim_x_high_x_young"
RB = "rb_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"

# Row label, term. The step is built from two of these and inserted where
# the None stands.
ROWS = [
    (r"Tightening months ($\hat\gamma_1$)", RB),
    (r"Interim, through 2023 ($\hat\gamma_0$)", INTER),
    (r"Adoption, from 2024 ($\hat\gamma_2$)", POST),
    ("STEP", None),
    ("Quarter 1", "q1_x_high_x_young"),
    ("Quarter 2", "q2_x_high_x_young"),
    ("Quarter 3", "q3_x_high_x_young"),
]
# Panel C: the sex split at 22-25. The sum is built last.
SEX_ROWS = [
    (r"Young men, tightening months ($\hat\gamma_1$)", RB),
    (r"Young men, interim through 2023 ($\hat\gamma_0$)", INTER),
    (r"Young men, adoption from 2024 ($\hat\gamma_2$)", POST),
    (r"Female differential, tightening months",
     "rb_x_high_x_young_x_female"),
    (r"Female differential, interim through 2023",
     "interim_x_high_x_young_x_female"),
    (r"Female differential, adoption from 2024", FEMALE),
]
# The industry-cluster count of each panel, as the export held it when
# this script was written. A different count means a different panel.
CLUSTERS = {("pooled", "22-25"): 260, ("pooled", "26-30"): 263,
            ("gender", "22-25"): 260}
COEF = re.compile(r"^\$([-+][0-9.]+)\$$")
SE = re.compile(r"^\(([0-9.]+)\)(\$\^\{\*\}\$)?$")


def source(name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def vcov(name: str) -> pd.DataFrame:
    """One exported variance-covariance matrix, terms on both axes."""
    return pd.read_csv(source(name), index_col=0)


def thousands(n: int) -> str:
    """A cluster count as the paper prints it."""
    return f"{n:,}".replace(",", "{,}")


def se_diff(v: pd.DataFrame, a: str, b: str) -> float:
    """Standard error of a minus b from the exported covariance."""
    return float(v.loc[a, a] + v.loc[b, b] - 2 * v.loc[a, b]) ** 0.5


def se_sum(v: pd.DataFrame, a: str, b: str) -> float:
    """Standard error of a plus b from the exported covariance."""
    return float(v.loc[a, a] + v.loc[b, b] + 2 * v.loc[a, b]) ** 0.5


def coef_cell(what: str, c: float) -> str:
    """The coefficient, checked against the export before it is printed."""
    out = f"${c:+.4f}$"
    m = COEF.match(out)
    if m is None or abs(float(m.group(1)) - c) > 5e-5:
        raise SystemExit(f"  {what}: the printed coefficient {out} disagrees "
                         f"with the export ({c:+.6f})")
    return out


def se_cell(what: str, c: float, se: float) -> str:
    """One standard error, with a star where the estimate is significant
    at five per cent under that clustering, checked before it is
    printed."""
    star = "$^{*}$" if abs(c) > 1.96 * se else ""
    out = f"({se:.4f}){star}"
    m = SE.match(out)
    if m is None or abs(float(m.group(1)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed standard error {out} "
                         f"disagrees with the export ({se:.6f})")
    return out


def one_count(values, what: str) -> int:
    n = sorted(set(int(x) for x in values))
    if len(n) != 1:
        raise SystemExit(f"  {what}: one count expected, found {n}")
    return n[0]


def panel_frame(spec: str, band: str) -> tuple[pd.DataFrame, int, int]:
    """The rows of one fit, with the gate that makes them quotable."""
    d = pd.read_csv(source("occ_rest_cluster.csv"))
    d = d[(d.get("status", "ok") == "ok") & (d.spec == spec)
          & (d.young_band == band)]
    if d.empty:
        raise SystemExit(f"  occ_rest_cluster.csv: no {spec} rows for {band}")
    if not bool(d.coef_match_4dp.all()):
        bad = list(d.loc[~d.coef_match_4dp.astype(bool), "term"])
        raise SystemExit(f"  {spec} {band}: {bad} do not reproduce their own "
                         f"employer-clustered run to four decimals; no "
                         f"standard error from this fit is quotable")
    off = (d.coef - d.coef_employer_run).abs().max()
    if off > 5e-5:
        raise SystemExit(f"  {spec} {band}: the coefficient column departs "
                         f"from the employer-clustered run by {off:.2e}")
    n_clusters = one_count(d.n_clusters_complete, f"{spec} {band} clusters")
    if n_clusters != CLUSTERS[(spec, band)]:
        raise SystemExit(f"  {spec} {band}: the export holds {n_clusters} "
                         f"industry clusters and this script was written on "
                         f"{CLUSTERS[(spec, band)]}; the panel head would be "
                         f"wrong, so nothing is written")
    n_firms = one_count(d.n_firms, f"{spec} {band} employers")
    return d.set_index("term"), n_clusters, n_firms


def check_diagonals(d: pd.DataFrame, terms: list, ve: pd.DataFrame,
                    vi: pd.DataFrame, what: str) -> None:
    """Each exported standard error must be the square root of its own
    diagonal, so the derived rows are built from the same matrices."""
    for term in terms:
        if term not in d.index:
            raise SystemExit(f"  {what}: no {term} in occ_rest_cluster.csv")
        for v, col, which in ((ve, "se_employer", "employer"),
                              (vi, "se_industry_complete", "industry")):
            if term not in v.index or term not in v.columns:
                raise SystemExit(f"  {what} {term}: not in the {which} "
                                 f"covariance export")
            diag = float(v.loc[term, term]) ** 0.5
            got = float(d.loc[term, col])
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {what} {term}: the exported {which} "
                                 f"standard error {got:.6f} is not the square "
                                 f"root of its own variance {diag:.6f}")


def pooled_panel(band: str) -> tuple[str, int, int, list]:
    """Panel A or B: one young band against the older bands pooled."""
    d, n_clusters, n_firms = panel_frame("pooled", band)
    us = band.replace("-", "_")
    ve = vcov(f"vcov_s80_clemp_{us}.csv")
    vi = vcov(f"vcov_s80_clind2_{us}.csv")
    terms = [t for _, t in ROWS if t is not None]
    check_diagonals(d, terms, ve, vi, band)

    step = float(d.loc[POST, "coef"]) - float(d.loc[INTER, "coef"])
    rows = []
    for label, term in ROWS:
        if term is None:
            rows.append(("Step from the 2023 level",
                         coef_cell(f"{band} step", step),
                         se_cell(f"{band} step, employer", step,
                                 se_diff(ve, POST, INTER)),
                         se_cell(f"{band} step, industry", step,
                                 se_diff(vi, POST, INTER))))
            continue
        r = d.loc[term]
        c, emp, ind = (float(r.coef), float(r.se_employer),
                       float(r.se_industry_complete))
        rows.append((label, coef_cell(f"{band} {term}", c),
                     se_cell(f"{band} {term}, employer", c, emp),
                     se_cell(f"{band} {term}, industry", c, ind)))
    return (band, n_clusters, n_firms, rows)


def sex_panel() -> tuple[str, int, int, list]:
    """Panel C: the sex split of Equation (2) at ages 22 to 25, with the
    step for young women built from the male step and the female
    differential."""
    band = "22-25"
    d, n_clusters, n_firms = panel_frame("gender", band)
    ve = vcov("vcov_s80_gender_clemp_22_25.csv")
    vi = vcov("vcov_s80_gender_clind2_22_25.csv")
    terms = [t for _, t in SEX_ROWS]
    check_diagonals(d, terms, ve, vi, "the sexes")

    rows = []
    for label, term in SEX_ROWS:
        r = d.loc[term]
        c, emp, ind = (float(r.coef), float(r.se_employer),
                       float(r.se_industry_complete))
        rows.append((label, coef_cell(f"sexes {term}", c),
                     se_cell(f"sexes {term}, employer", c, emp),
                     se_cell(f"sexes {term}, industry", c, ind)))

    women = float(d.loc[POST, "coef"]) + float(d.loc[FEMALE, "coef"])
    rows.append(("Young women, adoption from 2024",
                 coef_cell("young women", women),
                 se_cell("young women, employer", women,
                         se_sum(ve, POST, FEMALE)),
                 se_cell("young women, industry", women,
                         se_sum(vi, POST, FEMALE))))
    return ("sexes", n_clusters, n_firms, rows)


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE
    print(f"  lane {d.name}")
    panels = [pooled_panel("22-25"), pooled_panel("26-30"), sex_panel()]
    print("  every coefficient reproduces its own employer-clustered run to "
          "four decimals, and every standard error its own covariance")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Every term of the headline design under two "
           r"clusterings, employer and three-digit industry.}",
           r"\label{tab:cluster_industry}", r"\footnotesize",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Term & Coefficient & Employer SE & Industry SE \\",
           r"\midrule"]
    for i, (band, n_clusters, n_firms, rows) in enumerate(panels):
        if i:
            tex.append(r"\addlinespace[4pt]")
        if band == "sexes":
            head = (f"Panel {chr(65 + i)}. The sexes at ages 22--25, "
                    f"{thousands(n_clusters)} industry clusters")
        else:
            head = (f"Panel {chr(65 + i)}. Ages {band.replace('-', '--')}, "
                    f"{thousands(n_clusters)} industry clusters")
        tex.append(r"\multicolumn{4}{l}{\textit{" + head + r"}} \\")
        print(f"  {head}, {n_firms:,} employers")
        for label, c, emp, ind in rows:
            tex.append(f"{label} & {c} & {emp} & {ind} \\\\")
            print(f"    {label[:44]:44s} {c:>12s}  {emp:>14s}  {ind:>14s}")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.9\textwidth}\footnotesize\vspace{4pt}",
            r"The headline specification of Equation~(2), exposure the "
            r"employer's 2019 occupation mix, re-estimated with standard "
            r"errors clustered on the employer's three-digit NACE group "
            r"instead of on the employer, the code completed from a cascade "
            r"across register years and the employers it cannot resolve "
            r"sharing one residual group. Coefficients are identical to four "
            r"decimals under the two clusterings, so the exercise changes "
            r"inference and nothing else and the coefficient is printed once. "
            r"The step from the 2023 level is the adoption term minus the "
            r"interim term, and its employer-clustered standard error is the "
            r"one Table~1 of the paper reports; each standard error of a "
            r"derived row comes from the covariance of the two terms under "
            r"the clustering in whose column it stands. Panel~C is the sex "
            r"split of Equation~(2) at ages 22--25, every treatment term "
            r"interacted with female and the effects sex-specific; the row "
            r"for young women is the male step plus the female differential. "
            r"Stars mark $p<0.05$ under the clustering in whose column they "
            r"stand. Source: script 80, part B.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_cluster_industry.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

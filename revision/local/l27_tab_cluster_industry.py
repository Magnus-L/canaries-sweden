#!/usr/bin/env python3
"""
l27_tab_cluster_industry.py: Online Appendix Table III.2
(tab:cluster_industry), every term of the headline design under two
clusterings.

THE QUESTION
Exposure is a firm-level score built from the education mix of a
workforce, and that mix is correlated within industry, so employers in
the same sector are not independent draws. Clustering on the employer
treats them as if they were. The table asks how much of the paper's
inference survives when the standard errors instead allow a disturbance
common to every employer in a three-digit industry.

WHAT IS ESTIMATED
The headline specification of Equation (2) is estimated once and its
variance computed twice, on employer clusters and on the employer's
three-digit NACE group, the code completed from a cascade across register
years rather than read from 2019 alone (script 80, lane 27 part B). The
employers the cascade cannot resolve share one residual group instead of
each holding a cluster of its own, which is what the superseded lane 25
and lane 26 runs did and why they reported several thousand clusters
where there are some 260 industry groups. The point
estimates are therefore the same object under both clusterings, and the
exercise bears on inference alone; the coefficient is printed once, in
its own column, and the two standard errors beside it. Panels A and B are
one young band each against the older bands pooled, and the
industry-cluster count is the number of groups that band's panel
contains.

Panel C does the same for the sex split of Equation (2) at ages 22 to 25,
every treatment term interacted with female and the effects
sex-specific (script 80, lane 27 part B). It is the panel the paper reads
the hardest, because the female differential is a contrast drawn inside
the employer and inside one age band, so a disturbance common to an
industry has less of it to move. The step for young women is the male
step plus the differential, and each of its standard errors comes from
the covariance of those two terms in the run that column reports.

The step from the 2023 level is the adoption term minus the interim term,
which is gamma_2 minus gamma_0 of Equation (2). Its industry-clustered
standard error is computed from the exported covariance of the two terms
in this run. Its employer-clustered standard error is the one Table 1
reports, from the window specification of script 75, which was not
re-clustered; the script checks that the two runs give the same step
before it borrows that standard error.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names (or one
directory given on the command line): cluster_industry_v2.csv with
vcov_s80_clind2_<band>.csv and vcov_s80_gender_clind2_22_25.csv (script
80, lane 27 part B) for the coefficients, both standard errors and the
cluster counts, the pooled rows in its spec "pooled" and Panel C's in its
spec "gender", and se_industry_complete for every industry standard
error, se_industry_hybrid being the superseded column;
reference_window.csv with vcov_s75_<band>_stock.csv (script 75, lanes 21
and 22) for the employer-clustered standard error of the step, the one
Table 1 prints; and gender_eq2.csv with vcov_s78_gender_eq2_22_25.csv
(script 78, lane 25 part B) for Panel C's employer-clustered standard
errors. Nothing is typed in. Writes
revision/tables/tableA_cluster_industry.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l27_tab_cluster_industry.py [export_dir]

THE GATE
The table is written only if both fits reproduce the employer-clustered
estimates: every coef_match_4dp in the export must be True, and each
coefficient column must equal the employer-clustered run to four
decimals. Panel C carries the further check that its coefficients are the
ones the sex split of lane 25 part B reported, since those are the
coefficients Table 1 prints. The step from the 2023 level must come out the same in the
clustering run and in the window specification, since the employer
standard error printed for it is the window specification's. Each
industry standard error must equal the square root of its own diagonal in
the exported covariance, and every printed number is read back from the
string that goes into the table and compared with the export it came
from. Any disagreement beyond half of the last printed digit stops the
script and nothing is written.

IN THE PAPER
Online Appendix III.2, the paragraph on the same estimates under industry
clustering, Table tab:cluster_industry. The industry column of Table 1
comes from the same two exports, through l18_table1.py.
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
LANE21_22 = OUT / "round3_20260922-0712-lanes21-22"
LANE25 = OUT / "round3_20260922-1237-lane25bc-BCEF"
LANE26 = OUT / "round3_20260922-lane26ab-AB"
LANE27 = OUT / "round3_20260922-1820-lane27bc"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30"]
POST = "post_x_high_x_young"
INTER = "interim_x_high_x_young"
RB = "rb_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"
# The window specification names the tightening term differently; every
# other term carries the same name in the two runs.
WINDOW_POST, WINDOW_INTER = POST, INTER

# Row label, term in cluster_industry.csv. The step is built from two of
# these terms and is inserted after them.
ROWS = [
    (r"Tightening months ($\hat\gamma_1$)", "rb_x_high_x_young"),
    (r"Interim, through 2023 ($\hat\gamma_0$)", INTER),
    (r"Adoption, from 2024 ($\hat\gamma_2$)", POST),
    ("STEP", None),
    ("Quarter 1", "q1_x_high_x_young"),
    ("Quarter 2", "q2_x_high_x_young"),
    ("Quarter 3", "q3_x_high_x_young"),
]
# Panel C: the sex split at 22-25. Label, term; the sum is built last.
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
COEF = re.compile(r"^\$([-+][0-9.]+)\$$")
SE = re.compile(r"^\(([0-9.]+)\)(\$\^\{\*\}\$)?$")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def vcov(default_dir: Path, name: str) -> pd.DataFrame:
    """One exported variance-covariance matrix, terms on both axes."""
    return pd.read_csv(source(default_dir, name), index_col=0)


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


def sex_panel() -> tuple[str, int, list]:
    """Panel C: the sex split of Equation (2) at ages 22 to 25 under the
    two clusterings, with the step for young women built from the male
    step and the female differential."""
    gsex = pd.read_csv(source(LANE27, "cluster_industry_v2.csv"))
    gsex = gsex[gsex.get("status", "ok") == "ok"]
    gsex = gsex[gsex.spec == "gender"]
    if not bool(gsex.coef_match_4dp.all()):
        raise SystemExit("  lane 27B: a coefficient does not reproduce to "
                         "four decimals; no industry SE is quotable for the "
                         "sex rows")
    off = (gsex.coef - gsex.coef_employer_run).abs().max()
    if off > 5e-5:
        raise SystemExit(f"  lane 27B: the coefficient column departs from "
                         f"the employer-clustered run by {off:.2e}")
    d = gsex[gsex.young_band == "22-25"]
    if d.empty:
        raise SystemExit("  cluster_industry_v2.csv: no sex rows for 22-25")
    d = d.set_index("term")

    # The coefficients must be the ones the sex split of lane 25 reported,
    # because those are the coefficients Table 1 prints beside these
    # standard errors.
    eq2 = pd.read_csv(source(LANE25, "gender_eq2.csv"))
    eq2 = eq2[eq2.get("status", "ok") == "ok"]
    eq2 = eq2[eq2.young_band == "22-25"].set_index("term")
    for term in d.index:
        if term not in eq2.index:
            raise SystemExit(f"  {term}: not in the lane 25 sex split")
        if abs(float(d.loc[term, "coef"]) - float(eq2.loc[term, "coef"])) > 5e-5:
            raise SystemExit(f"  {term}: the industry-clustered sex run does "
                             f"not reproduce the lane 25 coefficient")

    n_clusters = sorted(set(int(x) for x in d.n_clusters_complete))
    if len(n_clusters) != 1:
        raise SystemExit(f"  the sexes: one industry-cluster count expected, "
                         f"found {n_clusters}")
    vi = vcov(LANE27, "vcov_s80_gender_clind2_22_25.csv")
    ve = vcov(LANE25, "vcov_s78_gender_eq2_22_25.csv")

    rows = []
    for label, term in SEX_ROWS:
        if term not in d.index:
            raise SystemExit(f"  the sexes: no {term} in the export")
        r = d.loc[term]
        c, emp, ind = (float(r.coef), float(r.se_employer),
                       float(r.se_industry_complete))
        for v, got, which in ((vi, ind, "industry"), (ve, emp, "employer")):
            diag = float(v.loc[term, term]) ** 0.5
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {term}: the exported {which} SE "
                                 f"{got:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")
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
    return ("sexes", n_clusters[0], rows)


def main() -> int:
    clind = pd.read_csv(source(LANE27, "cluster_industry_v2.csv"))
    clind = clind[clind.get("status", "ok") == "ok"]
    clind = clind[clind.spec == "pooled"]
    window = pd.read_csv(source(LANE21_22, "reference_window.csv"))
    window = window[window.get("status", "ok") == "ok"]

    # Part B is an inference exercise and nothing else: if a coefficient
    # moved, the panel is not the one the paper reports and no standard
    # error from this run may be quoted.
    if not bool(clind.coef_match_4dp.all()):
        raise SystemExit("  lane 27B: a coefficient does not reproduce to "
                         "four decimals; no industry SE is quotable")
    off = (clind.coef - clind.coef_employer_run).abs().max()
    if off > 5e-5:
        raise SystemExit(f"  lane 27B: the coefficient column departs from "
                         f"the employer-clustered run by {off:.2e}")

    panels = []
    for band in BANDS:
        d = clind[clind.young_band == band]
        if d.empty:
            raise SystemExit(f"  cluster_industry_v2.csv: no pooled rows for "
                             f"{band}")
        d = d.set_index("term")
        n_clusters = sorted(set(int(x) for x in d.n_clusters_complete))
        if len(n_clusters) != 1:
            raise SystemExit(f"  {band}: one industry-cluster count "
                             f"expected, found {n_clusters}")
        v = vcov(LANE27, f"vcov_s80_clind2_{band.replace('-', '_')}.csv")

        # The window specification's step, whose employer-clustered
        # standard error Table 1 prints and this table borrows.
        w = window[(window.young_band == band) & (window.outcome == "stock")]
        w = w.set_index("term")
        for t in (WINDOW_POST, WINDOW_INTER):
            if t not in w.index:
                raise SystemExit(f"  {band}: no {t} in reference_window.csv")
        step_window = float(w.loc[WINDOW_POST, "coef"]) - float(w.loc[WINDOW_INTER, "coef"])
        step = float(d.loc[POST, "coef"]) - float(d.loc[INTER, "coef"])
        if abs(step - step_window) > 5e-5:
            raise SystemExit(f"  {band}: the step from the 2023 level is "
                             f"{step:+.4f} in the clustering run and "
                             f"{step_window:+.4f} in the window specification, "
                             f"so the employer SE of Table 1 does not belong "
                             f"to it")
        vw = vcov(LANE21_22, f"vcov_s75_{band.replace('-', '_')}_stock.csv")
        step_se_emp = se_diff(vw, WINDOW_POST, WINDOW_INTER)
        step_se_ind = se_diff(v, POST, INTER)

        rows = []
        for label, term in ROWS:
            if term is None:
                rows.append(("Step from the 2023 level",
                             coef_cell(f"{band} step", step),
                             se_cell(f"{band} step, employer", step,
                                     step_se_emp),
                             se_cell(f"{band} step, industry", step,
                                     step_se_ind)))
                continue
            if term not in d.index:
                raise SystemExit(f"  {band}: no {term} in "
                                 f"cluster_industry_v2.csv")
            r = d.loc[term]
            c, emp, ind = (float(r.coef), float(r.se_employer),
                           float(r.se_industry_complete))
            # The covariance the step is built from must be the one the
            # standard error column reports for the same terms.
            diag = float(v.loc[term, term]) ** 0.5
            if abs(diag - ind) > 5e-5:
                raise SystemExit(f"  {band} {term}: the exported industry SE "
                                 f"{ind:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")
            rows.append((label, coef_cell(f"{band} {term}", c),
                         se_cell(f"{band} {term}, employer", c, emp),
                         se_cell(f"{band} {term}, industry", c, ind)))
        panels.append((band, n_clusters[0], rows))

    panels.append(sex_panel())

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Every term of the headline design under two "
           r"clusterings, employer and three-digit industry.}",
           r"\label{tab:cluster_industry}", r"\footnotesize",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"Term & Coefficient & Employer SE & Industry SE \\",
           r"\midrule"]
    for i, (band, n_clusters, rows) in enumerate(panels):
        if i:
            tex.append(r"\addlinespace[4pt]")
        if band == "sexes":
            head = (f"Panel {chr(65 + i)}. The sexes at ages 22--25, "
                    f"{thousands(n_clusters)} industry clusters")
        else:
            head = (f"Panel {chr(65 + i)}. Ages {band.replace('-', '--')}, "
                    f"{thousands(n_clusters)} industry clusters")
        tex.append(r"\multicolumn{4}{l}{\textit{" + head + r"}} \\")
        print(f"  {head}")
        for label, c, emp, ind in rows:
            tex.append(f"{label} & {c} & {emp} & {ind} \\\\")
            print(f"    {label[:44]:44s} {c:>12s}  {emp:>14s}  {ind:>14s}")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.9\textwidth}\footnotesize\vspace{4pt}",
            r"The headline specification of Equation~(2), re-estimated with "
            r"standard errors clustered on the employer's three-digit NACE "
            r"group instead of on the employer, the code completed from a "
            r"cascade across register years and the employers it cannot "
            r"resolve sharing one residual group. Coefficients are "
            r"identical to four decimals under the two clusterings, so the "
            r"exercise changes inference and nothing else and the coefficient "
            r"is printed once. The step from the 2023 level is the adoption "
            r"term minus the interim term; its employer-clustered standard "
            r"error is the one Table~1 reports, from the window "
            r"specification, and its industry-clustered standard error comes "
            r"from the covariance of the two terms in this run. Panel~C is "
            r"the sex split of Equation~(2) at ages 22--25, every treatment "
            r"term interacted with female and the effects sex-specific; the "
            r"row for young women is the male step plus the female "
            r"differential, each standard error from the covariance of those "
            r"two terms in its own run. Stars mark "
            r"$p<0.05$ under the clustering in whose column they stand. "
            r"Source: script 80, part B.",
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

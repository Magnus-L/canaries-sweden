#!/usr/bin/env python3
"""
l18_table1.py: Table 1 of the paper, assembled from the exported estimates.

WHAT THE TABLE REPORTS
The estimates the paper rests on, all from the within-employer age design
with the calendar cycle removed: employment of workers aged 22 to 25
relative to their older colleagues in the same employer, with exposure
frozen at the employer's 2019 education mix.

  Ages 22-25         the rise during the tightening months (gamma_1); the
                     additional step once firms adopt AI (gamma_2), with
                     the vintage re-scoring beside it; the step from the
                     2023 level (gamma_2 minus gamma_0, its standard error
                     from the covariance of the post and interim terms of
                     the window specification); the level after adoption
                     against the months before the rate rise.
  Ages 26-30         the additional step at adoption and the step from the
                     2023 level.
  Profile            22-25 and 50 and over, each against 41-49, from one
                     panel of all six bands.
  Margin, incidence  hires, separations, the young men's step, the female
                     differential, the young women's step, and the part of
                     the differential within broad education tracks.

TWO CLUSTERINGS
Exposure is a firm-level score built from a workforce composition that is
correlated within industry, so every young-band row that was re-estimated
with three-digit industry clusters (lane 25, part C) carries that standard
error in square brackets beside the employer-clustered one. Part C changes
no coefficient: the script refuses to write the table unless every
coefficient reproduces the employer-clustered run to four decimals. The
level row comes from the window specification, which was not re-clustered,
so it carries one standard error only.

THE GENDER ROWS
The three sex rows come from the sex split on Equation (2) (lane 25, part
B): every treatment term interacted with female, sex-specific
employer-by-age and month-by-age effects, so they sit on the same base as
the rest of the table. Young women is the sum of the male step and the
differential, its standard error from the exported covariance. The
within-track row is script 76's and sits on script 68's base, which has no
interim term; the note says so and the reconciliation is in the appendix.

The vintage re-scoring is the change in gamma_2 when each employer's 2019
incumbents are re-scored from the education register as it stood in 2021
and the same panel is re-estimated (the as-of arm of script 68). It is
not the backtest of Online Appendix IV.3, which the reported design admits
no occupation code after 2019 and is therefore not exposed to. The 26-30
arm was not re-scored, so that cell is empty.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names (or one
directory given on the command line): seasonal_pooled.csv (script 68);
reference_window.csv and vcov_s75_<band>_stock.csv (script 75);
contrast_seasonal.csv (script 74); gender_split.csv (script 76);
gender_eq2.csv, vcov_s78_gender_eq2_22_25.csv, cluster_industry.csv and
vcov_s78_clind_<band>.csv (script 78, lane 25). Nothing is typed in; a
missing or ambiguous row stops the script. Writes
revision/tables/table1_headline.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l18_table1.py [export_dir]

IN THE PAPER
Table 1, Section 3.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE14 = OUT / "round3_20260921-2152-lane14-seasonal-complete"
LANE14_GENDER = OUT / "round3_20260921-lane14-seasonal"
LANE20 = OUT / "round3_20260922-0105-lane20-seasonal-contrast"
LANE21_22 = OUT / "round3_20260922-0712-lanes21-22"
LANE25 = OUT / "round3_20260922-1237-lane25bc-BCEF"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

TERM = "post_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def one(df: pd.DataFrame, **cond) -> tuple[float, float]:
    """The coefficient and standard error of exactly one row."""
    r = df
    for k, v in cond.items():
        r = r[r[k] == v]
    if len(r) != 1:
        raise SystemExit(f"  expected one row for {cond}, found {len(r)}")
    return float(r.coef.iloc[0]), float(r.se.iloc[0])


def step_from_2023(window: pd.DataFrame, band: str) -> tuple[float, float]:
    """The change from the 2023 level to the level after adoption, on the
    window specification of script 75 (post minus interim, both measured
    against the months before the rate hike, so the difference equals
    gamma_2 minus gamma_0 of Equation (2)), with the standard error from
    the exported covariance of the two terms."""
    post = one(window, young_band=band, outcome="stock", term=TERM)
    inter = one(window, young_band=band, outcome="stock",
                term="interim_x_high_x_young")
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE21_22
    v = pd.read_csv(d / f"vcov_s75_{band.replace('-', '_')}_stock.csv",
                    index_col=0)
    var = (v.loc[TERM, TERM] + v.loc["interim_x_high_x_young",
                                     "interim_x_high_x_young"]
           - 2 * v.loc[TERM, "interim_x_high_x_young"])
    return post[0] - inter[0], float(var) ** 0.5


def vcov(default_dir: Path, name: str) -> pd.DataFrame:
    """One exported variance-covariance matrix, terms on both axes."""
    return pd.read_csv(source(default_dir, name), index_col=0)


def se_lin(v: pd.DataFrame, a: str, b: str, sign: int) -> float:
    """Standard error of a + b (sign +1) or a - b (sign -1)."""
    return float(v.loc[a, a] + v.loc[b, b]
                 + 2 * sign * v.loc[a, b]) ** 0.5


def est(c: float, se: float) -> str:
    """The estimate with its employer-clustered standard error."""
    return f"${c:+.4f}$ ({se:.4f})"


def ind(se: float | None) -> str:
    """The third column: the same coefficient's standard error on
    three-digit industry clusters, where that run exists."""
    return "" if se is None else f"({se:.4f})"


def main() -> int:
    pooled = pd.read_csv(source(LANE14, "seasonal_pooled.csv"))
    pooled = pooled[pooled.get("status", "ok") == "ok"]
    window = pd.read_csv(source(LANE21_22, "reference_window.csv"))
    window = window[window.get("status", "ok") == "ok"]
    profile = pd.read_csv(source(LANE20, "contrast_seasonal.csv"))
    split = pd.read_csv(source(LANE21_22, "gender_split.csv"))
    sexes = pd.read_csv(source(LANE25, "gender_eq2.csv"))
    sexes = sexes[sexes.get("status", "ok") == "ok"]
    clind = pd.read_csv(source(LANE25, "cluster_industry.csv"))

    # Part C is an inference exercise and nothing else: if a coefficient
    # moved, the panel is not the one the rest of the table reports and no
    # industry-clustered standard error from it may be quoted.
    if not bool(clind.coef_match_4dp.all()):
        raise SystemExit("  lane 25C: a coefficient does not reproduce to "
                         "four decimals; no industry SE is quotable")

    def se_ind(band: str, term: str) -> float:
        r = clind[(clind.young_band == band) & (clind.term == term)]
        if len(r) != 1:
            raise SystemExit(f"  expected one industry row for {band} {term}")
        return float(r.se_industry.iloc[0])

    def check(band: str, term: str, c: float) -> None:
        r = clind[(clind.young_band == band) & (clind.term == term)]
        if abs(float(r.coef.iloc[0]) - c) > 5e-5:
            raise SystemExit(f"  {band} {term}: the industry-clustered run "
                             f"does not reproduce the reported coefficient")

    # 22-25, the sequence
    g1 = one(pooled, young_band="22-25", outcome="stock", arm="true",
             term="rb_x_high_x_young")
    g2 = one(pooled, young_band="22-25", outcome="stock", arm="true",
             term=TERM)
    g2_asof = one(pooled, young_band="22-25", outcome="stock", arm="asof",
                  term=TERM)
    artefact = g2_asof[0] - g2[0]
    level = one(window, young_band="22-25", outcome="stock", term=TERM)
    step23 = step_from_2023(window, "22-25")
    # 26-30, the step
    g2_26 = one(pooled, young_band="26-30", outcome="stock", arm="true",
                term=TERM)
    step23_26 = step_from_2023(window, "26-30")
    # the profile against 41-49
    p22 = one(profile, arm="seasonal", band_vs_ref="22_25")
    p50 = one(profile, arm="seasonal", band_vs_ref="50p")
    # margin and incidence
    hires = one(pooled, young_band="22-25", outcome="hires", arm="true",
                term=TERM)
    seps = one(pooled, young_band="22-25", outcome="seps", arm="true",
               term=TERM)
    # the sex split on Equation (2): the male step, the differential, and
    # their sum, whose standard error comes from the exported covariance
    men = one(sexes, young_band="22-25", term=TERM)
    fem = one(sexes, young_band="22-25", term=FEMALE)
    vg = vcov(LANE25, "vcov_s78_gender_eq2_22_25.csv")
    women = (men[0] + fem[0], se_lin(vg, TERM, FEMALE, +1))
    if len(split) != 1:
        raise SystemExit("  gender_split.csv should hold one row")
    within = (float(split.within.iloc[0]), float(split.within_se.iloc[0]))

    # the industry-clustered standard errors, and the coefficient check
    check("22-25", "rb_x_high_x_young", g1[0])
    check("22-25", TERM, g2[0])
    check("26-30", TERM, g2_26[0])
    v22 = vcov(LANE25, "vcov_s78_clind_22_25.csv")
    v26 = vcov(LANE25, "vcov_s78_clind_26_30.csv")
    INTER = "interim_x_high_x_young"
    step23_ind = se_lin(v22, TERM, INTER, -1)
    step23_26_ind = se_lin(v26, TERM, INTER, -1)
    for band, v, got in (("22-25", v22, step23[0]),
                         ("26-30", v26, step23_26[0])):
        r = clind[clind.young_band == band].set_index("term")
        d = float(r.loc[TERM, "coef"]) - float(r.loc[INTER, "coef"])
        if abs(d - got) > 5e-5:
            raise SystemExit(f"  {band}: the step from the 2023 level in the "
                             f"industry-clustered run is {d:+.4f}, not {got:+.4f}")

    rows = [
        (r"\multicolumn{3}{l}{\textit{Ages 22--25, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Tightening months, April to November 2022 ($\hat\gamma_1$)",
         est(*g1), ind(se_ind("22-25", "rb_x_high_x_young"))),
        (r"Additional step at adoption, from January 2024 ($\hat\gamma_2$)",
         est(*g2), ind(se_ind("22-25", TERM))),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23), ind(step23_ind)),
        (r"Level after adoption, against the pre-hike months",
         est(*level), ""),
        (r"\addlinespace[3pt]", None, None),
        (r"\multicolumn{3}{l}{\textit{Ages 26--30, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Additional step at adoption ($\hat\gamma_2$)",
         est(*g2_26), ind(se_ind("26-30", TERM))),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23_26), ind(step23_26_ind)),
        (r"\addlinespace[3pt]", None, None),
        (r"\multicolumn{3}{l}{\textit{The profile, against 41--49 alone}} \\",
         None, None),
        (r"22--25", est(*p22), ""),
        (r"50 and over", est(*p50), ""),
        (r"\addlinespace[3pt]", None, None),
        (r"\multicolumn{3}{l}{\textit{Ages 22--25, margin and incidence}} \\",
         None, None),
        (r"Hires, additional step at adoption", est(*hires), ""),
        (r"Separations, additional step at adoption", est(*seps), ""),
        (r"Young men, additional step at adoption", est(*men), ""),
        (r"Young women minus young men", est(*fem), ""),
        (r"Young women, additional step at adoption", est(*women), ""),
        (r"\quad within broad education tracks", est(*within), ""),
    ]

    note = (
        r"Poisson on employer $\times$ age $\times$ month counts, with the "
        r"effects and calendar terms of Equation~(2); exposure frozen at the "
        r"employer's 2019 education mix; clustered by employer. Steps read "
        r"from the tightening level, the Riksbank interaction staying in the "
        r"model; the level row is a window against January 2021 to "
        r"March 2022, and the step from 2023 is post minus interim. The third "
        r"column clusters the same specification on three-digit industry, "
        r"5{,}545 groups at 22--25 and 9{,}368 at 26--30, and reproduces "
        r"every coefficient. The sex rows "
        r"interact every treatment term with female, with sex-specific "
        r"employer-by-age and month-by-age effects; young women is the male "
        r"step plus the differential, standard error from their "
        r"covariance (within-track row: earlier base). "
        r"Re-scoring the 2019 incumbents from the 2021 register "
        rf"moves the step at 22--25 by ${artefact:+.4f}$. "
        r"Full tables in Online Appendix~III.2."
    )

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Employment of young workers relative to their older "
           r"colleagues inside the same employer, with the calendar cycle "
           r"removed.}",
           r"\label{tab:headline}", r"\footnotesize",
           # the third column has to fit inside the text block; the table
           # was already a whisker over before it was added
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lcc}", r"\toprule",
           r" & Estimate (SE) & Industry SE \\",
           r"\midrule"]
    for lab, e, art in rows:
        if e is None:
            tex.append(lab)
        else:
            tex.append(f"{lab} & {e} & {art} \\\\" if art
                       else f"{lab} & {e} & \\\\")
            print(f"  {lab[:58]:58s} {e}  {art}")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "table1_headline.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

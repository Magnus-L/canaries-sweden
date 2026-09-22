#!/usr/bin/env python3
"""
l18_table1.py -- Table 1 of the paper, assembled from the exported estimates.

WHAT THE TABLE REPORTS. The estimates the paper rests on, all from the
within-employer age design with the calendar cycle removed: employment of
workers aged 22 to 25 relative to their older colleagues in the same
employer, with exposure frozen at the employer's 2019 education mix.

  Sequence for 22-25    the rise during the tightening months (gamma_1),
                        the additional step once firms adopt AI (gamma_2)
                        with the vintage re-scoring beside it, the step
                        from the 2023 level (gamma_2 minus gamma_0, its
                        standard error from the covariance of the two
                        terms in the window specification of lane 21),
                        and the level after adoption against the months
                        before the rate hike.
  26-30                 the additional step at adoption and the step from
                        the 2023 level.
  Profile               22-25 and 50 and over, each against 41-49, from
                        one panel of all six bands.
  Margin and incidence  hires, separations, the female differential and
                        the part of it that lies within broad education
                        tracks.

INPUTS, read from the export directories the final-code manifest names.
Nothing is typed in; a missing row stops the script rather than printing
a blank that could be read as a zero.

  lane 14  seasonal_pooled.csv    gamma_1, gamma_2, the as-of arm, hires,
                                  separations (script 68)
  lane 21  reference_window.csv   the level after adoption and, with the
           vcov_s75_*_stock.csv     covariance files, the step from the 2023
                                  level (script 75)
  lane 20  contrast_seasonal.csv  the six-band profile (script 74)
  lane 14  seasonal_gender.csv    the female differential (script 68)
  lane 22  gender_split.csv       the within-track differential (script 76)

The vintage re-scoring is the change in gamma_2 when each employer's 2019
incumbents are re-scored from the education register as it stood in
2021, the staleness the 2024-25 records inherit, and the same panel is
re-estimated (the as-of arm of script 68). It is not the backtest of
Online Appendix IV.3, which runs on 2019-2023 with a pseudo-dated
treatment; the reported design admits no occupation code after 2019, so
the backtest does not apply to it. The 26-30 arm was not re-scored, so
that cell is left empty.

OUTPUT. revision/tables/table1_headline.tex, copied to the manuscript
repository's tables/ folder. Pass one directory on the command line to
read every input from there instead of the pinned locations.

    python3 revision/local/l18_table1.py [export_dir]
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
    window specification of lane 21 (post minus interim, both measured
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


def est(c: float, se: float) -> str:
    return f"${c:+.4f}$ ({se:.4f})"


def main() -> int:
    pooled = pd.read_csv(source(LANE14, "seasonal_pooled.csv"))
    pooled = pooled[pooled.get("status", "ok") == "ok"]
    window = pd.read_csv(source(LANE21_22, "reference_window.csv"))
    window = window[window.get("status", "ok") == "ok"]
    profile = pd.read_csv(source(LANE20, "contrast_seasonal.csv"))
    gender = pd.read_csv(source(LANE14_GENDER, "seasonal_gender.csv"))
    gender = gender[gender.get("status", "ok") == "ok"]
    split = pd.read_csv(source(LANE21_22, "gender_split.csv"))

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
    fem = one(gender, term=FEMALE)
    if len(split) != 1:
        raise SystemExit("  gender_split.csv should hold one row")
    within = (float(split.within.iloc[0]), float(split.within_se.iloc[0]))
    if abs(float(split.pooled.iloc[0]) - fem[0]) > 5e-5:
        raise SystemExit("  the pooled differential in gender_split.csv "
                         "does not match seasonal_gender.csv")

    rows = [
        (r"\multicolumn{3}{l}{\textit{Ages 22--25, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Tightening months, April to November 2022 ($\hat\gamma_1$)",
         est(*g1), ""),
        (r"Additional step at adoption, from January 2024 ($\hat\gamma_2$)",
         est(*g2), f"${artefact:+.4f}$"),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23), ""),
        (r"Level after adoption, against the months before the hike",
         est(*level), ""),
        (r"\addlinespace[3pt]", None, None),
        (r"\multicolumn{3}{l}{\textit{Ages 26--30, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Additional step at adoption ($\hat\gamma_2$)", est(*g2_26), ""),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23_26), ""),
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
        (r"Young women minus young men", est(*fem), ""),
        (r"\quad of which within broad education tracks", est(*within), ""),
    ]

    note = (
        r"Poisson pseudo-maximum likelihood on employer $\times$ age $\times$ "
        r"month counts, employer-by-month, employer-by-age and month-by-age "
        r"effects, three calendar-quarter interactions with the fourth "
        r"quarter omitted, exposure frozen at the employer's 2019 education "
        r"mix, standard errors clustered by employer. Steps are measured "
        r"from the level reached during the tightening months, as "
        r"$\beta_2$ is on the posting margin; the level row re-estimates "
        r"with the Riksbank interaction as a window so that the "
        r"post-adoption term reads against January 2021 to March 2022, "
        r"and the step from the 2023 level is the difference between its "
        r"post-adoption and interim terms, with the standard error from "
        r"their covariance. The "
        r"profile rows come from one panel of all six bands with 41--49 as "
        r"the reference. The female differential is the interaction of the "
        r"adoption term with a female indicator, with "
        r"employer-by-age-and-sex and month-by-age-and-sex effects, in a "
        r"specification that carries the Riksbank and adoption terms "
        r"without the interim term, so that its steps are from the level "
        r"of April 2022 to December 2023 and the female interaction is the "
        r"differential change from January 2021 to December 2023; the "
        r"within-track row weights the same differential estimated inside "
        r"each broad education track by young women's track shares in "
        r"exposed firms. The vintage re-scoring column is the change in the "
        r"coefficient when each employer's 2019 incumbents are re-scored "
        r"from the education register as it stood in 2021, the staleness "
        r"the 2024--25 records inherit, and the same panel is "
        r"re-estimated; the threshold fixed before that test was 0.05. "
        r"Full tables in Online Appendix~III.2."
    )

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Employment of young workers relative to their older "
           r"colleagues inside the same employer, with the calendar cycle "
           r"removed.}",
           r"\label{tab:headline}", r"\footnotesize",
           r"\begin{tabular}{lcc}", r"\toprule",
           r" & Estimate (SE) & Vintage re-scoring \\", r"\midrule"]
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

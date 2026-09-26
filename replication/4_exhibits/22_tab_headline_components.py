#!/usr/bin/env python3
"""
22_tab_headline_components.py: Online Appendix Table A15 (Section III.2),
the components behind the headline: the steps from the tightening months,
the pre-launch drift, the age profile, the flows and the sexes.

These are the rows of Equation (2) in its cumulative parameterisation, where
the tightening indicator stays on through the later periods so that gamma_2
is the step from April to November 2022 to January 2024 to June 2025, a
different contrast from the headline tau of Table 1. Poisson pseudo-maximum
likelihood with employer-by-month, employer-by-age and month-by-age effects
and the calendar-quarter terms, exposure the top quartile of the employer's
2019 occupation mix, clustered by employer and, in the second column, by
three-digit industry.

Panel A: gamma_2 at 22-25 and 26-30 against the older bands pooled (script
82, part B; industry standard errors from script 80's clustering within 83).
Panel B: the pre-launch drift per month, January 2021 to November 2022
(script 83, part D). Panel C: the profile against 41-49 alone, the step at
22-25 and at 50 and over (script 82). Panel D: hires and separations at
22-25 (script 82). Panel E: the sexes at 22-25, young men, the female
differential and young women (the male step plus the differential, its
standard error from the exported covariance; script 82), and the female
differential within broad fields of education weighted by young women's track
shares (script 87).

The script refuses to write the table unless every industry-clustered fit
reproduces its employer-clustered coefficient to four decimals, the exported
row for young women equals the male step plus the differential on its own
covariance, and the pooled differential of the within-track split equals the
table's own female differential.

Exports read (3_register_mona/exports/):
  2026-09-23_0655_s82-partB_s83-partsBCD/  occ_route_headline.csv,
      occ_route_profile.csv, occ_route_gender.csv, occ_route_flows.csv,
      vcov_s82_gender_22_25.csv (script 82); occ_rest_drift.csv (script 83);
      occ_rest_cluster.csv with vcov_s80_gender_clind2_22_25.csv (script
      80's clustering within 83)
  2026-09-23_1352_s87/  occ_route_gender_split.csv (script 87)
Output: output/tables/tableA_headline_components.tex

    python 4_exhibits/22_tab_headline_components.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

OCC = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"
SPLIT = EXPORTS / "2026-09-23_1352_s87"

TERM = "post_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"
TREND = "trend_x_high_x_young"
ARM, FLOOR, LEVEL = "backward", 5, "uniform3"
TOL = 5e-5


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else OCC


def read(name: str, where: Path = None) -> pd.DataFrame:
    p = (where or d()) / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    t = pd.read_csv(p)
    return t[t.status.isin(["ok", "derived"])] if "status" in t.columns else t


def vcov(name: str) -> pd.DataFrame:
    p = d() / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return pd.read_csv(p, index_col=0)


def one(df: pd.DataFrame, **cond) -> tuple[float, float]:
    r = df
    for k, v in cond.items():
        r = r[r[k] == v]
    if len(r) != 1:
        raise SystemExit(f"  expected one row for {cond}, found {len(r)}")
    return float(r.coef.iloc[0]), float(r.se.iloc[0])


def se_sum(v: pd.DataFrame, a: str, b: str) -> float:
    return float(v.loc[a, a] + v.loc[b, b] + 2 * v.loc[a, b]) ** 0.5


def est(c: float, se: float) -> str:
    return f"${c:+.4f}$ ({se:.4f})"


def line(label: str, e: tuple[float, float], ind: float | None = None) -> str:
    """One row: the estimate with its employer standard error and, where
    the fit was also clustered by industry, that standard error."""
    if ind is None:
        return f"{label} & {est(*e)} & \\\\"
    return f"{label} & {est(*e)} & ({ind:.4f}) \\\\"


def main() -> int:
    print(f"  export {d().name}")
    head = read("occ_route_headline.csv")
    head = head[(head.arm == ARM) & (head.floor == FLOOR)
                & (head.level == LEVEL) & (head.outcome == "stock")]
    profile = read("occ_route_profile.csv")
    gender = read("occ_route_gender.csv")
    flows = read("occ_route_flows.csv")
    drift = read("occ_rest_drift.csv")
    cluster = read("occ_rest_cluster.csv")
    clind, gsex = cluster[cluster.spec == "pooled"], cluster[cluster.spec == "gender"]
    split = read("occ_route_gender_split.csv", SPLIT)

    # Inference only: a moved coefficient means a moved panel, and no
    # industry standard error from that fit may be quoted.
    for name, t in (("pooled", clind), ("sex", gsex)):
        if not bool(t.coef_match_4dp.all()):
            raise SystemExit(f"  a {name} coefficient does not reproduce to "
                             f"four decimals; no industry SE is quotable")

    def se_ind(frame: pd.DataFrame, band: str, term: str, c: float) -> float:
        r = frame[(frame.young_band == band) & (frame.term == term)]
        if len(r) != 1:
            raise SystemExit(f"  expected one industry row for {band} {term}")
        if abs(float(r.coef.iloc[0]) - c) > TOL:
            raise SystemExit(f"  {band} {term}: the industry-clustered run "
                             f"does not reproduce the reported coefficient")
        return float(r.se_industry_complete.iloc[0])

    g2 = one(head, young_band="22-25", term=TERM)
    g2_26 = one(head, young_band="26-30", term=TERM)
    drift22 = one(drift, young_band="22-25", term=TREND)
    drift26 = one(drift, young_band="26-30", term=TREND)
    p22 = one(profile, band="22-25")
    p50 = one(profile, band="50+")
    hires = one(flows, outcome="hires", young_band="22-25", term=TERM)
    seps = one(flows, outcome="seps", young_band="22-25", term=TERM)
    men = one(gender, block="term", young_band="22-25", term=TERM)
    fem = one(gender, block="term", young_band="22-25", term=FEMALE)
    women = one(gender, block="step", young_band="22-25", term="female_step")

    # The script exported the sum and its standard error; recompute it from
    # the covariance and refuse the table if the two disagree.
    vg = vcov("vcov_s82_gender_22_25.csv")
    if abs(se_sum(vg, TERM, FEMALE) - women[1]) > TOL \
            or abs(men[0] + fem[0] - women[0]) > TOL:
        raise SystemExit("  the exported young-women row does not equal the "
                         "male step plus the differential on its own covariance")
    vgi = vcov("vcov_s80_gender_clind2_22_25.csv")
    for t in (TERM, FEMALE):
        c = one(gender, block="term", young_band="22-25", term=t)[0]
        if abs(float(vgi.loc[t, t]) ** 0.5 - se_ind(gsex, "22-25", t, c)) > TOL:
            raise SystemExit(f"  {t}: the exported industry SE is not the "
                             f"square root of its own variance")
    women_ind = se_sum(vgi, TERM, FEMALE)
    if len(split) != 1:
        raise SystemExit("  occ_route_gender_split.csv should hold one row")
    if abs(float(split.pooled.iloc[0]) - fem[0]) > TOL:
        raise SystemExit(f"  the split's pooled differential "
                         f"{float(split.pooled.iloc[0]):+.6f} is not this "
                         f"table's female differential {fem[0]:+.6f}, so the "
                         f"within-track row is not on this panel")
    within = (float(split.within.iloc[0]), float(split.within_se.iloc[0]))

    rows = [
        r"\multicolumn{3}{l}{\textit{Panel A. Step from the tightening months "
        r"to the later period ($\hat\gamma_2$)}} \\",
        line("Ages 22--25, against the older bands pooled", g2,
             se_ind(clind, "22-25", TERM, g2[0])),
        line("Ages 26--30, against the older bands pooled", g2_26,
             se_ind(clind, "26-30", TERM, g2_26[0])),
        r"\addlinespace[3pt]",
        r"\multicolumn{3}{l}{\textit{Panel B. Pre-launch drift per month, "
        r"January 2021 to November 2022}} \\",
        line("Ages 22--25", drift22),
        line("Ages 26--30", drift26),
        r"\addlinespace[3pt]",
        r"\multicolumn{3}{l}{\textit{Panel C. The profile in the later "
        r"period, against 41--49 alone}} \\",
        line("22--25", p22),
        line("50 and over", p50),
        r"\addlinespace[3pt]",
        r"\multicolumn{3}{l}{\textit{Panel D. Flows at 22--25, step from the "
        r"tightening months}} \\",
        line("Hires", hires),
        line("Separations", seps),
        r"\addlinespace[3pt]",
        r"\multicolumn{3}{l}{\textit{Panel E. The sexes at 22--25, step from "
        r"the tightening months}} \\",
        line("Young men", men, se_ind(gsex, "22-25", TERM, men[0])),
        line("Young women minus young men", fem,
             se_ind(gsex, "22-25", FEMALE, fem[0])),
        line("Young women", women, women_ind),
        line("Young women minus young men, within broad education tracks",
             within),
    ]
    for r in rows:
        if not r.startswith("\\"):
            print("  " + r[:110])

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Components behind the headline: steps from the "
           r"tightening months, drift, the age profile, flows and the sexes}",
           r"\label{tab:headline_components}", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lcc}", r"\toprule",
           r" & Estimate (employer SE) & Industry SE \\", r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson with the effects and calendar terms of Equation~(2), exposure the employer's 2019 occupation mix, clustered by employer and, in the second column, by three-digit industry. Panels A, D and E report the cumulative parameterisation, the step from April to November 2022 to January 2024 to June 2025, a different contrast from the headline $\tau$ of Table~1 of the paper; the period coefficients behind $\tau$ are in Table~\ref{tab:window}. The drift is a linear monthly trend. Hires and separations are relative counts, not hazards. The within-track row is the female differential estimated within each broad field of the worker's own education and weighted by young women's track shares (Table~\ref{tab:gender_split}).",
            r"\end{minipage}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_headline_components.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

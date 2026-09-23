#!/usr/bin/env python3
"""
l35_table1_v3.py: Table 1 of the v3 paper, on the occupation route.

WHAT THIS IS
l18_table1.py assembles Table 1 with exposure routed through education:
an education group carries the mean DAIOE percentile of the occupations
its holders worked in during 2019, and the employer is ranked by the mean
over its incumbents. This script assembles the same table with the
intermediate step deleted: the employer is ranked directly by the
employment-weighted mean DAIOE percentile of the 2019 occupations of its
OWN incumbents aged 31 to 69, scored from the three-digit book, codes
carried back to 2015 where 2019 is missing, and a floor of five incumbent
person-months. No education record enters the score anywhere.

Same rows, same base, same two clusterings as l18. Every row is read
from the lane 28 and lane 29 exports and nothing is typed in; a missing
or ambiguous row stops the script.

THE ONE EDUCATION ROW, AND WHY IT STAYS
The last row, the part of the female differential that survives within
broad education tracks, is script 76's and is education-based by
construction: it asks how much of the differential is composition across
tracks. It is the one place the paper still uses the education register,
it is credited to Nordstrom Skans and Sokolow Romin in the text, and the
note says so. It is read from the education-route export unchanged.

THE INDUSTRY COLUMN
Script 80's Part B was refitted on THIS quartile in lane 29b, with its
prior-export search path emptied so the employer-clustered run the
four-decimal gate checks against is its own and not the education
route's. The script refuses to write the table unless every one of those
fits reproduces its own employer-clustered coefficient to four decimals.

INPUTS AND OUTPUTS
Reads, from the lane 28b/29b-d export directory (or one given on the
command line): occ_route_headline.csv and occ_route_profile.csv (script
82, part B); occ_route_gender.csv, occ_route_flows.csv and
vcov_s82_gender_22_25.csv (script 82, part C); occ_rest_window.csv with
vcov_s83_window_<band>.csv, occ_rest_drift.csv, occ_rest_cluster.csv with
vcov_s80_clind2_<band>.csv and vcov_s80_gender_clind2_22_25.csv (script
83, part B). Reads gender_split.csv from the education route's lane 21-22
export for the within-track row alone. Writes
revision/tables/table1_headline_v3.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l35_table1_v3.py [export_dir]

IN THE PAPER
Table 1 of main_v3.tex, Section 3.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
# Lane 28b and lanes 29b, c, d came back in one folder, which also carries
# lane 28c's tables, so one directory holds every occupation-route input.
OCC = OUT / "round3_20260923-0655-lanes28b-29bcd"
# The within-track row alone is the education route's.
EDU = OUT / "round3_20260922-0712-lanes21-22"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

TERM = "post_x_high_x_young"
INTER = "interim_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"
# The reported score: the uniform three-digit arm, the backward cascade,
# a floor of five incumbent person-months. The robustness arms are in the
# same export and are never read here.
ARM, FLOOR, LEVEL = "backward", 5, "uniform3"


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else OCC


def read(name: str, where: Path = None) -> pd.DataFrame:
    p = (where or d()) / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    t = pd.read_csv(p)
    return t[t.get("status", "ok").isin(["ok", "derived"])] \
        if "status" in t.columns else t


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


def se_lin(v: pd.DataFrame, a: str, b: str, sign: int) -> float:
    return float(v.loc[a, a] + v.loc[b, b] + 2 * sign * v.loc[a, b]) ** 0.5


def est(c: float, se: float) -> str:
    return f"${c:+.4f}$ ({se:.4f})"


def ind(se) -> str:
    return "" if se is None else f"({se:.4f})"


def main() -> int:
    head = read("occ_route_headline.csv")
    head = head[(head.arm == ARM) & (head.floor == FLOOR)
                & (head.level == LEVEL) & (head.outcome == "stock")]
    profile = read("occ_route_profile.csv")
    gender = read("occ_route_gender.csv")
    flows = read("occ_route_flows.csv")
    window = read("occ_rest_window.csv")
    drift = read("occ_rest_drift.csv")
    clind2 = read("occ_rest_cluster.csv")
    clind, gsex = clind2[clind2.spec == "pooled"], clind2[clind2.spec == "gender"]
    split = read("gender_split.csv", EDU)

    # Inference only: a moved coefficient means a moved panel, and no
    # industry standard error from that fit may be quoted.
    for name, t in (("pooled", clind), ("sex", gsex)):
        if not bool(t.coef_match_4dp.all()):
            raise SystemExit(f"  lane 29b: a {name} coefficient does not "
                             f"reproduce to four decimals; no industry SE "
                             f"is quotable")

    def se_ind(band: str, term: str, c: float) -> float:
        r = clind[(clind.young_band == band) & (clind.term == term)]
        if len(r) != 1:
            raise SystemExit(f"  expected one industry row for {band} {term}")
        if abs(float(r.coef.iloc[0]) - c) > 5e-5:
            raise SystemExit(f"  {band} {term}: the industry-clustered run "
                             f"does not reproduce the reported coefficient")
        return float(r.se_industry_complete.iloc[0])

    def se_ind_sex(term: str, c: float) -> float:
        r = gsex[(gsex.young_band == "22-25") & (gsex.term == term)]
        if len(r) != 1:
            raise SystemExit(f"  expected one sex row for {term}")
        if abs(float(r.coef.iloc[0]) - c) > 5e-5:
            raise SystemExit(f"  {term}: the industry-clustered sex run does "
                             f"not reproduce the reported coefficient")
        return float(r.se_industry_complete.iloc[0])

    def step_from_2023(band: str) -> tuple[float, float]:
        """Post minus interim on the window specification, both measured
        against the months before the rate rise, so the difference is
        gamma_2 minus gamma_0 of Equation (2); the standard error from the
        exported covariance of the two terms."""
        post = one(window, young_band=band, outcome="stock", term=TERM)
        inter = one(window, young_band=band, outcome="stock", term=INTER)
        v = vcov(f"vcov_s83_window_{band.replace('-', '_')}.csv")
        var = (v.loc[TERM, TERM] + v.loc[INTER, INTER]
               - 2 * v.loc[TERM, INTER])
        return post[0] - inter[0], float(var) ** 0.5

    # 22-25, the sequence
    g1 = one(head, young_band="22-25", term="rb_x_high_x_young")
    g2 = one(head, young_band="22-25", term=TERM)
    level = one(window, young_band="22-25", outcome="stock", term=TERM)
    step23 = step_from_2023("22-25")
    TREND = "trend_x_high_x_young"
    drift22 = one(drift, young_band="22-25", term=TREND)
    drift26 = one(drift, young_band="26-30", term=TREND)
    # 26-30
    g2_26 = one(head, young_band="26-30", term=TERM)
    step23_26 = step_from_2023("26-30")
    # the profile against 41-49
    p22 = one(profile, band="22-25")
    p50 = one(profile, band="50+")
    # margin and incidence
    hires = one(flows, outcome="hires", young_band="22-25", term=TERM)
    seps = one(flows, outcome="seps", young_band="22-25", term=TERM)
    men = one(gender, block="term", young_band="22-25", term=TERM)
    fem = one(gender, block="term", young_band="22-25", term=FEMALE)
    women = one(gender, block="step", young_band="22-25", term="female_step")
    # The script exported the sum and its standard error; recompute it from
    # the covariance and refuse the table if the two disagree.
    vg = vcov("vcov_s82_gender_22_25.csv")
    chk = se_lin(vg, TERM, FEMALE, +1)
    if abs(chk - women[1]) > 5e-5 or abs(men[0] + fem[0] - women[0]) > 5e-5:
        raise SystemExit("  the exported young-women row does not equal the "
                         "male step plus the differential on its own "
                         "covariance")
    vgi = vcov("vcov_s80_gender_clind2_22_25.csv")
    for t in (TERM, FEMALE):
        if abs(float(vgi.loc[t, t]) ** 0.5
               - se_ind_sex(t, one(gender, block="term",
                                   young_band="22-25", term=t)[0])) > 5e-5:
            raise SystemExit(f"  {t}: the exported industry SE is not the "
                             f"square root of its own variance")
    women_ind = se_lin(vgi, TERM, FEMALE, +1)
    if len(split) != 1:
        raise SystemExit("  gender_split.csv should hold one row")
    within = (float(split.within.iloc[0]), float(split.within_se.iloc[0]))

    v22, v26 = vcov("vcov_s80_clind2_22_25.csv"), vcov("vcov_s80_clind2_26_30.csv")
    step23_ind = se_lin(v22, TERM, INTER, -1)
    step23_26_ind = se_lin(v26, TERM, INTER, -1)
    for band, got in (("22-25", step23[0]), ("26-30", step23_26[0])):
        r = clind[clind.young_band == band].set_index("term")
        delta = float(r.loc[TERM, "coef"]) - float(r.loc[INTER, "coef"])
        if abs(delta - got) > 5e-5:
            raise SystemExit(f"  {band}: the step from the 2023 level in the "
                             f"industry-clustered run is {delta:+.4f}, not "
                             f"{got:+.4f}")

    rows = [
        (r"\multicolumn{3}{l}{\textit{Ages 22--25, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Tightening months, April to November 2022 ($\hat\gamma_1$)",
         est(*g1), ind(se_ind("22-25", "rb_x_high_x_young", g1[0]))),
        (r"Additional step at adoption, from January 2024 ($\hat\gamma_2$)",
         est(*g2), ind(se_ind("22-25", TERM, g2[0]))),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23), ind(step23_ind)),
        (r"Level after adoption, against the pre-hike months",
         est(*level), ""),
        (r"Pre-launch drift per month, to November 2022",
         est(*drift22), ""),
        (r"\addlinespace[3pt]", None, None),
        (r"\multicolumn{3}{l}{\textit{Ages 26--30, employment stock, "
         r"against the older bands pooled}} \\", None, None),
        (r"Additional step at adoption ($\hat\gamma_2$)",
         est(*g2_26), ind(se_ind("26-30", TERM, g2_26[0]))),
        (r"Step from the 2023 level ($\hat\gamma_2 - \hat\gamma_0$)",
         est(*step23_26), ind(step23_26_ind)),
        (r"Pre-launch drift per month, to November 2022",
         est(*drift26), ""),
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
        (r"Young men, additional step at adoption", est(*men),
         ind(se_ind_sex(TERM, men[0]))),
        (r"Young women minus young men", est(*fem),
         ind(se_ind_sex(FEMALE, fem[0]))),
        (r"Young women, additional step at adoption", est(*women),
         ind(women_ind)),
        (r"\quad within broad education tracks", est(*within), ""),
    ]

    note = (
        r"Poisson on employer $\times$ age $\times$ month counts, with the "
        r"effects and calendar terms of Equation~(2); clustered by employer. "
        r"Exposure is the employer's 2019 occupation mix. "
        r"Steps read from the tightening level; the level row is a window "
        r"against January 2021 to March 2022, and the step from 2023 is post "
        r"minus interim. The third column clusters the same fits on "
        r"three-digit industry. The sex rows interact every treatment term "
        r"with female; young women is the male step plus the differential. "
        r"The drift rows are a linear monthly trend to November 2022. The "
        r"within-track row alone is cut by education, and it alone is "
        r"estimated on the earlier education-mix score, whose pooled "
        r"differential on that panel is $-0.0659$ (0.0131); three quarters "
        r"of it survives within tracks, and it is not comparable with the "
        r"row above. Full tables in Online Appendix~III.2."
    )

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Employment of young workers relative to their older "
           r"colleagues inside the same employer, with the calendar cycle "
           r"removed.}",
           r"\label{tab:headline}", r"\footnotesize",
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
    out = V2_TAB / "table1_headline_v3.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

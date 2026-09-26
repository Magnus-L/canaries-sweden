#!/usr/bin/env python3
"""
01_table1_headline.py: Table 1 of the paper, exposure-related changes in
within-employer employment composition.

Every row is tau = b_L - b_I of Equation (2), the later period (January 2024
to June 2025) against the interim period (December 2022 to December 2023),
at top-quartile relative to less exposed employers: Poisson pseudo-maximum
likelihood on employer by age band by month counts with employer-by-month,
employer-by-age and month-by-age effects and three calendar-quarter terms,
exposure the top quartile of the employment-weighted mean DAIOE percentile
of the 2019 occupations of the employer's incumbents aged 31 to 69, standard
errors clustered by employer and, in the last column, by three-digit
industry. The pooled rows (22-25, 26-30) take tau and its employer standard
error from the window parameterisation of script 83, whose post and interim
terms are levels against January 2021 to March 2022, and the industry
standard error from script 80's industry-clustered fit of the same model.
The sex rows are linear combinations of the terms of one specification that
interacts every treatment term with being female (script 82), with the
covariance of the employer-clustered and the industry-clustered fit.

Nothing is written unless: every industry-clustered fit reproduces its
employer-clustered coefficient to four decimals; every exported standard
error is the square root of its own covariance diagonal; the window's tau
equals the cumulative parameterisation's post minus interim term; and every
tau and its employer standard error agree, to four decimals, with the gate
that script 97 re-estimated in a separate run (headline_checks.csv, parts
G and sex_gate), which is the second record of every estimate in the table.

Exports read (3_register_mona/exports/):
  2026-09-23_0655_s82-partB_s83-partsBCD/  occ_route_headline.csv,
      occ_route_gender.csv (script 82); occ_rest_window.csv with
      vcov_s83_window_<band>.csv (script 83); occ_rest_cluster.csv with
      vcov_s80_clind2_<band>.csv, vcov_s80_gender_clemp_22_25.csv and
      vcov_s80_gender_clind2_22_25.csv (script 80 within 83)
  2026-09-25_2250_s97/  headline_checks.csv (script 97, the second record)
Output: output/tables/table1_headline_v3.tex

    python 4_exhibits/01_table1_headline.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

# Scripts 82 (part B) and 83 (parts B to D) were exported together, so one
# folder holds every input of the table.
OCC = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"
# Script 97 re-estimated tau on the same panels as its gate before running
# its own checks; those rows are the independent record this table is
# checked against.
S97 = EXPORTS / "2026-09-25_2250_s97"

POST = "post_x_high_x_young"
INTER = "interim_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"
FEM_INTER = "interim_x_high_x_young_x_female"
# The reported score: the uniform three-digit arm, the backward cascade,
# a floor of five incumbent person-months.
ARM, FLOOR, LEVEL = "backward", 5, "uniform3"
BANDS = ["22-25", "26-30"]
TOL = 5e-5          # agreement between two records of one coefficient
GATE_DP = 4         # agreement with script 97's re-estimation, decimals


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


def lin(v: pd.DataFrame, w: dict) -> float:
    """Standard error of the linear combination sum(w[t] * b[t])."""
    t = list(w)
    x = pd.Series(w, dtype=float)
    return float(x @ v.loc[t, t] @ x) ** 0.5


def diag_check(v: pd.DataFrame, df: pd.DataFrame, terms: list, se_col: str,
               what: str, **cond) -> None:
    """Every exported standard error is the square root of its own
    variance in the exported covariance, so the covariance is the one the
    derived rows may be built from."""
    r = df
    for k, val in cond.items():
        r = r[r[k] == val]
    r = r.set_index("term")
    for t in terms:
        if t not in v.index or t not in v.columns:
            raise SystemExit(f"  {what}: {t} is not in the covariance export")
        diag = float(v.loc[t, t]) ** 0.5
        got = float(r.loc[t, se_col])
        if abs(diag - got) > TOL:
            raise SystemExit(f"  {what} {t}: the exported standard error "
                             f"{got:.6f} is not the square root of its own "
                             f"variance {diag:.6f}; nothing is written")


def gate(said: pd.DataFrame, spec: str, band: str, term: str,
         got: tuple[float, float], what: str) -> None:
    """Script 97's own re-estimation of tau must agree to four decimals."""
    r = said[(said.part == "G") & (said.spec == spec)
             & (said.young_band == band) & (said.term == term)]
    if len(r) != 1:
        raise SystemExit(f"  headline_checks.csv: one {spec} {term} row "
                         f"expected for {band}, found {len(r)}")
    c, se = float(r.coef.iloc[0]), float(r.se.iloc[0])
    if round(c, GATE_DP) != round(got[0], GATE_DP) \
            or round(se, GATE_DP) != round(got[1], GATE_DP):
        raise SystemExit(f"  {what}: this table gives {got[0]:+.6f} "
                         f"({got[1]:.6f}) and script 97's gate "
                         f"{c:+.6f} ({se:.6f}); nothing is written")


def cell(c: float, se_emp: float, se_ind: float) -> str:
    return f"${c:+.3f}$ & ({se_emp:.3f}) & ({se_ind:.3f})"


def main() -> int:
    print(f"  export {d().name}")
    head = read("occ_route_headline.csv")
    head = head[(head.arm == ARM) & (head.floor == FLOOR)
                & (head.level == LEVEL) & (head.outcome == "stock")]
    window = read("occ_rest_window.csv")
    cluster = read("occ_rest_cluster.csv")
    pooled = cluster[cluster.spec == "pooled"]
    sexes = cluster[cluster.spec == "gender"]
    gender = read("occ_route_gender.csv")
    gender = gender[gender.block == "term"]
    said = read("headline_checks.csv", S97)

    # Inference only: a moved coefficient means a moved panel, and no
    # industry standard error from that fit may be quoted.
    for name, t in (("pooled", pooled), ("sex", sexes)):
        if not bool(t.coef_match_4dp.all()):
            raise SystemExit(f"  a {name} coefficient does not reproduce its "
                             f"employer-clustered run to four decimals; no "
                             f"industry SE is quotable")

    rows, firms = {}, {}
    for band in BANDS:
        us = band.replace("-", "_")
        vw = vcov(f"vcov_s83_window_{us}.csv")
        vi = vcov(f"vcov_s80_clind2_{us}.csv")
        diag_check(vw, window, [POST, INTER], "se", f"{band} window",
                   young_band=band, outcome="stock")
        diag_check(vi, pooled, [POST, INTER], "se_industry_complete",
                   f"{band} industry", young_band=band)
        # tau = post minus interim on the window parameterisation, both
        # levels against the months before the rate rise; the covariance
        # of the two terms gives the standard error.
        post = one(window, young_band=band, outcome="stock", term=POST)
        inter = one(window, young_band=band, outcome="stock", term=INTER)
        tau = post[0] - inter[0]
        se_emp = lin(vw, {POST: 1, INTER: -1})
        # The cumulative parameterisation describes the same fitted means,
        # so its post minus interim term is the same tau.
        g2 = one(head, young_band=band, term=POST)
        g0 = one(head, young_band=band, term=INTER)
        if abs((g2[0] - g0[0]) - tau) > TOL:
            raise SystemExit(f"  {band}: the window gives tau {tau:+.6f} and "
                             f"the cumulative parameterisation "
                             f"{g2[0] - g0[0]:+.6f}; nothing is written")
        r = pooled[pooled.young_band == band].set_index("term")
        ind_tau = float(r.loc[POST, "coef"]) - float(r.loc[INTER, "coef"])
        if abs(ind_tau - tau) > TOL:
            raise SystemExit(f"  {band}: the industry-clustered run gives "
                             f"tau {ind_tau:+.6f}, not {tau:+.6f}")
        se_ind = lin(vi, {POST: 1, INTER: -1})
        gate(said, "gate", band, "hy_tau", (tau, se_emp), f"{band} tau")
        n = set(int(x) for x in window[window.young_band == band].n_firms)
        if len(n) != 1:
            raise SystemExit(f"  {band}: one employer count expected, {n}")
        firms[band] = n.pop()
        rows[band] = (tau, se_emp, se_ind)
        print(f"  {band}: tau {tau:+.4f} ({se_emp:.4f}) industry ({se_ind:.4f}), "
              f"{firms[band]:,} employers")

    # The sexes: one fit at 22-25 with every treatment term interacted with
    # being female. tau_M is the male post minus interim; the differential
    # tau_F - tau_M the female post minus the female interim; tau_F their sum.
    ve = vcov("vcov_s80_gender_clemp_22_25.csv")
    vi = vcov("vcov_s80_gender_clind2_22_25.csv")
    sex_terms = [POST, INTER, FEMALE, FEM_INTER]
    diag_check(ve, gender, sex_terms, "se", "sexes, employer", young_band="22-25")
    diag_check(vi, sexes, sex_terms, "se_industry_complete", "sexes, industry",
               young_band="22-25")
    g = gender[gender.young_band == "22-25"].set_index("term")
    w_men = {POST: 1, INTER: -1}
    w_diff = {FEMALE: 1, FEM_INTER: -1}
    w_women = {POST: 1, INTER: -1, FEMALE: 1, FEM_INTER: -1}
    est = {}
    for key, w in (("men", w_men), ("diff", w_diff), ("women", w_women)):
        c = sum(float(g.loc[t, "coef"]) * x for t, x in w.items())
        est[key] = (c, lin(ve, w), lin(vi, w))
        print(f"  {key:5s}: tau {c:+.4f} ({est[key][1]:.4f}) industry "
              f"({est[key][2]:.4f})")
    gate(said, "sex_gate", "22-25", "hy_tau", est["men"][:2], "young men")
    gate(said, "sex_gate", "22-25", "hyf_tau", est["diff"][:2],
         "the female differential")
    n_sex = set(int(x) for x in gender[gender.young_band == "22-25"].n_firms)
    if n_sex != {firms["22-25"]}:
        raise SystemExit(f"  the sex specification runs on {n_sex} employers, "
                         f"not the {firms['22-25']:,} of the pooled fit")
    print("  script 97's gate reproduces every tau and its standard error to "
          f"{GATE_DP} decimals")

    tex = [
        r"\begin{table}[ht!]", r"\centering",
        r"\caption{Exposure-related changes in within-employer employment "
        r"composition}",
        r"\label{tab:headline}", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"\multicolumn{4}{c}{\textit{January 2024 to June 2025 relative to "
        r"December 2022 to December 2023}} \\",
        r"\midrule",
        r" & Estimate & SE, employer & SE, industry \\",
        r"\midrule",
        f"Ages 22--25 relative to older colleagues & {cell(*rows['22-25'])} \\\\",
        f"Young women, 22--25, relative to older colleagues & "
        f"{cell(*est['women'])} \\\\",
        f"Young men, 22--25, relative to older colleagues & "
        f"{cell(*est['men'])} \\\\",
        r"Young women minus young men ($\hat\tau_F - \hat\tau_M$) & "
        + cell(*est["diff"]) + r" \\",
        r"\addlinespace[3pt]",
        f"Ages 26--30 relative to older colleagues & {cell(*rows['26-30'])} \\\\",
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
        r"Poisson estimates of $\tau = b_L - b_I$ in Equation~(2). Every row "
        r"compares top-quartile with less exposed employers on the same "
        r"contrast, the later period against the interim period. Older "
        r"colleagues are those aged 31 to 69 at the same employer. The sex "
        r"rows come from one specification that interacts every treatment "
        r"term with being female, with sex-specific employer-by-age and "
        r"age-by-month effects; their older colleagues are of both sexes. "
        r"Standard errors are clustered by employer and by three-digit "
        r"industry. In per cent, a row is $100[\exp(\tau)-1]$. Employers: "
        + f"{firms['22-25']:,}".replace(",", "{,}") + " at 22--25 and "
        + f"{firms['26-30']:,}".replace(",", "{,}") + " at 26--30.",
        r"\end{minipage}", r"\end{table}",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "table1_headline_v3.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

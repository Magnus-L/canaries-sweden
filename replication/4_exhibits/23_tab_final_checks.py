#!/usr/bin/env python3
"""
23_tab_final_checks.py: Online Appendix Table A21 (Section III.2), the checks
on tau from the final register run: pension ages, birth cohorts, credit, the
female differential and the exposure specification.

tau is the later-period coefficient minus the interim one from the same
Poisson fit of Equation (2), its standard error from the clustered covariance
of the two terms (V_LL + V_II - 2 V_LI), clustered by employer unless the row
says otherwise. Every fit reproduced Table 1 on its own sample before the
check was run (the gates in each script's summary).

Panel A: tau with the older reference restricted to 31-59 and 31-49, on the
employers that hold the young band and ages 31-49 (script 95, part R,
sample E49). Panel B: tau for each band against 41-49 in one fit of eight
bands (script 95, part P; the panel Figure 2 of the paper draws). Panel C:
fixed birth cohorts against those born 1956-1990, the cohorts the youth
payroll reduction never covered and those it did, their difference, and the
gradient per statutory year of eligibility (script 96, parts C and D).
Panel D: the credit test on tau, the employers with a 2019 balance sheet
split at median leverage (script 97, part K). Panel E: the female
differential at 22-25 with industry x age x sex x month effects and its
pre-launch drift, clustered by employer and by industry (script 97, parts I
and P). Panel F: the exposure specification, the continuous score per
baseline standard deviation and the quartiles against the first (script 97,
part Q). Panel G: tau and the female differential with eleven month-of-year
terms in place of the three calendar-quarter terms, each beside the same
run's gate on Table 1's specification, standard errors by employer and by
industry (script 102, parts G and M). Panel H: the headline design with employers
classified by the Eloundou et al. (2024) rating through the same chain, on the
employers both indices score, DAIOE beside Eloundou (script 103, parts D and E).

Nothing is written unless every exported standard error of a derived tau is
the square root of the exported variance of the difference, every count the
panel titles state is the one the export holds, and every estimate agrees to
its printed decimals with the summary its own run printed (95_summary.txt,
96_summary.txt, 97_summary.txt, 102_summary.txt, 103_summary.txt). The note's median leverage (0.693, 0.689),
third-quarter term and score standard deviation (17.6) are read from the
exports and the summaries too.

Exports read (3_register_mona/exports/):
  2026-09-25_1832_s95-s96-s98/  pension_reference.csv, 95_summary.txt,
      payroll_cohorts.csv, 96_summary.txt
  2026-09-25_2250_s97/  headline_checks.csv, 97_summary.txt
  2026-09-26_1208_s102/  month_of_year.csv, 102_summary.txt
  2026-09-26_1527_s103/  eloundou_classification.csv, classification_agreement.csv,
      103_summary.txt
Output: output/tables/tableA_final_checks.tex (a bare tabular and note; the
        appendix supplies the float and caption)

    python 4_exhibits/23_tab_final_checks.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S95 = EXPORTS / "2026-09-25_1832_s95-s96-s98"
S97 = EXPORTS / "2026-09-25_2250_s97"
S102 = EXPORTS / "2026-09-26_1208_s102"
S103 = EXPORTS / "2026-09-26_1527_s103"

RE_EST = r"([-+][0-9.]+) \(([0-9.]+)\)"


def need(folder: Path, name: str) -> Path:
    p = folder / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def rows_ok(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path)
    return t[t.status.isin(["ok", "derived"])]


def derived_se_check(r: pd.Series, what: str) -> None:
    """A derived tau's exported standard error must be the square root of
    the exported variance of the difference."""
    var = (float(r["var_post"]) + float(r["var_interim"])
           - 2.0 * float(r["cov_post_interim"]))
    if abs(np.sqrt(var) - float(r["se"])) > 1e-6:
        raise SystemExit(f"  {what}: the exported standard error {float(r['se']):.6f} "
                         f"is not the square root of the exported variance "
                         f"{np.sqrt(var):.6f}; nothing is written")


def pick(df: pd.DataFrame, what: str, **cond) -> pd.Series:
    r = df
    for k, v in cond.items():
        r = r[r[k] == v]
    if len(r) != 1:
        raise SystemExit(f"  {what}: expected one row for {cond}, found {len(r)}")
    return r.iloc[0]


def agree(what: str, got: tuple[float, float], said: tuple[float, float],
          dp: int = 4) -> None:
    if round(got[0], dp) != said[0] or round(got[1], dp) != said[1]:
        raise SystemExit(f"  {what}: the export gives {got[0]:+.{dp}f} "
                         f"({got[1]:.{dp}f}) and the run's summary "
                         f"{said[0]:+.{dp}f} ({said[1]:.{dp}f}); nothing is written")


def est(c: float, se: float, dp: int = 4) -> str:
    return f"${c:+.{dp}f}$ ({se:.{dp}f})"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def summary_lines(path: Path, pattern: str) -> dict:
    """Every line of a summary matching `pattern` (two groups: a key and
    the estimate), as key -> (coef, se)."""
    out = {}
    rx = re.compile(pattern)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = rx.match(line)
        if m:
            out[m.group(1).strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def panel_a() -> tuple[list, dict]:
    d = rows_ok(need(S95, "pension_reference.csv"))
    said = summary_lines(need(S95, "95_summary.txt"),
                         r"^\s*(\d\d-\d\d vs \d\d-\d\d): own [-+][0-9.]+ "
                         r"\([0-9.]+\)\s+E49 " + RE_EST)
    rows, firms = [], {}
    for band in ("22-25", "26-30"):
        cells = []
        for ref in ("31-69", "31-59", "31-49"):
            r = pick(d, "pension_reference.csv", part="R", spec=f"ref_{ref}",
                     sample="E49", young_band=band, term="tau")
            derived_se_check(r, f"Panel A {band} vs {ref}")
            agree(f"Panel A {band} vs {ref}", (float(r.coef), float(r.se)),
                  said[f"{band} vs {ref}"])
            firms.setdefault(band, set()).add(int(r.n_firms))
            cells.append(est(float(r.coef), float(r.se)))
        if len(firms[band]) != 1:
            raise SystemExit(f"  Panel A {band}: one employer count expected, "
                             f"{firms[band]}")
        n = firms[band].pop()
        rows.append(f"{band.replace('-', '--')} ({thousands(n)} employers) & "
                    + " & ".join(cells) + r" \\")
    return rows, {}


def panel_b() -> list:
    d = rows_ok(need(S95, "pension_reference.csv"))
    said = summary_lines(need(S95, "95_summary.txt"),
                         r"^\s*(\d\d-\d\d)\s+tau " + RE_EST)
    bands = ["22-25", "26-30", "31-34", "35-40", "50-59", "60-64", "65-69"]
    cells, firms = {}, set()
    for b in bands:
        r = pick(d, "pension_reference.csv", part="P", spec="p8_tau",
                 young_band=b, term="tau")
        derived_se_check(r, f"Panel B {b}")
        agree(f"Panel B {b}", (float(r.coef), float(r.se)), said[b])
        firms.add(int(r.n_firms))
        cells[b] = est(float(r.coef), float(r.se))
    if firms != {104333}:
        raise SystemExit(f"  Panel B: the eight-band panel holds {firms} "
                         f"employers, not the 104,333 the title states")
    left, right = bands[:4], bands[4:] + [None]
    rows = []
    for lb, rb in zip(left, right):
        if rb is None:
            rows.append(f"{lb.replace('-', '--')} & {cells[lb]} & & \\\\")
        else:
            rows.append(f"{lb.replace('-', '--')} & {cells[lb]} & "
                        f"{rb.replace('-', '--')} & {cells[rb]} \\\\")
    return rows


def panel_c() -> tuple[list, int]:
    d = rows_ok(need(S95, "payroll_cohorts.csv"))
    text = need(S95, "96_summary.txt").read_text(encoding="utf-8", errors="replace")
    said = {}
    for key, rx in (("nc", r"never covered, born 1994-1997\s+" + RE_EST),
                    ("ec", r"ever covered, born 1998-2003\s+" + RE_EST),
                    ("diff", r"1998-2003 minus 1994-1997\s+" + RE_EST),
                    ("dose", r"gradient_per_year_eligible\s+" + RE_EST)):
        m = re.search(rx, text)
        if not m:
            raise SystemExit(f"  96_summary.txt: no line for {key}")
        said[key] = (float(m.group(1)), float(m.group(2)))
    nc = pick(d, "payroll_cohorts.csv", part="C", spec="gradient_nc", term="gradient")
    ec = pick(d, "payroll_cohorts.csv", part="C", spec="gradient_ec", term="gradient")
    diff = pick(d, "payroll_cohorts.csv", part="C", spec="difference", term="ec_minus_nc")
    dose = pick(d, "payroll_cohorts.csv", part="D", spec="dose_linear",
                term="gradient_per_year_eligible")
    for what, r in (("nc", nc), ("ec", ec)):
        derived_se_check(r, f"Panel C {what}")
    for key, r in (("nc", nc), ("ec", ec), ("diff", diff), ("dose", dose)):
        agree(f"Panel C {key}", (float(r.coef), float(r.se)), said[key])
    if abs((float(ec.coef) - float(nc.coef)) - float(diff.coef)) > 1e-6:
        raise SystemExit("  Panel C: the exported difference is not ever "
                         "covered minus never covered")
    n_firms = int(diff.n_firms)
    if int(dose.n_firms) != n_firms:
        raise SystemExit("  Panel C: the cohort fits run on different panels")
    rows = [f"Never covered by the payroll reduction, born 1994--1997 & & & "
            f"{est(float(nc.coef), float(nc.se))} \\\\",
            f"Ever covered, born 1998--2003 & & & {est(float(ec.coef), float(ec.se))} \\\\",
            f"Difference & & & {est(float(diff.coef), float(diff.se))} \\\\",
            f"Per statutory year of eligibility, born 1994--2003 & & & "
            f"{est(float(dose.coef), float(dose.se))} \\\\"]
    return rows, n_firms


def panel_d(said_text: str) -> tuple[list, dict]:
    d = rows_ok(need(S97, "headline_checks.csv"))
    said = summary_lines(need(S97, "97_summary.txt"),
                         r"^\s*(\d\d-\d\d (?:base_balance_sheet|leverage)\s+"
                         r"\w+)\s+" + RE_EST)
    said = {re.sub(r"\s+", " ", k): v for k, v in said.items()}
    spec_term = [("Same-sample $\\tau$", "base_balance_sheet", "hy_tau"),
                 ("$\\tau$, less leveraged half", "leverage", "hy_tau"),
                 ("Additional $\\tau$, more leveraged half", "leverage", "hy_x_lev_tau"),
                 ("Young $\\times$ leverage, all employers", "leverage",
                  "young_x_lev_tau")]
    cells, firms, medians = {}, {}, {}
    for band in ("22-25", "26-30"):
        for label, spec, term in spec_term:
            r = pick(d, "headline_checks.csv", part="K", spec=spec,
                     young_band=band, term=term)
            derived_se_check(r, f"Panel D {band} {spec} {term}")
            agree(f"Panel D {band} {term}", (float(r.coef), float(r.se)),
                  said[f"{band} {spec} {term}"])
            firms.setdefault(band, set()).add(int(r.n_firms))
            cells[(band, label)] = est(float(r.coef), float(r.se))
        if len(firms[band]) != 1:
            raise SystemExit(f"  Panel D {band}: one employer count expected")
        m = re.search(rf"K/{band}: leverage split at ([0-9.]+)", said_text)
        if not m:
            raise SystemExit(f"  97_summary.txt: no leverage median for {band}")
        medians[band] = float(m.group(1))
    rows = [r" & & 22--25 & 26--30 \\"]
    for label, _, _ in spec_term:
        rows.append(f"{label} & & {cells[('22-25', label)]} & "
                    f"{cells[('26-30', label)]} \\\\")
    rows.append(f"Employers & & {thousands(firms['22-25'].pop())} & "
                f"{thousands(firms['26-30'].pop())} \\\\")
    return rows, medians


def panel_e(said_text: str) -> tuple[list, tuple[float, float]]:
    d = rows_ok(need(S97, "headline_checks.csv"))
    base = pick(d, "headline_checks.csv", part="I", spec="base_industry_sample",
                young_band="22-25", term="hyf_tau")
    ind = pick(d, "headline_checks.csv", part="I", spec="industry_age_sex_month",
               young_band="22-25", term="hyf_tau")
    for what, r in (("base", base), ("industry", ind)):
        derived_se_check(r, f"Panel E {what}")
    said_i = summary_lines(need(S97, "97_summary.txt"),
                           r"^\s*(base_industry_sample|industry_age_sex_month)\s+hyf_tau\s+"
                           + RE_EST)
    agree("Panel E same-sample", (float(base.coef), float(base.se)),
          said_i["base_industry_sample"])
    agree("Panel E industry", (float(ind.coef), float(ind.se)),
          said_i["industry_age_sex_month"])
    if int(base.n_firms) != int(ind.n_firms):
        raise SystemExit("  Panel E: the two industry fits run on different employers")
    tr_e = pick(d, "headline_checks.csv", part="P", spec="drift",
                young_band="22-25", term="trend_x_hyf")
    tr_i = pick(d, "headline_checks.csv", part="P", spec="drift_indcl",
                young_band="22-25", term="trend_x_hyf")
    if abs(float(tr_e.coef) - float(tr_i.coef)) > 1e-9:
        raise SystemExit("  Panel E: the two clusterings of the drift differ "
                         "in the coefficient")
    m = re.search(r"drift: trend ([-+][0-9.]+) per month, SE ([0-9.]+) by employer, "
                  r"([0-9.]+) by industry", said_text)
    if not m or round(float(tr_e.coef), 5) != float(m.group(1)) \
            or round(float(tr_e.se), 5) != float(m.group(2)) \
            or round(float(tr_i.se), 5) != float(m.group(3)):
        raise SystemExit("  Panel E: the drift does not match 97_summary.txt")
    q3 = pick(d, "headline_checks.csv", part="P", spec="drift",
              young_band="22-25", term="q3_x_hyf")
    rows = [f"Same-sample $\\tau$, {thousands(int(base.n_firms))} employers with an "
            f"industry code & & & {est(float(base.coef), float(base.se))} \\\\",
            f"With industry $\\times$ age $\\times$ sex $\\times$ month effects & & & "
            f"{est(float(ind.coef), float(ind.se))} \\\\",
            f"Pre-launch drift per month, Jan 2021 to Nov 2022, by employer & & & "
            f"{est(float(tr_e.coef), float(tr_e.se), 5)} \\\\",
            f"\\quad the same, clustered by industry & & & "
            f"{est(float(tr_i.coef), float(tr_i.se), 5)} \\\\"]
    return rows, (float(q3.coef), float(q3.se))


def panel_f(said_text: str) -> tuple[list, float]:
    d = rows_ok(need(S97, "headline_checks.csv"))
    z = pick(d, "headline_checks.csv", part="Q", spec="continuous_per_sd",
             young_band="22-25", term="z_tau")
    derived_se_check(z, "Panel F continuous")
    said = {}
    for key, rx in (("z", r"continuous score, per baseline SD\s+" + RE_EST),
                    ("q2", r"Q2 against Q1\s+" + RE_EST),
                    ("q3", r"Q3 against Q1\s+" + RE_EST),
                    ("q4", r"Q4 against Q1\s+" + RE_EST)):
        m = re.search(rx, said_text)
        if not m:
            raise SystemExit(f"  97_summary.txt: no line for {key}")
        said[key] = (float(m.group(1)), float(m.group(2)))
    agree("Panel F continuous", (float(z.coef), float(z.se)), said["z"])
    rows = [f"Continuous score, per baseline standard deviation & & & "
            f"{est(float(z.coef), float(z.se))} \\\\"]
    firms = {int(z.n_firms)}
    for q, label in (("q2", "Second"), ("q3", "Third"), ("q4", "Top")):
        r = pick(d, "headline_checks.csv", part="Q", spec="quartiles_vs_q1",
                 young_band="22-25", term=f"{q}_vs_q1_tau")
        derived_se_check(r, f"Panel F {q}")
        agree(f"Panel F {q}", (float(r.coef), float(r.se)), said[q])
        firms.add(int(r.n_firms))
        rows.append(f"{label} quartile against the first & & & "
                    f"{est(float(r.coef), float(r.se))} \\\\")
    if firms != {104217}:
        raise SystemExit(f"  Panel F: the fits run on {firms} employers, not "
                         f"the 104,217 the title states")
    m = re.search(r"SD ([0-9.]+) percentile points", said_text)
    if not m:
        raise SystemExit("  97_summary.txt: no standard deviation of the score")
    return rows, float(m.group(1))


def est2(c: float, se_emp: float, se_ind: float, dp: int = 4) -> str:
    return f"${c:+.{dp}f}$ ({se_emp:.{dp}f}; {se_ind:.{dp}f})"


def panel_g() -> list:
    """Eleven month-of-year terms in place of the three quarter terms
    (script 102): tau and the female differential, each with both
    clusterings, beside the same run's gate on Table 1's specification."""
    d = rows_ok(need(S102, "month_of_year.csv"))
    text = need(S102, "102_summary.txt").read_text(encoding="utf-8", errors="replace")
    # The summary prints the pair of lines twice: first for tau, then for the
    # female differential.
    said = re.findall(r"(quarter terms \(Table 1\)|month-of-year terms)\s+tau "
                      r"([-+][0-9.]+)\s+SE ([0-9.]+) by employer, ([0-9.]+) by industry",
                      text)
    if len(said) != 4:
        raise SystemExit("  102_summary.txt: expected four movement lines")
    rows, firms = [], set()
    for label, term, part_e, part_i, si in (
            ("$\\tau$ at 22--25", "hy_tau", ("gate", "month_of_year"),
             ("gate_indcl", "month_of_year_indcl"), 0),
            ("Female differential at 22--25", "hyf_tau",
             ("sex_gate", "sex_month_of_year"),
             ("sex_gate_indcl", "sex_month_of_year_indcl"), 2)):
        cells = []
        for j, (spec_e, spec_i) in enumerate(zip(part_e, part_i)):
            re_ = pick(d, "month_of_year.csv", spec=spec_e, young_band="22-25", term=term)
            ri_ = pick(d, "month_of_year.csv", spec=spec_i, young_band="22-25", term=term)
            for what, r in (("employer", re_), ("industry", ri_)):
                derived_se_check(r, f"Panel G {label} {spec_e} {what}")
                firms.add(int(r.n_firms))
            if abs(float(re_.coef) - float(ri_.coef)) > 1e-9:
                raise SystemExit(f"  Panel G {spec_e}: the two clusterings differ "
                                 "in the coefficient")
            _, s_c, s_e, s_i = said[si + j]
            if round(float(re_.coef), 4) != float(s_c) \
                    or round(float(re_.se), 4) != float(s_e) \
                    or round(float(ri_.se), 4) != float(s_i):
                raise SystemExit(f"  Panel G {spec_e}: the export gives "
                                 f"{float(re_.coef):+.4f} ({float(re_.se):.4f}; "
                                 f"{float(ri_.se):.4f}) and the summary {s_c} ({s_e}; "
                                 f"{s_i}); nothing is written")
            cells.append(est2(float(re_.coef), float(re_.se), float(ri_.se)))
        rows.append(f"{label}, SE by employer and by industry & & "
                    + " & ".join(cells) + r" \\")
    if firms != {104217}:
        raise SystemExit(f"  Panel G: the fits run on {firms} employers, not "
                         f"the 104,217 the title states")
    return [r" & & Quarter terms (Table 1) & Month-of-year terms \\"] + rows


def panel_h() -> list:
    """The headline design on the Eloundou classification (script 103): DAIOE
    and Eloundou on the employers both indices score, for tau at 22-25 (both
    clusterings on Eloundou), the female differential and tau at 26-30; each
    row checked against the run's summary and the agreement statistics."""
    d = rows_ok(need(S103, "eloundou_classification.csv"))
    text = need(S103, "103_summary.txt").read_text(encoding="utf-8", errors="replace")
    agree_ = pd.read_csv(need(S103, "classification_agreement.csv"))
    said = re.findall(r"(DAIOE, common employers|Eloundou, same employers)\s+tau ([-+][0-9.]+)\s+SE ([0-9.]+) by employer(?:, ([0-9.]+) by industry)?", text)
    if len(said) != 6:
        raise SystemExit(f"  103_summary.txt: expected six estimate lines, found {len(said)}")
    rows, firms = [], {}
    for k, (label, band, term, spec_d, spec_e, spec_ei) in enumerate((
            ("$\\tau$ at 22--25", "22-25", "hy_tau", "daioe_common", "eloundou", "eloundou_indcl"),
            ("Female differential at 22--25", "22-25", "hyf_tau", "sex_daioe_common", "sex_eloundou", "sex_eloundou_indcl"),
            ("$\\tau$ at 26--30", "26-30", "hy_tau", "daioe_common", "eloundou", None))):
        rd = pick(d, "eloundou_classification.csv", part="D", spec=spec_d, young_band=band, term=term)
        re_ = pick(d, "eloundou_classification.csv", part="E", spec=spec_e, young_band=band, term=term)
        for what, r in (("DAIOE", rd), ("Eloundou", re_)):
            derived_se_check(r, f"Panel H {label} {what}")
        if int(rd.n_firms) != int(re_.n_firms):
            raise SystemExit(f"  Panel H {label}: DAIOE and Eloundou fits run on different employers")
        firms[band] = int(rd.n_firms)
        sd, se = said[2 * k], said[2 * k + 1]
        agree(f"Panel H {label} DAIOE", (float(rd.coef), float(rd.se)), (float(sd[1]), float(sd[2])))
        agree(f"Panel H {label} Eloundou", (float(re_.coef), float(re_.se)), (float(se[1]), float(se[2])))
        cell_d = est(float(rd.coef), float(rd.se))
        if spec_ei:
            ri = pick(d, "eloundou_classification.csv", part="E", spec=spec_ei, young_band=band, term=term)
            derived_se_check(ri, f"Panel H {label} Eloundou industry")
            if abs(float(ri.coef) - float(re_.coef)) > 1e-9 or round(float(ri.se), 4) != float(se[3]):
                raise SystemExit(f"  Panel H {label}: the industry clustering does not match the summary")
            cell_e = est2(float(re_.coef), float(re_.se), float(ri.se))
        else:
            cell_e = est(float(re_.coef), float(re_.se))
        rows.append(f"{label}, {thousands(firms[band])} employers & & {cell_d} & {cell_e} \\\\")
    top = agree_[(agree_.population == "all_scored_employers")
                 & (agree_.statistic == "share_top_daioe_also_top_eloundou")].value.iloc[0]
    top_w = agree_[(agree_.population == "all_scored_employers")
                   & (agree_.statistic == "employment_share_top_daioe_also_top_eloundou")].value.iloc[0]
    rho = agree_[(agree_.population == "all_scored_employers")
                 & (agree_.statistic == "spearman_employer_scores")].value.iloc[0]
    if round(100 * float(top), 1) != 92.5 or round(100 * float(top_w), 1) != 91.8 or round(float(rho), 2) != 0.91:
        raise SystemExit("  Panel H: the agreement statistics do not reproduce the summary's 0.925 / 0.918 / 0.91")
    return ([r" & & DAIOE, common employers & Eloundou rating \\"] + rows,
            (float(top), float(top_w), float(rho)))


def main() -> int:
    said97 = need(S97, "97_summary.txt").read_text(encoding="utf-8", errors="replace")
    a, _ = panel_a()
    b = panel_b()
    c, n_c = panel_c()
    dd, medians = panel_d(said97)
    e, q3 = panel_e(said97)
    f, sd = panel_f(said97)
    g = panel_g()
    h, (top, top_w, rho) = panel_h()
    print("  every panel reproduces its run's own summary")

    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           r"\multicolumn{4}{l}{\emph{Panel A. $\tau$ with the older reference "
           r"restricted, common sample}} \\",
           r"Reference band & 31--69 & 31--59 & 31--49 \\", r"\midrule"]
    tex += a
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel B. $\tau$ for each band against "
            r"41--49, one panel of 104{,}333 employers}} \\"]
    tex += b
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel C. Fixed birth cohorts against "
            r"those born 1956--1990, later minus interim, "
            + thousands(n_c) + r" employers}} \\"]
    tex += c
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel D. The credit test on $\tau$, "
            r"employers with a 2019 balance sheet}} \\"]
    tex += dd
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel E. The female differential at "
            r"22--25}} \\"]
    tex += e
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel F. The exposure specification at "
            r"22--25, 104{,}217 employers}} \\"]
    tex += f
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel G. Eleven month-of-year terms in place "
            r"of the three calendar-quarter terms, 104{,}217 employers}} \\"]
    tex += g
    tex += [r"\midrule",
            r"\multicolumn{4}{l}{\emph{Panel H. Employers classified by the Eloundou et al. (2024) "
            r"rating in place of DAIOE, on the employers both indices score}} \\"]
    tex += h
    note = (
        r"$\tau$ is the later-period coefficient minus the interim one from the same Poisson fit, standard errors clustered by employer unless stated. Every fit reproduces Table~1 of the paper on its own sample before any check is run. Panel~A: the young band against the stated older bands, on the employers that hold both the young band and ages 31--49. Panel~B: eight bands, seven contrasts against 41--49, in one fit. Panel~C: groups defined by birth year alone, so no one enters or leaves a group as they age; the reduction covered pay from January 2021 to March 2023 for those who had turned 18 but not 23 at the start of the year \citep{sfs202155}. Panel~D: leverage is one minus equity over total assets in the 2019 balance sheet, split at the median ("
        + f"{medians['22-25']:.3f} at 22--25, {medians['26-30']:.3f} at 26--30"
        + r"); the rows are the terms of one fit. Panel~E: the drift is the monthly trend on the young $\times$ exposed $\times$ female term with calendar-quarter terms; a third-quarter term absorbs a seasonal dip ("
        + f"${q3[0]:+.3f}$, SE {q3[1]:.3f}"
        + r"). Panel~F: the score's baseline standard deviation is "
        + f"{sd:.1f}" + r" percentile points. Panel~G: the month-of-year terms, December omitted, are interacted with exposure and the young band (and with sex on the sex panel), everything else as in Equation~(2); the two columns come from separate fits, so the difference between them carries no standard error of its own. Panel~H: employers re-scored through the same chain (2019 codes of incumbents aged 31--69, three-digit book, incumbent floor, employment-weighted quartile cut) with the Eloundou rating as the occupation score, each index keeping its own top quartile; "
        + f"{100 * top:.1f} per cent of DAIOE's top-quartile employers ({100 * top_w:.1f} per cent by incumbent employment) are in the Eloundou top quartile, and the employer scores correlate at {rho:.2f} (Spearman); the Eloundou cells give the employer-clustered standard error and, at 22--25, the industry-clustered one; separate fits, so the differences carry no standard error.")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_final_checks.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

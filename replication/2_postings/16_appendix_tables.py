#!/usr/bin/env python3
"""
16_appendix_tables.py: the appendix tables of the posting margin and of the
within-employer posting design.

WHAT IT BUILDS
Every table is generated from the CSV its source script wrote, so nothing is
retyped. Coefficients print as estimate with stars at ten, five and one per
cent and the standard error in parentheses.

  coverage_by_source.tex     the valid-code share by year and source channel,
                             2020 to 2025, from postings_coverage_monthly.csv
                             (01); channels with fewer than 100 advertisements
                             in a year are omitted. 17 rewrites it to June 2026
  postings_extended.tex      Equation (1) on the windows to December 2025 and
                             to June 2026, OLS and Poisson (03)
  postings_seasonality.tex   the seasonal variants (06)
  firm_within_did.tex        the within-employer posting design (14)
  firm_heterogeneity.tex     the same by industry, employer age and size (15)
  postings_deciles.tex       the decile gradient (04)

INPUTS   output/results/*.csv
OUTPUTS  output/tables/ (the six files above)
SERVES   Online Appendix Tables A5 (through 17), A6, A7, A8, A30, A31
RUNTIME  seconds
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SRC = config.RESULTS
DEST = config.TABLES


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coef(c: float, se: float, p: float, d: int = 3) -> str:
    return f"{c:.{d}f}$^{{{stars(p)}}}$ ({se:.{d}f})" if stars(p) \
        else f"{c:.{d}f} ({se:.{d}f})"


def write(name: str, body: str):
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(body)
    print(f"  wrote {name}")


def t_coverage():
    """Monthly valid-SSYK share, collapsed to year x source, since 72 months
    by five sources is unreadable."""
    # 17 writes this table on the window to June 2026, after checking that
    # its 2020 to 2025 rows print exactly as they print here.
    if (SRC / "postings_coverage_monthly_extended.csv").exists():
        print("  coverage_by_source.tex: left to 17 (window to June 2026)")
        return
    d = pd.read_csv(SRC / "postings_coverage_monthly.csv")
    d["year"] = d["year_month"].str[:4]
    # Three ads carry dates outside the window (one 2051, two 2099) and 117
    # sit in 2019, before the sample opens; they are the "out of range" rows
    # of the accounting table and would print as spurious years here.
    d = d[d["year"].between("2020", "2025")]
    LABELS = {"(none)": "No source", "VIA_AIS": "AIS",
              "VIA_ANNONSERA": "Annonsera", "VIA_JOBPOSTING": "JobPosting",
              "VIA_PLATSBANKEN_DXA": "Platsbanken DXA"}
    d["source_type"] = d["source_type"].map(LABELS).fillna(d["source_type"])
    g = (d.groupby(["year", "source_type"])
         .agg(n_ads=("n_ads", "sum"), n_valid=("n_valid_code", "sum"))
         .reset_index())
    g = g[g["n_ads"] >= 100]
    g["share"] = 100 * g["n_valid"] / g["n_ads"]
    piv = g.pivot(index="year", columns="source_type", values="share")
    tot = (d.groupby("year").agg(n=("n_ads", "sum"), v=("n_valid_code", "sum")))
    piv["All sources"] = 100 * tot["v"] / tot["n"]
    cols = list(piv.columns)
    head = " & ".join(str(c) for c in cols)
    rows = [f"{y} & " + " & ".join(
        "--" if pd.isna(piv.loc[y, c]) else f"{piv.loc[y, c]:.1f}" for c in cols)
        + r" \\" for y in piv.index]
    write("coverage_by_source.tex", "\n".join([
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline\hline",
        "Year & " + head + r" \\", r"\hline", *rows, r"\hline\hline",
        r"\end{tabular}"]))


def t_extended():
    """The window to December 2025 against the window to June 2026, both
    estimators."""
    d = pd.read_csv(SRC / "postings_extended_did.csv")
    # Both rows start in January 2020; the first stops at the end of the last
    # annual archive, the second adds the two closed quarters of 2026.
    lab = {"submitted_to_2025-12": "January 2020 to December 2025",
           "extended_to_2026-06": "January 2020 to June 2026"}
    est = {"OLS_ln": r"OLS, $\ln(\text{ads})$", "Poisson": "Poisson PML"}
    rows = []
    for w in ["submitted_to_2025-12", "extended_to_2026-06"]:
        for e in ["OLS_ln", "Poisson"]:
            s = d[(d["window"] == w) & (d["estimator"] == e)]
            rb = s[s["term"] == "rb_x_high"].iloc[0]
            gp = s[s["term"] == "gpt_x_high"].iloc[0]
            rows.append(f"{lab[w]} & {est[e]} & "
                        f"{coef(rb.coef, rb.se, rb.pval)} & "
                        f"{coef(gp.coef, gp.se, gp.pval)} & "
                        f"{int(rb.n_obs):,}" + r" \\")
    write("postings_extended.tex", "\n".join([
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llccr}", r"\hline\hline",
        r"Window & Estimator & PostRB $\times$ High & PostGPT $\times$ High & $N$ \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}"]))


def t_seasonality():
    d = pd.read_csv(SRC / "postings_seasonality.csv")
    lab = {"S0_baseline": "Baseline (occupation and month FE)",
           "S1_groupseason": r"$+$ one-digit occupation group $\times$ calendar month",
           "S2_groupmonth": r"$+$ one-digit occupation group $\times$ month of sample",
           "S1_poisson": "Poisson, with group $\\times$ calendar month"}
    rows = []
    for k, v in lab.items():
        s = d[d["spec"] == k]
        if s.empty:
            continue
        rb = s[s["term"] == "rb_x_high"].iloc[0]
        gp = s[s["term"] == "gpt_x_high"].iloc[0]
        rows.append(f"{v} & {coef(rb.coef, rb.se, rb.pval)} & "
                    f"{coef(gp.coef, gp.se, gp.pval)}" + r" \\")
    write("postings_seasonality.tex", "\n".join([
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcc}", r"\hline\hline",
        r"Specification & PostRB $\times$ High & PostGPT $\times$ High \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}"]))


def t_firm_design():
    """The within-employer posting design and its heterogeneity."""
    d = pd.read_csv(SRC / "firm_within_did.csv")
    lab = {"a_all_firms": "All identifying employers",
           "b_excl_staffing_public": "Excluding staffing agencies and public employers",
           "c_entry_level_ads": "Entry-level advertisements only"}
    rows = []
    for k, v in lab.items():
        s = d[d["variant"] == k]
        rb = s[s["term"] == "rb_x_high"].iloc[0]
        gp = s[s["term"] == "gpt_x_high"].iloc[0]
        rows.append(f"{v} & {coef(rb.coef, rb.se, rb.pval)} & "
                    f"{coef(gp.coef, gp.se, gp.pval)} & "
                    f"{int(rb.n_firms):,} & {int(rb.n_obs):,}" + r" \\")
    write("firm_within_did.tex", "\n".join([
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccrr}", r"\hline\hline",
        r"Sample & PostRB $\times$ High & PostGPT $\times$ High & Employers & $N$ \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}"]))

    h = pd.read_csv(SRC / "firm_heterogeneity.csv")
    h = h[h["term"] == "gpt_x_high"].sort_values(["dimension", "coef"])
    # Raw labels carry underscores, which TeX reads as maths subscripts.
    DIM = {"section": "Industry", "firm_age": "Employer age",
           "size_proxy": "Employer size"}
    LEV = {"young_lt10y": "under 10 years", "older_10y+": "10 years or more",
           "small": "small", "mid": "medium", "large": "large",
           "C manufacturing": "C manufacturing", "G trade": "G trade",
           "J ICT": "J information and communication", "K finance": "K finance",
           "M professional": "M professional services",
           "N admin-support": "N administrative and support",
           "other-private": "other private", "public": "public"}
    rows = [f"{DIM.get(r.dimension, r.dimension)} & "
            f"{LEV.get(r.level, r.level)} & {coef(r.coef, r.se, r.pval)} "
            f"& {int(r.n_firms):,}" + r" \\" for r in h.itertuples()]
    write("firm_heterogeneity.tex", "\n".join([
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llcr}", r"\hline\hline",
        r"Dimension & Cell & PostGPT $\times$ High & Employers \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}"]))


def t_deciles():
    d = pd.read_csv(SRC / "postings_decile_gradient.csv")
    rows = []
    for dec in sorted(d["decile"].unique()):
        s = d[d["decile"] == dec]
        rb = s[s["period"] == "post_riksbank"].iloc[0]
        gp = s[s["period"] == "post_chatgpt"].iloc[0]
        rows.append(f"{int(dec)} & {coef(rb.coef, rb.se, rb.pval)} & "
                    f"{coef(gp.coef, gp.se, gp.pval)}" + r" \\")
    write("postings_deciles.tex", "\n".join([
        r"\begin{tabular}{lcc}", r"\hline\hline",
        r"Exposure decile & PostRB $\times$ decile & PostGPT $\times$ decile \\",
        r"\hline", *rows, r"\hline\hline", r"\end{tabular}",
        r"% Decile 5, the median, is the omitted reference."]))


if __name__ == "__main__":
    print(f"Writing appendix tables to {DEST}")
    t_coverage(); t_extended(); t_seasonality()
    t_firm_design(); t_deciles()
    print("done")

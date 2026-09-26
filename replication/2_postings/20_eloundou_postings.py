#!/usr/bin/env python3
"""
20_eloundou_postings.py: Equation (1) with the GPT-exposure score of
Eloundou et al. (2024) in place of DAIOE, on the window to June 2026, and
on the common occupation sample (Online Appendix Table A9).

WHAT IT ESTIMATES
ln(postings), zero cells dropped, occupation and month effects, PostRB x
High and PostGPT x High, standard errors clustered by occupation. High is
the top quartile of each measure as the submitted version defined it: DAIOE
Q4 from daioe_quartiles.csv; the Eloundou beta score above its 75th
percentile across the SSYK codes the crosswalk scores (unweighted). The
crosswalked Eloundou score is read from 3_register_mona/inputs/
eloundou_ssyk4.dta, the file the register scripts also read, checked against
its pinned SHA-256; it was built from the authors' occ_level.csv through
the SOC 2010 -> ISCO-08 -> SSYK 2012 route of 12. The quartile assignment
is NOT recomputed on the common sample, so the common-sample columns differ
from the full-sample columns only by which occupations enter. Three
columns per measure: Equation (1); a linear time trend interacted with
High; month effects replaced by SSYK one-digit group x month effects.

CHECK
The DAIOE full-sample baseline must reproduce the OLS baseline of 03
(postings_extended_did.csv) to 1e-8 in both coefficients and standard errors.

INPUTS   data/processed/postings_daioe_merged_extended.csv (03);
         3_register_mona/inputs/eloundou_ssyk4.dta;
         output/results/postings_extended_did.csv (03)
OUTPUTS  output/results/eloundou_postings_extended.csv;
         output/tables/tableA_eloundou_postings.tex
SERVES   Online Appendix II.8 (Table A9; the 341 occupations, -0.135 and
         -0.094 of the text) and Section 3 of the paper
RUNTIME  about 30 seconds
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfixest as pf  # noqa: E402

PROCESSED, RESULTS, TABLES = config.PROCESSED, config.RESULTS, config.TABLES
ELOUNDOU = config.PACKAGE / "3_register_mona" / "inputs" / "eloundou_ssyk4.dta"
ELOUNDOU_SHA256 = "d47b771e2f75a6166f3a855a3ba181e51a8510d8c9d07c82ebceb9e2f68eda93"
RB = pd.Timestamp(config.RIKSBANKEN_HIKE)
GPT = pd.Timestamp(config.CHATGPT_LAUNCH)


def eloundou_classification() -> pd.DataFrame:
    """The crosswalked Eloundou score and its top-quartile flag, from the
    pinned input file."""
    digest = hashlib.sha256(ELOUNDOU.read_bytes()).hexdigest()
    if digest != ELOUNDOU_SHA256:
        raise SystemExit(f"  {ELOUNDOU.name} has SHA-256 {digest}, not the pinned "
                         f"{ELOUNDOU_SHA256}")
    e = pd.read_stata(ELOUNDOU)
    e["ssyk4"] = e["ssyk4"].astype(int).astype(str).str.zfill(4)
    # The flag is the score above its 75th percentile across the scored
    # codes, each counting once; recomputed here as a check on the file.
    q75 = e["eloundou_score"].quantile(0.75)
    if not ((e["eloundou_score"] > q75).astype(int) == e["high_exposure_eloundou"]).all():
        raise SystemExit("  eloundou_ssyk4.dta: the high-exposure flag is not the top "
                         "quartile of the score")
    return e[["ssyk4", "eloundou_score", "high_exposure_eloundou"]]


def build(panel: pd.DataFrame, el: pd.DataFrame) -> pd.DataFrame:
    """Occupation-month panel carrying both High dummies (NaN where unscored)."""
    df = panel[["ssyk4", "year_month", "n_ads", "high_exposure"]].copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df = df.merge(el, on="ssyk4", how="left")
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])
    # The trend is measured from the first month of the panel, in months.
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days / 30.0
    df["group_time"] = df["ssyk4"].str[0] + "_" + df["year_month"]
    df["high_daioe"] = df["high_exposure"].astype(float)
    df["high_eloundou"] = df["high_exposure_eloundou"]
    return df


def fit(df: pd.DataFrame, high: str, spec: str) -> dict:
    d = df[df[high].notna()].copy()
    d["rb"] = ((d["date"] >= RB) & (d[high] == 1)).astype(int)
    d["gpt"] = ((d["date"] >= GPT) & (d[high] == 1)).astype(int)
    d["trend"] = d["time_idx"] * d[high]
    rhs = "rb + gpt" + (" + trend" if spec == "trend" else "")
    fe = "ssyk4 + group_time" if spec == "grp" else "ssyk4 + year_month"
    m = pf.feols(f"ln_ads ~ {rhs} | {fe}", data=d, vcov={"CRV1": "ssyk4"},
                 fixef_rm="none")  # keep singletons, as the submitted version did
    c, s, p = m.coef(), m.se(), m.pvalue()
    return dict(rb=c["rb"], se_rb=s["rb"], p_rb=p["rb"], gpt=c["gpt"], se_gpt=s["gpt"],
                p_gpt=p["gpt"], n_obs=m._N, n_occ=d.ssyk4.nunique(),
                n_high_occ=d.loc[d[high] == 1, "ssyk4"].nunique())


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(v, p):
    s = f"{v:.3f}".replace("-", "$-$")
    return s + (f"$^{{{stars(p)}}}$" if stars(p) else "")


def write_tex(res: pd.DataFrame):
    cols = [("full", "DAIOE"), ("common", "DAIOE"), ("full", "Eloundou")]
    specs = ["base", "trend", "grp"]

    def get(s, m, sp):
        return res[(res["sample"] == s) & (res.measure == m) & (res.spec == sp)].iloc[0]

    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Postings with DAIOE and \citet{eloundou2024gpts} exposure, January 2020 to June 2026}",
         r"\label{tab:eloundou_postings}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{l" + "c" * 9 + "}", r"\hline\hline",
         r" & \multicolumn{3}{c}{DAIOE, all occupations} & \multicolumn{3}{c}{DAIOE, common sample}"
         r" & \multicolumn{3}{c}{Eloundou $\beta$, common sample} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
         " & " + " & ".join(f"({i})" for i in range(1, 10)) + r" \\", r"\hline"]
    for term, lab in (("rb", r"Post-RB $\times$ High"), ("gpt", r"Post-GPT $\times$ High")):
        L.append(lab + "".join(" & " + cell(get(s, m, sp)[term], get(s, m, sp)["p_" + term])
                               for s, m in cols for sp in specs) + r" \\")
        L.append("".join(f" & ({get(s, m, sp)['se_' + term]:.3f})" for s, m in cols for sp in specs)
                 + (r" \\[3pt]" if term == "rb" else r" \\"))
    L += [r"\hline",
          r"Linear trend $\times$ High" + " & & Yes & " * 3 + r"\\",
          r"Group $\times$ month FE" + " & & & Yes" * 3 + r" \\"]
    L.append("Occupations" + "".join(f" & \\multicolumn{{3}}{{c}}{{{get(s, m, 'base')['n_occ']}}}"
                                     for s, m in cols) + r" \\")
    L.append("Observations" + "".join(f" & \\multicolumn{{3}}{{c}}{{{get(s, m, 'base')['n_obs']:,}}}"
                                      for s, m in cols) + r" \\")
    L += [r"\hline\hline", r"\end{tabular}}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize \textit{Notes:} Equation~(1) on ln postings, "
          r"occupation-by-month cells with at least one advertisement. High is the top quartile of the "
          r"respective measure across all occupations it scores; it is not re-cut on the common sample. The "
          r"common sample keeps the 341 occupations observed in the posting data and scored by both measures. "
          r"Columns (2), (5) and (8) add a linear time trend interacted with High; columns (3), (6) and (9) "
          r"replace month fixed effects with SSYK 1-digit group $\times$ month fixed effects. Standard errors "
          r"clustered by occupation. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}",
          r"\end{table}"]
    out = TABLES / "tableA_eloundou_postings.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out.name}")


def main():
    print("Equation (1) with the Eloundou et al. (2024) score, January 2020 to June 2026")
    el = eloundou_classification()
    ext = build(pd.read_csv(PROCESSED / "postings_daioe_merged_extended.csv"), el)
    common = ext[ext.high_eloundou.notna()]
    rows = []
    for sample, df in (("full", ext), ("common", common)):
        for measure, high in (("DAIOE", "high_daioe"), ("Eloundou", "high_eloundou")):
            for spec in ("base", "trend", "grp"):
                r = fit(df, high, spec)
                rows.append(dict(sample=sample, measure=measure, spec=spec, **r))
    res = pd.DataFrame(rows)
    # The Eloundou 'full' and 'common' samples coincide by construction.
    l08 = pd.read_csv(RESULTS / "postings_extended_did.csv").set_index(["window", "estimator", "term"])
    b = res[(res["sample"] == "full") & (res.measure == "DAIOE") & (res.spec == "base")].iloc[0]
    for t, k in (("rb_x_high", "rb"), ("gpt_x_high", "gpt")):
        ref = l08.loc[("extended_to_2026-06", "OLS_ln", t)]
        if abs(b[k] - ref["coef"]) > 1e-8 or abs(b["se_" + k] - ref["se"]) > 1e-8:
            raise SystemExit(f"  the DAIOE baseline {t} {b[k]:.6f} ({b['se_' + k]:.6f}) is not "
                             f"03's {ref['coef']:.6f} ({ref['se']:.6f})")
    print(f"  CHECK: DAIOE full-sample baseline = 03's ({b['rb']:.4f}, {b['gpt']:.4f})")
    n_common = int(res[(res["sample"] == "common") & (res.spec == "base")].n_occ.iloc[0])
    if n_common != 341:
        raise SystemExit(f"  the common sample holds {n_common} occupations; the note says 341")
    res.insert(0, "panel", f"{ext.year_month.min()} to {ext.year_month.max()}")
    out = RESULTS / "eloundou_postings_extended.csv"
    res.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    print(res.round(4).to_string())
    write_tex(res)


if __name__ == "__main__":
    main()

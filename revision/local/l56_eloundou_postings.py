#!/usr/bin/env python3
"""
l56_eloundou_postings.py: the Eloundou et al. (2024) posting check of the
submitted Online Appendix, restored on the paper's current window.

WHY THIS EXISTS
The submitted OA (appendix_v1.tex, section sec:posting_eloundou, table
tab:eloundou_robust, built by src/10_eloundou_robustness.py) re-estimated
Equation (1) with the Eloundou beta exposure instead of DAIOE. On the
submitted panel the Eloundou post-ChatGPT coefficient was -0.096 (p 0.025,
341 occupations) while DAIOE loaded the decline on the rate-hike period,
and both attenuated with occupation-specific trends or SSYK 1-digit group
x month effects. Every other posting estimate now runs on the extended
panel built by l08 (January 2020 to June 2026). A second external review
asked for the check on that window, and on a COMMON occupation sample so
that the two indices differ only in the ranking.

WHAT IT ESTIMATES (unchanged from src/10)
ln(postings), zero cells dropped, occupation and month effects,
PostRB x High and PostGPT x High, standard errors clustered by occupation.
High is the top quartile of each measure as the submitted version defined
it: DAIOE Q4 from daioe_quartiles.csv; Eloundou beta above its 75th
percentile across the SSYK codes the crosswalk scores (src/10's
assign_eloundou_quartiles, unweighted). The quartile assignment is NOT
recomputed on the common sample, so the common-sample rows differ from the
full-sample rows only by which occupations enter.
Three columns per measure, as in the submitted table:
  base    Equation (1);
  trend   + a linear time trend interacted with High (src/10 spec 3);
  grp     month effects replaced by SSYK 1-digit group x month effects
          (src/10 spec 4).

GATE
On the submitted panel (postings_daioe_merged.csv, October 2019 to February 2026, 26,672 cells) the
estimator must reproduce tables/eloundou_robustness.csv, coefficients to
1e-8 for both measures and all three specifications; then on the extended
panel the DAIOE full-sample baseline must reproduce l08's OLS_ln row
(-0.1271, -0.0593) to 1e-8.

Run:  python3 revision/local/l56_eloundou_postings.py
Out:  revision/tables/l56_eloundou_postings.csv
      ../canaries-sweden-paper/tables/tableA_eloundou_postings.tex
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "revision" / "tables" / "l56_eloundou_postings.csv"
PAPER_TAB = ROOT.parent / "canaries-sweden-paper" / "tables" / "tableA_eloundou_postings.tex"
OLD = ROOT / "tables" / "eloundou_robustness.csv"
L08 = ROOT / "revision" / "tables" / "postings_extended_did.csv"

sys.path.insert(0, str(SRC))
import config  # noqa: E402

_spec = importlib.util.spec_from_file_location("el10", SRC / "10_eloundou_robustness.py")
el10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(el10)

RB = pd.Timestamp(config.RIKSBANKEN_HIKE)
GPT = pd.Timestamp(config.CHATGPT_LAUNCH)


def eloundou_classification() -> pd.DataFrame:
    """Rebuild src/10's SSYK mapping and check it equals the saved file."""
    scores = el10.assign_eloundou_quartiles(el10.build_eloundou_ssyk(el10.download_eloundou()))
    saved = pd.read_csv(PROCESSED / "eloundou_ssyk_matched.csv", dtype={"ssyk4": str})
    saved["ssyk4"] = saved["ssyk4"].str.zfill(4)
    chk = scores.merge(saved, on="ssyk4", suffixes=("", "_saved"))
    assert len(chk) == len(saved) == len(scores)
    assert (chk.high_exposure_eloundou == chk.high_exposure_eloundou_saved).all()
    return scores[["ssyk4", "dv_rating_beta", "high_exposure_eloundou"]]


def build(panel: pd.DataFrame, el: pd.DataFrame) -> pd.DataFrame:
    """Occupation-month panel carrying both High dummies (NaN where unscored)."""
    df = panel[["ssyk4", "year_month", "n_ads", "high_exposure"]].copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df = df.merge(el, on="ssyk4", how="left")
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])
    # src/10 measures the trend from the first month of the panel it is given.
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
                  fixef_rm="none")  # keep singletons, as PanelOLS in src/10 did
    c, s, p = m.coef(), m.se(), m.pvalue()
    return dict(rb=c["rb"], se_rb=s["rb"], p_rb=p["rb"], gpt=c["gpt"], se_gpt=s["gpt"],
                p_gpt=p["gpt"], n_obs=m._N, n_occ=d.ssyk4.nunique(),
                n_high_occ=d.loc[d[high] == 1, "ssyk4"].nunique())


def main():
    el = eloundou_classification()

    # ---- Gate 1: submitted panel reproduces the submitted table --------
    old = pd.read_csv(OLD)
    # The submitted panel itself (October 2019 to February 2026); config.load_postings_merged
    # now points at the extended file.
    sub = build(pd.read_csv(PROCESSED / "postings_daioe_merged.csv"), el)
    for measure, high in (("DAIOE", "high_daioe"), ("Eloundou", "high_eloundou")):
        for spec, key in (("base", "spec2"), ("trend", "spec3"), ("grp", "spec4")):
            r = fit(sub, high, spec)
            o = old[(old.measure == measure) & (old.specification == key)].set_index("variable")
            assert abs(r["rb"] - o.loc["post_rb_x_high", "coefficient"]) < 1e-8, (measure, spec, r)
            assert abs(r["gpt"] - o.loc["post_gpt_x_high", "coefficient"]) < 1e-8, (measure, spec, r)
            assert r["n_obs"] == o["n_obs"].iloc[0]
            print(f"  gate {measure:8s} {spec:5s}: rb {r['rb']:+.4f} gpt {r['gpt']:+.4f} "
                  f"SE ratio to PanelOLS {r['se_gpt'] / o.loc['post_gpt_x_high', 'std_error']:.4f}")
    print("GATE 1 PASSED: submitted panel reproduces tables/eloundou_robustness.csv")

    # ---- Current window --------------------------------------------------
    ext = build(pd.read_csv(PROCESSED / "postings_daioe_merged_extended.csv"), el)
    common = ext[ext.high_eloundou.notna()]
    rows = []
    for sample, df in (("full", ext), ("common", common)):
        for measure, high in (("DAIOE", "high_daioe"), ("Eloundou", "high_eloundou")):
            for spec in ("base", "trend", "grp"):
                r = fit(df, high, spec)
                rows.append(dict(sample=sample, measure=measure, spec=spec, **r))
    res = pd.DataFrame(rows)
    # Eloundou 'full' and 'common' are the same sample by construction.
    l08 = pd.read_csv(L08).set_index(["window", "estimator", "term"])
    b = res[(res["sample"] == "full") & (res.measure == "DAIOE") & (res.spec == "base")].iloc[0]
    for t, k in (("rb_x_high", "rb"), ("gpt_x_high", "gpt")):
        ref = l08.loc[("extended_to_2026-06", "OLS_ln", t)]
        assert abs(b[k] - ref["coef"]) < 1e-8 and abs(b["se_" + k] - ref["se"]) < 1e-8, (t, b[k], ref)
    print(f"GATE 2 PASSED: DAIOE full baseline = l08 ({b['rb']:.4f}, {b['gpt']:.4f})")

    res.insert(0, "panel", f"{ext.year_month.min()} to {ext.year_month.max()}")
    res.to_csv(OUT, index=False)
    print(res.round(4).to_string())
    write_tex(res)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(v, p):
    s = f"{v:.3f}".replace("-", "$-$")
    return s + (f"$^{{{stars(p)}}}$" if stars(p) else "")


def write_tex(res: pd.DataFrame):
    cols = [("full", "DAIOE"), ("common", "DAIOE"), ("full", "Eloundou")]
    specs = ["base", "trend", "grp"]
    get = lambda s, m, sp: res[(res["sample"] == s) & (res.measure == m) & (res.spec == sp)].iloc[0]
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
          r"common sample keeps the 341 occupations that both measures score, which is every occupation the "
          r"Eloundou crosswalk reaches, so columns (7) to (9) are also its full sample. "
          r"Columns (2), (5) and (8) add a linear time trend interacted with High; columns (3), (6) and (9) "
          r"replace month fixed effects with SSYK 1-digit group $\times$ month fixed effects. Standard errors "
          r"clustered by occupation. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}",
          r"\end{table}"]
    PAPER_TAB.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {PAPER_TAB}")


if __name__ == "__main__":
    main()

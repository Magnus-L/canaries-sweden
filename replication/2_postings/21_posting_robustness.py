#!/usr/bin/env python3
"""
21_posting_robustness.py: the posting robustness table on the window to June
2026 (Online Appendix Table A8).

WHAT IT ESTIMATES
Equation (1), ln(postings) on occupation and month effects, PostRB x High
and PostGPT x High, High the top DAIOE quartile, standard errors clustered
by occupation, under six variations of the construction, each on the
extended panel of 03:
  baseline;
  vacancies    the outcome is ln(positions advertised), the sum over
               advertisements of the stated number of vacancies (missing or
               below one set to one, the rule of 1_data_public/02); it does
               not reweight. For 2026 the field is not in 03's counts, so
               the two closed-quarter archives are streamed again with 01's
               classification and 03's deduplication, and the advertisement
               counts are checked against 03's cached 2026 aggregates cell
               by cell;
  no pandemic  January to June 2020 dropped;
  terciles     High is the score above its 66.7th percentile over the
               panel's cells;
  excl. ICT    SSYK 25xx dropped;
  balanced     occupations with a positive count in every month of the
               window (78 months).
Two rows are carried from existing results on the same panel and months,
not re-estimated: Poisson on the counts (03, postings_extended_did.csv) and
one-digit group by month-of-year effects (06, postings_seasonality.csv).

CHECK
The baseline row must reproduce the OLS baseline of 03 to 1e-8.

INPUTS   data/processed/postings_daioe_merged_extended.csv (03);
         data/raw/postings_ssyk4_monthly_2026-02-24.csv (the frozen counts,
         which carry the positions advertised); config.JOBADS_DIR/2026-Q1
         and 2026-Q2 archives; output/results/postings_ssyk4_monthly_2026H1.csv,
         postings_extended_did.csv (03), postings_seasonality.csv (06)
OUTPUTS  output/results/posting_robustness_extended.csv,
         postings_ssyk4_monthly_2026H1_vacancies.csv (the 2026 positions,
         cached; delete it to re-stream); output/tables/tableA_posting_robustness.tex
SERVES   Online Appendix II.6, Table A8
RUNTIME  about 2 minutes (the two 2026 archives are streamed once)
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfixest as pf  # noqa: E402

PROCESSED, RESULTS, TABLES = config.PROCESSED, config.RESULTS, config.TABLES
VAC_CACHE = RESULTS / "postings_ssyk4_monthly_2026H1_vacancies.csv"
RB, GPT = pd.Timestamp(config.RIKSBANKEN_HIKE), pd.Timestamp(config.CHATGPT_LAUNCH)
acc = sibling("01_postings_accounting")


def vacancies_2026() -> pd.DataFrame:
    """2026 H1 advertisements and positions by occupation-month, on 03's
    kept sample and deduplication (one identifier set across the two
    quarter archives)."""
    if VAC_CACHE.exists():
        return pd.read_csv(VAC_CACHE, dtype={"ssyk4": str})
    seen, counts = set(), {}
    for stem in config.PLATSBANKEN_QUARTERS:
        zpath = config.platsbanken_zip(stem)
        if not zpath.exists():
            raise SystemExit(f"  archive missing: {zpath} (run 1_data_public/01)")
        print(f"  streaming {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
                with zf.open(name) as f:
                    for line in f:
                        try:
                            ad = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        reason, rec = acc.classify_ad(ad)
                        if reason != "ok":
                            continue
                        if rec["ad_id"] and rec["ad_id"] in seen:
                            continue
                        if rec["ad_id"]:
                            seen.add(rec["ad_id"])
                        nv = ad.get("number_of_vacancies")
                        nv = 1 if (nv is None or nv < 1) else int(nv)  # 02's rule
                        k = (rec["ssyk4"], rec["year_month"])
                        c = counts.setdefault(k, [0, 0])
                        c[0] += 1
                        c[1] += nv
    df = pd.DataFrame([{"ssyk4": k[0], "year_month": k[1], "n_ads": v[0], "n_vacancies": v[1]}
                       for k, v in counts.items()])
    df = df[(df.year_month >= "2026-01") & (df.year_month <= "2026-06")]
    df.to_csv(VAC_CACHE, index=False)
    return df


def extended_with_vacancies() -> pd.DataFrame:
    ext = pd.read_csv(PROCESSED / "postings_daioe_merged_extended.csv", dtype={"ssyk4": str})
    ext["ssyk4"] = ext["ssyk4"].str.zfill(4)
    v26 = vacancies_2026()
    v26["ssyk4"] = v26["ssyk4"].str.zfill(4)
    a26 = pd.read_csv(RESULTS / "postings_ssyk4_monthly_2026H1.csv", dtype={"ssyk4": str})
    a26["ssyk4"] = a26["ssyk4"].str.zfill(4)
    chk = a26.merge(v26, on=["ssyk4", "year_month"], how="outer", suffixes=("_03", ""))
    if chk.n_ads.isna().any() or chk.n_ads_03.isna().any() or (chk.n_ads != chk.n_ads_03).any():
        raise SystemExit("  the re-streamed 2026 advertisement counts do not equal 03's")
    print(f"  CHECK: the 2026 re-stream equals 03's aggregates ({len(chk):,} cells, "
          f"{int(chk.n_ads.sum()):,} advertisements, {int(chk.n_vacancies.sum()):,} positions)")
    base = pd.read_csv(config.POSTINGS_SSYK4, dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base.year_month >= "2020-01") & (base.year_month <= "2025-12")]
    vac = pd.concat([base[["ssyk4", "year_month", "n_ads", "n_vacancies"]],
                     v26[["ssyk4", "year_month", "n_ads", "n_vacancies"]]])
    out = ext.merge(vac, on=["ssyk4", "year_month"], how="left", suffixes=("", "_v"))
    if out.n_vacancies.isna().any() or (out.n_ads != out.n_ads_v).any():
        raise SystemExit("  the positions advertised do not line up with the panel's counts")
    return out.drop(columns="n_ads_v")


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["high_exposure"] = df["high_exposure"].astype(int)
    return df


def fit(df, high="high_exposure", outcome="n_ads") -> dict:
    d = df[df[outcome] > 0].copy()
    d["y"] = np.log(d[outcome])
    d["rb"] = ((d.date >= RB) & (d[high] == 1)).astype(int)
    d["gpt"] = ((d.date >= GPT) & (d[high] == 1)).astype(int)
    m = pf.feols("y ~ rb + gpt | ssyk4 + year_month", data=d, vcov={"CRV1": "ssyk4"},
                 fixef_rm="none")
    c, s, p = m.coef(), m.se(), m.pvalue()
    return dict(rb=c["rb"], se_rb=s["rb"], p_rb=p["rb"], gpt=c["gpt"], se_gpt=s["gpt"],
                p_gpt=p["gpt"], n_obs=m._N, n_occ=d.ssyk4.nunique(),
                n_high_occ=d.loc[d[high] == 1, "ssyk4"].nunique())


def variants(df: pd.DataFrame, balanced_window: tuple) -> dict:
    out = {"Baseline": fit(df),
           "Vacancies (positions advertised)": fit(df, outcome="n_vacancies"),
           "Excluding January to June 2020": fit(df[df.year_month >= "2020-07"])}
    t = df.copy()
    t["high_tercile"] = (t["pctl_rank_genai"] > t["pctl_rank_genai"].quantile(0.667)).astype(int)
    out["Top tercile"] = fit(t, high="high_tercile")
    out["Excluding ICT occupations (SSYK 25)"] = fit(df[~df.ssyk4.str.startswith("25")])
    b = df[(df.year_month >= balanced_window[0]) & (df.year_month <= balanced_window[1])]
    nm = b.year_month.nunique()
    keep = b[b.n_ads > 0].groupby("ssyk4").year_month.nunique()
    out["Balanced panel"] = fit(b[b.ssyk4.isin(keep[keep == nm].index)])
    return out


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(v, p):
    s = f"{v:.3f}".replace("-", "$-$")
    return s + (f"$^{{{stars(p)}}}$" if stars(p) else "")


def write_tex(out: pd.DataFrame):
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Posting robustness, January 2020 to June 2026}",
         r"\label{tab:posting_robustness}", r"\footnotesize",
         r"\resizebox{\linewidth}{!}{\begin{tabular}{lccrr}", r"\hline\hline",
         r" & Post-RB $\times$ High & Post-GPT $\times$ High & Occupations & Observations \\",
         r"\hline"]
    for i, r in out.iterrows():
        L.append(f"({i + 1}) {r['row']} & {cell(r.rb, r.p_rb)} & {cell(r.gpt, r.p_gpt)} & "
                 f"{int(r.n_occ)} & {int(r.n_obs):,} \\\\")
        L.append(f" & ({r.se_rb:.3f}) & ({r.se_gpt:.3f}) & & \\\\" + ("[2pt]" if i < len(out) - 1 else ""))
    L += [r"\hline\hline", r"\end{tabular}}",
          r"\par\smallskip\parbox{\linewidth}{\scriptsize \textit{Notes:} Equation~(1): ln postings on "
          r"occupation and month fixed effects, High is the top DAIOE quartile, cells with at least one "
          r"advertisement. Row (2) replaces the outcome with ln of the positions advertised (the sum of "
          r"each advertisement's stated number of vacancies); it does not reweight. Row (4) sets High to "
          r"the top tercile. Row (6) keeps occupations advertised in all 78 months. Row (7) is Poisson "
          r"pseudo-maximum likelihood on the counts; row (8) adds SSYK 1-digit group $\times$ month-of-year "
          r"fixed effects. Standard errors clustered by occupation. "
          r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}",
          r"\end{table}"]
    out_path = TABLES / "tableA_posting_robustness.tex"
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out_path.name}")


def main():
    print("Posting robustness, January 2020 to June 2026")
    ext = prep(extended_with_vacancies())
    res = variants(ext, ("2020-01", "2026-06"))
    l08 = pd.read_csv(RESULTS / "postings_extended_did.csv").set_index(["window", "estimator", "term"])
    for t, k in (("rb_x_high", "rb"), ("gpt_x_high", "gpt")):
        ref = l08.loc[("extended_to_2026-06", "OLS_ln", t), "coef"]
        if abs(res["Baseline"][k] - ref) > 1e-8:
            raise SystemExit(f"  the baseline {t} {res['Baseline'][k]:.6f} is not 03's {ref:.6f}")
    print("  CHECK: the baseline reproduces 03's OLS estimates")
    rows = [dict(row=k, source="21", **v) for k, v in res.items()]

    # Carried rows (same panel, same months).
    p = l08.xs(("extended_to_2026-06", "Poisson"), level=["window", "estimator"])
    s = pd.read_csv(RESULTS / "postings_seasonality.csv").set_index(["spec", "term"])
    for name, get, src in (
            ("Poisson on counts", lambda t: p.loc[t], "03 postings_extended_did.csv"),
            ("Group by month-of-year effects", lambda t: s.loc[("S1_groupseason", t)],
             "06 postings_seasonality.csv S1_groupseason")):
        rb, gp = get("rb_x_high"), get("gpt_x_high")
        rows.append(dict(row=name, source=src, rb=rb.coef, se_rb=rb.se, p_rb=rb.pval, gpt=gp.coef,
                         se_gpt=gp.se, p_gpt=gp.pval, n_obs=int(rb.n_obs), n_occ=369,
                         n_high_occ=res["Baseline"]["n_high_occ"]))
    out = pd.DataFrame(rows)
    out.insert(0, "panel", "2020-01 to 2026-06")
    out_path = RESULTS / "posting_robustness_extended.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved {out_path.name}")
    print(out.round(4).to_string())
    write_tex(out)


if __name__ == "__main__":
    main()

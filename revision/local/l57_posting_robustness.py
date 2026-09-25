#!/usr/bin/env python3
"""
l57_posting_robustness.py: the submitted posting robustness table
(OA tab:robustness, src/07_robustness.py) re-estimated on the paper's
current window, January 2020 to June 2026.

WHY THIS EXISTS
The submitted OA reported Equation (1) under a vacancy outcome, without
the first pandemic months, with a tercile cut, without IT occupations and
on a balanced panel, all on the submitted panel (October 2019 to February
2026). Every other posting estimate now runs on l08's extended panel. A
second external review asked for these checks on that window in one table.

DEFINITIONS (read from src/07 and matched)
  baseline     ln(ads), occupation and month effects, PostRB x High and
               PostGPT x High, High = DAIOE Q4, SEs clustered by occupation.
  vacancies    the OUTCOME changes, not the weights: ln(positions
               advertised), the sum over ads of Platsbanken's
               number_of_vacancies ("antal platser"), missing or <1 set to 1
               as in src/02. src/07 called this row "vacancy-weighted" but
               it was the same outcome swap. For 2026 the field is not in
               l08's aggregates, so the two closed-quarter files are
               re-streamed with l01.classify_ad (the kept sample and the
               ad-id deduplication of l08) and the ad counts are checked
               against l08's cached 2026 aggregates cell by cell.
  no pandemic  drop January to June 2020 (src/07: year_month >= 2020-07).
  terciles     High = pctl_rank_genai above its 66.7th percentile, the
               percentile taken over the panel's cells as src/07 did.
  excl. IT     drop SSYK 25xx (ICT professionals), as src/07.
  balanced     occupations with a positive count in every month of the
               window (src/07 used 2020-01 to 2025-12 on the submitted
               panel; here all 78 months, 2020-01 to 2026-06).
Two rows are carried from existing exports on the same panel and months,
not re-estimated: Poisson on the counts (l08, postings_extended_did.csv)
and one-digit group by calendar-month effects (l06, S1_groupseason).

GATE
On the submitted panel (postings_daioe_merged.csv, 26,672 cells) the
estimator must reproduce tables/robustness_results.csv for the six
re-estimated rows (coefficients to 1e-8, N and occupations exactly); the
current-window baseline must reproduce l08 (-0.1271, -0.0593) to 1e-8.

Run:  python3 revision/local/l57_posting_robustness.py
Out:  revision/tables/l57_posting_robustness.csv
      revision/output/postings_ssyk4_monthly_2026H1_vac.csv (cache)
      ../canaries-sweden-paper/tables/tableA_posting_robustness.tex
"""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision"
PROCESSED = ROOT / "data" / "processed"
OUT = REV / "tables" / "l57_posting_robustness.csv"
VAC_CACHE = REV / "output" / "postings_ssyk4_monthly_2026H1_vac.csv"
L08_ADS = REV / "output" / "postings_ssyk4_monthly_2026H1.csv"
L08 = REV / "tables" / "postings_extended_did.csv"
L06 = REV / "tables" / "postings_seasonality.csv"
OLD = ROOT / "tables" / "robustness_results.csv"
PAPER_TAB = ROOT.parent / "canaries-sweden-paper" / "tables" / "tableA_posting_robustness.tex"
CACHE = Path.home() / ".cache" / "aiel-jobads"

_s = importlib.util.spec_from_file_location("l01", REV / "local" / "l01_postings_accounting.py")
l01 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(l01)

RB, GPT = pd.Timestamp("2022-04-01"), pd.Timestamp("2022-12-01")


def vacancies_2026() -> pd.DataFrame:
    """2026 H1 ads and positions by occupation-month, l08's kept sample."""
    if VAC_CACHE.exists():
        return pd.read_csv(VAC_CACHE, dtype={"ssyk4": str})
    seen, counts = set(), {}
    for fname in ("2026-Q1.jsonl.zip", "2026-Q2.jsonl.zip"):
        print(f"  streaming {fname} ...")
        with zipfile.ZipFile(CACHE / fname) as zf:
            for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
                with zf.open(name) as f:
                    for line in f:
                        try:
                            ad = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        reason, rec = l01.classify_ad(ad)
                        if reason != "ok":
                            continue
                        if rec["ad_id"] and rec["ad_id"] in seen:
                            continue
                        if rec["ad_id"]:
                            seen.add(rec["ad_id"])
                        nv = ad.get("number_of_vacancies")
                        nv = 1 if (nv is None or nv < 1) else int(nv)  # src/02 rule
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
    # Check: the re-stream reproduces l08's 2026 ad counts cell by cell.
    a26 = pd.read_csv(L08_ADS, dtype={"ssyk4": str})
    a26["ssyk4"] = a26["ssyk4"].str.zfill(4)
    chk = a26.merge(v26, on=["ssyk4", "year_month"], how="outer", suffixes=("_l08", ""))
    assert chk.n_ads.notna().all() and chk.n_ads_l08.notna().all()
    assert (chk.n_ads == chk.n_ads_l08).all()
    print(f"  CHECK: 2026 re-stream equals l08's aggregates ({len(chk)} cells, "
          f"{int(chk.n_ads.sum()):,} ads, {int(chk.n_vacancies.sum()):,} positions)")
    base = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv", dtype={"ssyk4": str})
    base["ssyk4"] = base["ssyk4"].str.zfill(4)
    base = base[(base.year_month >= "2020-01") & (base.year_month <= "2025-12")]
    vac = pd.concat([base[["ssyk4", "year_month", "n_ads", "n_vacancies"]],
                     v26[["ssyk4", "year_month", "n_ads", "n_vacancies"]]])
    out = ext.merge(vac, on=["ssyk4", "year_month"], how="left", suffixes=("", "_v"))
    assert out.n_vacancies.notna().all() and (out.n_ads == out.n_ads_v).all()
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
    out = {"Baseline": fit(df), "Vacancies (positions advertised)": fit(df, outcome="n_vacancies"),
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


def main():
    # ---- Gate: submitted panel reproduces the submitted table ----------
    old = pd.read_csv(OLD).set_index("specification")
    sub = prep(pd.read_csv(PROCESSED / "postings_daioe_merged.csv"))
    names = {"Baseline": "Baseline (genAI Q4)", "Vacancies (positions advertised)": "Vacancy-weighted",
             "Excluding January to June 2020": "Excl. pandemic", "Top tercile": "Terciles",
             "Excluding ICT occupations (SSYK 25)": "Excl. IT/tech", "Balanced panel": "Balanced panel"}
    for k, r in variants(sub, ("2020-01", "2025-12")).items():
        o = old.loc[names[k]]
        assert abs(r["rb"] - o.beta_rb) < 1e-8 and abs(r["gpt"] - o.beta_gpt) < 1e-8, (k, r, o)
        assert r["n_obs"] == o.n_obs and r["n_occ"] == o.n_entities, (k, r, o)
    print("GATE PASSED: submitted panel reproduces tables/robustness_results.csv (six rows)")

    # ---- Current window --------------------------------------------------
    ext = prep(extended_with_vacancies())
    res = variants(ext, ("2020-01", "2026-06"))
    l08 = pd.read_csv(L08).set_index(["window", "estimator", "term"])
    for t, k in (("rb_x_high", "rb"), ("gpt_x_high", "gpt")):
        assert abs(res["Baseline"][k] - l08.loc[("extended_to_2026-06", "OLS_ln", t), "coef"]) < 1e-8
    print("GATE PASSED: current-window baseline = l08")
    rows = [dict(row=k, source="l57", **v) for k, v in res.items()]

    # Carried rows (same panel, same months).
    p = l08.xs(("extended_to_2026-06", "Poisson"), level=["window", "estimator"])
    s = pd.read_csv(L06).set_index(["spec", "term"])
    for name, get, src in (
            ("Poisson on counts", lambda t: p.loc[t], "l08 postings_extended_did.csv"),
            ("Group by calendar-month effects", lambda t: s.loc[("S1_groupseason", t)],
             "l06 postings_seasonality.csv S1_groupseason")):
        rb, gp = get("rb_x_high"), get("gpt_x_high")
        rows.append(dict(row=name, source=src, rb=rb.coef, se_rb=rb.se, p_rb=rb.pval, gpt=gp.coef,
                         se_gpt=gp.se, p_gpt=gp.pval, n_obs=int(rb.n_obs), n_occ=369,
                         n_high_occ=res["Baseline"]["n_high_occ"]))
    out = pd.DataFrame(rows)
    out.insert(0, "panel", "2020-01 to 2026-06")
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string())
    write_tex(out)


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
          r"pseudo-maximum likelihood on the counts; row (8) adds SSYK 1-digit group $\times$ calendar-month "
          r"fixed effects. Standard errors clustered by occupation. "
          r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}",
          r"\end{table}"]
    PAPER_TAB.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {PAPER_TAB}")


if __name__ == "__main__":
    main()

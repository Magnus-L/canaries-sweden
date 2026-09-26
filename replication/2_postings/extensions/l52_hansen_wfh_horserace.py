#!/usr/bin/env python3
"""
l52_hansen_wfh_horserace.py: Lambert and Schindler's AI-versus-remote-work
horse race on the posting margin, with THEIR remote-work measure.

QUESTION
Lambert and Schindler (2026, "The Broken Ladder", SSRN 6787638) find that
realised remote work, not generative-AI exposure, accounts for the fall
in junior hiring after 2022. Their measure (their p. 10) is Hansen et al.
(2023): the share of an occupation's 2021-2022 job postings that offer one
or more days of remote or hybrid work. l50 builds a Swedish text-based
analogue from Platsbanken, which flags only 3 to 5 per cent of
advertisements. This script is the complement: the Hansen et al. measure
itself, from the public WFH Map release, crosswalked to SSYK 2012 and put
through l50's designs unchanged, so the two are directly comparable.

THE MEASURE, AND HOW IT DIFFERS FROM LAMBERT AND SCHINDLER'S
The public (Category A) WFH Map file, data/raw/wfhmap/ (see its README),
carries two occupation tables for the United States only:
  - us_occ_by_month: monthly shares by 2018 SOC minor group (96 groups),
    January 2019 to June 2026, with the posting count N.
  - us_occ_detailed: shares by 2018 SOC detailed occupation (733 codes),
    pooled over 2017-2019 and over 2023-2026 only; no 2021-2022 column.
Lambert and Schindler use detailed occupations, 2021-2022, pooled over
four countries; that cut is not public. Two versions are therefore built:
  H1 (headline): the 2021-2022 share of the occupation's minor group,
     pooled over the 24 months with WFH Map's own posting counts as
     weights (their window, a coarser occupation). Each 2018 SOC detailed
     code takes its minor group's share (longest code-prefix match).
  H2 (robustness): the detailed 2023-2026 share (their occupation grain,
     a later window that overlaps our post period).

CROSSWALK
Exactly the route of the Dingel-Neiman and Eloundou scores already in the
paper: SOC 2010 -> ISCO-08 (BLS) -> SSYK 2012 (SCB), unweighted means at
each step, using the functions of src/09_remote_work_robustness.py
themselves (imported, with the crosswalk paths pointed at the local copies
in data/raw/). WFH Map codes are 2018 SOC, so one step is put in front:
2018 SOC -> 2010 SOC through the official BLS 2010-to-2018 crosswalk
(lab-infrastructure/ai-exposure-measures/raw/crosswalks/
soc_2010_to_2018_crosswalk.xlsx), a 2010 code taking the unweighted mean
of the 2018 codes it maps to. Gate: the same code path fed Dingel and
Neiman must reproduce revision/upload/dingel_neiman_ssyk4.dta exactly.

DESIGNS (l50's, unchanged, with the Hansen score in place of l50's)
(1) Occupation x month panel, Equation (1): ln postings, occupation and
    month effects, PostRB x High and PostGPT x High, clustered by
    occupation, 28,084 cells, January 2020 to June 2026. Gate: -0.1271
    and -0.0593. Then (i) PostRB x RemoteHigh and PostGPT x RemoteHigh,
    RemoteHigh = top quartile of H1 across the panel occupations; (ii)
    standardised continuous scores; (iii) Lambert and Schindler's layout,
    one post-launch dummy times each standardised score alone and then
    jointly; (iv) Poisson of (i); (i) also on entry-level advertisements
    (l50's ad-level extract, read through l50's own loader).
(2) Within employer, l09's OA V design (Poisson, employer x quartile and
    employer x month effects, clustered by employer, January 2021 to June
    2026). Gate: -0.158. Then l50's finer panel, employer x DAIOE quartile
    x RemoteHigh x month, so PostGPT x RemoteHigh is identified within the
    employer-month. There is no employer-level Hansen measure for Swedish
    employers, so l50's employer split has no counterpart here.

    python3 revision/local/l52_hansen_wfh_horserace.py

OUTPUTS
revision/tables/l52_hansen_occ_measure.csv, l52_hansen_coverage.csv,
l52_hansen_correlations.csv, l52_hansen_horserace.csv,
l52_hansen_within.csv; canaries-sweden-paper/tables/
tableA_remote_horserace_hansen.tex (not \\input anywhere).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

WFH = ROOT / "data" / "raw" / "wfhmap" / "remote_work_in_job_ads_public_data.xlsx"
SOC1018 = (ROOT.parents[1] / "lab-infrastructure" / "ai-exposure-measures" / "raw"
           / "crosswalks" / "soc_2010_to_2018_crosswalk.xlsx")
PAPER_TAB = ROOT.parent / "canaries-sweden-paper" / "tables"
T = _cfg.V2_TAB

RB_YM, GPT_YM = "2022-04", "2022-12"
MEAS_LO, MEAS_HI = 2021, 2022          # Lambert-Schindler's 2021-2022


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The paper's crosswalk code, imported rather than copied.
sys.path.insert(0, str(ROOT / "src"))
x09 = _load("x09", ROOT / "src" / "09_remote_work_robustness.py")
x09.BLS_CROSSWALK = ROOT / "data" / "raw" / "isco_soc_crosswalk2.xls"
x09.SCB_CROSSWALK = ROOT / "data" / "raw" / "ssyk2012_isco08.xlsx"
l50 = _load("l50", REV / "local" / "l50_remote_work_horserace.py")
stars, wcorr, zscore = l50.stars, l50.wcorr, l50.zscore


# ---------------------------------------------------------------------------
# 1. the measure
# ---------------------------------------------------------------------------
def soc_to_ssyk(soc2010_scores: pd.DataFrame, col: str) -> pd.DataFrame:
    """SOC 2010 -> ISCO-08 -> SSYK 2012 through src/09's own functions.
    build_ssyk_telework averages a column called 'teleworkable'; the score is
    renamed in and out, so the arithmetic is the paper's line for line."""
    s = soc2010_scores.rename(columns={col: "teleworkable"})
    out = x09.build_ssyk_telework(s, x09.load_soc_to_isco(), x09.load_isco_to_ssyk())
    return out.rename(columns={"teleworkable": col})


def crosswalk_gate():
    """The same path fed Dingel and Neiman reproduces the upload file."""
    dn = pd.read_csv(ROOT / "data" / "raw" / "dingel_neiman_telework.csv")
    dn["soc2010"] = dn["onetsoccode"].str[:7]
    soc = dn.groupby("soc2010")["teleworkable"].mean().reset_index()
    mine = soc_to_ssyk(soc, "teleworkable")
    mine["ssyk4"] = mine["ssyk4"].astype(int)
    up = pd.read_stata(REV / "upload" / "dingel_neiman_ssyk4.dta")
    m = up.merge(mine, on="ssyk4", how="outer", suffixes=("_up", "_mine"), indicator=True)
    assert (m["_merge"] == "both").all() and len(m) == len(up), "DN code sets differ"
    assert np.allclose(m["teleworkable_up"], m["teleworkable_mine"], atol=1e-9), \
        "DN scores differ"
    print(f"  GATE crosswalk: src/09 path reproduces dingel_neiman_ssyk4.dta "
          f"({len(up)} codes, max abs diff "
          f"{(m['teleworkable_up'] - m['teleworkable_mine']).abs().max():.1e})")


def soc2018_to_2010():
    x = pd.read_excel(SOC1018, header=None, skiprows=9, dtype=str).iloc[:, [0, 2]]
    x.columns = ["soc2010", "soc2018"]
    x = x.dropna()
    x = x[x["soc2010"].str.match(r"^\d\d-\d{4}$") & x["soc2018"].str.match(r"^\d\d-\d{4}$")]
    return x.apply(lambda c: c.str.strip()).drop_duplicates()


def build_hansen():
    """H1 (minor group, 2021-2022) and H2 (detailed, 2023-2026) on SOC 2018
    detailed codes, then on SOC 2010, then on SSYK 2012."""
    mo = pd.read_excel(WFH, sheet_name="us_occ_by_month")
    mo.columns = ["year", "month", "ym_num", "minor", "minor_name", "pct", "pct3", "n"]
    w = mo[(mo["year"] >= MEAS_LO) & (mo["year"] <= MEAS_HI)].copy()
    assert w.groupby("minor").size().eq(24).all(), "minor group without 24 months"
    w["remote_n"] = w["pct"] / 100 * w["n"]
    g = w.groupby(["minor", "minor_name"]).agg(remote_n=("remote_n", "sum"),
                                               n_2122=("n", "sum")).reset_index()
    g["h1"] = g["remote_n"] / g["n_2122"]
    g["prefix"] = g["minor"].str.replace("-", "").str.rstrip("0")

    det = pd.read_excel(WFH, sheet_name="us_occ_detailed")
    det.columns = ["soc2018", "name", "wfh_1719", "wfh_2326", "n_1719", "n_2326"]
    det["h2"] = det["wfh_2326"] / 100

    xw = soc2018_to_2010()
    codes = pd.DataFrame({"soc2018": sorted(set(xw["soc2018"]) | set(det["soc2018"]))})
    # longest prefix: 15-1252 -> "151252" starts with "1512" (15-1200)
    pref = sorted(g["prefix"], key=len, reverse=True)
    def minor_of(c):
        k = c.replace("-", "")
        for p in pref:
            if k.startswith(p):
                return p
        return None
    codes["prefix"] = codes["soc2018"].map(minor_of)
    codes = (codes.merge(g[["prefix", "minor", "h1"]], on="prefix", how="left")
             .merge(det[["soc2018", "h2"]], on="soc2018", how="left"))
    print(f"  SOC 2018 detailed codes: {len(codes)}; with a minor-group share: "
          f"{codes['h1'].notna().sum()}; with a detailed 2023-26 share: {codes['h2'].notna().sum()}")

    s10 = xw.merge(codes, on="soc2018", how="left")
    out = {}
    for col in ("h1", "h2"):
        soc = (s10.dropna(subset=[col]).groupby("soc2010")[col].mean().reset_index())
        print(f"  {col}: {len(soc)} SOC 2010 codes")
        out[col] = soc_to_ssyk(soc, col)
    h = out["h1"].merge(out["h2"], on="ssyk4", how="outer")
    h["ssyk4"] = h["ssyk4"].astype(str).str.zfill(4)
    return h, g


def build_occ(panel):
    h, g = build_hansen()
    d = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv", dtype={"ssyk4": str})
    d["ssyk4"] = d["ssyk4"].str.zfill(4)
    dn = pd.read_stata(REV / "upload" / "dingel_neiman_ssyk4.dta")
    dn["ssyk4"] = dn["ssyk4"].astype(str).str.zfill(4)
    l50o = pd.read_csv(T / "l50_remote_occ_measure.csv", dtype={"ssyk4": str})
    l50o["ssyk4"] = l50o["ssyk4"].str.zfill(4)
    vol = panel.groupby("ssyk4")["n_ads"].sum().rename("n_ads_all").reset_index()
    v2122 = (panel[panel["year_month"].between("2021-01", "2022-12")]
             .groupby("ssyk4")["n_ads"].sum().rename("n_ads_2122").reset_index())
    o = (pd.DataFrame({"ssyk4": sorted(panel["ssyk4"].unique())})
         .merge(d[["ssyk4", "pctl_rank_genai", "exposure_quartile"]], on="ssyk4", how="left")
         .merge(h, on="ssyk4", how="left")
         .merge(dn[["ssyk4", "teleworkable"]], on="ssyk4", how="left")
         .merge(l50o[["ssyk4", "remote_share", "emp_2024"]], on="ssyk4", how="left")
         .merge(vol, on="ssyk4", how="left").merge(v2122, on="ssyk4", how="left"))
    o["high"] = o["exposure_quartile"].astype(str).str.startswith("Q4").astype(int)
    cov = []
    for col, lab in (("h1", "H1 minor group 2021-22"), ("h2", "H2 detailed 2023-26"),
                     ("teleworkable", "Dingel-Neiman (reference)")):
        has = o[col].notna()
        cov.append({"measure": lab, "n_occ_panel": len(o), "n_occ_scored": int(has.sum()),
                    "share_occ": has.mean(),
                    "share_ads_all": o.loc[has, "n_ads_all"].sum() / o["n_ads_all"].sum(),
                    "share_ads_2122": o.loc[has, "n_ads_2122"].sum() / o["n_ads_2122"].sum(),
                    "share_emp_2024": o.loc[has, "emp_2024"].sum() / o["emp_2024"].sum()})
    cov = pd.DataFrame(cov)
    cov.to_csv(T / "l52_hansen_coverage.csv", index=False)
    print(cov.to_string(index=False))
    return o, g


def correlations(o):
    rows = []
    def add(lab, x, y):
        s = o[o[x].notna() & o[y].notna()]
        e = s[s["emp_2024"].fillna(0) > 0]
        rows.append({"pair": lab, "n_occ": len(s), "unweighted": wcorr(s[x], s[y]),
                     "ad_weighted_2122": wcorr(s[x], s[y], s["n_ads_2122"]),
                     "employment_weighted_2024": wcorr(e[x], e[y], e["emp_2024"]),
                     "spearman": wcorr(s[x].rank(), s[y].rank())})
    for h, hl in (("h1", "Hansen H1"), ("h2", "Hansen H2")):
        add(f"{hl} x DAIOE pctl_rank_genai", h, "pctl_rank_genai")
        add(f"{hl} x Dingel-Neiman teleworkable", h, "teleworkable")
        add(f"{hl} x l50 Platsbanken remote_share", h, "remote_share")
    add("Hansen H1 x Hansen H2", "h1", "h2")
    s = o[o["h1"].notna()]
    for h, hl in (("h1", "H1"), ("h2", "H2")):
        s = o[o[h].notna()]
        off = (s["pctl_rank_genai"] >= s["pctl_rank_genai"].median()) != (s[h] >= s[h].median())
        rows.append({"pair": f"off-diagonal share, median cuts DAIOE x Hansen {hl}",
                     "n_occ": len(s), "unweighted": off.mean(),
                     "ad_weighted_2122": np.average(off, weights=s["n_ads_2122"]),
                     "employment_weighted_2024": np.average(off, weights=s["emp_2024"].fillna(0)),
                     "spearman": np.nan})
    s = o[o["h1"].notna()]
    both = (s["high"] == 1) & (s["remote_high"] == 1)
    rows.append({"pair": "share of DAIOE Q4 occupations also RemoteHigh (H1)",
                 "n_occ": int(s["high"].sum()), "unweighted": both.sum() / s["high"].sum(),
                 "ad_weighted_2122": np.average(both[s["high"] == 1],
                                                weights=s.loc[s["high"] == 1, "n_ads_2122"]),
                 "employment_weighted_2024": np.nan, "spearman": np.nan})
    for h in ("h1", "h2"):
        for k, v in o[h].describe(percentiles=[.1, .25, .5, .75, .9]).items():
            rows.append({"pair": f"{h} distribution: {k}", "n_occ": int(o[h].notna().sum()),
                         "unweighted": v, "ad_weighted_2122": np.nan,
                         "employment_weighted_2024": np.nan, "spearman": np.nan})
    c = pd.DataFrame(rows)
    c.to_csv(T / "l52_hansen_correlations.csv", index=False)
    print(c.to_string(index=False))
    return c


# ---------------------------------------------------------------------------
# 2. estimation
# ---------------------------------------------------------------------------
def main():
    import pyfixest as pf
    print("L52: Lambert-Schindler horse race with the Hansen et al. (2023) measure")
    crosswalk_gate()

    panel = pd.read_csv(_cfg.PROCESSED / "postings_daioe_merged_extended.csv",
                        dtype={"ssyk4": str})
    panel["ssyk4"] = panel["ssyk4"].str.zfill(4)
    o, g = build_occ(panel)
    sc = o["h1"].notna()
    q75 = o.loc[sc, "h1"].quantile(0.75)
    o["remote_high"] = np.where(sc, (o["h1"] >= q75).astype(float), np.nan)
    o["z_ai"] = zscore(o["pctl_rank_genai"])
    o["z_rem"] = zscore(o["h1"])          # NaN-aware (pandas skips NaN)
    o["z_rem2"] = zscore(o["h2"])
    print(f"  RemoteHigh threshold (H1, 75th pct over {sc.sum()} occupations): {q75:.4f}")
    o.to_csv(T / "l52_hansen_occ_measure.csv", index=False)
    corr = correlations(o)

    p = panel.merge(o[["ssyk4", "remote_high", "z_ai", "z_rem", "z_rem2"]],
                    on="ssyk4", how="left")
    p["high"] = p["high_exposure"]
    def add_terms(df):
        df["post_rb"] = (df["year_month"] >= RB_YM).astype(int)
        df["post_gpt"] = (df["year_month"] >= GPT_YM).astype(int)
        for nm, dummy in (("high", "high"), ("rhigh", "remote_high")):
            df[f"rb_x_{nm}"] = df["post_rb"] * df[dummy]
            df[f"gpt_x_{nm}"] = df["post_gpt"] * df[dummy]
        for z in ("z_ai", "z_rem", "z_rem2"):
            df[f"rb_x_{z}"] = df["post_rb"] * df[z]
            df[f"gpt_x_{z}"] = df["post_gpt"] * df[z]
            df[f"post_x_{z}"] = df["post_gpt"] * df[z]
        return df
    p = add_terms(p)
    p["ln_ads"] = np.log(p["n_ads"])

    rows = []
    def fit(label, formula, data, est="OLS", sample="all ads"):
        f = pf.feols if est == "OLS" else pf.fepois
        r = f(formula, data=data, vcov={"CRV1": "ssyk4"})
        for t in r.coef().index:
            rows.append({"design": "occupation_panel", "sample": sample, "spec": label,
                         "estimator": est, "term": t, "coef": r.coef()[t], "se": r.se()[t],
                         "pval": r.pvalue()[t], "n_obs": r._N,
                         "n_occ": data["ssyk4"].nunique()})
        print(f"  [{sample} | {label} | {est}] N={r._N:,}  " + "  ".join(
            f"{t} {r.coef()[t]:+.4f} ({r.se()[t]:.4f})" for t in r.coef().index))
        return r

    FE = " | ssyk4 + year_month"
    base = fit("baseline", "ln_ads ~ rb_x_high + gpt_x_high" + FE, p)
    assert abs(base.coef()["gpt_x_high"] + 0.0593) < 5e-4 and \
        abs(base.coef()["rb_x_high"] + 0.1271) < 5e-4 and base._N == 28084, "baseline gate failed"
    print("  GATE baseline reproduced: -0.1271 / -0.0593 on 28,084")
    ps = p[p["remote_high"].notna()].copy()
    if len(ps) < len(p):
        fit("baseline_scored", "ln_ads ~ rb_x_high + gpt_x_high" + FE, ps,
            sample="all ads, scored occupations")
    fit("i_quartile", "ln_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE, ps)
    fit("i_remote_only", "ln_ads ~ rb_x_rhigh + gpt_x_rhigh" + FE, ps)
    fit("ii_continuous_ai_only", "ln_ads ~ rb_x_z_ai + gpt_x_z_ai" + FE, ps)
    fit("ii_continuous", "ln_ads ~ rb_x_z_ai + gpt_x_z_ai + rb_x_z_rem + gpt_x_z_rem" + FE, ps)
    fit("iii_LS_ai_alone", "ln_ads ~ post_x_z_ai" + FE, ps)
    fit("iii_LS_remote_alone", "ln_ads ~ post_x_z_rem" + FE, ps)
    fit("iii_LS_joint", "ln_ads ~ post_x_z_ai + post_x_z_rem" + FE, ps)
    fit("iv_poisson_baseline", "n_ads ~ rb_x_high + gpt_x_high" + FE, ps, est="Poisson")
    fit("iv_poisson_quartile", "n_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE,
        ps, est="Poisson")
    # H2, the detailed 2023-2026 share: continuous and Lambert-Schindler form
    p2 = p[p["z_rem2"].notna()].copy()
    fit("ii_continuous_H2", "ln_ads ~ rb_x_z_ai + gpt_x_z_ai + rb_x_z_rem2 + gpt_x_z_rem2" + FE,
        p2, sample="all ads, H2 scored occupations")
    fit("iii_LS_remote_alone_H2", "ln_ads ~ post_x_z_rem2" + FE, p2,
        sample="all ads, H2 scored occupations")
    fit("iii_LS_joint_H2", "ln_ads ~ post_x_z_ai + post_x_z_rem2" + FE, p2,
        sample="all ads, H2 scored occupations")

    # entry-level advertisements: l50's ad-level extract through l50's loader
    a = l50.load_occ_ads()
    occ_ids = set(o["ssyk4"])
    ent = (a[a["entry"] == 1].groupby(["ssyk4", "ym"]).size()
           .rename("n_ads").reset_index().rename(columns={"ym": "year_month"}))
    ent = ent[ent["ssyk4"].isin(occ_ids)]
    del a
    pe = ent.merge(p.drop(columns=["n_ads", "ln_ads"]).drop_duplicates(["ssyk4", "year_month"]),
                   on=["ssyk4", "year_month"], how="inner")
    pe["ln_ads"] = np.log(pe["n_ads"])
    print(f"  entry-level: {int(pe['n_ads'].sum()):,} ads in {len(pe):,} occupation-months")
    fit("baseline", "ln_ads ~ rb_x_high + gpt_x_high" + FE, pe, sample="entry-level ads")
    pes = pe[pe["remote_high"].notna()]
    fit("i_quartile", "ln_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE, pes,
        sample="entry-level ads")
    fit("iv_poisson_baseline", "n_ads ~ rb_x_high + gpt_x_high" + FE, pes, est="Poisson",
        sample="entry-level ads")
    fit("iv_poisson_quartile", "n_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE,
        pes, est="Poisson", sample="entry-level ads")
    fit("iii_LS_joint", "ln_ads ~ post_x_z_ai + post_x_z_rem" + FE, pes, sample="entry-level ads")
    res = pd.DataFrame(rows)
    res.to_csv(T / "l52_hansen_horserace.csv", index=False)

    # -------- within employer (l09 / l50's finer panel) ---------------------
    l09 = _load("l09", REV / "local" / "l09_firm_within_did.py")
    cube = l09.load_cube()
    d = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv", dtype={"ssyk4": str})
    d["ssyk4"] = d["ssyk4"].str.zfill(4)
    dq = d[["ssyk4", "exposure_quartile"]].copy()
    dq["exposure_quartile"] = dq["exposure_quartile"].astype(str).str.extract(r"Q(\d)").astype(int)
    # Hansen RemoteHigh on every DAIOE code the cube may carry (not only the
    # 369 panel occupations), with the panel threshold.
    hh, _ = build_hansen()
    rh = hh[["ssyk4", "h1"]].dropna()
    rh["remote_high"] = (rh["h1"] >= q75).astype(int)

    wrows = []
    def wfit(label, bal, terms, fe="fe_fq + fe_ft", sample="all ads", note=""):
        r = pf.fepois(f"n_ads ~ {' + '.join(terms)} | {fe}", data=bal, vcov={"CRV1": "orgnr"})
        for t in terms:
            wrows.append({"design": "within_employer", "sample": sample, "spec": label,
                          "estimator": "Poisson", "term": t, "coef": r.coef()[t],
                          "se": r.se()[t], "pval": r.pvalue()[t], "n_obs": r._N,
                          "n_firms": bal["orgnr"].nunique(), "note": note})
        print(f"  [within | {sample} | {label}] N={r._N:,} firms={bal['orgnr'].nunique():,}  "
              + "  ".join(f"{t} {r.coef()[t]:+.4f} ({r.se()[t]:.4f})" for t in terms))
        return r

    def finer_panel(outcome, employers):
        """l50's finer panel, verbatim, with the Hansen RemoteHigh."""
        cm = cube.merge(dq, on="ssyk4", how="inner").merge(
            rh[["ssyk4", "remote_high"]], on="ssyk4", how="inner")
        cm = cm[cm["orgnr"].isin(employers)]
        cell = (cm.groupby(["orgnr", "exposure_quartile", "remote_high", "month"])[outcome]
                .sum().rename("n_ads").reset_index().rename(columns={"month": "year_month"}))
        pairs = cell[cell["n_ads"] > 0][["orgnr", "exposure_quartile", "remote_high"]].drop_duplicates()
        months = sorted(cube["month"].unique())
        bal = (pairs.assign(_k=1).merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
               .drop(columns="_k")
               .merge(cell, on=["orgnr", "exposure_quartile", "remote_high", "year_month"], how="left"))
        bal["n_ads"] = bal["n_ads"].fillna(0).astype(int)
        bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
        pr, pg = (bal["year_month"] >= RB_YM).astype(int), (bal["year_month"] >= GPT_YM).astype(int)
        bal["rb_x_high"], bal["gpt_x_high"] = pr * bal["high"], pg * bal["high"]
        bal["rb_x_rhigh"], bal["gpt_x_rhigh"] = pr * bal["remote_high"], pg * bal["remote_high"]
        bal["fe_fqr"] = (bal["orgnr"] + "_" + bal["exposure_quartile"].astype(str)
                         + "_" + bal["remote_high"].astype(str))
        bal["fe_ft"] = bal["orgnr"] + "_" + bal["year_month"]
        share = cm[outcome].sum() / cube.merge(dq, on="ssyk4").query(
            "orgnr in @employers")[outcome].sum()
        print(f"  finer panel ({outcome}): {share:.2%} of the employers' DAIOE-coded ads carry a Hansen score")
        return bal

    bal_a = l09.build_panel(cube, dq, "ads")
    r0 = wfit("l09_a_baseline", bal_a, ["rb_x_high", "gpt_x_high"])
    assert abs(r0.coef()["gpt_x_high"] + 0.158) < 5e-4, "within-employer gate failed"
    print("  GATE within-employer reproduced: -0.158")
    fa = finer_panel("ads", sorted(bal_a["orgnr"].unique()))
    del bal_a
    wfit("finer_ai_only", fa, ["rb_x_high", "gpt_x_high"], fe="fe_fqr + fe_ft",
         note="employer x quartile x Hansen remote-high cells; l09 employers")
    wfit("finer_joint", fa, ["rb_x_high", "gpt_x_high", "rb_x_rhigh", "gpt_x_rhigh"],
         fe="fe_fqr + fe_ft", note="employer x quartile x Hansen remote-high cells; l09 employers")
    del fa
    bal_c = l09.build_panel(cube, dq, "entry")
    wfit("l09_c_baseline", bal_c, ["rb_x_high", "gpt_x_high"], sample="entry-level ads")
    fc = finer_panel("entry", sorted(bal_c["orgnr"].unique()))
    wfit("finer_ai_only", fc, ["rb_x_high", "gpt_x_high"], fe="fe_fqr + fe_ft",
         sample="entry-level ads", note="employer x quartile x Hansen remote-high cells; l09 entry employers")
    wfit("finer_joint", fc, ["rb_x_high", "gpt_x_high", "rb_x_rhigh", "gpt_x_rhigh"],
         fe="fe_fqr + fe_ft", sample="entry-level ads",
         note="employer x quartile x Hansen remote-high cells; l09 entry employers")
    wres = pd.DataFrame(wrows)
    wres.to_csv(T / "l52_hansen_within.csv", index=False)
    write_tex(res, wres, o, q75, int(sc.sum()))
    print("Saved l52 tables and tableA_remote_horserace_hansen.tex")


def write_tex(res, wres, o, q75, n_sc):
    def c(df, sample, spec, term, est=None):
        s = df[(df["sample"] == sample) & (df["spec"] == spec) & (df["term"] == term)]
        if est:
            s = s[s["estimator"] == est]
        if s.empty:
            return "", ""
        r = s.iloc[0]
        return (f"${r.coef:.3f}^{{{stars(r.pval)}}}$" if stars(r.pval) else f"${r.coef:.3f}$",
                f"({r.se:.3f})")

    def n(df, sample, spec):
        s = df[(df["sample"] == sample) & (df["spec"] == spec)]
        return f"{int(s.iloc[0].n_obs):,}"

    cols = [("all ads", "baseline", "OLS"), ("all ads", "i_quartile", "OLS"),
            ("all ads", "ii_continuous", "OLS"), ("all ads", "iv_poisson_quartile", "Poisson"),
            ("entry-level ads", "baseline", "OLS"), ("entry-level ads", "i_quartile", "OLS")]
    terms = [("rb_x_high", r"PostRB $\times$ High AI"), ("gpt_x_high", r"PostGPT $\times$ High AI"),
             ("rb_x_rhigh", r"PostRB $\times$ High remote"), ("gpt_x_rhigh", r"PostGPT $\times$ High remote")]
    cont = {"rb_x_high": "rb_x_z_ai", "gpt_x_high": "gpt_x_z_ai",
            "rb_x_rhigh": "rb_x_z_rem", "gpt_x_rhigh": "gpt_x_z_rem"}
    L = [r"\begin{table}[ht!]", r"\centering",
         r"\caption{AI exposure and remote work on the posting margin: the Hansen et al.\ (2023) measure}",
         r"\label{tab:remote_horserace_hansen}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         r" & \multicolumn{4}{c}{All advertisements} & \multicolumn{2}{c}{Entry-level} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r" & (1) & (2) & (3) & (4) & (5) & (6) \\",
         r" & OLS & OLS & OLS, $z$ & Poisson & OLS & OLS \\", r"\midrule"]
    for t, lab in terms:
        cs, ss = [], []
        for smp, spec, est in cols:
            a_, b_ = c(res, smp, spec, cont[t] if spec == "ii_continuous" else t, est)
            cs.append(a_), ss.append(b_)
        L.append(lab + " & " + " & ".join(cs) + r" \\")
        L.append(" & " + " & ".join(ss) + r" \\")
    L.append(r"\midrule")
    L.append("Observations & " + " & ".join(n(res, s, sp) for s, sp, _ in cols) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{7}{l}{\textit{One post-launch period, standardised scores (Lambert and Schindler's form)}} \\",
          r" & AI alone & Remote alone & Joint & Joint, H2 & Joint, entry & \\"]
    for term, lab, pos in (("post_x_z_ai", r"PostGPT $\times$ AI ($z$)", "ai"),
                           ("post_x_z_rem", r"PostGPT $\times$ Remote ($z$)", "rem")):
        alone = c(res, "all ads", "iii_LS_ai_alone" if pos == "ai" else "iii_LS_remote_alone", term)
        joint = c(res, "all ads", "iii_LS_joint", term)
        h2 = c(res, "all ads, H2 scored occupations",
               "iii_LS_joint_H2", term if pos == "ai" else "post_x_z_rem2")
        ent = c(res, "entry-level ads", "iii_LS_joint", term)
        c1 = alone if pos == "ai" else ("", "")
        c2 = alone if pos == "rem" else ("", "")
        L.append(f"{lab} & {c1[0]} & {c2[0]} & {joint[0]} & {h2[0]} & {ent[0]} & \\\\")
        L.append(f" & {c1[1]} & {c2[1]} & {joint[1]} & {h2[1]} & {ent[1]} & \\\\")
    L += [r"\midrule",
          r"\multicolumn{7}{l}{\textit{Within employer (Poisson; employer $\times$ cell and employer $\times$ month effects)}} \\",
          r" & Baseline & AI only & Joint & Entry, AI only & Entry, joint & \\"]
    specs = [("all ads", "l09_a_baseline"), ("all ads", "finer_ai_only"), ("all ads", "finer_joint"),
             ("entry-level ads", "finer_ai_only"), ("entry-level ads", "finer_joint")]
    for t, lab in (("gpt_x_high", r"PostGPT $\times$ High AI"),
                   ("gpt_x_rhigh", r"PostGPT $\times$ High remote")):
        cs, ss = zip(*[c(wres, s, sp, t) for s, sp in specs])
        L.append(lab + " & " + " & ".join(cs) + r" & \\")
        L.append(" & " + " & ".join(ss) + r" & \\")
    def wn(s, sp):
        return f"{int(wres[(wres['sample'] == s) & (wres['spec'] == sp)].iloc[0].n_firms):,}"
    L.append("Employers & " + " & ".join(wn(s, sp) for s, sp in specs) + r" & \\")
    notes = (
        "Remote share: Hansen et al.\\ (2023), the share of United States job postings in 2021 and 2022 "
        "that offer one or more days of remote or hybrid work, from the public WFH Map release, taken at "
        "the level of the 2018 SOC minor group (the finest grain the public file reports for those years) "
        "and crosswalked 2018 SOC $\\to$ 2010 SOC $\\to$ ISCO-08 $\\to$ SSYK 2012 with unweighted means "
        f"at each step, the route of the Dingel and Neiman score; {n_sc} of the {len(o)} panel "
        "occupations receive a score. High remote is its top quartile across those occupations (a share of "
        f"at least {q75*100:.1f} per cent). Columns (1), (2), (4) to (6): Equation (1) with occupation and "
        "month effects, January 2020 to June 2026, standard errors clustered by occupation; column (1) is "
        "the published specification. Column (3) replaces both dummies with the standardised DAIOE "
        "percentile and remote share. Columns (5) and (6) count only advertisements the Monitor's keyword "
        "rule flags as entry-level. The middle panel interacts one post-launch dummy (from December 2022) "
        "with each standardised score, alone and together; H2 uses the 2023 to 2026 share of the detailed "
        "2018 SOC occupation instead. The bottom panel is the within-employer design of Online Appendix V "
        "(Poisson, January 2021 to June 2026, clustered by employer), with each employer's cells cut by "
        "DAIOE quartile and by High remote. PostRB terms in the lower two panels are estimated and reported in the CSV. "
        "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.97\textwidth}\footnotesize\vspace{4pt}", notes, r"\end{minipage}",
          r"\end{table}"]
    PAPER_TAB.mkdir(parents=True, exist_ok=True)
    (PAPER_TAB / "tableA_remote_horserace_hansen.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

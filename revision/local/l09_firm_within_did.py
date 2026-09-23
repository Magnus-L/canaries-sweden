#!/usr/bin/env python3
"""
l09_firm_within_did.py: the within-employer design on public
advertisements, through June 2026.

QUESTION
The register design identifies from recomposition inside employers. Each
Platsbanken advertisement carries the employer's organisation number from
January 2021 in about 99 per cent of cases, recorded at publication, so
the same design runs on public data with no occupation register and no
register lag: do advertisements in exposed occupations fall relative to
the same employer's other advertisements after the launch, and is the
response different for entry-level advertisements?

DESIGN
Unit: distinct advertisements by employer, exposure quartile and month,
January 2021 to June 2026, from the AIEL Monitor's firm cube (the Monitor's
deduplication key and population, stated in every output). Sample:
employers with at least five distinct advertisements over the window and
advertisements in both an exposed and a less exposed occupation; the
panel is balanced and zero-filled over employer by quartile by month.
Specification: Poisson pseudo-maximum likelihood with PostRB x High (from
April 2022) and PostGPT x High (from December 2022) under employer-by-
quartile and employer-by-month effects, standard errors clustered by
employer; and the half-year event study with reference the first half of
2022. The pre-hike window is fifteen months, since organisation numbers
begin in January 2021. Variants: (a) all identifying employers; (b)
excluding staffing agencies (SNI 78, from the SCB register bulk) and
public employers (organisation numbers with prefix 2); (c) entry-level
advertisements only, flagged by the Monitor's keyword rule on the
advertisement text (nyexaminerad, nyutexaminerad, junior, trainee or
traineeprogram, ingen erfarenhet, utan erfarenhet, utan tidigare
erfarenhet).

With --variants: (d1, d2) the floor and the screen computed on January
2021 to March 2022 only, so that nothing after the rate rise decides
membership, keeping every quartile cell or only the quartiles used before
the hike; (e) d1 without staffing agencies and public employers; (f) all
advertisements and entry-level advertisements on the employers that pass
both screens; (g) the entry-level differential in one panel of employer by
quartile by entry flag by month, with employer-by-quartile-by-entry and
employer-by-entry-by-month effects and the triple interaction PostGPT x
High x Entry. The share of advertisements the entry rule flags, by year
and by quartile, is written beside the estimates.

INPUTS AND OUTPUTS
Reads lab-infrastructure/ai-monitor/demo/firm-dimension/firm_month_v2.csv.gz
and register/scb_bulkfil.zip, and data/processed/daioe_quartiles.csv.
Writes revision/tables/firm_within_did.csv, firm_within_es.csv and
firm_within_meta.txt; with --variants, firm_within_did_variants.csv,
firm_within_variants.tex and firm_within_entry_audit.csv.

IN THE PAPER
Section 3 (12,141 employers; +0.004 and -0.158; entry-level -0.196 against
-0.160 on the same 1,265 employers, differential -0.044); Online Appendix
V, Tables tab:firm_did (script l10) and tab:firm_variants (written here)
and Figure fig:firm_entry_es (script l07).
"""

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

FD = (REV.parent.parent.parent / "lab-infrastructure" / "ai-monitor"
      / "demo" / "firm-dimension")
CUBE = FD / "firm_month_v2.csv.gz"
SCB_BULK = FD / "register" / "scb_bulkfil.zip"

RB_YM, GPT_YM, REF = "2022-04", "2022-12", "2022H1"
MIN_FIRM_ADS = 5


def load_cube():
    cube = pd.read_csv(CUBE, dtype={"orgnr": str, "ssyk4": str,
                                    "kommun": str})
    cube = cube[(cube["month"] >= "2021-01") & (cube["month"] <= "2026-06")]
    cube["ssyk4"] = cube["ssyk4"].str.zfill(4)
    print(f"  cube: {len(cube):,} rows, {cube['orgnr'].nunique():,} orgnr, "
          f"{cube['month'].min()}..{cube['month'].max()}")
    return cube


def load_staffing_flags():
    """orgnr -> SNI-78 staffing flag from the SCB register bulk."""
    with zipfile.ZipFile(SCB_BULK) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reg = pd.read_csv(io.TextIOWrapper(f, encoding="cp1252"),
                              sep="\t", dtype=str, usecols=lambda c:
                              c in ("PeOrgNr", "Ng1", "SNI1", "Sni1"))
    sni_col = [c for c in reg.columns if c.lower().startswith(("ng", "sni"))][0]
    reg["orgnr"] = reg["PeOrgNr"].str[-10:]
    reg["staffing"] = reg[sni_col].astype(str).str.startswith("78")
    flags = reg.groupby("orgnr")["staffing"].max()
    print(f"  register: {len(flags):,} orgnr, "
          f"{flags.sum():,} staffing (SNI 78)")
    return flags


def build_panel(cube, daioe, outcome="ads", drop_staffing_public=False,
                staffing=None):
    m = cube.merge(daioe, on="ssyk4", how="inner")
    if drop_staffing_public:
        m = m[~m["orgnr"].str.startswith("2")]
        # .map() yields an object series with NaN for unmatched orgnr;
        # cast to bool BEFORE negating, or ~ does bitwise-int nonsense
        staff_mask = m["orgnr"].map(staffing).fillna(False).astype(bool)
        m = m[~staff_mask]
    firm = (m.groupby(["orgnr", "month", "exposure_quartile"],
                      observed=True)[outcome].sum().reset_index()
            .rename(columns={outcome: "n_ads", "month": "year_month"}))
    tot = firm.groupby("orgnr")["n_ads"].sum()
    firm = firm[firm["orgnr"].isin(tot[tot >= MIN_FIRM_ADS].index)]

    emp_q = firm[["orgnr", "exposure_quartile"]].drop_duplicates()
    hi = set(emp_q.loc[emp_q["exposure_quartile"] == 4, "orgnr"])
    lo = set(emp_q.loc[emp_q["exposure_quartile"] < 4, "orgnr"])
    emp_q = emp_q[emp_q["orgnr"].isin(hi & lo)]
    months = sorted(firm["year_month"].unique())
    cell = (firm.groupby(["orgnr", "exposure_quartile", "year_month"],
                         observed=True)["n_ads"].sum().reset_index())
    bal = (emp_q.assign(_k=1)
           .merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
           .drop(columns="_k")
           .merge(cell, on=["orgnr", "exposure_quartile", "year_month"],
                  how="left"))
    bal["n_ads"] = bal["n_ads"].fillna(0).astype(int)
    bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
    bal["post_rb"] = (bal["year_month"] >= RB_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= GPT_YM).astype(int)
    bal["rb_x_high"] = bal["post_rb"] * bal["high"]
    bal["gpt_x_high"] = bal["post_gpt"] * bal["high"]
    bal["fe_fq"] = bal["orgnr"] + "_" + bal["exposure_quartile"].astype(str)
    bal["fe_ft"] = bal["orgnr"] + "_" + bal["year_month"]
    bal["halfyear"] = (bal["year_month"].str[:4]
                       + np.where(bal["year_month"].str[5:7].astype(int)
                                  <= 6, "H1", "H2"))
    return bal


def estimate(bal, label, rows, es_rows, run_es=True):
    import pyfixest as pf
    print(f"  [{label}] {len(bal):,} cells, "
          f"{bal['orgnr'].nunique():,} firms, "
          f"{(bal['n_ads'] == 0).mean():.1%} zeros")
    fit = pf.fepois("n_ads ~ rb_x_high + gpt_x_high | fe_fq + fe_ft",
                    data=bal, vcov={"CRV1": "orgnr"})
    for t in ("rb_x_high", "gpt_x_high"):
        rows.append({"variant": label, "term": t, "coef": fit.coef()[t],
                     "se": fit.se()[t], "pval": fit.pvalue()[t],
                     "n_obs": fit._N,
                     "n_firms": bal["orgnr"].nunique()})
        print(f"      {t:>10}: {fit.coef()[t]:+.4f} "
              f"(SE {fit.se()[t]:.4f}, p {fit.pvalue()[t]:.4f})")
    if run_es:
        periods = sorted(bal["halfyear"].unique())
        terms = []
        for p_ in (p for p in periods if p != REF):
            col = f"hy_{p_}"
            bal[col] = ((bal["halfyear"] == p_) & (bal["high"] == 1)).astype(int)
            terms.append(col)
        fes = pf.fepois(f"n_ads ~ {' + '.join(terms)} | fe_fq + fe_ft",
                        data=bal, vcov={"CRV1": "orgnr"})
        for col in terms:
            es_rows.append({"variant": label, "period": col[3:],
                            "coef": fes.coef()[col], "se": fes.se()[col],
                            "pval": fes.pvalue()[col]})
        es_rows.append({"variant": label, "period": REF,
                        "coef": 0.0, "se": 0.0, "pval": 1.0})


PRE_END = "2022-03"      # last month before the Riksbank's first hike
ENTRY_RULE = ("nyexaminerad | nyutexaminerad | junior | trainee | traineeprogram | "
              "ingen erfarenhet | utan erfarenhet | utan tidigare erfarenhet")


def _cells(cube, daioe, outcome, drop_staffing_public=False, staffing=None):
    """Employer x quartile x month counts of `outcome` from the cube, no
    floor and no screen applied; the callers decide eligibility."""
    m = cube.merge(daioe, on="ssyk4", how="inner")
    if drop_staffing_public:
        m = m[~m["orgnr"].str.startswith("2")]
        staff_mask = m["orgnr"].map(staffing).fillna(False).astype(bool)
        m = m[~staff_mask]
    firm = (m.groupby(["orgnr", "month", "exposure_quartile"],
                      observed=True)[outcome].sum().reset_index()
            .rename(columns={outcome: "n_ads", "month": "year_month"}))
    return firm


def _balance(emp_q, firm):
    """Zero-filled balanced panel on the (employer, quartile) pairs in
    emp_q over every month in `firm`, with the design's regressors."""
    months = sorted(firm["year_month"].unique())
    cell = (firm.groupby(["orgnr", "exposure_quartile", "year_month"],
                         observed=True)["n_ads"].sum().reset_index())
    bal = (emp_q.assign(_k=1)
           .merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
           .drop(columns="_k")
           .merge(cell, on=["orgnr", "exposure_quartile", "year_month"],
                  how="left"))
    bal["n_ads"] = bal["n_ads"].fillna(0).astype(int)
    bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
    bal["post_rb"] = (bal["year_month"] >= RB_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= GPT_YM).astype(int)
    bal["rb_x_high"] = bal["post_rb"] * bal["high"]
    bal["gpt_x_high"] = bal["post_gpt"] * bal["high"]
    bal["fe_fq"] = bal["orgnr"] + "_" + bal["exposure_quartile"].astype(str)
    bal["fe_ft"] = bal["orgnr"] + "_" + bal["year_month"]
    bal["halfyear"] = (bal["year_month"].str[:4]
                       + np.where(bal["year_month"].str[5:7].astype(int)
                                  <= 6, "H1", "H2"))
    return bal


def build_panel_pre_eligible(cube, daioe, outcome="ads", pairs_only=False,
                             drop_staffing_public=False, staffing=None):
    """The design with eligibility decided before the rate hike.

    The floor (five distinct advertisements) and the screen (an exposed
    and a less exposed occupation) are computed on January 2021 to March
    2022 only. Eligible employers then carry every month to June 2026,
    zeros included. With pairs_only=False every quartile enters for every
    eligible employer (a quartile the employer never advertises in is an
    all-zero cell series, which the estimator drops on its own); with
    pairs_only=True only the quartiles the employer used before the hike
    enter, so a later move into a new quartile is not counted."""
    firm = _cells(cube, daioe, outcome, drop_staffing_public, staffing)
    pre = firm[firm["year_month"] <= PRE_END]
    tot = pre.groupby("orgnr")["n_ads"].sum()
    elig = tot[tot >= MIN_FIRM_ADS].index
    pre_q = (pre[pre["orgnr"].isin(elig) & (pre["n_ads"] > 0)]
             [["orgnr", "exposure_quartile"]].drop_duplicates())
    hi = set(pre_q.loc[pre_q["exposure_quartile"] == 4, "orgnr"])
    lo = set(pre_q.loc[pre_q["exposure_quartile"] < 4, "orgnr"])
    keep = sorted(hi & lo)
    if pairs_only:
        emp_q = pre_q[pre_q["orgnr"].isin(keep)].copy()
    else:
        emp_q = pd.DataFrame({"orgnr": np.repeat(keep, 4),
                              "exposure_quartile": np.tile([1, 2, 3, 4],
                                                           len(keep))})
    bal = _balance(emp_q, firm[firm["orgnr"].isin(keep)])
    return bal


def build_panel_triple(cube, daioe, employers):
    """Employer x quartile x entry-flag x month panel for the direct
    entry-level differential, on the employers given (the common sample
    of variant f). Non-entry counts are advertisements minus entry-level
    advertisements. The (employer, quartile) cells are those the employer
    uses at any time in the all-advertisement design, both entry flags,
    every month, zero-filled."""
    m = cube.merge(daioe, on="ssyk4", how="inner")
    m = m[m["orgnr"].isin(employers)].copy()
    m["nonentry"] = m["ads"] - m["entry"]
    assert (m["nonentry"] >= 0).all()
    firm = (m.groupby(["orgnr", "month", "exposure_quartile"],
                      observed=True)[["entry", "nonentry"]].sum()
            .reset_index().rename(columns={"month": "year_month"}))
    emp_q = (firm[(firm["entry"] + firm["nonentry"]) > 0]
             [["orgnr", "exposure_quartile"]].drop_duplicates())
    parts = []
    for flag, col in ((1, "entry"), (0, "nonentry")):
        f = firm[["orgnr", "year_month", "exposure_quartile", col]].rename(
            columns={col: "n_ads"})
        b = _balance(emp_q, f)
        b["entry"] = flag
        parts.append(b)
    bal = pd.concat(parts, ignore_index=True)
    bal["rb_x_high_x_entry"] = bal["rb_x_high"] * bal["entry"]
    bal["gpt_x_high_x_entry"] = bal["gpt_x_high"] * bal["entry"]
    bal["fe_fqe"] = bal["fe_fq"] + "_e" + bal["entry"].astype(str)
    bal["fe_fte"] = bal["fe_ft"] + "_e" + bal["entry"].astype(str)
    return bal


def _fit_rows(bal, label, rows, note=""):
    import pyfixest as pf
    print(f"  [{label}] {len(bal):,} cells, {bal['orgnr'].nunique():,} firms, "
          f"{(bal['n_ads'] == 0).mean():.1%} zeros")
    fit = pf.fepois("n_ads ~ rb_x_high + gpt_x_high | fe_fq + fe_ft",
                    data=bal, vcov={"CRV1": "orgnr"})
    for t in ("rb_x_high", "gpt_x_high"):
        rows.append({"variant": label, "term": t, "coef": fit.coef()[t],
                     "se": fit.se()[t], "pval": fit.pvalue()[t],
                     "n_obs": fit._N, "n_firms": bal["orgnr"].nunique(),
                     "note": note})
        print(f"      {t:>10}: {fit.coef()[t]:+.4f} "
              f"(SE {fit.se()[t]:.4f}, p {fit.pvalue()[t]:.4f})")


def entry_audit(cube, daioe):
    """Share of distinct advertisements the keyword rule flags as
    entry-level, by year and by exposure quartile."""
    m = cube.merge(daioe, on="ssyk4", how="inner")
    m["year"] = m["month"].str[:4]
    out = []
    for dim in ("year", "exposure_quartile"):
        g = m.groupby(dim)[["ads", "entry"]].sum()
        for k, r in g.iterrows():
            out.append({"dimension": dim, "level": k, "ads": int(r["ads"]),
                        "entry": int(r["entry"]),
                        "share_entry": r["entry"] / r["ads"]})
    g = m[["ads", "entry"]].sum()
    out.append({"dimension": "all", "level": "all", "ads": int(g["ads"]),
                "entry": int(g["entry"]), "share_entry": g["entry"] / g["ads"]})
    return pd.DataFrame(out)


def _tex(rows_df, trip):
    def cell(df, v, t):
        r = df[(df["variant"] == v) & (df["term"] == t)].iloc[0]
        star = "$^{***}$" if r.pval < .01 else "$^{**}$" if r.pval < .05 else "$^{*}$" if r.pval < .1 else ""
        return f"{r.coef:.3f}{star} ({r.se:.3f})"
    def nf(df, v):
        r = df[df["variant"] == v].iloc[0]
        return f"{int(r.n_firms):,}", f"{int(r.n_obs):,}"
    labels = [("d1_pre_eligible_all_quartiles", "Eligibility before the hike, every quartile cell"),
              ("d2_pre_eligible_pre_pairs", "Eligibility before the hike, pre-hike quartile cells only"),
              ("e_pre_eligible_excl_staffing_public", "As d1, excluding staffing agencies and public employers"),
              ("f_all_ads_common_sample", "All advertisements, employers passing both screens"),
              ("f_entry_ads_common_sample", "Entry-level advertisements, the same employers")]
    lines = [r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{lccrr}", r"\toprule",
             r"Sample & PostRB $\times$ High & PostGPT $\times$ High & Employers & $N$ \\", r"\midrule"]
    for v, lab in labels:
        if v not in set(rows_df["variant"]):
            continue
        f_, n_ = nf(rows_df, v)
        lines.append(f"{lab} & {cell(rows_df, v, 'rb_x_high')} & {cell(rows_df, v, 'gpt_x_high')} & {f_} & {n_} \\\\")
    lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{Entry-level differential, one panel (employer $\times$ quartile $\times$ entry flag $\times$ month)}} \\"]
    for t, lab in (("rb_x_high", "PostRB $\\times$ High, non-entry advertisements"),
                   ("gpt_x_high", "PostGPT $\\times$ High, non-entry advertisements"),
                   ("rb_x_high_x_entry", "PostRB $\\times$ High $\\times$ Entry"),
                   ("gpt_x_high_x_entry", "PostGPT $\\times$ High $\\times$ Entry")):
        r = trip[trip["term"] == t].iloc[0]
        star = "$^{***}$" if r.pval < .01 else "$^{**}$" if r.pval < .05 else "$^{*}$" if r.pval < .1 else ""
        lines.append(f"{lab} & \\multicolumn{{2}}{{c}}{{{r.coef:.3f}{star} ({r.se:.3f})}} & {int(r.n_firms):,} & {int(r.n_obs):,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main_variants():
    print("L9 variants: eligibility before the hike; entry-level on a common sample")
    cube = load_cube()
    daioe = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    daioe["exposure_quartile"] = (daioe["exposure_quartile"].astype(str)
                                  .str.extract(r"Q(\d)").astype(int))
    daioe = daioe[["ssyk4", "exposure_quartile"]]
    staffing = load_staffing_flags()
    rows = []

    # d and e: eligibility decided before the rate hike
    _fit_rows(build_panel_pre_eligible(cube, daioe, "ads"),
              "d1_pre_eligible_all_quartiles", rows,
              "floor and screen on 2021-01..2022-03; all four quartiles per employer, zero-filled to 2026-06")
    _fit_rows(build_panel_pre_eligible(cube, daioe, "ads", pairs_only=True),
              "d2_pre_eligible_pre_pairs", rows,
              "floor and screen on 2021-01..2022-03; only quartiles used before the hike")
    _fit_rows(build_panel_pre_eligible(cube, daioe, "ads",
                                       drop_staffing_public=True,
                                       staffing=staffing),
              "e_pre_eligible_excl_staffing_public", rows,
              "as d1, without SNI 78 and prefix-2 employers")

    # f: all ads and entry ads on the employers that pass both screens
    bal_a = build_panel(cube, daioe, "ads")
    bal_c = build_panel(cube, daioe, "entry")
    common = sorted(set(bal_a["orgnr"]) & set(bal_c["orgnr"]))
    print(f"  common employers (both screens): {len(common):,} of "
          f"{bal_a['orgnr'].nunique():,} (all) and {bal_c['orgnr'].nunique():,} (entry)")
    _fit_rows(bal_a[bal_a["orgnr"].isin(common)].copy(),
              "f_all_ads_common_sample", rows,
              "whole-window screens on all AND on entry-level advertisements; all-advertisement cells")
    _fit_rows(bal_c[bal_c["orgnr"].isin(common)].copy(),
              "f_entry_ads_common_sample", rows,
              "the same employers; entry-level cells")

    # g: the entry-level differential in one panel
    import pyfixest as pf
    trip_bal = build_panel_triple(cube, daioe, set(common))
    print(f"  [g_triple] {len(trip_bal):,} cells, {trip_bal['orgnr'].nunique():,} firms, "
          f"{(trip_bal['n_ads'] == 0).mean():.1%} zeros")
    fit = pf.fepois("n_ads ~ rb_x_high + gpt_x_high + rb_x_high_x_entry "
                    "+ gpt_x_high_x_entry | fe_fqe + fe_fte",
                    data=trip_bal, vcov={"CRV1": "orgnr"})
    trip = []
    for t in ("rb_x_high", "gpt_x_high", "rb_x_high_x_entry",
              "gpt_x_high_x_entry"):
        trip.append({"variant": "g_entry_triple", "term": t,
                     "coef": fit.coef()[t], "se": fit.se()[t],
                     "pval": fit.pvalue()[t], "n_obs": fit._N,
                     "n_firms": trip_bal["orgnr"].nunique(),
                     "note": "employer x quartile x entry and employer x entry x month effects; common employers"})
        print(f"      {t:>20}: {fit.coef()[t]:+.4f} (SE {fit.se()[t]:.4f}, p {fit.pvalue()[t]:.4f})")
    trip = pd.DataFrame(trip)

    out = pd.concat([pd.DataFrame(rows), trip], ignore_index=True)
    out.to_csv(_cfg.V2_TAB / "firm_within_did_variants.csv", index=False)
    audit = entry_audit(cube, daioe)
    audit.to_csv(_cfg.V2_TAB / "firm_within_entry_audit.csv", index=False)
    print("  entry-level rule:", ENTRY_RULE)
    print(audit.to_string(index=False))
    (_cfg.V2_TAB / "firm_within_variants.tex").write_text(_tex(pd.DataFrame(rows), trip))
    print("Saved firm_within_did_variants.csv, firm_within_entry_audit.csv, firm_within_variants.tex")


def main():
    print("L9: within-employer posting DiD on public data (2021-01..2026-06)")
    cube = load_cube()
    daioe = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    daioe["exposure_quartile"] = (daioe["exposure_quartile"].astype(str)
                                  .str.extract(r"Q(\d)").astype(int))
    daioe = daioe[["ssyk4", "exposure_quartile"]]
    staffing = load_staffing_flags()

    rows, es_rows = [], []
    bal_a = build_panel(cube, daioe, "ads")
    estimate(bal_a, "a_all_firms", rows, es_rows)

    bal_b = build_panel(cube, daioe, "ads", drop_staffing_public=True,
                        staffing=staffing)
    estimate(bal_b, "b_excl_staffing_public", rows, es_rows, run_es=False)

    bal_c = build_panel(cube, daioe, "entry")
    estimate(bal_c, "c_entry_level_ads", rows, es_rows)

    pd.DataFrame(rows).to_csv(_cfg.V2_TAB / "firm_within_did.csv",
                              index=False)
    pd.DataFrame(es_rows).to_csv(_cfg.V2_TAB / "firm_within_es.csv",
                                 index=False)
    meta = [
        "GENERATOR: AIEL Monitor firm cube (firm_month_v2, built 20 Aug",
        "2026, frozen v1.5 pipeline, distinct-ad unit). Population:",
        "employers with organisationsnummer in the ad (99.1-99.4% of ads",
        "from Jan 2021; 0% before -- the panel starts 2021-01).",
        "Window 2021-01..2026-06 (official closed-quarter files).",
        "Pre-Riksbank window is 15 months (orgnr constraint).",
        "Entry-level flag: the Monitor's entry defintion on the ad's",
        "experience requirement. Variant b drops SNI-78 staffing agencies",
        "(register match) and prefix-2 public orgnrs.",
    ]
    (_cfg.V2_TAB / "firm_within_meta.txt").write_text("\n".join(meta))
    print("Saved firm_within_did.csv, firm_within_es.csv, meta")


if __name__ == "__main__":
    if "--variants" in sys.argv:
        main_variants()
    else:
        main()

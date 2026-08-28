#!/usr/bin/env python3
"""
l09_firm_within_did.py -- NEW (ML, 28 Aug): the within-EMPLOYER posting
design on public data, through June 2026.

THE IDEA
========
The paper's register headline is a within-employer composition DiD that
MONA caps at the register frontier. Platsbanken ads carry the employer's
organisationsnummer natively at ~99% from January 2021, so the SAME
design runs on public data with no lag: employer x quartile and
employer x month fixed effects, Poisson, on posting counts. Nobody
publishes this (the Monitor's firm-dimension research found
identifier-based posting-firm linkage unoccupied in the literature);
for the revision it is a second within-employer margin, current through
June 2026, that no coverage objection can touch -- SCB never codes these
ads; the SSYK code is the advertiser's own, present or absent at
publication, with a valid-code share of 100.0%% in the treatment window
(L1).

INPUT: the AIEL Monitor's firm cube (built 20 Aug 2026 from the cached
corpus with the frozen v1.5 pipeline; distinct-ad unit):
  demo/firm-dimension/firm_month_v2.csv.gz
    orgnr, month, ssyk4, kommun, ads, ai, floor, genai, entry, ai_entry
plus the SCB register bulk for the SNI-78 staffing flag
  demo/firm-dimension/register/scb_bulkfil.zip (PeOrgNr 12-digit, cp1252).

Population and unit are the CUBE's, not the paper pipeline's: distinct
ads per the Monitor's dedup key, 2021-01..2026-06 (orgnr coverage begins
2021). Both stated in every output (generator rule).

DESIGN
======
  ads_{f,q,t} ~ PostRB_t x High_q + PostGPT_t x High_q
                | firm x quartile + firm x month          (Poisson)
  - firms observed in Q4 AND a lower quartile (identification restriction)
  - balanced zero-filled firm x quartile x month panel
  - cumulative >= 5 distinct ads per firm (small-firm noise floor)
  - variants: (a) all firms; (b) excluding SNI 78 staffing agencies and
    prefix-2 public orgnrs; (c) ENTRY-LEVEL ads only -- the public echo
    of the age margin (entry ~ inexperienced), same FE
  - half-year event study, ref 2022H1, for (a) and (c)

Pre-period caveat, stated wherever results appear: orgnr exists from
2021 only, so the pre-Riksbank window is Jan 2021 - Mar 2022 (15
months), shorter than the register design's.

Output: tables/firm_within_did.csv, tables/firm_within_es.csv,
        tables/firm_within_meta.txt
Runtime: ~5-10 min (the cube is 1.9M rows; panels are built per variant).
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
        m = m[~m["orgnr"].map(staffing).fillna(False)]
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
    main()

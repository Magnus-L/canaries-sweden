#!/usr/bin/env python3
"""
l09b_firm_heterogeneity.py -- T17 continued: WHERE the within-employer
posting decline sits. Splits the l09 design by firm characteristics from
the open register join.

Dimensions (all from open data; the SCB bulk has no size class -- that is
API-only -- so size uses the Monitor's ML-approved accumulated-hiring
proxy):
  A. Industry section (SNI Ng1): C manufacturing, G trade, J ICT,
     K finance, M professional services, N admin/support (incl. SNI 78),
     other-private. Public orgnr (prefix 2) kept as its own stratum.
  B. Firm age at ChatGPT (RegDatKtid): <10 years vs 10+ years.
  C. Size proxy: terciles of pre-2022 total distinct ads per firm.

Each cell re-runs the l09 Poisson DiD (firm x quartile + firm x month FE,
identification restriction, balanced zero-filled panel). Comparisons are
WITHIN each split's own sample; populations named in the output.

Output: tables/firm_heterogeneity.csv
Runtime: ~20-40 min (many medium fits); run in background.
"""

import importlib.util
import sys
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
_l09_spec = importlib.util.spec_from_file_location(
    "l09", REV / "local" / "l09_firm_within_did.py")
l09 = importlib.util.module_from_spec(_l09_spec)
_l09_spec.loader.exec_module(l09)
_cfg = l09._cfg

SECTION_MAP = {  # SNI2007 2-digit -> section label (the groups we report)
    **{f"{i:02d}": "C manufacturing" for i in range(10, 34)},
    **{f"{i:02d}": "G trade" for i in range(45, 48)},
    **{f"{i:02d}": "J ICT" for i in range(58, 64)},
    **{f"{i:02d}": "K finance" for i in range(64, 67)},
    **{f"{i:02d}": "M professional" for i in range(69, 76)},
    **{f"{i:02d}": "N admin-support" for i in range(77, 83)},
}


def load_register():
    with zipfile.ZipFile(l09.SCB_BULK) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reg = pd.read_csv(io.TextIOWrapper(f, encoding="cp1252"),
                              sep="\t", dtype=str,
                              usecols=["PeOrgNr", "Ng1", "RegDatKtid"])
    reg["orgnr"] = reg["PeOrgNr"].str[-10:]
    reg["sni2"] = reg["Ng1"].astype(str).str[:2]
    reg["section"] = reg["sni2"].map(SECTION_MAP).fillna("other-private")
    reg["regyear"] = pd.to_numeric(reg["RegDatKtid"].astype(str).str[:4],
                                   errors="coerce")
    reg["young_firm"] = reg["regyear"] >= 2013   # <10 years at ChatGPT
    reg = reg.drop_duplicates("orgnr").set_index("orgnr")
    return reg[["section", "young_firm"]]


def main():
    print("L9b: firm heterogeneity of the within-employer decline")
    cube = l09.load_cube()
    daioe = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    daioe["exposure_quartile"] = (daioe["exposure_quartile"].astype(str)
                                  .str.extract(r"Q(\d)").astype(int))
    daioe = daioe[["ssyk4", "exposure_quartile"]]
    reg = load_register()

    # attach firm strata to the cube
    cube["section"] = cube["orgnr"].map(reg["section"])
    cube.loc[cube["orgnr"].str.startswith("2"), "section"] = "public"
    cube["section"] = cube["section"].fillna("unmatched")
    cube["young_firm"] = cube["orgnr"].map(reg["young_firm"])
    pre_vol = (cube[cube["month"] < "2022-01"]
               .groupby("orgnr")["ads"].sum())
    terc = pre_vol.quantile([1 / 3, 2 / 3])
    size_map = pd.cut(pre_vol, [-np.inf, terc.iloc[0], terc.iloc[1], np.inf],
                      labels=["small", "mid", "large"])
    cube["size_proxy"] = cube["orgnr"].map(size_map)

    rows = []

    def run(sub, dim, level):
        es_dummy = []
        bal = l09.build_panel(sub, daioe, "ads")
        if bal["orgnr"].nunique() < 100:
            print(f"  [{dim}={level}] <100 firms, skipped")
            return
        try:
            import pyfixest as pf
            fit = pf.fepois(
                "n_ads ~ rb_x_high + gpt_x_high | fe_fq + fe_ft",
                data=bal, vcov={"CRV1": "orgnr"})
            for t in ("rb_x_high", "gpt_x_high"):
                rows.append({"dimension": dim, "level": level, "term": t,
                             "coef": fit.coef()[t], "se": fit.se()[t],
                             "pval": fit.pvalue()[t],
                             "n_firms": bal["orgnr"].nunique()})
            g = fit.coef()["gpt_x_high"]
            print(f"  [{dim}={level}] {bal['orgnr'].nunique():,} firms: "
                  f"gpt {g:+.4f}")
        except Exception as e:
            print(f"  [{dim}={level}] FAILED: {e}")

    for level in ("C manufacturing", "G trade", "J ICT", "K finance",
                  "M professional", "N admin-support", "other-private",
                  "public"):
        run(cube[cube["section"] == level], "section", level)
    for level, mask in (("young_lt10y", cube["young_firm"] == True),   # noqa: E712
                        ("older_10y+", cube["young_firm"] == False)):  # noqa: E712
        run(cube[mask], "firm_age", level)
    for level in ("small", "mid", "large"):
        run(cube[cube["size_proxy"] == level], "size_proxy", level)

    out = pd.DataFrame(rows)
    out.to_csv(_cfg.V2_TAB / "firm_heterogeneity.csv", index=False)
    print("Saved firm_heterogeneity.csv")


if __name__ == "__main__":
    main()

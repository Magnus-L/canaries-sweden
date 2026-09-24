#!/usr/bin/env python3
"""
15_within_employer_heterogeneity.py: where the within-employer decline in
exposed advertisements sits.

DESIGN
The design of 14 (PostRB x High and PostGPT x High, employer-by-quartile and
employer-by-month effects, the five-advertisement floor and the two-quartile
screen, a balanced zero-filled panel, Poisson, clustered by employer) is
re-estimated inside cells defined by employer characteristics from open data:
industry section from Statistics Sweden's business register (C manufacturing,
G trade, J information and communication, K finance, M professional services,
N administrative and support, other private, and public employers by
organisation-number prefix); employer age at the launch (registered from 2013,
so under ten years, against older); and size, proxied by terciles of the
employer's distinct advertisements before 2022, since the open register
carries no size class. Cells with fewer than 100 employers are skipped.

INPUTS   config.FIRM_CUBE, config.SCB_BULK (through 14);
         data/processed/daioe_quartiles.csv
OUTPUTS  output/results/firm_heterogeneity.csv
SERVES   Online Appendix V, Table A31 (through 16)
RUNTIME  about 5 minutes
"""

import sys
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

emp = sibling("14_within_employer")

SECTION_MAP = {  # SNI2007 2-digit -> section label (the groups we report)
    **{f"{i:02d}": "C manufacturing" for i in range(10, 34)},
    **{f"{i:02d}": "G trade" for i in range(45, 48)},
    **{f"{i:02d}": "J ICT" for i in range(58, 64)},
    **{f"{i:02d}": "K finance" for i in range(64, 67)},
    **{f"{i:02d}": "M professional" for i in range(69, 76)},
    **{f"{i:02d}": "N admin-support" for i in range(77, 83)},
}


def load_register():
    with zipfile.ZipFile(emp.SCB_BULK) as zf:
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
    print("Within-employer design by employer characteristics")
    cube = emp.load_cube()
    daioe = pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
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
        bal = emp.build_panel(sub, daioe, "ads")
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
    out.to_csv(config.RESULTS / "firm_heterogeneity.csv", index=False)
    print("Saved firm_heterogeneity.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
l04_public_yreg_check.py: the age pattern in Statistics Sweden's published
occupational employment, coded by the agency.

QUESTION
The register results classify employers from their 2019 education mix and
never code a worker's occupation. Statistics Sweden publishes employment
by occupation and age band as an official aggregate in which the agency
codes the worker, so any age pattern there cannot be an artefact of our
coding. Does it show the same direction?

WHAT IT BUILDS
Downloads the published table YREG54BAS for every year the agency has
released (through the downloader and parser of src/20, unchanged), merges
the four-digit occupations with the DAIOE quartiles, and for each
published age band computes the log employment gap between top-quartile
occupations and the rest, per year, and its change since 2022, the last
year before the launch. The output states the three caveats: the register
switches from RAMS to BAS at reference year 2022, on the treatment
boundary; the published series lags the employer declarations, so the
check reaches 2024 and not 2025; and the published bands (16-24, 25-34
and so on) are not the paper's.

INPUTS AND OUTPUTS
Reads the SCB API (cached in data/raw/scb_yreg54bas.json) and
data/processed/daioe_quartiles.csv. Writes
revision/tables/public_yreg_check.csv and public_yreg_summary.txt.

IN THE PAPER
Section 3 (16-24 is the band whose top-quartile gap falls most by 2024,
by 0.006) and Online Appendix III.5, Table tab:public_yreg (built by
script l10 from public_yreg_check.csv).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
# Import the revision's config explicitly by path: src/ also has a config.py, and
# whichever lands first on sys.path would shadow the other.
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
PROCESSED, RAW, V2_TAB = _cfg.PROCESSED, _cfg.RAW, _cfg.V2_TAB
sys.path.insert(0, str(REV.parent / "src"))   # for src/20's own config import

_spec = importlib.util.spec_from_file_location(
    "yreg20", REV.parent / "src" / "20_employment_age_yreg.py")
yreg20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(yreg20)


def main():
    print("L4: public YREG coverage-immune check (R2.2)")
    data = yreg20.download_scb_employment()
    emp = yreg20.process_employment(data)
    years = sorted(emp["year"].unique())
    print(f"  published years available: {years}")

    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)
    emp["ssyk4"] = emp["ssyk4"].astype(str).str.zfill(4)
    m = emp.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4",
                  how="inner")
    # exposure_quartile is stored as strings ("Q4 (highest)") in the
    # processed file, and newer pandas reads them as the Arrow-backed
    # 'str' dtype, which is not object. Match on the string content,
    # not on the dtype.
    m["high"] = (m["exposure_quartile"].astype(str)
                 .str.startswith("Q4").astype(int))

    # Young = the youngest published band containing 22-25 (SCB publishes
    # "16-24 years" / "25-34 years" bands); report both for transparency.
    bands = sorted(m["age_group"].unique())
    print(f"  published age bands: {bands}")

    grp = (m.groupby(["year", "age_group", "high"], observed=True)
           ["n_employed"].sum().reset_index())
    grp["ln_emp"] = np.log(grp["n_employed"])
    # Q4-vs-rest log gap per band-year, then its change from the last
    # pre-ChatGPT reference year (2022)
    wide = grp.pivot_table(index=["year", "age_group"], columns="high",
                           values="ln_emp").reset_index()
    wide["gap"] = wide[1] - wide[0]
    ref = wide[wide["year"] == 2022][["age_group", "gap"]].rename(
        columns={"gap": "gap_2022"})
    wide = wide.merge(ref, on="age_group", how="left")
    wide["dd_vs_2022"] = wide["gap"] - wide["gap_2022"]
    out = wide[["year", "age_group", "gap", "dd_vs_2022"]]
    out.to_csv(V2_TAB / "public_yreg_check.csv", index=False)

    lines = ["PUBLIC YREG CHECK (SCB-coded; immune to our cascade)",
             "=" * 56,
             f"years: {years}",
             "",
             "Q4-vs-rest log employment gap, change from 2022, by band:"]
    for band in bands:
        sub = out[(out["age_group"] == band) & (out["year"] > 2022)]
        for _, r in sub.iterrows():
            lines.append(f"  {band:>14} {int(r['year'])}: "
                         f"{r['dd_vs_2022']:+.4f}")
    lines += ["",
              "Caveats (state in the OA):",
              " - RAMS->BAS register switch at reference year 2022 sits on",
              "   the treatment boundary; SCB flags it as a series break.",
              " - Publication lag: the newest published year lags AGI; the",
              "   2025 occupation register will not exist before ~2027, so",
              "   this check covers the first problem year (2024), not 2025.",
              " - Annual frequency and published age bands (not 22-25)."]
    (V2_TAB / "public_yreg_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

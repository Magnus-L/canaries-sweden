#!/usr/bin/env python3
"""
l04_public_yreg_check.py -- T13/R2.2: the coverage-immune public-data check.

Extends src/20 (SCB YREG54BAS, published Yrkesregistret aggregates). The
revision's use of it is different from v1's: the published register is
coded BY SCB, not by our cascade, so any decline in exposed occupations
visible here cannot be an artefact of OUR coverage. This script:

  1. Re-downloads YREG54BAS taking WHATEVER years SCB has published (the
     v1 download stopped at 2024; the query already asks for all years,
     so a fresh pull picks up new releases automatically). Delete the
     cached data/raw/scb_yreg54bas.json to force a refresh.
  2. Computes, for the youngest published age band vs older bands, the
     log change in employment in Q4 vs Q1-Q3 occupations, per year --
     a public-data analogue of the register DiD at annual frequency.
  3. States the two structural caveats in the output: the RAMS->BAS
     switch at reference year 2022 (methodological break exactly at the
     treatment boundary), and publication lag (the newest year available
     lags the AGI window; 2025 will not exist before ~2027).

Reuses src/20's downloader and parser verbatim (imported), so the raw
pull and the JSON-stat2 handling stay identical to the submitted OA.

Output: tables/public_yreg_check.csv, tables/public_yreg_summary.txt
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
# Import the v2 config EXPLICITLY by path: src/ also has a config.py, and
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
    # 'str' dtype -- which is NOT == object. Match on the string content,
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

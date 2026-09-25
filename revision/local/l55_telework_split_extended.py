#!/usr/bin/env python3
"""
l55_telework_split_extended.py: the Dingel and Neiman teleworkability split
on the posting margin, re-estimated on the paper's current window.

WHY THIS EXISTS
OA II.3 and column (1) of Table tab:remote_measures (built by l53) took the
split from tables/telework_did_results.csv, which src/09 and
replication/2_postings/12 estimate on the SUBMITTED panel (October 2019 to
February 2026, postings_daioe_merged.csv, 26,672 occupation-months). Every
other posting estimate now runs on the extended panel built by l08
(January 2020 to June 2026, postings_daioe_merged_extended.csv, 28,084
occupation-months). This script moves the split onto that panel.

WHAT IT ESTIMATES (unchanged from replication/2_postings/12)
The Dingel and Neiman (2020) classification is carried O*NET-SOC -> SOC
2010 -> ISCO-08 -> SSYK 2012 with equal-weight averaging at each step, the
occupations are split at the median score, and specification (2),
ln(postings + 1) on occupation and month effects, PostRB x High and
PostGPT x High, is estimated by OLS in each half and on all occupations,
standard errors clustered by occupation. The functions are imported from
the replication script itself, so the specification cannot drift.

GATE
Before switching the input, the code is run on the submitted panel and
must reproduce tables/telework_did_results.csv (every column, every row;
counts exactly, floats to a relative 1e-10, since the least-squares solve
differs from the stored run only in the fourteenth digit). The script stops if it does not.

A CHECK ROW
On all occupations, the export also carries specification (2) with
ln(postings) as the outcome (the form of l08's baseline), which must
reproduce postings_extended_did.csv's OLS_ln gpt_x_high (-0.0593).

Run:  python3 revision/local/l55_telework_split_extended.py
Out:  revision/tables/telework_did_results_v3.csv
      ../canaries-sweden-paper/figures/figA_telework_robustness.png
      (OA Figure fig:posting_rivals, panel (b), redrawn on the current window
      by the replication script's own plotting function)
      (tables/telework_did_results.csv, the old export, is left untouched)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
REPL = ROOT / "replication" / "2_postings" / "12_telework_split.py"
PROCESSED = ROOT / "data" / "processed"
OLD_EXPORT = ROOT / "tables" / "telework_did_results.csv"
OUT = ROOT / "revision" / "tables" / "telework_did_results_v3.csv"
PAPER_FIG = ROOT.parent / "canaries-sweden-paper" / "figures"
L08_EXPORT = ROOT / "revision" / "tables" / "postings_extended_did.csv"

# Import the replication script as a module (its name starts with a digit).
sys.path.insert(0, str(REPL.parent))
_spec = importlib.util.spec_from_file_location("telework12", REPL)
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)


def mapping():
    """SSYK 2012 teleworkability scores via the replication crosswalk chain."""
    return tw.build_ssyk_telework(tw.load_dingel_neiman(), tw.load_soc_to_isco(),
                                  tw.load_isco_to_ssyk())


def main():
    ssyk_tw = mapping()
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")

    # ---- Gate: the old panel must reproduce the old export exactly ------
    old_panel = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    got = tw.run_did_by_telework(old_panel, daioe, ssyk_tw.copy())
    want = pd.read_csv(OLD_EXPORT)
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  want.reset_index(drop=True),
                                  check_exact=False, rtol=1e-10, atol=0, check_dtype=False)
    print("\nGATE PASSED: submitted panel reproduces telework_did_results.csv "
          "(integers exact, floats to a relative 1e-10, i.e. machine precision)")

    # ---- The current window ---------------------------------------------
    new_panel = pd.read_csv(PROCESSED / "postings_daioe_merged_extended.csv")
    res = tw.run_did_by_telework(new_panel, daioe, ssyk_tw.copy())
    # Panel (b) of the OA figure, same function, current window.
    tw.FIGURES = PAPER_FIG
    tw.plot_telework_comparison(res)
    res.insert(1, "outcome", "ln(ads+1)")
    res.insert(1, "panel", f"{new_panel.year_month.min()} to {new_panel.year_month.max()}")

    # Check row: all occupations, ln(ads) outcome, must equal l08's baseline.
    df = new_panel.copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df = df[df["n_ads"] > 0]
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["high"] = (df["exposure_quartile"] == "Q4 (highest)").astype(int)
    df["rb_high"] = ((df["date"] >= pd.Timestamp(tw.RIKSBANKEN_HIKE)) & (df["high"] == 1)).astype(int)
    df["gpt_high"] = ((df["date"] >= pd.Timestamp(tw.CHATGPT_LAUNCH)) & (df["high"] == 1)).astype(int)
    df["ln_ads"] = np.log(df["n_ads"])
    m = smf.ols("ln_ads ~ C(ssyk4) + C(year_month) + rb_high + gpt_high", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["ssyk4"]})
    chk = dict(group="All", panel=res.panel.iloc[0], outcome="ln(ads), check", n_occ=df.ssyk4.nunique(),
               n_obs=len(df), beta1_rb=m.params["rb_high"], se_rb=m.bse["rb_high"],
               p_rb=m.pvalues["rb_high"], beta2_gpt=m.params["gpt_high"],
               se_gpt=m.bse["gpt_high"], p_gpt=m.pvalues["gpt_high"])
    l08 = pd.read_csv(L08_EXPORT).set_index(["window", "estimator", "term"])
    ref = l08.loc[("extended_to_2026-06", "OLS_ln", "gpt_x_high"), "coef"]
    assert abs(chk["beta2_gpt"] - ref) < 1e-8, (chk["beta2_gpt"], ref)
    print(f"CHECK PASSED: all occupations, ln(ads): {chk['beta2_gpt']:.4f} = l08 {ref:.4f}")

    res = pd.concat([res, pd.DataFrame([chk])], ignore_index=True)
    res.to_csv(OUT, index=False)
    print(f"\nSaved {OUT.relative_to(ROOT)}")
    print(res.to_string())


if __name__ == "__main__":
    main()

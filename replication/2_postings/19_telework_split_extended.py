#!/usr/bin/env python3
"""
19_telework_split_extended.py: the Dingel and Neiman teleworkability split
of 12 on the window the paper's posting estimates use, January 2020 to June
2026.

WHAT IT ESTIMATES
Exactly what 12 estimates, through 12's own functions: the Dingel and
Neiman (2020) classification carried O*NET-SOC -> SOC 2010 -> ISCO-08 -> SSYK
2012 with equal-weight averaging at each step, occupations split at the
median score, and Equation (1) as ln(postings + 1) on occupation and month
effects, PostRB x High and PostGPT x High, by OLS in each half and on all
occupations, standard errors clustered by occupation. 12 runs it on the
submitted version's panel (October 2019 to February 2026); this script runs
it on the extended panel of 03, which every other posting estimate uses, and
draws panel (b) of Online Appendix Figure A2 from that run, replacing the
figure 12 wrote.

CHECKS
Before the input is switched, 12's functions are run on the submitted panel
and must reproduce 12's own result file (counts exactly, floats to a
relative 1e-10). On the extended panel, Equation (1) on all occupations with
ln(postings) as the outcome must reproduce the OLS baseline of 03
(postings_extended_did.csv) to 1e-8.

INPUTS   data/processed/postings_daioe_merged.csv (12's panel),
         postings_daioe_merged_extended.csv (03), daioe_quartiles.csv;
         output/results/telework_did_results.csv (12), postings_extended_did.csv (03)
OUTPUTS  output/results/telework_did_results_v3.csv;
         output/figures/figA_telework_robustness.png (panel (b) of Figure A2)
SERVES   Online Appendix II.3 (187 and 182 occupations; the post-launch
         coefficient of -0.216 in the non-teleworkable half) and column (1)
         of Table A4, through 22
RUNTIME  about 10 seconds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

PROCESSED, RESULTS = config.PROCESSED, config.RESULTS
tw = sibling("12_telework_split")


def main():
    print("Teleworkability split on the window to June 2026")
    ssyk_tw = tw.build_ssyk_telework(tw.load_dingel_neiman(), tw.load_soc_to_isco(),
                                     tw.load_isco_to_ssyk())
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")

    # ---- Gate: the submitted panel reproduces 12's result file ----------
    old = RESULTS / "telework_did_results.csv"
    if not old.exists():
        raise SystemExit(f"  run 12_telework_split.py first: {old} is missing")
    got = tw.run_did_by_telework(pd.read_csv(PROCESSED / "postings_daioe_merged.csv"),
                                 daioe, ssyk_tw.copy())
    want = pd.read_csv(old)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True),
                                  check_exact=False, rtol=1e-10, atol=0, check_dtype=False)
    print("  GATE: the submitted panel reproduces 12's telework_did_results.csv")

    # ---- The current window ---------------------------------------------
    new_panel = pd.read_csv(PROCESSED / "postings_daioe_merged_extended.csv")
    res = tw.run_did_by_telework(new_panel, daioe, ssyk_tw.copy())
    tw.plot_telework_comparison(res)      # panel (b) of Figure A2, current window
    res.insert(1, "outcome", "ln(ads+1)")
    res.insert(1, "panel", f"{new_panel.year_month.min()} to {new_panel.year_month.max()}")

    # Check row: all occupations, ln(ads), must equal 03's OLS baseline.
    df = new_panel.copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df = df[df["n_ads"] > 0].copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["high"] = (df["exposure_quartile"] == "Q4 (highest)").astype(int)
    df["rb_high"] = ((df["date"] >= pd.Timestamp(tw.RIKSBANKEN_HIKE)) & (df["high"] == 1)).astype(int)
    df["gpt_high"] = ((df["date"] >= pd.Timestamp(tw.CHATGPT_LAUNCH)) & (df["high"] == 1)).astype(int)
    df["ln_ads"] = np.log(df["n_ads"])
    m = smf.ols("ln_ads ~ C(ssyk4) + C(year_month) + rb_high + gpt_high", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["ssyk4"]})
    chk = dict(group="All", panel=res.panel.iloc[0], outcome="ln(ads), check",
               n_occ=df.ssyk4.nunique(), n_obs=len(df),
               beta1_rb=m.params["rb_high"], se_rb=m.bse["rb_high"], p_rb=m.pvalues["rb_high"],
               beta2_gpt=m.params["gpt_high"], se_gpt=m.bse["gpt_high"], p_gpt=m.pvalues["gpt_high"])
    l08 = pd.read_csv(RESULTS / "postings_extended_did.csv").set_index(["window", "estimator", "term"])
    ref = l08.loc[("extended_to_2026-06", "OLS_ln", "gpt_x_high"), "coef"]
    if abs(chk["beta2_gpt"] - ref) > 1e-8:
        raise SystemExit(f"  the all-occupation check {chk['beta2_gpt']:.6f} is not 03's {ref:.6f}")
    print(f"  CHECK: all occupations, ln(ads): {chk['beta2_gpt']:.4f} = 03's baseline")

    res = pd.concat([res, pd.DataFrame([chk])], ignore_index=True)
    out = RESULTS / "telework_did_results_v3.csv"
    res.to_csv(out, index=False)
    print(f"  Saved {out.name}")
    print(res.to_string())


if __name__ == "__main__":
    main()

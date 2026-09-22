#!/usr/bin/env python3
"""
46_wfh_horserace.py: the submitted occupation design with teleworkability
entered beside AI exposure.

QUESTION
A referee asked how much of the submitted employment result can be
explained by exposure to remote work. This script enters the Dingel and
Neiman teleworkability share beside the DAIOE quartile in one model, on
the submitted design's panel, per age band, and re-estimates the
DAIOE-only model on the same sample for comparison.

DESIGN
Panel: the withdrawn design's employer by exposure quartile by month
cells from the vintage-tagged panel with the vintages summed out,
employers with at least five person-months, balanced and zero-filled,
restricted to employers in both the top quartile and a lower one, one
panel per age band. Teleworkability of a cell is the employment-weighted
mean teleworkable share of its occupations over the months before the
launch, fixed thereafter. Specification: Poisson pseudo-maximum
likelihood with PostRB x High, PostGPT x High, PostRB x WFH and PostGPT x
WFH under employer-by-quartile and employer-by-month effects, standard
errors clustered by employer; then PostRB x High and PostGPT x High alone
on the same cells.

INPUTS AND OUTPUTS
Reads cache/panel_vintage.parquet (script 39), daioe_quartiles.dta and
dingel_neiman_ssyk4.dta. Writes to output_46/: wfh_horserace.csv,
wfh_daioe_only.csv and 46_summary.txt.

IN THE PAPER
Not quoted in the current manuscript. The design it runs on was withdrawn
after the backtest of Online Appendix IV.3; the teleworkability
comparison the paper reports is script 63's, on the 2019 firm-age
baseline (Online Appendix Table III.2, Panel B).
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_46"
OUT.mkdir(exist_ok=True)
CACHE = mc.PANEL_CACHE
WFH_PATH = mc.SHARE + r"\dingel_neiman_ssyk4.dta"

AGES = list(mc.AGE_GROUPS)
STEP1_MIN_CUMULATIVE = 5
TERMS = ["post_rb_x_high", "post_gpt_x_high",
         "post_rb_x_wfh", "post_gpt_x_wfh"]


def load_wfh():
    wfh = pd.read_stata(WFH_PATH) if WFH_PATH.endswith(".dta") else pd.read_csv(WFH_PATH)
    # accept either (ssyk4, teleworkable) or (ssyk4, telework_share)
    col = "teleworkable" if "teleworkable" in wfh.columns else "telework_share"
    wfh = wfh.rename(columns={col: "wfh"})[["ssyk4", "wfh"]]
    wfh["ssyk4"] = wfh["ssyk4"].astype(str).str.zfill(4)
    return wfh


def main():
    mc.Tee(OUT / "46_log.txt")
    print("=" * 70)
    print("46: WFH HORSE RACE (R1.7)")
    print("=" * 70)
    if not CACHE.exists():
        print("FATAL: run 39 first")
        sys.exit(1)

    panel = pd.read_parquet(CACHE)
    agg_occ = mc.collapse_vintage(panel)
    # Drop the vintage panel the moment the collapse has consumed it:
    # same rows, same numbers, roughly half the peak memory.
    del panel
    gc.collect()
    agg_occ["ssyk4"] = agg_occ["ssyk4"].astype(str).str.zfill(4)
    daioe = mc.load_daioe()
    wfh = load_wfh()

    occ = agg_occ.merge(daioe, on="ssyk4", how="inner").merge(
        wfh, on="ssyk4", how="left")
    occ["wfh"] = occ["wfh"].fillna(occ["wfh"].mean())
    size = occ.groupby("employer_id")["n_emp"].sum()
    occ = occ[occ["employer_id"].isin(
        size[size >= mc.MIN_EMPLOYER_SIZE].index)]

    # Cell-level WFH: employment-weighted mean over the PRE period, fixed.
    pre = occ[occ["year_month"] < mc.CHATGPT_YM]
    cell_wfh = (pre.assign(wxn=pre["wfh"] * pre["n_emp"])
                .groupby(["employer_id", "exposure_quartile"],
                         observed=True)
                .agg(wxn=("wxn", "sum"), n=("n_emp", "sum")).reset_index())
    cell_wfh["cell_wfh"] = cell_wfh["wxn"] / cell_wfh["n"]
    cell_wfh = cell_wfh[["employer_id", "exposure_quartile", "cell_wfh"]]

    agg = (occ.groupby(["employer_id", "year_month",
                        "exposure_quartile", "age_group"], observed=True)
           ["n_emp"].sum().reset_index())
    all_months = sorted(agg["year_month"].unique())

    race_rows, only_rows = [], []
    for age in AGES:
        print(f"\n--- {age} ---")
        sub = agg[agg["age_group"] == age]
        cum = sub.groupby("employer_id")["n_emp"].sum()
        sub = sub[sub["employer_id"].isin(
            cum[cum >= STEP1_MIN_CUMULATIVE].index)]
        bal = mc.add_treatment(mc.balance_panel(sub, all_months))
        bal = bal.merge(cell_wfh, on=["employer_id", "exposure_quartile"],
                        how="left")
        bal["cell_wfh"] = bal["cell_wfh"].fillna(bal["cell_wfh"].mean())
        bal["post_rb_x_wfh"] = bal["post_rb"] * bal["cell_wfh"]
        bal["post_gpt_x_wfh"] = bal["post_gpt"] * bal["cell_wfh"]
        print(f"  {len(bal):,} cells")

        res = mc.run_fepois_multi(bal, OUT, tag=f"race_{age}", terms=TERMS)
        if not res.empty:
            res["age_group"] = age
            race_rows.append(res)

        # DAIOE-only on the SAME sample (the Azar-style comparison)
        res0 = mc.run_fepois(bal, OUT, tag=f"only_{age}")
        if not res0.empty:
            res0["age_group"] = age
            only_rows.append(res0)

    if race_rows:
        pd.concat(race_rows).to_csv(OUT / "wfh_horserace.csv", index=False)
    if only_rows:
        pd.concat(only_rows).to_csv(OUT / "wfh_daioe_only.csv", index=False)

    if race_rows and only_rows:
        race = pd.concat(race_rows)
        only = pd.concat(only_rows)
        lines = ["WFH HORSE RACE -- gamma2 (PostGPT x High)", "=" * 50,
                 f"{'age':>6} {'with WFH terms':>15} {'DAIOE only':>12}"]
        for age in AGES:
            g_r = race[(race["age_group"] == age)
                       & (race["term"] == "post_gpt_x_high")]["coef"]
            g_o = only[(only["age_group"] == age)
                       & (only["term"] == "post_gpt_x_high")]["coef"]
            if len(g_r) and len(g_o):
                lines.append(f"{age:>6} {g_r.iloc[0]:>+15.4f} "
                             f"{g_o.iloc[0]:>+12.4f}")
        (OUT / "46_summary.txt").write_text("\n".join(lines))
        print("\n".join(lines))


if __name__ == "__main__":
    main()

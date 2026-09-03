#!/usr/bin/env python3
"""
44_decile_gradient.py -- T11/R1.4 (MONA run M5): employment DiD by
exposure DECILE.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Requires output_39/panel_vintage.parquet.
======================================================================

R1 asks why the exposure measure is so discrete; Kallberg's public-data
thesis uses the top decile where the paper uses the top quartile. This
script assigns each occupation its UNWEIGHTED DAIOE decile (the same
convention as the quartiles: each SSYK4 counts once; deciles computed
from pctl_rank_genai in the shared daioe_quartiles.csv) and estimates
ONE Poisson model per age group with eighteen interactions
(REFERENCE = MEDIAN DECILE 5; decile 1 is construction-heavy and its
own rate-cycle crash would contaminate every contrast against it):

    d in {1..10}\{5}:  PostRB x 1[decile=d],  PostGPT x 1[decile=d]

via r_fepois_multi.R, with employer x decile and employer x month FE.
Identification restriction: employers observed in decile 10 AND some
lower decile (the top-vs-rest logic of the quartile design, one level
finer). Balanced zero-filled panel across the employer's observed
deciles. Pre-committed read (plan T11): the gradient is reported as
estimated; the monotonicity statement quotes deciles 8-10 jointly, and
the local postings analogue (l03: minimum at d8) is quoted beside it.

Output (output_44/): decile_pooled.csv (age x decile x period),
44_summary.txt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_44"
OUT.mkdir(exist_ok=True)
CACHE = HERE / "output_39" / "panel_vintage.parquet"

AGES = ["22-25", "26-30", "50+"]     # headline + neighbours; extend if fast
STEP1_MIN_CUMULATIVE = 5


def load_daioe_deciles():
    daioe = pd.read_csv(mc.DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    if "pctl_rank_genai" not in daioe.columns:
        raise RuntimeError("daioe_quartiles.csv lacks pctl_rank_genai; "
                           "upload the full file from data/processed/")
    occ = daioe.drop_duplicates("ssyk4")[["ssyk4", "pctl_rank_genai"]].copy()
    occ["decile"] = pd.qcut(occ["pctl_rank_genai"], 10, labels=False) + 1
    return occ[["ssyk4", "decile"]]


def balance_panel_decile(sub, all_months):
    """Balanced panel over the employer's observed deciles; requires
    decile 10 plus at least one lower decile."""
    emp_d = sub[["employer_id", "decile"]].drop_duplicates()
    top = set(emp_d.loc[emp_d["decile"] == 10, "employer_id"])
    low = set(emp_d.loc[emp_d["decile"] < 10, "employer_id"])
    emp_d = emp_d[emp_d["employer_id"].isin(top & low)]
    months_df = pd.DataFrame({"year_month": sorted(all_months)})
    # Guarantee one row per (employer, decile, month) before the merge --
    # upstream aggregation already ensures this in production, but the
    # balance step must not silently duplicate cells if it ever does not.
    cell = (sub.groupby(["employer_id", "decile", "year_month"],
                        observed=True)["n_emp"].sum().reset_index())
    bal = (emp_d.assign(_k=1).merge(months_df.assign(_k=1), on="_k")
           .drop(columns="_k")
           .merge(cell, on=["employer_id", "decile", "year_month"],
                  how="left"))
    bal["n_emp"] = bal["n_emp"].fillna(0).astype(int)
    return bal


def main():
    mc.Tee(OUT / "44_log.txt")
    print("=" * 70)
    print("44: EMPLOYMENT DECILE GRADIENT (R1.4)")
    print("=" * 70)
    if not CACHE.exists():
        print("FATAL: run 39 first")
        sys.exit(1)

    panel = pd.read_parquet(CACHE)
    agg = mc.collapse_vintage(panel)
    dec = load_daioe_deciles()
    agg["ssyk4"] = agg["ssyk4"].astype(str).str.zfill(4)
    agg = agg.merge(dec, on="ssyk4", how="inner")
    size = agg.groupby("employer_id")["n_emp"].sum()
    agg = agg[agg["employer_id"].isin(
        size[size >= mc.MIN_EMPLOYER_SIZE].index)]
    agg = (agg.groupby(["employer_id", "year_month", "decile", "age_group"],
                       observed=True)["n_emp"].sum().reset_index())
    all_months = sorted(agg["year_month"].unique())

    DECS = [d for d in range(1, 11) if d != 5]   # median decile = reference
    terms = [f"{p}_d{d}" for d in DECS for p in ("rb", "gpt")]
    all_rows = []
    for age in AGES:
        print(f"\n--- {age} ---")
        sub = agg[agg["age_group"] == age]
        cum = sub.groupby("employer_id")["n_emp"].sum()
        sub = sub[sub["employer_id"].isin(
            cum[cum >= STEP1_MIN_CUMULATIVE].index)]
        bal = balance_panel_decile(sub, all_months)
        bal["post_rb"] = (bal["year_month"] >= mc.RIKSBANK_YM).astype(int)
        bal["post_gpt"] = (bal["year_month"] >= mc.CHATGPT_YM).astype(int)
        for d in DECS:
            isd = (bal["decile"] == d).astype(int)
            bal[f"rb_d{d}"] = bal["post_rb"] * isd
            bal[f"gpt_d{d}"] = bal["post_gpt"] * isd
        bal["fe_emp_bin"] = (bal["employer_id"].astype(str) + "_"
                             + bal["decile"].astype(str))
        bal["fe_emp_t"] = (bal["employer_id"].astype(str) + "_"
                           + bal["year_month"])
        print(f"  {len(bal):,} cells, "
              f"{bal['employer_id'].nunique():,} employers")
        res = mc.run_fepois_multi(bal, OUT, tag=f"dec_{age}", terms=terms)
        if not res.empty:
            res["age_group"] = age
            all_rows.append(res)

    if all_rows:
        out = pd.concat(all_rows)
        out["period"] = np.where(out["term"].str.startswith("rb"),
                                 "post_riksbank", "post_chatgpt")
        out["decile"] = out["term"].str.extract(r"d(\d+)").astype(int)
        out.to_csv(OUT / "decile_pooled.csv", index=False)
        g = out[(out["period"] == "post_chatgpt")
                & (out["age_group"] == "22-25")].sort_values("decile")
        lines = ["DECILE GRADIENT, 22-25, post-ChatGPT (Poisson)", "=" * 40]
        for _, r in g.iterrows():
            lines.append(f"  d{int(r['decile']):>2}: {r['coef']:+.4f} "
                         f"(SE {r['se']:.4f})")
        (OUT / "44_summary.txt").write_text("\n".join(lines))
        print("\n".join(lines))


if __name__ == "__main__":
    main()

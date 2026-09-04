#!/usr/bin/env python3
"""
42_frozen_cohort.py -- T3/E5 (MONA run M3): the coverage-immune cohort.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY.
  Own SQL pull (force_cascade); cached in output_42/.
======================================================================

Design. Freeze BOTH membership and exposure at the 2021-2023 registers:
the cohort is every worker who holds a cascade code (Individ_2023/22/21),
and their DAIOE quartile is fixed by that code for the WHOLE window
2019-2025 -- including pre-2023 years, which under the production
pipeline use own-year codes. One assignment rule, one closed population.

Under this design the editor's mechanical story cannot operate: a newly
hired worker without a code after 2023 is outside the cohort in every
period, so differential nonmatching of new hires cannot move measured
employment; and no worker's quartile can change mid-panel, so
misclassification drift cannot either. What remains identified is the
employment path OF THE FROZEN COHORT across exposure quartiles within
employers -- the separations-plus-rehiring margin for the coded stock.

Read side by side with the baseline: baseline (hiring in) vs frozen
cohort (stock out). If the 22-25 decline is real, the frozen cohort
shows it too, attenuated by construction; if it is a coverage artefact,
the frozen cohort is flat.

Output (output_42/): frozen_pooled.csv, frozen_es.csv, 42_summary.txt,
panel_frozen.parquet (cache).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_42"
OUT.mkdir(exist_ok=True)
mc.CACHE_DIR.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR / "panel_frozen.parquet"

AGES = list(mc.AGE_GROUPS)
STEP1_MIN_CUMULATIVE = 5


def main():
    mc.Tee(OUT / "42_log.txt")
    print("=" * 70)
    print("42: FROZEN COHORT (coverage-immune; E5)")
    print("=" * 70)

    conn = None if CACHE.exists() else mc.connect()
    panel = mc.pull_panel(range(2019, 2026), conn, CACHE,
                          force_cascade=True)
    agg = mc.collapse_vintage(panel)
    agg = mc.merge_daioe_and_filter(agg, mc.load_daioe())
    agg = mc.aggregate_to_quartile(agg)
    all_months = sorted(agg["year_month"].unique())

    pooled_rows, es_frames = [], []
    for age in AGES:
        print(f"\n--- {age} ---")
        sub = agg[agg["age_group"] == age]
        cum = sub.groupby("employer_id")["n_emp"].sum()
        sub = sub[sub["employer_id"].isin(
            cum[cum >= STEP1_MIN_CUMULATIVE].index)]
        bal = mc.add_treatment(mc.balance_panel(sub, all_months))
        bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
        print(f"  {len(bal):,} cells, "
              f"{bal['employer_id'].nunique():,} employers")

        pres = mc.run_fepois(bal, OUT, tag=f"frz_{age}")
        for _, r in pres.iterrows():
            pooled_rows.append({"age_group": age, **r.to_dict()})
        eres = mc.run_fepois_es(bal, OUT, tag=f"frz_es_{age}")
        if not eres.empty:
            eres["age_group"] = age
            es_frames.append(eres)

    pd.DataFrame(pooled_rows).to_csv(OUT / "frozen_pooled.csv", index=False)
    if es_frames:
        pd.concat(es_frames).to_csv(OUT / "frozen_es.csv", index=False)

    poi = pd.DataFrame(pooled_rows)
    g2 = poi[poi["term"] == "post_gpt_x_high"]
    lines = ["FROZEN COHORT -- gamma2 by age (Poisson)", "=" * 40]
    for _, r in g2.iterrows():
        lines.append(f"  {r['age_group']:>6}: {r['coef']:+.4f} "
                     f"(SE {r['se']:.4f}, p {r['pvalue']:.4f})")
    lines.append("")
    lines.append("Read: attenuation vs baseline is expected (the hiring")
    lines.append("margin is excluded by construction); a FLAT 22-25 path")
    lines.append("here with a steep baseline points at the hiring margin,")
    lines.append("not at coverage.")
    (OUT / "42_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

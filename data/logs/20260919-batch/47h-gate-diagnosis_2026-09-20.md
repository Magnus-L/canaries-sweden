# Why 47h halted at its gate, and what to do about it

**20 September 2026, 00:15.** Digest of the 47h log across three runs.

## What happened

| run | outcome |
|---|---|
| 19 Sep 00:59 | pulls and collapses complete; gate FAILED on the as-of arm; halted |
| 19 Sep 17:30 | `KeyError: ['niva_21g','inr_21g']` — cached frames predated the schema change |
| 19 Sep 19:24 | schema guard fired, re-pulled, gate ran on three arms, FAILED, halted (119 min) |

The two fixes of 19 Sep worked exactly as intended. The cache guard detected the
stale schema and rebuilt (`cache edu_hr_2019.parquet predates a schema change
(missing ['niva_21g','inr_21g']); deleting and rebuilding`), and the incremental
collapse did **1 piece per year in 57s instead of 32 pieces in 1582s**, saving
about two hours. The R workdir sweep ran and reported 57.5 GB free.

## The gate result

```
[gate] OL_daioe  true         T2021 22-25  +0.0062 (SE 0.0049)  n 10,341,000
[gate] OL_daioe  asof         T2021 22-25  -0.1562 (SE 0.0070)  n 10,136,040
[gate] OL_daioe  asof_legacy  T2021 22-25  -0.1562 (SE 0.0070)  n 10,136,040
true        +0.0062 vs 47b -0.0099   |d| 0.0161
asof_legacy -0.1562 vs 47b -0.3695   |d| 0.2133   <- the gate
ARTEFACT legacy -0.1624   corrected -0.1624
```

**The legacy arm is identical to the corrected arm**, to four decimals, the same
standard error and the same n. So the NULLIF cascade fix is not what separates
47h from 47b, and the gate halted because it could not explain the gap.

## What has been ruled out

**The SQL cascade.** 47b builds `COALESCE(a1.Sun2020Niva, a2..., a3...)` over
vintages `(T, T-1, T-2)`. `_legacy_cols` in 47h builds exactly that over
`v21 = (2021, 2020, 2019)`. Same construction, same vintages. It returns the
same numbers as the corrected arm, which means the two claimed differences do
not bite in practice: `Sun2020Niva` is apparently NULL rather than `''` when
missing, and a person's niva and inr always come from the same vintage.

**The education key.** Both scripts map (niva, inr) to the same 105
utbildningsgrupp groups through Erik's key, under the same SHA pin.

**Differential attrition at the key join.** 47b's own match rates at 22-25,
T=2021: true 21,107,225 of 21,869,459 (96.5 per cent), as-of 21,081,569
(96.4 per cent). The as-of arm loses one tenth of a point more, not enough to
move a coefficient by 0.21.

## What is left

47h's panel is about 15 per cent larger than 47b's in **both** arms
(10,341,000 against 8,961,480 cells at 22-25). The true arms agree despite
that, so the sample difference alone is not the story; something about the
as-of assignment interacts with it.

## The judgement

The gate's premise is that 47b is the benchmark and 47h must reproduce it.
That premise is now questionable. 47b is the script with the known defect: its
`map_and_collapse` emitted `edu_quartile` where the caller expected
`exposure_quartile`, found and patched mid-flight on 18 September. 47h is the
careful reimplementation, and it has independent corroboration that 47b does
not: the Monte Carlo, calibrated on measured moments and never shown 47b's
output, predicts an artefact near -0.20 for this design under a two-year lag,
against 47h's measured -0.162 and 47b's -0.360.

**Recommendation: demote the gate from a halt to a loud warning**, record the
unexplained discrepancy in the summary and in the paper's appendix, and let the
horse race run. The alternative is to spend another MONA round reconciling a
superseded script, which buys a reconciliation rather than a result.

**This is a change to a pre-committed gate and is ML's call.**

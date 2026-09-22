# Lane 24: script 73, Part B (the credit channel) on the 2019 balance sheet

Filed 22 September 2026 from `~/Downloads/MyFiles949/` (exported 09:49). Nine files:
`73_summary.txt`, `credit_test.csv`, `industry_fe.csv` (baseline rows only; Part A was not
re-run), and six covariance files `vcov_r73_{base,levbase,lev}_{22-25,26-30}.csv`. Every filed
copy was hash-checked against the original before the original was deleted.

## What ran

Script 73 at commit `0639a1f`, `CANARIES_73_PARTS=B`. The code review of 22 September
(reviewer 2, F1 and F2) found that the leverage loader filtered Serrano's `BSLSLUT` with
`pd.to_numeric` on a date column, matched nothing, and let the credit test run on whichever
accounting year came first for each firm. The loader now parses the date and keeps the latest
close inside 2019 (`leverage year filter: BSLSLUT in 2019, 581,879 of 9,681,255 balance-sheet
rows kept`, 561,460 firms), and Part B fits the baseline on its own sample, the employers with a
2019 balance sheet, before the leverage split enters.

## Numbers

Leverage is 1 minus equity over assets. Median split at 0.691 (22-25 panel) and 0.688
(26-30); 92,213 of 111,459 and 102,719 of 128,193 panel employers carry the covariate
(82.7 and 80.1 per cent).

| term | 22-25 | 26-30 |
|---|---|---|
| Full-panel baseline (lane 19) | -0.0509 (0.0122) | -0.0437 (0.0079) |
| Baseline on the balance-sheet sample | -0.0695 (0.0117) | -0.0516 (0.0088) |
| Exposure step, less leveraged half | -0.0875 (0.0175) | -0.0834 (0.0144) |
| Additional in the more leveraged half | +0.0326 (0.0267) | +0.0583 (0.0199) |
| Exposure step averaged over the halves (from the vcov) | -0.0712 (0.0116) | -0.0542 (0.0089) |
| Leverage x young, all firms in the sample | -0.0054 (0.0087) | -0.0147 (0.0072) |

## Read rule, fixed before the run

The exposure step with leverage in is compared with the same-sample baseline: MONETARY if it
loses more than half, AI SURVIVES otherwise; the leverage x young term is reported for both
bands whatever it shows. Verdict: AI SURVIVES in both bands (the averaged step is 102 and 105
per cent of the same-sample baseline). The credit channel that does operate, leverage x young,
reaches 26-30 (t = 2.1) and not 22-25 (t = 0.6). The exposure step is larger in the less
leveraged half of exposed employers, the opposite of what a credit story needs.

## Against lane 19

Lane 19's Part B (`round3_20260921-lane19-final/credit_test.csv`: exposure -0.0770 / -0.0683,
leverage x young -0.0125 / -0.0225) used an unknown accounting year per firm and had no
same-sample baseline; it is superseded and must not be quoted. Lane 19's Part A (industry x age
x month: 88 and 48 per cent retained) and Part C stand.

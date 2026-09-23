# Lane 28b and lanes 29b, 29c, 29d, exported 23 September 2026 06:55, filed 07:05

Handed over as `~/Downloads/MyFiles655`, fifty-eight files, every one hash-checked against the
filed copy before the original was deleted.

**One flat folder, so two summaries survived and three did not.** MONA's `output_82b`,
`output_82c`, `output_83b`, `output_83c` and `output_83d` were copied into a single directory,
where `82_summary.txt` and `83_summary.txt` collide. The `82_summary.txt` here is **Part B's**
(runtime 109.7 min); Part C's is already filed at `round3_20260922-2327-lane28c/` and is not
lost. The `83_summary.txt` here is **Part C's** (69.4 min). **Script 83's Part B and Part D
summaries are gone and are owed**; their CSVs are all here, so the verdicts below were read off
the exports rather than off the script's own text, which is not how the read rules are meant to
be settled.

**The "NOT RUN" banner at the top of `83_summary.txt` is a within-job artefact.** Part A did not
run in that slot, so that job could not print the first stage. The first stage was settled in
the 22:33 export (`round3_20260922-2333-lane29a/`): THE FIRST STAGE REPRODUCES, +20.87 points
against the education route's +21.54 at the firm level and +22.53 against +23.50 at the
individual level. Lanes 28 and 29 may be quoted.

## Lane 28b, script 82 Part B: the headline and the age profile

**Verdict 1, the adoption step at 22-25: REPRODUCES.** **Verdict 2, the age profile: THE PROFILE
REPRODUCES.** Both on the primary score (`uniform3`, backward cascade, floor of five).

| | occupation route | education route |
|---|---|---|
| adoption step, 22-25 | **-0.0578 (0.0155)** t -3.73 | -0.0408 (0.0150) |
| adoption step, 26-30 | **-0.0482 (0.0104)** t -4.66 | -0.0394 (0.0102) |
| 50+ against 41-49 | **+0.0675 (0.0063)** | +0.0589 (0.0062) |
| firms, 22-25 / 26-30 | 104,217 / 117,090 | 111,459 / 128,193 |

Every estimate is larger than the education route's on a near-identical standard error, which is
what removing a smoothing layer should do. The profile, lowest first: 22-25 -0.0192 (0.0125),
26-30 -0.0097 (0.0086), 41-49 reference, 35-40 +0.0180 (0.0053), 31-34 +0.0181 (0.0065), 50+
+0.0675 (0.0063), on 153,845 employers.

The nine stock fits agree with each other to the third decimal: `mixed43` -0.0586 (0.0155),
`four_only` -0.0587 (0.0155), floor of one -0.0579, floor of three -0.0575, the forward cascade
-0.0581. **The scoring arm does not matter here**, and the reason is in the coverage: under
`mixed43` only 0.25 per cent of incumbents fall back to the three-digit book, so the three arms
are nearly the same score. 6.3 per cent of coded incumbents come from a year before 2019 on the
backward arm, 11.3 per cent on the forward arm.

**The 82c straggler.** `vcov_s82_hires_22_25.csv` is here and completes the 22:27 lane-28c
export. The eight other 82c files in this folder (`occ_route_flows/gender/vintage/ssyk3_book`
and four `vcov_s82_*`) are byte-identical duplicates of the copies already filed there, checked
on filing.

## Lane 29b, script 83 Part B: the reference window, the pre-launch drift, the clustering

No summary. Read off `occ_rest_window.csv`, `occ_rest_drift.csv` and `occ_rest_cluster.csv`.

**Rule 4, the reference window: agrees in direction, and is sharper than the education route's.**
The level after adoption against the pre-hike months: 22-25 **-0.0419 (0.0189)** t -2.22, 26-30
**-0.0298 (0.0123)** t -2.43. Both negative, as the rule asks. Both are also distinguishable from
zero, where neither education-route figure was (-0.0194 (0.0184) and -0.0172 (0.0121)); the rule
sets no significance test, so this is not a verdict, but it is worth the paper's saying.

**Rule 5, the pre-launch drift: the same pattern as the education route.** 22-25 trend +0.00038
(0.00077), FLAT within two of its own standard errors; 26-30 +0.00161 (0.00044), NOT flat at 3.7
standard errors. The education route is flat at 22-25 (+0.0006 (0.0008)) and not flat at 26-30
(+0.0019 (0.0004)), so this is agreement and not a defect of this route.

**Rule 6, the clustering.** The four-decimal gate PASSES on every row: `coef_match_4dp` is True
throughout, so this is inference only. On the completed three-digit key (260 clusters at 22-25,
263 at 26-30):

| | employer SE | industry SE | t on industry |
|---|---|---|---|
| 22-25 adoption step (-0.0578) | 0.0155 | 0.0307 | **-1.88** |
| 26-30 adoption step (-0.0482) | 0.0104 | 0.0208 | -2.32 |
| female differential (-0.0858) | 0.0142 | 0.0174 | **-4.95** |
| young men (-0.0171) | 0.0179 | 0.0352 | -0.48 |

**The rule as written is not met in full**: it asks that the 22-25 step AND the female
differential both stay distinguishable from zero at five per cent under industry clustering, and
the 22-25 step sits at t -1.88. **This is the education route's own result, not a new weakness**
of the occupation route: there the same step gave t -1.35 on an industry SE of 0.0302. The
occupation route is the closer of the two to the five per cent line, and the female differential
survives comfortably on both. The paper already reports the youngest band's step as not
distinguishable from zero once common industry disturbances are allowed; that sentence stands
unchanged on the new route and must be kept.

## Lane 29c, script 83 Part C: the industry test and the credit test

**Verdict 7: THE STEP SURVIVES AT 22-25 AND NOT AT 26-30.** With industry by age band by month
absorbed, on the same firms in both fits: 22-25 -0.0579 -> -0.0388, **67 per cent retained**
(education route 85); 26-30 -0.0483 -> -0.0152, **31 per cent retained** (education route 47,
which would not itself pass the 50 per cent rule). 103,728 and 116,378 employers, 1.7 and 1.9
per cent of them coded from a source other than `Ftg_2019`, in 259 and 262 industry groups.

**Verdict 8: THE STEP IS NOT A CREDIT EFFECT.** On the balance-sheet sample (Serrano, no FEK
table answered the probe; leverage = 1 - EKSU/TILLGSU on 2019 balance sheets, 561,460 firms,
covering 85.0 and 84.3 per cent of the two panels): 22-25 baseline -0.0887 (0.0116), with the
leverage split -0.1062 (0.0174), **120 per cent** of it; 26-30 -0.0572 (0.0088) to -0.0925
(0.0147), **162 per cent**. The exposed-and-levered term is +0.0319 (0.0267) at 22-25 and
+0.0650 (0.0200) at 26-30. The full-panel baselines, -0.0658 (0.0125) and -0.0498 (0.0080),
reproduce lane 28's quantity, so the two lanes fit the same panel.

**The balance-sheet sample is limited companies.** The public sector and the unincorporated are
not in it, and the paper must say so rather than leave it implicit.

## Lane 29d, script 83 Part D: the firm-size robustness

No summary. Read off `occ_rest_reliability.csv` and `occ_rest_size.csv`.

**Rule 9, first half: PASSES.** At a floor of sixty incumbent person-months (five workers
employed all year) the step at 22-25 is **-0.0537 (0.0169)** against the reported -0.0578
(0.0155): the same sign, and 0.0041 apart, well inside one reported standard error. 52,333 of
104,217 employers survive the floor. At 26-30 it is -0.0498 (0.0110) against -0.0482.

**Rule 9, second half: no size group carries the step at 22-25**, but read the cut with care.
**The tercile cut degenerated: `tercile_1` is empty and produced no fit.** The incumbent-count
distribution is so skewed that the lower cut point falls at one incumbent, so what the export
calls terciles is a two-way split: **1 to 3 incumbents (170,488 employers) and 4 or more
(91,601)**. On that split the step at 22-25 is **-0.0578 (0.0223)** in the small group and
**-0.0569 (0.0165)** in the large one, which is as clean an answer to the concern as the data
can give. At 26-30 it divides: **+0.0286 (0.0183)** in the small group against **-0.0499
(0.0108)** in the large, so the older young band's step comes from employers with four or more
incumbents. Missing `vcov_s83_size_tercile_1_*` files are therefore expected and are not owed.

**The reliability table, descriptive and never to be used as a correction.** Between-firm
variance 318.7 net of 28.6 of sampling noise, within-firm 422.0, so the net between share is
**0.43**. A one-incumbent score has reliability 0.43 and the median employer, with two coded
incumbents, 0.60. **106,401 employers, 40.6 per cent of them, sit below a reliability of one
half, and they hold 3.0 per cent of incumbent employment.** Weighted by employment the picture
reverses: the tenth percentile of incumbent employment sits at five incumbents (0.79) and the
median at 645 (0.998). The thin scores are numerous and tiny, which is the answer the appendix
should give, and it should give it in both units rather than one.

## What is still owed

1. `output_83b/83_summary.txt` and `output_83d/83_summary.txt`. The CSVs are complete, but the
   verdict lines on rules 4, 5, 6 and 9, and Part D's NOTES (the share a recut floor-of-sixty
   quartile would have moved, the tercile ranges, and the `D/tercile_1/no employer` failure)
   exist only in those files.
2. Nothing else. `vcov_s82_vintage_asof_2021.csv` does not exist (that arm scored zero
   employers, recorded in the 28c summary) and neither do the `tercile_1` covariance files.

# MONA trip — EL67898 revision, round 1

Staged 3 Sep; destination rebuilt 4 Sep. **16 uploadable files**; `UPLOAD.md` stays local
(`.md` is not an allowed portal format). Largest file is 17 KB against a 10 MB cap.

## 0. A new, project-named folder

The v1 work lives in `Lydia P1207\CANARIES\`. Two different things are tangled there, and the
distinction matters.

The **scripts** are Lydia's, February to May 2026: 44 files of mixed vintage in `Code\Python\`
(`latest.py`, `gendercheck.py`, `gendercheck2.py`, `gendercheck3.py`,
`14_without_regression.py`). Written under deadline, and no criticism of anyone.

The **folder structure** is ours, from 7 July 2026 — `Code\Python`, `Code\R`, `Input\`,
`All_output\`. That was a sensible pass and it is most of the way to a good layout. But it was
left half-finished and unrecorded: `Input\` was created and never filled, so the shared input
files still sit in `All_output\` beside the results, and no note was written. Two months later the
revision code was built against a path that had not existed since July, which would have killed the
round on the first script. The lesson is not about Lydia's code; it is that **a structure pass that
is not written down is a defect waiting on a delay.**

Also still there: one stale `.pyc` from April, and 3.3 GB of dead cache (`raw_panel.parquet`
1.04 GB, `flows_nonzero_cells.csv` 2.2 GB).

**Nothing in v2 reads anything from it.** Verified by grep: the only external reads in the whole
battery are `daioe_quartiles.csv`, the new `dingel_neiman_ssyk4.csv`, and the SQL database. The
canary gate pulls 2019–2025 fresh from `monasql.micro.intra` and touches no v1 output. The cost of
starting clean is therefore one 14 KB file, and that file is hash-verified on arrival.

Destination, under the **group convention decided 4 Sep 2026** — every researcher one folder at
`P1207_Gem` root, every project one main owner, the project living in its owner's folder. Magnus
owns canaries (he runs all revision empirics), so it sits beside `proworker-gov`:

    \\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\
      input\                    daioe_quartiles.dta, dingel_neiman_ssyk4.dta
      round1_EL67898\           mona_common.py, 39..46, run_all_mona.py,
                                r_fepois*.R, MANIFEST.txt
                                output_39\ .. output_46\  (created on the run)

`Lydia P1207\CANARIES\` is left untouched as the v1 archive; `mona_common.V1_ARCHIVE` records the
path so provenance stays traceable.

Three mechanics new on 4 Sep, all tested:

- **Provenance stamps.** Every log opens `run by <MONA account> at <time> | <script>`, and every
  stage appends one line to `canaries-sweden\RUNLOG.txt` (when, who, what, exit, minutes). That is
  the who-touched-my-project convention, implemented rather than hoped for.
- **Storage discipline, implemented for the first time.** All expensive pulls cache under ONE
  disposable `cache\` (the panel, the frozen panel, both dual panels) — never in a results folder.
  Every run ends with a storage report. At close of round, after exports are out and verified:
  `python run_all_mona.py --retire-caches` (measures, lists, deletes). Never mid-round.
- **Hash integrity.** Pre-flight verifies `daioe_quartiles.csv` against the repo copy's sha256 and
  every `.py` against `MANIFEST.txt`.

**Create three folders before uploading:** `Magnus_P1207\canaries-sweden\`, and inside it
`input\` and `round1_EL67898\`.

## 1. Upload list — from, to, and what to do on arrival

`.py` uploads directly (since 12 Aug 2026) and so does `.dta`, which is why the two data files
now ship as Stata files per the house convention (9 Aug 2026) instead of doing the txt-rename
dance — the scripts read either extension. **Only the three R files still need a rename**, because
`.R` is not an allowed portal format.

| # | Upload this file | From (local) | To (MONA) | Rename after upload |
|---|---|---|---|---|
| 1 | `mona_common.py` | `revision/upload/` | `round1_EL67898\` | — |
| 2 | `run_all_mona.py` | `revision/upload/` | `round1_EL67898\` | — |
| 3 | `39_canary_gate.py` | `revision/upload/` | `round1_EL67898\` | — |
| 4 | `40_coverage_diagnostics.py` | `revision/upload/` | `round1_EL67898\` | — |
| 5 | `41_vintage_event_studies.py` | `revision/upload/` | `round1_EL67898\` | — |
| 6 | `42_frozen_cohort.py` | `revision/upload/` | `round1_EL67898\` | — |
| 7 | `43_poisson_primary.py` | `revision/upload/` | `round1_EL67898\` | — |
| 8 | `44_decile_gradient.py` | `revision/upload/` | `round1_EL67898\` | — |
| 9 | `45_asof_backtest.py` | `revision/upload/` | `round1_EL67898\` | — |
| 10 | `46_wfh_horserace.py` | `revision/upload/` | `round1_EL67898\` | — |
| 11 | `r_fepois.txt` | `revision/upload/` | `round1_EL67898\` | **→ `r_fepois.R`** |
| 12 | `r_fepois_es.txt` | `revision/upload/` | `round1_EL67898\` | **→ `r_fepois_es.R`** |
| 13 | `r_fepois_multi.txt` | `revision/upload/` | `round1_EL67898\` | **→ `r_fepois_multi.R`** |
| 14 | `dingel_neiman_ssyk4.dta` | `revision/upload/` | **`input\`** | — |
| 15 | `daioe_quartiles.dta` | `revision/upload/` | **`input\`** | — |
| 16 | `MANIFEST.txt` | `revision/upload/` | `round1_EL67898\` | — |



**Two things to check on arrival.** Uploads can silently get a date suffix when a name already
exists — glance at the folder and rename any back. Then run the pre-flight, which checks exactly
the three hand-renamed files, because those are the ones that get forgotten:

    python run_all_mona.py --dry-run

`MANIFEST.sha256` and this file stay on the laptop; neither is an allowed upload format. The
hashes are of the files as uploaded, before renaming.

## 2. What runs, in what order, and what may run at the same time

**One console, one stage at a time.** Each stage is a separate process on purpose: Python does not
return freed memory to the OS, so a single long-lived process accumulates. The node ceiling is
**100 GB and over-runs are killed without warning** — that is what happened to script 34 in April.

### Lane A — the main run, strictly sequential

    python run_all_mona.py

| Order | Script | Tier | Why here |
|---|---|---|---|
| 1 | `39_canary_gate.py` | **gate** | Pulls 2019–2025 once, writes `output_39/panel_vintage.parquet`, and must reproduce γ₂ = −0.010, Poisson −0.174, N = 11,970,426. A failure aborts the run — everything downstream would be wrong the same way. |
| 2 | `43_poisson_primary.py` | T4/E6 | The headline. The response letter cannot be written without it. |
| 3 | `42_frozen_cohort.py` | T3/E5 | Short, Tier 1, cheap insurance on the coverage defence. |
| 4 | `40_coverage_diagnostics.py` | T1/E3 | The accounting the editor lists. |
| 5 | `41_vintage_event_studies.py` | T2/E4 | ES by code vintage and by margin. |
| 6 | `44_decile_gradient.py` | T11 | Tier 2 — droppable if the trip runs short. |
| 7 | `46_wfh_horserace.py` | T10 | Tier 2 — droppable. Needs file 14. |

### Lane B — the one thing that genuinely parallelises

    python run_all_mona.py --lane b

`45_asof_backtest.py` is the **only** script that does not read the shared panel cache: it makes its
own SQL pulls (2019–2023 × two truncation scenarios). So it shares nothing with lane A but the
database connection, and can run in a **second console** — but only after 39 has finished, and only
if the node shows headroom. If in doubt, run everything in one console:

    python run_all_mona.py --lane all

Order there puts 45 fourth, so the centrepiece is known good by mid-trip rather than at the end.

### Round 2: script 47 (education exposure) — built and tested 4 Sep, runs STANDALONE

`47_edu_exposure.py` no longer waits on anyone: Erik confirmed no do-files exist, supplied the
recipe, and the script implements it with 2019 composition weights (first SUN2020 vintage, zero
nulls, last military/police year, pre-COVID, pre-ChatGPT; 2021 produced alongside as robustness).
Employment-weighted group quartiles; most-recent education per worker — own-year through 2023 and
the 2023→22→21 cascade after, i.e. the latest record that exists, with match rates reported per
age × year; headline = the event-study shape and the single-break post-ChatGPT effect, never the
RB/GPT split (the script-34 lesson). **Three weights tables** ship in the output: all-worker 2019
(primary), all-worker 2021 (temporal robustness), and **young-worker 2019 (ages 22–35)** — the
mapping that reflects where an education's recent holders actually go, with a 22–25 single-break
re-estimate under those weights (ML's point, 4 Sep; Uppsala's graduate-destination logic). DAIOE
vintage is 2023 (`DAIOE_REF_YEAR`), the paper's existing convention, unchanged. Eight dedicated tests in the suite, including a synthetic
end-to-end that recovers an injected −20 % effect at 22–25 only.

**To run it (after round 1 finishes, never beside it):**
1. Upload `utb_grupp2_sun2020_niva3_inr4_nyckel.dta` → `input\` (a `.dta`, uploads directly;
   the script refuses any file whose sha256 differs from Erik's delivered key).
2. Upload the new `47_edu_exposure.py` → `round1_EL67898\` (replace the stub), plus the
   refreshed `MANIFEST.txt`.
3. Submit **`47_edu_exposure.py` itself** to BatchClient — it is standalone, logs to
   `output_47\47_log.txt`, caches under `cache\`, and stamps `RUNLOG.txt` is not touched (the
   runner does that); its own Tee header carries the who-ran-what stamp.

## 3. What was tested before the trip, and what was found

`revision/mona/test_dryrun.py` runs the whole battery locally on synthetic data with no SQL:
it compiles and imports every file, builds a 1.1 M-row employer × ssyk4 × age × month panel with a
real −20 % post-ChatGPT effect written into 22–25, runs the scripts against it, and exercises the
three R wrappers against a known DGP. **35/35 pass.** Re-run it after any edit:

    python3 test_dryrun.py

Evidence it is testing something real, not just exit codes:

- **43** recovers the injected structure: 22–25 `post_gpt_x_high` = **−0.124** (p ≈ 1e-38), null in
  every other age band, and the rate period null everywhere.
- **45** recovers a spurious effect from injected staleness alone: with 8 % missing and 8 %
  misclassified codes after truncation, true γ₂ = −0.025 but as-of γ₂ = −0.085, so the measured
  **artefact is −0.060**. That is exactly the quantity the editor's mechanical story predicts, and
  the script measures it.
- **R wrappers** recover the DGP: `post_gpt_x_high` = −0.2952 against a true −0.2877, converged.
- **40, 41, 42** run to the point where they legitimately need SQL the cache does not hold
  (entrant splits, person-employer flags, the force_cascade panel) and produce their earlier
  outputs first. That boundary is the expected local stopping point.

Three defects were found and fixed:

1. **`46` would have died on the share.** It reads `dingel_neiman_ssyk4.csv` with no fallback and
   that file had never been uploaded — v1 script 26 never ran in MONA. Built and staged as item 14.
2. **`load_daioe` had a latent dtype bug.** It converted `"Q3"` → `3` only when
   `dtype == object`. Under pandas 3 a string column's dtype is `str`, so the conversion would be
   skipped silently, every `exposure_quartile == 4` test would be False, and the panel would come
   back empty rather than wrong. Now tests `is_numeric_dtype` and asserts the values land in 1–4.
   Harmless on MONA's pandas 2, fatal on any upgrade.
3. **`45` opened a SQL connection before checking its own cache**, so a re-run after a failure in
   the estimation stage would have re-pulled ten year-scenario queries. Now guarded like `42`.

One interpretive note, not a defect: in 45 the true and as-of panels have different row counts
(19,200 vs 30,720 on synthetic data) because the as-of cascade scatters workers across more
employer × quartile cells. That is the mechanism, not an error, but say so when the number is
reported.

## 4. Useful flags

    python run_all_mona.py --only 45        # one stage alone, after a fix
    python run_all_mona.py --from 44        # resume mid-lane
    python run_all_mona.py --skip-done      # skip stages with output_NN/_DONE
    python run_all_mona.py --dry-run        # plan + pre-flight, runs nothing

A non-gate failure does not stop the run: later stages do not depend on each other, so the runner
reports it and carries on, and prints the exact `--only` command to retry.

## 5. Export discipline

Aggregates and coefficients only, cell counts ≥ 5, no raw rows. Export budget is a rolling 7-day
50 MB / 1,000 files, max 5 MB per file. Keep `_Tee` echo capped so the master log stays under the
per-file cap.

## 6. 47h, the education-exposure horse race (added 18 Sep 2026, 23:30)

Standalone, like 47b: submit the file itself, not a console wrapper. Two files go up.

| File on this machine | Destination on MONA (full path) | Note |
|---|---|---|
| `revision/upload/47h_edu_horserace.py` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\47h_edu_horserace.py` | upload as `.txt`, rename to `.py` beside the other scripts |
| `revision/upload/eloundou_ssyk4.dta` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\input\eloundou_ssyk4.dta` | `.dta` uploads directly, no rename; next to `daioe_quartiles.dta` |

No new folder anywhere. `MANIFEST.txt` carries 47h's hash; re-upload it only if you want the
manifest row (47h checks its own three inputs by hash regardless).

**What it runs.** Eight education-exposure designs (two replicate Nordström Skans and Sokolow
Romin's mapping, with Eloundou and with DAIOE; the DAIOE one is script 47b's design and is the
gate), each put through the as-of backtest at T=2021 and T=2022 on 2019-2023, ages 22-25 first,
then 26-30 and 50+ for the two reference designs and for any design that clears "usable", then a
gradient tier (31-34, 35-40, 41-49) for the reference designs at T=2022. Winner = smallest 22-25
artefact with a near-zero 50+ artefact. The rule is in the docstring and was written before the run.

**Runtime.** Roughly 5.5 to 6.5 hours: weights 3 x ~1 min, year pulls 5 x 10-15 min, collapses
5 x ~5 min, Tier A 32 fits x ~6 min, Tier B and C ~1.5 h. Every pull is cached under `cache\`;
a resubmit after a failure skips all of them.

**Gate playbook.** The gate compares OL_daioe at T=2021, 22-25, to 47b (-0.3695 as-of,
-0.0099 true). A difference under 0.005 is PASS; under 0.05 is PASS with documented drift (47h
lets '' fall through the cascade with NULLIF, which 47b did not); above 0.05 the run stops
before Tier A and the log names both numbers. If it stops, export `output_47h\47h_log.txt` and
`horserace_estimates.csv` and read them before changing anything.

**Exports to bring out** (all aggregates, cells floored at 5):
`output_47h\47h_summary.txt`, `horserace_estimates.csv`, `score_<design>.csv` (eight files),
`score_diagnostics.csv`, `anchoring_rates.csv`, `47h_log.txt`.

**Tested locally** end to end on synthetic frames with the real key, DAIOE and Eloundou inputs
and real R + fixest: `revision/local/test_47h_synthetic.py` (five cases, all pass, ~5 min).

## 7. 50, calibration moments for the simulation study (added 19 Sep 2026, 01:10)

One file, standalone, about 15 minutes, no dependency on 47h (it reads 47h's caches only if
they happen to exist). Runs in any free slot.

| File on this machine | Destination on MONA (full path) |
|---|---|
| `revision/upload/50_sim_moments.py` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\50_sim_moments.py` (upload as `.txt`, rename to `.py`) |

Inputs already on the share: `input\daioe_quartiles.dta`, `input\utb_grupp2_sun2020_niva3_inr4_nyckel.dta`
(both hash-checked). No new folder.

**Exports to bring out** (`output_50\`, all aggregates, every count 0 or >= 5): `m1a_completion_age.csv`,
`m1b_level_change.csv`, `m2_occ_change.csv`, `m3_staleness.csv`, `m4a_enrolment_prevalence.csv`,
`m4b_field_switch.csv`, `m5a_employer_size.csv`, `m5b_retention_22_25.csv` (only if 47h has run),
`m6_matrix_2019.csv` ... `m6_matrix_2023.csv` and `m6b_inr_tertiary_2019.csv` ... `_2023.csv` (one per
year; each under the 5 MB file cap), `m7_validation_t2021_k0.csv` ... `m7_validation_t2023_k2.csv`
(six files: lagged education x current occupation, the predictive-validation table),
`50_summary.txt`, `50_log.txt`. Once out, run `python3 revision/local/l13_validate_edu_designs.py
<export dir>` locally: it scores every design from m6/m6b, predicts each m7 row and reports
agreement, Q4 precision and recall and percentile error by age, t and lag; no further MONA run. A moment whose query fails is skipped and named in the log; the
rest are still written.

Tested locally end to end: `revision/local/test_50_synthetic.py` (three cases, all pass).

## 8. 48 re-run after the gender fix (added 19 Sep 2026, 01:55)

`Kon` is a char column, so the pulled panel holds the strings "1"/"2" while the split filtered
on integers: every gendered subset was empty, and that surfaced as
`ValueError: You are trying to merge on float64 and object columns for key 'year_month'`
inside `balance_panel`. 48 now normalises the gender column whatever its dtype, prints the codes
and their counts, and raises with a readable message if a code is missing or a subset is empty.

| File on this machine | Destination on MONA (full path) |
|---|---|
| `revision/upload/48_gender_poisson.py` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\48_gender_poisson.py` (upload as `.txt`, rename to `.py`) |

Submit standalone, or `python run_all_mona.py --only 48`. **`panel_gender.parquet` is already
cached**, so the seven year pulls (38 min) are skipped and the run is the gate plus twelve fits,
roughly 30 to 40 minutes. Exports: `output_48\gender_poisson.csv`, `48_summary.txt`, `48_log.txt`.

Tested locally: `revision/local/test_48_gender.py` (four cases: the 18 Sep bug reproduces and is
now caught; the fix works on a string panel and an int panel; a panel missing a code raises).

## 9. 47i, firm-mix exposure -- the register route ML ruled for (19 Sep 2026)

Exposure becomes a property of the FIRM (the worker-weighted education mix of its whole
workforce, quartiles fixed on 2019), as in Nordstrom Skans and Sokolow Romin (2026), so a young
worker's own stale education record never enters the classification -- the mechanism that gave
47b its -0.36 artefact. The young are counted as the outcome, never classified.

| File on this machine | Destination on MONA (full path) |
|---|---|
| `revision/upload/47i_firmmix.py` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\47i_firmmix.py` (upload as `.txt`, rename to `.py`) |

**It needs no SQL of its own if 47h has run**: it reads 47h's cached year frames and weight
pulls, so with a warm cache it is about 10 minutes for 48 fits. With a cold cache it pulls
them itself and takes as long as 47h's pull stage.

Submit standalone, AFTER 47h (or after its caches exist). Exports:
`output_47i\47i_summary.txt`, `firmmix_estimates.csv`, `firmmix_quartile_sizes.csv`,
`47i_log.txt`.

**What it costs, stated in the output rather than buried:** a firm holds one quartile, so the
paper's within-employer comparison is not available in this design; identification is across
firms. The industry-reweighted contrast and hires-as-outcome are additions, not corrections,
and are named in the summary as not done.

Tested locally end to end: `revision/local/test_47i_synthetic.py` (three cases, no SQL, real R).

## 10. 47j, the within-employer triple difference (19 Sep 2026)

Their exposure idea with our identification. Exposure is the education mix of the firm's
INCUMBENTS (31+) in 2019, fixed; the within-employer variation comes from AGE, the one worker
attribute that cannot go stale. Employer x month, employer x age and month x age are all
absorbed, so the surviving term is PostGPT x High x Young: young versus older workers inside
one employer in one month.

| File on this machine | Destination on MONA (full path) |
|---|---|
| `revision/upload/47j_within_employer_triple.py` | `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\47j_within_employer_triple.py` (upload as `.txt`, rename to `.py`) |

Reads 47h's caches; about 10 minutes warm. Submit standalone, after 47h or 47i. Exports:
`output_47j\47j_summary.txt`, `triple_estimates.csv`, `triple_quartile_sizes.csv`, `47j_log.txt`.

**The estimand is the age gradient**, which is what the paper's title claims, so this is closer
to the stated contribution than the current design, not further from it. Its blind spot is
stated in the output: a shock that hit every age equally inside exposed firms would not appear.

Tested locally: corrupting every young education record, and deleting the young entirely, leaves
all 140 fixture firms' exposure bit-identical; a planted decline on the classifier's own Q4 firms
is recovered with the right sign and size; end to end with SQL forbidden.

## 11. ONE FILE for tonight (19 Sep 2026)

`revision/upload/run_tonight.py` -> `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\run_tonight.py`
(upload as `.txt`, rename to `.py`). Submit THAT and nothing else.

It runs, in dependency order and skipping anything already finished: 47h (resumed from its
caches if needed), then 47i, 47j, 48 and 50. A failed stage does not stop the rest. It prints
the plan with expected runtimes first, and the export list at the end.

Still upload the individual scripts it calls (sections 6 to 10) and `eloundou_ssyk4.dta`.

**Then on this machine, one command:** `python3 revision/assemble.py <export folder>`
-- writes `revision/EVIDENCE.md` and `.pdf`, one section per claim, each with the number, the
source file and the verdict under the pre-committed rule. Add `--sim` to run the simulation
study too (about three hours). It never deletes anything; use `fdr` for filing.

## 12. The final upload list (19 Sep 2026, after the cross-vendor review)

Submit ONE file: `run_tonight.py`. Upload all of these first, every one as `.txt` renamed to
`.py` unless stated, into
`\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`:

| # | File in `revision/upload/` | Note |
|---|---|---|
| 1 | `run_tonight.py` | the only one you submit |
| 2 | `50_sim_moments.py` | runs first, ~20 min; carries the validation table |
| 3 | `47L_age_baseline_exposure.py` | **new**, the design the review identified; own SQL, ~75 min |
| 4 | `47h_edu_horserace.py` | the long one, ~5.5 h, resumes from its caches |
| 5 | `47k_settled_sample.py` | **new**, keeps the paper's estimand; reads 47h's caches |
| 6 | `47i_firmmix.py` | supporting evidence |
| 7 | `47j_within_employer_triple.py` | supporting evidence |
| 8 | `48_gender_poisson.py` | with the char/int fix |
| 9 | `MANIFEST.txt` | optional, for the hash rows |

And one data file, uploaded directly with no rename, into
`\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\input\`:

| 10 | `eloundou_ssyk4.dta` | 47h needs it |

No new folders anywhere. Total if nothing has finished: about 10 hours; far less if 47h
already completed, since 47i, 47j and 47k then read its caches.

**Export afterwards:** `output_50`, `output_47L`, `output_47h`, `output_47k`, `output_47i`,
`output_47j`, `output_48`, plus `output_41`, `output_44` and `output_46`, which finished
earlier and were never fetched.

**Then here:** `python3 revision/assemble.py <the export folder>`.

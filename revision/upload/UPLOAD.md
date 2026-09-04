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

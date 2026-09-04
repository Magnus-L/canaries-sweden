# MONA trip — EL67898 revision, round 1

Staged 3 Sep, destination corrected 4 Sep against screenshots of the actual share. **14
uploadable files**; `UPLOAD.md` and `MANIFEST.sha256` stay local, because `.md` and `.sha256` are
not allowed portal formats. Largest file is 17 KB against a 10 MB cap.

## 0. The share is not flat, and the old path was wrong

The project note said scripts and data sat directly under `Lydia P1207\`. They do not. The folder
was reorganised on **2026-07-07** and now looks like this:

    \\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\CANARIES\
      Code\Python\      44 v1 scripts, 09_mona.py .. 38c_olsplus1_robustness.py
      Code\R\           23_mona_rambachan_roth.r, r_feols(.es).R, r_fepois(_es).R
      Input\            EMPTY
      All_output\       output\, output_16..38, output_hires_separations\,
                        the .std/.txt logs, AND the shared data files:
                        daioe_quartiles.csv, eloundou_ssyk4.csv,
                        finland_marginals_2022.txt, plus the v1 caches
                        raw_panel.parquet (1.04 GB), flows_nonzero_cells.csv
                        (2.2 GB), agg_panel_filtered.parquet (60 MB)
      __pycache__\

**This would have killed the trip on the first script.** `mona_common.DAIOE_PATH` resolved to
`Lydia P1207\daioe_quartiles.csv`, which does not exist; the file is in `CANARIES\All_output\`.
Every script that calls `load_daioe()` — that is, all of them — would have died at the first merge.
Fixed: `CANARIES` and `SHARE` are now separate constants and `SHARE` points at `All_output\`.

Reading the data in place rather than copying it is deliberate: one `daioe_quartiles.csv` exists,
the canary gate reproduces against it, and a second copy could silently drift. `All_output\` is a
poor name for input files, but moving them would break the v1 scripts the gate reproduces against.

## Destination for this round

    CANARIES\
      All_output\
        dingel_neiman_ssyk4.csv      <- NEW, item 14, beside daioe_quartiles.csv
      revision_EL67898\              <- NEW, everything else
        mona_common.py, 39..46, run_all_mona.py, r_fepois*.R
        output_39\ .. output_46\     <- created automatically beside the scripts

A self-contained round folder rather than `Code\Python\` + `Code\R\` + `All_output\`, for three
reasons: `R_FEPOIS = _THIS_DIR / "r_fepois.R"` already expects the R helpers beside the scripts, so
splitting them would mean editing tested code for no gain; `Code\R\` already holds `r_fepois.R`
and `r_fepois_es.R` from April, and uploading same-named files there invites the silent
date-suffix rename; and one directory per round is what makes the caches identifiable and
retirable at close, which the 17 August storage warning wants.

Nothing v1 is moved, renamed or overwritten.

## 1. Upload list — from, to, and what to do on arrival

`.py` uploads directly (the txt-rename dance has been obsolete for Python since 12 Aug 2026).
`.R` and `.csv` are **not** allowed formats, so they travel as `.txt` and are renamed in MONA —
the same route `eloundou_ssyk4.txt` already took.

| # | Upload this file | From (local) | To (MONA) | Rename after upload |
|---|---|---|---|---|
| 1 | `mona_common.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 2 | `run_all_mona.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 3 | `39_canary_gate.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 4 | `40_coverage_diagnostics.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 5 | `41_vintage_event_studies.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 6 | `42_frozen_cohort.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 7 | `43_poisson_primary.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 8 | `44_decile_gradient.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 9 | `45_asof_backtest.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 10 | `46_wfh_horserace.py` | `revision/upload/` | `CANARIES\revision_EL67898\` | — |
| 11 | `r_fepois.txt` | `revision/upload/` | `CANARIES\revision_EL67898\` | **→ `r_fepois.R`** |
| 12 | `r_fepois_es.txt` | `revision/upload/` | `CANARIES\revision_EL67898\` | **→ `r_fepois_es.R`** |
| 13 | `r_fepois_multi.txt` | `revision/upload/` | `CANARIES\revision_EL67898\` | **→ `r_fepois_multi.R`** |
| 14 | `dingel_neiman_ssyk4.txt` | `revision/upload/` | **`CANARIES\All_output\`** | **→ `dingel_neiman_ssyk4.csv`** |

**Already on the share, do not re-upload:** `daioe_quartiles.csv`, in `CANARIES\All_output\`.

**Two things to check on arrival.** Uploads can silently get a date suffix when a name already
exists — glance at the folder and rename any back. Then run the pre-flight, which checks exactly
the four hand-renamed files, because those are the ones that get forgotten:

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

### Not in this trip

`47_edu_exposure.py` waits on Erik Engberg's education-exposure measure and **must not hold the
trip**. It is Tier 2, and T13 (public YREG, local, already half-run) does its rhetorical job with
SCB's own occupation coding. See `notes/erik-delivery-vs-T12_2026-09-03.md`.

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

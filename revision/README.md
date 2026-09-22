# The revision package

The code, exports and outputs of the *Economics Letters* revision. The submitted
version's code is untouched in `../src/` and `../mona_package/`; nothing here
overwrites it. The repository README describes the paper, the two repositories and
what can be reproduced without register access; this file describes how the folder is
laid out and in what order it runs. The final-code manifest
(`../notes/final-code-manifest_2026-09-22.md`, local to the authors' working copy)
maps every quoted number to its script and export.

## Layout

```
revision/
  config.py          paths and analysis constants for the local side
  run_local.py       runs local/l01 to l07 in sequence
  assemble.py        collects every known export, applies the read rules fixed
                     before the runs, and writes EVIDENCE.md
  stage_upload.py    keeps upload/ identical to mona/ (--check exits 1 on drift)
  local/             scripts on public data and on the exports; test_*.py beside them
  mona/              scripts that run inside MONA, mona_common.py, the R wrappers,
                     _lane.py and the run_lane*.py wrappers, test_dryrun.py
  upload/            the staged copies of the MONA scripts, MANIFEST.txt, UPLOAD.md
  output/            the aggregate exports from MONA, one folder per export, and
                     the extended posting series
  tables/, figures/  what the local scripts produce; copied to the paper repository
  EVIDENCE.md        what each export supports, under the pre-committed read rules
```

## The local side, in run order

| Script | What it produces | Where it appears |
|---|---|---|
| `local/l01_postings_accounting.py` | drop reasons per year over the raw archives; the valid-code share by month and source | Section 2; OA II.11, II.12 |
| `local/l02_coverage_diagnostics.py` | active occupations, zero cells and quartile shares by month; the 400 to 369 reconciliation | Section 2; OA II.12 |
| `local/l08_extend_2026.py` | the 2026 quarters, the extended panel to June 2026, Equation (1) on both windows by OLS and Poisson | Section 2, Section 3; OA II.13 |
| `local/l03_decile_gradient_postings.py` | the posting difference-in-differences by exposure decile | OA II.15 |
| `local/l05_posting_estimators.py` | the posting difference-in-differences by Poisson with the zero cells | Section 3; OA II.4 |
| `local/l06_seasonality_variant.py` | occupation-group-by-calendar-month effects | Section 3; OA II.14 |
| `local/l04_public_yreg_check.py` | the top-quartile gap by published age band, change since 2022 | Section 3; OA III.5 |
| `local/l09_firm_within_did.py` (and `--variants`) | the within-employer posting design and its event study; eligibility before the hike; the entry-level differential | Section 3; OA V |
| `local/l09b_firm_heterogeneity.py` | the same design by industry, employer age and size | OA V |
| `local/l07_figures_rebuild.py` | Figure 1; the posting event study of OA V | Section 3; OA V |
| `local/l10_appendix_tables.py` | the posting, firm-lane and public-data tables, written into the paper repository | OA II, III.5, V |

The scripts that build the register exhibits from the exports, each reading the
export folder the manifest names and taking one other folder on the command line:

| Script | Exhibit | Reads |
|---|---|---|
| `local/l18_table1.py` | Table 1 | `seasonal_pooled.csv`, `seasonal_gender.csv` (68); `reference_window.csv`, `vcov_s75_*` (75); `contrast_seasonal.csv` (74); `gender_split.csv` (76) |
| `local/l23_fig_age_profile.py` | Figure 2 | `contrast_seasonal.csv` (74) |
| `local/l14_fig_spreading.py` (and `--monthly`) | Figure 3; the monthly diagnostic of OA III.2 | `seasonal_path.csv` (68) |
| `local/l24_tab_descriptive_bands.py` | OA Table III.1 | `output_66__plain_stock.csv` (66) |
| `local/l19_tab_age_profile.py` | OA Table III.2, the headline design by age | 68, 61, 75, 74, 63 |
| `local/l25_tab_industry_credit.py` | OA Table III.2, industry and credit | `industry_fe.csv`, `credit_test.csv`, `vcov_r73_lev_*` (73) |
| `local/l16_fig_firststage.py` | Figure A2 | `itftg_firststage.csv`, `bita_firststage.csv` (71) |
| `local/l21_tab_partIV.py` | OA Tables IV.1 to IV.3 | 40, 41, 45, 68 |
| `local/l15_fig_backtest.py` | Figure A1 | `asof_estimates.csv` (45) |
| `local/l22_tab_descriptives.py` | OA Table I.2 | `68_log.txt`, `seasonal_pooled.csv` (68); `73_summary.txt` (73) |

Four online-appendix tables are still assembled by hand from the exports:
`tableA_profile_seasonal` (74), `tableA_gender_split` and `tableA_education_mix`
(76) and `tableA_contrast_by_track` (77).

`_figsafe.py` refuses to overwrite a figure file that no script in the tree claims,
so an exhibit without a generating script cannot be lost by accident.

## The MONA side, in dependency order

Every script imports `mona/mona_common.py`: one register pull, one panel builder, one
route to R and fixest, one cache discipline and one export floor, so the estimates are
comparable across scripts. Each script's docstring states what it estimates, what it
reads and writes, and where its numbers appear.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `47h_edu_horserace.py` | the education-to-exposure score books, and the comparison of eight scoring rules by an as-of backtest | Individ 2019 to 2021, the declarations 2019 to 2023 with the education vintages, the key and score files | `cache/edu_hr_weights_*`, `edu_hr_*`; `output_47h/` |
| `47j_within_employer_triple.py` | the employer exposure (`incumbent_exposure`) and the within-employer age design on 2019 to 2023 | 47h's caches | `output_47j/` |
| `47L_age_baseline_exposure.py` | the monthly counts by employer and age band to June 2025; the 2019 occupation baseline; the continuous firm-age exposure | the declarations 2019 to 2025, Individ 2019 | `cache/L_baseline_2019`, `L_basepay_2019`, `L_counts_*`; `output_47L/` |
| `54_hiring_flows.py` | hires and separations by employer, age band and month | consecutive months of the declarations | `cache/flows_*`; `output_54/` |
| `61_redated_triple.py` | the panel of Equation (2) (`build_skeleton`, `attach_exposure`, `add_terms`) with the treatment dated at adoption | 47h's and 47L's caches | `output_61/` |
| `63_measure_robustness.py` | the continuous route on DAIOE, Eloundou et al. and teleworkability, stock and flows, both datings; the joint specification | 47L's and 54's caches, the score files | `output_63/` |
| `65_occupation_arm.py` | the same design classified by the 2019 occupations of incumbents | 47L's caches | `output_65/` |
| `66_plain_magnitudes.py` | the descriptive counterpart by quartile, band and period | 47h's, 47L's and 54's caches | `output_66/` |
| `67_gender_on_the_new_design.py` | the sex panel (`build_skeleton_sex`) and the design split by sex | pulls counts and flows by sex | `cache/L_counts_sex_*`, `flows_sex_*`; `output_67/` |
| `68_seasonal_control.py` | the headline with the calendar cycle removed; the year, quarter and month paths; the female differential | 47h's, 47L's, 54's and 67's caches | `output_68/` |
| `70_respecifications.py` | the six-band skeleton and the profile against 41 to 49; the payroll-tax expiry; the education-to-occupation ladder | 47h's and 47L's caches | `output_70/` |
| `71_adoption_validation.py` | the first stage on the ICT surveys of enterprises and individuals | the survey tables, the declarations for the respondent link, 47h's and 47L's caches | `output_71/` |
| `72_incumbent_floor.py` | the reliability of the incumbent mix as a proxy for the young; leave-one-band-out | 47h's and 47L's caches | `output_72/` |
| `73_industry_and_credit.py` | industry by age by month; the credit test on 2019 leverage; failed firms dropped | LISA's firm table, Serrano, 47h's and 47L's caches | `output_73/` (or `CANARIES_73_OUT`) |
| `74_contrast_seasonal.py` | the six-band profile with and without the calendar cycle | 47h's and 47L's caches | `output_74/` |
| `75_reference_window.py` | the headline with the Riksbank term as a window, so the level after adoption has a standard error | 47h's, 47L's and 54's caches | `output_75/`, `vcov_s75_*` |
| `76_gender_decomposition.py` | the female differential split into composition and within, by education track | pulls counts by sex and education | `cache/L_counts_sex_edu_*`; `output_76/` |
| `77_contrast_by_track.py` | the young against 41 to 49 inside each track | 76's caches | `output_77/` |
| `39_canary_gate.py`, `40`, `41`, `46`, `45`, `49` | the submitted design's panel and the coverage diagnostics of OA Part IV: match rates, vintages and exclusions; vintage event studies; the teleworkability check; the as-of backtest; the match-rate definitions | the declarations and Individ 2019 to 2023 | `cache/panel_vintage`, `panel_dual_T*`; `output_39/` to `output_49/` |

Scripts `42`, `43`, `44`, `47`, `47b`, `47i`, `47k`, `48`, `50` to `53`, `55` to `60`,
`62`, `64` and `69` belong to earlier stages of the revision and to designs the paper
does not report; their exports are in `output/` and `EVIDENCE.md` records what each
supports. `78_final_checks.py` runs seven checks on the headline design that the
final review asked for, on the caches above and two small pulls of its own.

**Running on the share.** A script is submitted as one file to the batch client,
which passes no arguments and keeps no standard output; each script therefore opens
its own log under `output_NN/` before anything else and appends a line to the
project's `RUNLOG.txt`. `_lane.py` with a `run_lane*.py` wrapper runs several scripts
in one job and skips a stage whose summary file exists, so a resubmitted job is
cheap. Options travel as environment variables set inside the wrapper
(`CANARIES_73_PARTS`, `CANARIES_73_OUT`, `CANARIES_78_PARTS`, `CANARIES_RWORK_TAG`,
`CANARIES_47H_FRESH`). Caches live under one disposable `cache/` folder on the share;
the R exchange files go to the batch node's local disk. `run_all_mona.py` is the
runner for the submitted design's battery (`39` to `46`) and retires the caches at the
end of a round with `--retire-caches`.

**Estimation.** Every Poisson fit runs in R through fixest (0.13.2 on R 4.5.0), since
pyfixest is not installed in MONA: `r_fepois_multi.R` for every reported estimate
(any term list, any fixed-effect list, the clustered covariance written beside the
coefficients), `r_fepois.R` and `r_fepois_es.R` for the submitted design's pooled and
event-study forms. The wrapper writes the panel as a compressed exchange file with the
fixed-effect keys as integer codes, bounds the thread count and retries at fewer
threads when R's allocator fails.

## Rules carried through every script

- Export safety: aggregates and coefficients only; counts below five are suppressed or
  the cell dropped; no raw rows and no identifiers leave MONA.
- Nothing is written to the project share while a stage runs, other than the stage's
  own output folder and its log.
- Every read rule (an artefact threshold, a reproduction gate, a retained share) is
  stated in the script and fixed before the run; the summary file reports the verdict
  either way.
- Descriptive posting series stop at December 2025, since the two most recent months
  of the bulk collection are under-counted; the posting regressions use the
  closed-quarter files to June 2026.
- Between the two repositories, a table or figure is edited only through the script
  that generates it.

## Status

The register chain is final and its exports are in `output/`; the manuscript reads the
folders the manifest names. The local chain is final. Four online-appendix tables are
still hand-built from the exports and are listed above.

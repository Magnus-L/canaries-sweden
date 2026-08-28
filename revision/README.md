# Revision package — EL67898R1

**Version 2 of the replication code, for the Economics Letters revision.** The submitted
version's code is untouched in `../src/` and `../mona_package/`; nothing here overwrites
it. Plan: `../notes/EL67898-revision-plan_2026-08-28.md` (task IDs T1–T15 referenced
below). Code-read defect list: `../notes/code-read-digest_2026-08-28.md` (D1–D10).

## Layout

```
revision/
  run_local.py       master runner, local side (postings + public data + figures)
  run_mona.py        master runner, MONA side (prints stage plan; each script also runs alone)
  config.py          single config for the local side — self-contained paths (fixes D1, D2)
  local/             l01..l07  local scripts (Platsbanken, SCB public data, figures)
  mona/              39..46    MONA scripts (numbering continues after 38c) + shared module
  output/, tables/, figures/   v2 outputs, kept apart from v1's
```

## Run order — local

| Step | Script | Plan | What |
|---|---|---|---|
| L1 | `local/l01_postings_accounting.py` | T1/E1 | Raw ad counts, drop reasons split (no code, invalid code, bad date, duplicate, parse error) per year; writes `tables/postings_accounting.csv` |
| L2 | `local/l02_coverage_diagnostics.py` | T1/E2 | Monthly valid-SSYK share by source and exposure group; active occupations; zero-posting cells; 400-vs-369 reconciliation |
| L3 | `local/l03_decile_gradient_postings.py` | T11/R1.4 | Posting DiD by exposure decile |
| L4 | `local/l04_public_yreg_check.py` | T13/R2.2 | Extends src/20: SCB YREG54BAS, coverage-immunity framing. **Run**: 16–24 is the only band whose Q4-vs-rest gap falls by 2024 (−0.006) while all older bands rise (+0.02 to +0.05) |
| L5 | `local/l05_posting_estimators.py` | T4/E6, R1.8 | Poisson postings variants. **Run**: β₁ −0.129*, β₂ −0.061 (p=0.11) — estimator-robust vs the OLS spec 2 |
| L6 | `local/l06_seasonality_variant.py` | T14/R1.9 | Group × calendar-month FE rider. **Run**: coefficients move < 0.004; conclusion unchanged |
| L7 | `local/l07_figures_rebuild.py` | T6/E8 | **Run**: Figure 1 rebuilt (two panels, legend, dated events). Figure 2 renders automatically once `mona/output_43/poisson_es.csv` is exported back |

## Run order — MONA (one trip, canary gate first)

All MONA scripts import `mona/mona_common.py` (fixes D8): one SQL pull with the cascade,
one DAIOE merge, one balanced-panel builder, one R-subprocess wrapper — inherited from
script 32's architecture (panel cache, staged outputs, `_Tee` logging, crash recovery).

| Step | Script | Plan | What |
|---|---|---|---|
| Gate | `mona/39_canary_gate.py` | — | Reproduce γ₂ = −0.010 (22–25 OLS+1) and Poisson −0.174, N = 11,970,426 from the cached panel before anything else runs |
| M1 | `mona/40_coverage_diagnostics.py` | T1/E3 | Match rates by month × age × quartile; incumbents / recent hires / entrants; code-vintage shares; exclusion counts |
| M2 | `mona/41_vintage_event_studies.py` | T2/E4 | ES by code vintage (2023/2022/2021); same-employer vs job-changer carry-forward; incumbents vs new matches |
| M3 | `mona/42_frozen_cohort.py` | T3/E5 | Coverage-immune cohort: coded by Dec 2023, followed forward |
| M4 | `mona/43_poisson_primary.py` | T4/E6 | Poisson primary battery: pooled + ES per age group, bridge table inputs, extensive-margin LPM |
| M5 | `mona/44_decile_gradient.py` | T11 | Employment deciles, pre-committed monotonicity read |
| M6 | `mona/45_asof_backtest.py` | T3/E5 | The centrepiece: as-of backtest (pretend the register ends 1–2 years early on 2019–23, rebuild the cascade, measure the artificial coefficient), the missingness × misclassification frontier, and the backdated-date variant (R1.14) |
| M8 | `mona/46_wfh_horserace.py` | T10/R1.7 | Telework-exposure × period interactions, both margins |
| T12 | `mona/47_edu_exposure.py` | T12/R2.3 | Port of Erik's education-exposure do-files — lands when his files arrive; design constraints from the script-34 failure are in the file header |

## Rules carried over

- Export safety: aggregates and coefficients only, cell counts ≥ 5, no raw rows.
- Never write to the MONA share mid-run; outputs under `output_NN/` beside the script.
- Poisson in MONA runs via `Rscript` + fixest 0.13.2 (`r_fepois.R`); pyfixest only locally.
- Posting series in descriptive figures are cut at December 2025: the last two collection
  months are artefacts (`reference_canaries_postings_tail_artefact`). Regressions frozen.

## Status, 28 August 2026

- **Implemented and run locally:** L1–L7 (L1 full pass over the 5.7 GB raw files done).
- **Implemented, MONA-ready, dry-run tested:** `mona_common.py`, 39 (canary gate),
  40 (coverage), 41 (vintage ES), 42 (frozen cohort), 43 (Poisson primary),
  44 (deciles), 45 (as-of backtest + frontier), 46 (WFH horse race);
  `r_fepois_multi.R` e2e-tested against a known DGP with local fixest 0.14.
- **Waiting:** 47 (Erik's education-exposure do-files).
- Upload set for the MONA trip: everything in `mona/` (scripts as .txt), plus
  `data/processed/daioe_quartiles.csv` (already on the share) and
  `dingel_neiman_ssyk4.csv` for 46.

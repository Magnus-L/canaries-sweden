# Session Summary — "Two Economies?" Paper

**Last updated:** 2026-02-24
**Project:** Economics Letters submission
**Repo:** ~/Documents/Research/papers/2026/canaries-sweden/

---

## What the paper does

Tests whether the US "scary chart" (stock prices up, job postings down) and "canaries in the coal mine" (AI displacing young workers) patterns appear in Sweden. Uses 4.6M Platsbanken job ads (2020–2026) matched to DAIOE genAI exposure at SSYK 4-digit level. A DiD design exploits the Riksbanken rate hike (April 2022) preceding ChatGPT (November 2022) by seven months.

**Main finding:** The posting decline is broad-based across all AI exposure quartiles, driven by macroeconomic tightening. No statistically significant additional decline in high-AI-exposure occupations after ChatGPT in the baseline spec, though 2 of 8 alternative specs show significant effects and the design has limited power (MDE ~10 log points).

---

## Current status: READY FOR AUTHOR REVIEW BEFORE SUBMISSION

The paper is complete and compiled. Everything below is done:

### Pipeline (src/)
- `01_download_platsbanken.py` — downloads historical JSONL + JobStream
- `02_process_platsbanken.py` — parses to SSYK4 × month aggregates
- `03_fetch_auxiliary.py` — OMXS30, Riksbanken, Indeed US, DAIOE
- `04_merge_and_classify.py` — merge with DAIOE, compute quartiles
- `05_analysis.py` — DiD regression (4 specs), summary stats
- `06_figures_tables.py` — all publication figures and tables
- `07_robustness.py` — 8 alternative specs, event studies, R-R sensitivity, quadratic trends
- `08_employment_age.py` — SCB YREG54BAS canaries test (public data)
- `09_mona_agi_canaries.py` — self-contained MONA script for AGI register data
- `run_all.py` — master script

### Paper (paper/)
- `main.tex` — 1,995 words (limit: 2,000). 2 figures, 1 table, 7 references.
- `appendix.tex` — online supplement. 7 figures (A1–A7), 2 tables (A1–A2).
- `references.bib` — 13 entries (7 cited in main, 3 in appendix)
- Both PDFs compiled and committed.

### Figures (figures/)
- `fig1_scary_chart.png` — OMXS30 vs postings by AI quartile (main)
- `fig2_exposure_gap.png` — Q4 minus Q1 gap over time (main)
- `figA1_sweden_vs_us.png` — US vs Sweden comparison
- `figA2_quartile_panels.png` — individual quartile trends
- `figA3_event_study.png` — monthly event study
- `figA4_omxspi_comparison.png` — OMXS30 vs OMXSPI
- `figA5_event_study_quarterly.png` — quarterly event study
- `figA6_rambachan_roth.png` — R-R sensitivity (breakdown M̄ = 0.25)
- `figA7_canaries_employment.png` — SCB employment by age × AI exposure

### Tables (tables/)
- `did_regression.tex` — main DiD table (4 specs)
- `robustness_results.tex` — 9 rows (baseline + 8 alternatives)
- `top_bottom_occupations.tex` — top/bottom 10 occupations by DAIOE
- Various CSV outputs

### MONA package (mona_package/)
Self-contained folder for co-author to run AGI register analysis in MONA:
- `README.txt` — explains two options (import .py vs CSV-only)
- `daioe_quartiles.csv` — AI exposure data (369 occupations)
- `09_mona_agi_canaries.py` — full Python script
- `MONA_INSTRUCTIONS.md` — step-by-step spec if only CSV import allowed

### Simulated peer review (completed, not saved to files)
- Round 1: Two reviewers + co-editor → led to event study, R-R, spec (4)
- Round 2: Brynjolfsson, Autor, Cullen personas → led to MDE, Platsbanken bias direction, task-composition channel, Nordic institutions, pre-trend interpretation

---

## Key regression results

| Spec | β₁ (Riksbank) | β₂ (ChatGPT) | Notes |
|------|---------------|---------------|-------|
| (1) Monetary only | -0.178*** | — | |
| (2) + ChatGPT | -0.127*** | -0.062 (p=0.11) | Baseline |
| (3) + Trends | -0.068 | 0.018 | |
| (4) Group×Time | -0.039 | -0.032 | Preferred spec |

Robustness: β₂ significant in 2/8 alternatives (all-apps p=0.018, terciles p=0.026). Negative in 7/8.
MDE at 80% power: ~10 log points. Ex-post power for β₂=-0.062: 37%.
R-R breakdown: M̄ = 0.25.
SCB employment triple-diff (canaries): +0.038 (insignificant).

---

## What remains before submission

1. **Authors review the paper** — Magnus, Erik, Michael, Lydia read and approve
2. **MONA package sent to Lydia** — she will run AGI register canaries analysis when time permits
3. **Incorporate MONA results** — update appendix when results come back from Lydia
4. **GitHub repository** — create public repo, replace "[GitHub repository URL]" placeholder in main.tex
5. **Cover letter** — write for Economics Letters submission
6. **Submit via Elsevier Editorial Manager**

### Optional improvements (not blocking)
- Benchmark DAIOE against Felten/Eloundou measures (Cullen suggestion)
- Additional robustness with OMXSPI in appendix (already has figure, could add regression)
- Save simulated reviewer reports to correspondence/ folder

---

## Git log (12 commits on main)

```
6af9497 Add compiled PDFs (main paper + appendix)
c9d72a5 Revise paper based on simulated review (Brynjolfsson, Autor, Cullen)
1f51bb8 Add R-R sensitivity, quadratic trends, employment-age canaries test, MONA script
507b70a Add conditional parallel trends spec, table improvements, quarterly event study
f69a7b3 Revise paper based on simulated peer review
3ab855a Fix co-author name: Lydia Löthman (not Oskar)
e7f82f0 Update paper text with actual results and add Michael Koch
4e74676 Add OMXSPI (All-Share) scary chart comparison to appendix
1982d66 Improve all 5 figures: smoothing, trimming, and label fixes
fb1af93 Add Brynjolfsson-inspired robustness checks and fix date validation
f609cc6 Fix enriched data schema + linearmodels time index
5640dc4 Initial project structure: pipeline, paper, and config
```

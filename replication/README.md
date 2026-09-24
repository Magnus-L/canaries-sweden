# Replication package for "Same Storm, Different Boats: Generative AI and Young Workers Within Firms"

Magnus Lodefalk, Lydia Löthman, Michael Koch and Erik Engberg. Manuscript under revision at
*Economics Letters* (EL67898), September 2026. Corresponding author: Magnus Lodefalk,
Örebro University School of Business, <magnus.lodefalk@oru.se>.

## Overview

The paper reads two dates, the Riksbank's first rate rise in April 2022 and the launch of
ChatGPT in November 2022, on two margins of the Swedish labour market. The **posting margin**
uses 4.9 million public job advertisements from Platsbanken, 2020 to June 2026, matched to the
DAIOE generative-AI exposure index. The **employment margin** uses the monthly employer
declarations for every employee in Sweden, linked to Statistics Sweden's registers inside its
secure environment MONA, and compares young workers with their older colleagues inside the
same employer.

The package reproduces the paper in three tiers:

1. **Public data (packs 1, 2 and 5).** Download and process the advertisements and the
   auxiliary series, estimate every posting result and build every table and figure that rests
   on public data: Figure 1, Online Appendix Tables A1, A3 to A8, A18, A24 and A29 to A31, and
   Figures A1 to A4 and A9. One command, about 45 minutes on a laptop once the archives are
   downloaded.
2. **Register exhibits (pack 4).** Rebuild every register table and figure (Table 1, Figures 2
   and 3, and the remaining Online Appendix exhibits) from the aggregated results exported from
   MONA, which the package holds. No register access needed; under a minute.
3. **Register estimation (pack 3).** The scripts that produced those exports, as they ran
   inside MONA on project P1207, with a runner that executes them in order. Rerunning them
   requires access to the registers (see "Data availability"); about 53 hours of batch time.

A verification layer (pack 0) checks that the code shipped for MONA is the code that ran, that
every generated table is identical to the table the manuscript prints, and that each of 215
numbers printed in the text of the paper and the online appendix agrees with the file it comes
from.

## Data availability and provenance statements

### Statement about rights

- [x] The authors of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- [x] The authors of the manuscript have documented permission to redistribute and publish the data contained within this replication package, for the public data shipped in `data/raw/` and for the aggregated exports in `3_register_mona/exports/`, which passed Statistics Sweden's output review before they left MONA.
- [ ] Some data cannot be made publicly available: the register microdata behind every employment estimate, described below.

### Summary of availability

- [ ] All data **are** publicly available.
- [x] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

The posting margin (Figure 1, Online Appendix Parts I.1, I.2, II and V, and the published-aggregates check of III.6) rests entirely on public data and can be reproduced end to end from this package. The employment margin (Table 1, Figures 2 and 3, Online Appendix Parts I.2, III and IV) rests on individual-level registers that never leave Statistics Sweden; the package holds the code that ran on them, as it ran, and every aggregate it exported, so every register table and figure can be rebuilt and every printed register number checked, but not recomputed without register access.

### Details on each data source

| Data | Used for | Provider and access | Licence | Shipped |
|---|---|---|---|---|
| Platsbanken historical job advertisements: annual archives 2020 to 2025 (downloaded 24 February 2026) and closed-quarter archives 2026-Q1 and 2026-Q2 (downloaded 24 July 2026), JSONL, zipped | all posting analyses | Swedish Public Employment Service (Arbetsförmedlingen), JobTech Development, <https://data.jobtechdev.se/annonser/historiska/>; downloaded by `1_data_public/01` | CC0 | no (6.2 GB); SHA-256 of each archive in `data/DATA-MANIFEST.csv`, checked by `01 --verify` |
| DAIOE generative-AI exposure by SSYK 2012 occupation and year, of which the paper uses the 2023 cross-section (`daioe_ssyk2012.csv`) | exposure quartiles, both margins | Engberg et al. (2024); the file as distributed by the index's authors in February 2026, two of whom are authors of this paper | redistributed with the authors' permission | yes |
| SSYK 2012 to ISCO-08 correspondence (`ssyk2012_isco08.xlsx`) | crosswalk to O*NET-based scores | Statistics Sweden, <https://www.scb.se> | CC0 | yes |
| SOC 2010 to ISCO-08 crosswalk (`isco_soc_crosswalk2.xls`) | crosswalk | US Bureau of Labor Statistics, <https://www.bls.gov/soc/> | public domain | yes |
| Teleworkability by SOC occupation (`dingel_neiman_telework.csv`) | telework split, OA II.3 and III.2 | Dingel and Neiman (2020), <https://github.com/jdingel/DingelNeiman-workathome> | GPL-3.0 (repository licence) | yes |
| GPT exposure ratings by occupation, crosswalked to SSYK 2012 (`3_register_mona/inputs/eloundou_ssyk4.dta`) | alternative exposure measure, OA Table A14 | Eloundou et al. (2024), <https://github.com/openai/GPTs-are-GPTs> | MIT (repository licence) | yes, as the crosswalked input |
| Indeed Hiring Lab job-postings index, United States (`indeed_us_aggregate.csv`) | Figure A1(a) | Indeed Hiring Lab, <https://github.com/hiring-lab/job_postings_tracker> | CC BY 4.0 | yes |
| OMX Stockholm 30, OMX Stockholm All-Share and S&P 500 daily closes | Figure 1, Figure A1 | Yahoo Finance (tickers `^OMX`, `^OMXSPI`, `^GSPC`), downloaded with `yfinance` (OMX series 18 September 2026, S&P 500 24 February 2026) | Yahoo terms of service (see note) | yes, see note |
| Riksbank policy rate | Figure A1(d) | the dates and levels of the Riksbank's policy-rate decisions, <https://www.riksbank.se>, written into `1_data_public/03_market_and_policy_series.py` | public information | in the script |
| Employment by occupation, age and sex, YREG54BAS (`scb_yreg54bas*.json`) | OA Tables A18 and A24 | Statistics Sweden's statistical database, <https://api.scb.se> (query files shipped) | CC0 | yes |
| Business-register bulk file (`scb_bulkfil.zip`) | industry and legal form of advertising employers, OA V | Statistics Sweden, distributed by Bolagsverket as an EU high-value dataset, <https://vardefulla-datamangder.bolagsverket.se> | open data, free re-use | no; see Part V below |
| Employer-by-month-by-occupation advertisement counts (`firm_month_v2.csv.gz`) | OA Part V | built by the authors from the Platsbanken archives above (employer organisation number as printed in each advertisement) | derived from CC0 data | no; see Part V below |
| Monthly employer declarations at the individual level (AGI, *Arbetsgivardeklaration på individnivå*), 2019 to June 2025; LISA (*Individ*) 2015 to 2023 with the embedded occupation register (*Yrkesregistret*); the education register (SUN 2020); the ICT surveys of enterprises (ITFtg) and individuals (BITA 2024); the enterprise register (*Företagsdatabasen*); Serrano balance sheets (2019) | every employment estimate | Statistics Sweden, through the MONA environment, project P1207 (ORU-MICRO-AI) | confidential | no (the aggregated exports are) |

**Yahoo Finance.** The daily index series were downloaded with the `yfinance` package and are shipped so that Figure 1 can be redrawn exactly as printed. Yahoo's terms restrict redistribution of its data; if the package is deposited in an archive whose terms require it, the three files can be removed and re-downloaded with `1_data_public/03_market_and_policy_series.py --refresh`, at the cost of small revisions to the most recent months.

**Register data.** The registers were delivered by Statistics Sweden (SCB) to the ORU-MICRO-AI database of Örebro University under project P1207, after approval by the Swedish Ethical Review Authority (Etikprövningsmyndigheten; decisions 2021-05040, 2022-03330-02, 2024-01714-0 and 2025-04205-02). They are analysed inside MONA (Microdata Online Access), SCB's secure remote environment; microdata never leave SCB's servers, and only aggregates that have passed SCB's output review are exported. Researchers affiliated with a Swedish institution may apply to SCB for access to the same registers (<https://www.scb.se/en/services/ordering-data-and-statistics/ordering-microdata/>, <mona@scb.se>); access requires an ethical approval and an SCB project agreement and typically takes several months. Researchers abroad can obtain access through a Swedish host institution. A replicator wishing to rerun the register scripts on P1207 itself should contact the corresponding author, who will assist with the application. The authors will preserve the data and the MONA project folder for at least five years after publication. One further register input cannot be shipped: `utb_grupp2_sun2020_niva3_inr4_nyckel.dta`, a correspondence from Statistics Sweden's education groups to SUN 2020 fields built by a co-author for another project and not released; it is used only by the cuts by field of education (scripts 76, 87 and 88) and is available from the authors on request.

## Computational requirements

### Software

Outside MONA (packs 0, 1, 2, 4 and 5):

- Python 3.12.9 with the packages pinned in `requirements.txt` (pandas 3.0.3, numpy 2.3.5,
  scipy 1.15.3, pyfixest 0.40.1, linearmodels 7.0, statsmodels 0.14.6, matplotlib 3.10.8,
  pyarrow 24.0.0, openpyxl 3.1.5, xlrd 2.0.2, requests 2.34.2, tqdm 4.67.3, yfinance 1.2.0,
  pillow 12.1.0): `python -m pip install -r requirements.txt`.
- R 4.6.0 with HonestDiD 0.2.8, for `2_postings/08_honestdid.R` only.
- Stata is not needed. The independent reproduction in Stata 18.5 described in Online
  Appendix II.1 is kept in `archive/` as a record.

Inside MONA (pack 3): Python 3 with numpy, pandas, pyarrow, pyodbc and statsmodels as
installed in MONA in September 2026, and R 4.5.0 with fixest 0.13.2 for every Poisson fit
(versions confirmed from the run logs; `3_register_mona/README.md`).

No random numbers are drawn anywhere in the package, so no seed is set.

### Hardware and runtime

| Tier | Machine | Runtime |
|---|---|---|
| Download of the eight Platsbanken archives (6.2 GB) | any, with a network connection | depends on the connection |
| Packs 1, 2 and 5 | Apple M2, 8 cores, 16 GB, macOS 26.6; peak memory 3.4 GB | 42 minutes (the longest steps are the within-employer design, 12 and 7 minutes, and the processing of the archives, about 5 minutes each) |
| HonestDiD bounds computed from scratch (`--cold`) | as above | several hours; a cache of the paper's run is shipped |
| Pack 4 | as above | under 30 seconds |
| Pack 0 | as above | under a minute |
| Pack 3, chapters 1 to 8 | a MONA batch server (100 GB of memory per job) | about 53 hours in sequence; the data build of chapter 1 about 23 hours |

## Description of the code

```
replication/
  README.md, LICENSE, CITATION.cff    this file, the licences, how to cite
  MAPPING.csv                         exhibit and claim -> script -> export -> output
  MANIFEST.csv                        every checked number: where printed, where computed
  FILES.csv                           every file of the package with its size and SHA-256
  VERIFICATION.md, CHANGELOG.md       what was checked and with what result; the history
  config.py                           every path and constant outside MONA, in one place
  requirements.txt, run_public.sh     the environment and the one command
  data/raw/, data/DATA-MANIFEST.csv   the small public inputs, with source, date, licence
  0_verification/                     the checks
  1_data_public/                      download and processing of the public data
  2_postings/                         the posting analyses
  3_register_mona/                    the register scripts, their inputs and their exports
  4_exhibits/                         the register tables and figures, from the exports
  5_occupation_register_public/       the two tables built from published occupational statistics
  archive/                            records kept for provenance, not run
  output/                             what a run writes (tables/, figures/, results/)
```

Each pack has a README listing its scripts in run order with what each estimates, the exhibit
it serves, what it reads and writes, and its runtime; each script's docstring says the same.
Scripts are numbered in run order within a pack. The register scripts keep the numbers under
which they ran (39 to 93), because those numbers name every exported file.

- `1_data_public/` downloads and verifies the Platsbanken archives (01), rebuilds the
  occupation-by-month counts (02), builds the stock-market, policy-rate and US series (03), and
  merges the counts with the DAIOE quartiles (04).
- `2_postings/` estimates Equation (1) and its variants, the event study and the HonestDiD
  bounds, the diagnostics of Online Appendix Part II, and the within-employer design of Part V
  (scripts 01 to 18; `2_postings/README.md`).
- `3_register_mona/` holds the 41 files that ran in MONA, the three public score files they
  read, the 25 export runs brought out of MONA (268 files, of which the package reads 59),
  `master.py`, and the disclosure rules (`DISCLOSURE.md`).
- `4_exhibits/` holds 21 builders, one per register exhibit, in the paper's order, and
  `run_all.py`. Each checks its inputs against a second record of the same fit before writing
  (standard errors against the exported covariance, rows against the run's own summary).
- `5_occupation_register_public/` builds Online Appendix Tables A24 and A18 from Statistics
  Sweden's published employment by occupation and age.
- `0_verification/`: `check_mona_scripts.py` (the shipped MONA code equals the code that ran,
  by syntax tree), `check_manifest.py` (every row of `MANIFEST.csv`), `build_file_inventory.py`
  (writes `FILES.csv`) and `_assemble_manifest.py` (assembles `MANIFEST.csv` from its parts).

## Instructions to replicators

Public data and register exhibits:

1. Install Python 3.12 and the packages in `requirements.txt`, and R with HonestDiD if the
   bounds are to be computed from scratch.
2. From the package root, run `bash run_public.sh`. It downloads the archives into
   `data/raw/platsbanken/` (or give `--no-download` and set `CANARIES_JOBADS_DIR` to a folder
   that already holds them), verifies them against the digests of the paper's run, and runs
   packs 1, 2, 5, 4 and 0 in that order. Part V additionally needs the two inputs described
   under "Data availability" (`CANARIES_FIRM_CUBE`, `CANARIES_SCB_BULK`); without them steps 14
   and 15 of pack 2 stop with a message and the rest of the run is unaffected.
3. Tables are written to `output/tables/`, figures to `output/figures/`, estimates to
   `output/results/`, each under the file name the manuscript uses. With the manuscript folder
   at `CANARIES_PAPER_DIR`, `python 0_verification/check_manifest.py` compares them with print.

Register estimation, inside MONA (requires access to the registers):

1. Copy `3_register_mona/scripts/` and `master.py` into the project folder and the input files
   into its `input` folder; set the project folder in `mona_common.py` (`PROJECT`).
2. `python master.py --list` prints the plan; `python master.py --chapter 1 --run` builds the
   caches, then chapters 2 to 8. Submit through MONA's batch client as described in
   `3_register_mona/README.md`.
3. Bring out the export folders through Statistics Sweden's output review and place them in
   `3_register_mona/exports/`; pack 4 then rebuilds the exhibits from them.

## List of tables and programs

`MAPPING.csv` is the complete list, with one row per exhibit and per register-backed claim in
the text, and the exact export files. In brief:

| Exhibit | Program | Output |
|---|---|---|
| Figure 1 | `2_postings/18_figures.py` | `fig1_two_panel.pdf` |
| Table 1 | `4_exhibits/01_table1_headline.py` | `table1_headline_v3.tex` |
| Figure 2 | `4_exhibits/02_figure2_age_profile.py` | `fig2_age_profile_v3.pdf` |
| Figure 3 | `4_exhibits/03_figure3_quarterly_path.py` | `fig2_spreading_v3.pdf` |
| OA Table A1 | `2_postings/13_top_bottom_occupations.py` | `top_bottom_occupations.tex` |
| OA Table A2 | `4_exhibits/04_tab_estimation_sample.py` | `tableI2_sumstats_employment.tex` |
| OA Table A3 | `2_postings/10_summary_statistics.py` | `tableI2b_sumstats_postings_v3.tex` |
| OA Figure A1 | `2_postings/18_figures.py` | `figA_posting_context.pdf` |
| OA Figure A2 | `2_postings/07_event_study.py`, `08_honestdid.R`, `09_honestdid_figure.py` | `figA3_event_study_v3.png`, `figA6_rambachan_roth_v3.png` |
| OA Figure A3 | `2_postings/11_rate_sensitivity.py`, `12_telework_split.py` | `figA_rate_sensitivity_scatter.png`, `figA_telework_robustness.png` |
| OA Tables A4, A5 | `2_postings/17_accounting_to_june_2026.py` | `postings_accounting.tex`, `coverage_by_source.tex` |
| OA Tables A6, A7 | `2_postings/03`, `06`, `16` | `postings_extended.tex`, `postings_seasonality.tex` |
| OA Figure A4, Table A8 | `2_postings/04_decile_gradient.py`, `16` | `postings_decile_gradient.pdf`, `postings_deciles.tex` |
| OA Table A9 | `4_exhibits/05_tab_descriptive_bands.py` | `tableA_descriptive_bands.tex` |
| OA Figure A5 | `4_exhibits/12_fig_first_stage.py` | `figA2_first_stage_v3.pdf` |
| OA Table A10 | typed in the manuscript (definitions only) | |
| OA Tables A11, A12 | `4_exhibits/14_tab_window.py`, `13_tab_fixed_contrasts.py` | `tableA_window.tex`, `tableA_fixed_contrasts.tex` |
| OA Figure A6 | `4_exhibits/03_figure3_quarterly_path.py --monthly` | `fig2_spreading_monthly_v3.pdf` |
| OA Tables A13, A14 | `4_exhibits/08_tab_profile_bands.py`, `11_tab_continuous_profile.py` | `tableA_profile_split65.tex`, `tableA_age_profile.tex` |
| OA Tables A15 to A17 | `4_exhibits/15`, `16`, `17` | `tableA_gender_split.tex`, `tableA_education_mix.tex`, `tableA_contrast_by_track.tex` |
| OA Table A18 | `5_occupation_register_public/02_occupation_mix_by_sex.py` | `tableA_occ_mix_by_sex.tex` |
| OA Figure A7, Table A19 | `4_exhibits/06_fig_prepath.py`, `07_tab_prepath.py` | `fig_prepath.pdf`, `tableA_prepath.tex` |
| OA Tables A20, A21 | `4_exhibits/09_tab_industry_credit.py`, `10_tab_cluster_industry.py` | `tableA_industry_credit.tex`, `tableA_cluster_industry.tex` |
| OA Tables A22, A28 | `4_exhibits/18_tab_score_precision_and_coverage.py` | `tableA_size_reliability.tex`, `tableA_occ_coverage.tex` |
| OA Tables A23, A32 | typed in the manuscript (literature and sources) | |
| OA Table A24 | `5_occupation_register_public/01_published_age_gap.py` | `public_yreg.tex` |
| OA Tables A25 to A27, Figure A8 | `4_exhibits/19_tab_register_coverage.py`, `20_fig_backtest.py` | `tableIV1_coverage.tex`, `tableIV2_vintage.tex`, `tableIV3_backtest.tex`, `figA1_asof_backtest.pdf` |
| OA Tables A29 to A31, Figure A9 | `2_postings/14_within_employer.py` (and `--variants`), `15`, `16`, `18` | `firm_within_variants.tex`, `firm_within_did.tex`, `firm_heterogeneity.tex`, `fig3_firm_entry_es.pdf` |
| OA Table A33 | `4_exhibits/21_tab_uncounted.py` | `tableA_uncounted.tex` |

Every register exhibit in pack 4 reads exports written by the MONA scripts named in
`MAPPING.csv` (column `mona_scripts`); `3_register_mona/README.md` lists the chapter of
`master.py` that produces each.

## References

Dingel, J. I. and Neiman, B. (2020). How many jobs can be done at home? *Journal of Public
Economics* 189, 104235. Data: <https://github.com/jdingel/DingelNeiman-workathome>.

Eloundou, T., Manning, S., Mishkin, P. and Rock, D. (2024). GPTs are GPTs: Labor market impact
potential of LLMs. *Science* 384(6702), 1306-1308. Data:
<https://github.com/openai/GPTs-are-GPTs>.

Engberg, E., Görg, H., Lodefalk, M., Javed, F., Längkvist, M., Monteiro, N. P., Kyvik Nordås,
H., Pulito, G., Schroeder, S. and Tang, A. (2024). AI Unboxed and Jobs: A Novel
Measure and Firm-Level Evidence from Three Countries. IZA Discussion Paper 16717.

Indeed Hiring Lab. Job postings tracker. <https://github.com/hiring-lab/job_postings_tracker>.

JobTech Development, Arbetsförmedlingen. Historical job advertisements (Platsbanken).
<https://data.jobtechdev.se/annonser/historiska/>.

Rambachan, A. and Roth, J. (2023). A more credible approach to parallel trends. *Review of
Economic Studies* 90(5), 2555-2591.

Statistics Sweden. Employed persons by occupation, age and sex (YREG54BAS). Statistical
database, <https://www.statistikdatabasen.scb.se>.

Statistics Sweden. ORU-MICRO-AI, project P1207: monthly employer declarations at individual
level (AGI), LISA and the occupation register, the education register, the ICT surveys of
enterprises and individuals, the enterprise register and Serrano balance sheets. Accessed
through MONA, 2026.

## Acknowledgements

The package was prepared by the authors with assistance from Claude (Anthropic) in writing,
running and verifying code, as the paper's declaration on generative AI states. Every script
and document was reviewed by the authors, who take responsibility for it.

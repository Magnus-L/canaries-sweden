# Pack 2: the posting margin

Every estimate on job advertisements, from the sample accounting to the within-employer
design of Online Appendix Part V. The scripts read the public files of pack 1 and write
tables to `output/tables/`, figures to `output/figures/` and estimates to `output/results/`,
each under the file name the manuscript uses. `run_public.sh` at the package root runs
them in this order.

## Scripts, in run order

| Script | Estimates or builds | Exhibit | Runtime |
|---|---|---|---|
| `01_postings_accounting.py` | advertisements by year and reason for removal, 2020 to 2025; valid-code share by month and channel | Section 2; inputs to Tables A4 and A5 | 5.5 min |
| `02_coverage_diagnostics.py` | active occupations by month, zero cells by quartile, the 400 to 369 reconciliation, 2020 to 2025 | Section 2; II.5 | 1 s |
| `03_extend_2026_and_did.py` | the counts for January to June 2026 from the closed-quarter archives; Equation (1) on the windows to December 2025 and June 2026, OLS and Poisson | Section 3 ($\beta_1=-0.127$, $\beta_2=-0.059$); Table A6; Figure 1 input | 1 min |
| `04_decile_gradient.py` | Equation (1) by exposure decile, the median decile the reference | Figure A4, Table A8 | 4 s |
| `05_poisson_estimators.py` | Equation (1) by Poisson on the balanced panel with its zeros | Section 3; note to Table A6 | 20 s |
| `06_seasonality.py` | Equation (1) with one-digit occupation group by calendar-month and by month-of-sample effects | Section 3; Table A7 | 20 s |
| `07_event_study.py` | monthly and quarterly event studies, joint Wald pre-tests, the post-launch average and its covariance | II.2; Figure A2, panel (a) | 8 s |
| `08_honestdid.R` | the Rambachan and Roth relative-magnitudes bounds (HonestDiD) | II.2; Figure A2, panel (b) | hours cold; 1 s from the cache |
| `09_honestdid_figure.py` | the figure of those bounds | Figure A2, panel (b) | 2 s |
| `10_summary_statistics.py` | summary statistics of the posting sample | Table A3 | 20 s |
| `11_rate_sensitivity.py` | revealed sensitivity to the rate rise against exposure | II.3; Figure A3, panel (a) | 3 s |
| `12_telework_split.py` | Equation (1) in teleworkable and non-teleworkable occupations | II.3; Figure A3, panel (b) | 6 s |
| `13_top_bottom_occupations.py` | the ten most and least exposed occupations | Table A1 | 1 s |
| `14_within_employer.py` (and `--variants`) | the within-employer posting design, its event study and its variants | Section 3; Tables A29, A30; Figure A9 | 12 min, and 7 with `--variants` |
| `15_within_employer_heterogeneity.py` | the same design by industry, employer age and size | Table A31 | 7.5 min |
| `16_appendix_tables.py` | the LaTeX tables of 03, 04, 06, 14 and 15 (and of 01 until 17 runs) | Tables A5 to A8, A30, A31 | 1 s |
| `17_accounting_to_june_2026.py` | the accounting and coverage carried to June 2026, after four checks against 01, 02 and 03 | Tables A4, A5; the coverage files of II.5 | 3 min |
| `18_figures.py` | Figure 1 and Figures A1 and A9 | Figure 1; Figures A1, A9 | 5 s |

`_common.py` holds the figure style of 07, 09, 11 and 12 and the loader through which 03, 10
and 17 reuse the advertisement classification of 01 and 15 reuses the panel of 14.
Runtimes were measured in one uninterrupted run of `run_public.sh --no-download` on an Apple
M2 laptop with 16 GB of memory; packs 1, 2 and 5 together take 42 minutes.

## Two windows

Every estimate the paper quotes runs from January 2020 to June 2026 (28,084 positive
occupation-month cells in 369 occupations), except the two diagnostics of Figure A3 (11 and
12), which were drawn on the panel of the submitted version, October 2019 to February 2026
(26,672 cells), whose last two months come from the live feed. The rate sensitivity of 11 is
measured on the months of 2022 alone; the window enters only through the posting-volume
weights. The scripts read the submitted panel so that the figures reproduce as printed; the
manuscript does not state the window of Figure A3.

## The exact HonestDiD bounds and their cache

`08_honestdid.R` solves one linear program per possible location of the largest pre-period
difference, per grid point and per value of $\bar M$, which takes about 16 CPU minutes per
value of $\bar M$ and several hours in all on eight cores. Every finished interval is written
to `output/results/posting_rr_honestdid_v3_cache.csv` and is not recomputed. The cache of the
paper's run is shipped as `cache/posting_rr_honestdid_v3_cache.csv`; `run_public.sh` copies it
into `output/results/` unless `--cold` is given, and 08 then reproduces
`posting_rr_honestdid_v3.csv` in a second. The first run was stopped after its coarse grid,
and the intervals it had finished were entered into the cache from its log, whose four
decimals are exact because the test grid moves in steps of 0.004.

## The within-employer design (Part V) and its input

Scripts 14 and 15 read `CANARIES_FIRM_CUBE`, counts of distinct advertisements by employer
(organisation number), month, four-digit occupation and municipality for January 2021 to
June 2026, and `CANARIES_SCB_BULK`, Statistics Sweden's business-register bulk file
(distributed free by Bolagsverket) for industry and registration date. The count file was
built from the same public Platsbanken archives by the AI-Econ Lab's advertisement monitor
(its extraction script `firm_dimension_extract2.py`, 21 August 2026). It is not shipped,
because it is a firm-level derived file with one row per employer and month; the package
documents its construction instead, which a replicator can repeat from the archives:

- archives 2021 to 2026-Q2 (the annual 2025 archive as republished on 22 July 2026);
- within each archive, advertisements are de-duplicated on a digest of the headline, the
  employer's name and the first 400 characters of the description;
- publication month January 2021 to June 2026; advertisements without a usable organisation
  number are dropped (about one per cent from January 2021; organisation numbers are absent
  before 2021);
- the occupation is the first `legacy_ams_taxonomy_id` of `occupation_group`;
- organisation numbers of sole traders, which are personal identity numbers, are replaced by
  a keyed hash at extraction and never stored;
- the entry-level flag is the regular expression
  `\b(nyexaminerad|nyutexaminerad|junior|trainee(?:program)?|ingen erfarenhet|utan (?:tidigare )?erfarenhet)\b`
  on the lower-cased headline, description, requirements and conditions.

Its SHA-256 is in `data/DATA-MANIFEST.csv`. Whether to ship the count file, or the extraction
script, is open.

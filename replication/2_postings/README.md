# Pack 2: the posting margin

Every estimate on job advertisements, from the sample accounting to the within-employer
design of Online Appendix Part V. The scripts read the public files of pack 1 and write
tables to `output/tables/`, figures to `output/figures/` and estimates to `output/results/`,
each under the file name the manuscript uses. `run_public.sh` at the package root runs
them in this order. Exhibit numbers are those of the online appendix as compiled on
28 September 2026.

## Scripts, in run order

| Script | Estimates or builds | Exhibit | Runtime |
|---|---|---|---|
| `00_employer_counts.py` | the employer-by-month-by-occupation advertisement counts of Part V, rebuilt from the archives (run when the file is absent) | input to 14 and 15 | 5 min |
| `01_postings_accounting.py` | advertisements by year and reason for removal, 2020 to 2025; valid-code share by month and channel | Section 2; inputs to Tables A10 and A11 | 5.5 min |
| `02_coverage_diagnostics.py` | active occupations by month, zero cells by quartile, the 400 to 369 reconciliation, 2020 to 2025 | Section 2; II.8 | 1 s |
| `03_extend_2026_and_did.py` | the counts for January to June 2026 from the closed-quarter archives; Equation (1) on the windows to December 2025 and June 2026, OLS and Poisson | Section 3 ($\beta_1=-0.127$, $\beta_2=-0.059$); Table A4; Figure 1 input | 1 min |
| `04_decile_gradient.py` | Equation (1) by exposure decile, the median decile the reference | Table A8; the decile figure of the offline appendix | 4 s |
| `05_poisson_estimators.py` | Equation (1) by Poisson on the balanced panel with its zeros | Section 3; note to Table A4 | 20 s |
| `06_seasonality.py` | Equation (1) with one-digit occupation group by month-of-year and by month-of-sample effects | Section 3; Table A7 | 20 s |
| `07_event_study.py` | monthly and quarterly event studies, joint Wald pre-tests, the post-launch average and its covariance | II.3; Figure A1, panel (a) | 8 s |
| `08_honestdid.R` | the Rambachan and Roth relative-magnitudes bounds (HonestDiD) | II.3; Figure A1, panel (b) | hours cold; 1 s from the cache |
| `09_honestdid_figure.py` | the figure of those bounds | Figure A1, panel (b) | 2 s |
| `10_summary_statistics.py` | summary statistics of the posting sample | Table A3 | 20 s |
| `11_rate_sensitivity.py` | revealed sensitivity to the rate rise against exposure | II.6; Figure A2, panel (a) | 3 s |
| `12_telework_split.py` | Equation (1) in teleworkable and non-teleworkable occupations, on the submitted version's window | II.6; the input 19 checks itself against | 6 s |
| `13_top_bottom_occupations.py` | the ten most and least exposed occupations | Table A1 | 1 s |
| `14_within_employer.py` (and `--variants`) | the within-employer posting design, its event study and its variants | Section 3; Tables A37, A38; the entry-level event study of the offline appendix | 12 min, and 7 with `--variants` |
| `15_within_employer_heterogeneity.py` | the same design by industry, employer age and size | Table A39 | 7.5 min |
| `16_appendix_tables.py` | the LaTeX tables of 03, 04, 06, 14 and 15 (and of 01 until 17 runs) | Tables A4, A7, A8, A11, A38, A39 | 1 s |
| `17_accounting_to_june_2026.py` | the accounting and coverage carried to June 2026, after four checks against 01, 02 and 03 | Tables A10, A11; the coverage files of II.8 | 3 min |
| `18_figures.py` | Figure 1; the posting-context and entry-level event-study figures, which moved to the offline appendix on 26 September 2026 | Figure 1 | 5 s |
| `19_telework_split_extended.py` | 12's split on the current window, January 2020 to June 2026, through 12's own functions; draws panel (b) of Figure A2 from that run | II.6; Figure A2, panel (b); column (1) of Table A9 | 6 s |
| `20_eloundou_postings.py` | Equation (1) with the Eloundou et al. (2024) score in place of DAIOE, all occupations and the common sample of 341 | II.2; Table A6 | 9 s |
| `21_posting_robustness.py` | Equation (1) under six variations of the construction (positions advertised as the outcome, no pandemic months, terciles, no ICT occupations, a balanced panel), with the Poisson and month-of-year rows carried from 03 and 06 | II.2; Table A5 | 25 s (streams the two 2026 archives once) |
| `22_tab_remote_measures.py` | the table of three remote-work measures against AI exposure, from 19 and from the result files of the two estimation scripts in `extensions/` | II.6; Table A9 | 1 s |
| `23_fig_posting_coverage_monthly.py` | the monthly coverage series: valid-code share by channel, active occupations, zero cells on the scored grid | II.8; Figure A3 | 3 s |
| `24_unscored_codes.py` | the 31 codes the index does not score, by kind (military, managerial without the level split, other), and their share of advertisements | Section 1; II.7 | 2 s |

`_common.py` holds the figure style of 07, 09, 11 and 12 and the loader through which 03, 10,
17, 19 and 21 reuse the advertisement classification of 01 and 15 reuses the panel of 14.
Runtimes were measured in one uninterrupted run of `run_public.sh --no-download` on an Apple
M2 laptop with 16 GB of memory; packs 1, 2 and 5 together take about 43 minutes.

## Two windows

Every estimate the paper quotes runs from January 2020 to June 2026 (28,084 positive
occupation-month cells in 369 occupations), except panel (a) of Figure A2 (11), which was
drawn on the panel of the submitted version, October 2019 to February 2026 (26,672 cells),
whose last two months come from the live feed. The rate sensitivity of 11 is measured on the
months of 2022 alone; the window enters only through the posting-volume weights, and the
manuscript does not state the window of that panel. Panel (b) of Figure A2, the
teleworkability split, is drawn by 19 on the current window; 12 keeps the submitted window
so that 19 can check its functions against 12's result before switching the input.

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

## The remote-work measures (Table A9) and `extensions/`

Table A9 sets three remote-work measures against AI exposure. Its Dingel and Neiman column is
19's result; its Platsbanken and Hansen et al. columns come from two estimation scripts that
ran in the authors' research repository on inputs the package does not ship (the
advertisement text of every archive, the employer counts of Part V and the WFH Map file).
`extensions/` holds those two scripts as they ran and the occupation-level results they wrote;
`extensions/README.md` states what each needs. 22 builds the table from those results after
checking that both scripts reproduce the baselines of 03 and 14.

## The within-employer design (Part V) and its input

Scripts 14 and 15 read `CANARIES_FIRM_CUBE`, counts of distinct advertisements by employer
(organisation number), month, four-digit occupation and municipality for January 2021 to
June 2026, and `CANARIES_SCB_BULK`, Statistics Sweden's business-register bulk file
(distributed free by Bolagsverket) for industry and registration date. The count file has
one row per employer and month, so it is not shipped; `00_employer_counts.py` rebuilds it
from the public archives under the rules the paper's build used, and writes it to
`CANARIES_FIRM_CUBE` (default `data/raw/firm_month_v2.csv.gz`):

- archives 2021 to 2026-Q2 (the annual 2025 archive as republished on 22 July 2026, which
  holds the same advertisements as the paper's copy);
- within each archive, advertisements are de-duplicated on a digest of the headline, the
  employer's name and the first 400 characters of the description;
- publication month January 2021 to June 2026; advertisements without a usable organisation
  number are dropped (about one per cent from January 2021; organisation numbers are absent
  before 2021);
- the occupation is the first `legacy_ams_taxonomy_id` of `occupation_group`;
- organisation numbers of sole traders, which are personal identity numbers, are replaced by
  a keyed hash and never written. The key is read from `CANARIES_EF_KEY`, or drawn at random
  for the run; the paper's key is not published. The pseudonyms depend on the key, the counts
  and estimates do not;
- the entry-level flag is the regular expression
  `\b(nyexaminerad|nyutexaminerad|junior|trainee(?:program)?|ingen erfarenhet|utan (?:tidigare )?erfarenhet)\b`
  on the lower-cased headline, description, requirements and conditions.

The SHA-256 of the paper's file is in `data/DATA-MANIFEST.csv`. A rebuild differs from it in
its bytes, for two reasons that change no count: the pseudonyms of sole traders depend on the
key, and the paper's file lists an employer-month-occupation-municipality cell once per archive
it appears in (12,987 cells appear twice), where 00 sums them into one row. Scripts 14 and 15
sum the counts before they estimate. On 28 September 2026 a rebuild with the paper's key
reproduced the paper's file cell by cell once its repeated cells were summed: 1,881,167 cells,
2,713,590 advertisements and 72,513 entry-level advertisements, with no cell differing.

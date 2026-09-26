# The register analysis (runs inside MONA)

Every estimate in the paper and the online appendix that uses Swedish administrative
registers was produced by the scripts in this folder, run inside Statistics Sweden's
secure remote-access environment, MONA, on the authors' research project P1207 at
Örebro University. The registers cannot leave MONA, so this folder holds the code that
ran there, the small public inputs it read, and the aggregated results that were
exported from it. No individual-level record is included, here or anywhere else in the
package.

## What the folder contains

| Path | What it is |
|---|---|
| `scripts/` | The 56 files that ran in MONA: 45 Python scripts (39 to 102, and `_lane.py`), the shared module `mona_common.py`, three R wrappers around `fixest`, and the six lane runners (`run_lane37a.py` to `run_lane38c.py`) under which the scripts of chapter 10 were submitted |
| `inputs/` | Three occupation-level score files the scripts read from the project's input folder |
| `exports/` | Every aggregated file brought out of MONA for this revision, one folder per export, listed with its SHA-256 in `exports/EXPORT_RUNS.csv` (31 exports, 337 files) |
| `master.py` | Runs the scripts inside MONA, chapter by chapter, with the settings each job ran with |
| `SCRIPTS.csv` | One row per file: its role, the chapters it serves, and the hashes that tie it to the copy that ran |
| `EXPORTS.csv` | One row per exported file: the script that wrote it and the exhibit or printed number it feeds |
| `DISCLOSURE.md` | The disclosure rules the scripts apply before anything is written for export |

## Data availability

The analysis uses four kinds of register held by Statistics Sweden (SCB) and linked
inside MONA by pseudonymised person and employer keys: the monthly employer
declarations at individual level (AGI, tables `Arb_AGIIndividYYYYMM`, from 2019), the
annual individual register built from LISA with occupation (SSYK 2012), education (SUN),
sex and age (`Individ_YYYY`), the firm registers (`Ftg_YYYY`, the Företagsdatabasen
files and the financial-statement files) and SCB's surveys of ICT use in enterprises
and by individuals (`ai_itftg_2019`, `ITFtg_Stora_2021` and `2023`, `BITA_2024` and
`2025`). The university enrolment register (`HREG_AKTIVITET`) is read by one
data-building script.

These data are confidential under the Swedish Public Access to Information and Secrecy
Act. SCB grants access to researchers at Swedish research organisations for a stated
project, after ethical approval; access is not transferable and the data cannot be
passed on by the authors. A replicator applies to SCB for a delivery of the same
registers through SCB's microdata service for researchers,
and the variable lists the scripts need can be read from the SQL in each script. The
authors will help with the application. The access for P1207 runs to 31 December 2027.

## The inputs

The scripts read four small files from the project's input folder (`SHARE` in
`mona_common.py`, `<project>\input`). Three are public and ship in `inputs/`; each is
checked against a pinned SHA-256 before any register is read, and the copies here carry
exactly those hashes.

| File | Content | SHA-256 | Built from |
|---|---|---|---|
| `daioe_quartiles.dta` | DAIOE generative-AI exposure and its quartile, 423 SSYK 2012 four-digit occupations | `e217df0d…41bbb` | `1_data_public/04`, `data/processed/daioe_quartiles.csv` |
| `dingel_neiman_ssyk4.dta` | Dingel and Neiman (2020) teleworkability mapped to 423 SSYK occupations | `a63bf527…868a58` | the SOC 2010 to ISCO-08 to SSYK 2012 route of `2_postings/12` applied to `data/raw/dingel_neiman_telework.csv` |
| `eloundou_ssyk4.dta` | Eloundou et al. (2024) exposure score and high-exposure flag for 394 SSYK occupations | `d47b771e…eda93` | the same route applied to the authors' `occ_level.csv` (MIT licence); `2_postings/20` reads this file for OA Table A10 and checks its hash |

The fourth, `utb_grupp2_sun2020_niva3_inr4_nyckel.dta`, maps SUN 2020 education codes
(level at three digits, field at four) to the education groups that define the
education-based exposure score and the fields of education in Section 3. It is
unpublished work by a co-author and is not distributed; it is available from the
authors on request. `47h_edu_horserace.py` reads it and verifies it against the pinned
hash `c760361ba21554951a0744ee00de2f02f22f2e021b87f0863d9ece049e786637`; the scripts
that need education groups load it through 47h's loader (47j, 61, 66, 67, 68, 70, 71,
73, 75, 76, 77, 78, 79, 80 and 87). Since chapter 1 builds its caches with it, the
register analysis cannot be rerun without it.

## Software

Python 3.13.7 with numpy, pandas, pyarrow (parquet caches), pyodbc (the connection to SCB's
SQL server, ODBC Driver 17) and statsmodels; R 4.5.0 with `fixest` 0.13.2 for every
Poisson fit, called from Python through `mona_common.run_fepois_multi` and the R files
in `scripts/`. The versions are those installed in MONA in September 2026. The path to
`Rscript` is found on the search path or read from a one-line file `rscript_path.txt`
beside the scripts.

## How to run it

1. Copy the contents of `scripts/` and `master.py` into the project folder on the MONA
   share, and the four inputs into its `input` subfolder. If the project folder is not
   the one named in `mona_common.py` (`PROJECT`), change that one line.
2. `python master.py --list` prints every job with its settings, its approximate run
   time and the exhibits it produces. Nothing runs without `--run`, and `--run` refuses
   to start outside MONA; with `CANARIES_DRYRUN=1` it prints the commands instead.
3. Run chapter 1 first (`python master.py --chapter 1 --run`); it builds the caches
   every later chapter reads. Chapters 2 to 8 and 10 follow in any order. Through MONA's
   batch submitter, which passes no arguments, put the arguments on one line in
   `master_run.txt` beside `master.py`.

Each script opens its own log in its `output_*` folder, prints a summary file that
states every read rule and its verdict, and appends one line to `RUNLOG.txt`. The run
time of chapters 1 to 8 and 10 is about 70 hours in sequence on one batch node; jobs
within a chapter can run side by side.

| Chapter | Scripts | Exhibits |
|---|---|---|
| 1 Data build | 47h, 47L, 54, 67, 76, 78 (parts E, F), 79 (part C), 80 (part A) | caches only |
| 2 Table 1 and the sample | 82 (parts A, B, C), 68 | Table 1; OA Tables A2, A15, A17, A34 |
| 3 First stage and specification | 83 (parts A to D) | Table 1; OA Figure A5; Tables A14, A15, A23, A24, A26, A27 |
| 4 The age profile | 85, 63 | OA Tables A12, A17, A18 |
| 5 Timing | 84, 86, 91 | Figure 3; OA Figure A7; Tables A16, A23; the boundary sweep of III.2 |
| 6 Heterogeneity | 87, 88 | OA Tables A15, A19, A20, A21 |
| 7 Register coverage | 39, 40, 41, 45, 49 | OA Part IV, Tables A30 to A32, Figure A8; the submitted estimate |
| 8 Rival explanations | 89, 90, 92, 93 | Section 3; OA II.3, III.2; Table A39 |
| 9 Comparison only | 47j, 61, 66, 70, 71, 73, 74, 75, 77, 78 (parts A to D, G) | the education route, not reported |
| 10 Final checks on tau | 95, 96, 97, 98, 99, 100, 101 (and 102, pending) | Figure 2; OA Figure A6; Tables A25, A33, A40, A41; the second record of Table 1 |

`EXPORTS.csv` gives the exhibit behind every exported file, and the package-level
`MANIFEST.csv` ties each printed number to its file; `4_exhibits/` turns the exports into
the tables and figures outside MONA.

### Chapter 10 and the lane runners

The scripts of chapter 10 were written after the second external review of 25 September
2026 and ran on 25 and 26 September as six lanes (37a to 37c, 38a to 38c), each a
one-line list of stages for `_lane.py`, which runs the stages in order, skips a stage
whose summary already exists, and caps what it echoes to the console (MONA's batch
submitter blocks on a full pipe). The runners are shipped so that the record of how the
scripts were submitted is complete; `master.py` sets the same variables. Every chapter-10
script reproduces Table 1's tau on its own panel before its checks run and stops if it
does not; its summary prints the read rules that were fixed before the run and their
verdicts. Script 98 ran twice: the second run (lane 38b) added to its export the number
of observations each Poisson fit retained, with every estimate unchanged, and OA Table A33
is built from that export while `4_exhibits/26` checks it against the first. Lane 38c
(script 102) replaces the three calendar-quarter terms with eleven month-of-year terms and
reports tau and the female differential beside the same run's gate on Table 1's
specification (OA Table A25, Panel G, built by `4_exhibits/23`).

## The code that ran

The scripts are the copies that ran, with one kind of change: comments and docstrings
were rewritten for a reader of the paper (internal working notes removed, the exhibit
each script feeds named). No statement, string, number or file name was changed.
`0_verification/check_mona_scripts.py` proves this for every file: it removes the
docstrings from both versions, parses each into Python's syntax tree and compares the
trees (for R, the token streams without comments). All 56 files are identical in code.
`SCRIPTS.csv` records for each file the SHA-256 of the copy that ran, the last commit
that changed it in the authors' repository, the SHA-256 of the copy shipped, and the
syntax-tree fingerprint of the copy that ran, so the check also works on this folder
alone (`--fingerprints-only`).

The column `code_at_run_vs_shipped` compares the code at the time of each export with
the code shipped. It agrees for every script behind a printed number, with these
exceptions, none of which changes an exported estimate:

- `mona_common.py` and the three R wrappers changed during September in how jobs run
  (error reporting, retries at fewer threads, one exchange folder per job, integer
  codes for the cluster variable, the covariance matrix written beside each fit). The
  exports of scripts 39 to 77 were produced by the earlier versions.
- `r_fepois.R` and `r_fepois_multi.R` gained, on 26 September, a column `n_obs_fit`
  in the fit's output, the observations the fit retained after `fixest` drops all-zero
  fixed-effect groups and singletons. The exports up to `2026-09-25_2250_s97` were
  produced by the version without it; the column is additive and changes no estimate,
  which the two exports of script 98 (with and without it) show estimate by estimate.
- `98_backtest_common.py` was extended the same day to print the retained
  observations; its first export (`2026-09-25_1832`) comes from the version without
  that print, its second (`2026-09-26_1040`) from the version shipped.
- `82_occupation_route.py`, part A: the scoring code was later moved into a function
  shared by all arms; the output is unchanged.
- `85_occupation_route_plain_profile.py`: a line printed to the console now divides by
  the length of each window.
- `86_occupation_route_prepath.py`: the read rule that checks the path against the
  headline was restated after the run. The exported summary reports that the four-
  decimal agreement check failed (by 5 to 11 units in the fifth decimal, a
  consequence of the 2019 start of the panel) and so declined to draw the path; the
  estimates are those shown in OA Figure A7 and Table A23.
- `66_plain_magnitudes.py`: a printed reading note was corrected after the run.
- `63_measure_robustness.py` and `68_seasonal_control.py` ran twice; the exports the
  exhibits read (`output_63` and the 21:52 export of 68) come from the code shipped.

A few printed strings inside the scripts still carry the vocabulary of the working
repository (for example a file name in a message of 47h, or the words "lane" and "trip"
in read rules and runner docstrings); they are part of the code that ran and are left as
they were.

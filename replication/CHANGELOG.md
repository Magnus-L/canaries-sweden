# Changelog

## 2026-09-26: the package brought to the manuscript of 26 September

The manuscript, online appendix and response letter were rewritten on 25 and 26 September
around tau, the later period against the interim period, and six further register runs
(lanes 37a to 38c) were brought out of MONA. The package follows.

- `3_register_mona/`: scripts 95 to 102, the lane helper `_lane.py` and the six lane runners
  join `scripts/` (56 files; `check_mona_scripts.py` finds the runners' as-run copies in
  `revision/upload/`); the R wrappers are the 26 September versions, which add the retained
  observations to a fit's output (`SCRIPTS.csv` records which exports the earlier versions
  produced). Six export runs join `exports/` under neutral names (69 files; the internal
  filing notes stay out), listed in `EXPORT_RUNS.csv` and `EXPORTS.csv`; `master.py` gains
  chapter 10 with the settings each job ran with.
- `5_occupation_register_public/03` states how alike DAIOE and the Eloundou score rank
  occupations (correlation 0.87 across 393 occupations, reproducing the register run's own
  number; 83 per cent of 2024 employment on the same side of the top-quartile cut), the
  sentence of the online appendix's data section; `4_exhibits/23` adds Panel G of OA Table
  A25 (month-of-year terms, lane 38c).
- `4_exhibits/`: `01` rebuilds Table 1 in its five-row design on tau, checked against
  script 97's re-estimation; `02` draws Figure 2 on tau from the eight-band panel of script
  95; `09` and `18` add the tau rows of OA Tables A24 and A27 from the exported covariances;
  `07`, `08`, `10`, `14` and `19` carry the current labels, captions and notes, and `19`
  adds the person-month panel of Table A31; six builders are new (`22` the components behind
  the headline, `23` the final checks, `24` the unlinked payslips, `25` the non-match rates,
  `26` the three-arm backtest, `27` the female pre-path figure). 27 builders, 1 figure and
  the monthly diagnostic of `03` now serve the offline appendix.
- `2_postings/`: `14` and `16` print the manuscript's labels ("Post-rate-rise",
  "Post-launch", "month-of-year"); five scripts are new (`19` the teleworkability split on
  the current window, which now draws panel (b) of Figure A2; `20` the Eloundou table;
  `21` the posting robustness table; `22` the remote-work measures table; `23` the monthly
  coverage figure). `extensions/` holds the two remote-work estimation scripts as they ran
  in the research repository, with their occupation-level results and a README of the
  inputs they need; `run_public.sh` runs 19 to 23.
- `0_verification/`: the table comparison accepts the co-author markup a printed file may
  still carry (`\add{x}` read as x, `\del{x}` and `\rem{x}` removed) and ignores whole-line
  comments; `MANIFEST.csv` regenerated against the 26 September texts (400 rows: 361 numbers
  and 39 tables; the rows of numbers no longer printed removed, 145 rows added). The numbers
  the manifest found to disagree with their source in the last printed digit are listed in
  `VERIFICATION.md`, section 5, for the authors to correct in the manuscript.
- `MAPPING.csv`, `README.md`, `CITATION.cff` and `VERIFICATION.md` updated to the
  manuscript's title, status and exhibit numbers; `FILES.csv` rebuilt.

## 2026-09-25: licences, archival and the scope of what is shipped

- Licences: MIT for the code (`LICENSE`), CC BY 4.0 for the documentation, the aggregated
  exports and the derived tables (`LICENSE-docs`). The repository's top-level `LICENSE`,
  `README.md` and `CITATION.cff` now state the same.
- Archival: the package will be deposited on Zenodo through a GitHub release on acceptance;
  the README and `CITATION.cff` carry a placeholder for the DOI.
- The Yahoo Finance daily series are no longer shipped, since Yahoo's terms restrict
  redistribution. `1_data_public/03` fetches them when they are absent, on the windows of the
  paper's downloads; the monthly series it builds are identical to the paper's.
- The employer counts of Online Appendix Part V remain unshipped as a firm-level derived
  file; the README states why and where their construction is documented.
- The offline appendix is not part of the publication; the drawing of its quarterly posting
  event study was removed from `2_postings/07` (the estimates it writes are unchanged).
- `archive/independent_reproduction/` keeps the do-file of the reproduction only.
- `MANIFEST.csv`: rows M009 and M037 follow the corrected wording of the manuscript.

## 2026-09-25: the package for the revision (EL67898R1)

The replication package was assembled in `replication/` for the revised manuscript, from the
research code of the revision, which remains in the repository under `revision/` and `src/`:

- The public tiers were copied from `src/` (the submitted version's download and processing
  steps and three of its figures) and `revision/local/`, renumbered in run order, their paths
  routed through `config.py`, and rerun end to end with identical output (VERIFICATION.md, 4).
- The register scripts were copied from `revision/mona/` with comments and docstrings rewritten;
  their code is identical to the code that ran (VERIFICATION.md, 1). A runner, `master.py`,
  replaces the batch wrappers under which they ran.
- The exports were copied from `revision/output/` into `3_register_mona/exports/` under neutral
  folder names (date and time of export, and the scripts that wrote them). Internal filing notes
  and R error files were left out. `exports/EXPORT_RUNS.csv` records the original folder of
  every file and its SHA-256.
- The register exhibit builders were copied from `revision/local/` and carry the table notes as
  the manuscript now prints them.
- Two records were moved into `archive/`: the independent reproduction of March 2026, which
  was previously the only content of `replication/`, and the submitted version's script 32.

## 2026-05-06: the submitted version

The code of the version submitted to *Economics Letters*, register scripts included, is in `src/`;
its outputs are in `data/output/`, `tables/` and `figures/`.

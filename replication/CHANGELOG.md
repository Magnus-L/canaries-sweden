# Changelog

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

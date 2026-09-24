# Changelog

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

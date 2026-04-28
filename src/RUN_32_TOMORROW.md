# Run script 32 tomorrow morning — Kauhanen staged robustness

**Created:** 2026-04-27 evening; updated 2026-04-28 with Steps 4 + 5
**Updated 2026-04-28 afternoon:** pyfixest install blocked in MONA (SCB
support, version conflicts). Poisson PML now runs via R + fixest 0.13.2
(confirmed installed in MONA) called as a subprocess from Python. New
companion file: `src/r_fepois.R`. Logic, formula, FE structure, and
cluster vcov are identical; only the estimator backend changed.
**Author:** Claude (Magnus's pickup file)
**Estimated total runtime:** 90–180 min, dominated by the SQL pull (one-off)

## What this script does

Tests five specification dimensions, **one at a time**, so you can localise
where (if anywhere) the Swedish canaries result breaks. Replaces the
original "single combined test" plan which would have changed everything
at once and left you with an ambiguous null if it failed.

| Step | Threshold | Exposure | FE | Tests |
|---|---|---|---|---|
| 1 | ≥5 cumulative (current) | DAIOE quartile, High = Q4 | emp×quartile + emp×month | OLS+1 vs Poisson |
| 2 | ≥10 mean monthly + ≥100 cum | DAIOE quartile, High = Q4 | emp×quartile + emp×month | Sample threshold |
| 3 | ≥10 + ≥100 (Kauhanen exact) | Eloundou β, quintile, High = Q5 | emp×quintile + emp×month | Full Kauhanen replication |
| 4 | ≥5 cumulative, weighted | DAIOE quartile, High = Q4 | emp×quartile + emp×month | Reweighted to Finnish ISCO-1 + NACE-2 composition (Eurostat LFS 2022). Conditional on `finland_marginals_2022.txt` being on the share. |
| 5 | ≥5 cumulative | DAIOE quartile, High = Q4 | emp×quartile + emp×month | SSYK 25xx (ICT specialists) excluded — same exclusion Brynjolfsson 2025 imposes |
| Diag | n/a | DAIOE quartile (last-known SSYK) | n/a | Post-2023 SSYK non-match attrition by quartile |

## Pre-flight checks (do these first)

1. **Verify Rscript + fixest are available in MONA:**
   ```bash
   Rscript -e 'cat(as.character(packageVersion("fixest")), "\n")'
   ```
   Should print `0.13.2` (or higher). If `Rscript` is not on PATH, ask SCB
   support — they confirmed fixest 0.13.2 is already installed (email
   2026-04-28). The Python script auto-checks at startup and FATALs early
   with a clear instruction if either is missing.

2. **Verify input files exist on the share:**
   - `\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv` — required
   - `\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\eloundou_ssyk4.csv` OR
     `eloundou_raw.csv` + `soc_isco_crosswalk.csv` + `isco_ssyk_crosswalk.csv` —
     required for Step 3 (the script will rebuild quintiles from raw if needed)
   - `\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\empirical_data\finland_marginals_2022.txt` — required for Step 4 only (script skips Step 4 with a clear log message if absent)

3. **Upload sequence (canaries-sweden MONA layout is flat — script and data at the same root, matching `daioe_quartiles.csv` and `eloundou_ssyk4.csv`):**
   - **Python script (.txt → rename to .py inside MONA via Spyder):** upload `mona_package/32_mona_kauhanen_robustness.txt` to the Lydia P1207 root folder. Open in Spyder and save as `32_mona_kauhanen_robustness.py` (or rename via the file browser).
   - **R wrapper (.txt → rename to .R inside MONA):** upload `mona_package/r_fepois.txt` to the same root folder, then rename to `r_fepois.R`. The Python script looks for `r_fepois.R` next to itself (i.e. the same directory).
   - **Data file:** upload `empirical_data/finland_marginals_2022.txt` to the same Lydia P1207 root folder. Stays `.txt`.
   - The Foretagsdatabasen NACE-2 lookup happens inside the script via SQL — no separate import file needed.

4. **Optional — if you have time before running:** the local logic tests pass
   already; you don't need to re-run them in MONA. (`test_32_kauhanen_logic.py`
   in `src/` runs synthetic data tests outside MONA. A separate
   `test_32_rfepois_e2e.py` exercises the R + fixest subprocess path locally
   and was used to verify coefficient recovery on a known-DGP fixture
   before shipping to MONA.)

## Running

From the MONA Python environment with the script in your project's `src/`:

```bash
python 32_mona_kauhanen_robustness.py
```

The script:
- Logs to `src/32_mona_kauhanen_robustness_log.txt` automatically (Tee logger)
- Pulls AGI 2019–2025 once (~30–45 min) and caches to `output_32/step0_panel_cache.csv`
- Subsequent runs reuse the cache (skip the SQL pull)
- Saves results incrementally after each (spec, age_group) — if the script
  crashes mid-way, what completed is already on disk

## Behaviour flags

In the script (top of the configuration block):

```python
RUN_ALL_REGARDLESS = True   # default
```

- `True` — runs all three steps regardless of Step 1 outcome. Recommended
  for tomorrow: you want the full picture even if one step nulls.
- `False` — stops after Step 1 if 22-25 (the headline group) is null or
  wrongly signed. Use this in a follow-up run if you've already seen Step 1
  pass and want to selectively re-run Steps 2-3.

## What you're looking for

**The kauhanen_summary.txt file is the first thing to read.** It is
prose-formatted; stars indicate significance. The decision tree:

| Step 1 result for 22-25 | Interpretation |
|---|---|
| Negative, p < 0.05 | OLS+1 ≈ Poisson confirmed. Paper line 440 is verified. Continue. |
| Null or wrongly signed | The estimator switch alone changes the result. Pause and re-think before promoting Poisson into the paper. |

| Step 2 result for 22-25 | Interpretation |
|---|---|
| Still negative, p < 0.05 | Sample threshold is not what drives Sweden-Finland divergence. Strong defence. |
| Now null | The Finnish null may be a sample-restriction artefact when applied to Swedish data. Worth reporting candidly. |

| Step 3 result for 22-25 | Interpretation |
|---|---|
| Still negative, p < 0.05 | The Swedish result holds under Kauhanen's exact spec. The Sweden-Finland divergence is **economy-dependent, not method-dependent**. This is the strongest possible defence of the EL contribution claim. |
| Now null | The combined Kauhanen spec flips the result. The paper's framing must downgrade to a method-dependent statement, and the appendix must report this candidly. |

## Output files (export from MONA)

Under `output_32/`:

| File | Contents |
|---|---|
| `kauhanen_summary.txt` | Prose summary, read first |
| `kauhanen_comparison.csv` | All three steps side by side, master table |
| `step1_poisson_current.csv` | Step 1 coefficients by age group |
| `step2_poisson_threshold.csv` | Step 2 coefficients by age group |
| `step3_poisson_kauhanen.csv` | Step 3 coefficients by age group |
| `step4_poisson_reweighted.csv` | Step 4 coefficients (Finnish-composition reweighting) — only if `finland_marginals_2022.txt` was on the share |
| `step5_poisson_no_ict.csv` | Step 5 coefficients (SSYK 25xx excluded) |
| `attrition_yearly_totals.csv` | Year-level AGI total vs SSYK-matched count |
| `attrition_by_quartile.csv` | Matched-worker share by year × DAIOE quartile (cells <5 set to NaN) |
| `step0_panel_cache.csv` | Re-usable AGI panel; keep for future runs |
| `eloundou_ssyk4_with_quintile.csv` | (only if Strategy B used) Eloundou with score and quintile |
| `32_mona_kauhanen_robustness_log.txt` | Full stdout/stderr capture |

## Disclosure / export-safety

- All regression coefficients and standard errors are aggregate statistics — no individual data, export-safe.
- The attrition diagnostic applies a `<5` cell-suppression rule: any `(year, quartile)` cell with a count below 5 is set to NaN before export.
- Per-step CSVs contain coefficients, SEs, p-values, N obs, N employers, total person-months — all aggregate.
- The `step0_panel_cache.csv` contains employer-level data and **does not export** as-is. Keep on the MONA share for re-use; do not request approval to export.

## If something fails

1. **Rscript / fixest not found** — see pre-flight check 1. Confirm with SCB
   support; fixest 0.13.2 was reported as installed on 2026-04-28.
2. **`r_fepois.R` not found** — the Python script looks for `r_fepois.R` in
   the same directory as itself. If you uploaded the .txt file but didn't
   rename it, the script will FATAL early with a clear path. Rename in
   Spyder.
3. **DAIOE or Eloundou file missing** — see error message; fix paths in
   the script's CONFIGURATION block.
4. **SQL connection timeout** — retry (the script will resume from cache
   on next run).
5. **Memory pressure during Poisson** — fepois does iterative demeaning;
   for the 50+ age group with ≥5 threshold, the panel can hit 30M cells.
   The R + Python subprocess path adds a CSV-write step before each
   regression call (script writes `_rfepois_in_*.csv` into `output_32/`,
   reads `_rfepois_out_*.csv`, then unlinks both). For a 30M-cell panel
   that's ~1–2 GB on disk transiently — fine, but make sure `output_32/`
   has space. If you get a disk-space error, comment out the age groups
   you don't immediately need. The `AGE_GROUPS` dict at the top of the
   script controls this.
6. **Step 1 takes too long** — fixest::fepois with high-D FEs converges in
   5–20 iterations. Expect 5–15 min per age group at the ≥5 threshold (full
   sample), plus ~10–30 s of CSV I/O overhead per call. If a single age
   group exceeds 30 min, kill and reduce the sample by raising the threshold.
7. **R subprocess output looks weird** — every R call's stdout/stderr is
   surfaced into the Python log with `[R]` and `[R-stderr]` prefixes. If
   fepois fails to converge, the R script writes a failure CSV and the
   Python wrapper returns `None` for that (spec, age_group) — the rest of
   the script continues.

## Contact

Magnus, if anything is unclear or fails: the local logic tests pass on the
synthetic fixture (`test_32_kauhanen_logic.py`), and the R + fixest
subprocess path passes coefficient-recovery on a known-DGP fixture
(`test_32_rfepois_e2e.py`). So any failure in MONA is either (a) data/path
issue at the network share, (b) Rscript / fixest absence, or (c)
memory/runtime issue — not a logic bug in the panel construction or the
estimator wrapper.

If the SQL pull hits SSYK column-name mismatches (e.g. `Ssyk4_2012_J16`
not present in some Individ table), update the COALESCE clauses in the
`pull_year_to_panel` function — script 15 and 29 use the same pattern, so
copy from the working version.

If the Foretagsdatabasen NACE pull fails (Step 4): check the table name in
the SQL is `dbo.Foretag_FDB` and that `NaceG1_2007` is the column name in
P1207. Cross-check against your MONA data dictionary; the script's column
names are based on standard P1207 schema but may need adjustment.

## If results turn ugly

See `notes/risk-preparedness-2026-04-28.md` (drafted *before* the run, as
a pre-commitment device). It contains scenario-by-scenario fallback
framings for the paper if any step nulls or attenuates substantially.
Read this before reframing the abstract or main text — it prevents
motivated reasoning after seeing results.

# Archive: records kept for provenance, not part of any run

Nothing in this folder is called by `run_public.sh`, `4_exhibits/run_all.py` or
`3_register_mona/master.py`. Each item is kept because the paper or the online
appendix quotes a number it produced, or because the appendix describes it.

## `submitted_design/`

The register script of the submitted version that the revised online appendix still
quotes, with the Poisson wrapper it calls. Comments and docstrings were rewritten for
this package; the executable code is the code that ran on 28 April 2026, which
`0_verification/check_mona_scripts.py` confirms by comparing syntax trees.

| File | Kept because |
|---|---|
| `32_mona_kauhanen_robustness.py` | Online Appendix VI.1 quotes its non-match series, 9.2 to 10.5 per cent of employer-declaration person-months (`3_register_mona/exports/2026-04-28_s32/attrition_yearly_totals.csv`); its first specification is the submitted design, whose estimate of $-0.174$ at ages 22 to 25 the paper quotes as "about $-0.17$" (reproduced by script 39 of the register tier). The design is withdrawn (Online Appendix Part IV). |
| `r_fepois.R` | The Poisson wrapper script 32 calls, as it stood when the script ran. |

## `independent_reproduction/`

An independent reproduction of the posting difference-in-differences on the window as
submitted, carried out by one of the authors in March 2026 with a separately
collected advertisement series and Stata 18.5 (`reghdfe`). Online Appendix II.1
reports its outcome: the two series correlate at $R^2 = 0.998$ at the
occupation-by-month level and the coefficients match to three decimals. The do-file
refers to folders on the author's own computer and is kept as the record of that
reproduction, not as a script to run.

| File | Content |
|---|---|
| `replication_20260311.do` | the Stata code of the reproduction |

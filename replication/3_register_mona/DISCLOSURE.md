# Disclosure control

The register data never leave MONA. What leaves are aggregated files, which the
researchers export through MONA's export function under Statistics Sweden's conditions
for project P1207. This note sets out the rules the scripts apply before they write
anything intended for export, so that a reader can check them in the code.

## What an export contains

Every file in `exports/` is one of four kinds:

1. Regression output: coefficients, standard errors, test statistics and the number of
   observations and employers behind each fit, as written by
   `mona_common.run_fepois_multi`.
2. Covariance matrices of reported coefficients (`vcov_<tag>.csv`), a few terms by a
   few terms, written so that a linear combination printed in the paper can be
   recomputed outside MONA. They contain no observation-level information.
3. Aggregate counts, shares and means over employers, age bands, months, sexes,
   education groups or occupations.
4. Plain-text logs and summaries, in which every count is printed through the same
   floor as the tables.

No file contains a person or employer identifier, a record-level row, or an occupation-
or employer-level value of a register variable.

## The rules in the code

- **The count floor.** `mona_common.enforce_min_cell` sets any count of one to four to
  missing and leaves zero as it is; the floor is five. Scripts 82 to 85 and 87 to 90
  write their tables through a `save()` function that applies it, scripts 91 to 93
  apply it directly, and the summaries print counts through `cnt()`, which prints
  "(suppressed)" for the same range. The regression files written through script 78's
  functions (by 78 and 86) carry only coefficients and the size of the whole
  estimation panel, which runs to hundreds of thousands of employers.
- **Shares with their numerators.** Script 79 (`floor_table`) suppresses a share
  whenever its numerator or its total is suppressed, since a share and a published
  total reproduce the count; script 93 reads its tables through the same function.
- **Cells of employers.** Script 66 prints a descriptive cell only when it rests on at
  least five employers (`MIN_FIRMS`) and drops thinner cells rather than blanking them.
- **Employer size.** The estimation panel keeps employers with at least five
  employees (`mona_common.MIN_EMPLOYER_SIZE`), so no estimate rests on an employer
  small enough for its workforce to be identified.
- **Scores built over many people.** An education group's exposure score uses at least
  200 completers (`MIN_CELL` in script 47h), and an employer's occupation score at
  least three coded workers in the cell (script 47L). Neither score is exported at
  the level at which it is built.

## What a replicator should check

The count floor is applied in code, not by inspection, and a replicator exporting new
files from MONA should rely on the same functions. Statistics Sweden's own rules for
exports from MONA apply in addition, and take precedence over anything stated here.

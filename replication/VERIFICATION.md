# Verification

What was checked, how, and with what result, against the manuscript at commit `a0bbade` of
the manuscript repository (25 September 2026). Every check can be rerun with the commands
given.

## 1. The code shipped for MONA is the code that ran

`python 0_verification/check_mona_scripts.py`

The 41 register files were copied from the versions that ran and only their comments and
docstrings were rewritten. The check parses the shipped file and the as-run file into Python
syntax trees with docstrings removed (for R, token streams without comments) and compares them.
**Result: 41 of 41 identical.** The same check confirms the two files of
`archive/submitted_design/` against their originals. `3_register_mona/SCRIPTS.csv` records,
per file, the SHA-256 of the copy that ran and its syntax-tree fingerprint, so the check also
runs on the package alone (`--fingerprints-only`). Where a script changed between its export
and the version shipped, the change and why it does not affect an exported estimate are listed
in `3_register_mona/README.md` ("The code that ran").

## 2. Every generated table is the table the manuscript prints

`python 0_verification/check_manifest.py` (rows of kind `table`)

**Result: 31 of 31 tables identical to the files the manuscript inputs**, line by line. The
figures were compared by rendering both versions to images: all 14 figure files are
pixel-identical (PDF bytes differ only in their creation timestamps).

The table notes of the online appendix were shortened in the manuscript after the builders
were written. The package's builders carry the current note text, so that a rebuilt table is
the printed one; the builders affected are 04, 05, 07, 08, 09, 10, 11, 13, 14, 15, 16, 17, 18,
19 and 21 of `4_exhibits/` and, in pack 2, the label and note strings of 13, 14 and of
`5_occupation_register_public/02`. Only strings changed; a comparison of syntax trees with
string constants masked confirms it for every builder, and every estimate in every table is
unchanged.

## 3. Every number in the text agrees with its source

`python 0_verification/check_manifest.py`

`MANIFEST.csv` holds 215 numbers printed in the running text, captions and hand-typed notes of
the paper (60) and the online appendix (155), each tied to the file it is computed from. A row
passes when the number appears in the manuscript and the source value rounds to it at the
printed precision. **Result: 215 PASS, 0 FAIL, 8 PENDING** (plus the 31 table rows above).
Two rows of the paper failed against earlier drafts and pass since the wording was corrected
in the manuscript: the share of occupation-month cells with no postings (M009, now printed as
2.4 per cent: 698 of the 28,782 cells of the balanced panel) and the standard error of the
contrast of ages 22 to 25 against 41 to 49 (M037, now printed as 0.012, from 0.012487).

The eight pending rows are numbers that no script of the package writes to a file: the
$R^2 = 0.998$ of the independent reproduction (`archive/`), the split of the
2,844 advertisements of the 2026 archives dated outside the half-year (2,657, 100 and 187), the
counts of military, managerial and other unpriced occupation codes (3, 26, 2; counted by hand
from `occupation_reconciliation_lists.txt`, which the pack writes), and the 3.7 per cent of
advertisements in the unpriced codes.

## 4. The public tiers reproduce from the archives

`bash run_public.sh --no-download`, on a clean copy of the package.

**Result:** 12 of 12 public-data tables identical to print, 41 of 41 result files identical to
the files the manuscript's numbers were taken from (one, `telework_did_results.csv`, to
$10^{-10}$), 8 of 8 processed data files byte-identical, every figure pixel-identical; 42
minutes. Two facts about the inputs belong here:

- The occupation-by-month counts the paper's figures and regressions use were built on
  24 February 2026 and included 50 advertisements from JobTech's live feed (one in October
  2025, one in November, 48 in December). Rebuilding the counts from the archives alone
  changes 86 cells and moves the posting estimates by at most 0.00005; no printed number
  changes. The package ships the February counts
  (`data/raw/postings_ssyk4_monthly_2026-02-24.csv`) so that the exhibits reproduce exactly,
  and `1_data_public/02` rebuilds and compares them.
- JobTech republished the 2025 annual archive after the paper's download; the two files differ
  in bytes but hold the same 582,241 advertisements and the same counts.

## 5. Known inconsistencies between code and text, reported and not changed

- **Online Appendix Table A7** is labelled "exposure group $\times$ calendar month"; the code
  (`2_postings/06_seasonality.py`) uses the one-digit occupation group.
- **Online Appendix Figure A3** is drawn on the submitted version's window, October 2019 to
  February 2026, whose last two months come from the live feed; the appendix does not state the
  window.
- **Online Appendix VI.1, the non-match series (9.2 to 10.5 per cent).** Its numerator counts
  employer-person-months of workers aged 22 to 69 with an occupation code, and its denominator
  distinct persons of every age per month (`archive/submitted_design/32`, `attrition_diagnostic`).
  The share is therefore not only the share without a code.
- **Online Appendix Table A28** gives 2.5 per cent of resolved codes as carried back from
  2015 to 2018, a share of codes, and 7.4 per cent for the average employer, the figure script
  82's summaries print; both are typed in the builder's note.
- **Script 86**'s exported summary reports that a four-decimal agreement check failed (by 5 to 11
  units in the fifth decimal) and declines to draw the path; Figure A7 and Table A19 draw it.
- **Figure 3**'s builder reads script 68's export for the set of quarters on the axis only.

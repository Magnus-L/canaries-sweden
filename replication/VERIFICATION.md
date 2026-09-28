# Verification

What was checked, how, and with what result, against the manuscript at commit `46baab9` of
the manuscript repository (28 September 2026: `main_v3.tex`, `appendix_v3.tex`, the table
files in `tables/` and the figures in `figures/`). Every check can be rerun with the commands
given.

## 1. The code shipped for MONA is the code that ran

`python 0_verification/check_mona_scripts.py`

The register files were copied from the versions that ran and only their comments and
docstrings were rewritten. The check parses the shipped file and the as-run file into Python
syntax trees with docstrings removed (for R, token streams without comments) and compares them.
**Result: 67 of 67 identical** (28 September 2026). The as-run copies are `revision/mona/<file>` in the research
repository and, for the eleven runners through which the last register scripts were submitted, `revision/upload/<file>`, where they were staged for
MONA. The same check confirms the two files of `archive/submitted_design/` against their
originals. `3_register_mona/SCRIPTS.csv` records, per file, the SHA-256 of the copy that ran
and its syntax-tree fingerprint, so the check also runs on the package alone
(`--fingerprints-only`). Where a script changed between an export and the version shipped, the
change and why it does not affect an exported estimate are listed in
`3_register_mona/README.md` ("The code that ran"); the 26 September change to the R wrappers
(the retained observations added to a fit's output) is shown to leave every estimate unchanged
by the two exports of script 98, which `4_exhibits/26` compares estimate by estimate.

## 2. Every generated table is the table the manuscript prints

`python 0_verification/check_manifest.py` (rows of kind `table`)

**Result: 41 of 41 tables pass.** Nineteen are identical line by line to the files the
manuscript inputs. In the other 22 the printed file was copy-edited after it was built, on
28 September: minus signs typeset as minus signs, age ranges as en dashes, some labels and
notes reworded, and wide tables wrapped to fit the page. For these the check compares the
numbers in the table cells, in order, and every cell number is the built one. The builders of
Tables A2 and A5 were brought up to date the same day (Panel B's labels; row 9, the unscored
occupations added back), so both of those are identical line by line.

Figures were compared by rendering both versions to images. The three figures added in the
revision (Figure 2 of the paper, `fig2_age_profile_v4`; Online Appendix Figures A3,
`fig_posting_coverage_monthly`, and A6, `fig_prepath_female`) are pixel-identical to the files
the manuscript includes (2.5, 4.9 and 2.5 million pixels, none differing). Both panels of
Figure A2 are, since 28 September, the files the pack builds: scripts 11 and 12 now save them
without an alpha channel (`save_png_rgb` in `_common.py`), because a transparent PNG left a
blank page in the online appendix, and the manuscript's copies were replaced by the builds.
The flattened builds are pixel-identical to the earlier transparent ones laid on white. Every
other figure is unchanged since the 25 September check, where all were pixel-identical.

The table notes of the online appendix are the text the manuscript prints on 28 September.
Where a builder's text changed between versions of the manuscript only strings changed; the
estimates in every table are read from the exports. Where a table reports $\tau$ and the
exports do not hold it as a row, the builder computes it as the later-period term minus the
interim term and takes its standard error from the exported covariance of the two terms, after
checking that every exported standard error is the square root of its own diagonal.

## 3. Every number in the text agrees with its source

`python 0_verification/check_manifest.py`

`MANIFEST.csv` holds 419 numbers printed in the running text, captions and hand-typed notes of
the paper and the online appendix, each tied to the file it is computed from. A row passes when
the number appears in the manuscript and the source value rounds to it at the printed
precision. **Result: 411 PASS, 0 FAIL, 8 PENDING** (plus the 41 table rows above, all PASS).
The last-digit discrepancies the 26 September check found have since been corrected in the
manuscript (section 5).

The pending rows are numbers that no script of the package writes to a file, among them the
$R^2 = 0.998$ of the independent reproduction (`archive/`), the split of the 2,844
advertisements of the 2026 archives dated outside the half-year (2,657, 100 and 187), two
readings of printed figures (the "eight" log points of Table A17 and the first-quarter
ratios of script 107's summary), and the precision and recall of the remote-work text
rule on hand-read samples (about 94 and 90 per cent; the rule's agreement with the structured
field is in `2_postings/extensions/results/l50_remote_validation.csv`). The counts of
unscored occupation codes (3 military, 26 managerial of which 25 lack the split into levels, 2
other) and their 3.7 per cent of advertisements, pending until 28 September, are now written
by `2_postings/24_unscored_codes.py` and pass. Every MONA run the manuscript reports has
returned; the last, script 102 (month-of-year terms), is Table A26, Panel G.

## 4. The public tiers reproduce from the archives

`bash run_public.sh --no-download`, on a clean copy of the package (25 September), and the
five scripts added on 26 September (`2_postings/19` to `23`) run on that copy's outputs.

**Result:** the 12 public-data tables of the 25 September check remain identical to print; the
four tables the new scripts write (Tables A5, A6 and A9 of the online appendix, and A37 with
its relabelled headings) are identical to print; the result files the earlier check compared
are unchanged; 45 minutes in all. `2_postings/21` re-streams the two 2026 closed-quarter
archives for the positions advertised and reproduces `03`'s advertisement counts cell by cell
(2,345 cells, 289,601 advertisements). Two facts about the inputs belong here:

- The occupation-by-month counts the paper's figures and regressions use were built on
  24 February 2026 and included 50 advertisements from JobTech's live feed (one in October
  2025, one in November, 48 in December). Rebuilding the counts from the archives alone
  changes 86 cells and moves the posting estimates by at most 0.00005; no printed number
  changes. The package ships the February counts
  (`data/raw/postings_ssyk4_monthly_2026-02-24.csv`) so that the exhibits reproduce exactly,
  and `1_data_public/02` rebuilds and compares them.
- JobTech republished the 2025 annual archive after the paper's download; the two files differ
  in bytes but hold the same 582,241 advertisements and the same counts.

The two estimation scripts behind the realised remote-work columns of Table A9 ran in the
research repository on inputs the package does not ship (`2_postings/extensions/README.md`);
`2_postings/22` checks that their AI-only baselines equal those of `03` and `14` before it
builds the table from their results.

The employer counts of Online Appendix Part V are not shipped; `2_postings/00` rebuilds them from
the archives. On 28 September 2026 a rebuild from the same archives with the paper's key
reproduced the paper's file in every cell once the cells that file lists once per archive were
summed (1,881,167 cells, 2,713,590 advertisements, 72,513 entry-level, no cell differing), and
scripts 14, 15 and 16 run on the rebuild write
result files and Tables A37 to A39 byte-identical to those from the paper's file.

## 5. Inconsistencies between the manuscript and its sources

Found by the manifest against the 26 September manuscript, each a last-digit matter, and corrected
in the manuscript since; none remains in the 28 September check:

- **Online Appendix III.2, the age profile:** "87 and 80 per cent of the same-sample
  $-0.041$". Table A26 prints the same-sample $\tau$ as $-0.0405$; the export holds
  $-0.040471$, which rounds to $-0.040$. The text rounds the four-decimal print rather
  than the value.
- **Online Appendix III.2, the sexes:** "6.5 standard errors under employer clustering and
  5.2 under industry clustering". The exports give $0.071381/0.010874 = 6.56$ and
  $0.071381/0.013566 = 5.26$, so 6.6 and 5.3; the text's figures follow from the rounded
  $-0.071$, 0.011 and 0.014 of Table 1.
- **Online Appendix IV.3:** "moves the coefficient by $-0.012$ at the 2021 truncation".
  Table A35 prints $-0.0115$; the export holds $-0.011475$, which rounds to $-0.011$.
- **Online Appendix VI.1:** "the gap between the groups narrows by 0.026 percentage
  points". On the unrounded shares the narrowing is 0.0269 points; 0.026 is the difference
  of the printed three-decimal shares ($0.066 - 0.040$). Table A43 itself prints the
  difference row as the difference of the printed shares, which the builder reproduces.
- **Online Appendix Table A43, last row:** the printed file gave $-0.0565$ (0.0111) for the
  female differential under the second allocation; the export holds $-0.056447$, so
  $-0.0564$, which the printed file now shows.
- **Online Appendix IV.3:** "the observations each fit retains ... differ by under two per
  cent across them". At the 2021 truncation the spread is 1.5 per cent of the input cells and
  2.3 per cent of the retained cells; at 2022, 0.9 and 0.9. The manifest reads the statement
  as a share of the input cells (row OA272), on which it holds.

Carried over from the 25 September check and still true:

- **Online Appendix Figure A2, panel (a)** is drawn on the submitted version's window, October
  2019 to February 2026, whose last two months come from the live feed; the appendix does not
  state the window. Panel (b) is drawn on the current window (`2_postings/19`).
- **Online Appendix Table A36** gives 2.5 per cent of resolved codes as carried back from
  2015 to 2018, a share of codes, and 7.4 per cent for the average employer, the figure script
  82's summaries print; both are typed in the builder's note.
- **Script 86**'s exported summary reports that a four-decimal agreement check failed (by 5 to 11
  units in the fifth decimal) and declines to draw the path; Figure A5 and Table A17 draw it.
- **Figure 3**'s builder reads script 68's export for the set of quarters on the axis only.

Resolved since the 25 September check: Table A7's label ("one-digit occupation group $\times$
month-of-year") now matches the code of `2_postings/06`; the non-match series of Online
Appendix VI.1 is rebuilt from the raw declarations by script 99 (Table A44), and the
submitted version's 9.2 to 10.5 per cent, whose numerator and denominator were not
commensurable, is no longer printed as a current figure.

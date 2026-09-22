#!/usr/bin/env python3
"""
l34_tab_uncounted.py: Online Appendix Part VI, Table tab:uncounted, the
payslips the employment panel never counts.

THE QUESTION
The editor's objection to the submitted design was that the register
stops in 2023, so the later years are measured with a coverage that the
earlier years do not share. The design the paper now reports classifies
no worker by occupation, but it still requires a birth year and a sex,
which are read from the 2023, 2021 and 2019 individual registers. A
worker in none of the three, in practice one first registered in Sweden
in 2024 or 2025, carries payslips the panel never counts. This table
counts them. It is the editor's mechanism in its purest form, and it is
a count rather than a test: it bounds what the channel could mechanically
produce and does not establish that it operates.

WHAT IS REPORTED
The share of employer-declaration person-months whose worker reaches none
of the three individual registers, by year, for all employers and then
for the top exposure quartile against the lower three (script 79, lane 26
part C). Quartiles are the employer's 2019 education-mix exposure
quartile; person-months at employers the 2019 score book does not score
sit outside the two quartile columns and inside the first, which is why
the first column is not a weighted average of the other two.

Two readings follow, both in the text. The share does not rise once the
register ends, because birth year and sex, unlike occupation, can be read
from any of three vintages and almost every worker appears in one. The
gap between the top quartile and the lower three is real but nearly flat,
and a level difference is differenced out by a design that compares
changes; what could manufacture a step is the movement in that gap, which
is an order of magnitude smaller than the step itself. 2019 is the first
year of the employer declarations and its gap takes the opposite sign, so
the pre-period averages quoted in the text run from 2020; 2025 covers the
first half of the year.

INPUTS AND OUTPUTS
Reads uncounted_share.csv (script 79, lane 26 part C; columns year,
quartile_group, decl_band, n_total, n_counted, n_no_register,
n_register_no_birth_or_sex, n_outside_age_range, share_no_register,
share_not_counted) and 79_summary.txt, the run's own report of the same
shares, from the export directory the final-code manifest names or one
given on the command line. Nothing is typed in. Writes
revision/tables/tableA_uncounted.tex and copies it to
canaries-sweden-paper/tables/; the appendix carries the float, the
caption and the label and inputs this file inside them.

    python3 revision/local/l34_tab_uncounted.py [export_dir]

THE GATE
This export fits no model, so its integrity check is arithmetic and the
script applies all of it. Every row's four parts, the counted, those with
no register row, those in a register without a birth year or a sex, and
those outside the age range, must sum to its total; each reported share
must equal its own numerator over its own total; the four quartiles and
the unscored must sum to the all-employer row, and the lower three
quartiles to the column the table prints, so the two quartile columns
partition what the first column holds. The export must cover the years
the table prints, and one declaration band, since a worker with no
register row has no register age and the share cannot be broken down.
Every printed share is then checked against part C of the run's summary,
which reports the same numbers independently of the CSV, and read back
from the string that goes into the table and compared with the export it
came from. Any disagreement beyond half of the last printed digit stops
the script and nothing is written.

IN THE PAPER
Online Appendix Part VI, the paragraph on non-match rates and sample
attrition, Table tab:uncounted; the bound it supports is quoted in the
same paragraph and in the response to the editor.
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE26C = OUT / "round3_20260922-lane26c-uncounted"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there, inside the float that carries its caption.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

YEARS = list(range(2019, 2026))
# Export key, column heading.
COLUMNS = [("all", "All employers"), ("Q4", "Top exposure quartile"),
           ("Q1-Q3", "Quartiles 1 to 3")]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
BAND = "all"          # the declaration carries no age, so there is one band
PARTS = ["n_counted", "n_no_register", "n_register_no_birth_or_sex",
         "n_outside_age_range"]
SHARE = re.compile(r"^([0-9]+\.[0-9]{3})$")
SUMMARY_ALL = re.compile(
    r"^\s+(\d{4})\s+([0-9,]+) person-months\s+no register\s+([0-9.]+)%\s+"
    r"not counted at all\s+([0-9.]+)%\s*$")
SUMMARY_SPLIT = re.compile(
    r"^\s+(\d{4})\s+Q4\s+([0-9.]+)%\s+Q1-Q3\s+([0-9.]+)%\s+"
    r"unscored\s+([0-9.]+)%\s*$")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def share_cell(what: str, x: float) -> str:
    """A share as a percentage, checked against the export before it is
    allowed into the table."""
    out = f"{100 * x:.3f}"
    m = SHARE.match(out)
    if m is None or abs(float(m.group(1)) - 100 * x) > 5e-4:
        raise SystemExit(f"  {what}: the printed share {out} disagrees with "
                         f"the export ({100 * x:.6f})")
    return out


def summary_part_c(path: Path) -> dict[tuple[int, str], str]:
    """Part C of the run's summary: the same shares as the run printed
    them, for all employers and for the two quartile groups. This is a
    second record, written by the run rather than derived from the CSV."""
    said = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ALL.match(line)
        if m:
            said[(int(m.group(1)), "all")] = m.group(3)
            continue
        s = SUMMARY_SPLIT.match(line)
        if s:
            said[(int(s.group(1)), "Q4")] = s.group(2)
            said[(int(s.group(1)), "Q1-Q3")] = s.group(3)
    if not said:
        raise SystemExit(f"  {path.name}: part C reports no uncounted share, "
                         f"so there is nothing to check the export against")
    return said


def main() -> int:
    d = pd.read_csv(source(LANE26C, "uncounted_share.csv"))
    bands = sorted(set(d.decl_band))
    if bands != [BAND]:
        raise SystemExit(f"  uncounted_share.csv: one declaration band "
                         f"expected, found {bands}; the declaration carries no "
                         f"age of its own")
    years = sorted(set(int(y) for y in d.year))
    if years != YEARS:
        raise SystemExit(f"  uncounted_share.csv: the table prints {YEARS}, "
                         f"the export covers {years}")

    # The arithmetic this export offers in place of a model's diagnostics.
    off = (d.n_total - d[PARTS].sum(axis=1)).abs().max()
    if off != 0:
        raise SystemExit(f"  uncounted_share.csv: a row's parts do not sum to "
                         f"its total (off by {off})")
    for col, num in (("share_no_register", "n_no_register"),
                     ("share_not_counted", None)):
        got = (d[num] / d.n_total) if num else (1 - d.n_counted / d.n_total)
        if (d[col] - got).abs().max() > 1e-9:
            raise SystemExit(f"  uncounted_share.csv: {col} is not its own "
                             f"numerator over its own total")
    for year in YEARS:
        g = d[d.year == year].set_index("quartile_group")
        parts = sum(int(g.loc[q, "n_total"]) for q in QUARTILES)
        if parts + int(g.loc["unscored", "n_total"]) != int(g.loc["all", "n_total"]):
            raise SystemExit(f"  {year}: the four quartiles and the unscored "
                             f"do not sum to all employers")
        if sum(int(g.loc[q, "n_total"]) for q in QUARTILES[:3]) != int(
                g.loc["Q1-Q3", "n_total"]):
            raise SystemExit(f"  {year}: Q1 to Q3 do not sum to the column the "
                             f"table prints")

    said = summary_part_c(source(LANE26C, "79_summary.txt"))
    d = d.set_index(["year", "quartile_group"])

    rows = []
    for year in YEARS:
        cells = []
        for key, _ in COLUMNS:
            if (year, key) not in d.index:
                raise SystemExit(f"  {year}: no {key} row in the export")
            text = share_cell(f"{year} {key}",
                              float(d.loc[(year, key), "share_no_register"]))
            if said.get((year, key)) != text:
                raise SystemExit(f"  {year} {key}: the table would print "
                                 f"{text}, the run's summary reports "
                                 f"{said.get((year, key), 'nothing')}")
            cells.append(text)
        rows.append(f"{year} & " + " & ".join(cells) + r" \\")
        print(f"  {year}  " + "  ".join(f"{c:>6s}" for c in cells))

    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           "Year & " + " & ".join(head for _, head in COLUMNS) + r" \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent of employer-declaration person-months. A worker enters "
            r"the employment panel only if a birth year and a sex can be read "
            r"from the 2023, 2021 or 2019 individual register; the table "
            r"reports the complement. Quartiles are the employer's 2019 "
            r"education-mix exposure quartile, and person-months at employers "
            r"the 2019 score book does not score are outside the two quartile "
            f"columns but inside the first. {YEARS[0]} is the first year of "
            r"the employer declarations and its gap takes the opposite sign, "
            r"so the pre-period averages quoted in the text run from "
            f"{YEARS[1]}; {YEARS[-1]} covers the first half of the year. The "
            r"employer declaration carries no birth year and no age of its "
            r"own, so the share cannot be broken down by age band. "
            r"Source: script 79.",
            r"\end{minipage}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_uncounted.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

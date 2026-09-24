#!/usr/bin/env python3
"""
21_tab_uncounted.py: Online Appendix Table A33 (Section VI.1), the payslips
the employment panel never counts.

A worker enters the panel only if a birth year and a sex can be read from the
2023, 2021 or 2019 individual register; a worker in none of them, in practice
one first registered in Sweden in 2024 or 2025, carries payslips the panel
never counts. The table reports that share of employer-declaration
person-months by year, for all employers and for the top exposure quartile
against the lower three (2019 occupation-mix quartiles); scripts 79 and 93. It
is a count, which bounds what the channel could mechanically produce.

Nothing is written unless each row's parts sum to its total, each share is
its own numerator over its own total, the quartiles and the unscored sum to
all employers, and every printed share matches the run's own summary. The file
is a bare tabular; the appendix supplies the float and caption.

Export read: 3_register_mona/exports/2026-09-24_1037_s93/
  uncounted_share.csv, 93_summary.txt
Output: output/tables/tableA_uncounted.tex

    python 4_exhibits/21_tab_uncounted.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

# Script 93: script 79's counts cut on the occupation-mix quartiles.
S93 = EXPORTS / "2026-09-24_1037_s93"

YEARS = list(range(2019, 2026))
# Export key, column heading.
COLUMNS = [("all", "All employers"), ("Q4", "Top exposure quartile"),
           ("Q1-Q3", "Quartiles 1 to 3")]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
BAND = "all"          # the declaration carries no age, so there is one band
PARTS = ["n_counted", "n_no_register", "n_register_no_birth_or_sex",
         "n_outside_age_range"]
SHARE = re.compile(r"^([0-9]+\.[0-9]{3})$")
# 93_summary.txt: year, all, Q4, Q1-Q3, unscored, gap.
SUMMARY_ROW = re.compile(
    r"^\s+(\d{4})\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([+-][0-9.]+)\s*$")


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
        m = SUMMARY_ROW.match(line)
        if m:
            y = int(m.group(1))
            said[(y, "all")] = m.group(2)
            said[(y, "Q4")] = m.group(3)
            said[(y, "Q1-Q3")] = m.group(4)
    if not said:
        raise SystemExit(f"  {path.name}: part C reports no uncounted share, "
                         f"so there is nothing to check the export against")
    return said


def main() -> int:
    d = pd.read_csv(source(S93, "uncounted_share.csv"))
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

    said = summary_part_c(source(S93, "93_summary.txt"))
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
            r"Per cent of employer-declaration person-months. A worker enters the employment panel only if a birth year and a sex can be read from the 2023, 2021 or 2019 individual register; the table reports the complement. Quartiles are the employer's 2019 occupation-mix exposure quartile, and person-months at employers the 2019 score book does not score are outside the two quartile columns but inside the first. 2019 is the first year of the employer declarations and its gap takes the opposite sign, so the pre-period averages quoted in the text run from 2020; 2025 covers the first half of the year. The employer declaration carries no age of its own, so the share cannot be broken down by age band.",
            r"\end{minipage}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_uncounted.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
25_tab_nonmatch.py: Online Appendix Table A42 (Section VI.1), occupation
non-match over all employer-declaration person-months, by age, rebuilt from
the raw declarations (script 99, part D).

The denominator is every distinct employer, person and month in the
declarations, before any exclusion by age, linkage or occupation. A
person-month has no code under the register rule of Table A30, Panel A: up
to 2022 the year's own register, and from 2023 the most recent code in the
2023, 2022 and 2021 registers. In the export, a code is therefore missing
when the reconciliation records no code (`none`), a code from a register
older than 2021 (`carried_earlier`), or, in the years up to 2022, any code
not observed in that year's own register. Age is the calendar year minus the
birth year from the 2023, 2021 or 2019 individual register; person-months
with no birth year (band `unknown`) enter the first row only. Cells with
fewer than five person-months were suppressed before export and enter as
zero.

Nothing is written unless the first row reproduces, to two decimals, the
production non-match rate that script 99's summary prints for every year,
and the yearly totals reproduce the declared person-months the summary
states.

Export read: 3_register_mona/exports/2026-09-25_1459_s99/
  measurement_reconciliation.csv, 99_summary.txt
Output: output/tables/tableA_nonmatch.tex (a bare tabular and note; the
        appendix supplies the float and caption)

    python 4_exhibits/25_tab_nonmatch.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S99 = EXPORTS / "2026-09-25_1459_s99"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
ADULT = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-69"]
RE_D = re.compile(r"(\d{4}): ([\d,]+) declared person-months; production non-match "
                  r"([0-9.]+)% of all")


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else S99


def need(name: str) -> Path:
    p = d() / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def main() -> int:
    print(f"  export {d().name}")
    r = pd.read_csv(need("measurement_reconciliation.csv"))
    n_rows, n_suppressed = len(r), int(r.person_months.isna().sum())
    r["pm"] = r.person_months.fillna(0.0)
    if sorted(r.year.unique()) != YEARS:
        raise SystemExit(f"  the export covers {sorted(r.year.unique())}, not {YEARS}")
    # The register rule, applied to the reconciliation's code categories.
    nocode = (r.codecat.eq("none") | r.codecat.eq("carried_earlier")
              | ((r.year <= 2022) & ~r.codecat.eq("current")))
    tot = r.groupby("year").pm.sum()

    def share(num: pd.Series, den: pd.Series | None = None) -> pd.Series:
        den_s = tot if den is None else r[den].groupby("year").pm.sum()
        return 100.0 * r[num].groupby("year").pm.sum().reindex(YEARS).fillna(0) / den_s

    said = {}
    for m in RE_D.finditer(need("99_summary.txt").read_text(encoding="utf-8",
                                                            errors="replace")):
        said[int(m.group(1))] = (int(m.group(2).replace(",", "")), float(m.group(3)))
    if sorted(said) != YEARS:
        raise SystemExit("  99_summary.txt: part D does not list every year")
    all_ages = share(nocode)
    supp = r[r.person_months.isna()].groupby("year").size().reindex(YEARS).fillna(0)
    for y in YEARS:
        # Each suppressed cell held at most four person-months, so the
        # export can fall short of the summary's total by at most four
        # times the number of cells suppressed in the year.
        gap = said[y][0] - int(tot[y])
        if not 0 <= gap <= 4 * int(supp[y]):
            raise SystemExit(f"  {y}: the export sums to {int(tot[y]):,} declared "
                             f"person-months, the summary says {said[y][0]:,}, and "
                             f"the {int(supp[y])} suppressed cells cannot account "
                             f"for the gap")
        if abs(float(all_ages[y]) - said[y][1]) > 0.01:
            raise SystemExit(f"  {y}: the register rule gives {all_ages[y]:.2f} per "
                             f"cent without a code, the summary {said[y][1]}; "
                             f"nothing is written")
    print("  the yearly totals (net of suppression) and the all-ages non-match "
          "rate reproduce 99_summary.txt")

    adult = r.band.isin(ADULT)
    young = r.band.eq("22-25")
    under = r.band.eq("under22")
    over = r.band.eq("over69")
    rows = {
        "All ages": all_ages,
        "Aged 22 to 69": share(nocode & adult, adult),
        r"\quad Aged 22 to 25": share(nocode & young, young),
        "Below 22": share(nocode & under, under),
        "Above 69": share(nocode & over, over),
    }
    shares = {"Aged 22 to 69": share(adult), "Above 69": share(over)}

    def line(label: str, s: pd.Series) -> str:
        cells = " & ".join(f"{float(s[y]):.1f}" for y in YEARS)
        print(f"  {label:26s} {cells}")
        return f"{label} & {cells} \\\\"

    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           " & " + " & ".join(str(y) for y in YEARS) + r" \\", r"\midrule",
           r"\multicolumn{7}{l}{\emph{Share of person-months with no code under the register rule}} \\"]
    tex += [line(k, v) for k, v in rows.items()]
    tex += [r"\midrule", r"\multicolumn{7}{l}{\emph{Share of all person-months}} \\"]
    tex += [line(k, v) for k, v in shares.items()]
    tex.append(line("Declared person-months (millions)", tot.reindex(YEARS) / 1e6))
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent. The denominator is every distinct employer, person and month in the employer declarations, before any exclusion by age, linkage or occupation. The register rule is that of Table~\ref{tab:iv_coverage}, Panel~A: each year's own register up to 2022, and from 2023 the most recent code in the 2023, 2022 and 2021 registers. Age is the calendar year minus the birth year from the 2023, 2021 or 2019 individual register. The 0.2 per cent of person-months with no birth year (Table~\ref{tab:uncounted}) are in the first row only. 2025 covers the first half of the year. Cells with fewer than five person-months are suppressed before export, "
            + f"{n_suppressed:,} of {n_rows:,}"
            + r". The paper's employment counts use no worker's own occupation code.",
            r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_nonmatch.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

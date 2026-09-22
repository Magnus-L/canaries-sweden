#!/usr/bin/env python3
"""
l24_tab_descriptive_bands.py: Online Appendix Table III.1, the descriptive
counterpart of the within-employer age design.

WHAT THE TABLE REPORTS
Script 66 counts workers by employer, age band and month on the 2019
exposure classification the regressions use and sums them over two
windows: the pre-adoption months, January 2022 to December 2023, and the
adoption window, January 2024 to June 2025. The table gives the per cent
change between the windows by exposure quartile of the employer and by
age band, in two panels.

  Panel A  Total headcount per month, summed over every employer in the
           quartile; an employer whose count in a band falls to zero
           contributes that zero, so the panel is the change in the number
           of people the quartile's employers had on the payroll.
  Panel B  Mean headcount per employer-month over the employer-months
           with at least one worker in the band; this conditions on the
           cell being populated, and the set of populated employer-months
           is not held fixed between the windows.

Neither panel controls for anything: composition, the business cycle and
the ageing of the workforce are inside the numbers. The regression
coefficients in Table 1 are within-employer contrasts net of the fixed
effects; this table is the raw movement they are a contrast within.

INPUTS AND OUTPUTS
Reads output_66__plain_stock.csv (script 66; columns fq, age_group,
period, mean_value, total, n_firms, n_cells) from the export directory the
final-code manifest names (or one given on the command line). Writes
revision/tables/tableA_descriptive_bands.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l24_tab_descriptive_bands.py [export_dir]

IN THE PAPER
Online Appendix III.1, Table tab:descriptive_bands; quoted in Section 3.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

ROUND2_66 = REV / "output" / "round2_20260921-0733-jobs646567"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
QUARTILES = [1, 2, 3, 4]
MONTHS = {"pre": 24, "post": 18}   # 2022-01 to 2023-12; 2024-01 to 2025-06


def source(default_dir: Path, name: str) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def change(d: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per cent change from the pre to the post window, band by quartile."""
    w = d.pivot_table(index=["fq", "age_group"], columns="period",
                      values=value, aggfunc="first")
    if w.isna().any().any():
        raise SystemExit(f"  a {value} cell is missing in the export")
    return (100 * (w["post"] / w["pre"] - 1)).unstack("fq")


def main() -> int:
    d = pd.read_csv(source(ROUND2_66, "output_66__plain_stock.csv"))
    d = d[d.fq.isin(QUARTILES) & d.age_group.isin(BANDS)].copy()
    d["per_month"] = d.total / d.period.map(MONTHS)
    a = change(d, "per_month")
    b = change(d, "mean_value")

    def block(tab: pd.DataFrame, title: str) -> list[str]:
        L = [r"\multicolumn{5}{l}{\textit{" + title + r"}} \\",
             r"\addlinespace[2pt]",
             r"Age band & Q1 (least) & Q2 & Q3 & Q4 (most) \\", r"\midrule"]
        for band in BANDS:
            cells = " & ".join(f"{tab.loc[band, q]:+.1f}" for q in QUARTILES)
            L.append(f"{band} & {cells} \\\\")
            print(f"  {title[:7]:7s} {band:6s} {cells}")
        return L

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Descriptive counterpart of the headline design: "
           r"employment by exposure quartile and age band, per cent change "
           r"from the pre-adoption months to the adoption window.}",
           r"\label{tab:descriptive_bands}", r"\footnotesize",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Exposure quartile of the employer "
           r"(2019 education mix)} \\",
           r"\cmidrule(lr){2-5}"]
    tex += block(a, "Panel A. Total headcount per month")
    tex += [r"\addlinespace[6pt]"]
    tex += block(b, "Panel B. Mean headcount per populated employer-month")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent change between the pre-adoption months (January 2022 "
            r"to December 2023) and the adoption window (January 2024 to "
            r"June 2025), on the same firm classification the regressions "
            r"use. Panel A sums headcount over every employer in the "
            r"quartile and divides by the months in the window, so an "
            r"employer whose count in a band falls to zero contributes that "
            r"zero. Panel B averages over employer-months with at least one "
            r"worker in the band; it conditions on the cell being populated, "
            r"and the set of populated employer-months is not held fixed "
            r"between the windows. Descriptive: no controls, so composition, "
            r"the business cycle and the ageing of the workforce are inside "
            r"every number. The regression coefficients in Table~1 of the "
            r"paper remove what the fixed effects remove and are contrasts, "
            r"not levels. Source: script 66 (\texttt{plain\_stock.csv}).",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_descriptive_bands.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

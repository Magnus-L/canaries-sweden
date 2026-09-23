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
Reads occ_route_descriptive_full.csv (script 85 part D, lane 31) from the
export directory pinned below, or output_66__plain_stock.csv if that is
what the directory holds, which is the education route's own version and
what this table stood on until 23 September 2026. Both carry the same
columns: fq, age_group, period, mean_value, total, n_firms, n_cells. Writes
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

LANE31 = REV / "output" / "round3_20260923-1125-lane31"
# The education route's own export, kept reachable by passing its directory.
ROUND2_66 = REV / "output" / "round2_20260921-0733-jobs646567"
NAMES = ("occ_route_descriptive_full.csv", "output_66__plain_stock.csv")
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
QUARTILES = [1, 2, 3, 4]
MONTHS = {"pre": 24, "post": 18}   # 2022-01 to 2023-12; 2024-01 to 2025-06


def source(default_dir: Path, names=NAMES) -> Path:
    """The two routes name the same frame differently, so the file is found
    by trying both rather than by assuming which directory was given."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    for n in names:
        p = d / n
        if p.exists():
            print(f"  reading {p}")
            return p
    raise SystemExit(f"  missing input: none of {names} in {d}")


def change(d: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per cent change from the pre to the post window, band by quartile."""
    w = d.pivot_table(index=["fq", "age_group"], columns="period",
                      values=value, aggfunc="first")
    if w.isna().any().any():
        raise SystemExit(f"  a {value} cell is missing in the export")
    return (100 * (w["post"] / w["pre"] - 1)).unstack("fq")


def main() -> int:
    d = pd.read_csv(source(LANE31))
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
           r"(2019 occupation mix)} \\",
           r"\cmidrule(lr){2-5}"]
    tex += block(a, "Panel A. Total headcount per month")
    tex += [r"\addlinespace[6pt]"]
    tex += block(b, "Panel B. Mean headcount per populated employer-month")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent change from the pre-adoption months (January 2022 to "
            r"December 2023) to the adoption window (January 2024 to June "
            r"2025), by the exposure quartile the regressions use. Panel A: "
            r"total headcount per month over every employer in the quartile, "
            r"zeros included. Panel B: mean headcount per employer-month with "
            r"at least one worker in the band, so it conditions on a populated "
            r"cell, and the set of such cells is not fixed across the windows. "
            r"No controls enter. Source: script 66 (\texttt{plain\_stock.csv}).",
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

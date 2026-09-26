#!/usr/bin/env python3
"""
05_tab_descriptive_bands.py: Online Appendix Table A10 (Section III.1), the
descriptive counterpart of the within-employer age design.

Workers counted by employer, age band and month on the 2019 exposure
classification the regressions use, summed over the pre-adoption months
(January 2022 to December 2023) and the adoption window (January 2024 to June
2025); the table gives the per cent change between the two by exposure
quartile and age band. Panel A: total headcount per month over every employer
in the quartile, zeros included. Panel B: mean headcount per employer-month
with at least one worker in the band. No controls enter.

Export read: 3_register_mona/exports/2026-09-23_1125_s85/occ_route_descriptive_full.csv
(script 85, part D). Script 66 wrote the same columns on the education-based
score (2026-09-21_0733_s66/output_66__plain_stock.csv); that file is read if its
folder is given on the command line.
Output: output/tables/tableA_descriptive_bands.tex

    python 4_exhibits/05_tab_descriptive_bands.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S85 = EXPORTS / "2026-09-23_1125_s85"
# Script 66's export on the education-based score, read if its folder is given.
S66 = EXPORTS / "2026-09-21_0733_s66"
NAMES = ("occ_route_descriptive_full.csv", "output_66__plain_stock.csv")

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
    d = pd.read_csv(source(S85))
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
           r"from the pre-adoption months (January 2022 to December 2023) to the adoption window (January 2024 to June 2025).}",
           r"\label{tab:descriptive_bands}", r"\footnotesize",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Exposure quartile of the employer "
           r"(2019 occupation mix)} \\",
           r"\cmidrule(lr){2-5}"]
    tex += block(a, "Panel A. Total headcount per month")
    tex += [r"\addlinespace[6pt]"]
    tex += block(b, "Panel B. Mean headcount per populated employer-month")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent change from the pre-adoption months (January 2022 to December 2023) to the adoption window (January 2024 to June 2025), by the employer's exposure quartile. Panel~A: total headcount per month over every employer in the quartile, zeros included. Panel~B: mean headcount per employer-month with at least one worker in the band, which conditions on a populated cell. No controls enter.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_descriptive_bands.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

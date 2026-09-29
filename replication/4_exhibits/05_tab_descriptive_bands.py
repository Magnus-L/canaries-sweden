#!/usr/bin/env python3
"""
05_tab_descriptive_bands.py: Online Appendix Table A12 (Section III.1), the
descriptive counterpart of the within-employer age design.

Workers counted by employer, age band and month on the 2019 exposure
classification the regressions use, summed over the pre-adoption months
(January 2022 to December 2023) and the adoption window (January 2024 to June
2025); the table gives the per cent change between the two by exposure
quartile and age band. Panel A: total headcount per month over every employer
in the quartile, zeros included. Panel B: mean headcount per employer-month
with at least one worker in the band. Panel C: hires per month. Panel D: the
monthly hire rate in the pre-adoption months, hires over headcount. No controls
enter.

Exports read: 3_register_mona/exports/2026-09-29_1317_s109/occ_route_plain_flows.csv
(script 109, Panels C and D; its headcount reproduces script 85's in every cell)
and 3_register_mona/exports/2026-09-23_1125_s85/occ_route_descriptive_full.csv
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
S109 = EXPORTS / "2026-09-29_1317_s109" / "occ_route_plain_flows.csv"
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


def num(v: float) -> str:
    """Signed, one decimal, TeX minus, as the paper prints it."""
    return f"{v:+.1f}".replace("-", "$-$")


def label(band: str) -> str:
    return band.replace("-", "--")


def main() -> int:
    d = pd.read_csv(source(S85))
    d = d[d.fq.isin(QUARTILES) & d.age_group.isin(BANDS)].copy()
    d["per_month"] = d.total / d.period.map(MONTHS)
    a = change(d, "per_month")
    b = change(d, "mean_value")

    f = pd.read_csv(S109)
    print(f"  reading {S109}")
    f = f[f.fq.isin(QUARTILES) & f.age_group.isin(BANDS)].copy()
    f["per_month"] = f.total / f.period.map(MONTHS)
    # The gate: 109's headcount must equal 85's, cell for cell.
    k = ["fq", "age_group", "period"]
    g = d.merge(f[f.outcome == "n_emp"], on=k, suffixes=("", "_109"))
    if len(g) != len(d) or (g.total != g.total_109).any():
        raise SystemExit("  109's headcount does not reproduce 85's")
    h = f[f.outcome == "n_hire"]
    c = change(h, "per_month")
    pre = lambda x: x[x.period == "pre"].set_index(["fq", "age_group"]).total
    r = (100 * pre(h) / pre(f[f.outcome == "n_emp"])).unstack("fq")

    def block(tab: pd.DataFrame, title: str, fmt=num) -> list[str]:
        L = [r"\multicolumn{5}{l}{\textit{" + title + r"}} \\",
             r"\addlinespace[2pt]",
             r"Age band & Q1 (least) & Q2 & Q3 & Q4 (most) \\", r"\midrule"]
        for band in BANDS:
            cells = " & ".join(fmt(tab.loc[band, q]) for q in QUARTILES)
            L.append(f"{label(band)} & {cells} \\\\")
            print(f"  {title[:7]:7s} {band:6s} {cells}")
        return L

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Descriptive counterpart of the headline design: "
           r"employment and hiring by exposure quartile and age band, "
           r"January 2022 to December 2023 against the later period "
           r"(January 2024 to June 2025).}",
           r"\label{tab:descriptive_bands}", r"\footnotesize",
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Exposure quartile of the employer "
           r"(2019 occupation mix)} \\",
           r"\cmidrule(lr){2-5}"]
    tex += block(a, "Panel A. Total monthly headcount, per cent change")
    tex += [r"\addlinespace[6pt]"]
    tex += block(b, "Panel B. Mean headcount per populated employer-month, "
                    "per cent change")
    tex += [r"\addlinespace[6pt]"]
    tex += block(c, "Panel C. Total monthly hires, per cent change")
    tex += [r"\addlinespace[6pt]"]
    tex += block(r, "Panel D. Monthly hire rate, January 2022 to December "
                    "2023, per cent", fmt=lambda v: f"{v:.1f}")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Per cent change in average monthly headcount or hires from January 2022 to December 2023 to January 2024 to June 2025, by the employer's exposure quartile; for example, $-8.5$ means that the monthly headcount aged 22--25 summed over top-quartile employers was 8.5 per cent lower in the later period. Panel~A: total headcount per month over every employer in the quartile, zeros included. Panel~B: mean headcount per employer-month with at least one worker in the band, which conditions on a populated cell. Panel~C: total hires per month, a hire being a worker present at an employer in a month and absent from it the month before. Panel~D: monthly hires as a percentage of monthly headcount over January 2022 to December 2023. No controls enter.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_descriptive_bands.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

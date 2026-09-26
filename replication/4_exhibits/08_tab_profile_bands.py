#!/usr/bin/env python3
"""
08_tab_profile_bands.py: Online Appendix Table A13 (Section III.2), the age
profile inside exposed employers after adoption: every band against 41-49,
and the oldest band split at 65.

Column one is the six-band profile that Figure 2 of the paper draws (script
82, part B); column two splits the 50-and-over band at 65 and re-estimates
all seven bands (script 85, part S). Both remove the calendar cycle, both use
the 2019 occupation-mix score, and they are different panels, so the employer
count of each is printed. Every standard error of both columns must equal the
square root of its own covariance diagonal to six decimals, or nothing is
written. The table carries the label tab:profile_bands.

Exports read (3_register_mona/exports/):
  2026-09-23_0655_s82-partB_s83-partsBCD/  occ_route_profile.csv,
      vcov_s82_profile_six_band.csv
  2026-09-23_1125_s85/  occ_route_split65.csv, vcov_s78_prof_split.csv
Output: output/tables/tableA_profile_split65.tex

    python 4_exhibits/08_tab_profile_bands.py [export_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S85 = EXPORTS / "2026-09-23_1125_s85"
S82B = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"

# The six-band profile is Figure 2's; the seven-band one splits its
# oldest band. 41--49 is the reference of both.
BANDS_SIX = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
BANDS_SEVEN = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-64", "65-69"]
ROWS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+", "50-64", "65-69"]
REFERENCE = "41-49"
DASH = "---"                    # the band this column does not estimate
SE_DP = 6                       # the second record, decimals
CELL = re.compile(r"^\$([-+][0-9.]+)(\^\{\*\})?\$ \(([0-9.]+)\)$")


def source(name: str, default_dir: Path) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def row_label(band: str) -> str:
    """'50+' reads as prose in the stub; the rest take an en-dash range."""
    return "50 and over" if band == "50+" else band.replace("-", "--")


def vcov_term(band: str) -> str:
    """The band's own adoption term as the covariance export names it."""
    return "gpt_x_high_" + band.replace("-", "_").replace("+", "p")


def cell(band: str, c: float, se: float) -> str:
    star = "^{*}" if abs(c) > 1.96 * se else ""
    out = f"${c:+.4f}{star}$ ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {band}: the formatted estimate is unreadable")
    if abs(float(m.group(1)) - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {band}: the printed estimate {out} disagrees "
                         f"with the export ({c:+.6f}, {se:.6f})")
    return out


def one_count(d: pd.DataFrame, name: str) -> int:
    n = sorted(set(int(x) for x in d.n_firms))
    if len(n) != 1:
        raise SystemExit(f"  {name}: one employer count expected, found {n}")
    return n[0]


def arm(csv: str, vcov: str, folder: Path, bands: list[str],
        tag: str) -> tuple[dict[str, str], int]:
    """One fitted column: its estimates, checked against its own
    clustered covariance, and the employer count of its panel."""
    d = pd.read_csv(source(csv, folder))
    missing = [b for b in bands if (d.band == b).sum() != 1]
    if missing:
        raise SystemExit(f"  {csv}: one row expected per band, not for "
                         f"{missing}")
    extra = sorted(set(d.band) - set(bands))
    if extra:
        raise SystemExit(f"  {csv}: unexpected bands {extra}; the column "
                         f"would print a profile the note does not describe")
    d = d.set_index("band")
    ref = d.loc[REFERENCE]
    if str(ref.status) != "reference" or float(ref.coef) != 0.0:
        raise SystemExit(f"  {csv}: {REFERENCE} is not exported as the "
                         f"reference")
    n = one_count(d, csv)

    # The second record: the clustered covariance of the same fit.
    v = pd.read_csv(source(vcov, folder), index_col=0)
    for band in bands:
        if band == REFERENCE:
            continue
        t = vcov_term(band)
        if t not in v.columns or t not in v.index:
            raise SystemExit(f"  {band}: {t} is not in {vcov}, so the "
                             f"estimate has only one record")
        said = float(np.sqrt(float(v.loc[t, t])))
        got = float(d.loc[band, "se"])
        if round(said, SE_DP) != round(got, SE_DP):
            raise SystemExit(f"  {band}: {csv} reports a standard error of "
                             f"{got:.6f} and its own covariance {said:.6f}; "
                             f"the table is not written")
    print(f"  {tag}: the covariance reproduces every standard error to "
          f"{SE_DP} decimals")

    cells = {}
    for band in bands:
        if band == REFERENCE:
            continue
        r = d.loc[band]
        if str(r.status) != "ok":
            raise SystemExit(f"  {band}: status '{r.status}', not quotable")
        cells[band] = cell(band, float(r.coef), float(r.se))
    return cells, n


def main() -> int:
    six, n_six = arm("occ_route_profile.csv", "vcov_s82_profile_six_band.csv",
                     S82B, BANDS_SIX, "six bands")
    seven, n_seven = arm("occ_route_split65.csv", "vcov_s78_prof_split.csv",
                         S85, BANDS_SEVEN, "seven bands")

    # Every printed row must be estimated by at least one column, and the
    # reference by both, or the table would carry an all-dash line.
    orphan = [b for b in ROWS
              if b != REFERENCE and b not in six and b not in seven]
    if orphan:
        raise SystemExit(f"  no column estimates {orphan}")

    rows = []
    for band in ROWS:
        stub = row_label(band)
        if band == REFERENCE:
            rows.append(f"{stub} & reference & reference \\\\")
            print(f"  {band:6s} reference                reference")
            continue
        a = six.get(band, DASH)
        b = seven.get(band, DASH)
        rows.append(f"{stub} & {a} & {b} \\\\")
        print(f"  {band:6s} {a:24s} {b}")
    print(f"  panels {n_six:,} employers on six bands, {n_seven:,} on seven")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The age profile inside exposed employers after "
           r"adoption: every band against 41--49, and the oldest band split "
           r"at 65.}",
           r"\label{tab:profile_bands}", r"\footnotesize",
           r"\begin{tabular}{lcc}", r"\toprule",
           r"Band, against 41--49 & Six bands & "
           r"Seven bands, 50 and over split at 65 \\",
           r"\midrule"]
    tex += rows
    tex += [r"\midrule",
            f"Employers & {thousands(n_six)} & {thousands(n_seven)} \\\\",
            r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson with employer-by-month, employer-by-age and month-by-age effects, calendar cycle removed, treatment January 2024, exposure the employer's 2019 occupation mix, clustered by employer. Both columns report the step from January 2024; Figure~2 of the paper draws $\tau$ on the eight-band panel of Table~\ref{tab:final_checks}, Panel~B. The second column splits the oldest band at 65 on its own, smaller panel. Only the 65--69 band contains the ages the 2020 and 2023 increases in the pension age reach. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_profile_split65.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

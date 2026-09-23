#!/usr/bin/env python3
"""
l42_tab_profile_split65_v3.py: Online Appendix Table III.3, the age
profile on the occupation route, in two columns: the six-band profile
and the same profile with the 50-and-over band split at 65.

WHY A NEW SCRIPT. l26 draws the same table on the EDUCATION route: its
pinned export is lane 25's prof_split.csv and its note says exposure is
frozen at the 2019 education mix. The v3 paper scores an employer from
the 2019 occupations of its own incumbents, and lane 31 part S refitted
the split on that score. l26 stays as it is so the v2 table remains
reproducible.

WHY TWO COLUMNS SINCE 23 SEPTEMBER 2026. The six-band profile is the
one Figure 2 draws and the one the paper argues from, and until now it
appeared in the appendix only as numbers inside a paragraph. The split
at 65 answers the pension-age rival and was the only column here. A
reader cannot check the rival against the profile if the profile is not
beside it, so both are printed, and the note says in one sentence that
they are different samples.

THE LABEL CHANGED WITH THE SECOND COLUMN. This table is no longer only
the split, so it carries \\label{tab:profile_bands} and not
\\label{tab:profile_split65}. The appendix reference has to follow.

THE SECOND RECORD IS THE COVARIANCE, NOT THE SUMMARY. l26 checked every
printed estimate against part E of the run's own summary, so that no
table was written on one record alone. Neither lane's folder carries the
summary of the job that fitted these arms, so that text does not exist
for either fit. The covariance export is a better second record anyway:
it comes from the same fit and is not a re-print of the same numbers,
and the standard error of each band is the square root of its own
diagonal entry. Every band of BOTH columns is checked against its own
covariance to six decimals and the table is not written if one
disagrees.

WHAT IT PRINTS
Eight rows against the 41--49 reference, in one order: the four young
bands, the reference, 50 and over, 50--64, 65--69. The six-band column
fills 50 and over and dashes the two halves; the seven-band column does
the reverse. A star at five per cent, the calendar cycle removed in
both, and the employer count of each panel as the last row, since the
two are different samples and the split is read beside the profile
rather than in place of it.

    python3 revision/local/l42_tab_profile_split65_v3.py [export_dir]

INPUTS AND OUTPUTS
Reads occ_route_profile.csv and vcov_s82_profile_six_band.csv from the
lane 28b export, and occ_route_split65.csv and vcov_s78_prof_split.csv
from the lane 31 export, both pinned below. Writes
revision/tables/tableA_profile_split65.tex and copies it to
canaries-sweden-paper/tables/.

IN THE PAPER
Online Appendix III.2, the paragraph on the age profile and the one on
the oldest band split at 65, Table tab:profile_bands.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE31 = OUT / "round3_20260923-1125-lane31"
LANE28B = OUT / "round3_20260923-0655-lanes28b-29bcd"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

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


def arm(csv: str, vcov: str, lane: Path, bands: list[str],
        tag: str) -> tuple[dict[str, str], int]:
    """One fitted column: its estimates, checked against its own
    clustered covariance, and the employer count of its panel."""
    d = pd.read_csv(source(csv, lane))
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
    v = pd.read_csv(source(vcov, lane), index_col=0)
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
                     LANE28B, BANDS_SIX, "six bands")
    seven, n_seven = arm("occ_route_split65.csv", "vcov_s78_prof_split.csv",
                         LANE31, BANDS_SEVEN, "seven bands")

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
            r"The first column is the six-band profile the paper's Figure 2 "
            r"draws; the second splits the oldest band at 65 and re-estimates "
            r"all seven bands. The two columns are different samples "
            f"({thousands(n_six)} and {thousands(n_seven)} employers), so the "
            r"split is read beside the profile rather than in place of it. "
            r"The calendar cycle is removed in both. Exposure is the "
            r"employer's 2019 occupation mix; Poisson with employer-by-month, "
            r"employer-by-age and month-by-age effects, treatment January "
            r"2024, standard errors clustered by employer. Only the 65--69 "
            r"band contains the ages the 2020 and 2023 increases in the "
            r"pension age reach, and the 50--64 band gains without them. "
            r"$^{*}$ $p<0.05$. Source: script 82, part B (six bands) and "
            r"script 85, part S (seven bands).",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_profile_split65.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(REV.parent)}")
    if PAPER_TAB.exists():
        shutil.copy2(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    print("  LABEL CHANGED: tab:profile_split65 -> tab:profile_bands; the "
          "appendix reference must follow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

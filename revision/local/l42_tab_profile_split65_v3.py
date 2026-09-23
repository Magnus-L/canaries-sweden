#!/usr/bin/env python3
"""
l42_tab_profile_split65_v3.py: Online Appendix Table III.3, the age
profile with the 50-and-over band split at 65, on the occupation route.

WHY A NEW SCRIPT. l26 draws the same table on the EDUCATION route: its
pinned export is lane 25's prof_split.csv and its note says exposure is
frozen at the 2019 education mix. The v3 paper scores an employer from
the 2019 occupations of its own incumbents, and lane 31 part S refitted
the split on that score. l26 stays as it is so the v2 table remains
reproducible.

THE SECOND RECORD IS THE COVARIANCE, NOT THE SUMMARY. l26 checked every
printed estimate against part E of the run's own summary, so that no
table was written on one record alone. Lane 31's folder was copied
before the three-part job wrote its summary, so that text does not
exist for this fit. The covariance export is a better second record
anyway: it comes from the same fit and is not a re-print of the same
numbers, and the standard error of each band is the square root of its
own diagonal entry. Every band is checked against it to six decimals
and the table is not written if one disagrees.

WHAT IT PRINTS
Seven bands against 41--49 with the calendar cycle removed, a star at
five per cent, and the employer count of the seven-band panel beside the
six-band one, since the two are different samples and the split is read
beside the profile rather than in place of it.

    python3 revision/local/l42_tab_profile_split65_v3.py [export_dir]

INPUTS AND OUTPUTS
Reads occ_route_split65.csv and vcov_s78_prof_split.csv from the lane 31
export pinned below, and occ_route_profile.csv from the lane 28b export
for the six-band employer count. Writes
revision/tables/tableA_profile_split65.tex and copies it to
canaries-sweden-paper/tables/.

IN THE PAPER
Online Appendix III.2, the paragraph on the oldest band split at 65,
Table tab:profile_split65.
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

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-64", "65-69"]
REFERENCE = "41-49"
TERM = "gpt_x_high_{}"          # the band's own adoption term in the vcov
SE_DP = 6                       # the second record, decimals
CELL = re.compile(r"^\$([-+][0-9.]+)(\^\{\*\})?\$ \(([0-9.]+)\)$")


def source(name: str, default_dir: Path = LANE31) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


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


def main() -> int:
    d = pd.read_csv(source("occ_route_split65.csv"))
    missing = [b for b in BANDS if (d.band == b).sum() != 1]
    if missing:
        raise SystemExit(f"  occ_route_split65.csv: one row expected per "
                         f"band, not for {missing}")
    d = d.set_index("band")
    ref = d.loc[REFERENCE]
    if str(ref.status) != "reference" or float(ref.coef) != 0.0:
        raise SystemExit(f"  {REFERENCE} is not exported as the reference")
    n_split = one_count(d, "occ_route_split65.csv")

    # The second record: the clustered covariance of the same fit.
    v = pd.read_csv(source("vcov_s78_prof_split.csv"), index_col=0)
    for band in BANDS:
        if band == REFERENCE:
            continue
        t = TERM.format(band.replace("-", "_"))
        if t not in v.columns or t not in v.index:
            raise SystemExit(f"  {band}: {t} is not in the covariance export, "
                             f"so the estimate has only one record")
        said = float(np.sqrt(float(v.loc[t, t])))
        got = float(d.loc[band, "se"])
        if round(said, SE_DP) != round(got, SE_DP):
            raise SystemExit(f"  {band}: the export reports a standard error "
                             f"of {got:.6f} and its own covariance "
                             f"{said:.6f}; the table is not written")
    print(f"  the covariance reproduces every standard error to {SE_DP} "
          f"decimals")

    six = pd.read_csv(source("occ_route_profile.csv", LANE28B))
    n_six = one_count(six, "occ_route_profile.csv")

    rows = []
    for band in BANDS:
        label = band.replace("-", "--")
        if band == REFERENCE:
            rows.append(f"{label} & reference \\\\")
            print(f"  {band:6s} reference")
            continue
        r = d.loc[band]
        if str(r.status) != "ok":
            raise SystemExit(f"  {band}: status '{r.status}', not quotable")
        c = cell(band, float(r.coef), float(r.se))
        rows.append(f"{label} & {c} \\\\")
        print(f"  {band:6s} {c}")
    print(f"  panel {n_split:,} employers, against {n_six:,} on six bands")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The age profile with the 50-and-over band split at 65: "
           r"every band against 41--49, calendar cycle removed.}",
           r"\label{tab:profile_split65}", r"\footnotesize",
           r"\begin{tabular}{lc}", r"\toprule",
           r"Band, against 41--49 & Calendar cycle removed \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.88\textwidth}\footnotesize\vspace{4pt}",
            r"The age profile with the oldest band split at 65, all seven "
            f"bands in one panel of {thousands(n_split)} employers; Poisson, "
            r"employer-by-month, employer-by-age and month-by-age effects, "
            r"exposure the employer's 2019 occupation mix, treatment January "
            r"2024, clustered by employer. The panel is smaller than the "
            f"six-band one ({thousands(n_six)} employers), so the split is "
            r"read beside the profile rather than in place of it. Of the "
            r"bands here, only 65--69 contains the ages the 2020 and 2023 "
            r"increases in the pension age reach; the 50--64 band gains "
            r"without them. $^{*}$ $p<0.05$. Source: script 85, part S.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_profile_split65.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(REV.parent)}")
    if PAPER_TAB.exists():
        shutil.copy2(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

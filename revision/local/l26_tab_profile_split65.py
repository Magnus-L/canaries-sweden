#!/usr/bin/env python3
"""
l26_tab_profile_split65.py: Online Appendix Table III.2
(tab:profile_split65), the age profile with the 50-and-over band split at
the pension age.

THE QUESTION
The profile against 41-49 has one large gain, the 50-and-over band. The
2020 and 2023 increases in the Swedish pension age reach workers in their
late sixties, so a gain that sits only there would be a fact about the
pension reform rather than about AI. Splitting the oldest band at 65 asks
whether the half of it the reform does not reach gains on its own.

WHAT IS ESTIMATED
Poisson pseudo-maximum likelihood on employer by age by month counts,
with employer-by-month, employer-by-age and month-by-age effects,
exposure frozen at the employer's 2019 education mix, treatment dated
January 2024, three calendar-quarter terms per band so that the calendar
cycle is removed, and standard errors clustered by employer (script 78,
lane 25 part E). Exposure is constant within an employer-month, so the
band levels are collinear with the employer-by-month effects and one band
must be the reference: 41-49 carries no estimate, and every other
coefficient is the post-adoption change of that band relative to 41-49
inside exposed employers.

The seven-band panel is smaller than the six-band panel of
Table tab:profile_seasonal, so the two are the same contrast estimated on
different samples and the split is read beside the six-band profile
rather than in place of it. Both employer counts are read from the
exports and printed in the note.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names (or one
directory given on the command line): prof_split.csv and 78_summary.txt
(script 78, lane 25 part E; the CSV carries band, coef, se, n_firms and
status, the summary the same seven rows as the run reported them) and
contrast_seasonal.csv (script 74, lane 20), the last only for the
employer count of the six-band panel the note compares with. Nothing is
typed in. Writes revision/tables/tableA_profile_split65.tex and copies it
to canaries-sweden-paper/tables/.

    python3 revision/local/l26_tab_profile_split65.py [export_dir]

THE GATE
Every printed estimate is read back from the string that goes into the
table and compared with the export it came from, and then against part E
of the run's own summary, which reports the same seven rows, their stars
and the employer count independently of the CSV. A disagreement beyond
half of the last printed digit, a star the summary does not carry, or an
employer count the summary does not confirm stops the script and nothing
is written. The reference row must be exported as the reference, each
band must appear exactly once, and the panel must report a single
employer count.

IN THE PAPER
Online Appendix III.2, the paragraph on the oldest band split at 65,
Table tab:profile_split65.
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
LANE20 = OUT / "round3_20260922-0105-lane20-seasonal-contrast"
LANE25 = OUT / "round3_20260922-1237-lane25bc-BCEF"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-64", "65-69"]
REFERENCE = "41-49"
CELL = re.compile(r"^\$([-+][0-9.]+)(\^\{\*\})?\$ \(([0-9.]+)\)$")
# Part E of the run's summary: one line per band, and the employer count.
SUMMARY_ROW = re.compile(r"^\s+(\S+)\s+([-+][0-9.]+)\s+\(([0-9.]+)\)(\s+\*)?\s*$")
SUMMARY_FIRMS = re.compile(r"^\s+firms\s+([0-9,]+)")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: int) -> str:
    """An employer count as the paper prints it."""
    return f"{n:,}".replace(",", "{,}")


def cell(band: str, c: float, se: float) -> str:
    """The estimate with a star at five per cent, checked against the
    export before it is allowed into the table."""
    star = "^{*}" if abs(c) > 1.96 * se else ""
    out = f"${c:+.4f}{star}$ ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {band}: the formatted estimate is unreadable")
    if abs(float(m.group(1)) - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {band}: the printed estimate {out} disagrees "
                         f"with prof_split.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_part_e(path: Path) -> tuple[dict[str, str], int]:
    """Part E of the run's summary as the run itself reported it: the
    printed estimate of each band, with its star, and the employer count.
    This is the second record of the same fit, and the table is written
    only if the two agree."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows, firms, inside = {}, None, False
    for line in lines:
        if re.match(r"^E\.\s", line):
            inside = True
            continue
        if inside and re.match(r"^([A-Z]\.\s|[A-Z][A-Z ]+:)", line):
            break
        if not inside:
            continue
        m = SUMMARY_ROW.match(line)
        if m:
            star = "^{*}" if m.group(4) else ""
            rows[m.group(1)] = f"${m.group(2)}{star}$ ({m.group(3)})"
            continue
        f = SUMMARY_FIRMS.match(line)
        if f:
            firms = int(f.group(1).replace(",", ""))
    if not rows or firms is None:
        raise SystemExit(f"  {path.name}: part E does not report the seven "
                         f"bands and the employer count")
    return rows, firms


def one_count(d: pd.DataFrame, name: str) -> int:
    """The employer count of a panel, which the export must report once."""
    n = sorted(set(int(x) for x in d.n_firms))
    if len(n) != 1:
        raise SystemExit(f"  {name}: one employer count expected, found {n}")
    return n[0]


def main() -> int:
    d = pd.read_csv(source(LANE25, "prof_split.csv"))
    missing = [b for b in BANDS if (d.band == b).sum() != 1]
    if missing:
        raise SystemExit(f"  prof_split.csv: one row expected per band, "
                         f"not for {missing}")
    d = d.set_index("band")
    ref = d.loc[REFERENCE]
    if str(ref.status) != "reference" or float(ref.coef) != 0.0:
        raise SystemExit(f"  prof_split.csv: {REFERENCE} is not exported as "
                         f"the reference band")
    n_split = one_count(d, "prof_split.csv")
    six = pd.read_csv(source(LANE20, "contrast_seasonal.csv"))
    n_six = one_count(six, "contrast_seasonal.csv")
    said, said_firms = summary_part_e(source(LANE25, "78_summary.txt"))
    if said_firms != n_split:
        raise SystemExit(f"  the summary reports {said_firms:,} employers and "
                         f"prof_split.csv {n_split:,}, so the two exports are "
                         f"not from the same fit")

    rows = []
    for band in BANDS:
        label = band.replace("-", "--")
        if band == REFERENCE:
            rows.append(f"{label} & reference \\\\")
            print(f"  {band:6s} reference")
            continue
        r = d.loc[band]
        if str(r.status) != "ok":
            raise SystemExit(f"  {band}: the export reports status "
                             f"'{r.status}', so it is not quotable")
        c = cell(band, float(r.coef), float(r.se))
        if said.get(band) != c:
            raise SystemExit(f"  {band}: the table would print {c}, the run's "
                             f"summary reports {said.get(band, 'nothing')}")
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
            r"exposure frozen at the 2019 education mix, treatment January "
            r"2024, clustered by employer. The panel is smaller than the "
            f"six-band one ({thousands(n_six)} employers), so the split is "
            r"read beside the profile rather than in place of it. Of the "
            r"bands here, "
            r"only 65--69 contains the ages the 2020 and 2023 increases in the "
            r"pension age reach; the 50--64 band gains without them. "
            r"$^{*}$ $p<0.05$. Source: script 78, part E.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_profile_split65.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

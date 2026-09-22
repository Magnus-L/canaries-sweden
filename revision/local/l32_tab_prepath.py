#!/usr/bin/env python3
"""
l32_tab_prepath.py: Online Appendix Table III.2 (tab:prepath), the
quarterly path of the within-employer age gradient on the plain
specification.

THE QUESTION
The design reads a step at adoption against the level before it, so the
reader is owed the path that step is a step in, and in particular whether
the pre-period drifts. The table reports every quarter from 2019Q1 to
2025Q2 on the plain specification, which carries no calendar terms, so
that the cycle between quarters and the movement within a quarter across
years can both be read. The verdict on drift is not taken from this
table: it comes from the linear trend fitted to the pre-launch months in
part A(ii) of the same run, which is quoted in the text. The table is the
evidence the verdict is read against.

WHAT IS ESTIMATED
Poisson pseudo-maximum likelihood on employer by age by month counts,
with employer-by-month, employer-by-age and month-by-age effects and no
calendar terms, one coefficient per calendar quarter interacted with
exposure and the young band, 2022Q1 omitted, and standard errors
clustered by employer (script 78, lane 25 part A). Each coefficient is
the change of the young band relative to the older bands inside exposed
employers in that quarter, read against 2022Q1. The two young bands are
fitted on their own panels, so the columns are two fits and not one, and
the panel sizes differ.

Because the plain specification carries no calendar terms, the fourth
quarter stands well above the rest in both columns throughout the panel:
that is the seasonal pattern the headline specification removes, and it
is the reason the path is read by quarter rather than as a single series.

INPUTS AND OUTPUTS
Reads prepath_plain.csv (script 78, lane 25 part A; columns young_band,
quarter, coef, se, n_obs, status), vcov_s78_prepath_<band>.csv (the
exported covariance of each fit) and 78_summary.txt, the run's own report
of the same coefficients, from the export directory the final-code
manifest names or one given on the command line. Nothing is typed in.
Writes revision/tables/tableA_prepath.tex and copies it to
canaries-sweden-paper/tables/; the appendix carries the float, the
caption and the label and inputs this file inside them.

    python3 revision/local/l32_tab_prepath.py [export_dir]

THE GATE
Each band must carry exactly one row per quarter and one panel size, the
omitted quarter must be exported as the reference with a zero
coefficient and a zero standard error, and it must appear in neither
covariance matrix, since a reference has no variance of its own. Every
other row must report status ok, and its standard error must equal the
square root of its own diagonal in that band's covariance matrix, which
is the integrity check this export offers. Every cell is then checked
against part A(i) of the run's summary, which reports the same
coefficients and their stars independently of the CSV. Finally each
printed number is read back from the string that goes into the table and
compared with the export it came from, and any disagreement beyond half
of the last printed digit stops the script and nothing is written.

IN THE PAPER
Online Appendix III.2, the paragraph on the pre-period, Table tab:prepath;
Figure fig:prepath draws the same coefficients by calendar quarter.
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
LANE25A = OUT / "round3_20260922-1333-lane25a-ADG"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there, inside the float that carries its caption.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30"]
REFERENCE = "2022Q1"
QUARTERS = [f"{y}Q{q}" for y in range(2019, 2026) for q in (1, 2, 3, 4)
            if not (y == 2025 and q > 2)]
CELL = re.compile(r"^\$([-+])\$([0-9.]+)(\$\^\{\*\}\$)? \(([0-9.]+)\)$")
SUMMARY_BAND = re.compile(r"^\s+(22-25|26-30):\s*$")
SUMMARY_ROW = re.compile(
    r"^\s+(\d{4}Q\d)\s+([-+][0-9.]+) \(([0-9.]+)\)(\s+\*)?\s*$")
SUMMARY_REF = re.compile(r"^\s+(\d{4}Q\d)\s+reference\s*$")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: int) -> str:
    """A cell count as the paper prints it."""
    return f"{n:,}".replace(",", "{,}")


def cell(what: str, c: float, se: float) -> str:
    """The estimate with a star at five per cent, in the sign-outside
    form this table uses, checked against the export before it is allowed
    into the table."""
    star = "$^{*}$" if abs(c) > 1.96 * se else ""
    out = f"${'+' if c >= 0 else '-'}${abs(c):.4f}{star} ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted estimate is unreadable")
    signed = float(m.group(1) + m.group(2))
    if abs(signed - c) > 5e-5 or abs(float(m.group(4)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed estimate {out} disagrees "
                         f"with prepath_plain.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_part_a(path: Path) -> dict[tuple[str, str], str]:
    """Part A(i) of the run's summary as the run itself reported it: the
    printed coefficient of every quarter, with its star. This is a second
    record of the same fits, and the table is written only if the two
    agree."""
    said, band = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "(ii)" in line:
            break
        m = SUMMARY_BAND.match(line)
        if m:
            band = m.group(1)
            continue
        if band is None:
            continue
        r = SUMMARY_ROW.match(line)
        if r:
            said[(band, r.group(1))] = (
                f"${r.group(2)[0]}${r.group(2)[1:]}"
                f"{'$^{*}$' if r.group(4) else ''} ({r.group(3)})")
            continue
        f = SUMMARY_REF.match(line)
        if f:
            said[(band, f.group(1))] = "reference"
    if not said:
        raise SystemExit(f"  {path.name}: part A(i) reports no quarterly path, "
                         f"so there is nothing to check the export against")
    return said


def main() -> int:
    d = pd.read_csv(source(LANE25A, "prepath_plain.csv"))
    said = summary_part_a(source(LANE25A, "78_summary.txt"))
    cells, panel = {}, {}
    for band in BANDS:
        b = d[d.young_band == band]
        bad = [q for q in QUARTERS if (b.quarter == q).sum() != 1]
        if bad or len(b) != len(QUARTERS):
            raise SystemExit(f"  prepath_plain.csv: {band} must carry one row "
                             f"per quarter; wrong for {bad or 'the row count'}")
        b = b.set_index("quarter")
        ref = b.loc[REFERENCE]
        if (str(ref.status) != "reference" or float(ref.coef) != 0.0
                or float(ref.se) != 0.0):
            raise SystemExit(f"  prepath_plain.csv: {band}'s {REFERENCE} is "
                             f"not exported as the reference")
        n = sorted(set(int(x) for x in b.loc[b.status == "ok", "n_obs"]))
        if len(n) != 1:
            raise SystemExit(f"  {band}: one panel size expected, found {n}")
        panel[band] = n[0]

        v = pd.read_csv(source(LANE25A,
                               f"vcov_s78_prepath_{band.replace('-', '_')}.csv"),
                        index_col=0)
        if f"pq_{REFERENCE}_x_high_x_young" in v.index:
            raise SystemExit(f"  {band}: the covariance carries the reference "
                             f"quarter, which has no variance of its own")
        for q in QUARTERS:
            r = b.loc[q]
            if q == REFERENCE:
                cells[(band, q)] = "reference"
            else:
                if str(r.status) != "ok":
                    raise SystemExit(f"  {band} {q}: the export reports status "
                                     f"'{r.status}', so it is not quotable")
                term = f"pq_{q}_x_high_x_young"
                if term not in v.index:
                    raise SystemExit(f"  {band} {q}: no {term} in the "
                                     f"exported covariance")
                diag = float(v.loc[term, term]) ** 0.5
                if abs(diag - float(r.se)) > 5e-5:
                    raise SystemExit(f"  {band} {q}: the exported standard "
                                     f"error {float(r.se):.6f} is not the "
                                     f"square root of its own variance "
                                     f"{diag:.6f}")
                cells[(band, q)] = cell(f"{band} {q}", float(r.coef),
                                        float(r.se))
            if said.get((band, q)) != cells[(band, q)]:
                raise SystemExit(f"  {band} {q}: the table would print "
                                 f"{cells[(band, q)]}, the run's summary "
                                 f"reports {said.get((band, q), 'nothing')}")

    rows = []
    for q in QUARTERS:
        line = " & ".join(cells[(band, q)] for band in BANDS)
        rows.append(f"{q} & {line} \\\\")
        print(f"  {q}  {line}")
    print(f"  panel {panel[BANDS[0]]:,} cells at {BANDS[0]} and "
          f"{panel[BANDS[1]]:,} at {BANDS[1]}")

    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           "Quarter & " + " & ".join(b.replace("-", "--") for b in BANDS)
           + r" \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}", "",
            r"\vspace{0.5em}",
            r"\begin{minipage}{0.86\textwidth}",
            r"\footnotesize \textit{Notes:} Poisson on employer-by-age-by-month counts, with",
            r"employer-by-month, employer-by-age and month-by-age effects and no calendar",
            f"terms; {REFERENCE} is the reference. Entries are the coefficient with the standard",
            r"error clustered by employer in parentheses; $^{*}$ marks significance at the five",
            f"per cent level. The panel holds {thousands(panel[BANDS[0]])} cells at "
            f"{BANDS[0].replace('-', '--')} and {thousands(panel[BANDS[1]])}",
            f"at {BANDS[1].replace('-', '--')}. "
            r"Figure~\ref{fig:prepath} draws the same coefficients by calendar",
            r"quarter.",
            r"\end{minipage}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_prepath.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

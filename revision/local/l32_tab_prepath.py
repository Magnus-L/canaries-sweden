#!/usr/bin/env python3
"""
l32_tab_prepath.py: Online Appendix Table III.2 (tab:prepath), the
quarterly path of the within-employer age gradient on the plain
specification, and beneath it the pre-launch drift test.

THE QUESTION
The design reads a step at adoption against the level before it, so the
reader is owed the path that step is a step in, and in particular whether
the pre-period drifts. The upper panel reports every quarter from 2019Q1
to 2025Q2 on the plain specification, which carries no calendar terms, so
that the cycle between quarters and the movement within a quarter across
years can both be read. The verdict on drift is not taken from that
panel: it comes from the linear trend fitted to the pre-launch months,
and the lower panel, added 23 September 2026, now carries that trend
rather than leaving it in running prose. The upper panel is the evidence
the verdict is read against.

THE LOWER PANEL
A linear trend on the pre-launch months, January 2021 to November 2022,
fitted with the calendar terms and the tightening window in, on the
occupation route (script 83, lane 29b part B). The pre-period counts as
flat if the trend lies within two of its own standard errors of zero.
The second row scales the monthly trend and its standard error by the
twenty-three months of the window, which is the drift the trend implies
over the pre-period; it is the same t and so carries the same star. The
third row is the tightening window itself, which is in the fit so that
the trend is not read off the months of the rate rise.

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
manifest names or one given on the command line. The lower panel reads
occ_rest_drift.csv (columns young_band, n_firms, first_month, last_month,
n_months, term, coef, se, t, flat_within_2se) and
vcov_s83_drift_<band>.csv from the lane 29b export, which is a separate
run and so is pinned separately and overridden by a second argument.
Nothing is typed in. Writes revision/tables/tableA_prepath.tex and copies
it to canaries-sweden-paper/tables/; the appendix carries the float, the
caption and the label and inputs this file inside them, so what is
written here stays a bare tabular with no float environment and no
caption of its own, both panels inside the one tabular.

    python3 revision/local/l32_tab_prepath.py [export_dir [drift_dir]]

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

The lower panel is gated the same way, with the covariance as the second
record: script 83's part B summary was lost when the MONA output folders
were flattened on export, so there is no text re-print to check the CSV
against, and each term's standard error must instead be the square root
of its own diagonal to six decimals. The exported verdict
flat_within_2se must also agree with the test recomputed from the
exported coefficient and standard error; an export that disagrees with
its own rule stops the script.

IN THE PAPER
Online Appendix III.2, the paragraph on the pre-period, Table tab:prepath;
Figure fig:prepath draws the upper panel's coefficients by calendar
quarter.
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
# Lane 32 ran script 78's part A on the occupation route, the score the
# paper reports. Lane 25a's export is the education route's and stays
# reachable by passing its directory; the two name the same files
# differently, so every name is tried in turn.
LANE32 = OUT / "round3_20260923-1234-lane32"
LANE25A = OUT / "round3_20260922-1333-lane25a-ADG"
# The drift panel is a different run from the quarterly path and has its
# own export, so it is pinned on its own and overridden by a second
# argument rather than by the first.
LANE29B = OUT / "round3_20260923-0655-lanes28b-29bcd"
PATH_NAMES = ("occ_route_prepath.csv", "prepath_plain.csv")
SUMMARY_NAMES = ("86_summary.txt", "78_summary.txt")
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there, inside the float that carries its caption.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30"]
REFERENCE = "2022Q1"
QUARTERS = [f"{y}Q{q}" for y in range(2019, 2026) for q in (1, 2, 3, 4)
            if not (y == 2025 and q > 2)]
CELL = re.compile(r"^\$([-+])\$([0-9.]+)(\$\^\{\*\}\$)? \(([0-9.]+)\)$")

# The drift panel.
TREND = "trend_x_high_x_young"
TIGHT = "rbw_x_high_x_young"
DRIFT_FIRST, DRIFT_LAST, DRIFT_MONTHS = "2021-01", "2022-11", 23
SE_DP = 6                       # the covariance check, decimals
SUMMARY_BAND = re.compile(r"^\s+(22-25|26-30):(\s*$|\s+\d+ quarters)")
SUMMARY_ROW = re.compile(
    r"^\s+(\d{4}Q\d)\s+([-+][0-9.]+) \(([0-9.]+)\)(\s+\*)?\s*$")
SUMMARY_REF = re.compile(r"^\s+(\d{4}Q\d)\s+reference\s*$")


def first_of(default_dir: Path, names) -> Path:
    """The first of `names` that the directory holds."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    for n in names:
        p = d / n
        if p.exists():
            print(f"  reading {p}")
            return p
    raise SystemExit(f"  missing input: none of {names} in {d}")


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


def cell_dp(what: str, c: float, se: float, dp: int) -> str:
    """The same cell at a chosen number of decimals, for the monthly
    trend, which is a ten-thousandth and would print as zero at four."""
    star = "$^{*}$" if abs(c) > 1.96 * se else ""
    out = (f"${'+' if c >= 0 else '-'}${abs(c):.{dp}f}{star} "
           f"({se:.{dp}f})")
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted estimate is unreadable")
    tol = 0.5 * 10 ** (-dp)
    signed = float(m.group(1) + m.group(2))
    if abs(signed - c) > tol or abs(float(m.group(4)) - se) > tol:
        raise SystemExit(f"  {what}: the printed estimate {out} disagrees "
                         f"with the export ({c:+.6f}, {se:.6f})")
    return out


def drift_source(name: str) -> Path:
    """The pinned lane 29b export, or the same file name under a
    directory given as the second argument."""
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else LANE29B
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    print(f"  reading {p}")
    return p


def drift_panel() -> tuple[list[str], dict[str, int]]:
    """The pre-launch drift test, as the rows that go under the
    quarterly path inside the same tabular.

    The test is a linear trend on the months from January 2021 to
    November 2022, fitted with the calendar terms and the tightening
    window in. The second record is the fit's own clustered covariance,
    since script 83's part B summary was lost on export; the exported
    verdict is also recomputed from the exported coefficient and
    standard error, so an export that disagrees with its own rule stops
    the script.
    """
    d = pd.read_csv(drift_source("occ_rest_drift.csv"))
    got: dict[str, dict[str, tuple[float, float]]] = {}
    flat: dict[str, bool] = {}
    firms: dict[str, int] = {}
    for band in BANDS:
        b = d[d.young_band == band]
        if b.empty:
            raise SystemExit(f"  occ_rest_drift.csv: no rows for {band}")
        bad = [t for t in (TREND, TIGHT) if (b.term == t).sum() != 1]
        if bad:
            raise SystemExit(f"  occ_rest_drift.csv: {band} must carry one "
                             f"row per term; wrong for {bad}")
        notok = sorted(set(b.loc[b.status != "ok", "status"].astype(str)))
        if notok:
            raise SystemExit(f"  {band}: occ_rest_drift.csv reports status "
                             f"{notok}, so the fit is not quotable")
        window = sorted(set(zip(b.first_month.astype(str),
                                b.last_month.astype(str),
                                b.n_months.astype(int))))
        if window != [(DRIFT_FIRST, DRIFT_LAST, DRIFT_MONTHS)]:
            raise SystemExit(f"  {band}: the drift export is fitted on "
                             f"{window}, not on {DRIFT_FIRST} to "
                             f"{DRIFT_LAST} over {DRIFT_MONTHS} months")
        n = sorted(set(int(x) for x in b.n_firms))
        if len(n) != 1:
            raise SystemExit(f"  {band}: one employer count expected in "
                             f"occ_rest_drift.csv, found {n}")
        firms[band] = n[0]
        b = b.set_index("term")

        v = pd.read_csv(drift_source(
            f"vcov_s83_drift_{band.replace('-', '_')}.csv"), index_col=0)
        for t in (TREND, TIGHT):
            if t not in v.index or t not in v.columns:
                raise SystemExit(f"  {band}: {t} is not in the drift "
                                 f"covariance, so it has one record only")
            said = float(np.sqrt(float(v.loc[t, t])))
            have = float(b.loc[t, "se"])
            if round(said, SE_DP) != round(have, SE_DP):
                raise SystemExit(f"  {band} {t}: the drift export reports a "
                                 f"standard error of {have:.6f} and its own "
                                 f"covariance {said:.6f}; not written")
            got.setdefault(band, {})[t] = (float(b.loc[t, "coef"]), have)
        print(f"  {band}: the drift covariance reproduces every standard "
              f"error to {SE_DP} decimals")

        c, se = got[band][TREND]
        said_flat = str(b.loc[TREND, "flat_within_2se"]).strip().lower()
        if said_flat not in ("true", "false"):
            raise SystemExit(f"  {band}: flat_within_2se reads "
                             f"'{b.loc[TREND, 'flat_within_2se']}', which is "
                             f"neither true nor false")
        is_flat = abs(c) <= 2.0 * se
        if is_flat != (said_flat == "true"):
            raise SystemExit(f"  {band}: the export declares "
                             f"flat_within_2se {said_flat}, but {c:+.6f} "
                             f"against twice {se:.6f} says "
                             f"{str(is_flat).lower()}; not written")
        flat[band] = is_flat

    rows = [r"\addlinespace",
            r"\multicolumn{3}{@{}l}{\textit{The pre-launch drift test, "
            r"January 2021 to November 2022}} \\",
            r"\addlinespace[2pt]"]
    line = " & ".join(cell_dp(f"{b} trend", *got[b][TREND], 5) for b in BANDS)
    rows.append(f"Linear trend per month & {line} \\\\")
    print(f"  {'trend per month':34s} {line}")
    line = " & ".join(
        cell_dp(f"{b} over {DRIFT_MONTHS} months",
                got[b][TREND][0] * DRIFT_MONTHS,
                got[b][TREND][1] * DRIFT_MONTHS, 4) for b in BANDS)
    rows.append(f"Over the twenty-three months & {line} \\\\")
    print(f"  {'over the twenty-three months':34s} {line}")
    line = " & ".join(cell_dp(f"{b} window", *got[b][TIGHT], 4)
                      for b in BANDS)
    rows.append(f"The tightening window & {line} \\\\")
    print(f"  {'the tightening window':34s} {line}")
    line = " & ".join("Yes" if flat[b] else "No" for b in BANDS)
    rows.append(f"Flat within two standard errors & {line} \\\\")
    print(f"  {'flat within two standard errors':34s} {line}")
    line = " & ".join(thousands(firms[b]) for b in BANDS)
    rows.append(f"Employers & {line} \\\\")
    print(f"  {'employers':34s} {line}")
    return rows, firms


def summary_part_a(path: Path) -> dict[tuple[str, str], str]:
    """Part A(i) of the run's summary as the run itself reported it: the
    printed coefficient of every quarter, with its star. This is a second
    record of the same fits, and the table is written only if the two
    agree."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # 78 prints the path first and ends it at A(ii); 86 prints a gate and a
    # drift block before it, so there the collection starts at its header.
    marker = "THE PLAIN PATH:"
    started = marker not in text
    said, band = {}, None
    for line in text.splitlines():
        if not started:
            started = line.strip() == marker
            continue
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
    d = pd.read_csv(first_of(LANE32, PATH_NAMES))
    said = summary_part_a(first_of(LANE32, SUMMARY_NAMES))
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

        v = pd.read_csv(source(LANE32,
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

    drift_rows, drift_firms = drift_panel()

    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           "Quarter & " + " & ".join(b.replace("-", "--") for b in BANDS)
           + r" \\",
           r"\midrule"]
    tex += rows
    tex += drift_rows
    tex += [r"\bottomrule", r"\end{tabular}", "",
            r"\vspace{0.5em}",
            r"\begin{minipage}{0.86\textwidth}",
            r"\footnotesize \textit{Notes:} The upper panel is Poisson on employer-by-age-by-month counts, with",
            r"employer-by-month, employer-by-age and month-by-age effects and no calendar",
            f"terms; {REFERENCE} is the reference. Entries are the coefficient with the standard",
            r"error clustered by employer in parentheses; $^{*}$ marks significance at the five",
            f"per cent level. The panel holds {thousands(panel[BANDS[0]])} cells at "
            f"{BANDS[0].replace('-', '--')} and {thousands(panel[BANDS[1]])}",
            f"at {BANDS[1].replace('-', '--')}. "
            r"Figure~\ref{fig:prepath} draws the same coefficients by calendar",
            r"quarter.",
            r"The lower panel is a separate fit: a linear trend on the pre-launch months,",
            r"January 2021 to November 2022, estimated with the calendar terms and the",
            r"tightening window in, on the occupation route. The trend is per month and the",
            r"row beneath it scales the trend and its standard error by the twenty-three",
            r"months of that window. The pre-period counts as flat if the trend lies within",
            f"two standard errors of zero. That fit holds "
            f"{thousands(drift_firms[BANDS[0]])} employers at "
            f"{BANDS[0].replace('-', '--')} and",
            f"{thousands(drift_firms[BANDS[1]])} at "
            f"{BANDS[1].replace('-', '--')}. Source: script 83.",
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

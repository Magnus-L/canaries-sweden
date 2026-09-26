#!/usr/bin/env python3
"""
07_tab_prepath.py: Online Appendix Table A19 (Section III.2), the quarterly
path on the plain specification and, beneath it, the pre-launch drift test.

Upper panel: Poisson on employer by age by month counts with
employer-by-month, employer-by-age and month-by-age effects and no calendar
terms, one coefficient per calendar quarter interacted with exposure and the
young band, 2022Q1 omitted, standard errors clustered by employer (script 86,
which runs script 78's part A on the occupation-mix score). The two young
bands are separate fits on their own panels. Lower panel: a linear trend on
January 2021 to November 2022, fitted with the calendar terms and the
tightening window in (script 83); the last row reports whether a zero trend
is rejected at two standard errors, which does not establish a flat
pre-period. The second row scales the monthly trend and its standard error
by the twenty-three months of the window.

The script writes nothing unless each band has one row per quarter and one
panel size, the omitted quarter is exported as the reference, every standard
error equals the square root of its own covariance diagonal, every cell
matches the run's own summary, and the exported flat-within-two-standard-errors
verdict agrees with the one recomputed from the coefficient and its error.
The file is a bare tabular; the appendix supplies the float and caption.

Exports read (3_register_mona/exports/):
  2026-09-23_1234_s86/  occ_route_prepath.csv, vcov_s78_prepath_<band>.csv,
      86_summary.txt
  2026-09-23_0655_s82-partB_s83-partsBCD/  occ_rest_drift.csv,
      vcov_s83_drift_<band>.csv (second argument overrides this folder)
Output: output/tables/tableA_prepath.tex

    python 4_exhibits/07_tab_prepath.py [export_dir [drift_dir]]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

# Script 86 ran script 78's part A on the occupation-mix score the paper
# reports. Script 78's own export is on the education-based score and is
# read if its folder is given; the two name the same files differently,
# so every name is tried in turn.
S86 = EXPORTS / "2026-09-23_1234_s86"
S78 = EXPORTS / "2026-09-22_1333_s78"
# The drift panel is a different run from the quarterly path and has its
# own export, so it is pinned on its own and overridden by a second
# argument rather than by the first.
S83 = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"
PATH_NAMES = ("occ_route_prepath.csv", "prepath_plain.csv")
SUMMARY_NAMES = ("86_summary.txt", "78_summary.txt")

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
    """The pinned script 83 export, or the same file name under a
    directory given as the second argument."""
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else S83
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
    rows.append(f"Zero linear drift not rejected & {line} \\\\")
    print(f"  {'zero linear drift not rejected':34s} {line}")
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
    d = pd.read_csv(first_of(S86, PATH_NAMES))
    said = summary_part_a(first_of(S86, SUMMARY_NAMES))
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

        v = pd.read_csv(source(S86,
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
            r"\footnotesize \textit{Notes:} Upper panel: Poisson on employer-by-age-by-month",
            r"counts with employer-by-month, employer-by-age and month-by-age effects and no",
            r"calendar terms, 2022Q1 the reference; standard errors clustered by employer;",
            r"$^{*}$ $p<0.05$. Lower panel: a separate fit of a linear monthly trend on",
            r"January 2021 to November 2022, with the calendar terms and the tightening",
            r"window in; the second row scales it to the twenty-three months. The last row reports whether a zero trend is rejected at two standard errors, which does not establish a flat pre-period.",
            r"\end{minipage}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_prepath.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

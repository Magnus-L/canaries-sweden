#!/usr/bin/env python3
"""
14_tab_window.py: Online Appendix Table A11 (Section III.2), the level after
adoption read against the months before the rate rise.

Equation (2) keeps the Riksbank interaction switched on through the post
period, so its adoption coefficient is a step from the tightening months.
Confining that interaction to a window, April to November 2022, makes the
later terms read against January 2021 to March 2022 (script 83). Poisson with
employer-by-month, employer-by-age and month-by-age effects and the calendar
terms, exposure the 2019 occupation mix, clustered by employer; each young band
is its own fit. The fourth row, the step from the 2023 level, is the post term
minus the interim term with the standard error from their covariance.

Nothing is written unless every row is status ok, every standard error equals
the square root of its own covariance diagonal to six decimals, and every
printed estimate equals, to four decimals, the value the appendix paragraph
states; those values are typed below as ANCHOR and serve only as a check.

Exports read: 3_register_mona/exports/2026-09-23_0655_s82-partB_s83-partsBCD/
  occ_rest_window.csv, vcov_s83_window_22_25.csv, vcov_s83_window_26_30.csv
Output: output/tables/tableA_window.tex

    python 4_exhibits/14_tab_window.py [export_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S83 = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"

BANDS = ["22-25", "26-30"]
TIGHT = "rbw_x_high_x_young"
INTERIM = "interim_x_high_x_young"
POST = "post_x_high_x_young"
FITTED = [
    (TIGHT, "Tightening months, April to November 2022"),
    (INTERIM, "The thirteen months after the launch"),
    (POST, "The level after adoption, from January 2024"),
]
DERIVED_LABEL = "The step from the 2023 level"
SE_DP = 6                       # the covariance check, decimals
CELL = re.compile(r"^\$([-+][0-9.]+)(\^\{\*\})?\$ \(([0-9.]+)\)$")

# The figures the appendix paragraph already states, to four decimals.
# They are the anchor, not the source: every number printed below is read
# from the export and then compared with these.
ANCHOR = {
    ("22-25", TIGHT): (+0.0159, 0.0075),
    ("22-25", INTERIM): (-0.0020, 0.0135),
    ("22-25", POST): (-0.0419, 0.0189),
    ("22-25", "step"): (-0.0399, 0.0102),
    ("26-30", TIGHT): (+0.0184, 0.0044),
    ("26-30", INTERIM): (+0.0104, 0.0080),
    ("26-30", POST): (-0.0298, 0.0123),
    ("26-30", "step"): (-0.0403, 0.0067),
}


def source(name: str, default_dir: Path = S83) -> Path:
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


def cell(what: str, c: float, se: float) -> str:
    """The estimate with a star at five per cent, read back from the
    string that goes into the table before it is allowed in."""
    star = "^{*}" if abs(c) > 1.96 * se else ""
    out = f"${c:+.4f}{star}$ ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted estimate is unreadable")
    if abs(float(m.group(1)) - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed estimate {out} disagrees "
                         f"with the export ({c:+.6f}, {se:.6f})")
    return out


def anchored(band: str, key: str, c: float, se: float) -> None:
    """The third check: what the appendix paragraph already states."""
    want_c, want_se = ANCHOR[(band, key)]
    if abs(round(c, 4) - want_c) > 1e-9 or abs(round(se, 4) - want_se) > 1e-9:
        raise SystemExit(
            f"  {band} {key}: the export gives {c:+.6f} ({se:.6f}), the "
            f"appendix paragraph states {want_c:+.4f} ({want_se:.4f}); the "
            f"table is not written")


def one_count(d: pd.DataFrame, band: str) -> int:
    n = sorted(set(int(x) for x in d.n_firms))
    if len(n) != 1:
        raise SystemExit(f"  {band}: one employer count expected, found {n}")
    return n[0]


def main() -> int:
    p = source("occ_rest_window.csv")
    print(f"  reading {p}")
    d = pd.read_csv(p)

    cells: dict[tuple[str, str], str] = {}
    firms: dict[str, int] = {}
    for band in BANDS:
        b = d[d.young_band == band]
        if b.empty:
            raise SystemExit(f"  occ_rest_window.csv: no rows for {band}")
        bad = [t for t, _ in FITTED if (b.term == t).sum() != 1]
        if bad:
            raise SystemExit(f"  occ_rest_window.csv: {band} must carry one "
                             f"row per term; wrong for {bad}")
        notok = sorted(set(b.loc[b.status != "ok", "status"].astype(str)))
        if notok:
            raise SystemExit(f"  {band}: the export reports status {notok}, "
                             f"so the fit is not quotable")
        firms[band] = one_count(b, band)
        b = b.set_index("term")

        # The second record: the clustered covariance of the same fit.
        vp = source(f"vcov_s83_window_{band.replace('-', '_')}.csv")
        print(f"  reading {vp}")
        v = pd.read_csv(vp, index_col=0)
        for t, _ in FITTED:
            if t not in v.index or t not in v.columns:
                raise SystemExit(f"  {band}: {t} is not in the covariance "
                                 f"export, so the estimate has one record")
            said = float(np.sqrt(float(v.loc[t, t])))
            got = float(b.loc[t, "se"])
            if round(said, SE_DP) != round(got, SE_DP):
                raise SystemExit(f"  {band} {t}: the export reports a "
                                 f"standard error of {got:.6f} and its own "
                                 f"covariance {said:.6f}; not written")
        print(f"  {band}: the covariance reproduces every standard error "
              f"to {SE_DP} decimals")

        for t, _ in FITTED:
            c, se = float(b.loc[t, "coef"]), float(b.loc[t, "se"])
            anchored(band, t, c, se)
            cells[(band, t)] = cell(f"{band} {t}", c, se)

        # The derived row: post minus interim, Var(a-b)=Vaa+Vbb-2Vab.
        c = float(b.loc[POST, "coef"]) - float(b.loc[INTERIM, "coef"])
        var = (float(v.loc[POST, POST]) + float(v.loc[INTERIM, INTERIM])
               - 2.0 * float(v.loc[POST, INTERIM]))
        if var <= 0:
            raise SystemExit(f"  {band}: the derived step has a variance of "
                             f"{var:.3e}, which is not a variance")
        se = float(np.sqrt(var))
        anchored(band, "step", c, se)
        cells[(band, "step")] = cell(f"{band} step", c, se)

    rows = []
    for t, label in FITTED + [("step", DERIVED_LABEL)]:
        line = " & ".join(cells[(band, t)] for band in BANDS)
        rows.append(f"{label} & {line} \\\\")
        print(f"  {label:44s} {line}")
    counts = " & ".join(thousands(firms[b]) for b in BANDS)
    rows.append(f"Employers & {counts} \\\\")
    print(f"  {'Employers':44s} {counts}")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The level after adoption, read against the months "
           r"before the rate rise.}",
           r"\label{tab:window}", r"\footnotesize",
           r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           "Term & " + " & ".join(b.replace("-", "--") for b in BANDS)
           + r" \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.88\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson with employer-by-month, employer-by-age and month-by-age effects and the calendar-quarter terms; exposure is the employer's 2019 occupation mix; standard errors clustered by employer. The Riksbank interaction is a window, April to November 2022, so the second and third rows are levels against January 2021 to March 2022 rather than steps; the fitted means are those of Equation~(2). The fourth row is the third minus the second, the step Table~1 of the paper reports. Each band is fitted on its own panel. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_window.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

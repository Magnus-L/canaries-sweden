#!/usr/bin/env python3
"""
17_tab_contrast_by_track.py: Online Appendix Table A16 (Section III.2), the
young against the prime-aged inside exposed employers after adoption, by
education track, calendar cycle removed.

Script 74's specification on a three-band panel (22-25, 26-30 and the
reference 41-49), refitted by script 88 on the 2019 occupation-mix score:
employer-by-month, employer-by-age and month-by-age effects, one adoption, one
Riksbank and three quarter-of-year terms per young band, clustered by employer;
on all workers and within each broad education track. Negative means the band
declined more than 41-49. The note compares the all-worker row with the same
contrast on the six-band panel (script 82) and quotes the ICT share of young
women and men in exposed employers (script 87).

Nothing is written unless every fit is status ok, every standard error equals
the square root of its own covariance diagonal, and every row, its star and
its employer count match the run's own summary.

Exports read (3_register_mona/exports/):
  2026-09-23_1407_s88/  occ_route_contrast_by_track.csv, vcov_s88_<track>.csv,
      88_summary.txt
  2026-09-23_0655_s82-partB_s83-partsBCD/occ_route_profile.csv
  2026-09-23_1352_s87/occ_route_education_mix_by_sex.csv
Output: output/tables/tableA_contrast_by_track.tex

    python 4_exhibits/17_tab_contrast_by_track.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

# Script 88 refits script 77's three-band panel on the occupation-mix
# score. The six-band figure the note compares with is script 82's
# profile on the same score, and the ICT shares are script 87's.
S88 = EXPORTS / "2026-09-23_1407_s88"
S87 = EXPORTS / "2026-09-23_1352_s87"
S82B = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"

# Export key, row label.
TRACKS = [
    ("all", "All workers"),
    ("ict", "ICT"),
    ("engineering", "Engineering, manufacturing, construction"),
    ("business_law_social", "Business, law, administration, social science"),
    ("health_education_care", "Health, education, social care"),
    ("other", "Other fields"),
]
BANDS = ["22-25", "26-30"]
REFERENCE = "41--49"
# The band whose six-band counterpart the note compares the panel with.
SIX_BAND = "22-25"
CELL = re.compile(r"^\$([-+][0-9.]+)\$(\$\^\{\*\}\$)? \(([0-9.]+)\)$")
# 88 lays its summary out differently from 77: the base rows name
# "all workers," and carry "vs 41-49", the track rows carry neither, and
# both print a t ratio where 77 printed a trailing star. One pattern
# reads both shapes; the star is derived from the t and checked.
SUMMARY_ROW = re.compile(
    r"^\s+(?:(all) workers,|(\S+))\s+(\d\d-\d\d)(?: vs 41-49)?\s+"
    r"([-+][0-9.]+) \(([0-9.]+)\) t ([-+][0-9.]+)\s+firms ([0-9,]+)\s*$")


def source(defaults: tuple[Path, ...], name: str) -> Path:
    """The pinned exports, or the same file name under a directory given
    on the command line and then its parent."""
    dirs = ((Path(sys.argv[1]), Path(sys.argv[1]).parent)
            if len(sys.argv) > 1 else defaults)
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    raise SystemExit(f"  missing input: {name} in {[str(d) for d in dirs]}")


def thousands(n: int) -> str:
    """An employer count as the paper prints it."""
    return f"{n:,}".replace(",", "{,}")


def cell(what: str, c: float, se: float) -> str:
    """The estimate with a star at five per cent, checked against the
    export before it is allowed into the table."""
    star = "$^{*}$" if abs(c) > 1.96 * se else ""
    out = f"${c:+.4f}${star} ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted estimate is unreadable")
    if abs(float(m.group(1)) - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed estimate {out} disagrees "
                         f"with occ_route_contrast_by_track.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_rows(path: Path) -> dict[tuple[str, str], tuple[str, float, int]]:
    """The run's own report of the same fits: the printed estimate of
    each track and band without its star, the t ratio the run printed,
    and the employer count. The star is not taken from the summary but
    derived from the estimate and checked against this t."""
    said = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ROW.match(line)
        if m is None:
            continue
        base, track, band, coef, se, t_ratio, firms = m.groups()
        said[(base or track, band)] = (f"${coef}$ ({se})", float(t_ratio),
                                       int(firms.replace(",", "")))
    if not said:
        raise SystemExit(f"  {path.name}: reports no track against 41-49, so "
                         f"there is nothing to check the export against")
    return said


def main() -> int:
    d = pd.read_csv(source((S88,), "occ_route_contrast_by_track.csv"))
    if not (d.status == "ok").all():
        raise SystemExit("  occ_route_contrast_by_track.csv: a fit reports a "
                         "status "
                         "other than ok, so it is not quotable")
    for key, _ in TRACKS:
        for band in BANDS:
            if ((d.track == key) & (d.band_vs_ref == band)).sum() != 1:
                raise SystemExit(f"  occ_route_contrast_by_track.csv: one row expected "
                                 f"for {key} at {band}")
    if len(d) != len(TRACKS) * len(BANDS):
        raise SystemExit(f"  occ_route_contrast_by_track.csv: {len(d)} rows, "
                         f"expected "
                         f"{len(TRACKS) * len(BANDS)}")

    # Each exported standard error must be the square root of its own
    # variance in the covariance matrix the same fit exported.
    for key, _ in TRACKS:
        v = pd.read_csv(source((S88,), f"vcov_s88_{key}.csv"),
                        index_col=0)
        for band in BANDS:
            term = f"gpt_x_high_{band.replace('-', '_')}"
            got = float(d[(d.track == key) & (d.band_vs_ref == band)].se.iloc[0])
            diag = float(v.loc[term, term]) ** 0.5
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {key} {band}: the exported standard error "
                                 f"{got:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")

    said = summary_rows(source((S88,), "88_summary.txt"))
    d = d.set_index(["track", "band_vs_ref"])

    rows, counts = [], {}
    for key, label in TRACKS:
        cells = []
        for band in BANDS:
            r = d.loc[(key, band)]
            text = cell(f"{key} {band}", float(r.coef), float(r.se))
            if (key, band) not in said:
                raise SystemExit(f"  {key} {band}: the run's summary does not "
                                 f"report it")
            quoted, t_said, firms = said[(key, band)]
            if text.replace("$^{*}$", "") != quoted:
                raise SystemExit(f"  {key} {band}: the table would print "
                                 f"{text}, the run's summary reports {quoted}")
            t_own = float(r.coef) / float(r.se)
            if abs(t_own - t_said) > 5e-3:
                raise SystemExit(f"  {key} {band}: the summary's t {t_said} is "
                                 f"not the estimate over its standard error "
                                 f"({t_own:+.4f})")
            # The star the table prints must agree with that same t.
            if ("$^{*}$" in text) != (abs(t_said) > 1.96):
                raise SystemExit(f"  {key} {band}: the table prints "
                                 f"{'a star' if '$^{*}$' in text else 'no star'}"
                                 f" and the run's summary reports t {t_said}")
            if firms != int(r.n_firms):
                raise SystemExit(f"  {key} {band}: the summary reports "
                                 f"{firms:,} employers and the CSV "
                                 f"{int(r.n_firms):,}")
            cells.append(text)
        n = sorted(set(int(d.loc[(key, b), "n_firms"]) for b in BANDS))
        if len(n) != 1:
            raise SystemExit(f"  {key}: one employer count expected across "
                             f"the bands, found {n}")
        counts[key] = n[0]
        rows.append(f"{label} & " + " & ".join(cells) + f" & {n[0]:,} \\\\")
        print(f"  {label[:44]:44s} " + " ".join(f"{c:>24s}" for c in cells)
              + f" {n[0]:>9,}")

    # The two figures the note compares the panel with: the same contrast
    # on the six-band panel, and the ICT shares of the mix table.
    six = pd.read_csv(source((S82B,), "occ_route_profile.csv"))
    six = six[(six.band == SIX_BAND) & (six.status == "ok")]
    if len(six) != 1:
        raise SystemExit(f"  occ_route_profile.csv: one row expected for "
                         f"{SIX_BAND}, found {len(six)}")
    six_cell = cell("six-band 22-25", float(six.coef.iloc[0]),
                    float(six.se.iloc[0])).replace("$^{*}$", "")
    mix = pd.read_csv(source((S87,),
                             "occ_route_education_mix_by_sex.csv"))
    mix = mix[(mix.dimension == "track") & (mix.exposed == 1)
              & (mix.cell == "ict")].set_index("gender")
    ict = {g: 100 * float(mix.loc[g, "share"]) for g in ("women", "men")}
    all_cell = cell("all 22-25", float(d.loc[("all", "22-25"), "coef"]),
                    float(d.loc[("all", "22-25"), "se"])).replace("$^{*}$", "")
    print(f"  panel {counts['all']:,} employers; all-worker {all_cell} against "
          f"{six_cell} on six bands; ICT {ict['women']:.1f} / {ict['men']:.1f} "
          f"per cent")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The young against 41--49 inside exposed firms after "
           r"adoption, by education track, calendar cycle removed.}",
           r"\label{tab:contrast_by_track}", r"\footnotesize",
           # The track labels are long; a narrower column gap keeps the
           # table inside the text block.
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lccr}", r"\toprule",
           f"Track & {BANDS[0].replace('-', '--')} vs {REFERENCE} & "
           f"{BANDS[1].replace('-', '--')} vs {REFERENCE} & Employers " + r"\\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"A three-band panel (22--25, 26--30 and the reference 41--49) with employer-by-month, employer-by-age and month-by-age effects, one adoption, one Riksbank and three quarter-of-year terms per young band, exposure the employer's 2019 occupation mix, clustered by employer. Negative means the band declined relative to 41--49. Tracks as in Table~\ref{tab:gender_split}; ICT holds 2.9 per cent of exposed employers' young women and 6.8 per cent of their young men. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_contrast_by_track.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l31_tab_contrast_by_track.py: Online Appendix Table III.2
(tab:contrast_by_track), the young against the prime-aged inside exposed
employers, cut by education track.

THE QUESTION
With the calendar cycle removed the pooled contrast of the young against
41-49 is a null, and a null can hide offsetting movements. The table asks
whether the young decline against their older colleagues in some fields
of education and gain in others, which is the heterogeneity the pooled
profile averages over. It is reported in full, nulls included, and no
track is promoted above the pooled profile.

WHAT IS ESTIMATED
Script 74's seasonal arm on a three-band panel, ages 22-25, 26-30 and the
reference 41-49 (script 77, lanes 22 and 23): Poisson pseudo-maximum
likelihood on employer by age by month counts, with employer-by-month,
employer-by-age and month-by-age effects, one adoption term, one Riksbank
term and three quarter-of-year terms per young band, exposure frozen at
the employer's 2019 education mix, and standard errors clustered by
employer. A coefficient is the post-adoption change of its band relative
to 41-49 inside exposed employers; negative means the band declined more
than 41-49 did.

The three-band panel is smaller than the six-band panel of
Table tab:profile_seasonal, so the all-worker row here and the 22-25 row
there are the same contrast on different samples; the note prints both.
Tracks are broad two-digit SUN 2020 fields and the education record for
2024 and 2025 is the 2023 one carried forward, which is why they are
broad.

The ICT fit failed at two threads in the lane 22 run and was re-estimated
alone in lane 23 once the retry ladder gained a one-thread attempt; lane
23 is the export the paper quotes, and it reproduces every other track of
lane 22 exactly. Lane 23 re-exported only the fits it re-estimated, so
the covariance of the all-worker fit is read from the lane 22 directory.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names or one
given on the command line: contrast_by_track.csv (script 77, lane 23;
columns track, band_vs_ref, coef, se, n_firms, status),
vcov_s77_<track>.csv (the exported covariance of each track fit) and
77_summary.txt, the run's own report of the same fits; and, for the two
figures the note compares with, contrast_seasonal.csv (script 74, lane
20) for the six-band contrast and education_mix_by_sex.csv (script 76,
lane 22) for the ICT shares. Nothing is typed in. Writes
revision/tables/tableA_contrast_by_track.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l31_tab_contrast_by_track.py [export_dir]

THE GATE
Every track must be exported once per band with status ok, and the panel
must report one employer count per track. Each exported standard error
must equal the square root of its own diagonal in that track's covariance
matrix, which is the integrity check this export offers. Every row is
then checked against the run's summary, which reports the same estimates,
their stars and their employer counts independently of the CSV; a star
the summary does not carry, or an employer count it does not confirm,
stops the script. Finally each printed number is read back from the
string that goes into the table and compared with the export it came
from, and any disagreement beyond half of the last printed digit stops
the script and nothing is written.

IN THE PAPER
Online Appendix III.2, the paragraph on the young against 41-49 by track,
Table tab:contrast_by_track; the ICT and business contrasts are quoted in
Section 3.
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
LANE22 = OUT / "round3_20260922-0712-lanes21-22"
LANE23 = LANE22 / "lane23-0807"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

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
SIX_BAND_KEY = "22_25"
CELL = re.compile(r"^\$([-+][0-9.]+)\$(\$\^\{\*\}\$)? \(([0-9.]+)\)$")
SUMMARY_ROW = re.compile(
    r"^\s+(\S+)\s+(\S+) vs 41-49\s+([-+][0-9.]+) \(([0-9.]+)\) "
    r"firms ([0-9,]+)(\s+\*)?\s*$")


def source(defaults: tuple[Path, ...], name: str) -> Path:
    """The pinned exports, or the same file name under a directory given
    on the command line and then its parent, since the lane 23 re-run
    exported only the fits it re-estimated."""
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
                         f"with contrast_by_track.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_rows(path: Path) -> dict[tuple[str, str], tuple[str, int]]:
    """The run's own report of the same fits: the printed estimate of
    each track and band, with its star, and the employer count."""
    said = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ROW.match(line)
        if m is None:
            continue
        track, band, coef, se, firms, star = m.groups()
        said[(track, band)] = (f"${coef}${'$^{*}$' if star else ''} ({se})",
                               int(firms.replace(",", "")))
    if not said:
        raise SystemExit(f"  {path.name}: reports no track against 41-49, so "
                         f"there is nothing to check the export against")
    return said


def main() -> int:
    d = pd.read_csv(source((LANE23, LANE22), "contrast_by_track.csv"))
    if not (d.status == "ok").all():
        raise SystemExit("  contrast_by_track.csv: a fit reports a status "
                         "other than ok, so it is not quotable")
    for key, _ in TRACKS:
        for band in BANDS:
            if ((d.track == key) & (d.band_vs_ref == band)).sum() != 1:
                raise SystemExit(f"  contrast_by_track.csv: one row expected "
                                 f"for {key} at {band}")
    if len(d) != len(TRACKS) * len(BANDS):
        raise SystemExit(f"  contrast_by_track.csv: {len(d)} rows, expected "
                         f"{len(TRACKS) * len(BANDS)}")

    # Each exported standard error must be the square root of its own
    # variance in the covariance matrix the same fit exported.
    for key, _ in TRACKS:
        v = pd.read_csv(source((LANE23, LANE22), f"vcov_s77_{key}.csv"),
                        index_col=0)
        for band in BANDS:
            term = f"gpt_x_high_{band.replace('-', '_')}"
            got = float(d[(d.track == key) & (d.band_vs_ref == band)].se.iloc[0])
            diag = float(v.loc[term, term]) ** 0.5
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {key} {band}: the exported standard error "
                                 f"{got:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")

    said = summary_rows(source((LANE23, LANE22), "77_summary.txt"))
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
            quoted, firms = said[(key, band)]
            if text != quoted:
                raise SystemExit(f"  {key} {band}: the table would print "
                                 f"{text}, the run's summary reports {quoted}")
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
    six = pd.read_csv(source((LANE20,), "contrast_seasonal.csv"))
    six = six[(six.arm == "seasonal") & (six.band_vs_ref == SIX_BAND_KEY)]
    if len(six) != 1:
        raise SystemExit(f"  contrast_seasonal.csv: one seasonal row expected "
                         f"for {SIX_BAND_KEY}, found {len(six)}")
    six_cell = cell("six-band 22-25", float(six.coef.iloc[0]),
                    float(six.se.iloc[0])).replace("$^{*}$", "")
    mix = pd.read_csv(source((LANE22,), "education_mix_by_sex.csv"))
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
           r"\begin{tabular}{lccr}", r"\toprule",
           f"Track & {BANDS[0].replace('-', '--')} vs {REFERENCE} & "
           f"{BANDS[1].replace('-', '--')} vs {REFERENCE} & Employers " + r"\\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Script 74's specification on a three-band panel (22--25, 26--30 "
            r"and the reference 41--49): employer-by-month, employer-by-age "
            r"and month-by-age effects; one adoption, one Riksbank and three "
            r"quarter-of-year terms per young band; exposure frozen at the "
            r"2019 education mix; clustered by employer. Negative means the "
            r"band declined more than 41--49. On this panel of "
            f"{thousands(counts['all'])} employers the all-worker contrast is "
            f"{all_cell}, a null. Tracks as in "
            r"Table~\ref{tab:gender_split}; the ICT track is small "
            f"({ict['women']:.1f} per cent of exposed firms' young women, "
            f"{ict['men']:.1f} per cent of their young men). "
            r"$^{*}$ $p<0.05$. Source: script 77.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_contrast_by_track.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

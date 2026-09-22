#!/usr/bin/env python3
"""
l28_tab_profile_seasonal.py: Online Appendix Table III.2
(tab:profile_seasonal), the age profile inside exposed employers after
adoption, with and without the calendar cycle.

THE QUESTION
The paper's claim is about the young relative to their older colleagues
inside the same employer. Read on the plain specification the youngest
band declines against 41-49, but employment of the young is strongly
seasonal and the adoption window opens in January, so part of that
contrast is the calendar. The table puts the two arms beside each other
and asks which features of the profile survive the removal of the cycle.

WHAT IS ESTIMATED
Poisson pseudo-maximum likelihood on employer by age by month counts,
with employer-by-month, employer-by-age and month-by-age effects,
exposure frozen at the employer's 2019 education mix, treatment dated
January 2024, and standard errors clustered by employer (script 74, lane
20). All six bands sit in one panel. Exposure is constant within an
employer-month, so the band levels are collinear with the
employer-by-month effects and one band must carry no estimate: 41-49 is
the reference, and every coefficient is the post-adoption change of its
band relative to 41-49 inside exposed employers. A negative number means
the band declined more than 41-49 did. The second arm adds three
quarter-of-year terms per band, which is the headline specification's
control; Q4 is omitted for the same collinearity reason.

The seasonal arm is the one the paper reads. With the cycle removed
neither young band differs from 41-49, and what remains is the gain of
the 50-and-over band; the read rule was fixed before the run.

INPUTS AND OUTPUTS
Reads contrast_seasonal.csv (script 74, lane 20; columns arm,
band_vs_ref, coef, se, n_firms) and 74_summary.txt, the run's own report
of the same fits, from the export directory the final-code manifest names
or one given on the command line. Nothing is typed in. Writes
revision/tables/tableA_profile_seasonal.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l28_tab_profile_seasonal.py [export_dir]

THE GATE
Every printed estimate is read back from the string that goes into the
table and compared with the export it came from; a disagreement beyond
half of the last printed digit stops the script and nothing is written.
Each arm must carry exactly one row per band, the reference band must
appear in neither arm, and the panel must report a single employer count
across both arms. The run's summary reports three of the ten cells with
their t ratios independently of the CSV: each such cell must print
exactly as the summary reports it, and the summary's t must equal the
CSV's coefficient over its standard error to the precision the summary
prints. That is the only integrity check this export offers, there being
no exported covariance for these fits.

IN THE PAPER
Online Appendix III.2, Table tab:profile_seasonal; Figure 2 draws the
same two arms, and the band rows of Table 1 come from the seasonal arm.
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
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

# Export key, row label. 41-49 is the reference and carries no estimate.
BANDS = [("22_25", "22--25"), ("26_30", "26--30"), ("31_34", "31--34"),
         ("35_40", "35--40"), ("50p", "50 and over")]
REFERENCE = "41--49"
ARMS = ["plain", "seasonal"]
CELL = re.compile(r"^\$([-+][0-9.]+)\$(\$\^\{\*\}\$)? \(([0-9.]+)\)$")
# One line of the run's summary: arm, band, estimate, standard error, t.
SUMMARY_ROW = re.compile(
    r"^(plain|seasonal)\s+(\S+) vs 41-49\s+([-+][0-9.]+)\s+"
    r"\(([0-9.]+)\)\s+t\s+([-+][0-9.]+)\s*$")


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
                         f"with contrast_seasonal.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_rows(path: Path) -> dict[tuple[str, str], tuple[str, float]]:
    """The cells the run's summary reports itself, as the printed string
    and the t ratio beside it. This is a second record of the same fits,
    written by the run rather than derived from the CSV."""
    said = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ROW.match(line)
        if m is None:
            continue
        arm, band, coef, se, t = m.groups()
        said[(arm, band)] = (f"${coef}$ ({se})", float(t))
    if not said:
        raise SystemExit(f"  {path.name}: reports no band against 41-49, so "
                         f"there is nothing to check the export against")
    return said


def one_count(d: pd.DataFrame) -> int:
    """The employer count of the panel, which the export must report once
    across both arms."""
    n = sorted(set(int(x) for x in d.n_firms))
    if len(n) != 1:
        raise SystemExit(f"  contrast_seasonal.csv: one employer count "
                         f"expected across both arms, found {n}")
    return n[0]


def main() -> int:
    d = pd.read_csv(source(LANE20, "contrast_seasonal.csv"))
    keys = [k for k, _ in BANDS]
    if set(d.arm) != set(ARMS):
        raise SystemExit(f"  contrast_seasonal.csv: expected the arms {ARMS}, "
                         f"found {sorted(set(d.arm))}")
    if "41_49" in set(d.band_vs_ref):
        raise SystemExit("  contrast_seasonal.csv: the reference band carries "
                         "an estimate, so it is not the reference")
    for arm in ARMS:
        a = d[d.arm == arm]
        bad = [k for k in keys if (a.band_vs_ref == k).sum() != 1]
        if bad or len(a) != len(keys):
            raise SystemExit(f"  contrast_seasonal.csv: the {arm} arm must "
                             f"carry one row per band; wrong for {bad or 'the row count'}")
    n_firms = one_count(d)
    said = summary_rows(source(LANE20, "74_summary.txt"))
    d = d.set_index(["arm", "band_vs_ref"])

    rows = []
    for key, label in BANDS:
        cells = []
        for arm in ARMS:
            r = d.loc[(arm, key)]
            c, se = float(r.coef), float(r.se)
            text = cell(f"{arm} {label}", c, se)
            # Where the run reported the same cell itself, the table must
            # print what the run reported, and the run's t must be this
            # coefficient over this standard error.
            if (arm, label.replace("--", "-")) in said:
                quoted, t = said[(arm, label.replace("--", "-"))]
                if text.replace("$^{*}$", "") != quoted:
                    raise SystemExit(f"  {arm} {label}: the table would print "
                                     f"{text}, the run's summary reports "
                                     f"{quoted}")
                if abs(c / se - t) > 5e-3:
                    raise SystemExit(f"  {arm} {label}: the summary's t {t} is "
                                     f"not {c:+.6f} over {se:.6f} "
                                     f"({c / se:+.4f})")
            cells.append(text)
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        print(f"  {label:12s} " + "  ".join(f"{x:>24s}" for x in cells))
    print(f"  panel {n_firms:,} employers, reference {REFERENCE}")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The age profile inside exposed firms after adoption: "
           r"every band against 41--49, with and without the calendar cycle.}",
           r"\label{tab:profile_seasonal}", r"\footnotesize",
           r"\begin{tabular}{lcc}", r"\toprule",
           r"Band, against 41--49 & Plain & Calendar cycle removed \\",
           r"\midrule"]
    tex += rows
    tex += [r"\addlinespace[2pt]",
            f"{REFERENCE} & reference & reference " + r"\\",
            r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.88\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson pseudo-maximum likelihood on employer $\times$ age "
            r"$\times$ month counts, all six bands in one panel of "
            f"{thousands(n_firms)} employers; employer-by-month, "
            r"employer-by-age and month-by-age effects; exposure frozen at the "
            r"2019 education mix; treatment January 2024; clustered by "
            r"employer. A coefficient is the post-adoption change of the band "
            r"relative to 41--49 inside exposed firms; negative means it "
            r"declined more. The second column adds three quarter-of-year "
            r"terms per band, the headline's control. $^{*}$ $p<0.05$. "
            r"Source: script 74 (\texttt{contrast\_seasonal.csv}).",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_profile_seasonal.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l33_tab_skill_placebo.py: Online Appendix Table III.2
(tab:skill_placebo), the adoption step under an education-intensity
placebo.

THE QUESTION
The paper's exposure is built from an employer's 2019 education mix, so
the obvious objection is that it is a proxy for how educated a workforce
is and nothing more. The placebo that answers it cuts employers on
education intensity alone and asks whether that cut reproduces the step.
The table enters the two cuts separately and then together.

WHAT IS ESTIMATED
The adoption step from January 2024 on the specification of Equation (2)
with the calendar terms in: Poisson pseudo-maximum likelihood on employer
by age by month counts, with employer-by-month, employer-by-age and
month-by-age effects, exposure frozen at the employer's 2019 education
mix, and standard errors clustered by employer (script 78, lane 25 part
D). Education intensity is the top quartile of the 2019 post-secondary
share of incumbents, employment weighted, at the same floor of five
person-months as exposure itself. The two cuts are collinear by
construction, since three quarters of the exposed employers are also
education-intensive, and the third and fourth rows are what a collinear
pair does when only one of the two carries the effect: the coefficients
move apart and the one carrying nothing takes the opposite sign.

Two rules were fixed before the run. The education cut reproduces the
step if it is at least as large as the exposure step allowing one
standard error; exposure survives if it keeps at least half its size with
both cuts in. The first fails at both bands and the second passes at
both, which the run's summary records as its own verdict.

INPUTS AND OUTPUTS
Reads skill_placebo.csv (script 78, lane 25 part D; columns young_band,
spec, term, coef, se, n_obs, status), vcov_s78_skill_<spec>_<band>.csv
(the exported covariance of each fit) and 78_summary.txt, the run's own
report of the same steps and of the cut, the share of employers above it
and the overlap with exposure that the note quotes, from the export
directory the final-code manifest names or one given on the command line.
Nothing is typed in. Writes revision/tables/tableA_skill_placebo.tex and
copies it to canaries-sweden-paper/tables/; the appendix carries the
float, the caption and the label and inputs this file inside them.

    python3 revision/local/l33_tab_skill_placebo.py [export_dir]

THE GATE
Each of the three specifications must be exported once per band with
status ok, the specifications entered alone must carry only their own
family of terms and the joint specification both, and each band must
report one panel size across all three. Every standard error must equal
the square root of its own diagonal in that fit's covariance matrix,
which is the integrity check this export offers. Every printed step is
then checked against part D of the run's summary, which reports the same
four steps per band independently of the CSV, and the cut, the share
above it and the overlap are read from the summary rather than typed.
Finally each printed number is read back from the string that goes into
the table and compared with the export it came from, and any disagreement
beyond half of the last printed digit stops the script and nothing is
written.

IN THE PAPER
Online Appendix III.2, the paragraph on the exposure measure against an
education-intensity cut, Table tab:skill_placebo.
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
AI, EDU = "post_x_high_x_young", "post_x_highedu_x_young"
# Specification in the export, the file tag of its covariance, and the
# family of terms it may carry.
SPECS = {"daioe_alone": ("ai", ("_x_high_x_young",)),
         "skill_alone": ("edu", ("_x_highedu_x_young",)),
         "both": ("both", ("_x_high_x_young", "_x_highedu_x_young"))}
# Row label, specification, term, and the name the run's summary gives it.
ROWS = [
    ("Exposure, entered alone", "daioe_alone", AI, "DAIOE step alone"),
    ("Education intensity, entered alone", "skill_alone", EDU,
     "skill-cut step alone"),
    (None, None, None, None),                       # \addlinespace
    ("Exposure, both entered", "both", AI, "DAIOE step, both in"),
    ("Education intensity, both entered", "both", EDU,
     "skill-cut step, both in"),
]
LABEL_WIDTH = 36
CELL = re.compile(r"^\$([-+])\$([0-9.]+) \(([0-9.]+)\)$")
SUMMARY_ROW = re.compile(
    r"^\s+(22-25|26-30) (DAIOE step alone|skill-cut step alone|"
    r"DAIOE step, both in|skill-cut step, both in)\s+"
    r"([-+][0-9.]+) \(([0-9.]+)\)\s*$")
SUMMARY_CUT = re.compile(
    r"^\s+D: post-secondary share cut at ([0-9.]+); ([0-9.]+)% of scored "
    r"employers are High_edu; ([0-9.]+)% of DAIOE Q4 employers are also "
    r"High_edu\s*$")


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
    """The step in the sign-outside form this table uses, checked against
    the export before it is allowed into the table. No stars: the
    paragraph reads the four steps against each other and against the two
    rules, not against zero."""
    out = f"${'+' if c >= 0 else '-'}${abs(c):.4f} ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted step is unreadable")
    signed = float(m.group(1) + m.group(2))
    if abs(signed - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed step {out} disagrees with "
                         f"skill_placebo.csv ({c:+.6f}, {se:.6f})")
    return out


def summary_part_d(path: Path) -> tuple[dict, tuple[str, str, str]]:
    """Part D of the run's summary: the four steps per band as the run
    printed them, and the cut, the share of scored employers above it and
    the overlap with exposure that the note quotes."""
    said, cut = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ROW.match(line)
        if m:
            said[(m.group(1), m.group(2))] = (
                f"${m.group(3)[0]}${m.group(3)[1:]} ({m.group(4)})")
            continue
        c = SUMMARY_CUT.match(line)
        if c:
            cut = c.groups()
    if len(said) != len(BANDS) * 4 or cut is None:
        raise SystemExit(f"  {path.name}: part D does not report the four "
                         f"steps of both bands and the cut")
    return said, cut


def main() -> int:
    d = pd.read_csv(source(LANE25A, "skill_placebo.csv"))
    if not (d.status == "ok").all():
        raise SystemExit("  skill_placebo.csv: a fit reports a status other "
                         "than ok, so it is not quotable")
    panel = {}
    for band in BANDS:
        b = d[d.young_band == band]
        if sorted(set(b.spec)) != sorted(SPECS):
            raise SystemExit(f"  skill_placebo.csv: {band} carries the "
                             f"specifications {sorted(set(b.spec))}, expected "
                             f"{sorted(SPECS)}")
        for spec, (tag, families) in SPECS.items():
            s = b[b.spec == spec]
            stray = [t for t in s.term
                     if not any(t.endswith(f) for f in families)]
            if stray:
                raise SystemExit(f"  {band} {spec}: carries terms outside its "
                                 f"own family: {stray}")
            v = pd.read_csv(source(LANE25A,
                                   f"vcov_s78_skill_{tag}_"
                                   f"{band.replace('-', '_')}.csv"),
                            index_col=0)
            for _, r in s.iterrows():
                if r.term not in v.index:
                    raise SystemExit(f"  {band} {spec} {r.term}: not in the "
                                     f"exported covariance")
                diag = float(v.loc[r.term, r.term]) ** 0.5
                if abs(diag - float(r.se)) > 5e-5:
                    raise SystemExit(f"  {band} {spec} {r.term}: the exported "
                                     f"standard error {float(r.se):.6f} is not "
                                     f"the square root of its own variance "
                                     f"{diag:.6f}")
        n = sorted(set(int(x) for x in b.n_obs))
        if len(n) != 1:
            raise SystemExit(f"  {band}: one panel size expected across the "
                             f"three specifications, found {n}")
        panel[band] = n[0]

    said, (cut, above, overlap) = summary_part_d(source(LANE25A,
                                                        "78_summary.txt"))
    d = d.set_index(["young_band", "spec", "term"])

    rows = []
    for label, spec, term, name in ROWS:
        if label is None:
            rows.append(r"\addlinespace")
            continue
        cells = []
        for band in BANDS:
            if (band, spec, term) not in d.index:
                raise SystemExit(f"  {band} {spec}: no {term} in "
                                 f"skill_placebo.csv")
            r = d.loc[(band, spec, term)]
            text = cell(f"{band} {spec} {term}", float(r.coef), float(r.se))
            if said.get((band, name)) != text:
                raise SystemExit(f"  {band} {name}: the table would print "
                                 f"{text}, the run's summary reports "
                                 f"{said.get((band, name), 'nothing')}")
            cells.append(text)
        rows.append(f"{label:<{LABEL_WIDTH}s} & " + " & ".join(cells) + r" \\")
        print(f"  {label:<{LABEL_WIDTH}s} " + "  ".join(cells))
    print(f"  cut {cut}; {above} per cent of scored employers above it; "
          f"{overlap} per cent of exposed employers also above it")
    print(f"  panel {panel[BANDS[0]]:,} cells at {BANDS[0]} and "
          f"{panel[BANDS[1]]:,} at {BANDS[1]}")

    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           "Adoption step on & " + " & ".join(b.replace("-", "--")
                                              for b in BANDS) + r" \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}", "",
            r"\vspace{0.5em}",
            r"\begin{minipage}{0.86\textwidth}",
            r"\footnotesize \textit{Notes:} The adoption step from January 2024, on the",
            r"specification of Equation~(2) with the calendar terms in, standard errors",
            r"clustered by employer in parentheses. Exposure is the top quartile of the 2019",
            r"firm education mix used throughout; education intensity is the top quartile of",
            r"the 2019 post-secondary share of incumbents, employment weighted, at the same",
            f"floor of five person-months and cut at {cut}. Of scored employers {above} per cent",
            f"are education-intensive on that cut, and {overlap} per cent of exposed employers are",
            f"also education-intensive. Each fit uses {thousands(panel[BANDS[0]])} cells at "
            f"{BANDS[0].replace('-', '--')} and",
            f"{thousands(panel[BANDS[1]])} at {BANDS[1].replace('-', '--')}.",
            r"\end{minipage}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_skill_placebo.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

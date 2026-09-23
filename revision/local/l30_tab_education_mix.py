#!/usr/bin/env python3
"""
l30_tab_education_mix.py: Online Appendix Table III.2
(tab:education_mix), where young women and young men work, by field and
level of education, inside and outside top-quartile employers.

THE QUESTION
The split of the female differential in Table tab:gender_split leaves a
residual it calls composition. This table is the descriptive object that
residual refers to: it shows what young women and young men in exposed
employers actually studied, beside the same distribution outside them.
It is what makes composition a candidate at all, and it is also the
answer to the reader who asks whether exposed employers simply hire
different people. Nothing here is an estimate; the table is a set of
shares.

WHAT IS REPORTED
Shares of person-months in 2023, the last year with a contemporaneous
education record, among employed workers aged 22 to 25 with a birth year,
a sex and a payslip (script 87, lane 33). The first four columns are the
field and the level of education by sex, in top-quartile employers and in
the other three quartiles. THE QUARTILES ARE THE OCCUPATION ROUTE'S: an
employer is scored by the 2019 occupations of its incumbents aged 31 to
69, as in Table 1, which is why this table is rebuilt on this route
rather than carried over from script 76. The last two columns give the
mean DAIOE score of the education group's occupations, which prices the
education GROUP and is not the employer exposure; it is loaded from
47h's score book for the descriptive columns alone. Shares are of the
sex-and-quartile column, so each block of six adds to a hundred.

Reading the columns beside each other is the point: young women in
exposed employers are concentrated in business, law and administration
and are rarer in engineering than young men, and both sexes in exposed
employers hold more post-secondary education than their counterparts
elsewhere.

INPUTS AND OUTPUTS
Reads occ_route_education_mix_by_sex.csv (script 87, lane 33; columns dimension,
exposed, gender, cell, person_months, persons_avg, share, mean_score,
scored_share, year) from the export directory the final-code manifest
names or one given on the command line. Nothing is typed in; the scored
share quoted in the note is read from the export as well. Writes
revision/tables/tableA_education_mix.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l30_tab_education_mix.py [export_dir]

THE GATE
The export must cover one year, and it must be the year the caption
names. Every cell of every block must appear exactly once, and the shares
of each block must sum to one, which is what makes them shares of a
column rather than of anything else. Each cell's person-months must be
twelve times its average person count, since the average is taken over
the twelve months of the year, and no cell may fall below the disclosure
floor of five persons. A mean score must be present exactly when the cell
reports a scored share above zero, and absent otherwise; the two cells
with no scored worker print a dash rather than a number. Every printed
share and score is then read back from the string that goes into the
table and compared with the export it came from, and any disagreement
beyond half of the last printed digit stops the script and nothing is
written.

IN THE PAPER
Online Appendix III.2, the paragraph on the female differential split,
Table tab:education_mix; the ICT shares are quoted in the note of
Table tab:contrast_by_track.
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
LANE33 = OUT / "round3_20260923-1352-lane33-script87"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

YEAR = 2023
MONTHS = 12          # the average person count is taken over the year
FLOOR = 5            # the disclosure floor, persons per cell
# The note spells the floor out; the map exists so that the number the
# gate enforces and the word the note prints cannot drift apart.
FLOOR_IN_WORDS = {3: "three", 5: "five", 10: "ten"}
# Block, export key, row label, and whether the row carries a mean score.
TRACKS = [
    ("business_law_social", "Business, law, administration, social science"),
    ("engineering", "Engineering, manufacturing, construction"),
    ("health_education_care", "Health, education, social care"),
    ("ict", "ICT"),
    ("other", "Other fields"),
    ("na", "No record"),
]
LEVELS = [
    ("below_upper_secondary", "Below upper secondary"),
    ("upper_secondary", "Upper secondary"),
    ("post_secondary", "Post-secondary"),
    ("na", "No record"),
]
# The four share columns: top quartile then the rest, women then men.
COLUMNS = [(1, "women"), (1, "men"), (0, "women"), (0, "men")]
SHARE = re.compile(r"^([0-9]+\.[0-9])$")
SCORE = re.compile(r"^([0-9]+)$")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def share_cell(what: str, x: float) -> str:
    """A share as a percentage, checked against the export before it is
    allowed into the table."""
    out = f"{100 * x:.1f}"
    m = SHARE.match(out)
    if m is None or abs(float(m.group(1)) - 100 * x) > 5e-2:
        raise SystemExit(f"  {what}: the printed share {out} disagrees with "
                         f"the export ({100 * x:.4f})")
    return out


def score_cell(what: str, x: float) -> str:
    """A mean DAIOE score, checked against the export before it is
    allowed into the table."""
    out = f"{x:.0f}"
    m = SCORE.match(out)
    if m is None or abs(float(m.group(1)) - x) > 0.5:
        raise SystemExit(f"  {what}: the printed score {out} disagrees with "
                         f"the export ({x:.4f})")
    return out


def main() -> int:
    d = pd.read_csv(source(LANE33, "occ_route_education_mix_by_sex.csv"))
    years = sorted(set(int(y) for y in d.year))
    if years != [YEAR]:
        raise SystemExit(f"  occ_route_education_mix_by_sex.csv: the "
                         f"caption names {YEAR}, the export covers {years}")

    blocks = {"track": [k for k, _ in TRACKS], "level": [k for k, _ in LEVELS]}
    for dimension, cells in blocks.items():
        for exposed, gender in COLUMNS:
            b = d[(d.dimension == dimension) & (d.exposed == exposed)
                  & (d.gender == gender)]
            bad = [c for c in cells if (b.cell == c).sum() != 1]
            if bad or len(b) != len(cells):
                raise SystemExit(f"  {dimension} {exposed} {gender}: one row "
                                 f"expected per cell, wrong for "
                                 f"{bad or 'the row count'}")
            total = float(b.share.sum())
            if abs(total - 1.0) > 1e-6:
                raise SystemExit(f"  {dimension} {exposed} {gender}: the "
                                 f"shares sum to {total:.6f}, so they are not "
                                 f"shares of this column")

    if (d.persons_avg < FLOOR).any():
        raise SystemExit(f"  occ_route_education_mix_by_sex.csv: a cell "
                         f"holds fewer than {FLOOR} persons on average, below "
                         f"the disclosure floor")
    off = (d.person_months - MONTHS * d.persons_avg).abs().max()
    if off > 1e-6:
        raise SystemExit(f"  occ_route_education_mix_by_sex.csv: "
                         f"person-months are not {MONTHS} times the average "
                         f"person count (off by {off:.2e})")
    scored = d.scored_share > 0
    if (scored != d.mean_score.notna()).any():
        raise SystemExit("  occ_route_education_mix_by_sex.csv: a mean "
                         "score is present without a scored worker, or "
                         "absent with one")
    lo = int(100 * d.loc[scored, "scored_share"].min())
    hi = int(round(100 * d.loc[scored, "scored_share"].max()))

    d = d.set_index(["dimension", "exposed", "gender", "cell"])

    def shares(dimension: str, key: str) -> list[str]:
        return [share_cell(f"{dimension} {key} {e} {g}",
                           float(d.loc[(dimension, e, g, key), "share"]))
                for e, g in COLUMNS]

    rows = [r"\multicolumn{7}{l}{\textit{Field of education (track)}} \\"]
    for key, label in TRACKS:
        cells = shares("track", key)
        for gender in ("women", "men"):
            score = d.loc[("track", 1, gender, key), "mean_score"]
            cells.append("--" if pd.isna(score)
                         else score_cell(f"track {key} {gender} score",
                                         float(score)))
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
        print(f"  track {label[:44]:44s} " + " ".join(f"{c:>5s}" for c in cells))
    rows += [r"\addlinespace[3pt]",
             r"\multicolumn{7}{l}{\textit{Level of education}} \\"]
    for key, label in LEVELS:
        cells = shares("level", key)
        rows.append(f"{label} & " + " & ".join(cells) + r" & & \\")
        print(f"  level {label[:44]:44s} " + " ".join(f"{c:>5s}" for c in cells))
    print(f"  scored share {lo} to {hi} per cent of each cell")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Where young women and young men work: education of "
           r"employed 22--25 year olds by sex, inside and outside "
           f"top-quartile employers, {YEAR}." + "}",
           r"\label{tab:education_mix}", r"\footnotesize",
           # Seven columns beside the long track labels: the same
           # overflow as the gender split, and the same remedy.
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Top quartile (\%)} & "
           r"\multicolumn{2}{c}{Other quartiles (\%)} & "
           # "Mean DAIOE score, top quartile" is wider than the two
           # columns under it and pushed the table past the text block
           # even at a reduced column separation; the top-quartile
           # restriction moves into the note.
           r"\multicolumn{2}{c}{Mean DAIOE score} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r" & Women & Men & Women & Men & Women & Men \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            f"Shares of person-months in {YEAR}, the last year with a "
            r"contemporaneous education record, among employed workers aged "
            r"22--25 with a birth year, a sex and a payslip; every cell holds "
            f"at least {FLOOR_IN_WORDS[FLOOR]} persons on average. The mean DAIOE "
            "score is "
            r"the generative-AI percentile of the education group's "
            r"occupations in 2019, averaged over the workers in the cell "
            f"with a scored group ({lo} to {hi} per cent of each cell), in "
            r"top-quartile employers; it prices the education group and is "
            r"not the exposure that defines the quartiles, which is the "
            r"employer's 2019 occupation mix. Tracks as in "
            r"Table~\ref{tab:gender_split}. Source: script 87.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_education_mix.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

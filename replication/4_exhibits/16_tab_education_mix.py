#!/usr/bin/env python3
"""
16_tab_education_mix.py: Online Appendix Table A16 (Section III.2), the
education of employed 22-25 year olds by sex, inside and outside top-quartile
employers, 2023.

Shares of person-months in 2023, the last year with a contemporaneous
education record, among employed workers aged 22 to 25 with a birth year, a
sex and a payslip (script 87): field and level of education by sex, in
top-quartile employers (2019 occupation-mix quartiles) and in the other three
quartiles. The last two columns give the mean DAIOE score of the education
group's occupations in top-quartile employers, which prices the education
group and is not the employer exposure. No estimate; a table of shares.

Nothing is written unless the export covers 2023 only, every block sums to
one, person-months equal twelve times the average person count, every cell
holds at least five persons, and a mean score is present exactly where a cell
has a scored worker.

Export read: 3_register_mona/exports/2026-09-23_1352_s87/occ_route_education_mix_by_sex.csv
Output: output/tables/tableA_education_mix.tex

    python 4_exhibits/16_tab_education_mix.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S87 = EXPORTS / "2026-09-23_1352_s87"

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
    d = pd.read_csv(source(S87, "occ_route_education_mix_by_sex.csv"))
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
            r"Shares of person-months in 2023, the last year with a contemporaneous education record, among employed workers aged 22--25. The mean DAIOE score prices the education group's 2019 occupations for workers in top-quartile employers; it is not the exposure that defines the quartiles, which is the employer's 2019 occupation mix. Tracks as in Table~\ref{tab:gender_split}.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_education_mix.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l41_fig_firststage_v3.py: Online Appendix Figure A6, the first stage, on the
occupation route.

WHY A NEW SCRIPT. l16 draws the same rows on the EDUCATION route: its filter
is `route == "education"` and its pinned export is lane 17. The v3 paper
scores an employer from the 2019 occupations of its own incumbents, and the
appendix paragraph beside the figure quotes the occupation-route numbers
(20.9 points in the 2023 firm survey, 22.5 in the 2024 worker survey). Until
this script existed the figure and its own paragraph were on different
measures. l16 stays as it is so the v2 figure remains reproducible.

WHAT IT DRAWS
The difference in reported AI use between employers in the top quartile of
the 2019 occupation-mix exposure and the rest, in percentage points with 95
per cent intervals: any AI use in 2019, 2021 and 2023 and language generation
in 2021 and 2023 from Statistics Sweden's ICT survey of enterprises
(controlling for log 2019 employment), and generative AI use by the
employer's workers in the 2024 survey of individuals (survey weighted). The
expenditure surveys are left out because they measure AI spending rather than
use and belong in the appendix text. A row missing from the export is
reported as missing, never drawn as zero.

THE GATE. The script refuses to write unless the two rows the appendix and
the main paper quote in prose come back at the printed precision. A figure
that disagrees with the sentence beside it is the defect this replaces, so it
is checked rather than assumed.

    python3 revision/local/l41_fig_firststage_v3.py [export_dir]

INPUTS AND OUTPUTS
Reads occ_rest_firststage.csv (script 83, lane 29a; columns term, coef, se,
n, source, route, outcome, coef_points, se_points, t, wave) from the pinned
export directory or one given on the command line. Writes
revision/figures/figA2_first_stage_v3.pdf and .png through _figsafe.save and
copies both to canaries-sweden-paper/figures/.

IN THE PAPER
Online Appendix III.2, Figure fig:first_stage; the numbers are quoted in
Section 2 of the paper and in the same appendix paragraph.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsafe import save  # noqa: E402

PAPER = REV.parents[1] / "canaries-sweden-paper"
plt.rcParams.update({"font.family": "serif", "font.size": 11})

LANE29A = REV / "output" / "round3_20260922-2333-lane29a"
EXPORT = "occ_rest_firststage.csv"
ROUTE = "occupation"

# (label, source, outcome, group, marker, facecolour)
ROWS = [
    ("Uses AI, 2019",                  "ai_itftg_2019",    "ai_any",
     "firm level",       "o", "white"),
    ("Uses any AI technology, 2021",   "ITFtg_Stora_2021", "ai_any",
     "firm level",       "o", "white"),
    ("Uses any AI technology, 2023",   "ITFtg_Stora_2023", "ai_any",
     "firm level",       "o", "white"),
    ("Uses language generation, 2021", "ITFtg_Stora_2021", "ai_genai",
     "firm level",       "s", "0.45"),
    ("Uses language generation, 2023", "ITFtg_Stora_2023", "ai_genai",
     "firm level",       "s", "0.45"),
    ("Worker used generative AI, 2024", "BITA_2024",       "genai",
     "individual level", "D", "black"),
]

# (source, outcome, points, se_points, t, n) exactly as the prose prints them.
QUOTED = [
    ("ITFtg_Stora_2023", "ai_any", 20.9, 1.5, 14.4, 3587),
    ("BITA_2024",        "genai",  22.5, 2.5, 9.0,  2285),
]


def gate(d: pd.DataFrame) -> list:
    """The prose is the specification; a figure that disagrees is wrong."""
    bad = []
    for source, outcome, pp, se, t, n in QUOTED:
        r = d[(d.source == source) & (d.outcome == outcome)]
        if r.empty:
            bad.append(f"{source}/{outcome} absent from the export")
            continue
        r = r.iloc[0]
        got = (round(float(r.coef) * 100, 1), round(float(r.se) * 100, 1),
               round(float(r.coef) / float(r.se), 1), int(r.n))
        if got != (pp, se, t, n):
            bad.append(f"{source}/{outcome}: export {got} against "
                       f"prose {(pp, se, t, n)}")
    return bad


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE29A
    p = d / EXPORT
    if not p.exists():
        print(f"  missing input: {p}")
        return 1
    print(f"  reading {p}")
    d = pd.read_csv(p)
    d = d[(d.term == "high") & (d.route == ROUTE)]
    if d.empty:
        print(f"  no {ROUTE}-route rows in the export")
        return 1

    bad = gate(d)
    if bad:
        print("  NOT WRITTEN, the figure would disagree with the prose:")
        for b in bad:
            print("    -", b)
        return 1

    got, missing = [], []
    for label, source, outcome, group, marker, face in ROWS:
        r = d[(d.source == source) & (d.outcome == outcome)]
        if r.empty:
            missing.append(f"{label} ({source}/{outcome})")
            continue
        got.append(dict(label=label, group=group, marker=marker, face=face,
                        pp=float(r.coef.iloc[0]) * 100,
                        se=float(r.se.iloc[0]) * 100))
    if missing:
        print(f"  MISSING, not zero: {'; '.join(missing)}")
    if not got:
        return 1

    n = len(got)
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * n + 1.9))
    for i, r in enumerate(got):
        y = n - 1 - i
        ax.plot([r["pp"] - 1.96 * r["se"], r["pp"] + 1.96 * r["se"]],
                [y, y], color="black", lw=1.1, solid_capstyle="butt",
                zorder=2)
        ax.plot(r["pp"], y, r["marker"], mfc=r["face"], mec="black",
                mew=1.1, ms=8 if r["marker"] == "D" else 9, zorder=3)

    seen, rules = [], []
    for i, r in enumerate(got):
        if r["group"] not in seen:
            seen.append(r["group"])
            if i:
                rules.append((n - 0.5 - i, r["group"]))
    ax.text(30, n - 0.42, "firm level", fontsize=10.5, style="italic",
            color="0.45", ha="right", va="bottom")
    for ypos, name in rules:
        ax.axhline(ypos, color="0.82", lw=0.9, zorder=1)
        ax.text(30, ypos + 0.10, name, fontsize=10.5, style="italic",
                color="0.45", ha="right", va="bottom")
    ax.axhline(n - 3.5, color="0.82", lw=0.9, zorder=1)

    ax.axvline(0, color="black", lw=1.0, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in reversed(got)], fontsize=11)
    ax.set_xlim(-1.2, 30)
    ax.set_ylim(-0.6, n - 0.25)
    ax.set_xlabel("Difference between top-quartile exposed firms and the "
                  "rest\n(percentage points, controlling for log 2019 "
                  "size)", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=3)

    save(fig, "figA2_first_stage_v3", __file__)
    plt.close(fig)
    for ext in (".pdf", ".png"):
        src = REV / "figures" / f"figA2_first_stage_v3{ext}"
        if src.exists():
            shutil.copy2(src, PAPER / "figures" / src.name)
    print(f"    saved figA2_first_stage_v3.pdf/.png ({n} rows, "
          f"{ROUTE} route) and copied to the paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

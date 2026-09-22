#!/usr/bin/env python3
"""
l16_fig_firststage.py: Figure A2, the first stage of the exposure measure.

WHAT IT DRAWS
The difference in reported AI use between employers in the top quartile of
the 2019 education-mix exposure and the rest, in percentage points with
95 per cent intervals, from the first-stage estimates of script 71 on the
education route: any AI use in 2019, 2021 and 2023 and language generation
in 2021 and 2023 from Statistics Sweden's ICT survey of enterprises
(controlling for log 2019 employment), and generative AI use by the
employer's workers in the 2024 survey of individuals (survey weighted).
The expenditure surveys are left out because they measure AI spending
rather than use and belong in the appendix text. A row whose estimate is
missing from the export is reported as missing, never drawn as zero.

    python3 revision/local/l16_fig_firststage.py [export_dir]

INPUTS AND OUTPUTS
Reads itftg_firststage.csv and bita_firststage.csv from the script 71
export directory the final-code manifest names (or from a directory given
on the command line). Writes revision/figures/figA2_first_stage.pdf and
.png through _figsafe.save.

IN THE PAPER
Online Appendix III.2, Figure fig:first_stage; the numbers are quoted in
Section 2.
"""
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
from config import V2_FIG  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 11})

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


# The script 71 export the final-code manifest names, whose adoption flag
# is built from the technology items only. Pass a directory on the command
# line to read from there instead.
LANE17 = REV / "output" / "round3_20260921-lane17-adoption-corrected"


def find(argv, name):
    d = Path(argv[1]) if len(argv) > 1 else LANE17
    p = d / name
    return p if p.exists() else None


def main() -> int:
    frames = []
    for nm in ("itftg_firststage.csv", "bita_firststage.csv"):
        f = find(sys.argv, nm)
        if f is not None:
            print(f"  reading {f}")
            frames.append(pd.read_csv(f))
    if not frames:
        print("  no first-stage export found; run lane 17 and export it")
        return 1
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.term == "high") & (d.route == "education")]

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

    # group rules, with the section label above the rule so it never sits
    # on the line
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
    # the marker-change inside the firm block gets a rule but no label
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

    save(fig, "figA2_first_stage", __file__)
    plt.close(fig)
    print(f"    saved figA2_first_stage.pdf/.png ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l16_fig_firststage.py -- do the firms we call exposed actually adopt AI?

WHY THIS EXISTS, AND A WARNING. A finished version of this figure sat in
`figures/figA2_first_stage.pdf` on 21 September with NO GENERATING
SCRIPT anywhere in the tree, and it was overwritten by accident. That
folder is gitignored, so there was no version history and the only
surviving copy was a window someone happened to have open. No figure in
this paper may depend on that again: every exhibit is produced by a
script that reads an export, and the script is the artefact we keep.

This reproduces that design from the export. If the original is
recovered, diff against it rather than assuming this matches.

WHAT IT SHOWS. Exposure is assigned from the 2019 education mix of
incumbents aged 31 and over, and never from anything a firm did after
the shock. The question is whether that assignment predicts AI adoption
measured independently in SCB's own surveys.

WHAT IS DELIBERATELY LEFT OUT. The R&D surveys and the IT-expenditure
survey measure AI SPENDING, not use. They belong to a different
construct, they would sit at +6.6 and +5.4 beside use rates near +20,
and a reader would read that as disagreement rather than as a different
question. They are reported in the appendix text instead.

    python3 revision/local/l16_fig_firststage.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
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


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if not r.exists():
            continue
        for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
            # the first version of 71 built its adoption flag by
            # pattern-matching "AI" and swept in nine barrier items plus
            # AI_USE_N; that export is withdrawn
            if "lane17" in str(p) and "corrected" not in str(p):
                continue
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


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

    # group rules, and the section label ABOVE the rule so it never sits
    # on the line -- that overlap is the defect in the version this
    # replaces
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

    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(V2_FIG / f"figA2_first_stage{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"    saved figA2_first_stage.pdf/.png ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

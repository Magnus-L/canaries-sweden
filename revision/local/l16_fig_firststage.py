#!/usr/bin/env python3
"""
l16_fig_firststage.py -- do the firms we call exposed actually adopt AI?

WHY THIS EXISTS. figA2_first_stage.pdf had no generating script. This
rebuilds it from lane 17's CORRECTED export: the first version of script
71 built its adoption flag by pattern-matching the string "AI", which
swept in nine barrier items answered by non-users plus AI_USE_N, a
variable that literally means "does not use AI". The +25.7 pp it
reported was withdrawn. Read only `*-corrected` exports here.

WHAT IT SHOWS. Exposure is assigned from the 2019 education mix of
incumbents aged 31+, and never from anything a firm did after the shock.
The question is whether that assignment predicts actual AI adoption
measured independently, in SCB's own firm and worker surveys. Each row
is the high-exposure coefficient in percentage points, from a separate
survey and year.

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
from config import V2_FIG, DARK_BLUE, ORANGE, GRAY, DARK_TEXT

PRETTY = {"ITFtg_Stora_2023": "IT survey, firms 10+, 2023",
          "ITFtg_Stora_2021": "IT survey, firms 10+, 2021",
          "ai_itftg_2019": "IT survey, AI module, 2019",
          "ai_fouoff_2019_2021": "R&D survey, public, 2019-21",
          "ai_fouftg_2019_2021": "R&D survey, firms, 2019-21",
          "ai_fufi_2019": "IT expenditure survey, 2019",
          "BITA_2024": "BITA, workers, genAI, 2024"}


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if not r.exists():
            continue
        for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
            if "corrected" not in str(p) and "lane17" in str(p):
                continue          # the withdrawn first version
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def main() -> int:
    rows = []
    f = find(sys.argv, "itftg_firststage.csv")
    if f is not None:
        d = pd.read_csv(f)
        d = d[(d.term == "high") & (d.route == "education")
              & (d.outcome.isin(["ai_any", "genai"]))]
        rows.append(d.assign(level="firm"))
        print(f"  reading {f}")
    f = find(sys.argv, "bita_firststage.csv")
    if f is not None:
        d = pd.read_csv(f)
        d = d[(d.term == "high") & (d.route == "education")]
        rows.append(d.assign(level="worker"))
        print(f"  reading {f}")
    if not rows:
        print("  no first-stage export found; run lane 17 and export it")
        return 1

    d = pd.concat(rows, ignore_index=True)
    d["pp"] = d["coef"] * 100
    d["lo"] = (d["coef"] - 1.96 * d["se"]) * 100
    d["hi"] = (d["coef"] + 1.96 * d["se"]) * 100
    d["label"] = d["source"].map(lambda s: PRETTY.get(s, s))
    d = d.sort_values("pp")

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(d) + 1.4))
    y = range(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        c = ORANGE if r["level"] == "worker" else DARK_BLUE
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=1.6,
                solid_capstyle="butt")
        ax.plot(r["pp"], yi, "o", color=c, ms=6)
        ax.text(r["hi"] + 1.2, yi, f"{r['pp']:+.1f}", fontsize=8.5,
                va="center", color=DARK_TEXT)
    ax.axvline(0, color=DARK_TEXT, lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlabel("High exposure, percentage points of AI adoption",
                  fontsize=9.5)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.text(0.99, 0.02, "blue: firm surveys   orange: worker survey",
            transform=ax.transAxes, ha="right", fontsize=8, color=GRAY)

    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(V2_FIG / f"figA2_first_stage{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"    saved figA2_first_stage.pdf/.png ({len(d)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

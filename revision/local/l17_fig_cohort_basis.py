#!/usr/bin/env python3
"""
l17_fig_cohort_basis.py -- is the calendar cycle an artefact of moving
age bands?

WHY THIS EXISTS. figA3_seasonal_basis.pdf had no generating script.

WHAT IT SHOWS. A band like 22-25 loses a whole birth cohort and gains
another every January, mechanically, so a cycle in the exposure
differential could be turnover rather than hiring behaviour. Script 69
rebuilds the same pre-period cycle on FIXED BIRTH COHORTS, where nobody
crosses a boundary. If the cycle is mechanical it should largely vanish;
if it is real it should survive.

THE RULE WAS FIXED BEFORE THE RUN, and it is drawn on the figure so a
reader can see the verdict was not chosen afterwards: on the LARGEST
pre-period calendar coefficient, a cohort-to-ageband ratio at or below
0.50 means mechanical, at or above 0.75 means real, and anything between
is ambiguous and settles nothing.

    python3 revision/local/l17_fig_cohort_basis.py [export_dir]
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

MECHANICAL, REAL = 0.50, 0.75
NICE = {"hy_q1": "Q1", "hy_q2": "Q2", "hy_q3": "Q3", "hy_q4": "Q4"}


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if r.exists():
            for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
                if best is None or p.stat().st_mtime > best.stat().st_mtime:
                    best = p
    return best


def main() -> int:
    src = find(sys.argv, "cohort_season.csv")
    if src is None:
        print("  no cohort_season.csv found; run lane 15 and export it")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src)
    band = sorted(d["young_band"].unique())[0]
    d = d[d["young_band"] == band]
    piv = d.pivot(index="term", columns="basis", values="coef")
    se = d.pivot(index="term", columns="basis", values="se")
    piv["ratio"] = piv["cohort"].abs() / piv["ageband"].abs()
    biggest = piv["ageband"].abs().idxmax()
    r = float(piv.loc[biggest, "ratio"])
    verdict = ("MECHANICAL" if r <= MECHANICAL else
               "REAL" if r >= REAL else "AMBIGUOUS")

    terms = list(piv.index)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    lows, labels = [], []
    for i, t in enumerate(terms):
        a, c = piv.loc[t, "ageband"], piv.loc[t, "cohort"]
        ax.plot([i, i], [a, c], color=GRAY, lw=1.2, zorder=1)
        ax.errorbar(i - 0.06, a, yerr=1.96 * se.loc[t, "ageband"], fmt="o",
                    color=DARK_BLUE, ms=6, capsize=3, lw=1.4,
                    label="moving age bands" if i == 0 else None, zorder=2)
        ax.errorbar(i + 0.06, c, yerr=1.96 * se.loc[t, "cohort"], fmt="s",
                    color=ORANGE, ms=6, capsize=3, lw=1.4,
                    label="fixed birth cohorts" if i == 0 else None,
                    zorder=2)
        # below the LOWER whisker, not below the lower marker, or the
        # label sits on top of the error bar it is describing
        floor = min(a - 1.96 * se.loc[t, "ageband"],
                    c - 1.96 * se.loc[t, "cohort"])
        lows.append(floor)
        labels.append((i, floor, piv.loc[t, "ratio"], t == biggest))

    pad = 0.055 * (max(piv[["ageband", "cohort"]].max()) - min(lows))
    for i, floor, ratio, is_big in labels:
        ax.text(i, floor - pad,
                f"ratio {ratio:.2f}" + ("\n(largest)" if is_big else ""),
                fontsize=8.2, ha="center", va="top", color=DARK_TEXT)
    ax.set_ylim(min(lows) - 4.2 * pad, None)

    ax.axhline(0, color=DARK_TEXT, lw=0.8)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([NICE.get(t, t) for t in terms], fontsize=10)
    ax.set_xlim(-0.5, len(terms) - 0.5)
    ax.set_ylabel(f"Pre-period calendar coefficient, {band}", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(
        f"Rule fixed before the run: <={MECHANICAL:.2f} mechanical, "
        f">={REAL:.2f} real.  Largest is {NICE.get(biggest, biggest)} "
        f"at {r:.2f}: {verdict}",
        fontsize=8.5, color=DARK_TEXT, loc="left", pad=10)

    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(V2_FIG / f"figA3_seasonal_basis{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"    saved figA3_seasonal_basis.pdf/.png  "
          f"(largest {biggest}, ratio {r:.3f}, {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

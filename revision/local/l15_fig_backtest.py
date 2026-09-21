#!/usr/bin/env python3
"""
l15_fig_backtest.py -- the as-of backtest, the exhibit the revision turns on.

WHY THIS EXISTS. `figures/figA1_asof_backtest.pdf` had no generating
script anywhere in the tree, and neither did figA2 or figA3. For the
figure that carries the paper's central argument that is not acceptable:
nobody can tell which export it came from, it cannot be re-rendered when
new numbers land, and it cannot go into a replication package.

WHAT IT SHOWS. Impose the register's two-year staleness on a period
where the truth is observable. At each truncation T the same design is
run twice: once on the codes as they will eventually read (`true`), once
on the codes as they read at T (`asof`). The gap between them is what
the lag manufactures out of nothing. It is large, negative, and it is
the whole reason the submitted headline was withdrawn.

Read the panel left to right: the true arm sits at zero, the as-of arm
sits far below it, and the distance is the artefact.

    python3 revision/local/l15_fig_backtest.py [export_dir]
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


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if not r.exists():
            continue
        for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def main() -> int:
    src = find(sys.argv, "asof_estimates.csv")
    if src is None:
        print("  no asof_estimates.csv found; run script 45 and export it")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src).sort_values(["trunc", "assignment"])

    truncs = sorted(d["trunc"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style = {"true": (DARK_BLUE, "o", "codes as they eventually read"),
             "asof": (ORANGE, "s", "codes as they read at T")}

    for i, t in enumerate(truncs):
        sub = d[d["trunc"] == t]
        for arm, (colour, marker, label) in style.items():
            r = sub[sub["assignment"] == arm]
            if r.empty:
                continue
            x = i + (-0.11 if arm == "true" else 0.11)
            g, se = float(r["gamma2"].iloc[0]), float(r["se2"].iloc[0])
            ax.errorbar(x, g, yerr=1.96 * se, fmt=marker, color=colour,
                        ms=7, capsize=4, lw=1.6,
                        label=label if i == 0 else None)
        # the gap IS the artefact: name it on the figure
        tr = sub[sub["assignment"] == "true"]
        af = sub[sub["assignment"] == "asof"]
        if not tr.empty and not af.empty:
            a, b = float(tr["gamma2"].iloc[0]), float(af["gamma2"].iloc[0])
            ax.annotate("", xy=(i, b), xytext=(i, a),
                        arrowprops=dict(arrowstyle="<->", color=GRAY,
                                        lw=1.1))
            ax.text(i + 0.055, (a + b) / 2, f"artefact {b - a:+.3f}",
                    fontsize=8.5, color=DARK_TEXT, va="center")

    ax.axhline(0, color=DARK_TEXT, lw=0.8)
    ax.set_xticks(range(len(truncs)))
    ax.set_xticklabels([f"truncated at {t}" for t in truncs], fontsize=9.5)
    ax.set_xlim(-0.5, len(truncs) - 0.35)
    ax.set_ylabel("Age gradient, log points", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    # upper left: lower left collides with the 2021 as-of marker, which
    # sits near -0.29 and is the point the figure exists to show
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(0.005, 0.72))

    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(V2_FIG / f"figA1_asof_backtest{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"    saved figA1_asof_backtest.pdf/.png ({len(truncs)} truncations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

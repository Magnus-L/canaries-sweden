#!/usr/bin/env python3
"""
l36_fig_age_profile_v3.py: Figure 2 of the v3 paper, the age profile
inside exposed employers after adoption, on the occupation route.

WHAT IT DRAWS
Each age band against 41-49, from script 82's part B: six bands in one
panel of 153,845 employers, with employer-by-month, employer-by-age and
month-by-age effects, the calendar-quarter terms in, treatment from
January 2024, and exposure frozen at the employer's 2019 OCCUPATION mix.
Filled markers are the paper's score. Hollow markers are the same profile
with exposure routed through education instead, the specification v2
reported, which the export carries on every row so the two are drawn from
one file and cannot drift apart.

l23 draws the v2 figure, where the hollow series was the same score
without the calendar terms. That comparison asked whether the cycle drove
the profile and the appendix keeps it; this one asks whether the routing
did, which is the question a referee puts to v3.

The reference band is drawn at zero without an interval. Whiskers are 95
per cent intervals clustered by employer, and stars mark ten, five and one
per cent from the exported standard errors against the normal thresholds.
A negative value means the band declined more than 41-49.

INPUTS AND OUTPUTS
Reads occ_route_profile.csv from the lane 28b export (or a directory given
on the command line). Writes revision/figures/fig2_age_profile_v3.pdf and
.png through _figsafe.save and copies both to canaries-sweden-paper/figures/.

    python3 revision/local/l36_fig_age_profile_v3.py [export_dir]

IN THE PAPER
Figure 2 of main_v3.tex (label fig:age_profile), Section 3.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsafe import save  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAPER = REPO.parent / "canaries-sweden-paper"
DEFAULT = REPO / "revision/output/round3_20260923-0655-lanes28b-29bcd"

DARK = "#222222"
MID = "#666666"
ORDER = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
LABEL = {"22-25": "22–25", "26-30": "26–30", "31-34": "31–34",
         "35-40": "35–40", "41-49": "41–49\n(reference)", "50+": "50 and over"}
REF = "41-49"


def stars(coef, se):
    z = abs(coef) / se if se else 0.0
    return "***" if z >= 2.576 else "**" if z >= 1.96 else "*" if z >= 1.645 else ""


def main(export_dir: Path) -> int:
    d = pd.read_csv(export_dir / "occ_route_profile.csv").set_index("band")
    missing = [b for b in ORDER if b not in d.index]
    if missing:
        raise SystemExit(f"  the profile export is missing {missing}")
    n_firms = int(d["n_firms"].iloc[0])
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(ORDER)))
    series = (("edu", -0.14, dict(mfc="white", mec=MID, color=MID),
               "Exposure routed through education (the earlier score)"),
              ("occ", +0.14, dict(mfc=DARK, mec=DARK, color=DARK),
               "Exposure from the employer's 2019 occupation mix (the paper's score)"))
    for which, off, style, label in series:
        for i, b in enumerate(ORDER):
            if b == REF:
                continue
            r = d.loc[b]
            c = float(r["coef"] if which == "occ" else r["edu_coef"])
            se = float(r["se"] if which == "occ" else r["edu_se"])
            x = xs[i] + off
            lo, hi = c - 1.96 * se, c + 1.96 * se
            ax.plot([x, x], [lo, hi], color=style["color"], lw=1.2, zorder=2)
            ax.plot(x, c, "o", ms=6, mfc=style["mfc"], mec=style["mec"],
                    zorder=3, label=label if i == 0 else None)
            s = stars(c, se)
            if s:
                ax.text(x, hi + 0.003 if c >= 0 else lo - 0.003, s,
                        ha="center", va="bottom" if c >= 0 else "top",
                        fontsize=8, color=style["color"])
    ax.plot(xs[ORDER.index(REF)], 0, "s", ms=6, color=DARK, zorder=3)
    ax.axhline(0, color=DARK, lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[b] for b in ORDER], fontsize=9.5)
    ax.set_ylabel("Change after adoption relative to 41–49,\nlog points, exposed firms",
                  fontsize=9.5)
    ax.set_xlabel("Age band", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.02, f"{n_firms:,} employers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig2_age_profile_v3", __file__)
    for ext in (".pdf", ".png"):
        shutil.copy(REPO / "revision/figures" / f"fig2_age_profile_v3{ext}",
                    PAPER / "figures" / f"fig2_age_profile_v3{ext}")
    print("wrote fig2_age_profile_v3.pdf/.png to revision/figures and the "
          "paper repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

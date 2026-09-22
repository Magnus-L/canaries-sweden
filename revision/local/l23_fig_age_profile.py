#!/usr/bin/env python3
"""
l23_fig_age_profile.py -- Figure 2 of the revision: every age band against
41-49 inside exposed firms after adoption, with and without the calendar
cycle, built from the exported lane 20 contrast rather than by hand.

WHAT IT SHOWS. Six bands in one panel of 172,396 employers (script 74).
Filled markers are the specification the paper uses, with three
quarter-of-year interactions per band; hollow markers are the plain arm.
A negative number means the band declined more than 41-49 did. The
reference band is drawn at zero with no interval. Stars at the whisker
ends mark significance at ten, five and one per cent (one to three
stars), computed from the exported standard errors against the normal
thresholds; coefficient plots in this paper keep their stars.

    python3 revision/local/l23_fig_age_profile.py [export_dir]

Writes canaries-sweden-paper/figures/fig2_age_profile.pdf and .png.
"""
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
DEFAULT = REPO / "revision/output/round3_20260922-0105-lane20-seasonal-contrast"

DARK = "#222222"
MID = "#666666"
LIGHT = "#bbbbbb"
ORDER = ["22_25", "26_30", "31_34", "35_40", "41_49", "50p"]
LABEL = {"22_25": "22–25", "26_30": "26–30", "31_34": "31–34",
         "35_40": "35–40", "41_49": "41–49\n(reference)", "50p": "50 and over"}


def stars(coef, se):
    z = abs(coef) / se if se else 0.0
    return "***" if z >= 2.576 else "**" if z >= 1.96 else "*" if z >= 1.645 else ""


def main(export_dir: Path):
    d = pd.read_csv(export_dir / "contrast_seasonal.csv")
    n_firms = int(d["n_firms"].iloc[0])
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(ORDER)))
    off = {"plain": -0.14, "seasonal": +0.14}
    style = {"plain": dict(mfc="white", mec=MID, color=MID, label="Plain"),
             "seasonal": dict(mfc=DARK, mec=DARK, color=DARK,
                              label="Calendar cycle removed (the paper's specification)")}
    for arm in ("plain", "seasonal"):
        sub = d[d["arm"] == arm].set_index("band_vs_ref")
        for i, b in enumerate(ORDER):
            x = xs[i] + off[arm]
            if b == "41_49":
                continue
            r = sub.loc[b]
            lo, hi = r.coef - 1.96 * r.se, r.coef + 1.96 * r.se
            ax.plot([x, x], [lo, hi], color=style[arm]["color"], lw=1.2, zorder=2)
            ax.plot(x, r.coef, "o", ms=6, mfc=style[arm]["mfc"], mec=style[arm]["mec"],
                    zorder=3, label=style[arm]["label"] if i == 0 else None)
            s = stars(r.coef, r.se)
            if s:
                ax.text(x, hi + 0.003 if r.coef >= 0 else lo - 0.003, s,
                        ha="center", va="bottom" if r.coef >= 0 else "top",
                        fontsize=8, color=style[arm]["color"])
    ax.plot(xs[ORDER.index("41_49")], 0, "s", ms=6, color=DARK, zorder=3)
    ax.axhline(0, color=DARK, lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[b] for b in ORDER], fontsize=9.5)
    ax.set_ylabel("Change after adoption relative to 41–49,\nlog points, exposed firms", fontsize=9.5)
    ax.set_xlabel("Age band", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.99, 0.02, f"{n_firms:,} employers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    # The tree's guard writes revision/figures/ and refuses to clobber an
    # orphan; the paper repo gets a copy, as for fig2_spreading.
    save(fig, "fig2_age_profile", __file__)
    import shutil
    for ext in (".pdf", ".png"):
        shutil.copy(REPO / "revision/figures" / f"fig2_age_profile{ext}",
                    PAPER / "figures" / f"fig2_age_profile{ext}")
    print("wrote fig2_age_profile.pdf/.png to revision/figures and the paper repo")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)

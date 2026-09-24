#!/usr/bin/env python3
"""
02_figure2_age_profile.py: Figure 2 of the paper, the age profile inside
exposed employers after adoption.

Each age band against 41-49 on one panel of six bands, Poisson with
employer-by-month, employer-by-age and month-by-age effects, treatment from
January 2024, exposure frozen at the employer's 2019 occupation mix. Filled
markers carry three quarter-of-year terms per band (the paper's
specification); hollow markers omit them. Both arms come from one fit frame
(script 85). The reference band is drawn at zero; whiskers are 95 per cent
intervals clustered by employer; stars mark ten, five and one per cent on the
normal thresholds.

Export read: 3_register_mona/exports/2026-09-23_1001_s85/occ_route_profile_arms.csv
Output: output/figures/fig2_age_profile_v3.pdf and .png

    python 4_exhibits/02_figure2_age_profile.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS  # noqa: E402

DEFAULT = EXPORTS / "2026-09-23_1001_s85"

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
    raw = pd.read_csv(export_dir / "occ_route_profile_arms.csv")
    raw = raw[raw["status"].isin(["ok", "reference"])]
    arms = {a: raw[raw["arm"] == a].set_index("band") for a in ("plain",
                                                               "seasonal")}
    for a, d in arms.items():
        missing = [b for b in ORDER if b not in d.index]
        if missing:
            raise SystemExit(f"  the {a} arm is missing {missing}")
    n_firms = int(raw["n_firms"].iloc[0])
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(ORDER)))
    series = (("plain", -0.14, dict(mfc="white", mec=MID, color=MID),
               "Plain"),
              ("seasonal", +0.14, dict(mfc=DARK, mec=DARK, color=DARK),
               "Calendar cycle removed (the paper's specification)"))
    for which, off, style, label in series:
        d = arms[which]
        for i, b in enumerate(ORDER):
            if b == REF:
                continue
            r = d.loc[b]
            c, se = float(r["coef"]), float(r["se"])
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
    print("wrote fig2_age_profile_v3.pdf/.png to output/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

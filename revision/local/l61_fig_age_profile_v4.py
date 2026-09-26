#!/usr/bin/env python3
"""
l61_fig_age_profile_v4.py: Figure 2 of the v3 paper redrawn on tau, the
paper's estimand since the 25 September redesign of Table 1.

WHAT IT DRAWS
Each age band against 41-49 on tau = b_L - b_I (the later period, January
2024 to June 2025, against the interim period, December 2022 to December
2023), eight bands in one panel of 104,333 employers, with employer-by-
month, employer-by-age and month-by-age effects and the calendar-quarter
terms, exposure frozen at the employer's 2019 occupation mix. The oldest
band is split at 60 and 65, which is what lane 37c (script 95, Part P)
added: the pension-age reading can be checked by eye.

WHY A NEW FIGURE
l36 drew the step from January 2024 (six bands, 153,845 employers), on
which the young band's shortfall against the prime-aged was not
distinguishable from zero (-0.019, SE 0.012). On tau it is (-0.024, SE
0.011), and Table 1 reports tau, so the figure and the text now share one
estimand. The step profile stays in OA Table tab:profile_bands.

The reference band is drawn at zero without an interval. Whiskers are 95
per cent intervals clustered by employer, and stars mark ten, five and one
per cent from the exported standard errors against the normal thresholds.
A negative value means the band declined more than 41-49.

INPUTS AND OUTPUTS
Reads pension_reference.csv (part P, spec p8_tau, term tau) from the lane
37c export, or a directory given on the command line. Writes
revision/figures/fig2_age_profile_v4.pdf and .png through _figsafe.save
and copies both to canaries-sweden-paper/figures/.

    python3 revision/local/l61_fig_age_profile_v4.py [export_dir]

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
DEFAULT = REPO / "revision/output/round3_20260925-1832-lane37c-and-98"

DARK = "#222222"
MID = "#666666"
ORDER = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-59", "60-64",
         "65-69"]
LABEL = {"22-25": "22–25", "26-30": "26–30", "31-34": "31–34",
         "35-40": "35–40", "41-49": "41–49\n(reference)", "50-59": "50–59",
         "60-64": "60–64", "65-69": "65–69"}
REF = "41-49"
N_FIRMS_EXPECTED = 104_333


def stars(coef, se):
    z = abs(coef) / se if se else 0.0
    return "***" if z >= 2.576 else "**" if z >= 1.96 else "*" if z >= 1.645 else ""


def main(export_dir: Path) -> int:
    raw = pd.read_csv(export_dir / "pension_reference.csv")
    d = raw[(raw["part"] == "P") & (raw["spec"] == "p8_tau")
            & (raw["term"] == "tau") & (raw["status"] == "derived")]
    d = d.set_index("young_band")
    missing = [b for b in ORDER if b != REF and b not in d.index]
    if missing:
        raise SystemExit(f"  the tau profile is missing {missing}")
    n_firms = int(d["n_firms"].iloc[0])
    if n_firms != N_FIRMS_EXPECTED:
        raise SystemExit(f"  expected {N_FIRMS_EXPECTED:,} employers, the "
                         f"export says {n_firms:,}")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs = list(range(len(ORDER)))
    for i, b in enumerate(ORDER):
        if b == REF:
            continue
        r = d.loc[b]
        c, se = float(r["coef"]), float(r["se"])
        x = xs[i]
        lo, hi = c - 1.96 * se, c + 1.96 * se
        ax.plot([x, x], [lo, hi], color=DARK, lw=1.2, zorder=2)
        ax.plot(x, c, "o", ms=6, mfc=DARK, mec=DARK, zorder=3)
        s = stars(c, se)
        if s:
            ax.text(x, hi + 0.003 if c >= 0 else lo - 0.003, s,
                    ha="center", va="bottom" if c >= 0 else "top",
                    fontsize=8, color=DARK)
    ax.plot(xs[ORDER.index(REF)], 0, "s", ms=6, color=DARK, zorder=3)
    ax.axhline(0, color=DARK, lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[b] for b in ORDER], fontsize=9.5)
    ax.set_ylabel("Later period against interim, relative to 41–49,\n"
                  "log points, exposed firms", fontsize=9.5)
    ax.set_xlabel("Age band", fontsize=9.5)
    ax.text(0.99, 0.02, f"{n_firms:,} employers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig2_age_profile_v4", __file__)
    for ext in (".pdf", ".png"):
        shutil.copy(REPO / "revision/figures" / f"fig2_age_profile_v4{ext}",
                    PAPER / "figures" / f"fig2_age_profile_v4{ext}")
    print("wrote fig2_age_profile_v4.pdf/.png to revision/figures and the "
          "paper repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

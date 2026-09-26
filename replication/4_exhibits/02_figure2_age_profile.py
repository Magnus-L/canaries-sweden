#!/usr/bin/env python3
"""
02_figure2_age_profile.py: Figure 2 of the paper, exposure-related changes
in the age profile.

Each age band against 41-49 on tau = b_L - b_I, the later period (January
2024 to June 2025) against the interim period (December 2022 to December
2023), at top-quartile relative to less exposed employers, from one Poisson
fit on eight bands with band-specific period coefficients (script 95, part
P): employer-by-month, employer-by-age and month-by-age effects and the
calendar-quarter terms, exposure the employer's 2019 occupation mix, 104,333
employers. The oldest band is split at 60 and 65 so that the pension-age
reading can be checked by eye. The reference band is drawn at zero without
an interval; whiskers are 95 per cent intervals clustered by employer, and
stars mark ten, five and one per cent on the normal thresholds. A negative
value means the band declined more than 41-49.

Nothing is drawn unless every exported standard error is the square root of
the exported variance of the difference (V_LL + V_II - 2 V_LI), the panel
holds the 104,333 employers the caption states, and every coefficient and
standard error agrees, to four decimals, with the run's own summary.

Export read: 3_register_mona/exports/2026-09-25_1832_s95-s96-s98/
  pension_reference.csv (part P, spec p8_tau, term tau), 95_summary.txt
Output: output/figures/fig2_age_profile_v4.pdf and .png

    python 4_exhibits/02_figure2_age_profile.py [export_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS  # noqa: E402

DEFAULT = EXPORTS / "2026-09-25_1832_s95-s96-s98"

DARK = "#222222"
MID = "#666666"
ORDER = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-59", "60-64",
         "65-69"]
LABEL = {"22-25": "22–25", "26-30": "26–30", "31-34": "31–34",
         "35-40": "35–40", "41-49": "41–49\n(reference)", "50-59": "50–59",
         "60-64": "60–64", "65-69": "65–69"}
REF = "41-49"
N_FIRMS_EXPECTED = 104_333
# The summary prints each band as "  22-25  tau -0.0239 (0.0110)   ..."
SUMMARY = re.compile(r"^\s*(\d\d-\d\d)\s+tau ([-+][0-9.]+) \(([0-9.]+)\)")


def stars(coef, se):
    z = abs(coef) / se if se else 0.0
    return "***" if z >= 2.576 else "**" if z >= 1.96 else "*" if z >= 1.645 else ""


def summary_profile(path: Path) -> dict:
    """Part P of 95_summary.txt, the run's own print of the same fit."""
    said, in_p = {}, False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("P. THE PROFILE"):
            in_p = True
            continue
        if in_p and line.startswith("VERDICTS"):
            break
        m = SUMMARY.match(line) if in_p else None
        if m:
            said[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    if len(said) < len(ORDER) - 1:
        raise SystemExit(f"  95_summary.txt: part P parses {len(said)} bands, "
                         f"{len(ORDER) - 1} expected; nothing is drawn")
    return said


def main(export_dir: Path) -> int:
    src = export_dir / "pension_reference.csv"
    if not src.exists():
        raise SystemExit(f"  missing input: {src}")
    raw = pd.read_csv(src)
    d = raw[(raw["part"] == "P") & (raw["spec"] == "p8_tau")
            & (raw["term"] == "tau") & (raw["status"] == "derived")]
    d = d.set_index("young_band")
    missing = [b for b in ORDER if b != REF and b not in d.index]
    if missing:
        raise SystemExit(f"  the tau profile is missing {missing}")
    n_firms = set(int(x) for x in d["n_firms"])
    if n_firms != {N_FIRMS_EXPECTED}:
        raise SystemExit(f"  expected {N_FIRMS_EXPECTED:,} employers, the "
                         f"export says {n_firms}")
    said = summary_profile(export_dir / "95_summary.txt")
    for b in ORDER:
        if b == REF:
            continue
        r = d.loc[b]
        c, se = float(r["coef"]), float(r["se"])
        var = (float(r["var_post"]) + float(r["var_interim"])
               - 2.0 * float(r["cov_post_interim"]))
        if abs(np.sqrt(var) - se) > 1e-6:
            raise SystemExit(f"  {b}: the exported standard error {se:.6f} is "
                             f"not the square root of the exported variance "
                             f"of the difference {np.sqrt(var):.6f}")
        if round(c, 4) != said[b][0] or round(se, 4) != said[b][1]:
            raise SystemExit(f"  {b}: the export gives {c:+.4f} ({se:.4f}) and "
                             f"95_summary.txt {said[b][0]:+.4f} "
                             f"({said[b][1]:.4f}); nothing is drawn")
        print(f"  {b}: tau {c:+.4f} ({se:.4f}) {stars(c, se)}")
    print(f"  the summary reproduces every band; {N_FIRMS_EXPECTED:,} employers")

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
    ax.text(0.99, 0.02, f"{N_FIRMS_EXPECTED:,} employers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MID)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "fig2_age_profile_v4", __file__)
    print("  wrote output/figures/fig2_age_profile_v4.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

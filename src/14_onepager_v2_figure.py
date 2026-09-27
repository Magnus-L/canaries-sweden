#!/usr/bin/env python3
"""
14_onepager_v2_figure.py: the figure for the September 2026 one-pager.

Two panels, both on the paper's headline contrast (the later period, January
2024 to June 2025, against the interim period, December 2022 to December
2023, at top-quartile against less exposed employers), shown in per cent,
100 * (exp(tau) - 1), so a lay reader does not meet log points:

  left   the age profile, every band against ages 41-49 (paper Figure 2),
         read from the same export and the same rows the paper's builder
         uses (replication/4_exhibits/02_figure2_age_profile.py);
  right  young women and young men at 22-25, each against older colleagues
         of the same sex at the same employers (OA Table A25, Panel C, and
         OA Section III.2): tau for each sex is the later-period step minus
         the interim step, derived here from the table's printed rows and
         checked against the appendix text (-0.077 and -0.006).

Why per cent and not log points: the one-pager is read by non-economists;
for changes this small the two differ by a few hundredths of a point.

Output: figures/onepager_v2_age_sex.pdf and .png (this repo), copied next to
the one-pager by the build.

    python src/14_onepager_v2_figure.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "replication/3_register_mona/exports/2026-09-25_1832_s95-s96-s98"
PAPER = ROOT.parent / "canaries-sweden-paper"
TABLE = PAPER / "tables/tableA_cluster_industry.tex"
OUT = ROOT / "figures"

ORDER = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-59", "60-64", "65-69"]
REF = "41-49"
NAVY, ORANGE, TEAL, GREY = "#1B3A5C", "#E8873A", "#2E7D6F", "#9A9A9A"


def pct(x):
    """Log change to per cent: 100 * (exp(x) - 1)."""
    return 100 * (np.exp(x) - 1)


def age_profile() -> pd.DataFrame:
    raw = pd.read_csv(EXPORT / "pension_reference.csv")
    d = raw[(raw["part"] == "P") & (raw["spec"] == "p8_tau")
            & (raw["term"] == "tau") & (raw["status"] == "derived")]
    d = d.set_index("young_band")[["coef", "se"]]
    d.loc[REF] = [0.0, np.nan]
    return d.loc[ORDER]


def sex_split() -> dict:
    """tau by sex from the printed rows of OA Table A25, Panel C."""
    txt = TABLE.read_text()
    def row(label):
        m = re.search(re.escape(label) + r".*?& \$([-+][0-9.]+)\$", txt)
        if not m:
            raise SystemExit(f"row not found: {label}")
        return float(m.group(1))
    men_int = row(r"Young men, interim through 2023")
    men_late = row(r"Young men, later period from 2024")
    fem_int = row(r"Female differential, interim through 2023")
    women_late = row(r"Young women, later period from 2024")
    men = men_late - men_int
    women = women_late - (men_int + fem_int)
    # the appendix prints these, with their SEs, from the covariance of the fit
    assert round(men, 3) == -0.006 and round(women, 3) == -0.077, (men, women)
    return {"Young men": (men, 0.012), "Young women": (women, 0.011)}


def main() -> None:
    prof = age_profile()
    sexes = sex_split()
    for b, r in prof.iterrows():
        print(f"  {b}: {r.coef:+.4f} ({r.se:.4f}) -> {pct(r.coef):+.1f}%")
    for k, (c, s) in sexes.items():
        print(f"  {k}: {c:+.4f} ({s:.3f}) -> {pct(c):+.1f}%")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9),
                                 gridspec_kw={"width_ratios": [2.6, 1]})

    # left: age profile, bars in per cent with 95 per cent whiskers
    xs = np.arange(len(ORDER))
    for i, b in enumerate(ORDER):
        c, s = prof.loc[b, "coef"], prof.loc[b, "se"]
        if b == REF:
            a1.plot(i, 0, "s", color=GREY, ms=5)
            continue
        col = ORANGE if c < 0 else TEAL
        a1.bar(i, pct(c), color=col, width=0.62, zorder=2)
        a1.plot([i, i], [pct(c - 1.96 * s), pct(c + 1.96 * s)],
                color=NAVY, lw=1, zorder=3)
    a1.axhline(0, color=NAVY, lw=0.8)
    a1.set_xticks(xs)
    a1.set_xticklabels([b.replace("-", "–") for b in ORDER], fontsize=8)
    a1.set_xlabel("Age band (41–49 = reference)", fontsize=8.5)
    a1.set_ylabel("Change at exposed employers, %", fontsize=8.5)
    a1.set_title("Young lose ground, older gain", fontsize=9.5,
                 color=NAVY, loc="left", fontweight="bold")

    # right: the sexes at 22-25
    for i, (k, (c, s)) in enumerate(sexes.items()):
        col = ORANGE if k == "Young women" else GREY
        a2.bar(i, pct(c), color=col, width=0.55, zorder=2)
        a2.plot([i, i], [pct(c - 1.96 * s), pct(c + 1.96 * s)],
                color=NAVY, lw=1, zorder=3)
        a2.text(i + 0.36, pct(c) / 2, f"{pct(c):+.1f}%".replace("-", "−"),
                ha="left", va="center", fontsize=9, color=NAVY, fontweight="bold")
    a2.axhline(0, color=NAVY, lw=0.8)
    a2.set_xticks([0, 1])
    a2.set_xlim(-0.5, 1.95)
    a2.set_xticklabels(["Men\n22–25", "Women\n22–25"], fontsize=8)
    a2.set_ylim(-12, 3)
    a2.set_title("Falls on young women", fontsize=9.5, color=NAVY,
                 loc="left", fontweight="bold")
    for ax in (a1, a2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.yaxis.grid(True, color="#E4E4E4", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.2)
    OUT.mkdir(exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(OUT / f"onepager_v2_age_sex{ext}", bbox_inches="tight", **kw)
    print("  wrote figures/onepager_v2_age_sex.pdf and .png")


if __name__ == "__main__":
    main()

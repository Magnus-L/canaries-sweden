#!/usr/bin/env python3
"""
03_figure3_quarterly_path.py: Figure 3 of the paper, the quarterly path of
young employment relative to older colleagues inside exposed employers, and
with --monthly the monthly diagnostic of Online Appendix III.2.

Each point is the coefficient on one quarter's interaction with High x Young
in a fit that keeps the Riksbank term and three quarter-of-year terms, so it
is a step from the level of the tightening months with the calendar cycle
removed; 95 per cent intervals clustered by employer (script 84). The 2022Q4
point is December 2022 alone and is drawn hollow. The lower panel is
Statistics Sweden's published share of enterprises with ten or more employees
using AI (10 per cent in 2023, 25 in 2024, 35 in 2025), typed in below from the
published statistic. A quarter with no estimate is drawn as a gap. The
quarter axis is the union of the quarters in this export and in script 68's
path export, which carry the same quarters.

Exports read (3_register_mona/exports/):
  2026-09-23_0917_s84/occ_route_path.csv
  2026-09-21_2152_s68/seasonal_path.csv (the quarter axis only)
Output: output/figures/fig2_spreading_v3.pdf and .png;
        with --monthly, fig2_spreading_monthly_v3.pdf and .png

    python 4_exhibits/03_figure3_quarterly_path.py [export_dir]
    python 4_exhibits/03_figure3_quarterly_path.py --monthly [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save  # noqa: E402
from config import EXPORTS, DARK_BLUE, ORANGE, GRAY, LIGHT_GRAY, DARK_TEXT  # noqa: E402

# Statistics Sweden's firm AI-use series, published and not estimated
# here. The paper carries no adoption figure later than 2024, because the
# dating of the window to January 2024 rests on the 2023-to-2024 change.
SCB_ADOPTION = {"2023": 10, "2024": 25, "2025": 35}   # scb2026aiuse: 2025 = 35
LAUNCH_Q = "2022Q4"
# Script 84's export.
S84 = EXPORTS / "2026-09-23_0917_s84"
# Script 68's path export, read only for its quarter axis.
S68 = EXPORTS / "2026-09-21_2152_s68"
BANDS = {"22-25": (ORANGE, "o"), "26-30": (DARK_BLUE, "s")}


def find(argv) -> Path | None:
    d = Path(argv[1]) if len(argv) > 1 else S84
    p = d / "occ_route_path.csv"
    return p if p.exists() else None


def edu_path() -> pd.DataFrame:
    p = S68 / "seasonal_path.csv"
    if not p.exists():
        print("  no script 68 path found; drawing this one alone")
        return pd.DataFrame(columns=["young_band", "shape", "period",
                                     "coef", "se", "status"])
    return pd.read_csv(p)


def ok(d: pd.DataFrame, shape: str) -> pd.DataFrame:
    return d[(d["shape"] == shape)
             & (d.get("status", "ok") == "ok")].copy()


def gaps(b: pd.DataFrame, order: list, band: str) -> pd.DataFrame:
    """The band's rows on the full period axis, NaN where it has none, so
    a failed fit breaks the line instead of being drawn through."""
    missing = [p for p in order if p not in set(b["period"])]
    if missing:
        print(f"    WARNING: {band} has no estimate for "
              f"{', '.join(missing)}; the line is broken there")
        b = pd.concat([b, pd.DataFrame({"period": missing})],
                      ignore_index=True)
    return b.sort_values("period")


def quarterly(d: pd.DataFrame, edu: pd.DataFrame) -> int:
    q, qe = ok(d, "quarter"), ok(edu, "quarter")
    if q.empty:
        print("  occ_route_path.csv has no quarterly rows")
        return 1
    order = sorted(set(q["period"]) | set(qe["period"]))
    xof = {p: i for i, p in enumerate(order)}

    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.12})

    for band, (colour, marker) in BANDS.items():
        b = gaps(q[q["young_band"] == band], order, band)
        b = b.assign(x=b["period"].map(xof))
        ax.fill_between(b["x"], b["coef"] - 1.96 * b["se"],
                        b["coef"] + 1.96 * b["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(b["x"], b["coef"], "-", color=colour, lw=1.9, zorder=3,
                label=band)
        post = b[b["period"] != LAUNCH_Q]
        pre = b[b["period"] == LAUNCH_Q]
        ax.plot(post["x"], post["coef"], marker, color=colour, ms=5,
                zorder=4)
        ax.plot(pre["x"], pre["coef"], marker, mfc="white", mec=colour,
                mew=1.4, ms=5, zorder=4)

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    if LAUNCH_Q in xof:
        xl = xof[LAUNCH_Q]
        for a in (ax, axb):
            a.axvline(xl, color=GRAY, ls="--", lw=0.9, zorder=1)
        ax.annotate("ChatGPT", xy=(xl, ax.get_ylim()[1]),
                    xytext=(xl + 0.15, ax.get_ylim()[1]),
                    fontsize=8, color=GRAY, va="top")
    ax.set_ylabel("Employment, log points\ncycle removed", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(axis="y", labelsize=8.5)

    for yr, pct in SCB_ADOPTION.items():
        xs = [i for i, p in enumerate(order) if p.startswith(yr)]
        if xs:
            axb.bar(xs, [pct] * len(xs), width=0.94, color=LIGHT_GRAY,
                    align="center")
            axb.text(sum(xs) / len(xs), pct + 3, f"{pct:.0f}%",
                     ha="center", fontsize=8, color=GRAY)
    axb.set_ylim(0, 52)
    axb.set_ylabel("SCB: firms\nusing AI", fontsize=8.5, color=GRAY)
    axb.tick_params(axis="y", labelsize=8, colors=GRAY)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels(order, rotation=45, ha="right", fontsize=8.5)

    save(fig, "fig2_spreading_v3", __file__)
    plt.close(fig)
    print(f"    saved fig2_spreading_v3.pdf/.png ({len(order)} quarters, "
          f"bands {sorted(q['young_band'].unique())})")
    return 0


def monthly(d: pd.DataFrame) -> int:
    m = ok(d, "month")
    if m.empty:
        print("  occ_route_path.csv has no monthly rows")
        return 1

    def to_date(p):
        return pd.Timestamp(int(str(p)[:4]), int(str(p)[5:7]), 15)

    m["t"] = m["period"].map(to_date)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for band, (colour, marker) in BANDS.items():
        b = m[m["young_band"] == band].sort_values("t")
        if b.empty:
            print(f"    WARNING: no monthly path for {band}")
            continue
        ax.fill_between(b["t"], b["coef"] - 1.96 * b["se"],
                        b["coef"] + 1.96 * b["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(b["t"], b["coef"], "-", color=colour, lw=1.5, marker=marker,
                ms=3.4, label=f"{band}, monthly", zorder=3)
    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    ax.axvline(pd.Timestamp(2022, 11, 30), color=GRAY, ls="--", lw=0.9)
    ax.text(pd.Timestamp(2022, 12, 5), ax.get_ylim()[1], " ChatGPT",
            fontsize=8, color=GRAY, va="top")
    ax.set_ylabel("Employment, log points, cycle removed", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.tick_params(labelsize=8.5)
    fig.autofmt_xdate(rotation=45, ha="right")
    save(fig, "fig2_spreading_monthly_v3", __file__)
    plt.close(fig)
    print("    saved fig2_spreading_monthly_v3.pdf/.png")
    return 0


def main() -> int:
    is_monthly = "--monthly" in sys.argv
    if is_monthly:
        sys.argv.remove("--monthly")
    src = find(sys.argv)
    if src is None:
        print("  no occ_route_path.csv found. Give the script 84 export "
              "directory on the command line.")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src)
    rc = monthly(d) if is_monthly else quarterly(d, edu_path())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l39_fig_spreading_v3.py: Figure 3 of the v3 paper, the quarterly path of
young employment relative to older colleagues, on the occupation route.

WHAT IT DRAWS
The quarterly path for 22-25 and for 26-30 on one pair of axes, from
script 84's export: each point is the coefficient on that quarter's
interaction with High x Young, a step from the level of the tightening
months (the Riksbank term stays in the fit), with the calendar cycle
removed by the quarter-of-year terms, and 95 per cent intervals clustered
by employer. The point for 2022Q4 is December 2022 alone, the first full
month after the launch, and is drawn hollow. A lower panel gives
Statistics Sweden's published share of enterprises with ten or more
employees using AI, which is the series the timing is read against.

WHAT IS NEW AGAINST l14. Exposure is the employer's 2019 OCCUPATION mix,
not its education mix, so this is the path that belongs beside Table 1 of
v3. The education route's own path is drawn behind it as a thin dashed
line, one per band, without an interval: the comparison a referee wants
is whether the dating moves with the routing, and two shaded bands per
colour would make the chart unreadable to answer it.

A MISSING QUARTER IS DRAWN AS A GAP, NOT INTERPOLATED. A fit that failed
leaves no row in the export, and a line drawn straight across the hole
would assert a value nothing estimated. Any quarter present for one band
and absent for another is filled with NaN, so the line breaks where the
evidence does, and the script says which quarters it broke for.

    python3 revision/local/l39_fig_spreading_v3.py [export_dir]
    python3 revision/local/l39_fig_spreading_v3.py --monthly [export_dir]

INPUTS AND OUTPUTS
Reads occ_route_path.csv from the lane 30 export directory (or one given
on the command line) and seasonal_path.csv from the lane 14 export the
final-code manifest names, for the education-route comparison. Writes
revision/figures/fig2_spreading_v3.pdf and .png, and with --monthly
fig2_spreading_monthly_v3.pdf and .png, through _figsafe.save, and copies
both to canaries-sweden-paper/figures/.

IN THE PAPER
Figure 3 of main_v3.tex (label fig:age_gradient), Section 3, and the
monthly diagnostic of Online Appendix III.2.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figsafe import save  # noqa: E402
from config import DARK_BLUE, ORANGE, GRAY, LIGHT_GRAY, DARK_TEXT  # noqa: E402

PAPER = REV.parents[1] / "canaries-sweden-paper"
# Statistics Sweden's firm AI-use series, published and not estimated
# here, as in l14.
SCB_ADOPTION = {"2023": 10, "2024": 25, "2025": 35}
LAUNCH_Q = "2022Q4"
# Lane 30's export. The folder is named on the day it is filed; pass it
# on the command line until this constant is updated to match.
LANE30 = REV / "output" / "round3_20260923-lane30-path"
# The education route's own path, the script 68 export the final-code
# manifest names.
LANE14 = REV / "output" / "round3_20260921-2152-lane14-seasonal-complete"
BANDS = {"22-25": (ORANGE, "o"), "26-30": (DARK_BLUE, "s")}


def find(argv) -> Path | None:
    d = Path(argv[1]) if len(argv) > 1 else LANE30
    p = d / "occ_route_path.csv"
    return p if p.exists() else None


def edu_path() -> pd.DataFrame:
    p = LANE14 / "seasonal_path.csv"
    if not p.exists():
        print("  no education-route path found; drawing ours alone")
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
        e = qe[qe["young_band"] == band].sort_values("period")
        if not e.empty:
            e = e.assign(x=e["period"].map(xof))
            ax.plot(e["x"], e["coef"], "--", color=colour, lw=1.0,
                    alpha=0.55, zorder=2,
                    label=f"{band}, education route")
        b = gaps(q[q["young_band"] == band], order, band)
        b = b.assign(x=b["period"].map(xof))
        ax.fill_between(b["x"], b["coef"] - 1.96 * b["se"],
                        b["coef"] + 1.96 * b["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(b["x"], b["coef"], "-", color=colour, lw=1.9, zorder=3,
                label=f"{band}, occupation route")
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
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
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
        print("  occ_route_path.csv has no monthly rows; lane 30b is the "
              "job that produces them")
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
        print("  no occ_route_path.csv found. Give the lane 30 export "
              "directory on the command line, or file it as "
              f"{LANE30.name} and run again.")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src)
    rc = monthly(d) if is_monthly else quarterly(d, edu_path())
    if rc == 0:
        stem = ("fig2_spreading_monthly_v3" if is_monthly
                else "fig2_spreading_v3")
        for ext in (".pdf", ".png"):
            f = REV / "figures" / f"{stem}{ext}"
            if f.exists() and (PAPER / "figures").exists():
                shutil.copy(f, PAPER / "figures" / f.name)
        print(f"    copied {stem} to the paper repository")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

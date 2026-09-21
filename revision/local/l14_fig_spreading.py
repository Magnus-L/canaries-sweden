#!/usr/bin/env python3
"""
l14_fig_spreading.py -- the revision's headline exhibit, built from the
exported lane 14 path rather than by hand.

WHY THIS EXISTS. `figures/fig2_quarterly_path.pdf` was produced ad hoc on
21 September: no PNG twin, no script in the tree, no way to tell which
export it came from. That is the wrong footing for the figure the paper
leads on, and the change list has since respecified it anyway, as two
series rather than one.

WHAT IT SHOWS. The quarterly path for 22-25 and for 26-30 on the same
axes, with SCB's measured firm AI adoption behind them. The claim is the
second line peeling away from zero about a year after the first: the
effect reaches the youngest band first and the next one later. Both
series have the calendar cycle removed, which matters -- the 22-25 and
26-30 paths must come from the SAME specification or the comparison is
between a seasonally adjusted series and a raw one.

READ RULES CARRIED INTO THE FIGURE.
  * 2022Q4 straddles the ChatGPT launch and is neither pre nor post. It
    is drawn hollow and excluded from any statement about the pre-period.
  * 2025 is a half year of preliminary AGI data, so the last points rest
    on less than the rest.
  * There is no yearly series here on purpose. `path/year/22-25` crashed
    and the figures it would have produced were never estimated.

    python3 revision/local/l14_fig_spreading.py [export_dir]
    python3 revision/local/l14_fig_spreading.py --monthly [export_dir]

THE MONTHLY VARIANT exists to let the 2025 endpoint be judged rather
than taken on trust. The quarterly figure shows 22-25 at -0.0629 in
2024Q4, -0.0031 in 2025Q1 and -0.0518 in 2025Q2, which reads as one
inexplicable point. Monthly shows what is behind it: February 2025 at
+0.0208 and June at -0.1003, a range of 0.121 log points across the six
preliminary months against 0.074 across all of complete 2024. Only
22-25 has a monthly path -- 26-30 was estimated at quarterly and yearly
frequency only -- so the monthly chart mixes frequencies and says so.

2025 IS NOT PRELIMINARY. SCB confirmed to ML on 21 September 2026 that
the AGI monthly figures are not revised after delivery. Earlier notes
in this project hedged the 2025 endpoint as preliminary; that hedge is
withdrawn. The endpoint still rests on six months rather than twelve,
which is a different and much weaker caveat.
"""
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
from config import V2_FIG, DARK_BLUE, ORANGE, GRAY, LIGHT_GRAY, DARK_TEXT

# SCB's firm AI-use series, the thing the timing is read against.
SCB_ADOPTION = {"2023": 10.4, "2024": 25.2, "2025": 35.0}
LAUNCH_Q = "2022Q4"


def find_path_csv(argv) -> Path | None:
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if not r.exists():
            continue
        for p in list(r.rglob("seasonal_path.csv")) + list(
                r.rglob("*__seasonal_path.csv")):
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def _to_date(period: str):
    """'2025-01' and '2025Q1' onto one axis; a quarter sits at its middle."""
    p = str(period)
    if "Q" in p:
        y, qq = p.split("Q")
        return pd.Timestamp(int(y), 3 * int(qq) - 1, 15)
    return pd.Timestamp(int(p[:4]), int(p[5:7]), 15)


def monthly_figure(d: pd.DataFrame) -> int:
    m = d[(d["shape"] == "month") & (d.get("status", "ok") == "ok")].copy()
    q = d[(d["shape"] == "quarter") & (d.get("status", "ok") == "ok")].copy()
    if m.empty:
        print("  no monthly rows in this export")
        return 1
    for f in (m, q):
        f["t"] = f["period"].map(_to_date)

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    mb = m[m["young_band"] == "22-25"].sort_values("t")
    ax.fill_between(mb["t"], mb["coef"] - 1.96 * mb["se"],
                    mb["coef"] + 1.96 * mb["se"], alpha=0.13,
                    color=ORANGE, lw=0, zorder=2)
    ax.plot(mb["t"], mb["coef"], "-o", color=ORANGE, lw=1.5, ms=3.6,
            label="22-25, monthly", zorder=3)

    qb = q[q["young_band"] == "26-30"].sort_values("t")
    if not qb.empty:
        ax.plot(qb["t"], qb["coef"], "--s", color=DARK_BLUE, lw=1.6,
                ms=4.5, label="26-30, quarterly (no monthly path)",
                zorder=3)

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    ax.axvline(pd.Timestamp(2022, 11, 30), color=GRAY, ls="--", lw=0.9)
    ax.text(pd.Timestamp(2022, 12, 5), ax.get_ylim()[1], " ChatGPT",
            fontsize=8, color=GRAY, va="top")

    # NO SHADING ON 2025. SCB confirmed to ML that the AGI months are not
    # revised after delivery, so 2025 is definitive and not preliminary.
    # What remains true is only that it is half a year, which is a
    # statement about how many months the endpoint rests on, not about
    # whether those months will change. That belongs in the caption, not
    # in a grey box that reads as a health warning.

    ax.set_ylabel("Employment, log points, cycle removed", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.tick_params(labelsize=8.5)
    fig.autofmt_xdate(rotation=45, ha="right")
    save(fig, "fig2_spreading_monthly", __file__)
    plt.close(fig)
    print(f"    saved fig2_spreading_monthly.pdf/.png "
          f"({len(mb)} months 22-25, {len(qb)} quarters 26-30)")
    return 0


def main() -> int:
    monthly = "--monthly" in sys.argv
    if monthly:
        sys.argv.remove("--monthly")
    src = find_path_csv(sys.argv)
    if src is None:
        print("  no seasonal_path.csv found; run lane 14 and export it")
        return 1
    print(f"  reading {src}")
    d = pd.read_csv(src)
    if monthly:
        return monthly_figure(d)
    q = d[(d["shape"] == "quarter") & (d.get("status", "ok") == "ok")].copy()
    if q.empty:
        print("  seasonal_path.csv has no quarterly rows")
        return 1

    order = sorted(q["period"].unique())
    q["x"] = q["period"].map({p: i for i, p in enumerate(order)})

    # SCB adoption goes in its OWN panel. Drawn behind the estimates on a
    # twin axis it read as part of the result: grey bars rising from the
    # bottom of a chart whose left axis is negative look like a series.
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(7.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.12})

    styles = {"22-25": (ORANGE, "o", "22-25"),
              "26-30": (DARK_BLUE, "s", "26-30")}
    for band, (colour, marker, label) in styles.items():
        b = q[q["young_band"] == band].sort_values("x")
        if b.empty:
            continue
        ax.fill_between(b["x"], b["coef"] - 1.96 * b["se"],
                        b["coef"] + 1.96 * b["se"], alpha=0.13,
                        color=colour, lw=0, zorder=2)
        ax.plot(b["x"], b["coef"], "-", color=colour, lw=1.9,
                label=label, zorder=3)
        post = b[b["period"] != LAUNCH_Q]
        pre = b[b["period"] == LAUNCH_Q]
        ax.plot(post["x"], post["coef"], marker, color=colour, ms=5,
                zorder=4)
        ax.plot(pre["x"], pre["coef"], marker, mfc="white", mec=colour,
                mew=1.4, ms=5, zorder=4)

    ax.axhline(0, color=DARK_TEXT, lw=0.8, zorder=1)
    if LAUNCH_Q in order:
        xl = order.index(LAUNCH_Q)
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

    save(fig, "fig2_spreading", __file__)
    plt.close(fig)
    print(f"    saved fig2_spreading.pdf/.png "
          f"({len(order)} quarters, bands "
          f"{sorted(q['young_band'].unique())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

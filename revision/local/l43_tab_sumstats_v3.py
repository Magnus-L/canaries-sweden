#!/usr/bin/env python3
"""
l43_tab_sumstats_v3.py: Online Appendix Table I.2, the estimation
sample, on the occupation route.

WHY A NEW SCRIPT. l22 describes the EDUCATION route's sample: it reads
the estimated cell counts from script 68's seasonal_pooled.csv, it prints
311,227 firms carrying an exposure score, and its panel counts come from
script 73. The v3 paper estimates on the occupation route, where 262,089
employers are scored and the panels hold 104,217 and 117,090, which is
what Table 1 and the size-reliability table already print. So the
appendix's own account of the estimation sample described a sample no
estimate in the paper uses. l22 stays as it is so the v2 table remains
reproducible.

WHAT IS ROUTE-DEPENDENT AND WHAT IS NOT
The balanced cells, the cells that are zero in every month and the
skeleton are counted BEFORE the exposure merge, so they belong to the
data and not to a route; they are read from script 68's log exactly as
l22 reads them. What the route changes is the last column, the cells
surviving the merge with the 2019 score, and the three employer counts
beneath the table.

THE TWO ROWS THAT CANNOT BE FILLED. The paper reports flows at 22--25
only, and lane 29 fitted them there only, so 26--30 hires and separations
have no estimate on this route. Their skeleton is still reported, because
it describes the data, and the last column prints an em dash rather than
a number from another route. Nothing is carried across.

    python3 revision/local/l43_tab_sumstats_v3.py

INPUTS AND OUTPUTS
Reads 68_log.txt (script 68) for the skeleton columns;
occ_route_headline.csv and occ_route_flows.csv (lane 28b/29) for the
estimated cells and the panel employer counts; occ_route_coverage.csv
(lane 28a) for the number of employers carrying a score. Writes
revision/tables/tableI2_sumstats_employment.tex and copies it to
canaries-sweden-paper/tables/.

IN THE PAPER
Online Appendix I.2, Table tab:sumstats_employment_new.
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LOG = OUT / "round3_20260921-lane14-seasonal/68_log.txt"
LANE28A = OUT / "round3_20260922-2232-lane28a"
LANE28B = OUT / "round3_20260923-0655-lanes28b-29bcd"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

ROWS = [("22-25", "stock"), ("22-25", "hires"), ("22-25", "seps"),
        ("26-30", "stock"), ("26-30", "hires"), ("26-30", "seps")]
TERM = "post_x_high_x_young"
# The reported arm, as in script 82: the uniform three-digit score, the
# backward cascade, a floor of five incumbent person-months.
ARM, FLOOR, LEVEL = "backward", 5, "uniform3"
DASH = "---"


def parse_log() -> dict:
    """Pair each 'dropped X of Y' with the skeleton line that follows it.
    These three columns are counted before any exposure merge, so they are
    the same on either route; this is l22's own parser, unchanged."""
    t = LOG.read_text(errors="replace").split("\n")
    out, pending = {}, None
    for ln in t:
        m = re.search(r"dropped ([\d,]+) of ([\d,]+) rows in cells that are "
                      r"zero in every month \((\d+)%\)", ln)
        if m:
            pending = (int(m.group(1).replace(",", "")),
                       int(m.group(2).replace(",", "")), int(m.group(3)))
            continue
        m = re.search(r"([\d-]+) (stock|hires|seps): skeleton ([\d,]+) rows",
                      ln)
        if m and pending:
            key = (m.group(1), m.group(2))
            skel = int(m.group(3).replace(",", ""))
            if pending[1] - pending[0] == skel:
                out.setdefault(key, (pending[1], pending[0], pending[2], skel))
            pending = None
    return out


def fitted() -> dict:
    """
    (band, outcome) -> (estimated cells, employers) on the REPORTED arm.

    The headline export carries every arm side by side: the floor
    variants, the two scoring levels beside the reported one and the
    forward cascade. Taking rows without naming the arm silently picks
    whichever sorts last, which here was the forward arm at 22--25 and
    four_only at 26--30, neither of which the paper reports. So the arm
    is named and exactly one row must survive per band.
    """
    out = {}
    h = pd.read_csv(LANE28B / "occ_route_headline.csv")
    h = h[(h.term == TERM) & (h.outcome == "stock")
          & (h.arm == ARM) & (h.floor == FLOOR) & (h.level == LEVEL)]
    for band in ("22-25", "26-30"):
        r = h[h.young_band == band]
        if len(r) != 1:
            raise SystemExit(f"  occ_route_headline.csv: {len(r)} rows for "
                             f"{band} on {ARM}/{FLOOR}/{LEVEL}, expected one")
        out[(band, "stock")] = (int(r.iloc[0].n_obs), int(r.iloc[0].n_firms))
    f = pd.read_csv(LANE28B / "occ_route_flows.csv")
    f = f[f.term == TERM]
    for _, r in f.iterrows():
        k = (str(r.young_band), str(r.outcome))
        if k in out:
            raise SystemExit(f"  occ_route_flows.csv: {k} appears twice")
        out[k] = (int(r.n_obs), int(r.n_firms))
    return out


def scored() -> int:
    d = pd.read_csv(LANE28A / "occ_route_coverage.csv")
    r = d[(d.block == "route") & (d.group == "occupation_route")
          & (d.item == "n_employers")]
    if len(r) != 1:
        raise SystemExit("  occ_route_coverage.csv: the occupation route's "
                         "scored employer count is not reported exactly once")
    return int(float(r.iloc[0]["n_employers"]))


def tex_thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main() -> int:
    log, fit = parse_log(), fitted()
    n_scored = scored()
    missing_fit = []

    L = [r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
         r"\caption{The estimation sample: employer $\times$ age band "
         r"$\times$ month, 2021:01--2025:06.}",
         r"\label{tab:sumstats_employment_new}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Band & Outcome & Balanced cells & Zero throughout & Skeleton & "
         r"Estimated \\", r"\midrule"]
    for band, oc in ROWS:
        k = (band, oc)
        if k not in log:
            raise SystemExit(f"  the log does not report {band} {oc}")
        bal, drop, pct, skel = log[k]
        if k in fit:
            est, _ = fit[k]
            est_s = f"{est:,}"
        else:
            missing_fit.append(f"{band} {oc}")
            est_s = DASH
        L.append(f"{band} & {oc} & {bal:,} & {drop:,} ({pct}\\%) & "
                 f"{skel:,} & {est_s} \\\\")
        print(f"  {band:6s} {oc:6s} balanced {bal:>12,}  zero-throughout "
              f"{drop:>11,} ({pct:2d}%)  skeleton {skel:>12,}  "
              f"estimated {est_s:>12}")

    L += [r"\midrule",
          r"\multicolumn{2}{l}{Firms carrying an exposure score} & "
          rf"\multicolumn{{4}}{{r}}{{{tex_thousands(n_scored)}}} \\"]
    for b in ("22-25", "26-30"):
        if (b, "stock") not in fit:
            raise SystemExit(f"  no stock fit for {b}, so its panel count "
                             f"cannot be reported")
        n = fit[(b, "stock")][1]
        L.append(f"\\multicolumn{{2}}{{l}}{{Firms in the {b} panel}} & "
                 f"\\multicolumn{{4}}{{r}}{{{n:,}}} \\\\")
        print(f"  panel firms {b}: {n:,}")
    print(f"  employers carrying a score: {n_scored:,}")

    note = (r"The panel is balanced over employers, age bands and months and "
            r"zero-filled, so \textbf{a firm whose young headcount falls to "
            r"zero contributes those months}: the extensive margin is in the "
            r"estimate, which is why Poisson rather than a log transform. "
            r"``Zero throughout'' counts employer-band cells with no "
            r"employment in any month of the window; these are perfectly "
            r"predicted by the employer-by-age effect and fixest separates "
            r"them regardless, so they are removed before estimation at no "
            r"cost to any coefficient. The first three columns are counted "
            r"before the exposure merge and so are the same whatever the "
            r"score. ``Skeleton'' is what remains, and ``Estimated'' the "
            r"cells surviving the merge with the employer's 2019 occupation "
            r"mix. A firm enters only if it holds the young band and at "
            r"least one older band at some point from January 2021, which is "
            r"a pre-treatment property and not an outcome. Exposure is "
            rf"scored for {tex_thousands(n_scored)} firms.")
    if missing_fit:
        note += (r" The paper reports the flow margins at 22--25 only, so "
                 r"the two 26--30 flow rows carry a skeleton and no "
                 r"estimate; nothing is carried across from another score.")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.95\textwidth}\footnotesize\vspace{4pt}" + note,
          r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableI2_sumstats_employment.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(REV.parent)}")
    if PAPER_TAB.exists():
        shutil.copy2(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    if missing_fit:
        print(f"  no estimate on this route for: {', '.join(missing_fit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

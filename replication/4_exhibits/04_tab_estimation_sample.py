#!/usr/bin/env python3
"""
04_tab_estimation_sample.py: Online Appendix Table A2 (Section I.2), the
employment estimation sample.

Panel A describes the employer by age band by month panel, January 2021 to
June 2025: balanced cells, cells zero in every month (removed before
estimation, since the employer-by-age effect predicts them perfectly), the
skeleton that remains, and the cells estimated after the merge with the 2019
occupation-mix score. The first three columns are counted before the exposure
merge and are read from script 68's log; the last column and the employer
counts from the estimation exports of scripts 82, 85 and 88. The 26-30 flow
rows have no estimate on this score and print a dash. Panel B gives the
employers behind each comparison the paper draws, each count read from the
export of the fit that uses it.

Exports read (3_register_mona/exports/):
  2026-09-21_0812_s68/68_log.txt
  2026-09-22_2232_s82-partA/occ_route_coverage.csv
  2026-09-23_0655_s82-partB_s83-partsBCD/occ_route_headline.csv,
      occ_route_flows.csv, occ_route_profile.csv
  2026-09-23_1125_s85/occ_route_split65.csv
  2026-09-23_1407_s88/occ_route_contrast_by_track.csv
Output: output/tables/tableI2_sumstats_employment.tex

    python 4_exhibits/04_tab_estimation_sample.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

LOG = EXPORTS / "2026-09-21_0812_s68" / "68_log.txt"
S82A = EXPORTS / "2026-09-22_2232_s82-partA"
S82B = EXPORTS / "2026-09-23_0655_s82-partB_s83-partsBCD"
S85 = EXPORTS / "2026-09-23_1125_s85"
S88 = EXPORTS / "2026-09-23_1407_s88"
S95 = EXPORTS / "2026-09-25_1832_s95-s96-s98"

ROWS = [("22-25", "stock"), ("22-25", "hires"), ("22-25", "seps"),
        ("26-30", "stock"), ("26-30", "hires"), ("26-30", "seps")]
TERM = "post_x_high_x_young"
# The reported arm, as in script 82: the uniform three-digit score, the
# backward cascade, a floor of five incumbent person-months.
ARM, FLOOR, LEVEL = "backward", 5, "uniform3"
DASH = "---"


def parse_log() -> dict:
    """Pair each 'dropped X of Y' with the skeleton line that follows it.
    These three columns are counted before any exposure merge, so they do
    not depend on the exposure score."""
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
    h = pd.read_csv(S82B / "occ_route_headline.csv")
    h = h[(h.term == TERM) & (h.outcome == "stock")
          & (h.arm == ARM) & (h.floor == FLOOR) & (h.level == LEVEL)]
    for band in ("22-25", "26-30"):
        r = h[h.young_band == band]
        if len(r) != 1:
            raise SystemExit(f"  occ_route_headline.csv: {len(r)} rows for "
                             f"{band} on {ARM}/{FLOOR}/{LEVEL}, expected one")
        out[(band, "stock")] = (int(r.iloc[0].n_obs), int(r.iloc[0].n_firms))
    f = pd.read_csv(S82B / "occ_route_flows.csv")
    f = f[f.term == TERM]
    for _, r in f.iterrows():
        k = (str(r.young_band), str(r.outcome))
        if k in out:
            raise SystemExit(f"  occ_route_flows.csv: {k} appears twice")
        out[k] = (int(r.n_obs), int(r.n_firms))
    return out


def scored() -> int:
    d = pd.read_csv(S82A / "occ_route_coverage.csv")
    r = d[(d.block == "route") & (d.group == "occupation_route")
          & (d.item == "n_employers")]
    if len(r) != 1:
        raise SystemExit("  occ_route_coverage.csv: the occupation route's "
                         "scored employer count is not reported exactly once")
    return int(float(r.iloc[0]["n_employers"]))


def tex_thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def one_firm_count(path: Path, what: str, **restrict) -> int:
    """The single employer count an export reports, or a loud failure.

    Every panel in Panel B is read from the export of the fit that runs on
    it, never typed, so the appendix cannot drift from the estimates.
    """
    if not path.exists():
        raise SystemExit(f"  missing input for {what}: {path}")
    d = pd.read_csv(path)
    for k, v in restrict.items():
        d = d[d[k] == v]
    n = sorted(set(int(x) for x in d.n_firms.dropna()))
    if len(n) != 1:
        raise SystemExit(f"  {what}: one employer count expected in "
                         f"{path.name}, found {n}")
    return n[0]


def main() -> int:
    log, fit = parse_log(), fitted()
    n_scored = scored()
    missing_fit = []

    L = [r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
         r"\caption{The estimation sample: employer $\times$ age band "
         r"$\times$ month, 2021:01--2025:06.}",
         r"\label{tab:sumstats_employment_new}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Band & Outcome & Balanced cells & Zero throughout & Retained & "
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

    for b in ("22-25", "26-30"):
        if (b, "stock") not in fit:
            raise SystemExit(f"  no stock fit for {b}, so its panel count "
                             f"cannot be reported")

    # Panel B. Which employers identify which comparison. Each count comes
    # from the export of the fit that runs on that panel.
    panels = [
        ("Scored: incumbents aged 31--69 on the 2019 payroll",
         "no age band required", n_scored),
        ("Headline, 22--25 against the older bands pooled",
         "holds 22--25 and an older band", fit[("22-25", "stock")][1]),
        ("Headline, 26--30 against the older bands pooled",
         "holds 26--30 and an older band", fit[("26-30", "stock")][1]),
        ("Age profile, six bands against 41--49",
         "holds 41--49 and another band",
         one_firm_count(S82B / "occ_route_profile.csv",
                        "the six-band profile")),
        ("Age profile, seven bands, 50 and over split at 65",
         "holds 22--25 and another band",
         one_firm_count(S85 / "occ_route_split65.csv",
                        "the seven-band split")),
        ("Age profile, eight bands (Figure~2 of the paper), seven contrasts against 41--49",
         "holds 22--25 and another band",
         one_firm_count(S95 / "pension_reference.csv",
                        "the eight-band profile", part="P", spec="p8_tau")),
        ("Contrast by field of education, three bands",
         "holds 41--49, 22--25 or 26--30",
         one_firm_count(S88 / "occ_route_contrast_by_track.csv",
                        "the three-band contrast", track="all")),
    ]
    L += [r"\midrule",
          r"\multicolumn{6}{l}{\textit{Panel B. Which employers identify "
          r"which comparison}} \\",
          r"\multicolumn{3}{l}{Comparison} & "
          r"\multicolumn{2}{l}{An employer enters if it} & Employers \\"]
    for label, rule, n in panels:
        L.append(f"\\multicolumn{{3}}{{l}}{{{label}}} & "
                 f"\\multicolumn{{2}}{{l}}{{{rule}}} & {n:,} \\\\")
        print(f"  {label:58s} {rule:34s} {n:>9,}")

    note = r"The panel is balanced over employers, age bands and months and zero-filled. ``Zero throughout'' counts employer-band cells with no employment in any month; the employer-by-age effect predicts them perfectly, so they are dropped before estimation at no cost to any coefficient. ``Retained'' is what remains, and ``Estimated'' the cells at employers with a 2019 exposure score. Flows are estimated at 22--25 only. Panel~B gives the employers in each comparison's own panel."
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.95\textwidth}\footnotesize\vspace{4pt}" + note,
          r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableI2_sumstats_employment.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    if missing_fit:
        print(f"  no estimate on this route for: {', '.join(missing_fit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

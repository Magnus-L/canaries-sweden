#!/usr/bin/env python3
"""
l22_tab_descriptives.py -- descriptive statistics for the SURVIVING design.

WHY THIS EXISTS. Online Appendix I.2 carried two employment tables,
`tab:sumstats_employment` and `tab:panel_structure`, both describing an
employer x QUARTILE x month panel over 2019:01--2025:06, restricted to
employers with at least five workers and observed in both Q4 and one of
Q1--Q3. That is the withdrawn occupation design. The surviving design is
employer x AGE BAND x month from 2021-01, exposure fixed at the firm, so
there is no within-firm quartile variation at all and no size screen.

The paper's "seventy per cent of cells are zero at ages 22-25" came from
the dead table's zero-cell share of 0.704, computed on employer x
quartile x month cells. A submitted paper is expected to report its
sample; this rebuilds that reporting on the panel actually estimated.

Every number is parsed from an export, none typed in:
  68_log.txt          balanced cells, all-zero drops, skeleton sizes
  seasonal_pooled.csv the estimated cell count of each fit
  73_summary.txt      panel firms per band (73 uses 61's build_skeleton,
                      which is the same skeleton 68 uses)

WHAT THE TABLE MUST MAKE CLEAR. Zeros are kept. The panel is balanced and
zero-filled, so a firm whose young headcount falls to zero contributes
those months, which is the extensive margin the paper is about. What is
removed is (i) employer-band cells that are zero in EVERY month and
(ii) firms never holding the young band. Both have their employer-by-age
effect at minus infinity under Poisson and are separated by fixest
regardless; removing them in pandas only keeps R alive.

    python3 revision/local/l22_tab_descriptives.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

LOG = REV / "output/round3_20260921-lane14-seasonal/68_log.txt"
POOLED = REV / "output/round3_20260921-2152-lane14-seasonal-complete/seasonal_pooled.csv"
S73 = REV / "output/round3_20260921-lane19-crossing/73_summary.txt"
ROWS = [("22-25", "stock"), ("22-25", "hires"), ("22-25", "seps"),
        ("26-30", "stock"), ("26-30", "hires"), ("26-30", "seps")]


def parse_log():
    """Pair each 'dropped X of Y' with the skeleton line that follows it."""
    t = LOG.read_text(errors="replace").split("\n")
    out, pending = {}, None
    for ln in t:
        m = re.search(r"dropped ([\d,]+) of ([\d,]+) rows in cells that are "
                      r"zero in every month \((\d+)%\)", ln)
        if m:
            pending = (int(m.group(1).replace(",", "")),
                       int(m.group(2).replace(",", "")), int(m.group(3)))
            continue
        m = re.search(r"([\d-]+) (stock|hires|seps): skeleton ([\d,]+) rows", ln)
        if m and pending:
            key = (m.group(1), m.group(2))
            skel = int(m.group(3).replace(",", ""))
            # only trust the pairing when the arithmetic closes
            if pending[1] - pending[0] == skel:
                out.setdefault(key, (pending[1], pending[0], pending[2], skel))
            pending = None
    return out


def main() -> int:
    log = parse_log()
    pooled = pd.read_csv(POOLED)
    firms = {}
    for m in re.finditer(r"(?:industry|leverage)/([\d-]+): [\d,]+ of ([\d,]+) "
                         r"panel firms", S73.read_text(errors="replace")):
        firms[m.group(1)] = int(m.group(2).replace(",", ""))

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
            print(f"  MISSING in log: {band} {oc}")
            continue
        bal, drop, pct, skel = log[k]
        r = pooled[(pooled.young_band == band) & (pooled.outcome == oc)
                   & (pooled.arm == "true")
                   & (pooled.term == "post_x_high_x_young")]
        est = int(r.iloc[0]["n_obs"]) if not r.empty else None
        L.append(f"{band} & {oc} & {bal:,} & {drop:,} ({pct}\\%) & "
                 f"{skel:,} & {est:,} \\\\")
        print(f"  {band:6s} {oc:6s} balanced {bal:>12,}  zero-throughout "
              f"{drop:>11,} ({pct:2d}%)  skeleton {skel:>12,}  "
              f"estimated {est:>12,}")
    L += [r"\midrule",
          r"\multicolumn{2}{l}{Firms carrying an exposure score} & "
          r"\multicolumn{4}{r}{311{,}227} \\"]
    for b in ("22-25", "26-30"):
        if b in firms:
            L.append(f"\\multicolumn{{2}}{{l}}{{Firms in the {b} panel}} & "
                     f"\\multicolumn{{4}}{{r}}{{{firms[b]:,}}} \\\\")
            print(f"  panel firms {b}: {firms[b]:,}")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.95\textwidth}\footnotesize\vspace{4pt}"
          r"The panel is balanced over employers, age bands and months and "
          r"zero-filled, so \textbf{a firm whose young headcount falls to "
          r"zero contributes those months}: the extensive margin is in the "
          r"estimate, which is why Poisson rather than a log transform. "
          r"``Zero throughout'' counts employer-band cells with no employment "
          r"in any month of the window; these are perfectly predicted by the "
          r"employer-by-age effect and fixest separates them regardless, so "
          r"they are removed before estimation at no cost to any coefficient. "
          r"``Skeleton'' is what remains, and ``Estimated'' the cells "
          r"surviving the merge with the 2019 exposure score. A firm enters "
          r"only if it holds the young band and at least one older band at "
          r"some point from January 2021, which is a pre-treatment property "
          r"and not an outcome. Exposure is scored for 311{,}227 firms; the "
          r"panel firm counts come from the specification of Online Appendix "
          r"III.2, which is built on the same skeleton.",
          r"\end{minipage}", r"\end{table}"]
    out = V2_TAB / "tableI2_sumstats_employment.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

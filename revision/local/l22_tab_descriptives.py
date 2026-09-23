#!/usr/bin/env python3
"""
l22_tab_descriptives.py: Online Appendix Table I.2, the estimation sample
of the reported design.

WHAT THE TABLE REPORTS
For each young band and outcome (the employment stock, hires,
separations): the number of balanced employer by age band by month cells
from January 2021 to June 2025; the cells in employer-band series that
are zero in every month, which the employer-by-age effect predicts
exactly and which are removed before estimation; the skeleton that
remains; and the cells estimated after the merge with the 2019 exposure
score. Below, the number of employers carrying an exposure score
(311,227) and the number in each young band's panel. The panel is
balanced and zero-filled, so an employer whose young headcount falls to
zero contributes those months, which is why Poisson rather than a log
transform.

Every number is parsed from an export: the cell counts and drops from
script 68's log, the estimated cell counts from seasonal_pooled.csv, and
the panel employers per band from script 73's summary, which is built on
the same skeleton.

INPUTS AND OUTPUTS
Reads 68_log.txt and seasonal_pooled.csv (script 68) and 73_summary.txt
(script 73) from the export directories the final-code manifest names.
Writes revision/tables/tableI2_sumstats_employment.tex.

    python3 revision/local/l22_tab_descriptives.py

IN THE PAPER
Online Appendix I.2, Table tab:sumstats_employment_new.
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
         r"\setlength{\tabcolsep}{3.5pt}",
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

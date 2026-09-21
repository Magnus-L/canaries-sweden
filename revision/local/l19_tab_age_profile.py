#!/usr/bin/env python3
"""
l19_tab_age_profile.py -- the appendix table for the design the paper
actually reports, plus the age profile that is NOT on that design.

WHY THIS EXISTS. Until 21 September the online appendix had no section
for the headline design; its only appendix support was Part IV, the
backtest. Meanwhile `main_v2` sent the reader to Online Appendix III.2
for "the full age profile", and III.2 was the SUPERSEDED employer-level
DiD. So a live claim cited a dead design.

THE TRAP THIS TABLE EXISTS TO AVOID. There is no six-band age profile
on the headline route. The headline exposure is a QUARTILE built from
the education mix of a firm's incumbents aged 31+, and it is estimated
for the two young bands only (scripts 61 and 68). The six-band profiles
that do exist (script 63) use a CONTINUOUS occupation-scaled measure,
so their coefficients are per standard deviation on that measure's own
scale and are a different estimand entirely: -0.0132 at 22-25 against
the headline's -0.0509 before the cycle is removed. Putting them in one
column would repeat exactly the error that produced the change list's
B1, where an occupation-route stock figure and two horse-race flow
figures were quoted as though they were one design.

So the table has two panels and says which is which.

  Panel A  the headline route, both young bands, before and after the
           calendar cycle is removed, with the backtest artefact beside
           each. Sources: 61 (pre-seasonal) and 68 (seasonal).
  Panel B  the six-band profile on the continuous occupation-scaled
           measures, which is a robustness exercise about the measure,
           not the headline estimand. Source: 63.

READ RULES.
  * Never compare a Panel A number with a Panel B number. Different
    exposure construction, different scale.
  * 41-49 is negative and significant in Panel B on both AI routes.
    What the paper must NOT say is that the young were hit harder than
    the prime-aged: tested directly with the calendar cycle removed,
    that contrast is -0.0153 (0.0126), a null.
  * Telework is the identification check, not a rival AI measure.

    python3 revision/local/l19_tab_age_profile.py [export_dir ...]
"""
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
TERM = "post_x_high_x_young"
POOLED = "post2024_x_high_x_young"


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if r.exists():
            for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
                if best is None or p.stat().st_mtime > best.stat().st_mtime:
                    best = p
    return best


def fmt(c, se, p=None):
    if c is None:
        return "PENDING"
    sig = (p < 0.05) if p is not None else (abs(c) > 1.96 * se)
    star = "^{*}" if sig else ""
    return f"${c:+.4f}{star}$ ({se:.4f})"


def panel_a(argv):
    """Headline route: education-mix quartile, the two young bands."""
    rows = []
    seas = find(argv, "seasonal_pooled.csv")
    pre = find(argv, "redated_pooled.csv")
    if seas is not None:
        d = pd.read_csv(seas)
        for band in ("22-25", "26-30"):
            t = d[(d["young_band"] == band) & (d["outcome"] == "stock")
                  & (d["arm"] == "true") & (d["term"] == TERM)]
            a = d[(d["young_band"] == band) & (d["outcome"] == "stock")
                  & (d["arm"] == "asof") & (d["term"] == TERM)]
            if t.empty:
                continue
            art = (f"${float(a.iloc[0]['coef']) - float(t.iloc[0]['coef']):+.4f}$"
                   if not a.empty else "--")
            rows.append((f"{band}, cycle removed",
                         fmt(float(t.iloc[0]["coef"]),
                             float(t.iloc[0]["se"])), art))
    if pre is not None:
        d = pd.read_csv(pre)
        for band in ("22-25", "26-30"):
            t = d[(d["young_band"] == band) & (d["arm"] == "true")
                  & (d["term"] == POOLED) & (d["design"] == "OL_daioe")]
            a = d[(d["young_band"] == band) & (d["arm"] == "asof")
                  & (d["term"] == POOLED) & (d["design"] == "OL_daioe")]
            if t.empty:
                continue
            art = (f"${float(a.iloc[0]['coef']) - float(t.iloc[0]['coef']):+.4f}$"
                   if not a.empty else "--")
            rows.append((f"{band}, before the cycle",
                         fmt(float(t.iloc[0]["coef"]),
                             float(t.iloc[0]["se"])), art))
    return rows


def panel_b(argv):
    """Six bands, continuous occupation-scaled measures. Different estimand."""
    src = find(argv, "robustness_gradient.csv")
    if src is None:
        return []
    d = pd.read_csv(src)
    d = d[d["dating"] == "adoption"]
    out = []
    for b in BANDS:
        cells = []
        for m in ("daioe", "eloundou", "telework"):
            r = d[(d["age_group"] == b) & (d["measure"] == m)
                  & (d["outcome"] == "stock")]
            cells.append("--" if r.empty else
                         fmt(float(r.iloc[0]["coef"]), float(r.iloc[0]["se"]),
                             float(r.iloc[0]["pvalue"])))
        out.append((b, cells))
    return out


def main() -> int:
    A, B = panel_a(sys.argv), panel_b(sys.argv)
    if not A and not B:
        print("  no exports found")
        return 1

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The headline design by age, and the age profile on a "
           r"different exposure construction.}",
           r"\label{tab:age_profile_rebuilt}", r"\footnotesize",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"\multicolumn{4}{l}{\textit{Panel A. Headline route: "
           r"top-quartile 2019 education mix of incumbents aged 31+}} \\",
           r"\addlinespace[2pt]",
           r" & Estimate (SE) & Artefact & \\", r"\midrule"]
    for lab, est, art in A:
        tex.append(f"{lab} & {est} & {art} & \\\\")
        print(f"  A  {lab:28s} {est}  art {art}")
    tex += [r"\addlinespace[6pt]",
            r"\multicolumn{4}{l}{\textit{Panel B. Continuous "
            r"occupation-scaled measures, per standard deviation}} \\",
            r"\addlinespace[2pt]",
            r"Age band & DAIOE & Eloundou & Teleworkable \\", r"\midrule"]
    for b, cells in B:
        tex.append(f"{b} & " + " & ".join(cells) + r" \\")
        print(f"  B  {b:6s} " + "  ".join(c.replace('$', '') for c in cells))
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson pseudo-maximum likelihood on employer $\times$ age "
            r"$\times$ month counts, employer-by-month, employer-by-age and "
            r"month-by-age effects, treatment dated January 2024, standard "
            r"errors clustered by employer, $^{*}$ denotes $p<0.05$. "
            r"Exposure is frozen in 2019 throughout and no occupation code "
            r"recorded after 2019 enters the exposure, the outcome or the "
            r"sample. \textbf{The two panels are not comparable.} Panel A "
            r"is a top-quartile indicator built from the education mix of a "
            r"firm's incumbents, and it is the estimand the paper reports; "
            r"it exists for the two young bands only. Panel B scores "
            r"occupations continuously, so a coefficient is the effect of a "
            r"one standard deviation more exposed occupation on that "
            r"measure's own scale. The artefact column gives what the as-of "
            r"backtest returns on the same specification when the "
            r"register's lag is imposed on years whose true gap is zero; "
            r"the threshold fixed before the test was 0.05. Employment at "
            r"41--49 declines on both AI measures in Panel B. The paper "
            r"does not claim the young were hit harder than the "
            r"prime-aged: tested directly with the calendar cycle removed, "
            r"that contrast is $-0.0153$ (0.0126), a null.",
            r"\end{minipage}", r"\end{table}"]
    out = V2_TAB / "tableA_age_profile.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
24_tab_unlinked.py: Online Appendix Table A40 (Section VI.1), the payslips
whose worker reaches none of the three individual registers, on the headline
sample and contrast: their share by exposure group and period, the tipping
point, and tau under two extremal allocations (script 100).

Panel A counts, on the 104,217 employers of the headline panel at 22-25, the
declared person-months of every age whose worker has no birth year and sex
in the 2023, 2021 or 2019 individual register, by exposure group and period,
as a share of the declared person-months of the group and period. The
difference row is the difference of the two shares as printed. Panel B: the
proportional increase in the young band's later-period counts at top-
quartile employers that sets tau to zero, found by scaling those counts by a
common factor and verified by a refit at that factor; the person-months it
implies and their ratio to the unlinked person-months at those employers in
the later period; the band's own share of declared person-months; and tau
with every unlinked person-month at top-quartile employers added to the band
(interim and later periods as they occur) and with only the later-period
excess over each employer's interim rate added. The second column does the
same for the female differential, the additions going to young women.

Nothing is written unless the gate rows reproduce Table 1 (tau -0.0399
(0.0102) pooled, -0.0714 (0.0109) for the differential), the exported
standard error of every derived tau is the square root of the exported
variance of the difference, the tipping point's refit returns a tau of zero
to four decimals, the person-months the tipping point implies equal the
factor applied to the band's later-period counts, and every printed number
agrees with the run's own summary (100_summary.txt).

Exports read: 3_register_mona/exports/2026-09-26_0827_s100-s101/
  unlinked_accounting.csv, tipping_point.csv, 100_summary.txt
Output: output/tables/tableA_unlinked.tex (a bare tabular and note; the
        appendix supplies the float and caption)

    python 4_exhibits/24_tab_unlinked.py [export_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S100 = EXPORTS / "2026-09-26_0827_s100-s101"
PERIODS = [("pre_hike", "Jan 2021--Mar 2022"), ("tightening", "Apr--Nov 2022"),
           ("interim", "Dec 2022--Dec 2023"), ("later", "Jan 2024--Jun 2025")]
GATES = {"P": (-0.0399, 0.0102), "S": (-0.0714, 0.0109)}


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else S100


def need(name: str) -> Path:
    p = d() / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def pick(df: pd.DataFrame, **cond) -> pd.Series:
    r = df
    for k, v in cond.items():
        r = r[r[k] == v]
    if len(r) != 1:
        raise SystemExit(f"  expected one row for {cond}, found {len(r)}")
    return r.iloc[0]


def tau_row(df: pd.DataFrame, part: str, spec: str) -> tuple[float, float]:
    r = pick(df, part=part, spec=spec, term="tau")
    var = (float(r.var_post) + float(r.var_interim) - 2.0 * float(r.cov_post_interim))
    if abs(np.sqrt(var) - float(r.se)) > 1e-6:
        raise SystemExit(f"  {part} {spec}: the exported standard error is not "
                         f"the square root of the exported variance")
    return float(r.coef), float(r.se)


def value(df: pd.DataFrame, part: str, term: str) -> float:
    return float(pick(df, part=part, spec="tipping", term=term).coef)


def summary_blocks(text: str) -> dict:
    """The POOLED and SEX blocks of 100_summary.txt as {block: {key: value}}."""
    blocks, cur = {}, None
    for line in text.splitlines():
        if line.startswith("POOLED"):
            cur = "P"
        elif line.startswith("SEX"):
            cur = "S"
        elif line.startswith("VERDICTS"):
            cur = None
        elif cur and line.strip():
            m = re.match(r"^\s*(\S+)\s+([-+]?[0-9,.]+)%?\s*$", line)
            if m:
                blocks.setdefault(cur, {})[m.group(1)] = float(m.group(2).replace(",", ""))
                continue
            m = re.match(r"^\s*tau (\S+)\s+([-+][0-9.]+) \(([0-9.]+)\)", line)
            if m:
                blocks.setdefault(cur, {})[f"tau {m.group(1)}"] = (float(m.group(2)),
                                                                  float(m.group(3)))
    if set(blocks) != {"P", "S"}:
        raise SystemExit("  100_summary.txt: the POOLED and SEX blocks do not parse")
    return blocks


def main() -> int:
    print(f"  export {d().name}")
    acc = pd.read_csv(need("unlinked_accounting.csv"))
    tip = pd.read_csv(need("tipping_point.csv"))
    tip = tip[tip.status.isin(["ok", "derived", "count"])]
    said = summary_blocks(need("100_summary.txt").read_text(encoding="utf-8",
                                                            errors="replace"))

    # Panel A
    share = {}
    for grp in ("high", "other"):
        for per, _ in PERIODS:
            r = pick(acc, group=grp, period=per)
            if float(r.u_pooled) != float(r.n_noreg) or float(r.n_nobirth) != 0:
                raise SystemExit(f"  {grp} {per}: unlinked person-months are not "
                                 f"all workers with no register record")
            share[(grp, per)] = 100.0 * float(r.u_pooled) / float(r.n_all)
    for per in ("interim", "later"):
        r = pick(acc, group="high", period=per)
        key = f"share_unlinked_high_{per}"
        if key in said["P"] and round(share[("high", per)], 4) != said["P"][key]:
            raise SystemExit(f"  high {per}: share {share[('high', per)]:.4f} is "
                             f"not the summary's {said['P'][key]}")
    u_high = {per: float(pick(acc, group="high", period=per).u_pooled)
              for per, _ in PERIODS}
    if u_high["later"] != said["P"]["U_high_later"]:
        raise SystemExit("  U_high_later disagrees with the summary")
    # The note names the headline panel, whose size is the gate fit's employer
    # count (104,217 at 22-25), not the number of panel employers declaring
    # in any one period (104,001 before the rate rise, 96,529 in the later period).
    gate = tip[(tip.part == "P") & (tip.spec == "gate") & (tip.term == "tau")]
    if len(gate) != 1:
        raise SystemExit("  the gate row of tipping_point.csv is missing")
    n_emp = int(float(gate.n_firms.iloc[0]))
    diff = {per: round(share[("high", per)], 3) - round(share[("other", per)], 3)
            for per, _ in PERIODS}
    rows_a = [
        "Top exposure quartile & " + " & ".join(f"{share[('high', p)]:.3f}" for p, _ in PERIODS) + r" \\",
        "Other employers & " + " & ".join(f"{share[('other', p)]:.3f}" for p, _ in PERIODS) + r" \\",
        "Difference, percentage points & " + " & ".join(f"{diff[p]:.3f}" for p, _ in PERIODS) + r" \\",
        "Unlinked person-months, top quartile & " + " & ".join(thousands(u_high[p]) for p, _ in PERIODS) + r" \\",
    ]
    for r in rows_a:
        print("  " + r)

    # Panel B
    cols = {}
    for part in ("P", "S"):
        gate = tau_row(tip, part, "gate")
        if (round(gate[0], 4), round(gate[1], 4)) != GATES[part]:
            raise SystemExit(f"  {part}: the gate tau {gate} is not Table 1's {GATES[part]}")
        kstar = tau_row(tip, part, "scaled_kstar")
        if round(kstar[0], 4) != 0.0:
            raise SystemExit(f"  {part}: the refit at k* gives tau {kstar[0]:+.6f}, not zero")
        m_star = value(tip, part, "m_star")
        y = value(tip, part, "young_pm_high_later_Y")
        needed = value(tip, part, "young_pm_needed")
        if abs(m_star * y - needed) > 0.5:
            raise SystemExit(f"  {part}: needed person-months {needed:.1f} are not "
                             f"m* times the band's later-period counts {m_star * y:.1f}")
        u_later = value(tip, part, "U_high_later")
        a_later = value(tip, part, "A_high_later")
        ratio = value(tip, part, "needed_over_U_high_later")
        if abs(needed / u_later - ratio) > 1e-9 or u_later != u_high["later"]:
            raise SystemExit(f"  {part}: the ratio to the unlinked total does not reproduce")
        x_all = tau_row(tip, part, "x_all")
        x_diff = tau_row(tip, part, "x_diff")
        pm_all = float(pick(tip, part=part, spec="x_all", term="allocated_pm").coef)
        pm_diff = float(pick(tip, part=part, spec="x_diff", term="allocated_pm").coef)
        s = said[part]
        checks = [("m_star", 100 * m_star, s["m_star"], 4),
                  ("young_pm_needed", needed, s["young_pm_needed"], 0),
                  ("needed_over_U_high_later", ratio, s["needed_over_U_high_later"], 2),
                  ("young_pm_high_later_Y", y, s["young_pm_high_later_Y"], 0)]
        for what, got, want, dp in checks:
            if round(got, dp) != round(want, dp):
                raise SystemExit(f"  {part} {what}: export {got} against summary {want}")
        for what, got in (("tau x_all", x_all), ("tau x_diff", x_diff), ("tau scaled_kstar", kstar)):
            want = s[what]
            if round(got[0], 5) != want[0] or round(got[1], 5) != want[1]:
                raise SystemExit(f"  {part} {what}: export {got} against summary {want}")
        cols[part] = dict(gate=gate, m=100 * m_star, needed=needed, ratio=ratio,
                          band_share=100 * y / a_later, x_all=x_all, pm_all=pm_all,
                          x_diff=x_diff, pm_diff=pm_diff)
        print(f"  {part}: tau {gate[0]:+.4f} ({gate[1]:.4f}); m* {100 * m_star:.1f} per cent, "
              f"{needed:,.0f} person-months, {ratio:.2f} of the unlinked; band share "
              f"{100 * y / a_later:.1f}; X_all {x_all[0]:+.4f} ({x_all[1]:.4f}); "
              f"X_diff {x_diff[0]:+.4f} ({x_diff[1]:.4f})")
    print("  100_summary.txt reproduces every printed number")

    def two(f) -> str:
        return f"{f(cols['P'])} & {f(cols['S'])}"

    e = lambda t: f"${t[0]:+.4f}$ ({t[1]:.4f})"  # noqa: E731
    rows_b = [
        r" & & & Ages 22--25 & Young women minus young men \\", r"\midrule",
        r"$\tau$ as estimated & & & " + two(lambda c: e(c["gate"])) + r" \\",
        r"Proportional increase in the young band's later-period counts at top-quartile employers that sets $\tau$ to zero & & & "
        + two(lambda c: f"{c['m']:.1f} per cent") + r" \\",
        r"\quad in person-months & & & " + two(lambda c: thousands(c["needed"])) + r" \\",
        r"\quad as a share of all unlinked person-months at those employers in the later period, any age & & & "
        + two(lambda c: f"{c['ratio']:.2f}") + r" \\",
        r"The band's own share of declared person-months at those employers & & & "
        + two(lambda c: f"{c['band_share']:.1f} per cent") + r" \\",
        r"$\tau$ with every unlinked person-month at top-quartile employers added to the band & & & "
        + two(lambda c: e(c["x_all"])) + r" \\",
        r"\quad person-months added & & & " + two(lambda c: thousands(c["pm_all"])) + r" \\",
        r"$\tau$ with only the later-period excess over each employer's interim rate added & & & "
        + two(lambda c: e(c["x_diff"])) + r" \\",
        r"\quad person-months added & & & " + two(lambda c: thousands(c["pm_diff"])) + r" \\",
    ]

    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"\multicolumn{5}{l}{\emph{Panel A. Payslips whose worker reaches none of the three registers, per cent of declared person-months, headline employers}} \\",
           " & " + " & ".join(lab for _, lab in PERIODS) + r" \\", r"\midrule"]
    tex += rows_a
    tex += [r"\midrule",
            r"\multicolumn{5}{l}{\emph{Panel B. What the unlinked person-months could do to $\tau$}} \\"]
    tex += rows_b
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            r"Panel~A: the " + thousands(n_emp) + r" employers of the headline panel at 22--25; declared person-months of every age in the employer declarations; a worker is unlinked when no birth year and sex can be read from the 2023, 2021 or 2019 individual register, in practice a worker first registered in Sweden in 2024 or 2025. Panel~B: the tipping point is found by scaling the later-period counts of the band at top-quartile employers by a common factor until $\tau$ is zero, verified by a refit at that factor ($\tau = -0.0000$). The allocations add unlinked person-months to the band's cell at the employer and month that declared them, in the interim and later periods as they occur, and leave every other cell as observed; the second allocation adds only the later-period person-months in excess of what each employer's own interim unlinked rate implies, where there is an excess. For the female differential the additions go to young women. The declaration carries no age, so these are counterfactuals on the counts, not a model of who the unlinked workers are, and the tipping point is not a bound. Standard errors clustered by employer.",
            r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_unlinked.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

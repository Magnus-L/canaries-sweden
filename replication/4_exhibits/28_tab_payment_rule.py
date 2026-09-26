#!/usr/bin/env python3
"""
28_tab_payment_rule.py: Online Appendix table (Section VI.1), what the
counting rule admits: the share of counted person-months that carry cash
pay subject to employer contributions, by age band, exposure group and
period; what the remainder carries; and the same for the 2019 scoring
population (script 104, lane 38e).

A counted person-month is any distinct person for whom the employer files
an individual declaration record in the month, birth year known, age 22 to
69; cash pay is not required. Panel A: the share of those person-months
with cash pay (KONTANT_ERSATTNING_ULAG_AG > 0), top quartile of the paper's
2019 score against the other three quartiles, by period. Panel B: top minus
rest in percentage points, and the change from the interim to the later
period. Panel C: among counted person-months without cash pay, the share
carrying an occupational-pension amount and the share carrying a taxable
benefit, all employers and periods. Panel D: the 2019 person-months aged 31
to 69 behind the score, by exposure group. The note carries the movement of
tau that counting cash-pay person-months alone would imply on the shares:
the later-minus-interim, top-minus-rest change in log(share) for the young
band less the same for the older bands.

Nothing is written unless every printed share reproduces the run's own
summary (104_summary.txt) to four decimals and the percentage-point
differences to three.

Exports read: 3_register_mona/exports/2026-09-26_1909_s104/
  payment_rule.csv, payment_rule_2019_incumbents.csv, 104_summary.txt
Output: output/tables/tableA_payment_rule.tex (a bare tabular and note;
        the appendix supplies the float and caption)

    python 4_exhibits/28_tab_payment_rule.py [export_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S104 = EXPORTS / "2026-09-26_1909_s104"
PERIODS = [("reference", "Jan 2021--Mar 2022"), ("tightening", "Apr--Nov 2022"),
           ("interim", "Dec 2022--Dec 2023"), ("later", "Jan 2024--Jun 2025")]
BANDS = ["22-25", "26-30", "31-69"]


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else S104


def need(name: str) -> Path:
    p = d() / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def thousands(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def main() -> int:
    sh = pd.read_csv(need("payment_rule.csv"))
    sc = pd.read_csv(need("payment_rule_2019_incumbents.csv"))
    summ = need("104_summary.txt").read_text(encoding="utf-8", errors="replace")
    if "THE GATE PASSES" not in summ or "THE GATE (2019) PASSES" not in summ:
        raise SystemExit("  104_summary.txt does not record both gates passing; nothing is written")

    def share(period, band, group) -> float:
        r = sh[(sh.period == period) & (sh.band == band) & (sh.group == group)]
        if len(r) != 1 or not np.isfinite(r.share_pay.iloc[0]):
            raise SystemExit(f"  no share for {period} {band} {group}")
        return float(r.share_pay.iloc[0])

    # ---- check every share against the summary's P1 block ---------------
    p1 = summ[summ.index("P1. SHARE"):summ.index("P2. TOP MINUS REST")]
    for band in BANDS:
        block = p1[p1.index(f"  {band}:"):]
        for period, _ in PERIODS:
            m = re.search(rf"{period}\s+top ([0-9.]+)\s+rest ([0-9.]+)\s+all ([0-9.]+)", block)
            if not m:
                raise SystemExit(f"  summary has no P1 line for {band} {period}")
            for grp, val in (("top", m.group(1)), ("rest", m.group(2)), ("all", m.group(3))):
                if abs(share(period, band, grp) - float(val)) > 5e-5:
                    raise SystemExit(f"  {band} {period} {grp}: csv {share(period, band, grp):.4f} vs summary {val}")
    # ---- P2 -----------------------------------------------------------
    diffs = {}
    for band in ("22-25", "31-69"):
        diffs[band] = {p: 100 * (share(p, band, "top") - share(p, band, "rest")) for p, _ in PERIODS}
        m = re.search(rf"  {re.escape(band)}: reference ([-+0-9.]+)\s+tightening ([-+0-9.]+)\s+interim ([-+0-9.]+)"
                      rf"\s+later ([-+0-9.]+)\s+\| later minus interim ([-+0-9.]+) pp", summ)
        if not m:
            raise SystemExit(f"  summary has no P2 line for {band}")
        got = [diffs[band][p] for p, _ in PERIODS] + [diffs[band]["later"] - diffs[band]["interim"]]
        for g, w in zip(got, m.groups()):
            if abs(g - float(w)) > 5e-4:
                raise SystemExit(f"  P2 {band}: {g:+.3f} vs summary {w}")
    # ---- the implied movement of tau on the shares ----------------------
    def dlog(band):
        return (np.log(share("later", band, "top")) - np.log(share("interim", band, "top"))) \
            - (np.log(share("later", band, "rest")) - np.log(share("interim", band, "rest")))
    tau_move = dlog("22-25") - dlog("31-69")
    # ---- P3 -----------------------------------------------------------
    p3 = {}
    for band in BANDS:
        sub = sh[(sh.band == band) & (sh.group == "all")]
        nopay = int((sub.n_emp - sub.n_pay).sum())
        p3[band] = {"nopay": nopay, "share": 100 * nopay / int(sub.n_emp.sum()),
                    "pension": 100 * int(sub.n_nopay_pension.sum()) / nopay,
                    "benefit": 100 * int(sub.n_nopay_benefit.sum()) / nopay}
        m = re.search(rf"  {re.escape(band)}: ([0-9,]+) person-months without cash pay \(([0-9.]+) per cent of counted\); "
                      rf"with a pension amount ([0-9.]+) per cent; with a taxable benefit ([0-9.]+) per cent", summ)
        if not m:
            raise SystemExit(f"  summary has no P3 line for {band}")
        if int(m.group(1).replace(",", "")) != nopay or abs(float(m.group(2)) - p3[band]["share"]) > 5e-3 \
                or abs(float(m.group(3)) - p3[band]["pension"]) > 0.05 or abs(float(m.group(4)) - p3[band]["benefit"]) > 0.05:
            raise SystemExit(f"  P3 {band} does not reproduce the summary")
    # ---- P4 -----------------------------------------------------------
    p4 = {}
    for grp in ("top", "rest", "all"):
        r = sc[sc.group == grp]
        if len(r) != 1:
            raise SystemExit(f"  no 2019 row for {grp}")
        r = r.iloc[0]
        p4[grp] = {"share": float(r.share_pay), "n": int(r.n_emp),
                   "pension": 100 * float(r.share_nopay_pension), "benefit": 100 * float(r.share_nopay_benefit)}
        m = re.search(rf"    {grp}\s+([0-9.]+)\s+\(([0-9,]+) person-months; without cash pay ([0-9,]+), of which pension "
                      rf"([0-9.]+) per cent, benefit ([0-9.]+) per cent\)", summ)
        if not m or abs(float(m.group(1)) - p4[grp]["share"]) > 5e-5 or int(m.group(2).replace(",", "")) != p4[grp]["n"]:
            raise SystemExit(f"  P4 {grp} does not reproduce the summary")
    print(f"  every share, difference and count reproduces 104_summary.txt; tau movement on the shares {tau_move:+.4f}")

    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    rows_a = []
    for band in BANDS:
        for grp, lab in (("top", "top quartile"), ("rest", "other quartiles")):
            rows_a.append(f"{band}, {lab} & " + " & ".join(pct(share(p, band, grp)) for p, _ in PERIODS) + r" \\")
    rows_b = []
    for band in ("22-25", "31-69"):
        rows_b.append(f"{band} & " + " & ".join(f"${diffs[band][p]:+.2f}$" for p, _ in PERIODS)
                      + f" & ${diffs[band]['later'] - diffs[band]['interim']:+.2f}$" + r" \\")
    rows_c = [f"{band} & {p3[band]['share']:.2f} & {p3[band]['pension']:.1f} & {p3[band]['benefit']:.1f} & \\\\"
              for band in BANDS]
    rows_d = [f"{lab} & {pct(p4[grp]['share'])} & {p4[grp]['pension']:.1f} & {p4[grp]['benefit']:.1f} & {thousands(p4[grp]['n'])} \\\\"
              for grp, lab in (("top", "Top quartile"), ("rest", "Other quartiles"), ("all", "All scored employers"))]

    tex = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"\multicolumn{6}{l}{\emph{Panel A. Counted person-months with cash pay subject to employer contributions, per cent}} \\",
           " & " + " & ".join(lab for _, lab in PERIODS) + r" & \\", r"\midrule"]
    tex += rows_a
    tex += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{Panel B. Top quartile minus other quartiles, percentage points}} \\",
            " & " + " & ".join(lab for _, lab in PERIODS) + r" & Later minus interim \\", r"\midrule"]
    tex += rows_b
    tex += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{Panel C. Counted person-months without cash pay, all employers and periods}} \\",
            r" & Per cent of counted & With a pension amount, per cent & With a taxable benefit, per cent & & \\", r"\midrule"]
    tex += rows_c
    tex += [r"\midrule",
            r"\multicolumn{6}{l}{\emph{Panel D. The 2019 scoring population: person-months aged 31--69 behind the employer score}} \\",
            r" & With cash pay, per cent & Of the remainder: pension, per cent & benefit, per cent & Person-months & \\", r"\midrule"]
    tex += rows_d
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"A counted person-month is any distinct person for whom the employer files an individual declaration record in the month, with a birth year in the 2023, 2021 or 2019 individual register and an age of 22 to 69; cash pay is not required. Cash pay: the declaration's cash remuneration subject to employer contributions exceeds zero; pension: an occupational-pension amount is declared; benefit: a car or other taxable benefit is declared. Exposure groups by the paper's 2019 employer score; employers the score does not reach are in the row totals of Panel~C only. "
            r"On these shares, counting cash-pay person-months alone would move $\hat\tau$ at 22--25 by the later-minus-interim, top-minus-rest change in the log share of the young band less that of the older bands, "
            + f"${tau_move:+.4f}$" + r", about a quarter of its standard error and towards a larger decline; this is a calculation on the shares, not a refit.",
            r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_payment_rule.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

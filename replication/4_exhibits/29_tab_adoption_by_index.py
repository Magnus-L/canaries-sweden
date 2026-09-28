#!/usr/bin/env python3
"""
29_tab_adoption_by_index.py: Online Appendix Table A27 (Section III.2), which
exposure index picks out the employers that report using AI: the paper's
DAIOE classification against the Eloundou rating, on the employers both
indices score (script 108, lane 39d).

Each row is one Statistics Sweden survey and outcome, the waves of OA
Figure A4 plus the two 2019-21 research-and-development surveys and the 2019
IT-expenditure survey. "Separately": the top-quartile differential in
percentage points on each classification alone (script 71's linear
probability model, log 2019 size for the firm surveys, survey weights for the
worker survey). "Together": both top-quartile indicators in one regression
on the same firms, each index's differential holding the other fixed, and
the difference DAIOE minus Eloundou with its standard error from the same
fit's robust covariance. The two top quartiles share most employers, so the
joint fit is identified from those on which they disagree.

Nothing is written unless every printed coefficient and standard error
reproduces the run's own summary (108_summary.txt, parts A1 and A3) to two
decimals, the difference equals DAIOE minus Eloundou in the joint fit, and
the run's gate passed.

Export read: 3_register_mona/exports/2026-09-27_1009_s108/
  adoption_by_index.csv, 108_summary.txt
Output: output/tables/tableA_adoption_by_index.tex (a bare tabular and
        note; the appendix supplies the float and caption)

    python 4_exhibits/29_tab_adoption_by_index.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

DEFAULT = EXPORTS / "2026-09-27_1009_s108"

# (source, outcome, label), in the order printed: pre-ChatGPT first
ROWS = [
    ("ai_itftg_2019", "ai_any", "ICT survey 2019, any AI"),
    ("ai_fufi_2019", "ai_any", "IT-expenditure survey 2019, any AI"),
    ("ai_fouftg_2019_2021", "ai_any", "R\\&D survey, business, 2019--21, AI in R\\&D"),
    ("ai_fouoff_2019_2021", "ai_any", "R\\&D survey, public sector, 2019--21, AI in R\\&D"),
    ("ITFtg_Stora_2021", "ai_any", "ICT survey 2021, any AI"),
    ("ITFtg_Stora_2021", "ai_genai", "ICT survey 2021, language generation"),
    ("ITFtg_Stora_2023", "ai_any", "ICT survey 2023, any AI"),
    ("ITFtg_Stora_2023", "ai_genai", "ICT survey 2023, language generation"),
    ("BITA_2024", "genai", "Worker survey 2024, generative AI"),
]
A1 = re.compile(r"^\s*(\S+)\s+(\S+)\s+DAIOE\s+([-+][0-9.]+) \(([0-9.]+)\)\s+Eloundou\s+([-+][0-9.]+) \(([0-9.]+)\)\s+n ([0-9,]+)")
A3 = re.compile(r"^\s*(\S+)\s+(\S+)\s+DAIOE\s+([-+][0-9.]+) \(([0-9.]+)\)\s+Eloundou\s+([-+][0-9.]+) \(([0-9.]+)\)\s+diff\s+([-+][0-9.]+) \(([0-9.]+)\) p ([0-9.]+)\s+n ([0-9,]+)")


def summary(path: Path) -> tuple:
    """Parts A1 and A3 of 108_summary.txt, and whether the gate passed."""
    text = path.read_text(encoding="utf-8", errors="replace")
    part, a1, a3 = None, {}, {}
    for line in text.splitlines():
        if line.startswith("A1."):
            part = "A1"
        elif line.startswith("A2."):
            part = None
        elif line.startswith("A3."):
            part = "A3"
        elif line.startswith("COUNTS"):
            part = None
        m = A1.match(line) if part == "A1" else A3.match(line) if part == "A3" else None
        if m and part == "A1":
            a1[(m.group(1), m.group(2))] = [float(x) for x in m.group(3, 4, 5, 6)] + [int(m.group(7).replace(",", ""))]
        elif m and part == "A3":
            a3[(m.group(1), m.group(2))] = [float(x) for x in m.group(3, 4, 5, 6, 7, 8, 9)] + [int(m.group(10).replace(",", ""))]
    return a1, a3, "GATE: PASSES" in text


def main(export_dir: Path) -> int:
    t = pd.read_csv(export_dir / "adoption_by_index.csv")
    a1, a3, gate = summary(export_dir / "108_summary.txt")
    if not gate:
        raise SystemExit("  108_summary.txt: the gate did not pass; nothing is written")

    def cell(route, src, out):
        r = t[(t.route == route) & (t.source == src) & (t.outcome == out)]
        if len(r) != 1:
            raise SystemExit(f"  {route} {src} {out}: {len(r)} rows in the export")
        return float(r.coef_points.iloc[0]), float(r.se_points.iloc[0]), int(r.n.iloc[0])

    def fmt(c, s):
        return f"${c:+.1f}$ & ({s:.1f})"

    body = []
    for src, out, lab in ROWS:
        dc, ec = cell("daioe_common", src, out), cell("eloundou_common", src, out)
        jd, je, jf = (cell(r, src, out) for r in ("joint_daioe", "joint_eloundou", "joint_diff"))
        # every printed number against the run's own summary
        s1, s3 = a1[(src, out)], a3[(src, out)]
        for got, said, what in ((dc[0], s1[0], "A1 DAIOE"), (dc[1], s1[1], "A1 DAIOE SE"),
                                (ec[0], s1[2], "A1 Eloundou"), (ec[1], s1[3], "A1 Eloundou SE"),
                                (jd[0], s3[0], "A3 DAIOE"), (jd[1], s3[1], "A3 DAIOE SE"),
                                (je[0], s3[2], "A3 Eloundou"), (je[1], s3[3], "A3 Eloundou SE"),
                                (jf[0], s3[4], "A3 diff"), (jf[1], s3[5], "A3 diff SE")):
            if round(got, 2) != said:
                raise SystemExit(f"  {src} {out} {what}: export {got:.4f}, summary {said}")
        if abs(jf[0] - (jd[0] - je[0])) > 1e-6:
            raise SystemExit(f"  {src} {out}: the difference is not DAIOE minus Eloundou")
        if not (dc[2] == ec[2] == jd[2] == s1[4] == s3[7]):
            raise SystemExit(f"  {src} {out}: the samples differ across the columns")
        body.append(f"{lab} & {fmt(*dc[:2])} & {fmt(*ec[:2])} & {fmt(*jd[:2])} & {fmt(*je[:2])} & "
                    f"{fmt(*jf[:2])} & {dc[2]:,} \\\\".replace(",", "{,}"))
        print(f"  {src} {out}: {dc[0]:+.1f} / {ec[0]:+.1f}; joint {jd[0]:+.1f} / {je[0]:+.1f}, diff {jf[0]:+.1f} ({jf[1]:.1f})")

    tex = [r"\begin{tabular}{l*{5}{r@{\,}l}r}", r"\toprule",
           r" & \multicolumn{4}{c}{Separately} & \multicolumn{6}{c}{Together} & \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-11}",
           r"Survey and outcome & \multicolumn{2}{c}{DAIOE} & \multicolumn{2}{c}{Eloundou} & "
           r"\multicolumn{2}{c}{DAIOE} & \multicolumn{2}{c}{Eloundou} & \multicolumn{2}{c}{Difference} & $N$ \\",
           r"\midrule"] + body[:4] + [r"\addlinespace"] + body[4:8] + [r"\addlinespace"] + body[8:] + [
           r"\bottomrule", r"\end{tabular}",
           r"\begin{minipage}{0.97\textwidth}\footnotesize\vspace{4pt}",
           r"Top-quartile differential in reported use, percentage points, with robust standard errors, on the employers both indices score. "
           r"Separately: each classification's top-quartile indicator alone. Together: both indicators in one regression; the difference is DAIOE minus Eloundou, its standard error from the same fit. "
           r"Firm surveys: linear probability models with log 2019 employment; $N$ firms. Worker survey: survey-weighted, $N$ respondents linked to one employer. "
           r"The two top quartiles share most employers, so the joint fit is identified from those on which they disagree.",
           r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_adoption_by_index.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))

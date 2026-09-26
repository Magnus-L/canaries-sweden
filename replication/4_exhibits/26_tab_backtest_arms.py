#!/usr/bin/env python3
"""
26_tab_backtest_arms.py: Online Appendix Table A33 (Section IV.3), the as-of
backtest decomposed into sample inclusion and re-coding on common support
(script 98).

The occupation-classified design's post-launch coefficient at 22-25
(employer x exposure-quartile x month cells, 2019-2023; rate-hike and launch
interactions with the top quartile; employer-by-quartile and employer-by-
month effects; Poisson; clustered by employer), with pseudo-dates for
truncation year T of April T-1 and December T-1. Three arms: A, completed-
vintage codes on all workers; B, completed-vintage codes on the workers the
as-of cascade retains; C, as-of codes on the same workers as B. All three are
estimated on the employers common to them (the harmonised support), and the
differences B-A, C-B and C-A come from one stacked fit, so that B-A isolates
sample inclusion and C-B re-coding.

Nothing is written unless the three arms and the stacked fit share their
support (the stacked fit's retained observations are the sum of the three),
the differences equal the differences of the arm coefficients, every arm
agrees with the earlier export of the same run (2026-09-25_1832, which
lacks only the retained-observation counts), and every printed number
agrees with the run's own summary.

Exports read (3_register_mona/exports/):
  2026-09-26_1040_s98b/  backtest_common.csv, 98_summary.txt (the run whose
      export carries the observations each fit retained)
  2026-09-25_1832_s95-s96-s98/  backtest_common.csv (the same fits, exported
      before the retained counts were added; the second record)
Output: output/tables/tableIV4_backtest_arms.tex (a bare tabular and note;
        the appendix supplies the float and caption)

    python 4_exhibits/26_tab_backtest_arms.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S98B = EXPORTS / "2026-09-26_1040_s98b"
S98A = EXPORTS / "2026-09-25_1832_s95-s96-s98"
ARMS = ["A", "B", "C"]
DIFFS = ["B_minus_A", "C_minus_B", "C_minus_A"]
RE_LINE = re.compile(r"^\s*(A|B|C|B_minus_A|C_minus_B|C_minus_A)\s+harmonised\s+"
                     r"([-+][0-9.]+) \(([0-9.]+)\)")


def d() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else S98B


def need(folder: Path, name: str) -> Path:
    p = folder / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def harmonised(folder: Path) -> pd.DataFrame:
    b = pd.read_csv(need(folder, "backtest_common.csv"))
    b = b[(b.support == "harmonised") & b.status.isin(["ok", "derived"])]
    return b.set_index(["trunc", "arm"])


def summary(path: Path) -> dict:
    said, cut = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^CUTOFF (\d{4}):", line)
        if m:
            cut = int(m.group(1))
            continue
        m = RE_LINE.match(line)
        if m and cut:
            said[(cut, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return said


def main() -> int:
    print(f"  export {d().name}")
    b = harmonised(d())
    a = harmonised(S98A)
    said = summary(need(d(), "98_summary.txt"))
    truncs = sorted(set(t for t, _ in b.index))
    rows, retained = [], {}
    for t in truncs:
        est, ses = {}, {}
        for arm in ARMS + DIFFS:
            if (t, arm) not in b.index:
                raise SystemExit(f"  backtest_common.csv: no harmonised {arm} at {t}")
            r = b.loc[(t, arm)]
            est[arm], ses[arm] = float(r.gamma2), float(r.se2)
            if (t, arm) in a.index:
                ra = a.loc[(t, arm)]
                if abs(float(ra.gamma2) - est[arm]) > 1e-9 or abs(float(ra.se2) - ses[arm]) > 1e-9:
                    raise SystemExit(f"  {t} {arm}: the two exports of the run disagree")
            if round(est[arm], 4) != said[(t, arm)][0] or round(ses[arm], 4) != said[(t, arm)][1]:
                raise SystemExit(f"  {t} {arm}: the export gives {est[arm]:+.4f} "
                                 f"({ses[arm]:.4f}) and the summary {said[(t, arm)]}")
        for diff, (x, y) in (("B_minus_A", ("B", "A")), ("C_minus_B", ("C", "B")),
                             ("C_minus_A", ("C", "A"))):
            if abs((est[x] - est[y]) - est[diff]) > 1e-6:
                raise SystemExit(f"  {t} {diff}: the stacked difference is not the "
                                 f"difference of the arm coefficients")
        firms = set(int(b.loc[(t, arm)].n_firms) for arm in ARMS)
        if len(firms) != 1:
            raise SystemExit(f"  {t}: the arms do not share their employers, {firms}")
        used = {arm: int(b.loc[(t, arm)].cells_used) for arm in ARMS}
        cells = {arm: int(b.loc[(t, arm)].cells) for arm in ARMS}
        stacked = set(int(b.loc[(t, diff)].cells_used) for diff in DIFFS)
        if stacked != {sum(used.values())}:
            raise SystemExit(f"  {t}: the stacked fit retains {stacked} observations, "
                             f"the three arms {sum(used.values()):,}")
        if len(set(round(c / 1e6, 1) for c in cells.values())) != 1:
            raise SystemExit(f"  {t}: the arms' input cells differ at the first "
                             f"decimal of a million, {cells}")
        retained[t] = (used, list(cells.values())[0])
        n = f"{firms.pop():,}".replace(",", "{,}")
        rows.append(f"{t} & " + " & ".join(f"${est[k]:+.4f}$" for k in ARMS + DIFFS)
                    + f" & {n} \\\\")
        rows.append("     & " + " & ".join(f"({ses[k]:.4f})" for k in ARMS + DIFFS)
                    + r" & \\")
        print(f"  {t}: " + "  ".join(f"{k} {est[k]:+.4f} ({ses[k]:.4f})" for k in ARMS + DIFFS)
              + f"; {n} employers; retained {used}")
    print("  the summary and the earlier export reproduce every estimate")

    def mill(x: float) -> str:
        return f"{x / 1e6:.2f}"

    def mill1(x: float) -> str:
        return f"{x / 1e6:.1f}"

    if truncs != [2021, 2022]:
        raise SystemExit(f"  the note describes the 2021 and 2022 truncations, "
                         f"the export holds {truncs}")
    u21, c21 = retained[2021]
    u22, c22 = retained[2022]
    note = (
        r"The occupation-classified design's post-launch coefficient at 22--25 (employer $\times$ exposure-quartile $\times$ month cells, 2019--2023; rate-hike and launch interactions with the top quartile; employer-by-quartile and employer-by-month effects; Poisson; standard errors clustered by employer). Pseudo-dates for truncation year $T$: rate rise April $T-1$, launch December $T-1$, reference period before the rate rise. A: completed-vintage codes, all workers. B: completed-vintage codes, the workers the as-of cascade retains. C: as-of codes (each year's own register to $T$, then the files of $T$, $T-1$ and $T-2$), the same workers as B. All three arms are estimated on the employers common to them, and the differences come from one stacked fit, so B$-$A isolates sample inclusion and C$-$B re-coding. The panel is balanced and zero-filled, and the Poisson fit drops the cells of all-zero employer-by-quartile groups and the singletons they leave, about a third of the input: at the 2021 truncation the three arms retain "
        + f"{mill(u21['A'])}, {mill(u21['B'])} and {mill(u21['C'])} million of {mill1(c21)} million cells, at 2022 "
        + f"{mill(u22['A'])}, {mill(u22['B'])} and {mill(u22['C'])} million of {mill1(c22)} million"
        + r", and the stacked fit retains exactly the sum of the three.")
    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r" & \multicolumn{3}{c}{Arm} & \multicolumn{3}{c}{Difference} & \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Register truncated at & A & B & C & B$-$A & C$-$B & C$-$A & Employers \\",
           r"\midrule"]
    tex += rows
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableIV4_backtest_arms.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

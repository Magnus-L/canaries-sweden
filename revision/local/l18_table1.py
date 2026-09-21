#!/usr/bin/env python3
"""
l18_table1.py -- Table 1, assembled from the exports rather than by hand.

THE SPECIFICATION, from the writing pack: "Table 1 four rows: the two
bands, the artefact beside each, the female differential. Nothing else."

Everything comes from lane 14, the seasonally controlled design, which
supersedes 61 and 67 for anything quoted. Nothing here is typed in: a
cell that has no export prints as PENDING and says which fit is
missing, because a blank that looks like a zero is how a reader is
misled. The 26-30 stock row is the one currently outstanding -- its fit
crashed and is being recovered.

Writes tables/table1_headline.tex and prints the same thing as text.

    python3 revision/local/l18_table1.py [export_dir]
"""
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

TERM = "post_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"


def find(argv, name):
    roots = [Path(a) for a in argv[1:]] + [REV / "output"]
    best = None
    for r in roots:
        if r.exists():
            for p in list(r.rglob(name)) + list(r.rglob(f"*__{name}")):
                if best is None or p.stat().st_mtime > best.stat().st_mtime:
                    best = p
    return best


def cell(coef, se):
    if coef is None:
        return "PENDING"
    t = coef / se if se else float("nan")
    return f"{coef:+.4f} ({se:.4f})  t {t:+.2f}"


def main() -> int:
    pooled = find(sys.argv, "seasonal_pooled.csv")
    gender = find(sys.argv, "seasonal_gender.csv")
    if pooled is None:
        print("  no seasonal_pooled.csv; run lane 14 and export it")
        return 1
    print(f"  reading {pooled}")
    d = pd.read_csv(pooled)
    d = d[d.get("status", "ok") == "ok"]

    def grab(band, arm):
        r = d[(d.young_band == band) & (d.outcome == "stock")
              & (d.arm == arm) & (d.term == TERM)]
        if r.empty:
            return None, None
        return float(r.coef.iloc[0]), float(r.se.iloc[0])

    rows, missing = [], []
    for band in ("22-25", "26-30"):
        tc, ts = grab(band, "true")
        ac, asd = grab(band, "asof")
        art = (f"{ac - tc:+.4f}" if (tc is not None and ac is not None)
               else "--")
        if tc is None:
            missing.append(f"{band} stock, true arm")
        rows.append((f"{band}, employment stock", cell(tc, ts), art))

    if gender is not None:
        print(f"  reading {gender}")
        g = pd.read_csv(gender)
        g = g[(g.get("status", "ok") == "ok") & (g.term == FEMALE)]
        if not g.empty:
            rows.append(("22-25, female differential",
                         cell(float(g.coef.iloc[0]), float(g.se.iloc[0])),
                         "--"))
        else:
            missing.append("female differential")
    sep = d[(d.young_band == "22-25") & (d.outcome == "seps")
            & (d.arm == "true") & (d.term == TERM)]
    if not sep.empty:
        rows.append(("22-25, separations",
                     cell(float(sep.coef.iloc[0]), float(sep.se.iloc[0])),
                     "--"))

    w = max(len(r[0]) for r in rows)
    print()
    print(f"  {'':<{w}}  {'estimate':<26} artefact")
    for lab, est, art in rows:
        print(f"  {lab:<{w}}  {est:<26} {art}")
    if missing:
        print(f"\n  PENDING, not zero: {'; '.join(missing)}")

    V2_TAB.mkdir(parents=True, exist_ok=True)
    # A full float, not a bare tabular: the manuscript \input{}s this and
    # needs the caption, the label and the note to travel with the numbers.
    # A caption written by hand beside a generated table is how a table and
    # its description drift apart.
    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Employment of young workers relative to their older "
           r"colleagues inside the same employer, with the calendar cycle "
           r"removed.}",
           r"\label{tab:headline}",
           r"\begin{tabular}{lcc}", r"\toprule",
           r" & Estimate (SE) & Artefact \\", r"\midrule"]
    for lab, est, art in rows:
        e = "PENDING" if est == "PENDING" else est.split("  t")[0]
        tex.append(f"{lab} & {e} & {art} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.86\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson pseudo-maximum likelihood on employer $\times$ age "
            r"$\times$ month counts, exposure frozen at the employer's 2019 "
            r"education mix. Standard errors clustered by employer. The "
            r"artefact column reports the coefficient the as-of backtest "
            r"returns on the same specification when the register's lag is "
            r"imposed on years where the true gap is zero; the threshold "
            r"fixed before that test was 0.05.",
            r"\end{minipage}", r"\end{table}"]
    out = V2_TAB / "table1_headline.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

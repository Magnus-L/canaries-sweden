#!/usr/bin/env python3
"""
l25_tab_industry_credit.py: Online Appendix Table (tab:industry_credit),
two rival explanations for the decline of the young inside exposed
employers, on the OCCUPATION route.

WHY THIS SCRIPT CHANGED. Until 23 September 2026 it read lane 19's
industry_fe.csv and lane 24's credit_test.csv, both fitted on the
education route, where an employer is scored from the 2019 education mix
of its workforce. The v3 paper scores an employer from the 2019
three-digit occupations of its own incumbents aged 31 to 69, and lanes
28b and 29 refitted both tests on that score. The education-route
version of this builder is in the history of this repository; nothing
from it is quoted any more.

THE QUESTION
Could the within-employer age pattern be an industry-specific age shock (a
sector that stopped hiring the young for reasons unrelated to AI), or the
credit channel of the rate cycle (leveraged employers cutting their
youngest staff as borrowing costs rose)?

WHAT IS ESTIMATED
Poisson pseudo-maximum likelihood on employer by age by month counts with
employer-by-month, employer-by-age and month-by-age effects, exposure the
employment-weighted mean DAIOE percentile of the 2019 occupations of the
employer's own incumbents, treatment dated January 2024; the calendar
cycle is not removed, so every row is read against the baseline in the
same panel and never against Table 1.

  Panel A  The adoption step with and without three-digit industry (2019)
           interacted with age band and month, on the employers that carry
           a three-digit code (script 80, part C). Retained is the
           industry-controlled step over the baseline on those same
           employers; the rule fixed before the run was that the step
           survives at a band if it keeps its sign and retains at least
           50 per cent. It retains 67 per cent at 22-25 and 31 per cent at
           26-30, so it survives at the younger band and not at the older.
           A carried-forward industry code absorbs less than a
           contemporaneous one and so flatters the test, which is why the
           share coded from a register year other than 2019 is in the note.
  Panel B  On the employers with a 2019 balance sheet, the adoption step
           interacted with an indicator for above-median leverage (one
           minus equity over assets), beside a term for leverage x young
           common to every employer in the sample (script 73). The step
           averaged over the two halves is the exposure step plus half the
           additional term, with its standard error from the exported
           covariance, and it is compared with the baseline on the same
           sample. Read rule fixed before the run: the step is not a
           credit effect if it keeps its sign and at least 80 per cent of
           that baseline. The full-panel baseline is printed beside it as
           the check that the two lanes fit the same panel.

INPUTS AND OUTPUTS
Reads, from the lane pinned below: occ_rest_industry.csv (script 80, part
C), occ_rest_credit.csv with vcov_r73_lev_<band>.csv (script 73), and
83_summary.txt, which is the second record every printed estimate is
checked against. Writes revision/tables/tableA_industry_credit.tex and
copies it to canaries-sweden-paper/tables/.

    python3 revision/local/l25_tab_industry_credit.py [export_dir]

THE GATE
Nothing is written unless every coefficient and standard error in the
table reproduces the number 83_summary.txt prints for it, to the decimals
that file prints, and unless the retained share the export carries equals
the ratio of the two coefficients to within half a point. The coverage of
the balance-sheet sample and the median leverage quoted in the note are
parsed from the same file and checked against the values recorded here
when this script was written.

IN THE PAPER
Online Appendix III.2, Table tab:industry_credit; the retained shares and
the leverage coefficients are quoted in Section 3.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
# Lanes 28b and 29b-d, the occupation route: one firm score for every fit.
LANE = OUT / "round3_20260923-0655-lanes28b-29bcd"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30"]
BASE_SPEC = "baseline_same_sample"
IND_SPEC = "industry_age_month"
EXPO = "post_x_high_x_young"
TRIPLE = "post_x_high_x_young_x_lev"
LEV = "post_x_young_x_lev"
SAME = "baseline_on_balance_sheet_sample"
FULL = "full_panel_baseline"

# Quoted in the note and parsed from 83_summary.txt, the lines that read
# "73: leverage/22-25: 88,613 of 104,217 panel firms carry the covariate
# (85.0%)" and "73: leverage/22-25: split at 0.693 (ge), 50.0% high".
# The values below are what that file held when this script was written;
# the parsed numbers are checked against them.
COVERAGE = {"22-25": 85.0, "26-30": 84.3}
SPLIT = {"22-25": 0.693, "26-30": 0.689}

RE_IND = re.compile(
    r"^\s*(?P<band>\d\d-\d\d): baseline (?P<bc>[-+]?[\d.]+) \((?P<bse>[\d.]+)\)"
    r" on the same firms -> with industry x age x month"
    r" (?P<ic>[-+]?[\d.]+) \((?P<ise>[\d.]+)\), retained (?P<kept>\d+)%")
RE_IND_N = re.compile(
    r"^\s*(?P<n>[\d,]+) employers in both fits, (?P<share>[\d.]+)% of them"
    r" coded from a source other than \S+, in (?P<groups>\d+) industry groups")
RE_CREDIT = re.compile(
    r"^\s*(?P<band>\d\d-\d\d): baseline on the balance-sheet sample"
    r" (?P<sc>[-+]?[\d.]+) \((?P<sse>[\d.]+)\); the step with the leverage"
    r" split (?P<ec>[-+]?[\d.]+) \((?P<ese>[\d.]+)\), (?P<pct>\d+)% of it")
RE_TRIPLE = re.compile(
    r"the exposed-and-levered term (?P<tc>[-+][\d.]+) \((?P<tse>[\d.]+)\)")
RE_FULL = re.compile(
    r"the full-panel baseline at this band is (?P<fc>[-+]?[\d.]+)"
    r" \((?P<fse>[\d.]+)\)")
RE_COV = re.compile(
    r"73: leverage/(?P<band>\d\d-\d\d): [\d,]+ of [\d,]+ panel firms carry"
    r" the covariate \((?P<cov>[\d.]+)%\)")
RE_SPLIT = re.compile(
    r"73: leverage/(?P<band>\d\d-\d\d): split at (?P<med>[\d.]+) \(ge\)")


def lane() -> Path:
    """The pinned export, or a directory given on the command line."""
    return Path(sys.argv[1]) if len(sys.argv) > 1 else LANE


def need(name: str) -> Path:
    p = lane() / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def fmt(c: float, se: float) -> str:
    """One estimate as the appendix prints it, starred at five per cent."""
    star = "^{*}" if abs(c) > 1.96 * se else ""
    return f"${c:+.4f}{star}$ ({se:.4f})"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def agree(what: str, got: float, said: str) -> None:
    """The second record. `said` is the string 83_summary.txt prints, and
    the export is rounded to that many decimals before the comparison, so
    the table can carry more decimals than the summary does."""
    dp = len(said.split(".")[1]) if "." in said else 0
    if float(f"{got:.{dp}f}") != float(said):
        raise SystemExit(f"  {what}: the export gives {got:.6f} and "
                         f"83_summary.txt says {said}; nothing is written")


def read_summary() -> dict:
    """83_summary.txt, sections 7 and 8 and the two leverage notes, as the
    second record for every number this table prints."""
    text = need("83_summary.txt").read_text(encoding="utf-8")
    ind, credit, cov, split = {}, {}, {}, {}
    band = None
    for line in text.splitlines():
        m = RE_IND.match(line)
        if m:
            band = m.group("band")
            ind[band] = dict(m.groupdict())
            continue
        m = RE_IND_N.match(line)
        if m and band in ind:
            ind[band].update(m.groupdict())
            continue
        m = RE_CREDIT.match(line)
        if m:
            band = m.group("band")
            credit[band] = dict(m.groupdict())
            continue
        for rx in (RE_TRIPLE, RE_FULL):
            m = rx.search(line)
            if m and band in credit:
                credit[band].update(m.groupdict())
        m = RE_COV.search(line)
        if m:
            cov[m.group("band")] = float(m.group("cov"))
        m = RE_SPLIT.search(line)
        if m:
            split[m.group("band")] = float(m.group("med"))
    for b in BANDS:
        for name, d, keys in (("section 7", ind, ("bc", "n", "groups")),
                              ("section 8", credit, ("sc", "tc", "fc"))):
            if b not in d or any(k not in d[b] for k in keys):
                raise SystemExit(f"  83_summary.txt: {name} does not parse "
                                 f"for {b}; nothing is written")
        if cov.get(b) != COVERAGE[b] or split.get(b) != SPLIT[b]:
            raise SystemExit(
                f"  83_summary.txt: the leverage note for {b} now says "
                f"coverage {cov.get(b)} and median {split.get(b)}, against "
                f"the {COVERAGE[b]} and {SPLIT[b]} this script was written "
                f"on; the note would be wrong, so nothing is written")
    return {"industry": ind, "credit": credit}


def panel_a(said: dict) -> dict:
    """The industry test, checked against section 7 of the summary."""
    d = pd.read_csv(need("occ_rest_industry.csv"))
    d = d[(d.term == EXPO) & (d.get("status", "ok") == "ok")]
    a = {}
    for band in BANDS:
        s = said["industry"][band]
        b = d[(d.young_band == band) & (d.spec == BASE_SPEC)]
        i = d[(d.young_band == band) & (d.spec == IND_SPEC)]
        for name, r in (("baseline", b), ("industry", i)):
            if len(r) != 1:
                raise SystemExit(f"  occ_rest_industry.csv: {len(r)} "
                                 f"{name} rows for {band}, one expected")
        b, i = b.iloc[0], i.iloc[0]
        agree(f"{band} baseline", float(b.coef), s["bc"])
        agree(f"{band} baseline SE", float(b.se), s["bse"])
        agree(f"{band} with industry", float(i.coef), s["ic"])
        agree(f"{band} with industry SE", float(i.se), s["ise"])

        # The retained share is taken from the export and not recomputed,
        # but it must be the ratio of the two coefficients all the same.
        kept = 100.0 * float(i.retained_share)
        ratio = 100.0 * float(i.coef) / float(b.coef)
        if abs(kept - ratio) > 0.5:
            raise SystemExit(f"  {band}: the exported retained share "
                             f"{kept:.2f} is not the ratio of the two "
                             f"coefficients ({ratio:.2f})")
        agree(f"{band} retained", kept, s["kept"])
        n_firms = int(b.n_firms)
        n_groups = int(b.n_groups)
        share = 100.0 * float(b.share_not_from_2019)
        if n_firms != int(i.n_firms) or n_groups != int(i.n_groups):
            raise SystemExit(f"  {band}: the two fits do not run on the "
                             f"same employers or the same industry groups")
        if n_firms != int(s["n"].replace(",", "")):
            raise SystemExit(f"  {band}: the export holds {n_firms} "
                             f"employers and the summary says {s['n']}")
        if n_groups != int(s["groups"]):
            raise SystemExit(f"  {band}: the export holds {n_groups} "
                             f"industry groups and the summary says "
                             f"{s['groups']}")
        agree(f"{band} share not from 2019", share, s["share"])
        a[band] = {"base": (float(b.coef), float(b.se)),
                   "ind": (float(i.coef), float(i.se)),
                   "kept": kept, "n_firms": n_firms, "n_groups": n_groups,
                   "share": share}
    return a


def panel_b(said: dict) -> dict:
    """The credit test, checked against section 8 of the summary."""
    d = pd.read_csv(need("occ_rest_credit.csv"))
    b = {}
    for band in BANDS:
        s = said["credit"][band]
        c = d[d.band == band]
        for term in (SAME, EXPO, TRIPLE, LEV, FULL):
            if (c.term == term).sum() != 1:
                raise SystemExit(f"  occ_rest_credit.csv: one {term} row "
                                 f"expected for {band}")
        c = c.set_index("term")
        agree(f"{band} same-sample baseline", float(c.loc[SAME, "coef"]), s["sc"])
        agree(f"{band} same-sample baseline SE", float(c.loc[SAME, "se"]), s["sse"])
        agree(f"{band} less leveraged half", float(c.loc[EXPO, "coef"]), s["ec"])
        agree(f"{band} less leveraged half SE", float(c.loc[EXPO, "se"]), s["ese"])
        agree(f"{band} exposed and levered", float(c.loc[TRIPLE, "coef"]), s["tc"])
        agree(f"{band} exposed and levered SE", float(c.loc[TRIPLE, "se"]), s["tse"])
        agree(f"{band} full-panel baseline", float(c.loc[FULL, "coef"]), s["fc"])
        agree(f"{band} full-panel baseline SE", float(c.loc[FULL, "se"]), s["fse"])

        v = pd.read_csv(need(f"vcov_r73_lev_{band}.csv"), index_col=0)
        for term in (EXPO, TRIPLE):
            diag = float(v.loc[term, term]) ** 0.5
            if abs(diag - float(c.loc[term, "se"])) > 5e-5:
                raise SystemExit(f"  {band} {term}: the exported standard "
                                 f"error {float(c.loc[term, 'se']):.6f} is "
                                 f"not the square root of its own variance "
                                 f"{diag:.6f}")
        # The step averaged over the two halves: the exposure step plus
        # half the additional term, its variance from the same covariance.
        avg = float(c.loc[EXPO, "coef"]) + 0.5 * float(c.loc[TRIPLE, "coef"])
        avg_se = float(np.sqrt(v.loc[EXPO, EXPO] + 0.25 * v.loc[TRIPLE, TRIPLE]
                               + v.loc[EXPO, TRIPLE]))
        b[band] = {"same": (float(c.loc[SAME, "coef"]), float(c.loc[SAME, "se"])),
                   "full": (float(c.loc[FULL, "coef"]), float(c.loc[FULL, "se"])),
                   "expo": (float(c.loc[EXPO, "coef"]), float(c.loc[EXPO, "se"])),
                   "triple": (float(c.loc[TRIPLE, "coef"]), float(c.loc[TRIPLE, "se"])),
                   "avg": (avg, avg_se),
                   "lev": (float(c.loc[LEV, "coef"]), float(c.loc[LEV, "se"])),
                   "kept": 100.0 * avg / float(c.loc[SAME, "coef"]),
                   "kept_expo": 100.0 * float(c.loc[EXPO, "coef"])
                   / float(c.loc[SAME, "coef"]),
                   "said_pct": int(s["pct"])}
    return b


def main() -> int:
    said = read_summary()
    print(f"  lane {lane().name}")
    a = panel_a(said)
    b = panel_b(said)
    print("  83_summary.txt reproduces every printed estimate")

    def row(label: str, key: str, src: dict) -> str:
        cells = [fmt(*src[band][key]) for band in BANDS]
        print(f"    {label[:48]:48s} " + "  ".join(f"{c:>24s}" for c in cells))
        return f"{label} & " + " & ".join(cells) + r" \\"

    def plain(label: str, values: list) -> str:
        print(f"    {label[:48]:48s} " + "  ".join(f"{v:>24s}" for v in values))
        return f"{label} & " + " & ".join(values) + r" \\"

    print("  Panel A. An industry-specific age shock")
    rows_a = [row("Baseline", "base", a),
              row(r"With industry $\times$ age $\times$ month", "ind", a),
              plain("Retained (per cent)",
                    [f"{a[band]['kept']:.0f}" for band in BANDS]),
              plain("Employers",
                    [thousands(a[band]["n_firms"]) for band in BANDS]),
              plain("Three-digit industry groups",
                    [f"{a[band]['n_groups']}" for band in BANDS])]
    for band in BANDS:
        verdict = "SURVIVES" if a[band]["kept"] >= 50 else "DOES NOT SURVIVE"
        print(f"      {band}: industry retains {a[band]['kept']:.0f} per cent, "
              f"{verdict} the 50 per cent rule")

    print("  Panel B. The credit channel")
    rows_b = [row("Baseline on this sample", "same", b),
              row("Baseline, full panel", "full", b),
              row("Adoption step, less leveraged half", "expo", b),
              row("Additional step, more leveraged half", "triple", b),
              row("Adoption step averaged over the halves", "avg", b),
              row(r"Leverage $\times$ young, all employers in the sample",
                  "lev", b),
              plain("Retained, averaged step over baseline (per cent)",
                    [f"{b[band]['kept']:.0f}" for band in BANDS])]
    for band in BANDS:
        verdict = "NOT A CREDIT EFFECT" if b[band]["kept"] >= 80 else "MONETARY"
        print(f"      {band}: the averaged step keeps {b[band]['kept']:.0f} "
              f"per cent of the same-sample baseline, {verdict}; the "
              f"unaveraged step of the less leveraged half keeps "
              f"{b[band]['kept_expo']:.0f} per cent, which is the "
              f"{b[band]['said_pct']} per cent 83_summary.txt reports")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{An industry-specific age shock and the credit channel as "
           r"rival explanations for the decline of the young inside exposed "
           r"employers.}",
           r"\label{tab:industry_credit}", r"\footnotesize",
           r"\begin{tabular}{lcc}", r"\toprule",
           r" & 22--25 & 26--30 \\", r"\midrule",
           r"\multicolumn{3}{l}{\textit{Panel A. An industry-specific age "
           r"shock}} \\"]
    tex += rows_a
    tex += [r"\addlinespace[4pt]",
            r"\multicolumn{3}{l}{\textit{Panel B. The credit channel, "
            r"employers with a 2019 balance sheet}} \\"]
    tex += rows_b
    tex += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.9\textwidth}\footnotesize\vspace{4pt}",
        r"Poisson with the paper's three fixed effects, exposure the "
        r"employer's 2019 occupation mix, treatment January 2024, clustered "
        r"by employer. The calendar cycle is not removed, so each row reads "
        r"against the baseline in its own panel. Industry is the employer's "
        r"three-digit NACE in 2019, interacted with age band and month, and "
        r"both fits of Panel~A run on the employers that carry such a code; "
        f"{a['22-25']['share']:.1f} per cent of them at 22--25 and "
        f"{a['26-30']['share']:.1f} per cent at 26--30 are coded from a "
        r"register year other than 2019, and a carried-forward code absorbs "
        r"less. Retained is the industry-controlled step over the baseline on "
        r"the same employers. Leverage is one minus equity over assets from "
        r"the 2019 balance sheet (Serrano), held by "
        f"{COVERAGE['22-25']:.1f} and {COVERAGE['26-30']:.1f} per cent of the "
        r"two panels and split at the median "
        f"({SPLIT['22-25']:.3f} and {SPLIT['26-30']:.3f}). "
        r"The balance-sheet sample is limited companies, so the public sector "
        r"and the unincorporated are outside it. Credit rows: the baseline on "
        r"the balance-sheet employers and on the full panel, the step among "
        r"the less leveraged exposed employers, the additional step among the "
        r"more leveraged, their average (the first plus half the second) "
        r"against the same-sample baseline, and leverage $\times$ young, the "
        r"channel common to all. "
        r"$^{*}$ $p<0.05$. Source: scripts 80 (part C) and 73.",
        r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_industry_credit.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

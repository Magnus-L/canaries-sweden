#!/usr/bin/env python3
"""
15_tab_gender_split.py: Online Appendix Table A15 (Section III.2), the female
differential at ages 22-25 by education track, and its split into a
within-track and a composition component.

Equation (2)'s sex specification (script 87): Poisson on employer by age by
sex by month counts with employer-by-month, employer-by-age-and-sex and
month-by-age-and-sex effects and three quarter-of-year terms, exposure the
employer's 2019 occupation mix, treatment January 2024, clustered by employer;
on all workers and separately inside each broad track (two-digit SUN 2020
fields of the worker's own education, 2023 record carried forward to 2024 and
2025). Education cuts the sample and does not measure exposure. The split
weights the five track differentials by young women's track shares in
top-quartile employers in 2023 and reads the residual as composition; the
within-track standard error treats the track fits as independent.

Nothing is written unless every track is exported once with status ok, every
standard error equals the square root of its own covariance diagonal, the
split's pooled differential is the all-worker differential, the composition
residual and the ratio satisfy their identities, the weights are the exported
shares, and every row matches the run's own summary.

Exports read: 3_register_mona/exports/2026-09-23_1352_s87/
  occ_route_gender_by_track.csv, occ_route_gender_split.csv,
  vcov_s87_<track>.csv, occ_route_education_mix_by_sex.csv, 87_summary.txt
Output: output/tables/tableA_gender_split.tex

    python 4_exhibits/15_tab_gender_split.py [export_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from config import EXPORTS, TABLES  # noqa: E402

S87 = EXPORTS / "2026-09-23_1352_s87"

# Export key, row label. "all" carries no weight; the five tracks do.
TRACKS = [
    ("all", "All workers"),
    ("ict", "ICT"),
    ("engineering", "Engineering, manufacturing, construction"),
    ("business_law_social", "Business, law, administration, social science"),
    ("health_education_care", "Health, education, social care"),
    ("other", "Other fields"),
]
WEIGHTED = [k for k, _ in TRACKS if k != "all"]
MALE = "post_x_high_x_young"
DIFF = "post_x_high_x_young_x_female"
CELL = re.compile(r"^\$([-+][0-9.]+)\$(\$\^\{\*\}\$)? \(([0-9.]+)\)$")
# One track line of the run's summary.
SUMMARY_ROW = re.compile(
    r"^\s+(\S+)\s+men ([-+][0-9.]+) \(([0-9.]+)\)\s+women minus men "
    r"([-+][0-9.]+) \(([0-9.]+)\) t ([-+][0-9.]+)\s+firms ([0-9,]+)\s*$")
SUMMARY_SPLIT = re.compile(
    r"^\s+(pooled differential|within tracks|composition(?: \(residual\))?|"
    r"ratio within / pooled)\s+([-+]?[0-9.]+)(?: \(([0-9.]+)\))?")


def source(default_dir: Path, name: str) -> Path:
    """The pinned export, or the same file name under a directory given
    on the command line."""
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def cell(what: str, c: float, se: float) -> str:
    """The estimate with a star at five per cent, checked against the
    export before it is allowed into the table."""
    star = "$^{*}$" if abs(c) > 1.96 * se else ""
    out = f"${c:+.4f}${star} ({se:.4f})"
    m = CELL.match(out)
    if m is None:
        raise SystemExit(f"  {what}: the formatted estimate is unreadable")
    if abs(float(m.group(1)) - c) > 5e-5 or abs(float(m.group(3)) - se) > 5e-5:
        raise SystemExit(f"  {what}: the printed estimate {out} disagrees "
                         f"with the export ({c:+.6f}, {se:.6f})")
    return out


def summary(path: Path) -> tuple[dict, dict]:
    """The run's own report of the same fits: one line per track, and the
    four lines of the split. This is a second record, written by the run
    rather than derived from the CSV."""
    rows, split = {}, {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SUMMARY_ROW.match(line)
        if m:
            track, mc, mse, dc, dse, t, firms = m.groups()
            rows[track] = (f"${mc}$ ({mse})", f"${dc}$ ({dse})", float(t),
                           int(firms.replace(",", "")))
            continue
        s = SUMMARY_SPLIT.match(line)
        if s:
            key = s.group(1)
            split["composition" if key.startswith("composition")
                  else key] = (s.group(2), s.group(3))
    if not rows or len(split) != 4:
        raise SystemExit(f"  {path.name}: does not report the tracks and the "
                         f"four lines of the split")
    return rows, split


def main() -> int:
    d = pd.read_csv(source(S87, "occ_route_gender_by_track.csv"))
    bad = [k for k, _ in TRACKS if (d.track == k).sum() != 1]
    if bad or len(d) != len(TRACKS):
        raise SystemExit(f"  occ_route_gender_by_track.csv: one row expected per "
                         f"track, "
                         f"wrong for {bad or 'the row count'}")
    if not (d.status == "ok").all():
        raise SystemExit(f"  occ_route_gender_by_track.csv: a track reports a status "
                         f"other than ok, so it is not quotable")
    d = d.set_index("track")

    # Each exported standard error must be the square root of its own
    # variance in the covariance matrix the same fit exported.
    for key, _ in TRACKS:
        v = pd.read_csv(source(S87, f"vcov_s87_{key}.csv"), index_col=0)
        for term, col in ((MALE, "male_se"), (DIFF, "diff_se")):
            diag = float(v.loc[term, term]) ** 0.5
            got = float(d.loc[key, col])
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {key} {term}: the exported standard error "
                                 f"{got:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")

    # The weights: young women's track shares in top-quartile employers.
    mix = pd.read_csv(source(S87, "occ_route_education_mix_by_sex.csv"))
    mix = mix[(mix.dimension == "track") & (mix.exposed == 1)
              & (mix.gender == "women")].set_index("cell")
    weights = {k: float(mix.loc[k, "share"]) for k in WEIGHTED}

    # The split, and the identities it must satisfy.
    s = pd.read_csv(source(S87, "occ_route_gender_split.csv"))
    if len(s) != 1:
        raise SystemExit(f"  occ_route_gender_split.csv: one split expected, "
                         f"found {len(s)}")
    s = s.iloc[0]
    pooled, pooled_se = float(s.pooled), float(s.pooled_se)
    within, within_se = float(s.within), float(s.within_se)
    composition, ratio = float(s.composition), float(s.ratio_within)
    if abs(pooled - float(d.loc["all", "diff"])) > 5e-5:
        raise SystemExit(f"  the split's pooled differential {pooled:+.6f} is "
                         f"not the all-worker differential "
                         f"{float(d.loc['all', 'diff']):+.6f}, so the two "
                         f"exports are not from the same fit")
    if abs(pooled - (within + composition)) > 5e-5:
        raise SystemExit(f"  the composition residual {composition:+.6f} is "
                         f"not the pooled differential less the within-track "
                         f"figure ({pooled - within:+.6f})")
    if abs(ratio - within / pooled) > 5e-4:
        raise SystemExit(f"  the exported ratio {ratio:.4f} is not the "
                         f"within-track figure over the pooled one "
                         f"({within / pooled:.4f})")
    over = [t.strip() for t in str(s.weights_renormalised_over).split(",")]
    if sorted(over) != sorted(WEIGHTED):
        raise SystemExit(f"  the split renormalises its weights over {over}, "
                         f"not over the five tracks the table prints")

    said, said_split = summary(source(S87, "87_summary.txt"))
    for key, quoted in (("pooled differential", f"{pooled:+.4f}"),
                        ("within tracks", f"{within:+.4f}"),
                        ("composition", f"{composition:+.4f}"),
                        ("ratio within / pooled", f"{ratio:.3f}")):
        if said_split[key][0] != quoted:
            raise SystemExit(f"  the split's '{key}' is {quoted} in the CSV "
                             f"and {said_split[key][0]} in the run's summary")

    rows = []
    for key, label in TRACKS:
        r = d.loc[key]
        male = cell(f"{key} men", float(r.male), float(r.male_se))
        diff = cell(f"{key} women minus men", float(r["diff"]), float(r.diff_se))
        n = int(r.n_firms)
        if key not in said:
            raise SystemExit(f"  {key}: the run's summary does not report it")
        s_male, s_diff, s_t, s_firms = said[key]
        if male.replace("$^{*}$", "") != s_male or diff.replace("$^{*}$", "") != s_diff:
            raise SystemExit(f"  {key}: the table would print {male} and "
                             f"{diff}, the run's summary reports {s_male} and "
                             f"{s_diff}")
        if s_firms != n:
            raise SystemExit(f"  {key}: the summary reports {s_firms:,} "
                             f"employers and the CSV {n:,}")
        if abs(float(r["diff"]) / float(r.diff_se) - s_t) > 5e-3:
            raise SystemExit(f"  {key}: the summary's t {s_t} is not the "
                             f"differential over its standard error "
                             f"({float(r['diff']) / float(r.diff_se):+.4f})")
        w = f"{weights[key]:.3f}" if key in weights else ""
        rows.append(f"{label} & {male} & {diff} & {w} & {n:,} \\\\")
        print(f"  {label[:44]:44s} {male:>24s} {diff:>24s} {w:>5s} {n:>9,}")
    print(f"  split: pooled {pooled:+.4f}, within {within:+.4f}, "
          f"composition {composition:+.4f}, ratio {ratio:.3f}")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The female differential at 22--25 within education tracks, "
           r"and its weighted average.}",
           r"\label{tab:gender_split}", r"\scriptsize",
           # The track labels are long; a narrower column gap keeps the
           # table inside the text block.
           r"\setlength{\tabcolsep}{3.5pt}",
           r"\begin{tabular}{lcccr}", r"\toprule",
           r"Track & Young men & Young women minus men & Weight & Employers \\",
           r"\midrule"]
    tex += rows
    tex += [r"\midrule",
            r"\multicolumn{5}{l}{\textit{The split (pooled differential $=$ "
            r"within $+$ composition)}} \\",
            "Pooled differential & & "
            + cell("pooled", pooled, pooled_se) + r" & & \\",
            "Within tracks (weighted) & & "
            + cell("within", within, within_se) + r" & & \\",
            f"Pooled minus weighted within-track estimate & & ${composition:+.4f}$" + r" & & \\",
            f"Ratio within / pooled & & {ratio:.2f}" + r" & & \\",
            r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"The sex specification of Equation~(2), with sex-specific employer-by-age and month-by-age effects, calendar cycle removed, clustered by employer, on all workers and within each broad track of the worker's own education (two-digit SUN 2020 fields, carried forward from 2023). Exposure is the employer's 2019 occupation mix; education only cuts the sample. Young men is the step from the tightening months; young women minus men is its female interaction. Weights are young women's track shares in top-quartile employers in 2023. $^{*}$ $p<0.05$.",
            r"\end{minipage}", r"\end{table}"]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tableA_gender_split.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(PACKAGE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

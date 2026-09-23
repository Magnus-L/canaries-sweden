#!/usr/bin/env python3
"""
l29_tab_gender_split.py: Online Appendix Table III.2
(tab:gender_split), the female differential at ages 22 to 25 by education
track, and its split into a within-track and a composition component.

THE QUESTION
Young women inside exposed employers lose ground relative to young men
after adoption. Two things could produce that. Young women may hold
different educations from young men, so that the sexes sit in different
parts of the labour market and the differential is composition; or young
women may lose ground against young men who hold the same broad
education, which is a differential within tracks. The table runs the
gender specification inside each track and splits the pooled differential
between the two.

WHAT IS ESTIMATED
Script 68's gender specification, re-estimated by script 76 (lane 22):
Poisson pseudo-maximum likelihood on employer by age by sex by month
counts, with employer-by-month, employer-by-age-and-sex and
month-by-age-and-sex effects, three quarter-of-year terms so that the
calendar cycle is removed, exposure frozen at the employer's 2019
education mix, treatment dated January 2024, and standard errors
clustered by employer. Young men is the post by exposure by young term;
young women minus men is its female interaction. The same fit is run on
all workers and then separately inside each broad education track, so
that young women are compared with young men holding the same broad
education, against their same-track older colleagues.

The split weights the five track differentials by young women's track
shares in top-quartile employers in 2023 and reads the residual as
composition. The weighted within-track figure treats the five track fits
as independent, which they are not exactly, so its standard error is a
working approximation and is reported as such. The reporting rule was
fixed before the run: a within-track share at or below a half reads as
composition, at or above three quarters as a differential within tracks.

Tracks are broad two-digit SUN 2020 fields and do not hold job content
constant, so the split bounds composition rather than identifying a
mechanism. The education record is the worker's own, and the individual
register ends in 2023, so 2024 and 2025 carry the 2023 record forward;
that is the reason the tracks are broad.

INPUTS AND OUTPUTS
Reads, from the export directory the final-code manifest names or one
given on the command line: gender_by_track.csv (script 76; the male
effect and the female differential per track, with employer counts and a
status), gender_split.csv (the same run's split of the pooled
differential), vcov_s76_<track>.csv (the exported covariance of each
track fit), education_mix_by_sex.csv (the track shares that are the
weights) and 76_summary.txt, the run's own report of the same fits.
Nothing is typed in. Writes revision/tables/tableA_gender_split.tex and
copies it to canaries-sweden-paper/tables/.

    python3 revision/local/l29_tab_gender_split.py [export_dir]

THE GATE
Every track must be exported once with status ok, and each exported
standard error must equal the square root of its own diagonal in that
track's covariance matrix. The pooled differential of the split must be
the all-worker differential of the track file, which is the run's own
reproduction gate; the composition residual must equal the pooled figure
minus the within-track figure, and the ratio must equal the within-track
figure over the pooled one. The weights must be the track shares of young
women in top-quartile employers that the mix export reports, and the
tracks they are renormalised over must be the five the table prints.
Every row is then checked against the run's summary, which reports the
same estimates, their t ratios and their employer counts independently of
the CSV. Finally each printed number is read back from the string that
goes into the table and compared with the export it came from. Any
disagreement beyond half of the last printed digit stops the script and
nothing is written.

IN THE PAPER
Online Appendix III.2, the paragraph on the female differential split,
Table tab:gender_split; the pooled differential and the within-track
figure are quoted in Section 3.
"""
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE22 = OUT / "round3_20260922-0712-lanes21-22"
# The manuscript repository is a sibling of this one; the paper \input{}s
# the table from there.
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

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
    r"^\s+(pooled differential|within tracks|composition \(residual\)|"
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
            split[s.group(1)] = (s.group(2), s.group(3))
    if not rows or len(split) != 4:
        raise SystemExit(f"  {path.name}: does not report the tracks and the "
                         f"four lines of the split")
    return rows, split


def main() -> int:
    d = pd.read_csv(source(LANE22, "gender_by_track.csv"))
    bad = [k for k, _ in TRACKS if (d.track == k).sum() != 1]
    if bad or len(d) != len(TRACKS):
        raise SystemExit(f"  gender_by_track.csv: one row expected per track, "
                         f"wrong for {bad or 'the row count'}")
    if not (d.status == "ok").all():
        raise SystemExit(f"  gender_by_track.csv: a track reports a status "
                         f"other than ok, so it is not quotable")
    d = d.set_index("track")

    # Each exported standard error must be the square root of its own
    # variance in the covariance matrix the same fit exported.
    for key, _ in TRACKS:
        v = pd.read_csv(source(LANE22, f"vcov_s76_{key}.csv"), index_col=0)
        for term, col in ((MALE, "male_se"), (DIFF, "diff_se")):
            diag = float(v.loc[term, term]) ** 0.5
            got = float(d.loc[key, col])
            if abs(diag - got) > 5e-5:
                raise SystemExit(f"  {key} {term}: the exported standard error "
                                 f"{got:.6f} is not the square root of its "
                                 f"own variance {diag:.6f}")

    # The weights: young women's track shares in top-quartile employers.
    mix = pd.read_csv(source(LANE22, "education_mix_by_sex.csv"))
    mix = mix[(mix.dimension == "track") & (mix.exposed == 1)
              & (mix.gender == "women")].set_index("cell")
    weights = {k: float(mix.loc[k, "share"]) for k in WEIGHTED}

    # The split, and the identities it must satisfy.
    s = pd.read_csv(source(LANE22, "gender_split.csv"))
    if len(s) != 1:
        raise SystemExit(f"  gender_split.csv: one split expected, found {len(s)}")
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

    said, said_split = summary(source(LANE22, "76_summary.txt"))
    for key, quoted in (("pooled differential", f"{pooled:+.4f}"),
                        ("within tracks", f"{within:+.4f}"),
                        ("composition (residual)", f"{composition:+.4f}"),
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
           r"\caption{The female differential at 22--25 by education track, "
           r"and its split into composition and within-track components.}",
           r"\label{tab:gender_split}", r"\footnotesize",
           # The track labels are long and the table ran past the text
           # block in both the online appendix and its v2 predecessor,
           # losing the right-hand column off the page.
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
            f"Composition (residual) & & ${composition:+.4f}$" + r" & & \\",
            f"Ratio within / pooled & & {ratio:.2f}" + r" & & \\",
            r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"Script 68's gender specification (Poisson; employer-by-month, "
            r"employer-by-age-and-sex and month-by-age-and-sex effects; "
            r"calendar cycle removed; treatment January 2024; clustered by "
            r"employer), on all workers and within each broad education track. "
            r"Young men is the post $\times$ high $\times$ young term; young "
            r"women minus men is its female interaction. Tracks are two-digit "
            r"SUN 2020 fields: ICT is data; engineering is technology, "
            r"materials and construction; business is social science, "
            r"business, administration and law; health is teaching, health "
            r"care and social work; other is the rest. Weights are young "
            r"women's track shares in top-quartile employers in 2023; the "
            r"within-track standard error treats the track fits as "
            r"independent. The education record is the worker's own, carried "
            r"forward from 2023 in 2024 and 2025. $^{*}$ $p<0.05$. "
            r"Source: script 76.",
            r"\end{minipage}", r"\end{table}"]

    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_gender_split.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

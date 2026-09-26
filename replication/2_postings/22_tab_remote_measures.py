#!/usr/bin/env python3
"""
22_tab_remote_measures.py: Online Appendix Table A4 (Section II.3), AI
exposure against three measures of remote work on the posting margin.

WHAT IT BUILDS
One table setting three remote-work measures side by side against AI
exposure: the Dingel and Neiman (2020) teleworkability classification (a
measure of feasibility, split at its median, Equation (1) in each half, from
19); the share of an occupation's 2021 to 2022 Platsbanken advertisements
that offer remote or hybrid work, read from the advertisement text; and the
same share in United States postings (Hansen et al. 2023, the WFH Map
release), crosswalked to SSYK. Panel A is Lambert and Schindler's layout
across occupations, one post-launch dummy interacted with each standardised
score, alone and together; Panel B is the within-employer design of Part V
with each employer's cells cut by the top quartile of AI exposure and of
remote work, and the split of employers by whether their own 2021 to 2022
advertisements offered remote work; Panel C is Panel A on entry-level
advertisements.

NOTHING IS ESTIMATED HERE. The Platsbanken and Hansen columns are read from
the result files of the two estimation scripts in extensions/, which ran in
the authors' research repository on inputs the package does not ship (the
advertisement text of every archive, the employer-level advertisement
counts of Part V, and the WFH Map file); extensions/README.md states what
each needs. The Dingel and Neiman column is read from 19's result on the
current window. Every cell of the table is also written to a CSV.

INPUTS   2_postings/extensions/results/l50_remote_{horserace,within,correlations}.csv,
         l52_hansen_{horserace,within,correlations}.csv;
         output/results/telework_did_results_v3.csv (19)
OUTPUTS  output/tables/tableA_remote_measures.tex;
         output/results/remote_measures_cells.csv
SERVES   Online Appendix II.3, Table A4, and the remote-work paragraphs of
         Section 3 of the paper
RUNTIME  seconds
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config  # noqa: E402

import pandas as pd  # noqa: E402

RESULTS, TABLES = config.RESULTS, config.TABLES
EXT = Path(__file__).resolve().parent / "extensions" / "results"


def stars(p):
    """Significance stars on the paper's convention (10/5/1 per cent)."""
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def pick(df, **kw):
    """Return exactly one row matching every column=value pair, or fail loudly."""
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    hit = df[m]
    if len(hit) != 1:
        raise SystemExit(f"  {kw}: {len(hit)} rows, one expected")
    return hit.iloc[0]


def cell(r):
    """(coef, se, p) triple from a result row."""
    return (float(r["coef"]), float(r["se"]), float(r["pval"]))


def need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"  missing input: {path}")
    return pd.read_csv(path)


def main():
    print("The remote-work measures against AI exposure (Table A4)")
    l50h = need(EXT / "l50_remote_horserace.csv")
    l50w = need(EXT / "l50_remote_within.csv")
    l50c = need(EXT / "l50_remote_correlations.csv")
    l52h = need(EXT / "l52_hansen_horserace.csv")
    l52w = need(EXT / "l52_hansen_within.csv")
    l52c = need(EXT / "l52_hansen_correlations.csv")
    dn = need(RESULTS / "telework_did_results_v3.csv")
    dn = dn[dn["outcome"] == "ln(ads+1)"].set_index("group")

    # The two estimation scripts share their AI-only baselines with 03 and
    # 14; those baselines are the check that they ran on the paper's panels.
    l08 = need(RESULTS / "postings_extended_did.csv").set_index(["window", "estimator", "term"])
    for h, name in ((l50h, "l50"), (l52h, "l52")):
        b = pick(h, design="occupation_panel", sample="all ads", spec="baseline", term="gpt_x_high")
        ref = l08.loc[("extended_to_2026-06", "OLS_ln", "gpt_x_high"), "coef"]
        if abs(float(b.coef) - ref) > 1e-8:
            raise SystemExit(f"  {name}: its baseline {float(b.coef):.6f} is not 03's {ref:.6f}")
    firm = need(RESULTS / "firm_within_did.csv")
    ref = float(pick(firm, variant="a_all_firms", term="gpt_x_high").coef)
    for w, name in ((l50w, "l50"), (l52w, "l52")):
        b = pick(w, design="within_employer", sample="all ads", spec="l09_a_baseline", term="gpt_x_high")
        if abs(float(b.coef) - ref) > 1e-8:
            raise SystemExit(f"  {name}: its within-employer baseline {float(b.coef):.6f} is not 14's {ref:.6f}")
    print("  CHECK: both estimation scripts reproduce 03's and 14's baselines")

    # Correlation with DAIOE across the 369 panel occupations, weighted by
    # 2024 employment: how much room there is to separate the two at all.
    corr = {
        "dn": float(pick(l50c, pair="DAIOE pctl_rank_genai x Dingel-Neiman teleworkable")["employment_weighted_2024"]),
        "pb": float(pick(l50c, pair="remote_share x DAIOE pctl_rank_genai")["employment_weighted_2024"]),
        "h": float(pick(l52c, pair="Hansen H1 x DAIOE pctl_rank_genai")["employment_weighted_2024"]),
    }

    occ = dict(design="occupation_panel", sample="all ads")
    ent = dict(design="occupation_panel", sample="entry-level ads")
    wall = dict(design="within_employer", sample="all ads")
    rows = []  # (panel, label, dn, pb, h); each cell None or (coef, se, p)

    def add(panel, label, dn_=None, pb=None, h=None):
        rows.append((panel, label, dn_, pb, h))

    dn_lo, dn_hi = dn.loc["Non-teleworkable"], dn.loc["Teleworkable"]
    add("A", "Below the median of the measure", dn_=(dn_lo.beta2_gpt, dn_lo.se_gpt, dn_lo.p_gpt))
    add("A", "Above the median of the measure", dn_=(dn_hi.beta2_gpt, dn_hi.se_gpt, dn_hi.p_gpt))
    add("A", "AI exposure alone",
        pb=cell(pick(l50h, **occ, spec="iii_LS_ai_alone", term="post_x_z_ai")),
        h=cell(pick(l52h, **occ, spec="iii_LS_ai_alone", term="post_x_z_ai")))
    add("A", "Remote work alone",
        pb=cell(pick(l50h, **occ, spec="iii_LS_remote_alone", term="post_x_z_rem")),
        h=cell(pick(l52h, **occ, spec="iii_LS_remote_alone", term="post_x_z_rem")))
    add("A", "Joint: AI exposure",
        pb=cell(pick(l50h, **occ, spec="iii_LS_joint", term="post_x_z_ai")),
        h=cell(pick(l52h, **occ, spec="iii_LS_joint", term="post_x_z_ai")))
    add("A", "Joint: remote work",
        pb=cell(pick(l50h, **occ, spec="iii_LS_joint", term="post_x_z_rem")),
        h=cell(pick(l52h, **occ, spec="iii_LS_joint", term="post_x_z_rem")))
    add("B", "AI exposure alone",
        pb=cell(pick(l50w, **wall, spec="l09_a_baseline", term="gpt_x_high")),
        h=cell(pick(l52w, **wall, spec="l09_a_baseline", term="gpt_x_high")))
    add("B", "Joint: AI exposure",
        pb=cell(pick(l50w, **wall, spec="finer_joint", term="gpt_x_high")),
        h=cell(pick(l52w, **wall, spec="finer_joint", term="gpt_x_high")))
    add("B", "Joint: remote work",
        pb=cell(pick(l50w, **wall, spec="finer_joint", term="gpt_x_rhigh")),
        h=cell(pick(l52w, **wall, spec="finer_joint", term="gpt_x_rhigh")))
    # The employer split exists only for the Swedish measure: it is read
    # from each employer's own 2021-22 advertisements.
    add("B", "AI exposure, employers advertising remote",
        pb=cell(pick(l50w, design="within_employer", sample="all ads, high-remote employers",
                     spec="split", term="gpt_x_high")))
    add("B", "AI exposure, employers not advertising remote",
        pb=cell(pick(l50w, design="within_employer", sample="all ads, low-remote employers",
                     spec="split", term="gpt_x_high")))
    add("B", "Difference",
        pb=cell(pick(l50w, **wall, spec="split_interaction", term="gpt_x_high_x_rememp")))
    add("C", "Joint: AI exposure",
        pb=cell(pick(l50h, **ent, spec="iii_LS_joint", term="post_x_z_ai")),
        h=cell(pick(l52h, **ent, spec="iii_LS_joint", term="post_x_z_ai")))
    add("C", "Joint: remote work",
        pb=cell(pick(l50h, **ent, spec="iii_LS_joint", term="post_x_z_rem")),
        h=cell(pick(l52h, **ent, spec="iii_LS_joint", term="post_x_z_rem")))

    # The counts printed under each panel, read from the same result rows.
    n_a = int(pick(l50h, **occ, spec="iii_LS_joint", term="post_x_z_ai").n_obs)
    n_a_h = int(pick(l52h, **occ, spec="iii_LS_joint", term="post_x_z_ai").n_obs)
    n_b = int(pick(l50w, **wall, spec="finer_joint", term="gpt_x_high").n_firms)
    n_b_h = int(pick(l52w, **wall, spec="finer_joint", term="gpt_x_high").n_firms)
    n_c = int(pick(l50h, **ent, spec="iii_LS_joint", term="post_x_z_ai").n_obs)
    n_c_h = int(pick(l52h, **ent, spec="iii_LS_joint", term="post_x_z_ai").n_obs)
    if n_a != n_a_h or n_b != n_b_h or n_c != n_c_h:
        raise SystemExit("  the two realised measures do not run on the same cells")

    rec = []
    for panel, label, a, b, c in rows:
        for meas, v in (("dingel_neiman", a), ("platsbanken", b), ("hansen", c)):
            if v is not None:
                rec.append(dict(panel=panel, row=label, measure=meas, coef=v[0], se=v[1], pval=v[2]))
    for meas, k in (("dingel_neiman", "dn"), ("platsbanken", "pb"), ("hansen", "h")):
        rec.append(dict(panel="0", row="correlation with AI exposure (employment-weighted)",
                        measure=meas, coef=corr[k], se=None, pval=None))
    pd.DataFrame(rec).to_csv(RESULTS / "remote_measures_cells.csv", index=False)

    def c1(v):
        return "" if v is None else f"${v[0]:.3f}{stars(v[2])}$"

    def c2(v):
        return "" if v is None else f"({v[1]:.3f})"

    dash = "--"
    heads = {"A": r"\textit{A. Across occupations, all advertisements (OLS)}",
             "B": r"\textit{B. Within employer, all advertisements (Poisson)}",
             "C": r"\textit{C. Across occupations, entry-level advertisements (OLS)}"}
    n_obs = {"A": (f"{int(dn.loc['All'].n_obs):,}", f"{n_a:,}", f"{n_a_h:,}", "Occupation-months"),
             "B": ("", f"{n_b:,}", f"{n_b_h:,}", "Employers"),
             "C": ("", f"{n_c:,}", f"{n_c_h:,}", "Occupation-months")}
    L = [r"\begin{table}[ht!]", r"\centering",
         r"\caption{AI exposure and three measures of remote work on the posting margin}",
         r"\label{tab:remote_measures}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{0.92}", r"\begin{tabular}{lccc}", r"\toprule",
         r" & Dingel and Neiman & Platsbanken & Hansen et al. \\",
         r" & (feasible) & (realised, Sweden) & (realised, US) \\",
         r" & (1) & (2) & (3) \\", r"\midrule",
         rf"Correlation with AI exposure & {corr['dn']:.2f} & {corr['pb']:.2f} & {corr['h']:.2f} \\"]
    for panel in ("A", "B", "C"):
        L += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{{heads[panel]}}} \\"]
        for p, label, a, b, c in rows:
            if p != panel:
                continue
            cells = [c1(x) if x is not None else dash for x in (a, b, c)]
            ses = [c2(x) for x in (a, b, c)]
            L.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
            L.append(r" & " + " & ".join(ses) + r" \\")
        o = n_obs[panel]
        L.append(rf"\quad {o[3]} & {o[0] or dash} & {o[1]} & {o[2]} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.97\textwidth}\footnotesize\vspace{4pt}",
          (r"Post-launch coefficients (from December 2022); standard errors, in parentheses, "
           r"clustered by occupation (panels A and C) or employer (panel B). "
           r"Column (1): the \citet{dingel2020many} teleworkability classification, a measure of "
           r"feasibility, split at its median as in Figure~\ref{fig:posting_rivals}(b), with "
           r"Equation~(1) estimated in each half. Column (2): the share of an occupation's "
           r"advertisements in 2021 and 2022 that offer remote or hybrid work, read from the "
           r"advertisement text. Column (3): the same share in United States postings "
           r"\citep{hansen2023remote}, the measure of \citet{lambert2026brokenladder}, "
           r"crosswalked to SSYK. Correlations are across the 369 occupations, weighted by 2024 "
           r"employment. Panels A and C follow Lambert and Schindler: one post-launch dummy "
           r"interacted with each standardised score, alone and together, with occupation and "
           r"month effects. Panel B is the design of Section~\ref{sec:firm_design}, with each "
           r"employer's cells cut by the top quartile of AI exposure and of remote work. Its "
           r"employer split, possible only in column (2), separates employers by whether any of "
           r"their own 2021 to 2022 advertisements offered remote work. "
           r"$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."),
          r"\end{minipage}", r"\end{table}"]
    out = TABLES / "tableA_remote_measures.tex"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {out.name} and remote_measures_cells.csv")


if __name__ == "__main__":
    main()

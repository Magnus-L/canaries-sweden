#!/usr/bin/env python3
"""
l53_tab_remote_measures.py: one table setting three remote-work measures
side by side against AI exposure on the posting margin (OA II.3,
label tab:remote_measures).

WHY THIS EXISTS
Lambert and Schindler (2026) argue that generative-AI exposure stands in
for the post-pandemic shift to remote work. OA II.3 answered with Dingel
and Neiman's feasibility classification only. l50 (realised remote work
read from Swedish advertisement text) and l52 (Hansen et al. 2023, the
measure Lambert and Schindler use, crosswalked SOC -> SSYK) add two
realised measures. Each had its own six-column table; this script folds
the comparable rows into one exhibit, funnelled broad to fine:
across occupations, within employers, entry-level advertisements.

NOTHING IS ESTIMATED HERE. Every number is read from an export:
- revision/tables/l50_remote_{horserace,within,correlations}.csv
- revision/tables/l52_hansen_{horserace,within,correlations}.csv
- tables/telework_did_results.csv (the Dingel and Neiman median split
  behind Figure panel (b) in OA II.3)

Run:  python3 revision/local/l53_tab_remote_measures.py
Out:  revision/tables/l53_remote_measures.csv
      ../canaries-sweden-paper/tables/tableA_remote_measures.tex
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "revision" / "tables"
PAPER = ROOT.parent / "canaries-sweden-paper" / "tables"


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
        raise ValueError(f"{kw}: {len(hit)} rows")
    return hit.iloc[0]


def cell(r):
    """(coef, se, p) triple from an export row."""
    return (float(r["coef"]), float(r["se"]), float(r["pval"]))


def main():
    l50h = pd.read_csv(TAB / "l50_remote_horserace.csv")
    l50w = pd.read_csv(TAB / "l50_remote_within.csv")
    l50c = pd.read_csv(TAB / "l50_remote_correlations.csv")
    l52h = pd.read_csv(TAB / "l52_hansen_horserace.csv")
    l52w = pd.read_csv(TAB / "l52_hansen_within.csv")
    l52c = pd.read_csv(TAB / "l52_hansen_correlations.csv")
    dn = pd.read_csv(ROOT / "tables" / "telework_did_results.csv").set_index("group")

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

    # Panel A: across occupations. Dingel-Neiman as the median split of
    # OA II.3; the realised measures in Lambert and Schindler's layout
    # (one post-launch dummy, standardised scores).
    dn_lo = dn.loc["Non-teleworkable"]
    dn_hi = dn.loc["Teleworkable"]
    add("A", "Below the median of the measure",
        dn_=(dn_lo.beta2_gpt, dn_lo.se_gpt, dn_lo.p_gpt))
    add("A", "Above the median of the measure",
        dn_=(dn_hi.beta2_gpt, dn_hi.se_gpt, dn_hi.p_gpt))
    ai_alone_pb = cell(pick(l50h, **occ, spec="iii_LS_ai_alone", term="post_x_z_ai"))
    ai_alone_h = cell(pick(l52h, **occ, spec="iii_LS_ai_alone", term="post_x_z_ai"))
    add("A", "AI exposure alone", pb=ai_alone_pb, h=ai_alone_h)
    add("A", "Remote work alone",
        pb=cell(pick(l50h, **occ, spec="iii_LS_remote_alone", term="post_x_z_rem")),
        h=cell(pick(l52h, **occ, spec="iii_LS_remote_alone", term="post_x_z_rem")))
    add("A", "Joint: AI exposure",
        pb=cell(pick(l50h, **occ, spec="iii_LS_joint", term="post_x_z_ai")),
        h=cell(pick(l52h, **occ, spec="iii_LS_joint", term="post_x_z_ai")))
    add("A", "Joint: remote work",
        pb=cell(pick(l50h, **occ, spec="iii_LS_joint", term="post_x_z_rem")),
        h=cell(pick(l52h, **occ, spec="iii_LS_joint", term="post_x_z_rem")))

    # Panel B: within employer (Poisson, employer x cell and employer x
    # month effects). AI alone is the published l09 estimate.
    add("B", "AI exposure alone",
        pb=cell(pick(l50w, **wall, spec="l09_a_baseline", term="gpt_x_high")),
        h=cell(pick(l52w, **wall, spec="l09_a_baseline", term="gpt_x_high")))
    add("B", "Joint: AI exposure",
        pb=cell(pick(l50w, **wall, spec="finer_joint", term="gpt_x_high")),
        h=cell(pick(l52w, **wall, spec="finer_joint", term="gpt_x_high")))
    add("B", "Joint: remote work",
        pb=cell(pick(l50w, **wall, spec="finer_joint", term="gpt_x_rhigh")),
        h=cell(pick(l52w, **wall, spec="finer_joint", term="gpt_x_rhigh")))
    # Employer split exists only for the Swedish measure: it is read from
    # each employer's own 2021-22 advertisements.
    add("B", "AI exposure, employers advertising remote",
        pb=cell(pick(l50w, design="within_employer", sample="all ads, high-remote employers",
                     spec="split", term="gpt_x_high")))
    add("B", "AI exposure, employers not advertising remote",
        pb=cell(pick(l50w, design="within_employer", sample="all ads, low-remote employers",
                     spec="split", term="gpt_x_high")))
    add("B", "Difference",
        pb=cell(pick(l50w, **wall, spec="split_interaction", term="gpt_x_high_x_rememp")))

    # Panel C: entry-level advertisements, Lambert and Schindler's layout.
    add("C", "Joint: AI exposure",
        pb=cell(pick(l50h, **ent, spec="iii_LS_joint", term="post_x_z_ai")),
        h=cell(pick(l52h, **ent, spec="iii_LS_joint", term="post_x_z_ai")))
    add("C", "Joint: remote work",
        pb=cell(pick(l50h, **ent, spec="iii_LS_joint", term="post_x_z_rem")),
        h=cell(pick(l52h, **ent, spec="iii_LS_joint", term="post_x_z_rem")))

    # CSV record of every cell in the table.
    rec = []
    for panel, label, a, b, c in rows:
        for meas, v in (("dingel_neiman", a), ("platsbanken", b), ("hansen", c)):
            if v is not None:
                rec.append(dict(panel=panel, row=label, measure=meas, coef=v[0], se=v[1], pval=v[2]))
    for meas, k in (("dingel_neiman", "dn"), ("platsbanken", "pb"), ("hansen", "h")):
        rec.append(dict(panel="0", row="correlation with AI exposure (employment-weighted)",
                        measure=meas, coef=corr[k], se=None, pval=None))
    pd.DataFrame(rec).to_csv(TAB / "l53_remote_measures.csv", index=False)

    # LaTeX.
    def c1(v):
        return "" if v is None else f"${v[0]:.3f}{stars(v[2])}$"

    def c2(v):
        return "" if v is None else f"({v[1]:.3f})"

    dash = "--"
    heads = {
        "A": r"\textit{A. Across occupations, all advertisements (OLS)}",
        "B": r"\textit{B. Within employer, all advertisements (Poisson)}",
        "C": r"\textit{C. Across occupations, entry-level advertisements (OLS)}",
    }
    n_obs = {
        "A": ("26,672", "28,084", "28,084", "Occupation-months"),
        "B": ("", "12,141", "12,141", "Employers"),
        "C": ("", "12,159", "12,159", "Occupation-months"),
    }
    L = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{AI exposure and three measures of remote work on the posting margin}",
        r"\label{tab:remote_measures}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & Dingel and Neiman & Platsbanken & Hansen et al. \\",
        r" & (feasible) & (realised, Sweden) & (realised, US) \\",
        r" & (1) & (2) & (3) \\",
        r"\midrule",
        rf"Correlation with AI exposure & {corr['dn']:.2f} & {corr['pb']:.2f} & {corr['h']:.2f} \\",
    ]
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
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.97\textwidth}\footnotesize\vspace{4pt}",
        (r"Post-launch coefficients (from December 2022); standard errors, in parentheses, "
         r"clustered by occupation (panels A and C) or employer (panel B). "
         r"Column (1): the \citet{dingel2020many} teleworkability classification, a measure of "
         r"feasibility, split at its median as in Figure~\ref{fig:posting_rivals}(b), with "
         r"Equation~(1) estimated in each half on the sample of the original submission "
         r"(January 2020 to February 2026). Column (2): the share of an occupation's "
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
        r"\end{minipage}",
        r"\end{table}",
    ]
    (PAPER / "tableA_remote_measures.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

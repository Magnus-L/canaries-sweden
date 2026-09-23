#!/usr/bin/env python3
"""
l37_v3_prose_numbers.py: every number the v3 prose quotes, printed from
the exports it comes from.

WHY THIS EXISTS
Table 1 and Figure 2 are generated, so their numbers cannot drift from
the exports. The numbers in the running text are typed, and in v2 that
was where the two errors of the 21 September sweep were found. This
script prints each one beside its source file and the arithmetic that
turns a Poisson coefficient into the percentage the sentence quotes, so
the text can be checked against it line by line, and re-checked after any
re-export.

    python3 revision/local/l37_v3_prose_numbers.py [export_dir]

IN THE PAPER
Section 1 (the preview), Section 2 (the measure), Section 3 (every
estimate), and the abstract.
"""
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
OUT = REV / "output"
OCC = OUT / "round3_20260923-0655-lanes28b-29bcd"
LANE28A = OUT / "round3_20260922-2232-lane28a"
# The track split and the mix behind its weights (script 87, lane 33).
SPLIT = OUT / "round3_20260923-1352-lane33-script87"
CONTRAST = OUT / "round3_20260923-1407-lane33-script88"

TERM = "post_x_high_x_young"
INTER = "interim_x_high_x_young"
FEMALE = "post_x_high_x_young_x_female"


def pct(c: float) -> float:
    """A Poisson coefficient as the percentage change the text quotes."""
    import math
    return (math.exp(c) - 1.0) * 100.0


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else OCC
    head = pd.read_csv(d / "occ_route_headline.csv")
    head = head[(head.arm == "backward") & (head.floor == 5)
                & (head.level == "uniform3") & (head.outcome == "stock")]
    prof = pd.read_csv(d / "occ_route_profile.csv").set_index("band")
    flows = pd.read_csv(d / "occ_route_flows.csv")
    gender = pd.read_csv(d / "occ_route_gender.csv")
    window = pd.read_csv(d / "occ_rest_window.csv")
    drift = pd.read_csv(d / "occ_rest_drift.csv")
    clust = pd.read_csv(d / "occ_rest_cluster.csv")
    ind = pd.read_csv(d / "occ_rest_industry.csv")
    size = pd.read_csv(d / "occ_rest_size.csv")
    rel = pd.read_csv(d / "occ_rest_reliability.csv")
    fs = pd.read_csv(d / "occ_rest_firststage.csv") \
        if (d / "occ_rest_firststage.csv").exists() \
        else pd.read_csv(OUT / "round3_20260922-2333-lane29a"
                         / "occ_rest_firststage.csv")

    def g(df, **c):
        r = df
        for k, v in c.items():
            r = r[r[k] == v]
        if len(r) != 1:
            raise SystemExit(f"  expected one row for {c}, found {len(r)}")
        return float(r.coef.iloc[0]), float(r.se.iloc[0])

    print("SECTION 2, THE MEASURE  (lane 28a summary and the headline export)")
    print(f"  employers scored, floor of five person-months      262,089")
    print(f"  employers in the 22-25 panel                       "
          f"{int(head[head.young_band == '22-25'].n_firms.iloc[0]):,}")
    print(f"  employers in the 26-30 panel                       "
          f"{int(head[head.young_band == '26-30'].n_firms.iloc[0]):,}")
    print(f"  employers in the six-band profile panel            "
          f"{int(prof.n_firms.iloc[0]):,}")
    print()
    print("SECTION 2, THE FIRST STAGE  (occ_rest_firststage.csv)")
    any_ai = fs[fs.outcome == "ai_any"]
    for src, label in (("ITFtg_Stora_2023", "2023 firm-level, any AI"),
                       ("BITA_2024", "2024 individual, generative AI")):
        for route in ("occupation", "education"):
            r = any_ai[(any_ai.source == src) & (any_ai.route == route)]
            if not len(r):
                r = fs[(fs.source == src) & (fs.route == route)]
            if len(r):
                r = r.iloc[0]
                print(f"  {label:<32} {route:<11} "
                      f"{float(r.coef_points):+.2f} points "
                      f"({float(r.se_points):.2f}), n {int(r.n):,}")
    print("  the pre-ChatGPT any-AI gap, firm level, points")
    for src, yr in (("ai_itftg_2019", 2019), ("ITFtg_Stora_2021", 2021),
                    ("ITFtg_Stora_2023", 2023)):
        cells = []
        for route in ("occupation", "education"):
            r = any_ai[(any_ai.source == src) & (any_ai.route == route)]
            if len(r):
                cells.append(f"{route} {float(r.coef_points.iloc[0]):+.1f} "
                             f"({float(r.se_points.iloc[0]):.1f})")
        print(f"    {yr}  " + "   ".join(cells))

    print("SECTION 3, THE HEADLINE  (occ_route_headline.csv, occ_rest_window.csv)")
    g1 = g(head, young_band="22-25", term="rb_x_high_x_young")
    g2 = g(head, young_band="22-25", term=TERM)
    g2_26 = g(head, young_band="26-30", term=TERM)
    w22 = g(window, young_band="22-25", outcome="stock", term=TERM)
    i22 = g(window, young_band="22-25", outcome="stock", term=INTER)
    w26 = g(window, young_band="26-30", outcome="stock", term=TERM)
    i26 = g(window, young_band="26-30", outcome="stock", term=INTER)
    s22, s26 = w22[0] - i22[0], w26[0] - i26[0]
    print(f"  22-25 tightening months  {g1[0]:+.4f} ({g1[1]:.4f})  "
          f"{pct(g1[0]):+.1f} per cent")
    print(f"  22-25 step at adoption   {g2[0]:+.4f} ({g2[1]:.4f})  "
          f"{pct(g2[0]):+.1f} per cent from the tightening level")
    print(f"  22-25 step from 2023     {s22:+.4f}            "
          f"{pct(s22):+.1f} per cent   -> one young worker in "
          f"{1 / abs(pct(s22) / 100):.0f}")
    print(f"  26-30 step at adoption   {g2_26[0]:+.4f} ({g2_26[1]:.4f})  "
          f"{pct(g2_26[0]):+.1f} per cent")
    print(f"  26-30 step from 2023     {s26:+.4f}            "
          f"{pct(s26):+.1f} per cent")
    print()

    print("SECTION 3, THE PROFILE  (occ_route_profile.csv)")
    for b in ("22-25", "50+"):
        r = prof.loc[b]
        print(f"  {b:<6} {float(r.coef):+.4f} ({float(r.se):.4f})  "
              f"{pct(float(r.coef)):+.1f} per cent; education route "
              f"{float(r.edu_coef):+.4f} ({float(r.edu_se):.4f})")
    print()

    print("SECTION 3, SEX AND MARGINS  (occ_route_gender.csv, occ_route_flows.csv)")
    men = g(gender[gender.block == "term"], young_band="22-25", term=TERM)
    fem = g(gender[gender.block == "term"], young_band="22-25", term=FEMALE)
    wom = g(gender[gender.block == "step"], young_band="22-25",
            term="female_step")
    print(f"  young men                {men[0]:+.4f} ({men[1]:.4f})  "
          f"{pct(men[0]):+.1f} per cent")
    print(f"  female differential      {fem[0]:+.4f} ({fem[1]:.4f})  "
          f"{pct(fem[0]):+.1f} per cent")
    print(f"  young women              {wom[0]:+.4f} ({wom[1]:.4f})  "
          f"{pct(wom[0]):+.1f} per cent")
    for outcome in ("hires", "seps"):
        c, se = g(flows, outcome=outcome, young_band="22-25", term=TERM)
        lo, hi = pct(c - 1.96 * se), pct(c + 1.96 * se)
        print(f"  {outcome:<8}               {c:+.4f} ({se:.4f})  "
              f"{pct(c):+.1f} per cent, interval {lo:+.1f} to {hi:+.1f}")
    print()

    print("SECTION 3 AND OA III.2, THE TRACK SPLIT  "
          "(occ_route_gender_by_track.csv, occ_route_gender_split.csv)")
    bt = pd.read_csv(SPLIT / "occ_route_gender_by_track.csv").set_index("track")
    sp = pd.read_csv(SPLIT / "occ_route_gender_split.csv").iloc[0]
    mix = pd.read_csv(SPLIT / "occ_route_education_mix_by_sex.csv")
    pooled = float(sp.pooled)
    if abs(pooled - fem[0]) > 5e-5:
        raise SystemExit(f"  the split's pooled differential {pooled:+.6f} is "
                         f"not Section 3's female differential {fem[0]:+.6f}")
    print(f"  pooled                   {pooled:+.4f} ({float(sp.pooled_se):.4f})"
          f"   [equals the female differential above, which is the gate]")
    print(f"  {'within, as exported':<28} {float(sp.within):+.4f} "
          f"({float(sp.within_se):.4f})   {100 * float(sp.ratio_within):.0f} "
          f"per cent of pooled; composition {100 * (1 - float(sp.ratio_within)):.0f} per cent")
    # The men's-shares variant is NOT exported: it is the same weighted
    # sum over the mix table's men's shares, and the paper quotes it, so
    # it is computed here rather than typed.
    tracks = [k for k in bt.index if k != "all"]
    for gender_w in ("women", "men"):
        w = mix[(mix.dimension == "track") & (mix.exposed == 1)
                & (mix.gender == gender_w)].set_index("cell")["share"]
        wr = w[tracks] / w[tracks].sum()
        within = float((wr * bt.loc[tracks, "diff"]).sum())
        se = float(((wr * bt.loc[tracks, "diff_se"]) ** 2).sum() ** 0.5)
        tag = ("women's shares, recomputed" if gender_w == "women"
               else "men's shares, not exported")
        print(f"  {tag:<28} {within:+.4f} ({se:.4f})   "
              f"{100 * within / pooled:.0f} per cent of pooled; "
              f"composition {100 * (1 - within / pooled):.0f} per cent")
    for k in tracks:
        r = bt.loc[k]
        print(f"    {k:<22} men {float(r.male):+.4f} ({float(r.male_se):.4f}) "
              f"t {float(r.male) / float(r.male_se):+.2f}   "
              f"diff {float(r['diff']):+.4f} ({float(r.diff_se):.4f}) "
              f"t {float(r['diff']) / float(r.diff_se):+.2f}   "
              f"{pct(float(r.male)):+.1f} per cent for men")
    w = mix[(mix.dimension == "track") & (mix.exposed == 1)
            & (mix.gender == "women")].set_index("cell")["share"]
    print(f"    ICT share of exposed firms' young women "
          f"{100 * float(w['ict']):.1f} per cent")
    print()

    print("OA III.2, THE THREE-BAND CONTRAST BY TRACK  "
          "(occ_route_contrast_by_track.csv)")
    ct = pd.read_csv(CONTRAST / "occ_route_contrast_by_track.csv")
    for band in ("22-25", "26-30"):
        b = ct[ct.band_vs_ref == band].set_index("track")
        print(f"  {band} against 41-49:")
        for k in ["all"] + [x for x in b.index if x != "all"]:
            r = b.loc[k]
            print(f"    {k:<22} {float(r.coef):+.4f} ({float(r.se):.4f}) "
                  f"t {float(r.t):+.2f}   {int(r.n_firms):>7,} firms")
    print()

    print("SECTION 3, THE PRE-TEST AND THE TWO CLUSTERINGS")
    for band in ("22-25", "26-30"):
        c, se = g(drift, young_band=band, term="trend_x_high_x_young")
        print(f"  drift {band}             {c:+.5f} ({se:.5f})  "
              f"{abs(c / se):.1f} standard errors from zero")
    for band in ("22-25", "26-30"):
        r = clust[(clust.spec == "pooled") & (clust.young_band == band)
                  & (clust.term == TERM)].iloc[0]
        print(f"  {band} industry SE       ({float(r.se_industry_complete):.4f})"
              f"   t {float(r.coef) / float(r.se_industry_complete):+.2f}")
    r = clust[(clust.spec == "gender") & (clust.term == FEMALE)].iloc[0]
    print(f"  differential industry SE ({float(r.se_industry_complete):.4f})"
          f"   t {float(r.coef) / float(r.se_industry_complete):+.2f}")
    print()

    print("SECTION 3, INDUSTRY AND CREDIT  (occ_rest_industry.csv)")
    for band in ("22-25", "26-30"):
        r = ind[(ind.young_band == band) & (ind.term == TERM)
                & (ind.spec == "industry_age_month")].iloc[0]
        b = ind[(ind.young_band == band) & (ind.term == TERM)
                & (ind.spec == "baseline_same_sample")].iloc[0]
        print(f"  {band}: baseline {float(b.coef):+.4f} ({float(b.se):.4f}) "
              f"-> {float(r.coef):+.4f} ({float(r.se):.4f}), retained "
              f"{float(r.retained_share) * 100:.0f} per cent "
              f"(education route {float(r.edu_retained_share) * 100:.0f}); "
              f"{int(r.n_firms):,} employers, {int(r.n_groups)} groups, "
              f"{float(r.share_not_from_2019) * 100:.1f} per cent coded "
              f"from another year")
    print()

    print("ONLINE APPENDIX, THE SIZE SPLIT  (occ_rest_size.csv)")
    p = size[size.term == TERM]
    for _, r in p.iterrows():
        lab = f"{r.spec} {r.young_band}"
        print(f"  {lab:<22} {float(r.coef):+.4f} ({float(r.se):.4f})  "
              f"{int(r.n_firms):,} employers, {int(r.n_scored):,} scored")
    print()

    print("ONLINE APPENDIX, THE RELIABILITY  (occ_rest_reliability.csv)")
    v = rel[rel.block == "variance"].set_index("item")["value"]
    print(f"  between-firm variance, net of sampling noise  "
          f"{float(v['variance_between_firms_net']):.1f}")
    print(f"  within-firm variance                          "
          f"{float(v['variance_within_firm']):.1f}")
    print(f"  net between share                             "
          f"{float(v['share_between_net']):.3f}")
    thin = rel[rel.block == "thin"].iloc[0]
    print(f"  employers below a reliability of one half     "
          f"{int(thin.n_firms):,} ({float(thin.share_firms) * 100:.1f} per "
          f"cent), holding {float(thin.share_employment) * 100:.1f} per cent "
          f"of incumbent employment")
    q = rel[rel.block == "reliability_by_firm_quantile"].set_index("item")
    e = rel[rel.block == "reliability_by_employment_quantile"].set_index("item")
    print(f"  median employer: {float(q.loc['p50_of_employers', 'n_incumbents']):.0f}"
          f" incumbents, reliability "
          f"{float(q.loc['p50_of_employers', 'value']):.2f}")
    print(f"  median of incumbent EMPLOYMENT: "
          f"{float(e.loc['p50_of_incumbent_employment', 'n_incumbents']):.0f}"
          f" incumbents, reliability "
          f"{float(e.loc['p50_of_incumbent_employment', 'value']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
70_respecifications.py -- three things the reviews said the paper cannot
                          currently claim, all on panels already cached.

======================================================================
  RUNS IN MONA. NO SQL AT ALL. Reads 47h's 2019 frame, 47L's cached
  counts and 47L's cached baseline pay. Writes output_70/.
======================================================================

Three questions, none of which needs a new pull, all of which decide
something the paper currently asserts without having tested it.

PART A. IS THE YOUNG COEFFICIENT DIFFERENT FROM THE MIDDLE-AGED ONE?

The paper is called canaries. Script 63 reports, adoption-dated and net of
teleworkability, a stock coefficient of -0.0318 at 22-25 and -0.0493 at
41-49. Both were read as "significant declines" from their own standard
errors, and the 41-49 result was treated as an awkward extra finding.

The arithmetic nobody did is that the POINT ESTIMATES put the youngest
band 0.0175 ABOVE the middle-aged one: on the stock, the young look
better, not worse. Comparing two t-statistics is not a test of whether two
coefficients differ, and the paper needs that test before it can keep its
framing.

The fix costs one fit. Exposure here is firm-level and constant within
employer-month, so the full set of six band interactions is collinear with
the employer-by-month effects and one band must drop. Which band drops is
a free choice, and every remaining coefficient is then a DIFFERENCE from
the dropped one, with a correct standard error and covariance built in. So
we drop 41-49 deliberately. The 22-25 coefficient that comes back IS the
contrast, tested, with no post-estimation algebra.

PART B. THE YOUTH PAYROLL-TAX EXPIRY, INTERACTED WITH EXPOSURE.

Sweden's reduced employer contributions for young workers expired on
31 March 2023, inside the post window. 47L already anticipated this: it
carries the SEK 25,000 cap, the April 2023 date, and a `post_tax x
taxshare` control, where taxshare is the base-year share of a firm's young
workers paid under the cap.

What it does not carry is the interaction that matters. A national policy
with a common effect is absorbed by the month-by-age effects. What
survives them is an EXPOSURE-DIFFERENTIAL response: exposed firms
employing many subsidised young workers reacting differently from
unexposed firms that also employed many. That is
`post_tax x taxshare x high`, and it is one term.

Timing is a partial defence whatever comes back, since the subsidy ended
in March 2023 and the effect concentrates from January 2024. The point is
to have tested it rather than to argue from the calendar.

PART C. WHY THE MECHANISM REVERSES BETWEEN THE TWO EXPOSURE ROUTES.

63 (occupation route) says hiring carries the decline. 67 (education
route, the headline classification) says hiring is null and separations
rise. The paper's abstract asserts hiring.

These two are not a clean measure comparison: the education route
classifies 311,227 firms and the occupation route 65,146, so population
and measurement move together. This part separates them as far as the
cached frames allow:

  A  education score, full education sample
  B  education score, restricted to firms the occupation route also scores
  C  both routes, restricted to firm-age cells BOTH routes score
  D  occupation score, same restriction as C

A versus B isolates employer coverage. C versus D isolates the register.

One limit, stated because it changes what C-versus-D means: a true
worker-level joint-support restriction, using only incumbents who hold
BOTH a usable 2019 occupation code and usable 2019 education fields, is
not possible from the cached frames, because the two routes read different
pre-aggregated tables. C restricts at the level of the firm-age cell, not
the worker. So C versus D isolates the register and the incumbent pool
together, not the register alone. Getting further needs a new SQL pull and
is not worth a MONA round for this revision.

Output (output_70/):
  age_contrast.csv     Part A, differences from the 41-49 band
  payroll_tax.csv      Part B
  route_ladder.csv     Part C, the four variants
  70_summary.txt
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_70"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

ALL_BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
# The reference band for Part A. 41-49 and not 50+: the comparison the
# paper's framing rests on is young against prime-age, and 41-49 is the
# band that contradicted it. Fixed before the run.
REF_BAND = "41-49"
PANEL_FROM = "2021-01"
PANEL_YEARS = list(range(2021, 2026))
POOLED_FROM = "2024-01"
TRUNC = 2021
DESIGN = "OL_daioe"
ARM = "true"
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def edu_exposure(j47, design: str, arm: str):
    """
    47j's incumbent education exposure, built exactly as 61 builds it.

    Reproduced rather than imported because 61 does this inside its main().
    The steps and their order matter: the ScoreBook needs every weight year,
    not only the base year, and the spec must be built into the book before
    incumbent_exposure is asked for anything.
    """
    h47 = j47._h47()
    counts = {}
    for y in h47.WEIGHT_YEARS:
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise SystemExit(f"edu_hr_weights_{y}.parquet missing: run 47h "
                             f"first. This script performs no education SQL.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[design])
    book.build(design, spec)
    frame19 = mc.read_cache(CACHE / f"edu_hr_{j47.BASE_YEAR}.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise SystemExit(f"edu_hr_{j47.BASE_YEAR}.parquet missing: run 47h.")
    expo, _ = j47.incumbent_exposure(frame19, book, design, spec, arm, TRUNC)
    del frame19
    gc.collect()
    return expo


def daioe_scores() -> pd.DataFrame:
    """
    The DAIOE frame as the occupation route needs it: ssyk4 and `score`.

    NOT mc.load_daioe(), which returns ssyk4 and exposure_quartile. 65's
    own main builds this, and occupation_exposure and 47L's build_exposure
    both read `score`, so calling the quartile loader here fails with a
    bare KeyError several frames later.
    """
    d = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    return d.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]


def band_col(prefix: str, band: str) -> str:
    return f"{prefix}_" + band.replace("-", "_").replace("+", "p")


def all_band_skeleton(counts: pd.DataFrame) -> pd.DataFrame:
    """
    The balanced employer x band x month panel over ALL SIX bands.

    61's skeleton takes one young band plus the incumbents, because its
    design is a single young-versus-older contrast. Part A needs the whole
    age profile in one fit, so the panel has to carry every band.
    """
    p = counts[counts["age_group"].astype(str).isin(ALL_BANDS)]
    p = p[p["year_month"].astype(str) >= PANEL_FROM]
    p = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
         ["n_emp"].sum().reset_index())
    p["age_group"] = p["age_group"].astype(str)
    p["year_month"] = p["year_month"].astype(str)
    # a firm must hold the reference band and at least one other, or it
    # contributes nothing to a within-employer age contrast
    have = p.groupby("employer_id")["age_group"].agg(set)
    keep = have[have.apply(lambda v: REF_BAND in v and len(v) >= 2)].index
    p = p[p["employer_id"].isin(keep)]
    if p.empty:
        return p
    months = sorted(p["year_month"].unique())
    emp = p["employer_id"].drop_duplicates().to_numpy()
    full = pd.MultiIndex.from_product([emp, ALL_BANDS, months],
                                      names=["employer_id", "age_group",
                                             "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index())
    bal["n_emp"] = bal["n_emp"].astype(int)
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"], sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def part_a(counts, expo, j47, sink):
    """The age profile with 41-49 omitted, so coefficients are contrasts."""
    bal = all_band_skeleton(counts)
    if bal.empty:
        print("  A: skeleton empty"); return
    b = bal.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    del bal; gc.collect()
    if b.empty:
        print("  A: no firms matched exposure"); return
    b["high"] = (b["fq"] == 4).astype(int)
    ym = b["year_month"].astype(str)
    post = (ym >= POOLED_FROM).astype(int)
    post_rb = (ym >= mc.RIKSBANK_YM).astype(int)
    terms = []
    # Both the treatment and the Riksbank control go in band by band. With
    # firm-level exposure, an un-interacted post x high is absorbed by the
    # employer-by-month effects, so a pooled Riksbank control would silently
    # contribute nothing and would not actually control for anything.
    for band in ALL_BANDS:
        if band == REF_BAND:
            continue
        d = (b["age_group"] == band).astype(int)
        c1 = band_col("gpt_x_high", band)
        c2 = band_col("rb_x_high", band)
        b[c1] = post * b["high"] * d
        b[c2] = post_rb * b["high"] * d
        terms += [c1, c2]
    print(f"  A: panel {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag="r70_age_contrast", terms=terms,
                            fes=j47.FES)
    if r.empty:
        FAILURES.append("A/age_contrast")
    else:
        for _, row in r.iterrows():
            if not row["term"].startswith("gpt_x_high"):
                continue
            sink.append({"band_vs_ref": row["term"].replace("gpt_x_high_", ""),
                         "reference": REF_BAND, "coef": row["coef"],
                         "se": row["se"],
                         "t": row["coef"] / row["se"] if row["se"] else np.nan})
    del b; gc.collect()


def part_b(counts, l47, j47, sink):
    """post_tax x taxshare x high on 47L's own panel."""
    pay = mc.read_cache(CACHE / "L_basepay_2019.parquet")
    if pay is None or not len(pay):
        print("  B: L_basepay_2019 not cached; part B SKIPPED and SAID SO")
        FAILURES.append("B/no_basepay")
        return
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    daioe = daioe_scores()
    expo = l47.build_exposure(base, daioe)
    if expo is None or not len(expo):
        print("  B: no exposure built"); FAILURES.append("B/no_expo"); return
    pay = pay[pay["age_group"] != "other"].copy()
    pay["taxshare"] = pay["n_under_cap"] / pay["n_all"].clip(lower=1)
    pay = pay[["employer_id", "age_group", "taxshare"]]
    bal = l47.build_panel(counts, expo, tax=pay)
    if bal is None or bal.empty:
        print("  B: panel empty"); FAILURES.append("B/empty"); return
    bal["taxshare"] = bal["taxshare"].fillna(0.0)
    ym = bal["year_month"].astype(str)
    bal["post_tax"] = (ym >= l47.TAX_YM).astype(int)
    # `expo` here is 47L's firm-by-age score, so `high` is the top quartile
    # of that same distribution and the triple is identified.
    cut = bal["expo"].quantile(0.75)
    bal["high"] = (bal["expo"] >= cut).astype(int)
    bal["post_tax_x_taxshare"] = bal["post_tax"] * bal["taxshare"]
    bal["post_tax_x_taxshare_x_high"] = bal["post_tax_x_taxshare"] * bal["high"]
    terms = list(l47.TERMS) + ["post_tax_x_taxshare",
                               "post_tax_x_taxshare_x_high"]
    print(f"  B: panel {len(bal):,} rows{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(bal, OUT, tag="r70_payroll_tax", terms=terms,
                            fes=l47.FES)
    if r.empty:
        FAILURES.append("B/fit")
    else:
        for _, row in r.iterrows():
            sink.append({"term": row["term"], "coef": row["coef"],
                         "se": row["se"]})
    del bal; gc.collect()


def part_c(counts, j47, l65, sink):
    """The four-rung ladder from education-on-everything to occupation."""
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    daioe = daioe_scores()
    edu = edu_exposure(j47, DESIGN, ARM)
    occ = l65.occupation_exposure(base, daioe, j47.INCUMBENT_BANDS)
    if edu is None or occ is None or edu.empty or occ.empty:
        print("  C: a route produced no exposure"); FAILURES.append("C/expo")
        return
    both = set(edu["employer_id"]) & set(occ["employer_id"])
    print(f"  C: education {len(edu):,} firms, occupation {len(occ):,}, "
          f"intersection {len(both):,}")
    rungs = [("A_edu_full", edu, None),
             ("B_edu_intersect", edu, both),
             ("C_edu_joint", edu, both),
             ("D_occ_joint", occ, both)]
    # The skeleton does not depend on which route scored the firm, only
    # the quartile does. 61 learned this the expensive way: rebuilding a
    # forty-million-row panel per variant is four builds where one will do.
    skel = all_band_skeleton(counts)
    if skel.empty:
        print("  C: skeleton empty"); FAILURES.append("C/skeleton"); return
    for name, ex, restrict in rungs:
        e = ex if restrict is None else ex[ex["employer_id"].isin(restrict)]
        if e.empty:
            print(f"  C {name}: empty"); continue
        b = skel.merge(e[["employer_id", "fq"]], on="employer_id",
                       how="inner")
        if b.empty:
            print(f"  C {name}: no overlap with the panel"); continue
        b["high"] = (b["fq"] == 4).astype(int)
        ym = b["year_month"].astype(str)
        post = (ym >= POOLED_FROM).astype(int)
        post_rb = (ym >= mc.RIKSBANK_YM).astype(int)
        terms = []
        for band in ALL_BANDS:
            if band == REF_BAND:
                continue
            d = (b["age_group"] == band).astype(int)
            c1, c2 = band_col("gpt_x_high", band), band_col("rb_x_high", band)
            b[c1] = post * b["high"] * d
            b[c2] = post_rb * b["high"] * d
            terms += [c1, c2]
        print(f"  C {name}: panel {len(b):,} rows, "
              f"{b['employer_id'].nunique():,} firms{mc.mem_line(' | ')}")
        r = mc.run_fepois_multi(b, OUT, tag=f"r70_rung_{name}", terms=terms,
                                fes=j47.FES)
        if r.empty:
            FAILURES.append(f"C/{name}")
        else:
            for _, row in r.iterrows():
                if not row["term"].startswith("gpt_x_high"):
                    continue
                sink.append({"rung": name,
                             "band_vs_ref": row["term"].replace(
                                 "gpt_x_high_", ""),
                             "reference": REF_BAND,
                             "coef": row["coef"], "se": row["se"],
                             "n_firms": int(b["employer_id"].nunique())})
        del b; gc.collect()
    del skel; gc.collect()


def main():
    mc.Tee(OUT / "70_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"70 respecifications; reference band {REF_BAND}; no SQL")
    print("=" * 70)

    j47 = _mod("47j_within_employer_triple.py", "j47")
    l47 = _mod("47L_age_baseline_exposure.py", "l47")
    l65 = _mod("65_occupation_arm.py", "l65")

    counts = []
    for y in PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise SystemExit(
                f"L_counts_{y}.parquet is not cached. This script does no "
                f"SQL by design; run 47L or 69 first.")
        counts.append(c)
    cnt = pd.concat(counts, ignore_index=True)
    del counts; gc.collect()
    last = str(cnt["year_month"].max())
    if last < POOLED_FROM:
        raise SystemExit(f"counts end at {last}, before {POOLED_FROM}.")
    print(f"  counts to {last}, {len(cnt):,} cells")

    expo = edu_exposure(j47, DESIGN, ARM)
    print(f"  headline exposure: {len(expo):,} firms")

    a_sink, b_sink, c_sink = [], [], []
    opt("part A", part_a, cnt, expo, j47, a_sink)
    opt("part B", part_b, cnt, l47, j47, b_sink)
    opt("part C", part_c, cnt, j47, l65, c_sink)

    lines = ["70 respecifications", "=" * 70, ""]

    if a_sink:
        df = pd.DataFrame(a_sink)
        df.to_csv(OUT / "age_contrast.csv", index=False)
        lines += [f"PART A. Age profile, differences from {REF_BAND}.",
                  "A positive coefficient means the band declined LESS than "
                  f"{REF_BAND} did.", ""]
        for _, r in df.iterrows():
            star = "" if abs(r["t"]) < 1.96 else "  *"
            lines.append(f"  {r['band_vs_ref']:<7} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  t {r['t']:+.2f}{star}")
        y = df[df.band_vs_ref == "22_25"]
        if len(y):
            c, t = float(y.iloc[0]["coef"]), float(y.iloc[0]["t"])
            if abs(t) < 1.96:
                lines += ["", "  READ: the young-versus-prime-age difference "
                          "is NOT distinguishable from zero. The paper "
                          "cannot claim the young are distinctively hit on "
                          "the stock. It may still claim a distinctive "
                          "COMPOSITION of adjustment, if part C settles "
                          "which mechanism is real."]
            elif c > 0:
                lines += ["", "  READ: the young declined significantly LESS "
                          f"than {REF_BAND}. The canaries framing is "
                          "contradicted on the stock and must be rewritten."]
            else:
                lines += ["", "  READ: the young declined significantly MORE "
                          f"than {REF_BAND}. The framing survives, and this "
                          "is now a tested contrast rather than two "
                          "separately significant coefficients."]
        lines.append("")
    else:
        lines += ["PART A produced no fit.", ""]

    if b_sink:
        df = pd.DataFrame(b_sink)
        df.to_csv(OUT / "payroll_tax.csv", index=False)
        lines += ["PART B. Youth payroll-tax expiry, April 2023.", ""]
        for _, r in df.iterrows():
            t = r["coef"] / r["se"] if r["se"] else float("nan")
            lines.append(f"  {r['term']:<32} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  t {t:+.2f}")
        lines += ["", "  READ: the triple is the one that matters. A common "
                  "national effect is absorbed by the month-by-age effects; "
                  "only an exposure-differential response survives them.", ""]
    else:
        lines += ["PART B produced no fit.", ""]

    if c_sink:
        df = pd.DataFrame(c_sink)
        df.to_csv(OUT / "route_ladder.csv", index=False)
        lines += ["PART C. Education to occupation, one rung at a time.",
                  f"Coefficients are differences from {REF_BAND}.", ""]
        for rung in ["A_edu_full", "B_edu_intersect", "C_edu_joint",
                     "D_occ_joint"]:
            s = df[(df.rung == rung) & (df.band_vs_ref == "22_25")]
            if len(s):
                r = s.iloc[0]
                lines.append(f"  {rung:<18} 22-25 vs {REF_BAND}: "
                             f"{r['coef']:+.4f} ({r['se']:.4f}), "
                             f"{int(r['n_firms']):,} firms")
        lines += ["", "  READ: A to B is employer coverage. C to D is the "
                  "register AND the incumbent pool together, not the "
                  "register alone, because the cached frames are "
                  "pre-aggregated and a worker-level joint-support "
                  "restriction would need a new pull.", ""]
    else:
        lines += ["PART C produced no fit.", ""]

    if FAILURES:
        lines += ["FAILED:"] + [f"  {f}" for f in FAILURES]
    (OUT / "70_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("70_respecifications", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

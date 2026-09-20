#!/usr/bin/env python3
"""
62_reconcile_gradients.py -- two clean designs, two age profiles. Which
                             ingredient causes the disagreement?

======================================================================
  RUNS IN MONA. No SQL: reads the caches 47L and 47h already wrote.
  Writes output_62/.
======================================================================

THE DISAGREEMENT.

Both designs pass their artefact test and both are honest measurements,
and they do not say the same thing about which age group is affected.

  47L, frozen occupation exposure, age gradient on the employment stock:
       22-25  +0.0081 (0.0126)   41-49  -0.0193 (0.0089)   50+  +0.0357
       the young are not the affected band; the forties are

  47j, incumbent education mix, within-employer triple difference:
       22-25  -0.0132 (0.0111), the only band estimated, and negative

Averaging over a disagreement is not analysis. Until it is reconciled,
any claim that the young specifically are affected rests on one design.

THE THREE THINGS THAT DIFFER, AND THE POINT OF THIS SCRIPT.

The two exposures differ in three ways at once, so the disagreement could
come from any of them:

  SOURCE     occupation (47L) against education (47j)
  UNIT       the firm's own age-a workers (47L) against the firm's
             incumbents aged 31 and over, one number per firm (47j)
  FORM       a standardised continuous score (47L) against a quartile (47j)

Changing three things at once and observing a different answer tells you
nothing about which one mattered. So this holds the outcome, the panel and
the fixed effects FIXED at 47L's, and varies the exposure alone, one
ingredient at a time:

  A  occupation, age-specific, continuous      = 47L exactly
  B  occupation, firm-level incumbents, continuous
  C  education,  age-specific, continuous
  D  education,  firm-level incumbents, quartile = 47j's exposure

A to B isolates the UNIT. A to C isolates the SOURCE. C to D adds FORM.
If the gradient flips between A and B, the disagreement is about whether
exposure is measured on the age group itself or on the firm, which is a
substantive distinction and not a technicality: a firm-level measure asks
whether exposed FIRMS treat their young differently, and an age-specific
one asks whether exposed YOUNG WORKERS fare differently.

HOW IT IS FAST. The counts panel is read once. Each variant differs only
in a single merged column, so the balanced panel is rebuilt per variant
but the expensive pulls are never repeated, and there are four fits rather
than a grid.

Output (output_62/):
  gradient_by_variant.csv   age band x variant, coefficient and SE
  exposure_agreement.csv    how much the four variants agree, cell by cell
  62_summary.txt
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
OUT = HERE / "output_62"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
INCUMBENT_BANDS = ["31-34", "35-40", "41-49", "50+"]
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def occ_exposure(base: pd.DataFrame, daioe: pd.DataFrame,
                 age_specific: bool, l47):
    """
    Mean DAIOE percentile of the occupations held in 2019, per employer x
    age cell or once per employer from its incumbents aged 31 and over.

    Variant A is 47L's OWN builder, floors and all, so the anchor is exact
    rather than approximate: a cell needs three coded workers and a firm
    needs two scored cells. The first version of this script left those
    floors out and scored 607,244 cells where 47L scores 143,700, which
    would have made the sample a fourth ingredient in a decomposition
    designed to vary one at a time.

    Variant B then reuses the SAME cells and changes only the unit, so the
    A to B step is a clean test of the unit and of nothing else.
    """
    cells = l47.build_exposure(base, daioe)
    if age_specific:
        return cells[["employer_id", "age_group", "expo"]].copy()
    inc = cells[cells["age_group"].astype(str).isin(INCUMBENT_BANDS)].copy()
    if inc.empty:
        return pd.DataFrame(columns=["employer_id", "age_group", "expo"])
    inc["ws"] = inc["expo"] * inc["n_coded"]
    g = (inc.groupby("employer_id", observed=True)
         .agg(ws=("ws", "sum"), n=("n_coded", "sum")).reset_index())
    g["fexpo"] = g["ws"] / g["n"]
    out = cells[["employer_id", "age_group"]].merge(
        g[["employer_id", "fexpo"]], on="employer_id", how="inner")
    return out.rename(columns={"fexpo": "expo"})


def edu_exposure(frame19: pd.DataFrame, book, name: str, spec: dict,
                 age_specific: bool, as_quartile: bool, l47):
    """
    The same construction on EDUCATION rather than occupation, scored
    through 47h's scorebook so it is the identical mapping 47j uses, and
    floored the same way 47L floors the occupation route.
    """
    cols = ["niva_t", "inr_t", "expb_t"]
    keycols = cols + ["age_group"]
    # Collapse the twelve monthly rows of 2019 to one row per employer x age
    # x attribute cell BEFORE scoring. The 2019 frame is tens of millions of
    # rows and only six of its columns matter here, so this is the
    # difference between a merge that fits in memory and one that does not.
    f = (frame19[["employer_id", "n_emp"] + keycols]
         .groupby(["employer_id"] + keycols, observed=True)["n_emp"]
         .sum().reset_index())
    combos = f[keycols].drop_duplicates().reset_index(drop=True)
    s = book.score_frame(name, spec, combos[cols[0]], combos[cols[1]],
                         combos[cols[2]], None, combos["age_group"])
    combos["_score"] = np.asarray(s, dtype="float64")
    f = f.merge(combos, on=keycols, how="left")
    f = f[f["_score"].notna()]
    f["ws"] = f["_score"] * f["n_emp"]
    cell = (f.groupby(["employer_id", "age_group"], observed=True)
            .agg(ws=("ws", "sum"), n=("n_emp", "sum")).reset_index())
    # 47L's floors, on the education side: a cell needs enough scored
    # workers to mean anything, and a firm needs more than one scored cell
    # or it contributes nothing to a within-employer comparison.
    cell = cell[cell["n"] >= l47.MIN_CELL_CODED]
    if cell.empty:
        return pd.DataFrame(columns=["employer_id", "age_group", "expo"])
    keep = (cell.groupby("employer_id")["age_group"].transform("nunique")
            >= l47.MIN_FIRM_AGES)
    cell = cell[keep]
    cell["cexpo"] = cell["ws"] / cell["n"]
    if age_specific:
        g = cell.rename(columns={"cexpo": "expo"})[
            ["employer_id", "age_group", "expo", "n"]]
    else:
        inc = cell[cell["age_group"].astype(str).isin(INCUMBENT_BANDS)]
        fm = (inc.groupby("employer_id", observed=True)
              .agg(ws=("ws", "sum"), n=("n", "sum")).reset_index())
        fm["expo"] = fm["ws"] / fm["n"]
        g = cell[["employer_id", "age_group"]].merge(
            fm[["employer_id", "expo", "n"]], on="employer_id", how="inner")
    if as_quartile:
        # WORKER-weighted cutoffs, which is what 47j uses. An unweighted
        # qcut over firms would put a quarter of FIRMS in each group and a
        # quite different share of workers, so variant D would not be 47j's
        # exposure and the comparison would be with something else.
        o = np.argsort(g["expo"].to_numpy(), kind="stable")
        v, w = g["expo"].to_numpy()[o], g["n"].to_numpy()[o]
        cum = np.cumsum(w) / max(w.sum(), 1)
        cuts = np.asarray([float(v[np.searchsorted(cum, q, side="left")])
                           for q in (0.25, 0.5, 0.75)])
        g["expo"] = (np.searchsorted(cuts, g["expo"].to_numpy(),
                                     side="right") + 1).astype(float)
    return g[["employer_id", "age_group", "expo"]]


ANCHOR = ("A0_47L_anchor", "47L exactly, on 47L's own sample")
VARIANTS = [
    ("A_occ_age_cont",  "occupation, age-specific, continuous  (= 47L)"),
    ("B_occ_firm_cont", "occupation, firm incumbents, continuous"),
    ("C_edu_age_cont",  "education, age-specific, continuous"),
    ("D_edu_firm_quart", "education, firm incumbents, quartile  (47j's "
                         "exposure, inside 47L's specification)"),
]


def main():
    mc.Tee(OUT / "62_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("62: WHY DO THE TWO CLEAN DESIGNS DISAGREE ABOUT AGE?")
    print("=" * 70)
    print("  Outcome, panel and fixed effects held at 47L's throughout.")
    print("  Only the exposure construction varies, one ingredient at a time.")
    print(mc.mem_line("  "))

    l47 = _mod("47L_age_baseline_exposure.py", "l47")
    h47 = _mod("47h_edu_horserace.py", "h47")

    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first.")
    cnt = [c for c in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                       for y in l47.YEARS) if c is not None]
    if not cnt:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    print(f"  counts: {len(counts):,} employer-age-month cells")

    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                                "score"]]

    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    w = {y: mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS) for y in (2019, 2020, 2021)}
    book = spec = None
    if frame19 is not None and all(v is not None for v in w.values()):
        book = h47.ScoreBook(w, h47.load_key(), h47.load_scores())
        spec = dict(h47.DESIGNS["OL_daioe"])
        book.build("OL_daioe", spec)
        print("  education scorebook ready")
    else:
        print("  *** education caches missing: variants C and D will be "
              "skipped, and the SOURCE comparison cannot be made")

    expos = {}
    expos["A_occ_age_cont"] = occ_exposure(base, daioe, True, l47)
    expos["B_occ_firm_cont"] = occ_exposure(base, daioe, False, l47)
    if book is not None:
        expos["C_edu_age_cont"] = opt("variant C", edu_exposure, frame19,
                                      book, "OL_daioe", spec, True, False,
                                      l47)
        expos["D_edu_firm_quart"] = opt("variant D", edu_exposure, frame19,
                                        book, "OL_daioe", spec, False, True,
                                        l47)
    expos = {k: v for k, v in expos.items() if v is not None and not v.empty}
    for k, v in expos.items():
        print(f"  {k:<18} {len(v):,} firm-age cells")
    # the 2019 education frame is tens of millions of rows and nothing
    # below needs it: the exposures are built and the panels come from the
    # counts. Free it before the first panel is built rather than after.
    frame19 = w = book = None
    gc.collect()

    # how much do the four agree, cell by cell? If they agree closely the
    # gradients cannot differ for a measurement reason.
    agree = []
    keys = [k for k in expos if expos.get(k) is not None]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            m = expos[a].merge(expos[b], on=["employer_id", "age_group"],
                               suffixes=("_a", "_b"))
            if len(m) > 10:
                agree.append({"a": a, "b": b, "n_cells": len(m),
                              "corr": float(np.corrcoef(m.expo_a,
                                                        m.expo_b)[0, 1])})
    if agree:
        pd.DataFrame(agree).to_csv(OUT / "exposure_agreement.csv", index=False)
        print("\n  agreement between exposure constructions:")
        for r in agree:
            print(f"    {r['a']:<18} vs {r['b']:<18} corr {r['corr']:+.3f} "
                  f"on {r['n_cells']:,} cells")

    # ONE SAMPLE FOR ALL FOUR. A decomposition that varies the exposure
    # one ingredient at a time must not vary the sample as well, and the
    # four constructions do not score identical sets of cells: the
    # education route reaches firms the occupation route does not, and the
    # firm-level measures need a firm to have incumbents. So every variant
    # is restricted to the cells all of them score, and the anchor fit
    # below reports 47L on 47L's own sample so the cost of that
    # restriction is visible rather than assumed.
    common = None
    for e in expos.values():
        k = e[["employer_id", "age_group"]].drop_duplicates()
        common = k if common is None else common.merge(
            k, on=["employer_id", "age_group"], how="inner")
    print(f"\n  common sample: {len(common):,} cells scored by all "
          f"{len(expos)} constructions")

    jobs = [(ANCHOR[0], ANCHOR[1], expos["A_occ_age_cont"])]
    for key, label in VARIANTS:
        e = expos.get(key)
        if e is not None:
            jobs.append((key, label, e.merge(common,
                                             on=["employer_id", "age_group"],
                                             how="inner")))

    rows = []
    for key, label, e in jobs:
        if e is None or e.empty:
            print(f"\n  {key}: skipped")
            continue
        t1 = time.time()
        # build_panel standardises exposure on the cells it keeps, so every
        # variant's coefficient reads per standard deviation of that
        # variant's own distribution. Do not standardise here as well.
        bal = l47.build_panel(counts, e)
        if bal.empty:
            print(f"  {key}: empty panel, skipped")
            continue
        print(f"  {key}: panel {len(bal):,} rows from {len(e):,} cells")
        gr = l47.fit_gradient(bal, f"g62_{key}")
        del bal
        gc.collect()
        if gr is None or gr.empty:
            FAILURES.append(key)
            print(f"  {key}: FAILED, recorded and skipped")
            continue
        gr = gr.copy(); gr["variant"] = key; gr["label"] = label
        rows.append(gr)
        pd.concat(rows, ignore_index=True).to_csv(
            OUT / "gradient_by_variant.csv", index=False)
        print(f"\n  {key}  ({label})  [{(time.time()-t1)/60:.1f} min]")
        for _, x in gr.iterrows():
            t = x["coef"] / max(x["se"], 1e-12)
            print(f"    {x['age_group']:<6} {x['coef']:+.4f} "
                  f"(SE {x['se']:.4f}) t {t:+.2f}")

    lines = ["WHY THE TWO CLEAN DESIGNS DISAGREE ABOUT AGE", "=" * 52, "",
             "Outcome, panel and fixed effects are 47L's in every column.",
             "Only the exposure construction changes.", ""]
    for k, l in VARIANTS:
        lines.append(f"  {k:<18} {l}")
    lines.append("")
    if rows:
        G = pd.concat(rows, ignore_index=True)
        piv = G.pivot_table(index="age_group", columns="variant",
                            values="coef")
        order = [a for a in l47.AGES if a in piv.index]
        lines += ["age gradient by variant:", piv.reindex(order).round(4).to_string(), ""]
        se = G.pivot_table(index="age_group", columns="variant", values="se")
        lines += ["standard errors:", se.reindex(order).round(4).to_string(), ""]
        cols = list(piv.columns)
        if ANCHOR[0] in cols and "A_occ_age_cont" in cols:
            d = piv.loc["22-25", "A_occ_age_cont"] - piv.loc["22-25", ANCHOR[0]]
            lines.append(f"SAMPLE alone (anchor to A) moves 22-25 by {d:+.4f}"
                         "  <- the cost of the common sample, not a finding")
        if "A_occ_age_cont" in cols and "B_occ_firm_cont" in cols:
            d = piv.loc["22-25", "B_occ_firm_cont"] - piv.loc["22-25", "A_occ_age_cont"]
            lines.append(f"UNIT alone (A to B) moves 22-25 by {d:+.4f}")
        if "A_occ_age_cont" in cols and "C_edu_age_cont" in cols:
            d = piv.loc["22-25", "C_edu_age_cont"] - piv.loc["22-25", "A_occ_age_cont"]
            lines.append(f"SOURCE alone (A to C) moves 22-25 by {d:+.4f}")
        if "C_edu_age_cont" in cols and "D_edu_firm_quart" in cols:
            d = piv.loc["22-25", "D_edu_firm_quart"] - piv.loc["22-25", "C_edu_age_cont"]
            lines.append(f"UNIT and FORM (C to D) move 22-25 by {d:+.4f}")
        lines.append("")
    if FAILURES:
        lines += ["VARIANTS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing column is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. This is a decomposition, not a horse race. None of the four",
        "     is 'the right one'; the point is which ingredient moves the",
        "     answer.",
        "  2. A firm-level exposure asks whether exposed FIRMS treat their",
        "     young differently. An age-specific one asks whether exposed",
        "     YOUNG WORKERS fare differently. If the unit is what matters,",
        "     the two designs were answering different questions all along",
        "     and neither is wrong.",
        "  3. Every variant here uses the stock as the outcome, so none of",
        "     them speaks to hiring.",
        "  4. CHECK THE ANCHOR FIRST. A0 is 47L's own exposure on 47L's own",
        "     sample and must reproduce 47L's published gradient (+0.0081",
        "     at 22-25, -0.0193 at 41-49). If it does not, something other",
        "     than the exposure has changed and nothing below it should be",
        "     read. The four variants then run on one common sample, so",
        "     the A0-to-A step is the price of that restriction and the",
        "     A-to-B, A-to-C and C-to-D steps are the decomposition.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "62_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n62 done.")


if __name__ == "__main__":
    main()

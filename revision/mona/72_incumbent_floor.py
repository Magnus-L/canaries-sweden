#!/usr/bin/env python3
"""
72_incumbent_floor.py -- is the 31+ incumbent mix a good proxy for what
                         a firm's YOUNG workers actually do?

======================================================================
  RUNS IN MONA. NO SQL. Reads 47h's cached education frames and 47L's
  cached counts. Writes output_72/.
======================================================================

THE QUESTION, WHICH IS ML'S AND IS A GOOD ONE.

Every headline in the revision scores a firm from the 2019 education mix
of its incumbents aged 31 and over. The floor is not a taste: 31 is the
LOWEST floor that keeps both studied bands, 22-25 and 26-30, out of their
own treatment. Let a young worker's own record into the exposure measure
and the circularity the as-of backtest just killed comes straight back.

But the floor buys that immunity at a price, and the price is what this
script measures. If a firm's young workers do different work from its
older ones, trainees and assistants against the professionals who employ
them, then the 31+ mix is a NOISY proxy for whether the young in that
firm are exposed. Classical measurement error in a binary treatment
attenuates toward zero, so the direction is knowable: our estimates would
be conservative rather than inflated. That is worth being able to say,
and it is worth QUANTIFYING rather than asserting.

Note what this script does not do. It does not attenuation-correct
anything. The measurement error here is not classical, the correction
would be a fiction, and the honest move is to report the reliability and
let the reader see how much room it leaves.

WHAT IT MEASURES

  A  RELIABILITY, descriptive, no estimation. For every firm, the
     education-based exposure of its 22-25 workers, of its 26-30
     workers, and of its 31+ incumbents. Then the correlation between
     young and incumbent scores, the share of firms landing in the same
     quartile on both, and the share that flip between the top and
     bottom quartile. Broken out by firm size, because the concern is
     sharpest where a handful of workers decide the mix.

  B  LEAVE ONE BAND OUT. For band a, exposure built from every band
     EXCEPT a. Estimating 22-25 then uses 26-30 and everyone older, so
     representation improves while the estimated band still never enters
     its own treatment. The headline is re-estimated on that measure.

  C  THE ARTEFACT ON B. Letting 26-30's records into exposure admits
     staler ones, so the leave-one-out measure has to carry its own
     as-of arm. A measure is not usable here merely because it is better
     represented.

THE READ RULE, FIXED BEFORE THE RUN

  1. PROXY GOOD if the young-to-incumbent correlation is at least
     R_GOOD and quartile agreement at least Q_GOOD. Then the floor costs
     little, say so in one sentence and keep the headline as it is.

  2. PROXY WEAK if the correlation is below R_WEAK. Then the headline is
     attenuated, and the leave-one-out estimate should be LARGER in
     absolute value. If it is not larger, attenuation is not the story
     and something else is going on, which must be chased rather than
     written around.

  3. Anything between is PARTIAL: report the reliability beside the
     estimate and claim nothing further.

  4. The leave-one-out estimate is USABLE only if its artefact stays
     under ARTEFACT_MAX, the project's standing threshold. Better
     representation does not buy a pass on staleness.

Output (output_72/):
  reliability.csv    Part A, overall and by size
  loo_headline.csv   Part B, with the 31+ headline beside it
  loo_artefact.csv   Part C
  72_summary.txt
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
OUT = HERE / "output_72"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YOUNG_BANDS = ["22-25", "26-30"]
INCUMBENT_BANDS = ["31-34", "35-40", "41-49", "50+"]
PANEL_FROM = "2021-01"
PANEL_YEARS = list(range(2021, 2026))
POOLED_FROM = "2024-01"
TRUNC = 2021
DESIGN = "OL_daioe"

# Read-rule thresholds, fixed here before the run.
R_GOOD, Q_GOOD = 0.70, 0.60
R_WEAK = 0.40
ARTEFACT_MAX = 0.05
SIZE_CUTS = [0, 10, 50, 250, np.inf]
SIZE_LABS = ["1-9", "10-49", "50-249", "250+"]
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(label)
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def edu_inputs(j47):
    """47h's scorebook, spec and 2019 frame, exactly as 61 assembles them."""
    h47 = j47._h47()
    counts = {}
    for y in h47.WEIGHT_YEARS:
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise SystemExit(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[DESIGN])
    book.build(DESIGN, spec)
    frame19 = mc.read_cache(CACHE / f"edu_hr_{j47.BASE_YEAR}.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise SystemExit(f"edu_hr_{j47.BASE_YEAR}.parquet missing: run 47h.")
    return book, spec, frame19


def exposure_on(j47, frame19, book, spec, bands, arm="true"):
    """
    47j's incumbent_exposure over an ARBITRARY band set.

    incumbent_exposure reads the band list from its own module global, so
    the set is swapped around the call and restored in a finally. That is
    deliberate: reimplementing the function here would let the two copies
    drift, and the whole value of this script is that its exposure is
    built by the same code as the headline's, differing only in which
    workers are counted.
    """
    saved = j47.INCUMBENT_BANDS
    try:
        j47.INCUMBENT_BANDS = list(bands)
        expo, cuts = j47.incumbent_exposure(frame19, book, DESIGN, spec,
                                            arm, TRUNC)
    finally:
        j47.INCUMBENT_BANDS = saved
    return expo, cuts


def part_a(j47, frame19, book, spec, sink):
    """Reliability of the 31+ mix as a proxy for the firm's young."""
    inc, _ = exposure_on(j47, frame19, book, spec, INCUMBENT_BANDS)
    if inc is None or inc.empty:
        print("  A: no incumbent exposure"); return
    inc = inc.rename(columns={"fq": "fq_inc", "mix": "mix_inc",
                              "n": "n_inc"})
    size = (frame19.groupby("employer_id", observed=True)["n_emp"].sum()
            .reset_index().rename(columns={"n_emp": "size19"}))
    for band in YOUNG_BANDS:
        y, _ = exposure_on(j47, frame19, book, spec, [band])
        if y is None or y.empty:
            print(f"  A {band}: the young band scores no firms"); continue
        y = y.rename(columns={"fq": "fq_y", "mix": "mix_y", "n": "n_y"})
        m = inc.merge(y, on="employer_id", how="inner").merge(
            size, on="employer_id", how="left")
        if len(m) < 30:
            print(f"  A {band}: only {len(m)} firms score on both"); continue
        m["szg"] = pd.cut(m["size19"], bins=SIZE_CUTS, labels=SIZE_LABS,
                          right=False)

        def row(d, label):
            if len(d) < 30 or d["mix_y"].nunique() < 2:
                return None
            return {
                "young_band": band, "group": label, "n_firms": int(len(d)),
                "pearson": float(d["mix_y"].corr(d["mix_inc"])),
                "spearman": float(d["mix_y"].corr(d["mix_inc"],
                                                  method="spearman")),
                "same_quartile": float((d["fq_y"] == d["fq_inc"]).mean()),
                "top_vs_bottom_flip": float(
                    (((d["fq_y"] == 4) & (d["fq_inc"] == 1))
                     | ((d["fq_y"] == 1) & (d["fq_inc"] == 4))).mean()),
                "median_young_coded": float(d["n_y"].median()),
            }

        r = row(m, "all")
        if r:
            sink.append(r)
            print(f"  A {band}: {len(m):,} firms, r {r['pearson']:.3f}, "
                  f"same quartile {r['same_quartile']:.1%}")
        for lab in SIZE_LABS:
            r = row(m[m["szg"] == lab], lab)
            if r:
                sink.append(r)
        # how many firms the young band cannot score AT ALL is part of the
        # answer: a proxy is also weak where the thing it proxies is absent
        missing = len(set(inc["employer_id"]) - set(y["employer_id"]))
        sink.append({"young_band": band, "group": "unscorable_young",
                     "n_firms": missing, "pearson": np.nan,
                     "spearman": np.nan, "same_quartile": np.nan,
                     "top_vs_bottom_flip": np.nan,
                     "median_young_coded": np.nan})


def fit_headline(counts, expo, band, j47, tag):
    """61's pooled headline on a supplied exposure. One fit."""
    l61 = _mod("61_redated_triple.py", "l61")
    skel = l61.build_skeleton(counts, band, j47)
    if skel.empty:
        return None
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id",
                   how="inner")
    del skel
    gc.collect()
    if b.empty:
        return None
    b["high"] = (b["fq"] == 4).astype(int)
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    b["post_rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    b["post_x_high_x_young"] = (ym >= POOLED_FROM).astype(int) * hy
    n_firms = int(b["employer_id"].nunique())
    print(f"    {tag}: panel {len(b):,} rows, {n_firms:,} firms"
          f"{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag=f"r72_{tag}",
                            terms=["post_rb_x_high_x_young",
                                   "post_x_high_x_young"],
                            fes=j47.FES)
    del b
    gc.collect()
    if r.empty:
        FAILURES.append(tag)
        return None
    row = r[r["term"] == "post_x_high_x_young"]
    if row.empty:
        return None
    return {"coef": float(row.iloc[0]["coef"]), "se": float(row.iloc[0]["se"]),
            "n_firms": n_firms}


def part_bc(counts, j47, frame19, book, spec, sink, art_sink):
    """Leave-one-band-out headline, and its own as-of artefact."""
    for band in YOUNG_BANDS:
        loo = [b for b in YOUNG_BANDS + INCUMBENT_BANDS if b != band]
        for label, bands in (("incumbent31", INCUMBENT_BANDS),
                             ("leave_one_out", loo)):
            e, _ = exposure_on(j47, frame19, book, spec, bands)
            if e is None or e.empty:
                print(f"  B {band}/{label}: no exposure"); continue
            r = fit_headline(counts, e, band, j47, f"{label}_{band}")
            if r:
                r.update({"young_band": band, "measure": label,
                          "bands_used": "+".join(bands)})
                sink.append(r)
            if label != "leave_one_out":
                continue
            # C: the as-of arm on the SAME band set, so the artefact is
            # measured for the measure actually being proposed
            ea, _ = exposure_on(j47, frame19, book, spec, bands, arm="asof")
            if ea is None or ea.empty:
                continue
            ra = fit_headline(counts, ea, band, j47, f"asof_{band}")
            if ra and r:
                art_sink.append({"young_band": band,
                                 "true": r["coef"], "asof": ra["coef"],
                                 "artefact": ra["coef"] - r["coef"]})


def verdict(rel, loo, art) -> list:
    """
    The pre-committed read rule, evaluated.

    Part B or C can fail on their own without invalidating Part A, and a
    reliability number is worth reporting even when no fit came back. So
    every frame is probed for the column before it is filtered: an empty
    DataFrame has no `young_band` attribute and would otherwise take the
    whole summary down with an AttributeError.
    """
    def has(df, col="young_band"):
        return df is not None and len(df) and col in df.columns

    out = []
    for band in YOUNG_BANDS:
        d = (rel[(rel.young_band == band) & (rel.group == "all")]
             if has(rel) else pd.DataFrame())
        if d.empty:
            out.append(f"{band}: reliability UNAVAILABLE"); continue
        r = float(d.iloc[0]["pearson"]); q = float(d.iloc[0]["same_quartile"])
        if r >= R_GOOD and q >= Q_GOOD:
            v = ("PROXY GOOD. The 31+ mix represents the firm's young well; "
                 "the floor costs little and the headline stands as it is.")
        elif r < R_WEAK:
            v = ("PROXY WEAK. The headline is attenuated, so the "
                 "leave-one-out estimate should be LARGER in absolute value.")
        else:
            v = ("PARTIAL. Report the reliability beside the estimate and "
                 "claim nothing further.")
        out.append(f"{band}: r {r:.3f}, same quartile {q:.1%}. {v}")

        a = (loo[(loo.young_band == band) & (loo.measure == "incumbent31")]
             if has(loo) else pd.DataFrame())
        b = (loo[(loo.young_band == band)
                 & (loo.measure == "leave_one_out")]
             if has(loo) else pd.DataFrame())
        if len(a) and len(b):
            ca, cb = float(a.iloc[0]["coef"]), float(b.iloc[0]["coef"])
            out.append(f"{band}: 31+ {ca:+.4f} ({float(a.iloc[0]['se']):.4f}), "
                       f"leave-one-out {cb:+.4f} "
                       f"({float(b.iloc[0]['se']):.4f})")
            if r < R_WEAK and abs(cb) <= abs(ca):
                out.append(f"{band}: the proxy is weak but the leave-one-out "
                           f"estimate is NOT larger. Attenuation does not "
                           f"explain this and it needs chasing.")
        s = art[art.young_band == band] if has(art) else pd.DataFrame()
        if len(s):
            av = float(s.iloc[0]["artefact"])
            out.append(
                f"{band}: leave-one-out artefact {av:+.4f}. "
                + ("USABLE." if abs(av) < ARTEFACT_MAX else
                   f"NOT USABLE: at or above {ARTEFACT_MAX}, so the better "
                   f"representation was bought with staleness."))
    return out


def main():
    mc.Tee(OUT / "72_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("72 incumbent floor: is the 31+ mix a good proxy for the young?")
    print(f"read rule fixed before the run: good r>={R_GOOD} and "
          f"q>={Q_GOOD}; weak r<{R_WEAK}; artefact<{ARTEFACT_MAX}")
    print("=" * 70)

    j47 = _mod("47j_within_employer_triple.py", "j47")
    book, spec, frame19 = edu_inputs(j47)
    print(f"  exposure frame: {len(frame19):,} cells (2019)")

    cnt = []
    for y in PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise SystemExit(f"L_counts_{y}.parquet is not cached. This "
                             f"script performs no SQL; run 47L first.")
        cnt.append(c)
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(counts["year_month"].max())
    if last < POOLED_FROM:
        raise SystemExit(f"counts end at {last}, before {POOLED_FROM}.")
    print(f"  counts to {last}, {len(counts):,} cells")

    rel_sink, loo_sink, art_sink = [], [], []
    opt("part A", part_a, j47, frame19, book, spec, rel_sink)
    opt("part B/C", part_bc, counts, j47, frame19, book, spec, loo_sink,
        art_sink)

    rel = pd.DataFrame(rel_sink)
    loo = pd.DataFrame(loo_sink)
    art = pd.DataFrame(art_sink)
    for df, nm in ((rel, "reliability.csv"), (loo, "loo_headline.csv"),
                   (art, "loo_artefact.csv")):
        if len(df):
            df.to_csv(OUT / nm, index=False)

    lines = ["72 incumbent floor", "=" * 70, "",
             "The 31+ floor is forced, not chosen: 31 is the lowest floor "
             "that keeps BOTH 22-25 and 26-30 out of their own treatment. "
             "This measures what that costs.", ""]
    if len(rel):
        lines += ["RELIABILITY of the 31+ mix as a proxy for the young", ""]
        for _, r in rel[rel.group != "unscorable_young"].iterrows():
            lines.append(
                f"  {r['young_band']:<7} {str(r['group']):<8} "
                f"n {int(r['n_firms']):>7,}  r {r['pearson']:+.3f}  "
                f"rho {r['spearman']:+.3f}  same quartile "
                f"{r['same_quartile']:.1%}  top-bottom flip "
                f"{r['top_vs_bottom_flip']:.1%}")
        for _, r in rel[rel.group == "unscorable_young"].iterrows():
            lines.append(f"  {r['young_band']:<7} firms the young band "
                         f"cannot score at all: {int(r['n_firms']):,}")
        lines.append("")
    if len(rel) or len(loo):
        lines += verdict(rel, loo, art) + [""]
    lines += ["No attenuation correction is applied anywhere here. The "
              "measurement error is not classical, so a correction would "
              "be a fiction; the reliability is reported and the reader "
              "can see how much room it leaves.", ""]
    if FAILURES:
        lines += ["FAILED:"] + [f"  {f}" for f in FAILURES]
    (OUT / "72_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("72_incumbent_floor", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
77_contrast_by_track.py -- the young against the prime-aged, inside each
                           broad education track.

======================================================================
  RUNS IN MONA. No SQL: reads the L_counts_sex_edu_YYYY caches that 76
  wrote (run 76 first, or in the same lane ahead of this). Six fits,
  three to five hours. Writes output_77/.
======================================================================

WHY (ML, 22 Sep 2026, 01:55). With the calendar cycle removed, the
22-25 and 26-30 bands are not distinguishable from 41-49 inside exposed
firms (lane 20: -0.0099 (0.0121) and -0.0096 (0.0084) on 172,396
firms). That is a pooled null with standard errors of a hundredth, and
it says nothing about workers in particular tracks: a steeper youth
gradient among, say, ICT graduates is neither claimed nor ruled out by
it. This script asks the question track by track, so the paper can say
in one sentence what the cut shows and the appendix can show it.

THE DESIGN. 74's seasonal arm on the THREE-band panel (22-25, 26-30,
41-49 as the omitted reference), the specification that produced the
-0.0153 (0.0126) the appendix quotes on 120,359 firms: employer-by-month,
employer-by-age and month-by-age effects, one post and one Riksbank term
per young band and three quarter-of-year terms per young band, Q4
omitted. Run first on all workers, as a reproduction gate against
-0.0153, then on the workers of each track.

THE READ RULE, FIXED BEFORE THE RUN.

  * GATE: the all-worker 22-25 contrast must land within one standard
    error of -0.0153 or the panel is not the paper's.
  * The cut is HETEROGENEITY. Every track is reported, sign and size,
    including the nulls. No track is promoted to a headline, and the
    paper's sentence on the pooled profile stands whatever the cut shows.
  * A track contrast beyond two standard errors is reported as such in
    the OA with its standard error and its firm count, with the
    carried-forward education record stated beside it.

Output (output_77/):
  contrast_by_track.csv   both young bands against 41-49, per track
  vcov_s77_*.csv          clustered covariances
  77_summary.txt
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
OUT = HERE / "output_77"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BANDS = ["22-25", "26-30", "41-49"]
REF_BAND = "41-49"
YOUNG_BANDS = ["22-25", "26-30"]
POST_FROM = "2024-01"
PANEL_FROM = "2021-01"
S74_2225 = (-0.0153, 0.0126)             # the three-band seasonal contrast (gate)
FAILURES = []
GATE = {"ok": None, "detail": ""}


def _mod(fname, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def quarter_of_year(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


def band_col(stem: str, band: str) -> str:
    return f"{stem}_{band.replace('-', '_').replace('+', 'p')}"


def three_band_skeleton(counts: pd.DataFrame) -> pd.DataFrame:
    """
    61's balanced employer x band x month panel on BANDS: a firm enters if
    it holds the reference band and at least one young band; cells are
    zero-filled over the firm's months; dead firm-bands (zero in every
    month) are dropped, since the employer-by-age effect predicts them
    exactly; the three fixed effects are integer codes as in 61.
    """
    p = counts[counts["age_group"].astype(str).isin(BANDS)]
    p = p[p["year_month"].astype(str) >= PANEL_FROM]
    p = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
         ["n_emp"].sum().reset_index())
    p["age_group"] = p["age_group"].astype(str)
    p["year_month"] = p["year_month"].astype(str)
    have = p.groupby("employer_id")["age_group"].agg(set)
    keep = have[have.apply(lambda v: REF_BAND in v
                           and bool(v & set(YOUNG_BANDS)))].index
    p = p[p["employer_id"].isin(keep)]
    if p.empty:
        return p
    months = sorted(p["year_month"].unique())
    emp = p["employer_id"].drop_duplicates().to_numpy()
    full = pd.MultiIndex.from_product([emp, BANDS, months],
                                      names=["employer_id", "age_group",
                                             "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index())
    bal["n_emp"] = bal["n_emp"].astype(int)
    alive = bal.groupby(["employer_id", "age_group"], observed=True)["n_emp"] \
        .transform("sum") > 0
    bal = bal[alive]
    nb = bal.groupby("employer_id", observed=True)["age_group"].transform("nunique")
    bal = bal[nb >= 2]
    if bal.empty:
        return bal
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"], sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def contrast_terms(b: pd.DataFrame) -> tuple:
    """74's seasonal arm: per young band, post and Riksbank interactions
    and three quarter terms; everything is a difference from 41-49."""
    ym = b["year_month"].astype(str)
    post = (ym >= POST_FROM).astype(int)
    post_rb = (ym >= mc.RIKSBANK_YM).astype(int)
    q = quarter_of_year(ym)
    terms = []
    for band in YOUNG_BANDS:
        d = (b["age_group"] == band).astype(int)
        c1, c2 = band_col("gpt_x_high", band), band_col("rb_x_high", band)
        b[c1] = post * b["high"] * d
        b[c2] = post_rb * b["high"] * d
        terms += [c1, c2]
        for qq in (1, 2, 3):
            c = band_col(f"q{qq}_x_high", band)
            b[c] = (q == qq).astype(int) * b["high"] * d
            terms.append(c)
    return b, terms


def fit_contrast(counts: pd.DataFrame, expo: pd.DataFrame, j47, tag: str):
    skel = three_band_skeleton(counts)
    if skel.empty:
        print(f"  {tag}: empty skeleton")
        return None
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    del skel
    gc.collect()
    if b.empty:
        return None
    b["high"] = (b["fq"] == 4).astype(int)
    b, terms = contrast_terms(b)
    n_firms = int(b["employer_id"].nunique())
    print(f"  {tag}: {len(b):,} rows, {n_firms:,} firms{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=j47.FES)
    del b
    gc.collect()
    if r.empty:
        return None
    g = r.set_index("term")
    out = {"n_firms": n_firms}
    for band in YOUNG_BANDS:
        c = band_col("gpt_x_high", band)
        if c in g.index:
            out[band] = (float(g.loc[c, "coef"]), float(g.loc[c, "se"]),
                         str(g.loc[c].get("status", "ok")))
    return out if any(bd in out for bd in YOUNG_BANDS) else None


def main():
    mc.Tee(OUT / "77_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("77: THE YOUNG AGAINST 41-49, BY EDUCATION TRACK")
    print("=" * 70)
    print(mc.mem_line("  "))

    s76 = _mod("76_gender_decomposition.py", "s76")
    s61 = _mod("61_redated_triple.py", "s61")
    j47 = s61._j47()
    h47 = j47._h47()

    wt = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        wt[y] = w
    book = h47.ScoreBook(wt, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    expo, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, "true",
                                     s61.TRUNC)
    print(f"  exposure: {len(expo):,} firms")
    del frame19
    gc.collect()

    frames = []
    for y in s76.YEARS:
        c = mc.read_cache(CACHE / f"L_counts_sex_edu_{y}.parquet",
                          require=s76.EDU_COLS + ["n_emp"])
        if c is None:
            raise RuntimeError(f"L_counts_sex_edu_{y}.parquet missing: run 76 "
                               f"first. This script performs no SQL.")
        frames.append(s76.tag_frame(c, h47))
        del c
        gc.collect()
    allf = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    rows = []

    def record(track, res):
        for band in YOUNG_BANDS:
            if band in res:
                c, se, st = res[band]
                rows.append({"track": track, "band_vs_ref": band, "coef": c,
                             "se": se, "n_firms": res["n_firms"], "status": st})
        pd.DataFrame(rows).to_csv(OUT / "contrast_by_track.csv", index=False)

    res = fit_contrast(s76.collapse(allf), expo, j47, "s77_all")
    if res is None:
        FAILURES.append("contrast/all")
    else:
        record("all", res)
        c, se, _ = res.get("22-25", (np.nan, np.nan, ""))
        GATE["ok"] = abs(c - S74_2225[0]) <= max(se, S74_2225[1])
        GATE["detail"] = (f"all-worker 22-25 vs 41-49 {c:+.4f} ({se:.4f}) against "
                          f"lane 16/20's {S74_2225[0]:+.4f} ({S74_2225[1]:.4f})")
        print(f"  GATE {'PASS' if GATE['ok'] else 'FAIL'}: {GATE['detail']}")
    for g in s76.TRACK_ORDER:
        sub = allf[allf["track"] == g]
        if sub.empty:
            print(f"  track {g}: no workers")
            continue
        res = opt(f"contrast, track {g}", fit_contrast, s76.collapse(sub), expo,
                  j47, f"s77_{g}")
        if res is None:
            FAILURES.append(f"contrast/{g}")
            continue
        record(g, res)
        for band in YOUNG_BANDS:
            if band in res:
                c, se, _ = res[band]
                print(f"    {g:<24} {band} vs 41-49 {c:+.4f} ({se:.4f}) "
                      f"t {c/max(se,1e-12):+.2f}")
    del allf
    gc.collect()

    lines = ["THE YOUNG AGAINST 41-49, BY EDUCATION TRACK", "=" * 52, "",
             "74's seasonal arm on the three-band panel, per track. A negative",
             "number means the band declined MORE than 41-49 did, inside exposed",
             "firms, with the calendar cycle removed.", ""]
    if GATE["ok"] is not None:
        lines += [f"REPRODUCTION GATE: {'PASS' if GATE['ok'] else 'FAIL'}. "
                  f"{GATE['detail']}",
                  "  A FAIL means this panel is not the paper's; quote nothing.", ""]
    for r in rows:
        star = "" if abs(r["coef"]) < 2 * r["se"] else "  *"
        lines.append(f"  {r['track']:<24} {r['band_vs_ref']} vs 41-49 "
                     f"{r['coef']:+.4f} ({r['se']:.4f}) firms {r['n_firms']:,}{star}")
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The gate must PASS.",
              "  2. This is heterogeneity. Every track is reported, including the",
              "     nulls; none becomes a headline; the pooled profile stands.",
              "  3. A starred track goes in the OA with its SE, its firm count and",
              "     the carried-forward education record stated beside it.",
              "  4. The education record for 2024-25 is the 2023 one (Individ",
              "     ends there), so the tracks are broad.", ""]
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "77_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    mc.runlog("77_contrast_by_track", 0, (time.time() - t0) / 60)
    print("\n77 done.")


if __name__ == "__main__":
    main()

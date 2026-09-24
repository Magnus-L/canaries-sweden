#!/usr/bin/env python3
"""
74_contrast_seasonal.py: the age profile against 41-49, with and without
the calendar cycle.

QUESTION
The headline compares the young with the older bands pooled. Against the
prime-aged alone, is either young band distinguishable, and does the
answer depend on the calendar cycle? The cycle is exposure-differential
and age-specific, so it can move a contrast between bands and not only a
level. This script fits the six-band profile of script 70 twice on one
panel, once without and once with the quarter-of-year terms, on the
education-route classification; scripts 82 and 85 reuse its terms on the
occupation route.

DESIGN
Panel: script 70's all_band_skeleton, employer by age band by month over
all six bands from January 2021 to June 2025, an employer entering if it
holds 41-49 and at least one other band, employer-band cells zero in
every month dropped. Exposure: the headline classification (script 70's
edu_exposure, which is script 47j's incumbent_exposure on the OL_daioe
score book). Plain arm: for every band except 41-49, PostGPT x High x Band
from January 2024 and PostRB x High x Band from April 2022. Seasonal arm:
the same plus Q1, Q2 and Q3 x High x Band per band, with the fourth
quarter omitted; the quarter terms are band-specific because High x
Quarter alone is constant within an employer-month and absorbed. Every
coefficient is a difference from the 41-49 band. Fixed effects employer by
month, employer by age, month by age; Poisson pseudo-maximum likelihood;
standard errors clustered by employer. One skeleton serves both arms, so
they differ in the term list and in nothing else. The band list is set in
this file and checked against script 70's at start, so the panel and the
terms cannot describe different age sets.

Read rule fixed before the run: the plain arm must reproduce script 70's
Part A within one standard error or neither arm is reported; the contrast
is stable if the seasonally adjusted 22-25 coefficient keeps at least half
its size and its significance, and seasonal otherwise.

INPUTS AND OUTPUTS
Reads the caches edu_hr_weights_2019 to 2021 and edu_hr_2019 (script 47h)
and L_counts_2021 to 2025 (script 47L); performs no SQL. Writes to
output_74/: contrast_seasonal.csv (both arms, every band, with the number
of employers) and 74_summary.txt.

IN THE PAPER
The coefficients written here are on the education route and are read
by the exhibit builders only for comparison. BANDS, POOLED_FROM and
build_terms are imported by script 82 (the profile rows of Table 1) and
script 85 (Figure 2, both arms, on the occupation route).
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
OUT = HERE / "output_74"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

PANEL_YEARS = list(range(2021, 2026))
POOLED_FROM = "2024-01"
REF_QUARTER = 4                 # omitted, matching 68
LANE16_2225 = -0.0357           # script 70 Part A: what the plain arm must reproduce
LANE16_SE = 0.0135
ATTEN_MAX = 0.50
FAILURES = []


# ──────────────────────────────────────────────────────────────────────
# THE BAND LIST IS SET HERE, DELIBERATELY, AND CHECKED AGAINST SCRIPT 70's
# AT START, so that a change to 70's band list cannot change this script's
# sample without the mismatch being caught. Two configurations exist and
# they are not interchangeable:
#
#   THREE  ["22-25", "26-30", "41-49"]                  120,359 employers
#          The three-band contrast Online Appendix III.2 quotes, which
#          script 77 reproduces as its gate.
#
#   SIX    all bands                                    172,396 employers
#          The profile of Figure 2 and Table 1, which is what this file is
#          set to.
#
BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
# ──────────────────────────────────────────────────────────────────────


def _mod(fname, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def quarter_of(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


def build_terms(b: pd.DataFrame, l70, seasonal: bool):
    """70's Part A terms, optionally plus a per-band calendar cycle."""
    ym = b["year_month"].astype(str)
    post = (ym >= POOLED_FROM).astype(int)
    post_rb = (ym >= mc.RIKSBANK_YM).astype(int)
    q = quarter_of(ym)
    terms = []
    for band in BANDS:
        if band == l70.REF_BAND:
            continue
        d = (b["age_group"] == band).astype(int)
        c1 = l70.band_col("gpt_x_high", band)
        c2 = l70.band_col("rb_x_high", band)
        b[c1] = post * b["high"] * d
        b[c2] = post_rb * b["high"] * d
        terms += [c1, c2]
        if not seasonal:
            continue
        for qq in (1, 2, 3):
            c = l70.band_col(f"q{qq}_x_high", band)
            b[c] = (q == qq).astype(int) * b["high"] * d
            terms.append(c)
    return b, terms


def run_arm(skel, expo, l70, j47, seasonal: bool, sink):
    label = "seasonal" if seasonal else "plain"
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id",
                   how="inner")
    if b.empty:
        print(f"  {label}: no firms matched exposure"); return
    b["high"] = (b["fq"] == 4).astype(int)
    b, terms = build_terms(b, l70, seasonal)
    print(f"  {label}: panel {len(b):,} rows, "
          f"{b['employer_id'].nunique():,} firms, {len(terms)} terms"
          f"{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag=f"r74_{label}", terms=terms,
                            fes=j47.FES)
    n_firms = int(b["employer_id"].nunique())
    del b
    gc.collect()
    if r.empty:
        FAILURES.append(label)
        return
    for _, row in r.iterrows():
        if not row["term"].startswith("gpt_x_high"):
            continue
        sink.append({"arm": label,
                     "band_vs_ref": row["term"].replace("gpt_x_high_", ""),
                     "coef": float(row["coef"]), "se": float(row["se"]),
                     "n_firms": n_firms})


def verdict(df) -> list:
    out = []
    if df.empty:
        return ["NO VERDICT: no fit came back."]
    p = df[(df.arm == "plain") & (df.band_vs_ref == "22_25")]
    s = df[(df.arm == "seasonal") & (df.band_vs_ref == "22_25")]
    if p.empty or s.empty:
        return ["NO VERDICT: one arm is missing, so the two are not "
                "comparable."]
    pc, ps = float(p.iloc[0]["coef"]), float(p.iloc[0]["se"])
    sc, ss = float(s.iloc[0]["coef"]), float(s.iloc[0]["se"])
    out.append(f"plain    22-25 vs 41-49 {pc:+.4f} ({ps:.4f}) t {pc/ps:+.2f}")
    out.append(f"seasonal 22-25 vs 41-49 {sc:+.4f} ({ss:.4f}) t {sc/ss:+.2f}")
    if abs(pc - LANE16_2225) > LANE16_SE:
        out.append(f"VOID: the plain arm is {pc:+.4f} against lane 16's "
                   f"{LANE16_2225:+.4f}, more than one standard error "
                   f"apart. The two runs are not on the same sample and "
                   f"neither number should be reported.")
        return out
    out.append(f"the plain arm reproduces lane 16 ({LANE16_2225:+.4f}), so "
               f"the comparison is valid")
    keeps = abs(sc) >= ATTEN_MAX * abs(pc)
    sig = abs(sc / ss) >= 1.96 if ss else False
    if keeps and sig:
        out.append("STABLE. The calendar cycle does not drive the age "
                   "difference. Quote the seasonally adjusted contrast and "
                   "say in one clause that it survives the control.")
    else:
        out.append("SEASONAL. The contrast is partly the calendar. The "
                   "paper cannot lead on the young being distinctively hit; "
                   "the framing moves to the spreading pattern and the "
                   "composition of adjustment.")
    for band in ("26_30",):
        q = df[(df.arm == "seasonal") & (df.band_vs_ref == band)]
        if len(q):
            c, e = float(q.iloc[0]["coef"]), float(q.iloc[0]["se"])
            out.append(f"seasonal {band.replace('_','-')} vs 41-49 "
                       f"{c:+.4f} ({e:.4f}) t {c/e:+.2f}")
    return out


def main():
    mc.Tee(OUT / "74_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("74 the age contrast, with and without the calendar cycle")
    print(f"the plain arm must reproduce lane 16's {LANE16_2225:+.4f} "
          f"within {LANE16_SE:.4f}")
    print("=" * 70)

    l70 = _mod("70_respecifications.py", "l70")
    # HARD GUARD. all_band_skeleton() builds the panel from
    # l70.CONTRAST_BANDS, while the interaction terms below are built from
    # BANDS. If the two ever disagree the terms and the panel describe
    # different age sets and the fit is quietly wrong. Fail here instead.
    if list(BANDS) != list(l70.CONTRAST_BANDS):
        raise SystemExit(
            f"BAND MISMATCH: 74 has {BANDS}, but 70 builds the skeleton from "
            f"{l70.CONTRAST_BANDS}. The panel and the terms would not agree. "
            f"Set them the same in both files before running.")
    j47 = _mod("47j_within_employer_triple.py", "j47")

    cnt = []
    for y in PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise SystemExit(f"L_counts_{y}.parquet missing: run 47L first. "
                             f"This script performs no SQL.")
        cnt.append(c)
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(counts["year_month"].max())
    if last < POOLED_FROM:
        raise SystemExit(f"counts end at {last}, before {POOLED_FROM}.")
    print(f"  counts to {last}; bands {BANDS} (set in 74, checked against 70), "
          f"reference {l70.REF_BAND}")

    expo = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
    print(f"  exposure: {len(expo):,} firms")

    # ONE skeleton for both arms: the arms must differ in the term list
    # and in nothing else, or the comparison measures the sample too.
    skel = l70.all_band_skeleton(counts)
    del counts
    gc.collect()
    if skel.empty:
        raise SystemExit("skeleton empty")
    print(f"  skeleton {len(skel):,} rows")

    sink = []
    for seasonal in (False, True):
        try:
            run_arm(skel, expo, l70, j47, seasonal, sink)
        except BaseException as ex:
            print(f"  arm failed: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            FAILURES.append("seasonal" if seasonal else "plain")
    del skel
    gc.collect()

    df = pd.DataFrame(sink)
    if len(df):
        df.to_csv(OUT / "contrast_seasonal.csv", index=False)

    lines = ["74 the age contrast, with and without the calendar cycle",
             "=" * 70, "",
             "Coefficients are differences from the 41-49 band. A negative "
             "number means the band declined MORE than 41-49 did.", ""]
    lines += verdict(df)
    lines += ["", "The seasonal terms are per band for the same reason the "
              "treatment is: with firm-level exposure, high x quarter is "
              "constant within employer-month and the employer-by-month "
              "effects absorb it. Q4 is omitted, matching 68."]
    if FAILURES:
        lines += ["", "FAILED:"] + [f"  {f}" for f in FAILURES]
    (OUT / "74_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("74_contrast_seasonal", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

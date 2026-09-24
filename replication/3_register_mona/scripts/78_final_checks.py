#!/usr/bin/env python3
"""
78_final_checks.py: seven robustness checks on the within-employer design,
each answerable from the caches of earlier scripts.

======================================================================
  RUNS IN MONA. No SQL except, in Parts C and F, one read of LISA's firm
  table for the 2019 industry code (the read script 73 makes) and, in
  Part E, one counts pull per panel year, cached. Reads the caches
  47h, 47L and 67 wrote. Writes output_78/. Parts are chosen with the
  environment variable CANARIES_78_PARTS (default ABCDEFG), which
  master.py sets. Budget: A about one hour, B one to two, C one to two,
  D two, E two to three (one pull, one wide fit), F two (one fit with
  four effects), G under one.
======================================================================

HOW THE PAPER USES THIS SCRIPT. The parts below are written for the
education-route score, and the coefficients this script writes itself are
not quoted. The occupation-route scripts import its term sets and parts
and run them on script 82's score: eq2_terms, gender_eq2_terms,
with_exposure, rows_of and lincomb (script 82: Table 1); part_a
(script 86: the pre-period path, Figure A7 and Table A19); part_e
(script 85: the profile with 50 and over split at 65, Table A13);
drift_terms and build_skeleton_bands (script 83: the drift rows of
Table 1); and the constants POST_FROM, PRE_LAUNCH_END, PRE_TREND_FROM,
REF_QUARTER and EXTENDED_FROM.

THE DESIGN EVERY PART USES. Employer x age band x month counts, the
young band beside the four bands aged 31 to 69, exposure the top quartile
of the employer's 2019 education mix (47h, incumbents aged 31 to 69),
employer-by-month, employer-by-age and month-by-age effects (47j), Poisson
pseudo-maximum likelihood, standard errors clustered by employer, the
calendar cycle removed by three quarter-of-year interactions with the
fourth quarter omitted (68). Nothing here changes that design; each part
asks one question of it.

A. THE PRE-PERIOD. The paper's Figure 3 starts at the launch, because the
   calendar cycle is identified off the pre-launch months and a path that
   spans the whole panel already spans the cycle. A reader still wants to
   see the pre-launch quarters. Two exhibits: (i) the quarterly path on
   the PLAIN specification, one dummy per quarter from 2021Q1 to the end
   of the panel with 2022Q1 as the reference and no calendar terms, so
   the pre-launch quarters and the cycle itself are drawn as they are;
   (ii) the drift test: on the pre-launch months alone (January 2021 to
   November 2022) the three calendar terms, the tightening window
   (April to November 2022) and a linear trend in months, all interacted
   with High x Young. The trend is the testable direction; a joint test
   of every pre-launch quarter net of the cycle is not identified from
   two years, and the summary says so.
   Read rule: FLAT if the trend is within two standard errors of zero.
   The path is reported whatever it shows.

B. THE SEX SPLIT ON EQUATION (2) EXACTLY. The paper's female
   differential (68) was estimated without the interim term, so its base
   differs from Table 1's. Here the sex panel (67's employer x age x sex x
   month cell) carries every term of Equation (2), the tightening switch,
   the interim window, the adoption step and the three calendar terms,
   each interacted with High x Young, with High x Female and with High x
   Young x Female, under employer-by-age-and-sex and month-by-age-and-sex
   effects. The summary prints the male adoption step, the female
   differential, the female step (their sum, with its standard error from
   the covariance) and each sex's step from the 2023 level, all from the
   same base as Table 1.

C. INFERENCE BY INDUSTRY. Exposure is a firm-level score shared by firms
   with the same education mix, and the industry test halves the 26-30
   step, so standard errors clustered by three-digit industry are
   reported beside the employer ones. The headline stock fit is re-run
   with the cluster set to the 2019 SNI 2007 three-digit code from LISA's
   firm table (script 73's source); a firm without a code is its own
   cluster and the count is printed. The coefficients must equal 68's to
   four decimals, since only the covariance changes; the check is printed.

D. THE SKILL-INTENSITY PLACEBO. A 2019 education mix that scores high on
   generative-AI exposure is also a graduate-heavy mix, so any 2024 shock
   to graduate hiring would be attributed to AI. High_edu is the top
   quartile, weighted by incumbent employment as the DAIOE cut is, of the
   share of the employer's 2019 incumbents aged 31 to 69 with
   post-secondary education (SUN 2020 level 4, 5 or 6, 47h's rule).
   Equation (2) is fitted with High_edu in place of High, then with both
   sets of interactions together.
   Read rule, fixed before the run: SKILL CUT REPRODUCES THE STEP if the
   skill cut alone returns an adoption step at least as large as the
   DAIOE step, allowing one standard error of the DAIOE step (a true skill
   effect also leaks into the DAIOE contrast's control group, so the DAIOE
   step alone understates it, and "within one SE" would miss the case the
   test exists for); AI SURVIVES THE SKILL CUT if, with both in, the DAIOE
   step keeps at least half of its size alone. Both numbers are reported
   either way.

E. THE PROFILE WITH THE OLDEST BAND SPLIT. Script 74's six-band profile
   against 41-49 (calendar cycle removed) with 50 and over split into
   50-64 and 65-69, so the pension-age rival (the general retirement age
   rose from 65 to 66 in 2023) can be read off: if the 50-and-over gain
   sits in 65-69 it is a retirement story, if in 50-64 it is not. The
   counts caches hold 50 and over as one band, so this part makes one
   pull of its own (47L's query with the band split), cached as
   L_counts_split_YYYY. No gate; every band is reported.

F. INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS. Script 73's
   industry test ran on the pre-cycle specification. Here the headline
   22-25 stock fit carries both the three calendar terms and
   industry x age x month effects (the 2019 three-digit code from LISA's
   firm table, as 73 reads it), beside its own baseline on the same
   industry-linked sample. Read rule: the retained share of the adoption
   step against that baseline is reported; no gate.

G. THE RATE-PERIOD BOUNDARY. The Riksbank announced its first increase on
   28 April 2022 and it took effect on 4 May. Equation (2) is re-run for
   the 22-25 stock with the tightening indicator from May 2022 and April
   2022 dropped from the panel. Read rule: the step and the tightening
   rise are reported beside the April-boundary ones; no gate.

THE PRE-PERIOD EXTENSION. 47L pulled counts for 2019 to 2025. When
L_counts_2019 and L_counts_2020 are on the share, Part A(i)'s path runs
from 2019Q1 with the same reference quarter; when they are not, it runs
from 2021Q1 and the summary says that extending it needs two more year
pulls of 47L's query. The drift test (A(ii)) always uses January 2021 to
November 2022, the pre-launch window the paper describes.

Output (output_78/):
  prepath_plain.csv      A(i): band, quarter, coef, se
  predrift.csv           A(ii): the trend, the cycle and the window
  gender_eq2.csv         B: every term of Equation (2) by sex
  cluster_industry.csv   C: coefficients with both standard errors
  skill_placebo.csv      D: the three specifications, both bands
  prof_split.csv         E: seven bands against 41-49
  industry_seasonal.csv  F: baseline and industry effects, same sample
  boundary_may.csv       G: every treatment term, May boundary
  vcov_s78_*.csv         the clustered covariance of every fit
  78_summary.txt
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_78_OUT", "output_78")
PARTS = os.environ.get("CANARIES_78_PARTS", "ABCDEFG").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

POST_FROM = "2024-01"                      # adoption, as in 68
RB_FROM, RB_TO = mc.RIKSBANK_YM, mc.CHATGPT_YM   # the tightening window
PRE_LAUNCH_END = mc.CHATGPT_YM             # the drift test stops here
REF_QUARTER = "2022Q1"                     # A(i): the omitted quarter
YOUNG_BANDS = ["22-25", "26-30"]
YOUNG_SEX = "22-25"                        # B runs on the paper's band
S68_EXPORT = HERE / "output_68" / "seasonal_pooled.csv"   # C's comparison
DRIFT_RULE_SE = 2.0                        # A(ii): FLAT within this many SE
SKILL_KEEP = 0.50                          # D: AI survives if it keeps half
EXTENDED_FROM = "2019-01"                  # A(i) runs from here if cached
PRE_TREND_FROM = "2021-01"                 # A(ii)'s window starts here
SPLIT_BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-64", "65-69"]
PROFILE_REF = "41-49"                      # E: the reference band, as in 74
MAY_BOUNDARY = "2022-05"                   # G: the increase took effect 4 May
NOTES = []
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


def quarter_of_year(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


def quarter_label(ym: pd.Series) -> pd.Series:
    return ym.str.slice(0, 4) + "Q" + quarter_of_year(ym).astype(str)


def month_index(ym: pd.Series) -> pd.Series:
    """Months since January 2021, so a trend coefficient is per month."""
    y = ym.str.slice(0, 4).astype(int)
    m = ym.str.slice(5, 7).astype(int)
    return (y - 2021) * 12 + (m - 1)


# ----------------------------------------------------------------------
# Shared inputs: exposure, counts, skeletons
# ----------------------------------------------------------------------

def load_modules():
    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    j47 = s61._j47()
    h47 = j47._h47()
    return s61, s67, j47, h47


def score_book(h47):
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
    return book, spec


def frame_2019(h47) -> pd.DataFrame:
    f = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                      require=h47.YEAR_COLS + ["n_emp"])
    if f is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    return f


def exposure(frame19, book, spec, j47, s61) -> pd.DataFrame:
    e, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, "true",
                                  s61.TRUNC)
    return e


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def build_skeleton_bands(counts: pd.DataFrame, bands: list, young: str,
                         j47, from_ym: str) -> pd.DataFrame:
    """
    61's balanced employer x band x month skeleton for ANY band list and
    start month: zero-filled, cells that are zero in every month dropped,
    employers with fewer than two bands dropped, the three fixed-effect
    keys as integers. `young` marks the band whose indicator is `young`;
    an employer must hold that band and at least one other.
    """
    p = counts[counts["age_group"].astype(str).isin(bands)]
    p = p[p["year_month"].astype(str) >= from_ym]
    p = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
         ["n_emp"].sum().reset_index())
    p["age_group"] = p["age_group"].astype(str)
    p["year_month"] = p["year_month"].astype(str)
    have = p.groupby("employer_id")["age_group"].agg(set)
    others = set(bands) - {young}
    keep = have[have.apply(lambda v: young in v and bool(v & others))].index
    p = p[p["employer_id"].isin(keep)]
    if p.empty:
        return p
    months = sorted(p["year_month"].unique())
    emp = p["employer_id"].drop_duplicates().to_numpy()
    full = pd.MultiIndex.from_product([emp, bands, months],
                                      names=["employer_id", "age_group",
                                             "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index())
    bal["n_emp"] = bal["n_emp"].astype(int)
    bal = j47._drop_dead_cells(bal)
    if bal.empty:
        return bal
    bal["young"] = (bal["age_group"] == young).astype(int)
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"], sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def with_exposure(skel: pd.DataFrame, expo: pd.DataFrame) -> pd.DataFrame:
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    if not b.empty:
        b["high"] = (b["fq"] == 4).astype(int)
    return b


def eq2_terms(b: pd.DataFrame, high_col: str = "high",
              suffix: str = "", rb_from: str = mc.RIKSBANK_YM) -> tuple:
    """
    Equation (2)'s term set for one exposure indicator, as script 68
    builds it: the cumulative tightening switch from April 2022, the
    interim window from the launch to December 2023, the adoption step
    from January 2024, and the three calendar terms with the fourth
    quarter omitted. `suffix` lets two indicators sit in one model (D).
    """
    ym = b["year_month"].astype(str)
    hy = b[high_col] * b["young"]
    q = quarter_of_year(ym)
    post_any = ym >= mc.CHATGPT_YM
    s = suffix
    b[f"rb_x_high{s}_x_young"] = (ym >= rb_from).astype(int) * hy
    terms = [f"rb_x_high{s}_x_young"]
    for qq in (1, 2, 3):
        b[f"q{qq}_x_high{s}_x_young"] = (q == qq).astype(int) * hy
        terms.append(f"q{qq}_x_high{s}_x_young")
    b[f"interim_x_high{s}_x_young"] = (post_any & (ym < POST_FROM)).astype(int) * hy
    b[f"post_x_high{s}_x_young"] = (ym >= POST_FROM).astype(int) * hy
    terms += [f"interim_x_high{s}_x_young", f"post_x_high{s}_x_young"]
    return b, terms


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """One Poisson fit; returns (coefficient table indexed by term, vcov
    DataFrame or None). A failure is recorded and returns (None, None)."""
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s78_{tag}", terms=terms,
                                fes=fes, cluster=cluster)
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        print(f"    {tag}: FAILED, recorded and skipped")
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return g, v


def lincomb(v: pd.DataFrame, weights: dict) -> float | None:
    """Standard error of sum(w_i * term_i) from the clustered covariance."""
    if v is None:
        return None
    ks = [k for k in weights if k in v.index and k in v.columns]
    if len(ks) != len(weights):
        return None
    var = 0.0
    for a in ks:
        for b_ in ks:
            var += weights[a] * weights[b_] * float(v.loc[a, b_])
    return float(np.sqrt(max(var, 0.0)))


def rows_of(g: pd.DataFrame, terms: list, **extra) -> list:
    out = []
    for t_ in terms:
        if t_ not in g.index:
            continue
        out.append({**extra, "term": t_, "coef": float(g.loc[t_, "coef"]),
                    "se": float(g.loc[t_, "se"]),
                    "n_obs": int(g.loc[t_, "n_obs"]),
                    "status": str(g.loc[t_].get("status", "ok"))})
    return out


# ----------------------------------------------------------------------
# Part A: the pre-period
# ----------------------------------------------------------------------

def plain_path_terms(b: pd.DataFrame) -> tuple:
    """One dummy per quarter of the panel, REF_QUARTER omitted, no
    calendar terms and no tightening switch: the path as it is."""
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    lab = quarter_label(ym)
    terms = []
    for qq in sorted(lab.unique()):
        if qq == REF_QUARTER:
            continue
        col = f"pq_{qq}_x_high_x_young"
        b[col] = (lab == qq).astype(int) * hy
        terms.append(col)
    return b, terms


def drift_terms(b: pd.DataFrame) -> tuple:
    """Pre-launch months only: the cycle, the tightening window and a
    linear trend in months, each x High x Young."""
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    q = quarter_of_year(ym)
    terms = []
    for qq in (1, 2, 3):
        b[f"q{qq}_x_high_x_young"] = (q == qq).astype(int) * hy
        terms.append(f"q{qq}_x_high_x_young")
    b["rbw_x_high_x_young"] = ((ym >= RB_FROM) & (ym < RB_TO)).astype(int) * hy
    b["trend_x_high_x_young"] = month_index(ym).astype(float) * hy
    terms += ["rbw_x_high_x_young", "trend_x_high_x_young"]
    return b, terms


def part_a(counts, expo, s61, j47, extended: pd.DataFrame | None = None) -> tuple:
    """`extended` is the counts frame from 2019 when the 2019 and 2020
    caches are on the share (A(i) then runs from 2019Q1); otherwise the
    2021 frame is used and the summary says so."""
    path_rows, drift_rows = [], []
    src, from_ym = (extended, EXTENDED_FROM) if extended is not None \
        else (counts, s61.PANEL_FROM)
    NOTES.append(f"A(i): path from {from_ym} "
                 + ("(2019 and 2020 counts cached)" if extended is not None
                    else "(L_counts_2019 and L_counts_2020 are not on the "
                         "share; extending the path needs two more year "
                         "pulls of 47L's query)"))
    for band in YOUNG_BANDS:
        skel = build_skeleton_bands(src, [band] + j47.INCUMBENT_BANDS, band,
                                    j47, from_ym)
        if skel.empty:
            FAILURES.append(f"A/{band}/empty")
            continue
        b = with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"A/{band}/no exposure")
            continue
        # (i) the plain quarterly path
        b, pterms = plain_path_terms(b)
        g, _ = fit(b, f"prepath_{band.replace('-', '_')}", pterms, j47.FES)
        if g is not None:
            for t_ in pterms:
                if t_ in g.index:
                    path_rows.append(
                        {"young_band": band,
                         "quarter": t_.removeprefix("pq_").split("_x_high")[0],
                         "coef": float(g.loc[t_, "coef"]),
                         "se": float(g.loc[t_, "se"]),
                         "n_obs": int(g.loc[t_, "n_obs"]),
                         "status": str(g.loc[t_].get("status", "ok"))})
            path_rows.append({"young_band": band, "quarter": REF_QUARTER,
                              "coef": 0.0, "se": 0.0, "n_obs": 0,
                              "status": "reference"})
            pd.DataFrame(path_rows).to_csv(OUT / "prepath_plain.csv", index=False)
        b = b.drop(columns=pterms)
        # (ii) the drift test on the pre-launch months
        ym = b["year_month"].astype(str)
        pre = b[(ym >= PRE_TREND_FROM) & (ym < PRE_LAUNCH_END)].copy()
        del b
        gc.collect()
        pre = j47._drop_dead_cells(pre)
        pre, dterms = drift_terms(pre)
        g, _ = fit(pre, f"predrift_{band.replace('-', '_')}", dterms, j47.FES)
        del pre
        gc.collect()
        if g is not None:
            drift_rows += rows_of(g, dterms, young_band=band)
            pd.DataFrame(drift_rows).to_csv(OUT / "predrift.csv", index=False)
    return path_rows, drift_rows


# ----------------------------------------------------------------------
# Part B: the sex split on Equation (2)
# ----------------------------------------------------------------------

def gender_eq2_terms(b: pd.DataFrame) -> tuple:
    """Every term of Equation (2), each x High x Young, x High x Female
    and x High x Young x Female."""
    ym = b["year_month"].astype(str)
    q = quarter_of_year(ym)
    post_any = ym >= mc.CHATGPT_YM
    periods = {"rb": (ym >= mc.RIKSBANK_YM).astype(int),
               "interim": (post_any & (ym < POST_FROM)).astype(int),
               "post": (ym >= POST_FROM).astype(int)}
    for qq in (1, 2, 3):
        periods[f"q{qq}"] = (q == qq).astype(int)
    hy = b["high"] * b["young"]
    hf = b["high"] * b["female"]
    hyf = hy * b["female"]
    terms = []
    for p, ind in periods.items():
        b[f"{p}_x_high_x_young"] = ind * hy
        b[f"{p}_x_high_x_female"] = ind * hf
        b[f"{p}_x_high_x_young_x_female"] = ind * hyf
        terms += [f"{p}_x_high_x_young", f"{p}_x_high_x_female",
                  f"{p}_x_high_x_young_x_female"]
    return b, terms


def part_b(expo, s61, s67, j47) -> tuple:
    sex = load_counts("L_counts_sex", s61.PANEL_YEARS,
                      require=["employer_id", "year_month", "age_group",
                               "gender", "n_emp"])
    if sex is None:
        FAILURES.append("B/no L_counts_sex cache")
        print("  B: L_counts_sex_* missing, skipped")
        return [], {}
    skel = s67.build_skeleton_sex(sex, YOUNG_SEX, j47, "n_emp")
    del sex
    gc.collect()
    if skel.empty:
        FAILURES.append("B/empty")
        return [], {}
    b = with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("B/no exposure")
        return [], {}
    b, terms = gender_eq2_terms(b)
    g, v = fit(b, f"gender_eq2_{YOUNG_SEX.replace('-', '_')}", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return [], {}
    rows = rows_of(g, terms, young_band=YOUNG_SEX)
    pd.DataFrame(rows).to_csv(OUT / "gender_eq2.csv", index=False)
    c = {t_: float(g.loc[t_, "coef"]) for t_ in terms if t_ in g.index}
    e = {t_: float(g.loc[t_, "se"]) for t_ in terms if t_ in g.index}
    m, d = "post_x_high_x_young", "post_x_high_x_young_x_female"
    mi, di = "interim_x_high_x_young", "interim_x_high_x_young_x_female"
    summ = {}
    if m in c and d in c:
        summ["male_step"] = (c[m], e[m])
        summ["female_differential"] = (c[d], e[d])
        summ["female_step"] = (c[m] + c[d], lincomb(v, {m: 1, d: 1}))
    if all(k in c for k in (m, d, mi, di)):
        summ["male_step_from_2023"] = (c[m] - c[mi], lincomb(v, {m: 1, mi: -1}))
        summ["female_step_from_2023"] = (
            c[m] + c[d] - c[mi] - c[di],
            lincomb(v, {m: 1, d: 1, mi: -1, di: -1}))
    return rows, summ


# ----------------------------------------------------------------------
# Part C: standard errors clustered by industry
# ----------------------------------------------------------------------

def industry_map() -> dict:
    """employer_id (normalised string) -> 2019 three-digit SNI, from
    script 73's loader, so both scripts read the same source."""
    s73 = _mod("73_industry_and_credit.py", "s73")
    conn = mc.connect()
    schema = s73.discover(conn)
    if schema.empty:
        raise RuntimeError("no LISA firm table visible; Part C cannot run")
    ind = s73.firm_industry(conn, schema)
    NOTES.extend(n for n in s73.NOTES
                 if n.startswith("industry") and n not in NOTES)
    try:
        conn.close()
    except Exception:
        pass
    return dict(zip(s73.norm_id(ind["employer_id"]), ind["ind3"].astype(str)))


def s68_reference() -> pd.DataFrame | None:
    if S68_EXPORT.exists():
        try:
            return pd.read_csv(S68_EXPORT)
        except Exception:
            return None
    return None


def part_c(counts, expo, s61, j47) -> list:
    imap = industry_map()
    s73 = _mod("73_industry_and_credit.py", "s73b")
    ref = s68_reference()
    rows = []
    for band in YOUNG_BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"C/{band}/empty")
            continue
        b = with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{band}/no exposure")
            continue
        key = s73.norm_id(b["employer_id"])
        cl = key.map(imap)
        n_own = int(cl.isna().groupby(b["employer_id"]).any().sum())
        b["cl_ind"] = cl.fillna("own_" + key)
        msg = (f"C/{band}: {b['employer_id'].nunique() - n_own:,} of "
               f"{b['employer_id'].nunique():,} employers carry a 2019 industry "
               f"code; {n_own:,} without one are their own cluster; "
               f"{b['cl_ind'].nunique():,} clusters")
        print(f"  {msg}")
        NOTES.append(msg)
        b, terms = eq2_terms(b)
        g_ind, _ = fit(b, f"clind_{band.replace('-', '_')}", terms, j47.FES,
                       cluster="cl_ind")
        if g_ind is None:
            del b
            gc.collect()
            continue
        # The employer-clustered comparison: 68's export if it is on the
        # share, else the same fit once more under the design's clustering.
        emp = None
        if ref is not None:
            sub = ref[(ref["young_band"] == band) & (ref["outcome"] == "stock")
                      & (ref["arm"] == "true")]
            if len(sub):
                emp = sub.set_index("term")
        if emp is None:
            emp, _ = fit(b, f"clemp_{band.replace('-', '_')}", terms, j47.FES)
        del b
        gc.collect()
        for t_ in terms:
            if t_ not in g_ind.index:
                continue
            ce = float(emp.loc[t_, "coef"]) if (emp is not None and t_ in emp.index) else np.nan
            se_e = float(emp.loc[t_, "se"]) if (emp is not None and t_ in emp.index) else np.nan
            ci = float(g_ind.loc[t_, "coef"])
            rows.append({"young_band": band, "term": t_, "coef": ci,
                         "se_employer": se_e, "se_industry": float(g_ind.loc[t_, "se"]),
                         "coef_employer_run": ce,
                         "coef_match_4dp": (bool(round(ci, 4) == round(ce, 4))
                                            if not np.isnan(ce) else None),
                         "n_clusters_industry": int(cl.nunique() + n_own)})
        pd.DataFrame(rows).to_csv(OUT / "cluster_industry.csv", index=False)
    return rows


# ----------------------------------------------------------------------
# Part D: the skill-intensity placebo
# ----------------------------------------------------------------------

def skill_cut(frame19: pd.DataFrame, j47, h47) -> tuple:
    """
    Share of each employer's 2019 incumbents (aged 31 to 69) with
    post-secondary education, and High_edu = the top quartile of that share
    weighted by incumbent employment, with the same five person-month floor
    as the DAIOE cut. Returns (frame with employer_id, edu_share, high_edu)
    and the cut point.
    """
    f = frame19[frame19["age_group"].astype(str).isin(j47.INCUMBENT_BANDS)].copy()
    f["ter"] = h47.is_tertiary(f["niva_t"]).astype(int) * f["n_emp"]
    g = (f.groupby("employer_id", observed=True)
         .agg(ter=("ter", "sum"), n=("n_emp", "sum")).reset_index())
    g = g[g["n"] >= j47.MIN_FIRM_INCUMBENTS]
    g["edu_share"] = g["ter"] / g["n"]
    cuts = h47.wquantile_cutoffs(g["edu_share"].to_numpy(dtype=float),
                                 g["n"].to_numpy(dtype=float))
    cut = cuts[2]
    g["high_edu"] = (g["edu_share"] >= cut).astype(int)
    return g[["employer_id", "edu_share", "high_edu"]], cut


def part_d(counts, expo, frame19, s61, j47, h47) -> tuple:
    edu, cut = skill_cut(frame19, j47, h47)
    share_hi = float(edu["high_edu"].mean()) if len(edu) else float("nan")
    overlap = edu.merge(expo[["employer_id", "fq"]], on="employer_id")
    both = float(((overlap["fq"] == 4) & (overlap["high_edu"] == 1)).sum())
    q4 = float((overlap["fq"] == 4).sum())
    msg = (f"D: post-secondary share cut at {cut:.3f}; {share_hi:.1%} of scored "
           f"employers are High_edu; {both/ max(q4,1):.1%} of DAIOE Q4 employers "
           f"are also High_edu")
    print(f"  {msg}")
    NOTES.append(msg)
    rows, verdicts = [], {}
    for band in YOUNG_BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"D/{band}/empty")
            continue
        b = with_exposure(skel, expo)
        del skel
        gc.collect()
        b = b.merge(edu[["employer_id", "high_edu"]], on="employer_id", how="inner")
        if b.empty:
            FAILURES.append(f"D/{band}/no overlap")
            continue
        # the DAIOE step alone, on the SAME sample, so the three are comparable
        b, t_ai = eq2_terms(b, "high", "")
        g_ai, _ = fit(b, f"skill_ai_{band.replace('-', '_')}", t_ai, j47.FES)
        b = b.drop(columns=t_ai)
        b, t_ed = eq2_terms(b, "high_edu", "edu")
        g_ed, _ = fit(b, f"skill_edu_{band.replace('-', '_')}", t_ed, j47.FES)
        b = b.drop(columns=t_ed)
        b, t_ai2 = eq2_terms(b, "high", "")
        b, t_ed2 = eq2_terms(b, "high_edu", "edu")
        g_both, _ = fit(b, f"skill_both_{band.replace('-', '_')}", t_ai2 + t_ed2,
                        j47.FES)
        del b
        gc.collect()
        for spec, g, terms in (("daioe_alone", g_ai, t_ai),
                               ("skill_alone", g_ed, t_ed),
                               ("both", g_both, t_ai2 + t_ed2)):
            if g is not None:
                rows += rows_of(g, terms, young_band=band, spec=spec)
        pd.DataFrame(rows).to_csv(OUT / "skill_placebo.csv", index=False)
        v = {}
        if g_ai is not None and "post_x_high_x_young" in g_ai.index:
            v["ai_alone"] = (float(g_ai.loc["post_x_high_x_young", "coef"]),
                             float(g_ai.loc["post_x_high_x_young", "se"]))
        if g_ed is not None and "post_x_highedu_x_young" in g_ed.index:
            v["skill_alone"] = (float(g_ed.loc["post_x_highedu_x_young", "coef"]),
                                float(g_ed.loc["post_x_highedu_x_young", "se"]))
        if g_both is not None:
            for k, t_ in (("ai_both", "post_x_high_x_young"),
                          ("skill_both", "post_x_highedu_x_young")):
                if t_ in g_both.index:
                    v[k] = (float(g_both.loc[t_, "coef"]),
                            float(g_both.loc[t_, "se"]))
        if "ai_alone" in v and "skill_alone" in v:
            a, s = v["ai_alone"], v["skill_alone"]
            v["reproduces"] = (np.sign(s[0]) == np.sign(a[0])
                               and abs(s[0]) >= abs(a[0]) - a[1])
        if "ai_alone" in v and "ai_both" in v:
            a, ab = v["ai_alone"], v["ai_both"]
            v["survives"] = (np.sign(ab[0]) == np.sign(a[0])
                             and abs(ab[0]) >= SKILL_KEEP * abs(a[0]))
        verdicts[band] = v
    return rows, verdicts


# ----------------------------------------------------------------------
# Part E: the profile with 50 and over split at the pension age
# ----------------------------------------------------------------------

AGE_CASE_SPLIT = """CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 64 THEN '50-64'
             WHEN age BETWEEN 65 AND 69 THEN '65-69'
             ELSE NULL END"""


def q_counts_split(year: int, conn) -> pd.DataFrame:
    """47L's counts query with the oldest band split at 65. Birth year
    from whichever Individ vintage holds the person, as in 47L."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    q = f"""
    WITH base AS ({monthly}),
    aged AS (
        SELECT employer_id, period, person_id, {year} - fodelse AS age
        FROM base WHERE fodelse IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {AGE_CASE_SPLIT} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, {AGE_CASE_SPLIT}
    """
    return pd.read_sql(q, conn)


def split_counts(years) -> pd.DataFrame | None:
    out, conn = [], None
    for y in years:
        cf = CACHE / f"L_counts_split_{y}.parquet"
        c = mc.read_cache(cf, require=["employer_id", "year_month",
                                       "age_group", "n_emp"])
        if c is None:
            if conn is None:
                conn = mc.connect()
            t = time.time()
            c = q_counts_split(y, conn)
            mc.write_cache(c, cf)
            print(f"  E: counts with the oldest band split, {y}: {len(c):,} "
                  f"cells ({(time.time()-t)/60:.1f} min)")
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else None


def profile_terms(b: pd.DataFrame) -> tuple:
    """74's seasonal arm: per band except the reference, the adoption
    step, the tightening switch and the three calendar terms, each x High
    x that band."""
    ym = b["year_month"].astype(str)
    post = (ym >= POST_FROM).astype(int)
    post_rb = (ym >= mc.RIKSBANK_YM).astype(int)
    q = quarter_of_year(ym)
    terms = []
    for band in SPLIT_BANDS:
        if band == PROFILE_REF:
            continue
        d = (b["age_group"] == band).astype(int)
        tag = band.replace("-", "_").replace("+", "plus")
        b[f"gpt_x_high_{tag}"] = post * b["high"] * d
        b[f"rb_x_high_{tag}"] = post_rb * b["high"] * d
        terms += [f"gpt_x_high_{tag}", f"rb_x_high_{tag}"]
        for qq in (1, 2, 3):
            b[f"q{qq}_x_high_{tag}"] = (q == qq).astype(int) * b["high"] * d
            terms.append(f"q{qq}_x_high_{tag}")
    return b, terms


def part_e(expo, s61, j47) -> list:
    counts = split_counts(s61.PANEL_YEARS)
    if counts is None or counts.empty:
        FAILURES.append("E/no counts")
        return []
    skel = build_skeleton_bands(counts, SPLIT_BANDS, "22-25", j47,
                                s61.PANEL_FROM)
    del counts
    gc.collect()
    if skel.empty:
        FAILURES.append("E/empty")
        return []
    b = with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("E/no exposure")
        return []
    b, terms = profile_terms(b)
    n_firms = int(b["employer_id"].nunique())
    g, _ = fit(b, "prof_split", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return []
    rows = []
    for band in SPLIT_BANDS:
        tag = band.replace("-", "_").replace("+", "plus")
        t_ = f"gpt_x_high_{tag}"
        if band == PROFILE_REF:
            rows.append({"band": band, "coef": 0.0, "se": 0.0,
                         "n_firms": n_firms, "status": "reference"})
        elif t_ in g.index:
            rows.append({"band": band, "coef": float(g.loc[t_, "coef"]),
                         "se": float(g.loc[t_, "se"]), "n_firms": n_firms,
                         "status": str(g.loc[t_].get("status", "ok"))})
    pd.DataFrame(rows).to_csv(OUT / "prof_split.csv", index=False)
    return rows


# ----------------------------------------------------------------------
# Part F: industry x age x month effects with the calendar terms
# ----------------------------------------------------------------------

def part_f(counts, expo, s61, j47) -> list:
    imap = industry_map()
    s73 = _mod("73_industry_and_credit.py", "s73f")
    band = "22-25"
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        FAILURES.append("F/empty")
        return []
    b = with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("F/no exposure")
        return []
    ind = pd.DataFrame({"_key": list(imap), "ind3": list(imap.values())})
    b["_key"] = s73.norm_id(b["employer_id"])
    n_all = int(b["employer_id"].nunique())
    b = b.merge(ind, on="_key", how="inner").drop(columns="_key")
    n_firms = int(b["employer_id"].nunique())
    msg = (f"F: {n_firms:,} of {n_all:,} employers carry a 2019 industry code "
           f"and form the sample for both fits")
    print(f"  {msg}")
    NOTES.append(msg)
    if b.empty:
        FAILURES.append("F/no industry")
        return []
    ic = pd.factorize(b["ind3"], sort=False)[0].astype("int64")
    ac = pd.factorize(b["age_group"], sort=False)[0].astype("int64")
    tc = pd.factorize(b["year_month"], sort=False)[0].astype("int64")
    n_a, n_t = int(ac.max()) + 1, int(tc.max()) + 1
    b["fe_ind_age_t"] = (ic * n_a + ac) * n_t + tc
    b, terms = eq2_terms(b)
    rows = []
    g_base, _ = fit(b, "indseas_base_22_25", terms, j47.FES)
    if g_base is not None:
        rows += rows_of(g_base, terms, spec="baseline_same_sample",
                        n_firms=n_firms)
    # fe_t_age is nested inside industry x age x month and is dropped, as
    # script 73 explains: the model is the same and one effect fewer fits.
    fes = tuple(f for f in j47.FES if f != "fe_t_age") + ("fe_ind_age_t",)
    g_ind, _ = fit(b, "indseas_ind_22_25", terms, fes)
    del b
    gc.collect()
    if g_ind is not None:
        rows += rows_of(g_ind, terms, spec="industry_age_month", n_firms=n_firms)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "industry_seasonal.csv", index=False)
    return rows


# ----------------------------------------------------------------------
# Part G: the tightening boundary at May 2022
# ----------------------------------------------------------------------

def part_g(counts, expo, s61, j47) -> list:
    band = "22-25"
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        FAILURES.append("G/empty")
        return []
    b = with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("G/no exposure")
        return []
    n_before = len(b)
    b = b[b["year_month"].astype(str) != mc.RIKSBANK_YM].copy()
    NOTES.append(f"G: {n_before - len(b):,} cells of {mc.RIKSBANK_YM} dropped; "
                 f"the tightening indicator starts in {MAY_BOUNDARY}")
    b, terms = eq2_terms(b, rb_from=MAY_BOUNDARY)
    g, _ = fit(b, "boundary_may_22_25", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return []
    rows = rows_of(g, terms, young_band=band, boundary=MAY_BOUNDARY)
    pd.DataFrame(rows).to_csv(OUT / "boundary_may.csv", index=False)
    return rows


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def fmt(x):
    if x is None:
        return "(no SE)"
    return f"({x:.4f})"


def main():
    mc.Tee(OUT / "78_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"78: SEVEN CHECKS ON THE HEADLINE DESIGN   parts {PARTS}")
    print("=" * 70)
    print(mc.mem_line("  "))

    s61, s67, j47, h47 = load_modules()
    book, spec = score_book(h47)
    frame19 = frame_2019(h47)
    expo = exposure(frame19, book, spec, j47, s61)
    print(f"  exposure: {len(expo):,} firms")
    if "D" not in PARTS:
        del frame19
        frame19 = None
    gc.collect()

    counts = None
    if any(p in PARTS for p in "ACDFG"):
        counts = load_counts("L_counts", s61.PANEL_YEARS)
        if counts is None:
            raise RuntimeError("L_counts_* missing: run 47L first.")

    path_rows, drift_rows, gender_rows, gsum, cl_rows = [], [], [], {}, []
    skill_rows, skill_v, prof_rows, ind_rows, bnd_rows = [], {}, [], [], []
    if "A" in PARTS:
        early = load_counts("L_counts", [2019, 2020])
        extended = (pd.concat([early, counts], ignore_index=True)
                    if early is not None else None)
        del early
        r = opt("Part A", part_a, counts, expo, s61, j47, extended)
        del extended
        gc.collect()
        if r:
            path_rows, drift_rows = r
    if "B" in PARTS:
        r = opt("Part B", part_b, expo, s61, s67, j47)
        if r:
            gender_rows, gsum = r
    if "C" in PARTS:
        r = opt("Part C", part_c, counts, expo, s61, j47)
        if r:
            cl_rows = r
    if "D" in PARTS:
        r = opt("Part D", part_d, counts, expo, frame19, s61, j47, h47)
        if r:
            skill_rows, skill_v = r
    if "E" in PARTS:
        r = opt("Part E", part_e, expo, s61, j47)
        if r:
            prof_rows = r
    if "F" in PARTS:
        r = opt("Part F", part_f, counts, expo, s61, j47)
        if r:
            ind_rows = r
    if "G" in PARTS:
        r = opt("Part G", part_g, counts, expo, s61, j47)
        if r:
            bnd_rows = r
    del counts
    gc.collect()

    # ---- summary ------------------------------------------------------
    L = ["SEVEN CHECKS ON THE HEADLINE DESIGN", "=" * 52, "",
         "Employer x age x month counts, exposure the top quartile of the",
         "2019 education mix, employer-by-month, employer-by-age and",
         "month-by-age effects, Poisson, calendar cycle removed. Each part",
         "asks one question of that design and changes nothing else.", ""]
    if "A" in PARTS:
        L += ["A. THE PRE-PERIOD"]
        if path_rows:
            L += [f"  (i) quarterly path on the plain specification, {REF_QUARTER} "
                  "omitted, no calendar terms:"]
            for band in YOUNG_BANDS:
                rs = [d for d in path_rows if d["young_band"] == band]
                if not rs:
                    continue
                L.append(f"    {band}:")
                for d in sorted(rs, key=lambda d: d["quarter"]):
                    if d["status"] == "reference":
                        L.append(f"      {d['quarter']:<8} reference")
                    else:
                        star = "" if abs(d["coef"]) < 2 * d["se"] else "  *"
                        L.append(f"      {d['quarter']:<8} {d['coef']:+.4f} "
                                 f"({d['se']:.4f}){star}")
        if drift_rows:
            L += ["  (ii) drift test on the pre-launch months (January 2021 to "
                  "November 2022), cycle and tightening window in:"]
            D = pd.DataFrame(drift_rows)
            for band in YOUNG_BANDS:
                g = D[D["young_band"] == band].set_index("term")
                if "trend_x_high_x_young" not in g.index:
                    continue
                c, s = (float(g.loc["trend_x_high_x_young", "coef"]),
                        float(g.loc["trend_x_high_x_young", "se"]))
                flat = abs(c) <= DRIFT_RULE_SE * s
                L.append(f"    {band}: trend {c:+.5f} ({s:.5f}) per month, "
                         f"t {c/max(s,1e-12):+.2f}; over the twenty-three months "
                         f"{23*c:+.4f}: "
                         f"{'FLAT' if flat else 'NOT FLAT'} on the rule "
                         f"(within {DRIFT_RULE_SE:.0f} SE)")
                if "rbw_x_high_x_young" in g.index:
                    L.append(f"      tightening window {float(g.loc['rbw_x_high_x_young','coef']):+.4f} "
                             f"({float(g.loc['rbw_x_high_x_young','se']):.4f})")
            L += ["    A joint test of every pre-launch quarter net of the cycle is",
                  "    not identified from two years of pre-launch months; the",
                  "    linear drift is the testable direction and is what is tested."]
        L.append("")
    if "B" in PARTS:
        L += ["B. THE SEX SPLIT ON EQUATION (2), 22-25 stock, every term x female:"]
        if gsum:
            for k, nm in (("male_step", "young men, adoption step"),
                          ("female_differential", "young women minus young men"),
                          ("female_step", "young women, adoption step"),
                          ("male_step_from_2023", "young men, step from the 2023 level"),
                          ("female_step_from_2023", "young women, step from the 2023 level")):
                if k in gsum:
                    c, s = gsum[k]
                    L.append(f"  {nm:<40} {c:+.4f} {fmt(s)}")
            L += ["  All from the same base as Table 1: steps are from the level",
                  "  of the tightening months; the 2023 rows are post minus interim."]
        else:
            L.append("  no fit came back")
        L.append("")
    if "C" in PARTS:
        L += ["C. STANDARD ERRORS BY THREE-DIGIT INDUSTRY beside employer:"]
        if cl_rows:
            C = pd.DataFrame(cl_rows)
            for band in YOUNG_BANDS:
                g = C[C["young_band"] == band].set_index("term")
                for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                           "post_x_high_x_young"):
                    if t_ in g.index:
                        r_ = g.loc[t_]
                        L.append(f"  {band} {t_:<26} {r_['coef']:+.4f}  "
                                 f"employer ({r_['se_employer']:.4f})  "
                                 f"industry ({r_['se_industry']:.4f})  "
                                 f"clusters {int(r_['n_clusters_industry']):,}")
                ok = g["coef_match_4dp"].dropna()
                if len(ok):
                    L.append(f"  {band}: coefficients equal the employer-clustered "
                             f"run to four decimals: {'YES' if bool(ok.all()) else 'NO'}")
        else:
            L.append("  no fit came back")
        L.append("")
    if "D" in PARTS:
        L += ["D. THE SKILL-INTENSITY PLACEBO (High_edu = top quartile of the",
              "   2019 post-secondary share, employment weighted):"]
        for band, v in skill_v.items():
            if "ai_alone" in v:
                L.append(f"  {band} DAIOE step alone        {v['ai_alone'][0]:+.4f} ({v['ai_alone'][1]:.4f})")
            if "skill_alone" in v:
                L.append(f"  {band} skill-cut step alone    {v['skill_alone'][0]:+.4f} ({v['skill_alone'][1]:.4f})")
            if "ai_both" in v:
                L.append(f"  {band} DAIOE step, both in     {v['ai_both'][0]:+.4f} ({v['ai_both'][1]:.4f})")
            if "skill_both" in v:
                L.append(f"  {band} skill-cut step, both in {v['skill_both'][0]:+.4f} ({v['skill_both'][1]:.4f})")
            if "reproduces" in v:
                L.append(f"  {band}: {'SKILL CUT REPRODUCES THE STEP' if v['reproduces'] else 'skill cut alone does not reproduce the step'} "
                         "(rule: at least as large as the DAIOE step, allowing one SE)")
            if "survives" in v:
                L.append(f"  {band}: {'AI SURVIVES THE SKILL CUT' if v['survives'] else 'SKILL CUT ABSORBS THE STEP'} "
                         f"(rule: DAIOE step keeps at least {SKILL_KEEP:.0%} of its size with both in)")
        if not skill_v:
            L.append("  no fit came back")
        L.append("")
    if "E" in PARTS:
        L += ["E. THE PROFILE AGAINST 41-49, CYCLE REMOVED, 50 AND OVER SPLIT AT 65:"]
        for d in prof_rows:
            if d["status"] == "reference":
                L.append(f"  {d['band']:<6} reference")
            else:
                star = "" if abs(d["coef"]) < 2 * d["se"] else "  *"
                L.append(f"  {d['band']:<6} {d['coef']:+.4f} ({d['se']:.4f}){star}")
        if prof_rows:
            L.append(f"  firms {prof_rows[0]['n_firms']:,}. Read 50-64 against 65-69: "
                     "a gain confined to 65-69 is the retirement-age rival, a gain "
                     "shared by 50-64 is not.")
        else:
            L.append("  no fit came back")
        L.append("")
    if "F" in PARTS:
        L += ["F. INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS, 22-25 stock:"]
        if ind_rows:
            I = pd.DataFrame(ind_rows)
            for spec_ in ("baseline_same_sample", "industry_age_month"):
                g = I[I["spec"] == spec_].set_index("term")
                for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                           "post_x_high_x_young"):
                    if t_ in g.index:
                        L.append(f"  {spec_:<22} {t_:<26} "
                                 f"{float(g.loc[t_,'coef']):+.4f} ({float(g.loc[t_,'se']):.4f})")
            gb = I[I["spec"] == "baseline_same_sample"].set_index("term")
            gi = I[I["spec"] == "industry_age_month"].set_index("term")
            if "post_x_high_x_young" in gb.index and "post_x_high_x_young" in gi.index:
                pb, pi = float(gb.loc["post_x_high_x_young", "coef"]), \
                    float(gi.loc["post_x_high_x_young", "coef"])
                L.append(f"  retained share of the adoption step against the same-sample "
                         f"baseline: {pi/pb if pb else float('nan'):.0%}")
        else:
            L.append("  no fit came back")
        L.append("")
    if "G" in PARTS:
        L += [f"G. THE TIGHTENING BOUNDARY AT {MAY_BOUNDARY}, {mc.RIKSBANK_YM} DROPPED, 22-25 stock:"]
        if bnd_rows:
            B = pd.DataFrame(bnd_rows).set_index("term")
            for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                       "post_x_high_x_young"):
                if t_ in B.index:
                    L.append(f"  {t_:<26} {float(B.loc[t_,'coef']):+.4f} ({float(B.loc[t_,'se']):.4f})")
            L.append("  Read beside 68's April-boundary rows; no gate.")
        else:
            L.append("  no fit came back")
        L.append("")
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += ["FITS THAT FAILED: " + "; ".join(FAILURES),
              "A missing row is a missing fit, never a zero.", ""]
    L += ["READ THIS BEFORE QUOTING ANY OF IT:",
          "  1. A(i) is drawn, not read as a pooled estimate; A(ii)'s rule is",
          "     the only verdict in Part A.",
          "  2. B replaces 68's female differential wherever the paper wants",
          "     the sexes on Table 1's base; 68's number stays in the OA with",
          "     its own base stated.",
          "  3. C changes no coefficient; a coefficient that differs from 68's",
          "     means the panel differs and nothing in C is quoted.",
          "  4. D's two verdicts are both reported whichever way they fall.",
          "  5. E, F and G carry no gate; each is read beside the paper's own",
          "     number for the same object.",
          "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "78_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("78_final_checks", 0, (time.time() - t0) / 60)
    print("\n78 done.")


if __name__ == "__main__":
    main()

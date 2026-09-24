#!/usr/bin/env python3
"""
79_last_gaps.py: three further checks beside script 78: inference by
industry for the sexes, the industry-and-cycle fit at 26 to 30, and a
count of the payslips the panel never sees.

======================================================================
  RUNS IN MONA. Parts A and B read the caches 47h, 47L and 67 wrote and
  make one SQL read each of LISA's firm table for the 2019 industry code
  (the read script 73 makes). Part C is SQL and aggregation only: one
  pull per year 2019 to 2025 over the employer declarations, cached, and
  no fit. Writes output_79/. Parts are chosen with the environment
  variable CANARIES_79_PARTS (default ABC) and the folder with
  CANARIES_79_OUT (default output_79); master.py sets both.
  Budget: A two to three hours and the heaviest thing here (one Poisson
  fit on the 40.5 million-row sex panel, two if script 78's employer-
  clustered export is not on the share), B about two hours (two fits,
  one of them with four sets of effects), C well under one hour.
======================================================================

THE DESIGN EVERY PART USES. Employer x age band x month counts, the
young band beside the four bands aged 31 to 69, exposure the top quartile
of the employer's 2019 education mix (47h, incumbents aged 31 to 69),
employer-by-month, employer-by-age and month-by-age effects (47j), Poisson
pseudo-maximum likelihood, the calendar cycle removed by three
quarter-of-year interactions with the fourth quarter omitted (68). Nothing
here changes that design. Part A changes only what the standard errors are
clustered on, Part B only which effects absorb the cell, and Part C fits
nothing at all.

A. THE SEXES, CLUSTERED BY INDUSTRY. Script 78's Part C showed that the
   22-25 adoption step is not distinguishable from zero once the standard
   errors allow common three-digit industry disturbances, while the 26-30
   step survives. The paper's sex result (78's Part B: young women -0.0793,
   the differential -0.0746) is estimated on the same employers and carries
   the same objection, and it is a headline sentence. This part re-runs
   Part B exactly, the sex panel of 67 with every term of Equation (2)
   interacted with female and with sex-specific employer-by-age and
   month-by-age effects, changing one thing: the cluster is the employer's
   2019 SNI 2007 three-digit code from LISA's firm table, a firm without a
   code being its own cluster. Econometrically nothing but the covariance
   moves, so the coefficients must reproduce, and the check is the gate.
   The employer-clustered standard errors beside them are script 78's own
   export (output_78b/gender_eq2.csv) when it is on the share, and one
   further fit here when it is not. The clustered covariance of the fit
   leaves with the exports, so the women's step, which is the male step
   plus the differential, carries an industry-clustered standard error as
   well as an employer-clustered one.
   Read rule, fixed before the run: the coefficients must reproduce to
   four decimals or nothing from this part is quoted. The differential and
   the women's step are reported under both clusterings whatever they show.

B. INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS, AT 26 TO 30.
   78's Part F put the calendar terms and industry-by-age-by-month effects
   in one fit at 22-25 and kept 82 per cent of the adoption step. The 26-30
   band is the one whose step survives industry clustering, so it is the
   band whose industry test the paper now leans on, and the only estimate
   for it is script 73's, fitted before the cycle was removed. This part is
   the 26-30 counterpart: the stock fit with the three calendar terms and
   three-digit industry by age by month effects, the month-by-age effect
   dropped because it is nested inside them (73 and 78 drop it for the same
   reason), beside a baseline on the same industry-linked sample.
   Read rule: no gate. The retained share of the adoption step against the
   same-sample baseline is reported whatever it is.

C. THE UNCOUNTED PAYSLIPS. Every count in this paper comes from a worker
   whose birth year and sex could be read from the 2023, 2021 or 2019
   individual register. A worker in none of the three, in practice somebody
   first registered in Sweden during 2024 or 2025, carries a payslip that
   the panel never counts. If those workers are more common in exposed
   firms and at young ages, the fall we measure is partly a measurement
   artefact, and that is the coverage mechanism stated as a testable
   quantity. This part counts them: for 2024 and 2025, and for 2019 to 2023
   as the comparison, the person-months in the employer declarations whose
   worker returns no birth year and no sex from any of the three registers,
   by the employer's 2019 exposure quartile with the unscored employers as
   their own group, and by age band where the declaration itself carries
   age information. It does not, in this delivery: the Arb_AGIIndivid
   tables hold the employer, the person, the period, the cash compensation
   and the benefit bases, and no birth year. The script probes
   INFORMATION_SCHEMA for an age-bearing column rather than assuming that,
   uses it if one is there, and says in the summary that the share is
   unbroken by age when there is not.
   Read rule: the paper reports the figure whatever it shows. A share that
   is higher in exposed firms at young ages is the coverage mechanism and
   is stated plainly if it appears. Counts below five are suppressed before
   anything leaves MONA, as everywhere in this tree.

WHAT EACH PART COSTS IF IT DIES. A and B lose their own fits and nothing
else. C caches each year as it lands, so a kill costs the year that was
running, not the pull.

Output (output_79/):
  gender_cluster_industry.csv   A: every term, both standard errors, the
                                cluster count and the reproduction check
  industry_seasonal_2630.csv    B: both specifications, every treatment
                                term, the employer count, the retained share
  uncounted_share.csv           C: person-months by year, exposure quartile
                                and age band, counts and shares, floored
  vcov_s79_*.csv                the clustered covariance of every fit
  79_summary.txt

IN THE PAPER. The fits of Parts A and B are on the education-route score
and are not quoted. Part C's functions (probe_declaration_age,
uncounted_counts, uncounted_table, floor_table) are imported by script 93,
which re-cuts the uncounted payslips on the occupation-route quartiles for
Online Appendix VI.1 and Table A33.
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
OUT = HERE / os.environ.get("CANARIES_79_OUT", "output_79")
PARTS = os.environ.get("CANARIES_79_PARTS", "ABC").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

POST_FROM = "2024-01"                 # adoption; checked against 78's below
YOUNG_SEX = "22-25"                   # A: the band the paper's sex result is on
BAND_B = "26-30"                      # B: the band whose step survives industry SEs
UNCOUNTED_YEARS = list(range(2019, 2026))   # C: the declaration years available
NEW_YEARS = [2024, 2025]              # C: where the editor's mechanism would sit
FLOOR = 5                             # C: the export floor, as in mona_common
# A reads script 78's employer-clustered export from whichever folder that
# run wrote: output_78b when Parts B and C ran as a job of their own,
# output_78 for the full script; the others are listed so a differently
# split submission is still found.
LANE25_ROOT = HERE
LANE25_DIRS = ("output_78b", "output_78", "output_78a", "output_78c")
GENDER_EXPORT = "gender_eq2.csv"
GENDER_VCOV = "vcov_s78_gender_eq2_22_25.csv"
MATCH_DP = 4                          # A: the reproduction check, decimals
NOTES = []
FAILURES = []


def opt(label, fn, *a, **kw):
    """Run one part. A part that dies is recorded and the others still run;
    nothing partial is silently treated as a result."""
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


def load_modules():
    """
    The scripts this one reuses rather than reimplements.

    61 builds the balanced employer x band x month skeleton, 67 the same
    skeleton with sex as a fourth dimension, 78 the term sets of Equation
    (2) and of its sex split, 47j the exposure and the fixed-effect list,
    47h the education score book. Importing 78 rather than copying its
    term builders is the point: Part A must be Part B of 78 with one
    argument changed, and a copied term list could drift from it.
    """
    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    s78 = _mod("78_final_checks.py", "s78")
    # 78's module-level OUT and CACHE are its own. Point them here so that
    # anything it writes lands with this script's exports rather than in
    # script 78's folder, and so a test that moves the cache moves both.
    s78.OUT, s78.CACHE = OUT, CACHE
    # The term builders are 78's, so the adoption date is 78's too. If the
    # two ever disagree, this script's docstring and read rules describe a
    # model it is not fitting, which is worse than a crash.
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(
            f"78's adoption date is {s78.POST_FROM} and this script says "
            f"{POST_FROM}; the terms come from 78, so settle it there first")
    j47 = s61._j47()
    h47 = j47._h47()
    return s61, s67, s78, j47, h47


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


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """
    One Poisson fit. Returns (coefficient table indexed by term, the
    clustered covariance as a DataFrame or None). A failure is recorded
    and returns (None, None), because a missing row must never be read as
    a zero.
    """
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s79_{tag}", terms=terms,
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
    """Standard error of sum(w_i * term_i) from a clustered covariance."""
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


def industry_map() -> dict:
    """
    employer_id (normalised string) -> 2019 three-digit SNI 2007 code.

    Read through script 73's own loader, so that this script, 73 and 78
    cluster and absorb on exactly the same codes from exactly the same
    table. 73's source order and its coverage note travel with it.
    """
    s73 = _mod("73_industry_and_credit.py", "s73_ind")
    conn = mc.connect()
    schema = s73.discover(conn)
    if schema.empty:
        raise RuntimeError("no LISA firm table visible; the industry parts "
                           "cannot run")
    ind = s73.firm_industry(conn, schema)
    NOTES.extend(n for n in s73.NOTES
                 if n.startswith("industry") and n not in NOTES)
    try:
        conn.close()
    except Exception:
        pass
    return dict(zip(s73.norm_id(ind["employer_id"]), ind["ind3"].astype(str)))


# ----------------------------------------------------------------------
# Part A: the sexes, with the standard errors clustered by industry
# ----------------------------------------------------------------------

def lane25_gender(terms: list):
    """
    Script 78's employer-clustered sex fit (its Part B), if it is on the share.

    78's Part B fitted exactly this panel with the standard errors
    clustered by employer and exported the coefficient table and its
    covariance. Reading them costs nothing and saves an hour of fitting;
    the reproduction check below is what proves the two runs are the same
    panel, so reading rather than refitting cannot hide a difference.
    Returns (coefficient table indexed by term, covariance or None, a
    sentence naming the source) or (None, None, "") when nothing is found.
    """
    for d in LANE25_DIRS:
        f = Path(LANE25_ROOT) / d / GENDER_EXPORT
        if not f.exists():
            continue
        try:
            g = pd.read_csv(f)
        except Exception as ex:
            print(f"  A: {f} unreadable ({type(ex).__name__}); ignored")
            continue
        if "term" not in g.columns or "coef" not in g.columns:
            print(f"  A: {f} is not a coefficient table; ignored")
            continue
        if "young_band" in g.columns:
            g = g[g["young_band"].astype(str) == YOUNG_SEX]
        if not set(terms) <= set(g["term"].astype(str)):
            print(f"  A: {f} does not carry every term of Equation (2); "
                  f"ignored")
            continue
        v = None
        vf = Path(LANE25_ROOT) / d / GENDER_VCOV
        if vf.exists():
            try:
                v = pd.read_csv(vf).set_index("term")
            except Exception:
                v = None
        return (g.set_index("term"), v,
                f"employer-clustered numbers read from {d}/{GENDER_EXPORT}"
                + ("" if v is not None else
                   f" (no {GENDER_VCOV} beside it, so the women's step has "
                   f"no employer-clustered standard error)"))
    return None, None, ""


def part_a(expo, s61, s67, s78, j47) -> tuple:
    """
    78's Part B, clustered by three-digit industry.

    Estimates: the sex panel of 67 (employer x age band x sex x month
    counts from January 2021, the young band beside the four incumbent
    bands, sex-specific employer-by-age and month-by-age effects and the
    usual employer-by-month effect), Poisson, with every term of Equation
    (2) entered three ways, as High x Young, as High x Female and as
    High x Young x Female. The one change is the cluster: the employer's
    2019 three-digit industry rather than the employer.

    Reads the L_counts_sex caches 67 wrote and LISA's firm table for the
    code. Writes gender_cluster_industry.csv and, through the fit,
    vcov_s79_gender_clind_22_25.csv.
    """
    imap = industry_map()
    s73 = _mod("73_industry_and_credit.py", "s73_a")
    sex = load_counts("L_counts_sex", s61.PANEL_YEARS,
                      require=["employer_id", "year_month", "age_group",
                               "gender", "n_emp"])
    if sex is None:
        FAILURES.append("A/no L_counts_sex cache")
        print("  A: L_counts_sex_* missing, skipped")
        return [], {}
    skel = s67.build_skeleton_sex(sex, YOUNG_SEX, j47, "n_emp")
    del sex
    gc.collect()
    if skel.empty:
        FAILURES.append("A/empty")
        return [], {}
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("A/no exposure")
        return [], {}
    # The cluster label is built once per EMPLOYER and then mapped, not
    # built once per cell: this panel is forty million rows, and a text
    # column that long costs gigabytes for a grouping key whose labels are
    # never read. The label is an integer, which is what the exchange file
    # would have factorised it to in any case.
    emp_u = b["employer_id"].drop_duplicates()
    key = s73.norm_id(emp_u)
    code = key.map(imap)
    n_own = int(code.isna().sum())
    label = np.where(code.isna(), "own_" + key.astype(str), code.astype(str))
    lut = pd.Series(pd.factorize(label)[0], index=emp_u.to_numpy())
    b["cl_ind"] = b["employer_id"].map(lut).astype("int64")
    n_firms = int(len(emp_u))
    n_clusters = int(lut.nunique())
    msg = (f"A: {n_firms - n_own:,} of {n_firms:,} employers on the sex panel "
           f"carry a 2019 industry code; {n_own:,} without one are their own "
           f"cluster; {n_clusters:,} clusters")
    print(f"  {msg}")
    NOTES.append(msg)
    b, terms = s78.gender_eq2_terms(b)
    g_ind, v_ind = fit(b, "gender_clind_22_25", terms, j47.FES,
                       cluster="cl_ind")
    if g_ind is None:
        del b
        gc.collect()
        return [], {}
    emp, v_emp, src = lane25_gender(terms)
    if emp is None:
        print("  A: no lane 25 export on the share; the employer-clustered "
              "run is repeated here")
        emp, v_emp = fit(b, "gender_clemp_22_25", terms, j47.FES)
        src = "employer-clustered numbers refitted in this run"
    del b
    gc.collect()
    NOTES.append(f"A: {src}")
    if emp is not None and "n_obs" in emp.columns and "n_obs" in g_ind.columns:
        n_e = int(emp["n_obs"].iloc[0])
        n_i = int(g_ind["n_obs"].iloc[0])
        if n_e != n_i:
            NOTES.append(f"A: the employer-clustered source has {n_e:,} cells "
                         f"and this fit {n_i:,}; the panels differ and the "
                         f"reproduction check below is the verdict")
    rows = []
    for t_ in terms:
        if t_ not in g_ind.index:
            continue
        ci = float(g_ind.loc[t_, "coef"])
        has = emp is not None and t_ in emp.index
        ce = float(emp.loc[t_, "coef"]) if has else np.nan
        se_e = float(emp.loc[t_, "se"]) if has else np.nan
        rows.append({"young_band": YOUNG_SEX, "term": t_, "coef": ci,
                     "se_employer": se_e,
                     "se_industry": float(g_ind.loc[t_, "se"]),
                     "coef_employer_run": ce,
                     "coef_match_4dp": (bool(round(ci, MATCH_DP)
                                             == round(ce, MATCH_DP))
                                        if not np.isnan(ce) else None),
                     "n_clusters_industry": n_clusters,
                     "n_obs": int(g_ind.loc[t_, "n_obs"]),
                     "status": str(g_ind.loc[t_].get("status", "ok"))})
    pd.DataFrame(rows).to_csv(OUT / "gender_cluster_industry.csv", index=False)
    # The three numbers the paper quotes, each under both clusterings. The
    # women's step is the male step plus the differential, so its standard
    # error comes from the covariance of the fit and not from adding two
    # standard errors.
    m, d = "post_x_high_x_young", "post_x_high_x_young_x_female"
    mi, di = "interim_x_high_x_young", "interim_x_high_x_young_x_female"
    summ = {}
    if m in g_ind.index and d in g_ind.index:
        ci_m, ci_d = float(g_ind.loc[m, "coef"]), float(g_ind.loc[d, "coef"])
        summ["male_step"] = (ci_m,
                             float(emp.loc[m, "se"]) if (emp is not None and m in emp.index) else None,
                             float(g_ind.loc[m, "se"]))
        summ["female_differential"] = (ci_d,
                                       float(emp.loc[d, "se"]) if (emp is not None and d in emp.index) else None,
                                       float(g_ind.loc[d, "se"]))
        summ["female_step"] = (ci_m + ci_d, lincomb(v_emp, {m: 1, d: 1}),
                               lincomb(v_ind, {m: 1, d: 1}))
    if all(k in g_ind.index for k in (m, d, mi, di)):
        c = {k: float(g_ind.loc[k, "coef"]) for k in (m, d, mi, di)}
        summ["male_step_from_2023"] = (
            c[m] - c[mi], lincomb(v_emp, {m: 1, mi: -1}),
            lincomb(v_ind, {m: 1, mi: -1}))
        summ["female_step_from_2023"] = (
            c[m] + c[d] - c[mi] - c[di],
            lincomb(v_emp, {m: 1, d: 1, mi: -1, di: -1}),
            lincomb(v_ind, {m: 1, d: 1, mi: -1, di: -1}))
    ok = [r["coef_match_4dp"] for r in rows if r["coef_match_4dp"] is not None]
    summ["reproduces"] = bool(ok) and all(ok)
    summ["n_clusters"] = n_clusters
    summ["n_firms"] = n_firms
    return rows, summ


# ----------------------------------------------------------------------
# Part B: industry x age x month effects with the calendar terms, 26-30
# ----------------------------------------------------------------------

def part_b(counts, expo, s61, s78, j47) -> list:
    """
    78's Part F at 26 to 30.

    Estimates: the 26-30 stock fit of Equation (2), the tightening switch,
    the interim window, the adoption step and the three calendar terms,
    each interacted with High x Young, twice on the same firms. The
    baseline carries 47j's three effects; the industry specification
    replaces the month-by-age effect, which is nested inside it, with
    three-digit industry by age band by month. Both fits run on the
    employers that carry a 2019 industry code, so the two coefficients are
    comparable and the ratio between them is a retained share rather than
    a sample difference.

    Reads 47L's counts caches and LISA's firm table. Writes
    industry_seasonal_2630.csv and the two covariance files.
    """
    imap = industry_map()
    s73 = _mod("73_industry_and_credit.py", "s73_b")
    band = BAND_B
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        FAILURES.append("B/empty")
        return []
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("B/no exposure")
        return []
    # As in Part A, the industry is resolved once per employer and mapped
    # in as an integer code; an employer without a 2019 code leaves the
    # sample, which is what the inner join in 78's Part F does, and both
    # fits then run on exactly the firms that carry a code.
    emp_u = b["employer_id"].drop_duplicates()
    code = s73.norm_id(emp_u).map(imap)
    lut = pd.Series(pd.factorize(code)[0], index=emp_u.to_numpy())
    n_all = int(len(emp_u))
    b["ind_code"] = b["employer_id"].map(lut).astype("int64")
    b = b[b["ind_code"] >= 0].copy()
    if b.empty:
        FAILURES.append("B/no industry")
        return []
    n_firms = int(b["employer_id"].nunique())
    msg = (f"B: {n_firms:,} of {n_all:,} employers on the {band} panel carry a "
           f"2019 industry code and form the sample for both fits")
    print(f"  {msg}")
    NOTES.append(msg)
    ic = b["ind_code"].to_numpy(dtype="int64")
    ac = pd.factorize(b["age_group"], sort=False)[0].astype("int64")
    tc = pd.factorize(b["year_month"], sort=False)[0].astype("int64")
    n_a, n_t = int(ac.max()) + 1, int(tc.max()) + 1
    b["fe_ind_age_t"] = (ic * n_a + ac) * n_t + tc
    b, terms = s78.eq2_terms(b)
    tag = band.replace("-", "_")
    g_base, _ = fit(b, f"indseas_base_{tag}", terms, j47.FES)
    # fe_t_age is nested inside industry x age x month and is dropped, as
    # scripts 73 and 78 drop it: the model is the same and one effect fewer
    # fits.
    fes = tuple(f for f in j47.FES if f != "fe_t_age") + ("fe_ind_age_t",)
    g_ind, _ = fit(b, f"indseas_ind_{tag}", terms, fes)
    del b
    gc.collect()
    post = "post_x_high_x_young"
    pb = (float(g_base.loc[post, "coef"])
          if g_base is not None and post in g_base.index else np.nan)
    pi = (float(g_ind.loc[post, "coef"])
          if g_ind is not None and post in g_ind.index else np.nan)
    retained = (pi / pb) if (pb == pb and pi == pi and pb != 0) else np.nan
    rows = []
    for spec_, g, share in (("baseline_same_sample", g_base, 1.0),
                            ("industry_age_month", g_ind, retained)):
        if g is None:
            continue
        rows += s78.rows_of(g, terms, young_band=band, spec=spec_,
                            n_firms=n_firms, retained_share=share)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "industry_seasonal_2630.csv",
                                  index=False)
    return rows


# ----------------------------------------------------------------------
# Part C: the payslips the panel never counts
# ----------------------------------------------------------------------

STATUSES = ["counted", "no_register", "register_no_birth_or_sex",
            "outside_age_range"]
# What a column has to be called before this script will read an age out of
# it. Deliberately narrow: a loose pattern that matched a benefit base
# would put every payslip in the wrong band and nothing downstream would
# notice.
AGE_PATTERNS = ((r"fodelse|birth", "birth"), (r"^alder$|_alder$|^age$|_age$", "age"))


def agi_table(year: int, month: int) -> str:
    """The monthly employer declaration table, named as 47L names it."""
    suffix = "_def" if year < 2025 else "_prel"
    return f"Arb_AGIIndivid{year}{month:02d}{suffix}"


def agi_months(year: int) -> int:
    return 12 if year < 2025 else 6


def probe_declaration_age(conn, years) -> tuple:
    """
    Does the employer declaration itself carry an age?

    The panel reads birth year and sex from the individual registers, so
    the workers it cannot place are exactly the ones with no register row,
    and their age is unknown FROM THE REGISTER. If the declaration carried
    a birth year or an age of its own, those workers could still be put in
    a band, which is what the age breakdown of this part would need. The
    delivered Arb_AGIIndivid tables carry the employer, the person, the
    period, the cash compensation, the benefit bases, the workplace number
    and a first-employee flag, and no age. That is read out of
    INFORMATION_SCHEMA here rather than assumed, because a delivery can
    change.

    Returns (column, kind, note). kind is 'birth' for a birth year and
    'age' for an age in years; (None, None, note) when the declaration
    carries neither, and then the share is reported unbroken by age.
    """
    import re
    tables = [agi_table(y, 1) for y in years]
    names = ", ".join("'" + t + "'" for t in tables)
    q = (f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
         f"FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME IN ({names})")
    cat = pd.read_sql(q, conn)
    if cat.empty:
        return None, None, ("C: INFORMATION_SCHEMA returned no columns for "
                            "the January declaration tables, so no age column "
                            "could be looked for and the share is unbroken by "
                            "age")
    cat["COLUMN_NAME"] = cat["COLUMN_NAME"].astype(str)
    for pattern, kind in AGE_PATTERNS:
        hit = sorted({c for c in cat["COLUMN_NAME"]
                      if re.search(pattern, c, re.I)})
        if not hit:
            continue
        col = hit[0]
        have = set(cat.loc[cat["COLUMN_NAME"] == col, "TABLE_NAME"]
                   .astype(str).str.lower())
        missing = [t for t in tables if t.lower() not in have]
        if missing:
            return None, None, (
                f"C: the declaration carries {col!r} in some years but not in "
                f"{', '.join(missing)}, so it is not used and the share is "
                f"unbroken by age")
        return col, kind, (f"C: the declaration carries {col!r}, read as a "
                           f"{'birth year' if kind == 'birth' else 'age in years'}, "
                           f"so the share is broken down by the band the worker "
                           f"would fall in")
    return None, None, (
        "C: the employer declaration carries no birth year and no age of its "
        "own (columns seen: "
        + ", ".join(sorted(set(cat["COLUMN_NAME"]))[:12])
        + "), so a worker with no register row cannot be placed in a band and "
          "the share is reported unbroken by age")


def _band_case(age_col) -> str:
    """The age band a declaration row falls in, read off the decl_age the
    query has already computed from the declaration's own age column.
    'all' when there is no such column, so the shape of the answer does not
    change with the delivery."""
    if not age_col:
        return "'all'"
    arms = "\n".join(
        f"             WHEN decl_age BETWEEN {lo} AND {hi} THEN '{b}'"
        for b, (lo, hi) in mc.AGE_GROUPS.items())
    return ("CASE\n"
            "             WHEN decl_age IS NULL THEN 'unknown'\n"
            "             WHEN decl_age < 22 THEN 'under 22'\n"
            f"{arms}\n"
            "             ELSE 'over 69' END")


def q_uncounted(year: int, conn, age_col=None, age_kind=None) -> pd.DataFrame:
    """
    Person-months in one year of employer declarations, by employer, by
    what the individual registers could say about the worker, and by the
    band the declaration itself would put the worker in.

    Four statuses, and they exhaust the declarations:
      counted                   a birth year and a sex are readable from
                                Individ_2023, 2021 or 2019 and the age is
                                22 to 69, which is the panel's own rule
      no_register               the person is in none of the three, which
                                in practice means a first registration in
                                Sweden after the 2023 register closed
      register_no_birth_or_sex  a register row exists but returns no birth
                                year, or no sex coded 1 or 2
      outside_age_range         everything is readable and the worker is
                                under 22 or over 69

    Person-months rather than rows: the count is distinct persons within
    employer and month, summed over the months of the year, so a corrected
    declaration or a duplicated register row cannot inflate it.
    """
    n_months = agi_months(year)
    age_sel = ("NULL" if not age_col
               else (f"TRY_CAST(agi.[{age_col}] AS INT)" if age_kind == "age"
                     else f"{year} - TRY_CAST(agi.[{age_col}] AS INT)"))
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               CASE WHEN a.P1207_LopNr_PersonNr IS NOT NULL
                      OR b.P1207_LopNr_PersonNr IS NOT NULL
                      OR c.P1207_LopNr_PersonNr IS NOT NULL
                    THEN 1 ELSE 0 END AS in_register,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               LTRIM(RTRIM(COALESCE(a.Kon, b.Kon, c.Kon))) AS kon,
               {age_sel} AS decl_age
        FROM dbo.{agi_table(year, m)} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, n_months + 1))
    q = f"""
    WITH base AS ({monthly}),
    cls AS (
        SELECT employer_id, period, person_id,
               CASE
                 WHEN in_register = 0 THEN 'no_register'
                 WHEN fodelse IS NULL OR kon IS NULL OR kon NOT IN ('1','2')
                   THEN 'register_no_birth_or_sex'
                 WHEN {year} - fodelse BETWEEN 22 AND 69 THEN 'counted'
                 ELSE 'outside_age_range' END AS status,
               {_band_case(age_col)} AS decl_band
        FROM base),
    percell AS (
        SELECT employer_id, period, status, decl_band,
               COUNT(DISTINCT person_id) AS n
        FROM cls
        GROUP BY employer_id, period, status, decl_band)
    SELECT employer_id, status, decl_band,
           SUM(n) AS n_personmonths
    FROM percell
    GROUP BY employer_id, status, decl_band
    """
    return pd.read_sql(q, conn)


def uncounted_counts(years, age_col, age_kind, conn) -> dict:
    """One cached frame per year. A kill costs the year that was running."""
    tag = (age_col or "noage").lower()
    out = {}
    for y in years:
        cf = CACHE / f"U_uncounted_{y}_{tag}.parquet"
        c = mc.read_cache(cf, require=["employer_id", "status", "decl_band",
                                       "n_personmonths"])
        if c is None:
            t = time.time()
            c = q_uncounted(y, conn, age_col, age_kind)
            mc.write_cache(c, cf)
            print(f"  C: declarations {y}: {len(c):,} employer cells, "
                  f"{int(c['n_personmonths'].sum()):,} person-months "
                  f"({(time.time()-t)/60:.1f} min)")
        out[y] = c
    return out


def uncounted_table(frames: dict, expo, s73) -> pd.DataFrame:
    """
    Counts and shares by year, 2019 exposure quartile and age band.

    The quartile is the employer's own 2019 education-mix quartile, the
    one the design treats on; an employer the score book never scored is
    its own group rather than being dropped, since the uncounted payslips
    are as likely to sit in a firm too small or too new to be scored as
    in a scored one. Rows are added for every quartile and for every band
    together, so the paper can quote one share and the contrast behind it
    from the same table.
    """
    qmap = dict(zip(s73.norm_id(expo["employer_id"]),
                    expo["fq"].astype(int)))
    out = []
    for y in sorted(frames):
        d = frames[y].copy()
        d["status"] = d["status"].astype(str).str.strip()
        d["decl_band"] = d["decl_band"].astype(str).str.strip()
        g = s73.norm_id(d["employer_id"]).map(qmap)
        grp = pd.Series("unscored", index=d.index, dtype=object)
        ok = g.notna()
        grp[ok] = "Q" + g[ok].astype(int).astype(str)
        d["quartile_group"] = grp
        share_scored = float((d.loc[ok, "n_personmonths"].sum())
                             / max(d["n_personmonths"].sum(), 1))
        NOTES.append(f"C: {y}: {share_scored:.1%} of person-months sit in an "
                     f"employer the 2019 score book scored")
        base = (d.groupby(["quartile_group", "decl_band", "status"],
                          observed=True)["n_personmonths"].sum().reset_index())
        # The cell, then the two margins, then the year as a whole: the
        # paper quotes one share and the contrast behind it, and both come
        # off this table rather than out of a second calculation.
        for k in (["quartile_group", "decl_band"], ["quartile_group"],
                  ["decl_band"], []):
            if k:
                p = (base.groupby(k + ["status"], observed=True)
                     ["n_personmonths"].sum()
                     .unstack("status", fill_value=0).reset_index())
            else:
                p = (base.groupby("status", observed=True)["n_personmonths"]
                     .sum().to_frame().T.reset_index(drop=True))
            for s_ in STATUSES:
                if s_ not in p.columns:
                    p[s_] = 0
            for c in ("quartile_group", "decl_band"):
                if c not in p.columns:
                    p[c] = "all"
            p["year"] = y
            out.append(p[["year", "quartile_group", "decl_band"] + STATUSES])
        # The three lower quartiles pooled, which is the comparison the
        # contrast is read against, so the paper does not have to add up
        # three rows of a floored table to get it.
        low = base[base["quartile_group"].isin(["Q1", "Q2", "Q3"])]
        for k in (["decl_band"], []):
            if low.empty:
                break
            if k:
                p = (low.groupby(k + ["status"], observed=True)
                     ["n_personmonths"].sum()
                     .unstack("status", fill_value=0).reset_index())
            else:
                p = (low.groupby("status", observed=True)["n_personmonths"]
                     .sum().to_frame().T.reset_index(drop=True))
            for s_ in STATUSES:
                if s_ not in p.columns:
                    p[s_] = 0
            if "decl_band" not in p.columns:
                p["decl_band"] = "all"
            p["quartile_group"] = "Q1-Q3"
            p["year"] = y
            out.append(p[["year", "quartile_group", "decl_band"] + STATUSES])
    tab = pd.concat(out, ignore_index=True)
    tab = tab.drop_duplicates(subset=["year", "quartile_group", "decl_band"])
    tab = tab.rename(columns={s_: f"n_{s_}" for s_ in STATUSES})
    ncols = [f"n_{s_}" for s_ in STATUSES]
    tab[ncols] = tab[ncols].fillna(0).astype("int64")
    tab["n_total"] = tab[ncols].sum(axis=1)
    tab["share_no_register"] = tab["n_no_register"] / tab["n_total"].replace(0, np.nan)
    tab["share_not_counted"] = ((tab["n_total"] - tab["n_counted"])
                                / tab["n_total"].replace(0, np.nan))
    tab = tab[["year", "quartile_group", "decl_band", "n_total"] + ncols
              + ["share_no_register", "share_not_counted"]]
    return tab.sort_values(["year", "quartile_group", "decl_band"],
                           ignore_index=True)


def floor_table(tab: pd.DataFrame) -> pd.DataFrame:
    """
    The export floor, as everywhere in this tree: a count of one to four is
    suppressed, zero stays. A share is suppressed with its own numerator,
    since a share and a published total reproduce the count.
    """
    t = tab.copy()
    for c in ["n_total"] + [f"n_{s_}" for s_ in STATUSES]:
        t = mc.enforce_min_cell(t, count_col=c, floor=FLOOR)
    t.loc[t["n_no_register"].isna() | t["n_total"].isna(),
          "share_no_register"] = np.nan
    t.loc[t["n_counted"].isna() | t["n_total"].isna(),
          "share_not_counted"] = np.nan
    return t


def part_c(expo) -> tuple:
    """
    Counts, no fit. Reads the employer declarations year by year and the
    three individual registers, and writes uncounted_share.csv.
    """
    s73 = _mod("73_industry_and_credit.py", "s73_c")
    conn = mc.connect()
    try:
        age_col, age_kind, note = probe_declaration_age(conn, UNCOUNTED_YEARS)
        print(f"  {note}")
        NOTES.append(note)
        frames = uncounted_counts(UNCOUNTED_YEARS, age_col, age_kind, conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    tab = uncounted_table(frames, expo, s73)
    del frames
    gc.collect()
    out = floor_table(tab)
    out.to_csv(OUT / "uncounted_share.csv", index=False)
    return out, {"age_col": age_col, "age_kind": age_kind, "note": note}


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  A. The coefficients must reproduce the employer-clustered run to four",
    "     decimals. If any term fails, nothing from Part A is quoted, because",
    "     clustering changes the covariance and nothing else, so a moved",
    "     coefficient means a moved panel. The female differential and the",
    "     women's step are reported under both clusterings whatever they show.",
    "  B. No gate. The retained share of the adoption step against the",
    "     same-sample baseline is reported whatever it is.",
    "  C. The paper reports the figure whatever it shows. A share of uncounted",
    "     payslips that is higher in exposed firms at young ages is the",
    "     editor's mechanism and is stated plainly if it appears. Counts below",
    f"     {FLOOR} are suppressed before anything leaves MONA.",
]


def fmt(x):
    return "(no SE)" if x is None else f"({x:.4f})"


def pct(x):
    return "     n/a" if x != x else f"{x:7.3%}"


def c_summary_lines(tab: pd.DataFrame, info: dict) -> list:
    """
    What Part C found, said plainly.

    Three readings, in the order the paper needs them: how many payslips
    the panel never counts in the new years, whether that is new, and
    whether it falls differently on exposed employers and on the young.
    """
    L = [info["note"].replace("C: ", "  "), ""]
    allb = tab[tab["decl_band"] == "all"]
    tot = allb[allb["quartile_group"] == "all"].set_index("year")
    L.append("  person-months in the employer declarations, and the share whose")
    L.append("  worker is in none of the three individual registers:")
    for y in sorted(tot.index):
        r = tot.loc[y]
        n = r["n_total"]
        L.append(f"    {y}  {('%15s' % f'{n:,.0f}') if n == n else '(suppressed)'} "
                 f"person-months   no register {pct(r['share_no_register'])}"
                 f"   not counted at all {pct(r['share_not_counted'])}")
    old = [y for y in sorted(tot.index) if y not in NEW_YEARS]
    if old and any(y in tot.index for y in NEW_YEARS):
        o = float(np.nanmean([tot.loc[y, "share_no_register"] for y in old]))
        n_ = float(np.nanmean([tot.loc[y, "share_no_register"] for y in NEW_YEARS
                               if y in tot.index]))
        L.append(f"    the mean share over {old[0]} to {old[-1]} is {o:.3%} and over "
                 f"{'/'.join(str(y) for y in NEW_YEARS)} it is {n_:.3%}, "
                 f"{'higher' if n_ > o else 'no higher'} in the new years, which is "
                 f"what a register that ends in 2023 implies.")
    L.append("")
    L.append("  the same share by 2019 exposure quartile (Q4 is the treated group):")
    for y in sorted(tot.index):
        row = []
        for ggroup in ("Q4", "Q1-Q3", "unscored"):
            s = allb[(allb["year"] == y) & (allb["quartile_group"] == ggroup)]
            v = float(s["share_no_register"].iloc[0]) if len(s) else float("nan")
            row.append(f"{ggroup} {pct(v)}")
        L.append(f"    {y}  " + "   ".join(row))
    bands = sorted(set(tab["decl_band"]) - {"all"})
    if bands:
        L.append("")
        L.append("  and by the band the declaration itself puts the worker in, "
                 "new years only:")
        for y in [y for y in NEW_YEARS if y in set(tab["year"])]:
            for bnd in bands:
                q4 = tab[(tab["year"] == y) & (tab["quartile_group"] == "Q4")
                         & (tab["decl_band"] == bnd)]
                lo = tab[(tab["year"] == y) & (tab["quartile_group"] == "Q1-Q3")
                         & (tab["decl_band"] == bnd)]
                if not len(q4) or not len(lo):
                    continue
                a = float(q4["share_no_register"].iloc[0])
                b_ = float(lo["share_no_register"].iloc[0])
                gap = a - b_
                L.append(f"    {y} {bnd:<9} Q4 {pct(a)}   Q1-Q3 {pct(b_)}   "
                         f"gap {100*gap:+7.3f} pp")
    # The plain statement the read rule asks for.
    worst = None
    for y in [y for y in NEW_YEARS if y in set(tab["year"])]:
        for bnd in (bands or ["all"]):
            q4 = tab[(tab["year"] == y) & (tab["quartile_group"] == "Q4")
                     & (tab["decl_band"] == bnd)]
            lo = tab[(tab["year"] == y) & (tab["quartile_group"] == "Q1-Q3")
                     & (tab["decl_band"] == bnd)]
            if not len(q4) or not len(lo):
                continue
            gap = (float(q4["share_no_register"].iloc[0])
                   - float(lo["share_no_register"].iloc[0]))
            if gap == gap and (worst is None or gap > worst[0]):
                worst = (gap, y, bnd)
    L.append("")
    if worst is None:
        L.append("  No Q4 against Q1-Q3 comparison survived the export floor, so "
                 "the contrast is not stated.")
    elif worst[0] > 0:
        L.append(f"  STATED PLAINLY: the uncounted share is HIGHER in the top "
                 f"exposure quartile than in the lower three, by at most "
                 f"{100*worst[0]:.3f} percentage points ({worst[1]}, band "
                 f"{worst[2]}). Uncounted "
                 f"payslips in exposed firms are a mechanism that would show up "
                 f"in our counts as a fall, and the paper says so and gives the "
                 f"size, whether or not it is large.")
    else:
        L.append(f"  STATED PLAINLY: the uncounted share is NOT higher in the top "
                 f"exposure quartile in any year or band (the largest difference "
                 f"is {100*worst[0]:+.3f} percentage points, {worst[1]}, band "
                 f"{worst[2]}), so the "
                 f"editor's measurement mechanism does not run in the direction "
                 f"it would need to.")
    L.append("  The band is what the declaration itself says, not the register, "
             "since a worker with no register row has no register age.")
    return L


def main():
    mc.Tee(OUT / "79_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"79: THE LAST THREE GAPS   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s61, s67, s78, j47, h47 = load_modules()
    book, spec = score_book(h47)
    frame19 = frame_2019(h47)
    expo = s78.exposure(frame19, book, spec, j47, s61)
    del frame19
    gc.collect()
    print(f"  exposure: {len(expo):,} firms")

    counts = None
    if "B" in PARTS:
        counts = load_counts("L_counts", s61.PANEL_YEARS)
        if counts is None:
            raise RuntimeError("L_counts_* missing: run 47L first.")

    gender_rows, gsum, ind_rows = [], {}, []
    ctab, cinfo = None, {}
    if "A" in PARTS:
        r = opt("Part A", part_a, expo, s61, s67, s78, j47)
        if r:
            gender_rows, gsum = r
    if "B" in PARTS:
        r = opt("Part B", part_b, counts, expo, s61, s78, j47)
        if r:
            ind_rows = r
    del counts
    gc.collect()
    if "C" in PARTS:
        r = opt("Part C", part_c, expo)
        if r:
            ctab, cinfo = r

    # ---- summary ------------------------------------------------------
    L = ["THE LAST THREE GAPS", "=" * 52, "",
         "Employer x age x month counts, exposure the top quartile of the",
         "2019 education mix, employer-by-month, employer-by-age and",
         "month-by-age effects, Poisson, calendar cycle removed. Part A",
         "changes the cluster, Part B the effects, Part C fits nothing.", ""]
    if "A" in PARTS:
        L += ["A. THE SEXES, CLUSTERED BY THREE-DIGIT INDUSTRY (22-25 stock,",
              "   every term of Equation (2) interacted with female):"]
        if gender_rows:
            G = pd.DataFrame(gender_rows).set_index("term")
            L.append(f"  {gsum.get('n_firms', 0):,} employers, "
                     f"{gsum.get('n_clusters', 0):,} industry clusters")
            L.append(f"  coefficients equal the employer-clustered run to "
                     f"{MATCH_DP} decimals: "
                     f"{'YES' if gsum.get('reproduces') else 'NO'}")
            if not gsum.get("reproduces"):
                L.append("  THE RULE SAYS NOTHING FROM PART A IS QUOTED.")
            L.append(f"  {'term':<34} {'coef':>9}  {'employer SE':>12}  "
                     f"{'industry SE':>12}")
            for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                       "post_x_high_x_young", "post_x_high_x_female",
                       "post_x_high_x_young_x_female"):
                if t_ in G.index:
                    r_ = G.loc[t_]
                    L.append(f"  {t_:<34} {r_['coef']:+9.4f}  "
                             f"{r_['se_employer']:12.4f}  {r_['se_industry']:12.4f}")
            for k, nm in (("male_step", "young men, adoption step"),
                          ("female_differential", "young women minus young men"),
                          ("female_step", "young women, adoption step"),
                          ("male_step_from_2023", "young men, step from 2023"),
                          ("female_step_from_2023", "young women, step from 2023")):
                if k in gsum:
                    c, se_e, se_i = gsum[k]
                    L.append(f"  {nm:<34} {c:+.4f}  employer {fmt(se_e)}  "
                             f"industry {fmt(se_i)}")
            L += ["  The women's step is the male step plus the differential and",
                  "  its standard error comes from the covariance of each fit, so",
                  "  it is not the sum of two standard errors."]
        else:
            L.append("  no fit came back")
        L.append("")
    if "B" in PARTS:
        L += [f"B. INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS, "
              f"{BAND_B} stock:"]
        if ind_rows:
            I = pd.DataFrame(ind_rows)
            for spec_ in ("baseline_same_sample", "industry_age_month"):
                g = I[I["spec"] == spec_].set_index("term")
                for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                           "post_x_high_x_young"):
                    if t_ in g.index:
                        L.append(f"  {spec_:<22} {t_:<26} "
                                 f"{float(g.loc[t_,'coef']):+.4f} "
                                 f"({float(g.loc[t_,'se']):.4f})")
            gi = I[I["spec"] == "industry_age_month"]
            if len(gi):
                sh = float(gi["retained_share"].iloc[0])
                L.append(f"  employers in both fits: {int(I['n_firms'].iloc[0]):,}")
                L.append(f"  retained share of the adoption step against the "
                         f"same-sample baseline: "
                         f"{'n/a' if sh != sh else format(sh, '.0%')}")
                L.append("  Read beside 78's Part F, which is the same fit at 22-25.")
        else:
            L.append("  no fit came back")
        L.append("")
    if "C" in PARTS:
        L += ["C. THE PAYSLIPS THE PANEL NEVER COUNTS:"]
        if ctab is not None and len(ctab):
            L += c_summary_lines(ctab, cinfo)
        else:
            L.append("  no counts came back")
        L.append("")
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += ["WHAT FAILED: " + "; ".join(FAILURES),
              "A missing row is a missing fit or a missing pull, never a zero.",
              ""]
    L += READ_RULES + [
        "",
        "AND WHEN QUOTING:",
        "  1. A is inference only. The coefficients are 78's Part B, and the",
        "     paper quotes the industry-clustered standard errors beside the",
        "     employer-clustered ones rather than in place of them.",
        "  2. B is read beside 78's Part F at 22-25; the two bands answer the",
        "     same question and the paper reports both retained shares.",
        "  3. C is a count, not an estimate. It bounds a measurement channel",
        "     and does not test it.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "79_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("79_last_gaps", 0, (time.time() - t0) / 60)
    print("\n79 done.")


if __name__ == "__main__":
    main()

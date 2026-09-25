#!/usr/bin/env python3
"""
95_pension_reference.py -- the age profile with the oldest band split at
                           60 and 65, and the headline contrast against
                           older references that stop before the ages the
                           pension reforms reach.

======================================================================
  RUNS IN MONA (lane 37a). Output folder CANARIES_95_OUT (default
  output_95); parts with CANARIES_95_PARTS (default RP). ONE new SQL
  pull: 47L's counts query with eight age bands, 2021 to 2025, cached as
  L_counts_age8_YYYY (about 70 seconds a year). Everything else is a
  cache: script 82's occupation cascade and 2019 counts for the score.
======================================================================

QUESTION
An external review (25 Sep 2026) made two points about the older side of
the headline contrast. First, the oldest band's gain after adoption was
defended as not being a pension-age effect because "only 65-69" is reached
by the 2020 and 2023 pension reforms; that is wrong, since the earliest
withdrawal age of the income pension rose from 61 to 62 in 2020 and from
62 to 63 in 2023 (the review's facts), so 60-64 is reached too. Second,
the headline tau = b_L - b_I compares the young with ALL colleagues aged 31
to 69, so a pension-driven rise at 60 and over would make the young look
worse relative to a reference that is itself moving. This script measures
both instead of arguing them.

THE OBJECT
tau = b_L - b_I: Equation (2)'s adoption step (from January 2024) minus
its interim window (December 2022 to December 2023), each x High x Young,
Poisson with employer-by-month, employer-by-age and month-by-age effects,
the tightening switch and three calendar-quarter terms, clustered by
employer; exposure the top quartile of the employer's 2019 occupation mix
(script 82's build_exposure, the reported arm). The standard error of tau
is from the clustered covariance of the same fit: Var = V_LL + V_II -
2 V_LI, and the three components are exported beside it.

THE GATE (hard stop, before anything else)
The eight-band pull is collapsed to the paper's six bands and must
reproduce Table 1 within 0.0005 on coefficient AND standard error:
22-25 adoption step -0.0578 (0.0155) and tau -0.0399 (0.0102); 26-30
-0.0482 (0.0104) and -0.0403 (0.0067). A miss stops the script and the
summary prints the arithmetic. Before the fit, the collapsed pull is
compared cell by cell with 47L's L_counts caches, which the gate's panel
has always been built from; the count of cells that differ is reported
(a diagnostic: the fit gate decides).

PART R. THE REFERENCE RESTRICTED (both young bands)
tau with the older reference restricted to 31-59 (31-34, 35-40, 41-49,
50-59) and to 31-49 (31-34, 35-40, 41-49). Each is fitted on its own
natural sample AND on the common sample E49, the employers the 31-49
panel keeps (every employer there is in the other two panels), where the
headline reference 31-69 and 31-59 are refitted, so that the three
references differ in the reference and in nothing else.

PART P. THE PROFILE WITH THE OLDEST BAND SPLIT (eight bands, one sample)
All bands against 41-49 on one skeleton: 22-25, 26-30, 31-34, 35-40,
41-49 (reference), 50-59, 60-64, 65-69; the employer must hold 22-25 and
at least one other band, which is the panel of the existing split-at-65
table (tab:profile_bands, 104,333 employers).
  (P7) the seven-band profile with 50-64 and 65-69, 74's seasonal terms
       (adoption, tightening and three calendar terms per band, no
       interim): must reproduce lane 31's occ_route_split65.csv within
       0.0005 and on the same employer count, or Part P stops (a moved
       panel). This proves the eight-band pull and the term builder.
  (P8g) the same terms with 50-59, 60-64 and 65-69: the gamma_2-style
       step comparable with the existing table.
  (P8t) the same eight bands with an interim term per band added, so
       every band gets its own tau = adoption minus interim, with the
       standard error from the covariance.
All three fits run on the same employers, and the script asserts it.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  R1. THE HEADLINE DOES NOT REST ON THE PENSION AGES if, on the common
      sample E49 at 22-25, tau against 31-59 AND against 31-49 are each
      negative and distinguishable from zero at five per cent. Otherwise
      the summary says which reference loses it. The share of the 31-69
      tau that each keeps on E49 is printed either way.
  P1. THE OLDER GAIN IS NOT A PENSION-AGE EFFECT if 50-59, which neither
      reform reaches, has a positive tau distinguishable from zero at five
      per cent. If the gain is confined to 60-64 and 65-69 (50-59 not
      distinguishable from zero), pension ages are a live explanation of
      the older gain, and the paper must say so.
  Every other cell is reported and settles nothing. A missing row is a
  missing fit, never a zero.

EXPORT (output_95/)
  pension_reference.csv  every coefficient the parts report, the tau rows
                         with var_post, var_interim and cov_post_interim,
                         n_obs and n_firms (employer counts below five
                         suppressed together with the statistic)
  95_summary.txt         the gate arithmetic, the tables, the read rules
  95_log.txt             the full log
  vcov_s95_*.csv         clustered covariances; stay on the share

IN THE PAPER
Online Appendix tab:profile_bands (the split at 65 becomes 50-59, 60-64
and 65-69, on tau and on the gamma_2-style step); the OA sentence and
the response-letter paragraph on pension ages (rebuttal replaced by the
estimate); Table 1's note or OA III.2 for the 31-59 and 31-49 references.

    python 95_pension_reference.py
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402

OUT = HERE / os.environ.get("CANARIES_95_OUT", "output_95")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_95_PARTS", "RP").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
SIG5 = 1.959963984540054
BANDS = ["22-25", "26-30"]
PROFILE_REF = "41-49"
FINE_PREFIX = "L_counts_age8"
FINE_BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-59",
              "60-64", "65-69"]
TO_SIX = {"50-59": "50+", "60-64": "50+", "65-69": "50+"}
TO_SEVEN = {"50-59": "50-64", "60-64": "50-64"}
SEVEN_BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50-64", "65-69"]
REFS = {"31-69": ["31-34", "35-40", "41-49", "50+"],
        "31-59": ["31-34", "35-40", "41-49", "50-59"],
        "31-49": ["31-34", "35-40", "41-49"]}
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]

# Table 1, export 2026-09-23_0655 (occ_route_headline.csv, vcov files).
GATE = {"22-25": {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)},
        "26-30": {"post": (-0.0482, 0.0104), "tau": (-0.0403, 0.0067)}}
GATE_TOL = 0.0005
# Lane 31's occ_route_split65.csv (round3_20260923-1125-lane31), which the
# seven-band fit must reproduce: band -> (coef, se); and its employers.
SPLIT65 = {"22-25": (-0.0217, 0.0124), "26-30": (-0.0105, 0.0087),
           "31-34": (+0.0167, 0.0067), "35-40": (+0.0184, 0.0056),
           "50-64": (+0.0392, 0.0039), "65-69": (+0.1902, 0.0321)}
SPLIT65_FIRMS = 104_333

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"

AGE_CASE8 = """CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 59 THEN '50-59'
             WHEN age BETWEEN 60 AND 64 THEN '60-64'
             WHEN age BETWEEN 65 AND 69 THEN '65-69'
             ELSE NULL END"""

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. The eight-band pull, collapsed to six bands, must reproduce",
    "  Table 1 within 0.0005 (coef and SE): 22-25 -0.0578 (0.0155), tau",
    "  -0.0399 (0.0102); 26-30 -0.0482 (0.0104), tau -0.0403 (0.0067).",
    "  A miss stops the script; nothing from it is quoted.",
    "  R1. The headline does not rest on the pension ages if, on the",
    "  common sample E49 at 22-25, tau against 31-59 AND against 31-49 are",
    "  each negative and distinguishable from zero at five per cent.",
    "  P1. The older gain is not a pension-age effect if 50-59 has a",
    "  positive tau distinguishable from zero at five per cent; if the",
    "  gain sits only in 60-64 and 65-69, pension ages are a live",
    "  explanation and the paper must say so.",
    "  Part P runs only if the seven-band profile reproduces lane 31's",
    "  split at 65 within 0.0005 on 104,333 employers.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


# ----------------------------------------------------------------------
# plumbing
# ----------------------------------------------------------------------

def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    for attr in ("NOTES", "FAILURES"):
        if hasattr(mod, attr):
            getattr(mod, attr).clear()


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s78.OUT, s78.CACHE = OUT, CACHE
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    if s78.POST_FROM != "2024-01":
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}")
    if list(j47.INCUMBENT_BANDS) != REFS["31-69"]:
        raise RuntimeError(f"47j's incumbent bands are {j47.INCUMBENT_BANDS}"
                           f", not {REFS['31-69']}; the gate's reference "
                           f"would not be the paper's")
    return s82, s61, s78, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def add(part, spec, sample, band, reference, term, coef, se, n_obs, n_firms,
        status="ok", vp=np.nan, vi=np.nan, cpi=np.nan):
    ROWS.append({"part": part, "spec": spec, "sample": sample,
                 "young_band": band, "reference": reference, "term": term,
                 "coef": float(coef) if coef == coef else np.nan,
                 "se": float(se) if se is not None and se == se else np.nan,
                 "t": tstat(coef, se), "var_post": vp, "var_interim": vi,
                 "cov_post_interim": cpi, "n_obs": n_obs,
                 "n_firms": n_firms, "status": status})


def save() -> pd.DataFrame:
    """The one export; the employer floor applies to every row."""
    df = pd.DataFrame(ROWS)
    if df.empty:
        df.to_csv(OUT / "pension_reference.csv", index=False)
        return df
    had = df["n_firms"].notna()
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = had & df["n_firms"].isna()
    if small.any():
        df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                       "cov_post_interim"]] = np.nan
    df.to_csv(OUT / "pension_reference.csv", index=False)
    return df


def get(part, spec, sample, band, reference, term):
    for r in ROWS:
        if (r["part"], r["spec"], r["sample"], r["young_band"],
                r["reference"], r["term"]) == (part, spec, sample, band,
                                               reference, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    """One Poisson fit; (coefficients by term, clustered vcov or None). A
    failure returns (None, None) and is recorded: a missing row is a
    missing fit. R's own stderr is written in full beside the log by
    mona_common._r_failed, never truncated here."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s95_{tag}", terms=terms,
                                fes=fes, cluster="employer_id")
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
    missing = [x for x in terms if x not in g.index]
    if missing:
        NOTES.append(f"{tag}: {len(missing)} terms absent from the fit "
                     f"({', '.join(missing[:4])})")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def tau(g, v, post=POST, interim=INTERIM) -> tuple:
    """tau = post - interim with the delta-method SE from the clustered
    covariance: Var = V_pp + V_ii - 2 V_pi. Returns (tau, se, V_pp, V_ii,
    V_pi); NaN where a piece is missing (never a zero)."""
    nan = (np.nan,) * 5
    if g is None or post not in g.index or interim not in g.index:
        return nan
    c = float(g.loc[post, "coef"]) - float(g.loc[interim, "coef"])
    if v is None or post not in v.index or interim not in v.index:
        return c, np.nan, np.nan, np.nan, np.nan
    vp, vi = float(v.loc[post, post]), float(v.loc[interim, interim])
    cpi = float(v.loc[post, interim])
    var = vp + vi - 2.0 * cpi
    return c, (float(np.sqrt(var)) if var > 0 else np.nan), vp, vi, cpi


def record_eq2(g, v, part, spec, sample, band, reference, n_firms):
    """The tightening, interim and adoption terms and the derived tau."""
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for lab, t_ in (("rb", "rb_x_high_x_young"), ("interim", INTERIM),
                    ("post", POST)):
        if t_ in g.index:
            add(part, spec, sample, band, reference, lab, g.loc[t_, "coef"],
                g.loc[t_, "se"], n_obs, n_firms,
                str(g.loc[t_].get("status", "ok")))
    c, s, vp, vi, cpi = tau(g, v)
    add(part, spec, sample, band, reference, "tau", c, s, n_obs, n_firms,
        "derived", vp, vi, cpi)
    save()


# ----------------------------------------------------------------------
# the eight-band counts
# ----------------------------------------------------------------------

def q_counts_age8(year: int, conn) -> pd.DataFrame:
    """47L's counts query (and 78's q_counts_split) with the oldest band
    split into 50-59, 60-64 and 65-69. Birth year from whichever Individ
    vintage holds the person; age = calendar year minus birth year."""
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
           {AGE_CASE8} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, {AGE_CASE8}
    """
    return pd.read_sql(q, conn)


def load_age8(years) -> pd.DataFrame:
    """The eight-band counts, pulled once per year and cached; the
    connection is opened only if a year is missing, so a rerun needs no
    SQL. The band set of every year is probed before use."""
    out, conn = [], None
    for y in years:
        cf = CACHE / f"{FINE_PREFIX}_{y}.parquet"
        c = mc.read_cache(cf, require=COUNT_COLS)
        if c is None:
            if conn is None:
                print("  opening a connection for the eight-band pull")
                conn = mc.connect()
            t = time.time()
            c = q_counts_age8(y, conn)
            mc.write_cache(c, cf)
            print(f"  eight-band counts {y}: {len(c):,} cells "
                  f"({(time.time() - t) / 60:.1f} min)")
        else:
            print(f"  eight-band counts {y}: cached ({len(c):,} cells)")
        got = set(c["age_group"].astype(str).unique())
        if got != set(FINE_BANDS):
            raise RuntimeError(f"{cf.name}: bands {sorted(got)}, expected "
                               f"{FINE_BANDS}")
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    d = pd.concat(out, ignore_index=True)
    d["age_group"] = d["age_group"].astype(str)
    d["year_month"] = d["year_month"].astype(str)
    return d


def relabel(counts: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Collapse bands (e.g. 50-59, 60-64, 65-69 -> 50+) and re-sum the
    cells, so a collapsed band is exactly the sum of its parts."""
    c = counts.copy()
    c["age_group"] = c["age_group"].replace(mapping)
    return (c.groupby(["employer_id", "year_month", "age_group"],
                      observed=True)["n_emp"].sum().reset_index())


def compare_with_lcounts(six: pd.DataFrame, s82, s61) -> None:
    """The collapsed pull against 47L's L_counts, cell by cell, year by
    year. A diagnostic for the summary; the fit gate decides. Identifier
    dtype is normalised on BOTH sides (failure class 3)."""
    tot_bad, tot_cells = 0, 0
    for y in s61.PANEL_YEARS:
        old = mc.read_cache(CACHE / f"L_counts_{y}.parquet", require=COUNT_COLS)
        if old is None:
            NOTES.append(f"data check: L_counts_{y} is not on the share; the "
                         f"comparison is skipped for {y}")
            continue
        new = six[six["year_month"].str.slice(0, 4) == str(y)]
        a = new.assign(employer_id=pd.to_numeric(new["employer_id"]).astype("int64"))
        b = old.assign(employer_id=pd.to_numeric(old["employer_id"]).astype("int64"),
                       year_month=old["year_month"].astype(str),
                       age_group=old["age_group"].astype(str))
        k = ["employer_id", "year_month", "age_group"]
        j = a.groupby(k)["n_emp"].sum().rename("new").to_frame().join(
            b.groupby(k)["n_emp"].sum().rename("old"), how="outer").fillna(0)
        bad = int((j["new"] != j["old"]).sum())
        tot_bad += bad
        tot_cells += len(j)
        print(f"  data check {y}: {len(j):,} cells, {bad:,} differ from "
              f"L_counts_{y}")
        del a, b, j
        gc.collect()
    msg = (f"data check: the eight-band pull collapsed to six bands differs "
           f"from 47L's L_counts in {tot_bad:,} of {tot_cells:,} "
           f"employer-band-month cells")
    NOTES.append(msg)
    print(f"  {msg}")


# ----------------------------------------------------------------------
# skeletons and terms
# ----------------------------------------------------------------------

def restrict(b: pd.DataFrame, emp: set) -> pd.DataFrame:
    """A built panel cut to a set of employers. The integer fixed-effect
    keys stay valid: they are unique per employer, month and band."""
    return b[b["employer_id"].isin(emp)].copy()


def profile_terms(b: pd.DataFrame, bands: list, interim: bool) -> tuple:
    """74's seasonal profile terms for any band list (78's profile_terms
    generalised): per band except 41-49, the adoption step from January
    2024, the cumulative tightening switch from April 2022 and three
    calendar-quarter terms, each x High x that band; with `interim`, also
    the interim window (December 2022 to December 2023), which makes each
    band's adoption term a step from the tightening level and gives every
    band its own tau."""
    ym = b["year_month"].astype(str)
    post = (ym >= "2024-01").astype(int)
    rb = (ym >= mc.RIKSBANK_YM).astype(int)
    itm = ((ym >= mc.CHATGPT_YM) & (ym < "2024-01")).astype(int)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    terms = []
    for band in bands:
        if band == PROFILE_REF:
            continue
        d = (b["age_group"] == band).astype(int) * b["high"]
        tag = band.replace("-", "_").replace("+", "plus")
        b[f"gpt_x_high_{tag}"] = post * d
        b[f"rb_x_high_{tag}"] = rb * d
        terms += [f"gpt_x_high_{tag}", f"rb_x_high_{tag}"]
        for qq in (1, 2, 3):
            b[f"q{qq}_x_high_{tag}"] = (q == qq).astype(int) * d
            terms.append(f"q{qq}_x_high_{tag}")
        if interim:
            b[f"int_x_high_{tag}"] = itm * d
            terms.append(f"int_x_high_{tag}")
    return b, terms


def check_gate(band: str) -> None:
    """The Table 1 gate; stops the script on a miss and prints why."""
    bad = []
    for key, term in (("post", "post"), ("tau", "tau")):
        c, s = get("G", "gate", "full", band, "31-69", term)
        wc, ws = GATE[band][key]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"{band} {key}: this run {c:+.4f} ({s:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f}), |diff| {abs(c - wc):.4f} and "
                       f"{abs(s - ws):.4f} against {GATE_TOL}")
    if bad:
        msg = "THE GATE FAILED. Nothing from this run is quotable. " + \
              "; ".join(bad)
        print(f"\n  {msg}")
        FAILURES.append(msg)
        write_summary()
        raise SystemExit("95: the gate failed; stopping before any other fit.")
    c, s = get("G", "gate", "full", band, "31-69", "tau")
    print(f"  THE GATE PASSES at {band}: tau {c:+.4f} ({s:.4f}), Table 1 "
          f"{GATE[band]['tau'][0]:+.4f} ({GATE[band]['tau'][1]:.4f})")


# ----------------------------------------------------------------------
# the gate and Part R
# ----------------------------------------------------------------------

def gate_and_r(six, age8, expo, s61, s78, j47) -> None:
    for band in BANDS:
        tag = band.replace("-", "_")
        print(f"\n  GATE at {band} (the paper's panel, reference 31-69):")
        skel = s61.build_skeleton(six, band, j47)
        b69 = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b69.empty:
            raise RuntimeError(f"{band}: the gate's panel is empty")
        b69, terms = s78.eq2_terms(b69)
        g, v = fit(b69, f"gate_{tag}", terms, j47.FES)
        record_eq2(g, v, "G", "gate", "full", band, "31-69",
                   int(b69["employer_id"].nunique()))
        check_gate(band)
        if "R" not in PARTS:
            del b69
            gc.collect()
            continue
        print(f"\n  PART R at {band}:")
        # One reference panel at a time beside the headline's, so Python
        # never holds three panels while R fits (failure class 4). 31-49
        # first: its employers define the common sample E49.
        e49 = None
        for ref in ("31-49", "31-59"):
            sk = s78.build_skeleton_bands(age8, [band] + REFS[ref], band, j47,
                                          s61.PANEL_FROM)
            bb = s78.with_exposure(sk, expo)
            del sk
            gc.collect()
            if bb.empty:
                FAILURES.append(f"R/{band}/{ref}/empty")
                continue
            bb, terms = s78.eq2_terms(bb)
            n = int(bb["employer_id"].nunique())
            g, v = fit(bb, f"r{ref[-2:]}_own_{tag}", terms, j47.FES)
            record_eq2(g, v, "R", f"ref_{ref}", "own", band, ref, n)
            if ref == "31-49":
                e49 = set(bb["employer_id"].unique())
                NOTES.append(f"R/{band}: E49 keeps {len(e49):,} of the "
                             f"{b69['employer_id'].nunique():,} employers of "
                             f"the headline panel")
                # on E49 the 31-49 fit IS the common-sample fit
                record_eq2(g, v, "R", f"ref_{ref}", "E49", band, ref, n)
            elif e49 is not None:
                bc = restrict(bb, e49)
                g, v = fit(bc, f"r{ref[-2:]}_e49_{tag}", terms, j47.FES)
                record_eq2(g, v, "R", f"ref_{ref}", "E49", band, ref,
                           int(bc["employer_id"].nunique()))
                del bc
            del bb
            gc.collect()
        if e49 is not None:
            bc = restrict(b69, e49)
            g, v = fit(bc, f"r69_e49_{tag}", terms, j47.FES)
            record_eq2(g, v, "R", "ref_31-69", "E49", band, "31-69",
                       int(bc["employer_id"].nunique()))
            del bc
        save()
        del b69
        gc.collect()


# ----------------------------------------------------------------------
# Part P
# ----------------------------------------------------------------------

def part_p(age8, expo, s61, s78, j47) -> None:
    print("\n  PART P, the profile on one sample:")
    seven = relabel(age8, TO_SEVEN)
    sk7 = s78.build_skeleton_bands(seven, SEVEN_BANDS, "22-25", j47,
                                   s61.PANEL_FROM)
    del seven
    gc.collect()
    b7 = s78.with_exposure(sk7, expo)
    del sk7
    gc.collect()
    n7 = int(b7["employer_id"].nunique())
    b7, t7 = profile_terms(b7, SEVEN_BANDS, interim=False)
    g, v = fit(b7, "p7_gamma", t7, j47.FES)
    emp7 = set(b7["employer_id"].unique())
    del b7
    gc.collect()
    bad = []
    if g is None:
        bad.append("the seven-band fit did not come back")
    else:
        n_obs = int(g["n_obs"].max())
        add("P", "p7_gamma", "profile", "all", PROFILE_REF, "reference",
            0.0, 0.0, n_obs, n7, "reference")
        for band, (wc, ws) in SPLIT65.items():
            t_ = f"gpt_x_high_{band.replace('-', '_')}"
            c = float(g.loc[t_, "coef"]) if t_ in g.index else np.nan
            s = float(g.loc[t_, "se"]) if t_ in g.index else np.nan
            add("P", "p7_gamma", "profile", band, PROFILE_REF, "gamma2_style",
                c, s, n_obs, n7)
            if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
                bad.append(f"{band}: {c:+.4f} ({s:.4f}) against lane 31's "
                           f"{wc:+.4f} ({ws:.4f})")
        if n7 != SPLIT65_FIRMS:
            bad.append(f"{n7:,} employers against lane 31's "
                       f"{SPLIT65_FIRMS:,}")
    save()
    if bad:
        msg = ("P/the seven-band profile does NOT reproduce lane 31's split "
               "at 65, so the eight-band fits were not run: " + "; ".join(bad))
        FAILURES.append(msg)
        print(f"  {msg}")
        return
    print("  the seven-band profile reproduces lane 31's split at 65")
    sk8 = s78.build_skeleton_bands(age8, FINE_BANDS, "22-25", j47,
                                   s61.PANEL_FROM)
    b8 = s78.with_exposure(sk8, expo)
    del sk8
    gc.collect()
    n8 = int(b8["employer_id"].nunique())
    if set(b8["employer_id"].unique()) != emp7:
        FAILURES.append(f"P/the eight-band panel holds {n8:,} employers and "
                        f"the seven-band {n7:,}: not one sample")
        return
    NOTES.append(f"P: the seven- and eight-band fits run on the same "
                 f"{n8:,} employers")
    b8, t8 = profile_terms(b8, FINE_BANDS, interim=False)
    g, _ = fit(b8, "p8_gamma", t8, j47.FES)
    b8 = b8.drop(columns=t8)
    if g is not None:
        n_obs = int(g["n_obs"].max())
        for band in FINE_BANDS:
            if band == PROFILE_REF:
                continue
            t_ = f"gpt_x_high_{band.replace('-', '_')}"
            if t_ in g.index:
                add("P", "p8_gamma", "profile", band, PROFILE_REF,
                    "gamma2_style", g.loc[t_, "coef"], g.loc[t_, "se"],
                    n_obs, n8)
        save()
    b8, t8 = profile_terms(b8, FINE_BANDS, interim=True)
    g, v = fit(b8, "p8_tau", t8, j47.FES)
    del b8
    gc.collect()
    if g is None:
        return
    n_obs = int(g["n_obs"].max())
    for band in FINE_BANDS:
        if band == PROFILE_REF:
            continue
        tg = band.replace("-", "_")
        p_, i_ = f"gpt_x_high_{tg}", f"int_x_high_{tg}"
        for lab, t_ in (("rb", f"rb_x_high_{tg}"), ("interim", i_),
                        ("post", p_)):
            if t_ in g.index:
                add("P", "p8_tau", "profile", band, PROFILE_REF, lab,
                    g.loc[t_, "coef"], g.loc[t_, "se"], n_obs, n8)
        c, s, vp, vi, cpi = tau(g, v, p_, i_)
        add("P", "p8_tau", "profile", band, PROFILE_REF, "tau", c, s, n_obs,
            n8, "derived", vp, vi, cpi)
    save()


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def sig(c, s, sign) -> bool:
    return bool(s == s and s > 0 and sign * c > 0 and abs(c) >= SIG5 * s)


def verdicts() -> list:
    L = []
    t69, s69 = get("R", "ref_31-69", "E49", "22-25", "31-69", "tau")
    t59, s59 = get("R", "ref_31-59", "E49", "22-25", "31-59", "tau")
    t49, s49 = get("R", "ref_31-49", "E49", "22-25", "31-49", "tau")
    if t59 == t59 and t49 == t49:
        ok = sig(t59, s59, -1) and sig(t49, s49, -1)
        L.append(f"  R1: {'THE HEADLINE DOES NOT REST ON THE PENSION AGES' if ok else 'R1 NOT MET: see which reference loses it below'}")
        for nm, c, s in (("31-59", t59, s59), ("31-49", t49, s49)):
            share = f"{c / t69:.0%} of the 31-69 tau" if t69 == t69 and t69 else ""
            L.append(f"      E49 22-25 against {nm}: {c:+.4f} ({s:.4f}) t "
                     f"{tstat(c, s):+.2f}  {share}")
    else:
        L.append("  R1: NO VERDICT, a reference fit is missing (not a null)")
    c5, s5 = get("P", "p8_tau", "profile", "50-59", PROFILE_REF, "tau")
    if c5 == c5:
        ok = sig(c5, s5, +1)
        L.append(f"  P1: {'THE OLDER GAIN IS NOT A PENSION-AGE EFFECT' if ok else 'P1 NOT MET: the older gain is not established at 50-59; pension ages are a live explanation'}")
        L.append(f"      50-59 tau {c5:+.4f} ({s5:.4f}) t {tstat(c5, s5):+.2f}")
    else:
        L.append("  P1: NO VERDICT, the eight-band tau fit is missing")
    return L


def write_summary() -> None:
    L = ["PENSION AGES: THE PROFILE SPLIT AT 60 AND 65, AND THE REFERENCE",
         "RESTRICTED TO 31-59 AND 31-49", "=" * 64, "",
         "tau = adoption step minus interim (b_L - b_I), SE from the",
         "clustered covariance of the same fit (V_LL + V_II - 2 V_LI).", ""]
    L += ["GATE (the paper's six bands, from the eight-band pull):"]
    for band in BANDS:
        c, s = get("G", "gate", "full", band, "31-69", "tau")
        p, ps = get("G", "gate", "full", band, "31-69", "post")
        if c == c:
            L.append(f"  {band}: tau {c:+.4f} ({s:.4f})   adoption "
                     f"{p:+.4f} ({ps:.4f})   Table 1 tau "
                     f"{GATE[band]['tau'][0]:+.4f} ({GATE[band]['tau'][1]:.4f})")
    L.append("")
    if any(r["part"] == "R" for r in ROWS):
        L += ["R. TAU WITH THE REFERENCE RESTRICTED (own sample | E49):"]
        for band in BANDS:
            for ref in ("31-69", "31-59", "31-49"):
                o = get("R", f"ref_{ref}", "own", band, ref, "tau")
                e = get("R", f"ref_{ref}", "E49", band, ref, "tau")
                if ref == "31-69":
                    o = get("G", "gate", "full", band, "31-69", "tau")
                n = next((r["n_firms"] for r in ROWS if r["part"] == "R"
                          and r["sample"] == "E49" and r["young_band"] == band
                          and r["reference"] == ref), -1)
                L.append(f"  {band} vs {ref}: own {o[0]:+.4f} ({o[1]:.4f})"
                         f"   E49 {e[0]:+.4f} ({e[1]:.4f})   E49 employers "
                         f"{int(n) if n == n else -1:,}")
        L.append("")
    if any(r["part"] == "P" for r in ROWS):
        L += ["P. THE PROFILE AGAINST 41-49, ONE SAMPLE (tau | gamma_2-style):"]
        for band in FINE_BANDS + ["50-64"]:
            if band == PROFILE_REF:
                continue
            c, s = get("P", "p8_tau", "profile", band, PROFILE_REF, "tau")
            gc_, gs = get("P", "p8_gamma", "profile", band, PROFILE_REF,
                          "gamma2_style")
            c7, s7 = get("P", "p7_gamma", "profile", band, PROFILE_REF,
                         "gamma2_style")
            if c == c or gc_ == gc_ or c7 == c7:
                L.append(f"  {band:<6} tau {c:+.4f} ({s:.4f})   gamma_2-style "
                         f"8 bands {gc_:+.4f} ({gs:.4f})   7 bands "
                         f"{c7:+.4f} ({s7:.4f})")
        L.append("")
    L += ["VERDICTS:"] + verdicts() + [""]
    L += [f"FITS: {DONE} of {PLANNED} attempted came back. A run far shorter "
          "than the estimate (2 to 3 hours) is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "95_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "95_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"95: PENSION AGES AND THE REFERENCE GROUP   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s78, l47, l70, j47 = load_modules()
        age8 = load_age8(s61.PANEL_YEARS)
        last = str(age8["year_month"].max())
        print(f"  eight-band counts: {len(age8):,} cells, "
              f"{age8['year_month'].min()} to {last}")
        if last < "2025-06" or str(age8["year_month"].min()) > s61.PANEL_FROM:
            raise RuntimeError(f"the counts span {age8['year_month'].min()} "
                               f"to {last}, not {s61.PANEL_FROM} to 2025-06")
        six = relabel(age8, TO_SIX)
        compare_with_lcounts(six, s82, s61)
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers")
        gate_and_r(six, age8, expo, s61, s78, j47)
        del six
        gc.collect()
        if "P" in PARTS:
            try:
                part_p(age8, expo, s61, s78, j47)
            except BaseException as ex:
                if isinstance(ex, SystemExit):
                    raise
                print(f"  Part P FAILED ({type(ex).__name__}: {ex})")
                traceback.print_exc()
                FAILURES.append(f"P/{type(ex).__name__}")
        drain(s78, "78")
    except SystemExit:
        mc.runlog("95_pension_reference", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"95 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("95_pension_reference", rc, (time.time() - T0) / 60)
    print("\n95 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

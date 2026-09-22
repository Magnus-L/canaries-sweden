#!/usr/bin/env python3
"""
80_industry_key.py -- the employer industry code completed, and the two
                      exercises that rest on it run again.

======================================================================
  RUNS IN MONA. Part A is SQL and counting only: up to nine reads of
  LISA's small firm tables, one of Serrano's firm table and one of the
  business register, and no fit. Parts B and C read the caches 47h, 47L
  and 67 wrote and fit Poisson models on the same panels as scripts 78
  and 79. Writes output_80/. Parts are chosen with the environment
  variable CANARIES_80_PARTS (default ABC) and the folder with
  CANARIES_80_OUT (default output_80); the lane runners set both.
  Budget: A about an hour, B three to four hours (the sex panel is the
  heavy one), C one and a half to two hours.
======================================================================

WHAT WENT WRONG, AND WHY IT MATTERS

Script 73 reads the employer's three-digit industry from LISA's firm
table for 2019 alone (Ftg_2019, Org_Sni2007). That table holds 552,099
coded firms and it is the right source: it is keyed on the same legal
entity our panel is built on, and the code is the one Statistics Sweden
itself attaches to the employer. But 5,285 of the 111,459 employers on
the 22 to 25 panel, and 9,106 of the 128,193 on the 26 to 30 panel, are
not in it.

Scripts 73, 78 and 79 then gave each unmatched firm a cluster of its
own. That is why the industry-clustered runs report 5,545 and 9,368
clusters against only 265 real three-digit groups: nineteen clusters in
twenty are a single firm. The reported inference is therefore neither
industry clustering nor employer clustering but a hybrid of the two, and
a referee who counts the clusters will see it before we do.

The likely mechanism is a reference-date mismatch rather than a bad
join. Ftg is LISA's firm table and LISA is built on a November reference
date, while our panel comes from the monthly employer declarations. A
firm that employed somebody in, say, March 2019 but nobody in November
is in our panel and absent from Ftg_2019. If that is the mechanism the
missing firms should be small and short-lived, and Part A measures
exactly that rather than asserting it.

A. THE KEY. A complete employer-to-industry key, built by cascade, with
   the coverage reported at every step and the first source that answers
   keeping the firm.
     1. Ftg_2019, Org_Sni2007. The preferred source, unchanged.
     2. Ftg_2018, then Ftg_2020, then outward: 2017, 2021, 2016, 2022,
        2015, 2023. Nearest year first, and the earlier of two equally
        near years first, because the industry must be frozen before the
        shock: a 2018 code cannot carry a firm's own response to it and
        a 2023 code might. Each table is three columns, keyed on
        P1207_LopNr_PeOrgNr and carrying the same Org_Sni2007, so these
        are small pulls.
     3. Serrano_Serrano_20230614, bransch_sni3, keyed on
        P1207_Lopnr_ORGNR, the key script 73 already uses for the credit
        test. The column is a FLOAT, so 011 arrives as 11.0 and the text
        route would file it as 110; it is cast and padded instead.
     4. FDB_JE_2014_2021, ng3, at reference year 2019. Last, because
        73's own docstring records that this delivery's extract is not
        the employer population and matches only a small share.
   Statistics Sweden writes "no code" in several ways and a plain null
   test catches only one of them, so the missing conventions are handled
   by name: null and empty, a run of asterisks ("****"), dots or dashes,
   an all-zero code, and the usual words. Each is counted and the counts
   go in the summary, because a convention read as a code would become
   an industry group of its own and pool unrelated firms.
   One consequence to expect rather than to debug: script 73 counted
   552,099 coded firms in Ftg_2019, and this script may count slightly
   fewer, because 73 strips the non-digits out of a code and an unclassified
   "00000" survives that as the group "000", which is not a group SNI 2007
   has. The difference is the conventions firing, and the counts in the
   summary say which and how many.
   Export: industry_key_coverage.csv, per panel, the number and share of
   employers resolved at each step, and the size distribution of the
   employers Ftg_2019 alone missed beside the ones it found.

B. THE CLUSTERING REDONE. The industry-clustered standard errors of
   script 78's Part C (the pooled stock fit at both bands) and script
   79's Part A (the full sex specification at 22 to 25) are computed
   again on the complete key, on the identical panels, with the
   identical terms and therefore the identical coefficients. Only the
   covariance moves, so the coefficients must reproduce, and that is the
   gate. An employer the cascade cannot resolve at all joins ONE
   residual group rather than becoming a cluster of one: the hybrid is
   what this script exists to remove, and the size of that group is
   reported so a reader can judge what is left of it.
   The cluster count falls from about 5,500 to about 266, which is the
   point rather than a cost: the earlier figure counted several thousand
   near-singletons, and 266 groups is comfortably above the range in
   which a cluster-robust variance stops being trustworthy. Expect the
   standard errors to RISE, because they were part employer-clustered
   before, and expect that to bear on the 22 to 25 step, which lane 25
   already could not distinguish from zero.
   Read rule, fixed before the run: the coefficients must reproduce the
   employer-clustered run to four decimals or nothing from this part is
   quoted. The new standard errors are reported beside the
   employer-clustered ones and beside the earlier hybrid ones, whatever
   they show, so the change is visible.
   Export: cluster_industry_v2.csv and the clustered covariance of every
   fit, so the women's step, which is the male step plus the
   differential, keeps a standard error of its own.

C. THE INDUSTRY EFFECTS REDONE. Script 78's Part F at 22 to 25 and
   script 79's Part B at 26 to 30 absorb three-digit industry by age
   band by month beside the calendar terms, and both ran on the
   employers Ftg_2019 could place. They are run again on the complete
   key, at both bands, each beside a baseline on the same sample.
   A carried-forward code is noisier than a contemporaneous one, and
   noise in the absorbing dimension makes it absorb less, which flatters
   us: the retained share would rise for a mechanical reason. The
   summary therefore reports how many firms in each fit carry a code
   from somewhere other than Ftg_2019.
   Read rule: no gate. The retained share is reported whatever it is.

WHAT EACH PART COSTS IF IT DIES. A loses its pulls, which are minutes.
B and C lose their own fits and nothing else. The key itself is cached
under I_industry_key.parquet as soon as it is built, so the second and
third parts read it rather than pulling again; three jobs building it at
once write the same frame and the write is atomic, so a race is harmless.

Output (output_80/):
  industry_key_coverage.csv   A: coverage by step and the size of what
                              Ftg_2019 missed, per panel
  cluster_industry_v2.csv     B: every term, three standard errors, the
                              cluster counts and the reproduction check
  industry_seasonal_v2.csv    C: both specifications at both bands, the
                              retained share and the non-2019 share
  vcov_s80_*.csv              the clustered covariance of every fit
  80_summary.txt

IN THE PAPER. A is a sentence in Online Appendix III.2 on how the
industry code is built and what it covers. B replaces the industry
standard errors in Table 1 and in the appendix table, wherever the
hybrid figures now stand. C replaces the retained shares of the industry
test at both bands.
"""

import gc
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_80_OUT", "output_80")
PARTS = os.environ.get("CANARIES_80_PARTS", "ABC").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

POST_FROM = "2024-01"            # adoption; checked against 78's below
BANDS = ["22-25", "26-30"]       # the two young bands the paper reports
SEX_BAND = "22-25"               # the band the paper's sex result is on
BASE_YEAR = 2019                 # the year the industry is frozen at
# Nearest year first, and the earlier of two equally near years first: the
# industry has to be frozen before the shock, so a backward step is safer
# than a forward one of the same length.
FTG_YEARS = [2019, 2018, 2020, 2017, 2021, 2016, 2022, 2015, 2023]
PRIMARY_SOURCE = f"Ftg_{BASE_YEAR}"
UNRESOLVED = "unresolved"        # ONE residual group, never a cluster each
MATCH_DP = 4                     # the reproduction check, decimals
FLOOR = 5                        # the export floor, as in mona_common
KEY_CACHE = CACHE / "I_industry_key.parquet"
KEY_COLS = ["employer_id", "ind3", "source"]
# Where lanes 25 and 26 wrote the runs this script is re-inferring. Listed
# rather than fixed because a differently split submission writes to a
# different folder, and a missing file costs a refit, not a wrong number.
PRIOR_ROOT = HERE
PRIOR_DIRS = ("output_78b", "output_78", "output_78a", "output_78c",
              "output_79a", "output_79", "output_79b", "output_79c")
POOLED_PRIOR = "cluster_industry.csv"          # 78 Part C
GENDER_EMP = "gender_eq2.csv"                  # 78 Part B
GENDER_EMP_VCOV = "vcov_s78_gender_eq2_22_25.csv"
GENDER_PRIOR = "gender_cluster_industry.csv"   # 79 Part A
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

    61 builds the balanced employer by band by month skeleton, 67 the same
    skeleton with sex as a fourth dimension, 78 the term sets of Equation
    (2) and of its sex split and the exposure, 73 the identifier
    normalisation and the catalogue probe, 47j the fixed-effect list, 47h
    the education score book. Importing 78 and 73 rather than copying
    their builders is the point: Part B must be 78's Part C and 79's Part
    A with one column changed, and a copied term list could drift.
    """
    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    s78 = _mod("78_final_checks.py", "s78")
    s73 = _mod("73_industry_and_credit.py", "s73")
    # 78's module-level OUT and CACHE are its own. Point them here so that
    # anything it writes lands with this script's exports rather than in
    # lane 25's folder, and so a test that moves the cache moves both.
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
    return s61, s67, s78, s73, j47, h47


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
        r = mc.run_fepois_multi(b, OUT, tag=f"s80_{tag}", terms=terms,
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


def cnt(v) -> str:
    """A count as the summary may print it: the export floor applies to the
    text that leaves MONA as much as to the CSV beside it."""
    if v is None or v != v:
        return "(suppressed)"
    v = int(v)
    return "(suppressed)" if 0 < v < FLOOR else f"{v:,}"


def prior(name: str) -> Path | None:
    """The first of lanes 25 and 26's export folders that holds a file."""
    for d in PRIOR_DIRS:
        f = Path(PRIOR_ROOT) / d / name
        if f.exists():
            return f
    return None


# ----------------------------------------------------------------------
# Part A: the key
# ----------------------------------------------------------------------

# Statistics Sweden says "no code" in more than one way, and every one of
# them reads as a perfectly good string. A code left in by mistake becomes
# an industry group of its own and pools firms that have nothing to do with
# each other, which is the defect this script exists to remove, so the
# conventions are matched by name and counted rather than assumed away.
MISSING_WORDS = ("na", "n/a", "missing", "unknown", "okand", "saknas",
                 "uppgift saknas")


def missing_reasons(raw: pd.Series) -> tuple:
    """
    (a boolean mask of the values that mean "no code", a count per
    convention). The conventions, in the order they are tested: a true
    null or an empty string; a run of asterisks, which is how "****"
    arrives; a run of dots or dashes; an all-zero code, which LISA uses
    for an unclassified firm; and the usual words.
    """
    s = raw.astype(str).str.strip()
    low = s.str.lower()
    null = pd.isna(raw) | low.isin(("", "nan", "none", "null", "<na>"))
    stars = s.str.fullmatch(r"\*+").fillna(False).astype(bool) & ~null
    dots = s.str.fullmatch(r"[.\-]+").fillna(False).astype(bool) & ~null
    zeros = s.str.fullmatch(r"0+(\.0+)?").fillna(False).astype(bool) & ~null
    words = low.isin(MISSING_WORDS) & ~null
    miss = null | stars | dots | zeros | words
    return miss, {"null or empty": int(null.sum()),
                  "asterisks": int(stars.sum()),
                  "dots or dashes": int(dots.sum()),
                  "all zeros": int(zeros.sum()),
                  "words": int(words.sum())}


def sni3_numeric(raw: pd.Series) -> pd.Series:
    """
    The three-digit group from a column stored as a floating-point number.

    Serrano's bransch_sni3 is a float, so group 011 (growing of crops)
    arrives as 11.0. Stripping the non-digits out of the text "11.0",
    which is what the text route does, would give "110" and file that
    firm under the manufacture of beverages. Cast, round and pad instead.
    """
    v = pd.to_numeric(raw, errors="coerce")
    ok = v.notna() & (v > 0) & (v < 1000)
    out = pd.Series(np.nan, index=raw.index, dtype=object)
    if ok.any():
        out.loc[ok] = (v[ok].round().astype(int).astype(str)
                       .str.zfill(3).to_numpy())
    return out


def to_sni3(raw: pd.Series, s73, numeric: bool = False) -> tuple:
    """(three-digit codes with the missing conventions removed, the counts)."""
    miss, reasons = missing_reasons(raw)
    kept = raw.where(~miss)
    code = sni3_numeric(kept) if numeric else s73._sni3(kept)
    reasons["unparseable"] = int((~miss & pd.isna(code)).sum())
    return code, reasons


def _cols_of(schema, table) -> list:
    return schema[schema["TABLE_NAME"].astype(str).str.lower()
                  == str(table).lower()]["COLUMN_NAME"].tolist()


def _find_table(schema, pattern: str):
    for t in sorted(schema["TABLE_NAME"].astype(str).unique()):
        if re.fullmatch(pattern, t, re.I):
            return t
    return None


def _reason_line(src: str, n_raw: int, n_ok: int, reasons: dict) -> str:
    named = ", ".join(f"{k} {v:,}" for k, v in reasons.items() if v)
    return (f"industry key: {src} gave {n_ok:,} coded firms of {n_raw:,} rows"
            + (f"; dropped by convention: {named}" if named else
               "; no missing convention fired"))


def pull_ftg(conn, schema, year: int, s73):
    """One LISA firm table, three columns, as script 73 reads Ftg_2019."""
    tab = _find_table(schema, rf"Ftg_{year}")
    if tab is None:
        NOTES.append(f"industry key: Ftg_{year} is not in the catalogue")
        return None
    cols = _cols_of(schema, tab)
    key = s73.pick(cols, r"^P1207_LopNr_PeOrgNr$", r"^LopNr_PeOrgNr$",
                   r"PeOrgNr")
    ind = s73.pick(cols, r"^Org_Sni2007$", r"Sni2007", r"^Sni")
    if not (key and ind):
        NOTES.append(f"industry key: {tab} has key {key} and industry {ind}; "
                     f"unusable, and the cascade moves on")
        return None
    d = pd.read_sql(f"SELECT [{key}] AS employer_id, [{ind}] AS ind "
                    f"FROM dbo.[{tab}]", conn)
    n_raw = len(d)
    d["employer_id"] = s73.norm_id(d["employer_id"])
    d["ind3"], reasons = to_sni3(d["ind"], s73)
    d = d.dropna(subset=["ind3"]).drop_duplicates("employer_id")
    NOTES.append(_reason_line(f"{tab} via {ind}", n_raw, len(d), reasons))
    d["source"] = tab
    return d[KEY_COLS]


def pull_serrano(conn, schema, s73):
    """
    Serrano's own firm table, at the 2019 accounting year.

    The key is P1207_Lopnr_ORGNR, the one script 73 already joins the
    balance sheet on, so the identifier handling is 73's and not a second
    version of it. The industry column is a float and goes through the
    numeric route.
    """
    tab = _find_table(schema, r"Serrano_Serrano_\d+")
    if tab is None:
        NOTES.append("industry key: no Serrano_Serrano table in the catalogue")
        return None
    cols = _cols_of(schema, tab)
    key = s73.pick(cols, r"^P1207_Lopnr_ORGNR$", r"ORGNR")
    ind = s73.pick(cols, r"^bransch_sni3$", r"^bransch_sni073$")
    yrc = s73.pick(cols, r"^ser_year$", r"^year$")
    if not (key and ind):
        NOTES.append(f"industry key: {tab} has key {key} and industry {ind}; "
                     f"unusable, and the cascade moves on")
        return None
    q = (f"SELECT [{key}] AS employer_id, [{ind}] AS ind FROM dbo.[{tab}]"
         + (f" WHERE [{yrc}] = {BASE_YEAR}" if yrc else ""))
    d = pd.read_sql(q, conn)
    n_raw = len(d)
    d["employer_id"] = s73.norm_id(d["employer_id"])
    d["ind3"], reasons = to_sni3(d["ind"], s73, numeric=True)
    d = d.dropna(subset=["ind3"]).drop_duplicates("employer_id")
    NOTES.append(_reason_line(
        f"{tab} via {ind}" + (f" at {yrc}={BASE_YEAR}" if yrc else ""),
        n_raw, len(d), reasons))
    d["source"] = "Serrano"
    return d[KEY_COLS]


def pull_fdb(conn, schema, s73):
    """
    The business register's legal-entity file, at reference year 2019.

    Last in the cascade for the reason 73's docstring gives: the delivered
    extract is not the employer population, so a low match rate here
    measures the register chosen and not the firms.
    """
    tabs = [t for t in sorted(schema["TABLE_NAME"].astype(str).unique())
            if t.lower().startswith("fdb_je")]

    def covers(t):
        yrs = [int(x) for x in re.findall(r"((?:19|20)\d{2})", t)]
        if len(yrs) == 2:
            return yrs[0] <= BASE_YEAR <= yrs[1]
        return len(yrs) == 1 and yrs[0] == BASE_YEAR

    tab = (next((t for t in tabs if covers(t)), None)
           or next((t for t in tabs if "all_years" in t.lower()), None))
    if tab is None:
        NOTES.append("industry key: no FDB_JE table covers 2019")
        return None
    cols = _cols_of(schema, tab)
    key = s73.pick(cols, r"^P1207_Lopnr_peorgnr$", r"PeOrgNr", r"PEORGNR")
    ind = s73.pick(cols, r"^ng3$", r"^ngs1$", r"^ng2$")
    yrc = s73.pick(cols, r"^ar$", r"^year$")
    if not (key and ind):
        NOTES.append(f"industry key: {tab} has key {key} and industry {ind}; "
                     f"unusable, and the cascade ends here")
        return None
    # `ar` is stored as text in this delivery, so the year is compared
    # after a cast rather than as a number against a string, which is how
    # a filter silently matches nothing.
    q = (f"SELECT [{key}] AS employer_id, [{ind}] AS ind FROM dbo.[{tab}]"
         + (f" WHERE TRY_CAST([{yrc}] AS INT) = {BASE_YEAR}" if yrc else ""))
    d = pd.read_sql(q, conn)
    n_raw = len(d)
    d["employer_id"] = s73.norm_id(d["employer_id"])
    d["ind3"], reasons = to_sni3(d["ind"], s73)
    d = d.dropna(subset=["ind3"]).drop_duplicates("employer_id")
    NOTES.append(_reason_line(
        f"{tab} via {ind}" + (f" at {yrc}={BASE_YEAR}" if yrc else ""),
        n_raw, len(d), reasons))
    d["source"] = "FDB_JE"
    return d[KEY_COLS]


SOURCE_ORDER = [f"Ftg_{y}" for y in FTG_YEARS] + ["Serrano", "FDB_JE"]


def build_key(s73) -> pd.DataFrame:
    """
    The cascade, in one frame: employer_id, ind3, source.

    The first source that answers keeps the firm, so a row's `source` is
    the step that resolved it and the steps are comparable across panels.
    """
    conn = mc.connect()
    try:
        schema = s73.discover(conn)
        if schema.empty:
            raise RuntimeError("no firm table is visible to this project; "
                               "the industry key cannot be built")
        parts, seen = [], set()
        for year in FTG_YEARS:
            d = pull_ftg(conn, schema, year, s73)
            if d is None or d.empty:
                continue
            d = d[~d["employer_id"].isin(seen)]
            if d.empty:
                NOTES.append(f"industry key: Ftg_{year} added no firm the "
                             f"earlier steps had not already placed")
                continue
            seen |= set(d["employer_id"])
            parts.append(d)
            print(f"  key: Ftg_{year} adds {len(d):,} firms "
                  f"({len(seen):,} so far)")
        for puller in (pull_serrano, pull_fdb):
            d = puller(conn, schema, s73)
            if d is None or d.empty:
                continue
            d = d[~d["employer_id"].isin(seen)]
            if d.empty:
                continue
            seen |= set(d["employer_id"])
            parts.append(d)
            print(f"  key: {d['source'].iloc[0]} adds {len(d):,} firms "
                  f"({len(seen):,} so far)")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not parts:
        raise RuntimeError("no source answered; the industry key is empty")
    key = pd.concat(parts, ignore_index=True)
    key["ind3"] = key["ind3"].astype(str)
    mc.write_cache(key, KEY_CACHE)
    msg = (f"industry key: {len(key):,} firms, "
           f"{key['ind3'].nunique()} three-digit groups, "
           f"{int((key['source'] == PRIMARY_SOURCE).sum()):,} of them from "
           f"{PRIMARY_SOURCE}")
    print(f"  {msg}")
    NOTES.append(msg)
    return key


def industry_key(s73) -> pd.DataFrame:
    """The cached key if it is there, else built and cached."""
    k = mc.read_cache(KEY_CACHE, require=KEY_COLS)
    if k is not None:
        k["ind3"] = k["ind3"].astype(str)
        NOTES.append(f"industry key: read from {KEY_CACHE.name}, "
                     f"{len(k):,} firms")
        return k
    return build_key(s73)


def key_maps(key: pd.DataFrame, s73) -> tuple:
    """(employer_id -> three-digit code, employer_id -> source)."""
    ids = s73.norm_id(key["employer_id"])
    return (dict(zip(ids, key["ind3"].astype(str))),
            dict(zip(ids, key["source"].astype(str))))


def panel_employers(b: pd.DataFrame) -> pd.DataFrame:
    """
    One row per employer on a panel: mean monthly headcount and the number
    of months in which it employed anybody.

    Both are read off the balanced skeleton the fits themselves run on, so
    "the employers Ftg_2019 missed" is the set the fits would have given a
    cluster of their own, not an approximation of it.
    """
    months = int(b["year_month"].nunique())
    tot = b.groupby("employer_id", observed=True)["n_emp"].sum()
    bym = b.groupby(["employer_id", "year_month"], observed=True)["n_emp"].sum()
    act = (bym > 0).groupby(level=0).sum()
    out = pd.DataFrame({"n_emp_total": tot, "months_active": act})
    out["mean_headcount"] = out["n_emp_total"] / max(months, 1)
    out["months_in_panel"] = months
    return out.reset_index()


SIZE_STATS = ["mean_headcount_p10", "mean_headcount_p25", "mean_headcount_p50",
              "mean_headcount_p75", "mean_headcount_p90",
              "mean_headcount_mean", "months_active_mean",
              "share_active_every_month", "share_active_under_6_months"]


def size_rows(panel: str, group: str, d: pd.DataFrame) -> list:
    """The size distribution of one group of employers, floored."""
    n = int(len(d))
    rows = [{"panel": panel, "block": "size", "group": group,
             "item": "n_employers", "n_employers": n, "share": np.nan,
             "value": float(n)}]
    if n < FLOOR:
        # Too few employers to describe without describing individuals.
        for it in SIZE_STATS:
            rows.append({"panel": panel, "block": "size", "group": group,
                         "item": it, "n_employers": n, "share": np.nan,
                         "value": np.nan})
        return rows
    h = d["mean_headcount"].astype(float)
    m = d["months_active"].astype(float)
    full = float(d["months_in_panel"].iloc[0])
    vals = {"mean_headcount_p10": float(h.quantile(0.10)),
            "mean_headcount_p25": float(h.quantile(0.25)),
            "mean_headcount_p50": float(h.quantile(0.50)),
            "mean_headcount_p75": float(h.quantile(0.75)),
            "mean_headcount_p90": float(h.quantile(0.90)),
            "mean_headcount_mean": float(h.mean()),
            "months_active_mean": float(m.mean()),
            "share_active_every_month": float((m >= full).mean()),
            "share_active_under_6_months": float((m < 6).mean())}
    for it in SIZE_STATS:
        rows.append({"panel": panel, "block": "size", "group": group,
                     "item": it, "n_employers": n, "share": np.nan,
                     "value": vals[it]})
    return rows


def coverage_rows(panel: str, emp: pd.DataFrame, src_map: dict,
                  s73) -> tuple:
    """
    Coverage by step for one panel, and the size distribution of what
    Ftg_2019 alone missed beside what it found.
    """
    ids = s73.norm_id(emp["employer_id"])
    src = ids.map(src_map).fillna(UNRESOLVED)
    n = int(len(emp))
    rows, cum = [], 0
    for step in SOURCE_ORDER + [UNRESOLVED]:
        k = int((src == step).sum())
        if k == 0 and step not in (PRIMARY_SOURCE, UNRESOLVED):
            continue
        cum += k
        rows.append({"panel": panel, "block": "step", "group": step,
                     "item": "resolved", "n_employers": k,
                     "share": k / max(n, 1), "value": cum / max(n, 1)})
    rows.append({"panel": panel, "block": "step", "group": "all",
                 "item": "employers_on_panel", "n_employers": n,
                 "share": 1.0, "value": 1.0})
    missed = emp[(src != PRIMARY_SOURCE).to_numpy()]
    found = emp[(src == PRIMARY_SOURCE).to_numpy()]
    # The residual group of Part B is this set and no other: firms the
    # whole cascade leaves uncoded. It is profiled separately from
    # "missed by Ftg_2019", most of which a later step recovers, because
    # what decides whether pooling them into one cluster matters is their
    # employment weight, not their number.
    unres = emp[(src == UNRESOLVED).to_numpy()]
    rows += size_rows(panel, f"in_{PRIMARY_SOURCE}", found)
    rows += size_rows(panel, f"missed_by_{PRIMARY_SOURCE}", missed)
    rows += size_rows(panel, UNRESOLVED, unres)
    tot_emp = float(emp["n_emp_total"].sum())
    n_unres = int(len(unres))
    share_emp_unres = (float(unres["n_emp_total"].sum()) / tot_emp
                       if (n_unres >= FLOOR and tot_emp > 0) else np.nan)
    rows.append({"panel": panel, "block": "size", "group": UNRESOLVED,
                 "item": "share_of_panel_employment",
                 "n_employers": n_unres, "share": share_emp_unres,
                 "value": share_emp_unres})
    summ = {"panel": panel, "n": n,
            "n_primary": int(len(found)), "n_missed": int(len(missed)),
            "n_unresolved": n_unres,
            "share_emp_unresolved": share_emp_unres,
            "med_unres": (float(unres["mean_headcount"].median())
                          if n_unres >= FLOOR else np.nan),
            "months_unres": (float(unres["months_active"].mean())
                             if n_unres >= FLOOR else np.nan),
            "med_found": (float(found["mean_headcount"].median())
                          if len(found) >= FLOOR else np.nan),
            "med_missed": (float(missed["mean_headcount"].median())
                           if len(missed) >= FLOOR else np.nan),
            "months_found": (float(found["months_active"].mean())
                             if len(found) >= FLOOR else np.nan),
            "months_missed": (float(missed["months_active"].mean())
                              if len(missed) >= FLOOR else np.nan)}
    return rows, summ


def part_a(counts, sexcounts, expo, key, s61, s67, s78, s73, j47) -> tuple:
    """
    What the cascade resolves, panel by panel, and what the first step
    misses.

    No fit. The panels are rebuilt exactly as the fits build them, because
    the question is how many of THOSE employers the 2019 table cannot
    place, and a shortcut employer list would answer a different one.
    """
    kmap, src_map = key_maps(key, s73)
    rows, summaries = [], []
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"A/{band}/empty")
            continue
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"A/{band}/no exposure")
            continue
        emp = panel_employers(b)
        del b
        gc.collect()
        r, s = coverage_rows(f"{band} stock", emp, src_map, s73)
        rows += r
        summaries.append(s)
        print(f"  A: {band} stock, {cnt(s['n'])} employers, "
              f"{cnt(s['n_missed'])} not in {PRIMARY_SOURCE}, "
              f"{cnt(s['n_unresolved'])} unresolved")
        del emp
        gc.collect()
    if sexcounts is not None:
        skel = s67.build_skeleton_sex(sexcounts, SEX_BAND, j47, "n_emp")
        if not skel.empty:
            b = s78.with_exposure(skel, expo)
            del skel
            gc.collect()
            if not b.empty:
                emp = panel_employers(b)
                del b
                gc.collect()
                r, s = coverage_rows(f"{SEX_BAND} sex", emp, src_map, s73)
                rows += r
                summaries.append(s)
                print(f"  A: {SEX_BAND} sex panel, {cnt(s['n'])} "
                      f"employers, {cnt(s['n_missed'])} not in "
                      f"{PRIMARY_SOURCE}")
                del emp
                gc.collect()
    else:
        NOTES.append("A: the L_counts_sex caches are not on the share, so the "
                     "sex panel's coverage is not reported")
    if not rows:
        return None, []
    tab = pd.DataFrame(rows)
    tab = mc.enforce_min_cell(tab, count_col="n_employers", floor=FLOOR)
    # A share and a published denominator reproduce a suppressed count, and
    # so does a cumulative share differenced across a suppressed step, so
    # the share goes with its own numerator and the cumulative column stops
    # at the first suppressed step of each panel. The headline figure
    # survives that: the unresolved row carries the coverage the key
    # reaches, and it is the last step rather than one of the small ones.
    gone = tab["n_employers"].isna()
    tab.loc[gone, ["share", "value"]] = np.nan
    for panel in sorted(set(tab["panel"])):
        m = (tab["panel"] == panel) & (tab["block"] == "step") \
            & (tab["item"] == "resolved")
        idx = list(tab.index[m])
        hit = [i for i in idx if bool(gone.get(i, False))]
        if hit:
            first = idx.index(hit[0])
            tab.loc[idx[first:], "value"] = np.nan
    tab.to_csv(OUT / "industry_key_coverage.csv", index=False)
    # The floored table, not the raw rows, is what the summary reads: the
    # summary leaves MONA with the exports.
    return tab, summaries


# ----------------------------------------------------------------------
# Part B: the clustering redone on the complete key
# ----------------------------------------------------------------------

def attach_cluster(b: pd.DataFrame, kmap: dict, src_map: dict, s73) -> tuple:
    """
    The cluster column, built once per EMPLOYER and mapped in as an
    integer.

    Two reasons for the integer. A forty-million-row text column costs
    gigabytes for a grouping key whose labels are never read, and the
    exchange file would factorise it anyway. Employers the cascade cannot
    resolve share ONE residual group: giving each of them a cluster of its
    own is the hybrid this script exists to remove, and dropping them
    would change the sample and so the coefficients.
    """
    emp_u = b["employer_id"].drop_duplicates()
    ids = s73.norm_id(emp_u)
    code = ids.map(kmap)
    src = ids.map(src_map)
    n_firms = int(len(emp_u))
    n_unres = int(code.isna().sum())
    n_other = int(((code.notna()) & (src != PRIMARY_SOURCE)).sum())
    label = code.where(code.notna(), UNRESOLVED)
    lut = pd.Series(pd.factorize(label)[0], index=emp_u.to_numpy())
    b["cl_ind"] = b["employer_id"].map(lut).astype("int64")
    n_clusters = int(lut.nunique())
    share_unres = float(
        (b["employer_id"].isin(emp_u[code.isna().to_numpy()]).mean())
        if n_unres else 0.0)
    return b, {"n_firms": n_firms, "n_clusters": n_clusters,
               "n_not_from_2019": n_other, "n_unresolved": n_unres,
               "share_cells_unresolved": share_unres}


def read_prior(fname: str, band: str | None, terms: list):
    """
    A coefficient table from lane 25 or lane 26, if it reached the share.

    Reading rather than refitting saves an hour and hides nothing: the
    reproduction check below is what proves the two runs are the same
    panel, and it runs on whatever is read.
    """
    f = prior(fname)
    if f is None:
        return None, ""
    try:
        g = pd.read_csv(f)
    except Exception as ex:
        print(f"  B: {f} unreadable ({type(ex).__name__}); ignored")
        return None, ""
    if "term" not in g.columns:
        return None, ""
    if band is not None and "young_band" in g.columns:
        g = g[g["young_band"].astype(str) == band]
    if not set(terms) <= set(g["term"].astype(str)):
        print(f"  B: {f} does not carry every term; ignored")
        return None, ""
    return g.set_index("term"), f"{f.parent.name}/{f.name}"


def _se(g, term) -> float | None:
    """
    The standard error of one term from a prior export.

    Lane 25's cluster_industry.csv names the employer-clustered column
    se_employer and lane 25's gender_eq2.csv names it se, and a refit here
    names it se as well, so the three are read through one accessor rather
    than through three branches at the call site.
    """
    if g is None or term not in g.index:
        return None
    for c in ("se_employer", "se"):
        if c in g.columns:
            v = g.loc[term, c]
            return None if pd.isna(v) else float(v)
    return None


def _coef(g, term) -> float:
    """
    The employer-clustered coefficient from a prior export.

    Lane 25's cluster_industry.csv carries the industry run's coefficient
    in `coef` and the employer run's in `coef_employer_run`, and the two
    agree because that run's own check passed. The employer column is
    preferred all the same: the check here has to compare this fit with
    the employer-clustered run, not with a copy of an industry one.
    """
    if g is None or term not in g.index:
        return np.nan
    for c in ("coef_employer_run", "coef"):
        if c in g.columns:
            v = g.loc[term, c]
            if not pd.isna(v):
                return float(v)
    return np.nan


def _row_of(spec: str, band: str, term: str, g_new, v_new, emp, hyb,
            info: dict) -> dict:
    ci = float(g_new.loc[term, "coef"])
    ce = _coef(emp, term)
    se_e = _se(emp, term)
    se_e = np.nan if se_e is None else se_e
    sh = np.nan
    if hyb is not None and term in hyb.index and "se_industry" in hyb.columns:
        sh = float(hyb.loc[term, "se_industry"])
    nch = np.nan
    if hyb is not None and term in hyb.index \
            and "n_clusters_industry" in hyb.columns:
        nch = float(hyb.loc[term, "n_clusters_industry"])
    return {"spec": spec, "young_band": band, "term": term, "coef": ci,
            "se_employer": se_e,
            "se_industry_hybrid": sh,
            "se_industry_complete": float(g_new.loc[term, "se"]),
            "coef_employer_run": ce,
            "coef_match_4dp": (bool(round(ci, MATCH_DP) == round(ce, MATCH_DP))
                               if not np.isnan(ce) else None),
            "n_clusters_complete": info["n_clusters"],
            "n_clusters_hybrid": nch,
            "n_firms": info["n_firms"],
            "n_not_from_2019": info["n_not_from_2019"],
            "n_unresolved": info["n_unresolved"],
            "n_obs": int(g_new.loc[term, "n_obs"]),
            "status": str(g_new.loc[term].get("status", "ok"))}


def part_b(counts, sexcounts, expo, key, s61, s67, s78, s73, j47) -> tuple:
    """
    78's Part C and 79's Part A, clustered on the complete key.

    Estimates: the pooled stock fit of Equation (2) at each young band,
    and the sex panel with every term of Equation (2) entered three ways,
    as High x Young, as High x Female and as High x Young x Female. The
    panels, the terms and the fixed effects are the ones those scripts
    used, so the coefficients must come back unchanged and only the
    covariance moves.

    Writes cluster_industry_v2.csv and, through the fits, the three
    vcov_s80_*.csv files.
    """
    kmap, src_map = key_maps(key, s73)
    rows, summ = [], {}
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"B/{band}/empty")
            continue
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"B/{band}/no exposure")
            continue
        b, info = attach_cluster(b, kmap, src_map, s73)
        msg = (f"B/{band}: {cnt(info['n_firms'] - info['n_unresolved'])} of "
               f"{cnt(info['n_firms'])} employers carry a three-digit code "
               f"({cnt(info['n_not_from_2019'])} of them from a source other "
               f"than {PRIMARY_SOURCE}); {cnt(info['n_unresolved'])} "
               f"unresolved share one residual group holding "
               f"{info['share_cells_unresolved']:.2%} of the panel cells; "
               f"{info['n_clusters']:,} clusters")
        print(f"  {msg}")
        NOTES.append(msg)
        b, terms = s78.eq2_terms(b)
        tag = band.replace("-", "_")
        g_new, v_new = fit(b, f"clind2_{tag}", terms, j47.FES,
                           cluster="cl_ind")
        if g_new is None:
            del b
            gc.collect()
            continue
        hyb, src_h = read_prior(POOLED_PRIOR, band, terms)
        emp, src_e = (hyb, src_h)
        if emp is None:
            print("  B: lane 25's cluster_industry.csv is not on the share; "
                  "the employer-clustered run is repeated here")
            emp, _ = fit(b, f"clemp_{tag}", terms, j47.FES)
            src_e = "refitted in this run"
        del b
        gc.collect()
        NOTES.append(f"B/{band}: employer-clustered and earlier hybrid "
                     f"numbers from {src_e or 'nowhere'}")
        for t_ in terms:
            if t_ in g_new.index:
                rows.append(_row_of("pooled", band, t_, g_new, v_new, emp,
                                    hyb, info))
        summ[f"pooled_{band}"] = {
            "info": info, "v": v_new,
            "coef": {t_: float(g_new.loc[t_, "coef"]) for t_ in terms
                     if t_ in g_new.index}}
        pd.DataFrame(rows).to_csv(OUT / "cluster_industry_v2.csv", index=False)
    # ---- the sex specification ---------------------------------------
    if sexcounts is None:
        FAILURES.append("B/no L_counts_sex cache")
        print("  B: L_counts_sex_* missing, the sex specification is skipped")
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "cluster_industry_v2.csv",
                                      index=False)
        return rows, summ
    skel = s67.build_skeleton_sex(sexcounts, SEX_BAND, j47, "n_emp")
    if skel.empty:
        FAILURES.append("B/sex/empty")
        return rows, summ
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("B/sex/no exposure")
        return rows, summ
    b, info = attach_cluster(b, kmap, src_map, s73)
    msg = (f"B/sex {SEX_BAND}: {cnt(info['n_firms'] - info['n_unresolved'])} "
           f"of {cnt(info['n_firms'])} employers carry a three-digit code "
           f"({cnt(info['n_not_from_2019'])} from a source other than "
           f"{PRIMARY_SOURCE}); {cnt(info['n_unresolved'])} unresolved share "
           f"one residual group holding "
           f"{info['share_cells_unresolved']:.2%} of the panel cells; "
           f"{info['n_clusters']:,} clusters")
    print(f"  {msg}")
    NOTES.append(msg)
    b, terms = s78.gender_eq2_terms(b)
    g_new, v_new = fit(b, f"gender_clind2_{SEX_BAND.replace('-', '_')}",
                       terms, j47.FES, cluster="cl_ind")
    if g_new is None:
        del b
        gc.collect()
        return rows, summ
    emp, src_e = read_prior(GENDER_EMP, SEX_BAND, terms)
    hyb, src_h = read_prior(GENDER_PRIOR, SEX_BAND, terms)
    v_emp = None
    vf = prior(GENDER_EMP_VCOV)
    if vf is not None:
        try:
            v_emp = pd.read_csv(vf).set_index("term")
        except Exception:
            v_emp = None
    if emp is None:
        print("  B: lane 25's gender_eq2.csv is not on the share; the "
              "employer-clustered run is repeated here")
        emp, v_emp = fit(b, f"gender_clemp_{SEX_BAND.replace('-', '_')}",
                         terms, j47.FES)
        src_e = "refitted in this run"
    del b
    gc.collect()
    NOTES.append(f"B/sex: employer-clustered numbers from "
                 f"{src_e or 'nowhere'}; earlier hybrid numbers from "
                 f"{src_h or 'nowhere'}")
    for t_ in terms:
        if t_ in g_new.index:
            rows.append(_row_of("gender", SEX_BAND, t_, g_new, v_new, emp,
                                hyb, info))
    pd.DataFrame(rows).to_csv(OUT / "cluster_industry_v2.csv", index=False)
    # The three numbers the paper quotes, each under all three variances.
    # The women's step is the male step plus the differential, so its
    # standard error comes from the covariance of the fit and not from
    # adding two standard errors.
    m, d = "post_x_high_x_young", "post_x_high_x_young_x_female"
    mi, di = "interim_x_high_x_young", "interim_x_high_x_young_x_female"
    c = {t_: float(g_new.loc[t_, "coef"]) for t_ in terms if t_ in g_new.index}
    g = {}
    if m in c and d in c:
        hy = (float(hyb.loc[m, "se_industry"])
              if hyb is not None and m in hyb.index
              and "se_industry" in hyb.columns else None)
        hd = (float(hyb.loc[d, "se_industry"])
              if hyb is not None and d in hyb.index
              and "se_industry" in hyb.columns else None)
        g["male_step"] = (c[m], _se(emp, m), hy,
                          float(g_new.loc[m, "se"]))
        g["female_differential"] = (c[d], _se(emp, d), hd,
                                    float(g_new.loc[d, "se"]))
        g["female_step"] = (c[m] + c[d], lincomb(v_emp, {m: 1, d: 1}), None,
                            lincomb(v_new, {m: 1, d: 1}))
    if all(k in c for k in (m, d, mi, di)):
        g["male_step_from_2023"] = (
            c[m] - c[mi], lincomb(v_emp, {m: 1, mi: -1}), None,
            lincomb(v_new, {m: 1, mi: -1}))
        g["female_step_from_2023"] = (
            c[m] + c[d] - c[mi] - c[di],
            lincomb(v_emp, {m: 1, d: 1, mi: -1, di: -1}), None,
            lincomb(v_new, {m: 1, d: 1, mi: -1, di: -1}))
    summ["gender"] = {"info": info, "steps": g}
    return rows, summ


# ----------------------------------------------------------------------
# Part C: the industry effects redone on the complete key
# ----------------------------------------------------------------------

def part_c(counts, expo, key, s61, s78, s73, j47) -> list:
    """
    78's Part F and 79's Part B on the complete key, at both bands.

    Estimates: the stock fit of Equation (2), the tightening switch, the
    interim window, the adoption step and the three calendar terms, each
    interacted with High x Young, twice on the same firms. The baseline
    carries 47j's three effects; the industry specification replaces the
    month-by-age effect, which is nested inside it, with three-digit
    industry by age band by month. An employer with no code at all leaves
    the sample, as it did in 78 and 79, because a residual group is not an
    industry and absorbing on one would mean something different in each
    fit; both fits then run on exactly the same firms.

    Writes industry_seasonal_v2.csv and the four covariance files.
    """
    kmap, src_map = key_maps(key, s73)
    rows = []
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"C/{band}/empty")
            continue
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{band}/no exposure")
            continue
        emp_u = b["employer_id"].drop_duplicates()
        ids = s73.norm_id(emp_u)
        code = ids.map(kmap)
        src = ids.map(src_map)
        n_all = int(len(emp_u))
        lut = pd.Series(pd.factorize(code)[0], index=emp_u.to_numpy())
        other = pd.Series(((code.notna()) & (src != PRIMARY_SOURCE)).to_numpy(),
                          index=emp_u.to_numpy())
        b["ind_code"] = b["employer_id"].map(lut).astype("int64")
        b = b[b["ind_code"] >= 0].copy()
        if b.empty:
            FAILURES.append(f"C/{band}/no industry")
            continue
        kept = b["employer_id"].drop_duplicates()
        n_firms = int(len(kept))
        n_other = int(other.reindex(kept.to_numpy()).fillna(False).sum())
        n_groups = int(b["ind_code"].nunique())
        share_other = n_other / max(n_firms, 1)
        msg = (f"C/{band}: {cnt(n_firms)} of {cnt(n_all)} employers carry a "
               f"three-digit code and form the sample for both fits; "
               f"{cnt(n_other)} of them ({share_other:.1%}) from a source "
               f"other than {PRIMARY_SOURCE}; {n_groups:,} industry groups")
        print(f"  {msg}")
        NOTES.append(msg)
        ic = b["ind_code"].to_numpy(dtype="int64")
        ac = pd.factorize(b["age_group"], sort=False)[0].astype("int64")
        tc = pd.factorize(b["year_month"], sort=False)[0].astype("int64")
        n_a, n_t = int(ac.max()) + 1, int(tc.max()) + 1
        b["fe_ind_age_t"] = (ic * n_a + ac) * n_t + tc
        b, terms = s78.eq2_terms(b)
        tag = band.replace("-", "_")
        g_base, _ = fit(b, f"indseas2_base_{tag}", terms, j47.FES)
        # fe_t_age is nested inside industry x age x month and is dropped,
        # as scripts 73, 78 and 79 drop it: the model is the same and one
        # effect fewer fits.
        fes = tuple(f for f in j47.FES if f != "fe_t_age") + ("fe_ind_age_t",)
        g_ind, _ = fit(b, f"indseas2_ind_{tag}", terms, fes)
        del b
        gc.collect()
        post = "post_x_high_x_young"
        pb = (float(g_base.loc[post, "coef"])
              if g_base is not None and post in g_base.index else np.nan)
        pi = (float(g_ind.loc[post, "coef"])
              if g_ind is not None and post in g_ind.index else np.nan)
        retained = (pi / pb) if (pb == pb and pi == pi and pb != 0) else np.nan
        for spec_, g_, share in (("baseline_same_sample", g_base, 1.0),
                                 ("industry_age_month", g_ind, retained)):
            if g_ is None:
                continue
            rows += s78.rows_of(g_, terms, young_band=band, spec=spec_,
                                n_firms=n_firms, n_not_from_2019=n_other,
                                share_not_from_2019=share_other,
                                n_groups=n_groups, retained_share=share)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / "industry_seasonal_v2.csv",
                                      index=False)
    return rows


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  A. Descriptive. Coverage is reported at every step of the cascade and",
    "     the employers the 2019 table alone missed are described beside the",
    "     ones it found, whichever way the comparison falls. Employer counts",
    f"     below {FLOOR} are suppressed before anything leaves MONA.",
    "  B. GATE. The coefficients must reproduce the employer-clustered run to",
    f"     {MATCH_DP} decimals. If any term fails, nothing from Part B is",
    "     quoted, because the cluster changes the covariance and nothing",
    "     else, so a moved coefficient means a moved panel. The new standard",
    "     errors are reported beside the employer-clustered ones and beside",
    "     the earlier hybrid ones whatever they show.",
    "  C. No gate. The retained share of the adoption step against the",
    "     same-sample baseline is reported whatever it is, and so is the",
    "     share of firms in each fit whose code is not from Ftg_2019, since a",
    "     carried-forward code absorbs less and so flatters the test.",
]


def fmt(x):
    return "(no SE)" if x is None or x != x else f"({x:.4f})"


def a_summary_lines(summaries: list) -> list:
    """What the cascade found, said plainly."""
    L = []
    for s in summaries:
        L.append(f"  {s['panel']}: {cnt(s['n'])} employers, "
                 f"{cnt(s['n'] - s['n_missed'])} placed by {PRIMARY_SOURCE} "
                 f"({(s['n'] - s['n_missed']) / max(s['n'], 1):.1%}), "
                 f"{cnt(s['n_missed'] - s['n_unresolved'])} placed by a later "
                 f"step, {cnt(s['n_unresolved'])} unresolved")
        if s["med_missed"] == s["med_missed"] and s["med_found"] == s["med_found"]:
            L.append(f"    median monthly headcount {s['med_missed']:.1f} in "
                     f"the employers {PRIMARY_SOURCE} missed against "
                     f"{s['med_found']:.1f} in the ones it found; months "
                     f"employing anybody {s['months_missed']:.1f} against "
                     f"{s['months_found']:.1f}")
            smaller = (s["med_missed"] < s["med_found"]
                       and s["months_missed"] < s["months_found"])
            L.append("    " + ("The missing firms are smaller and shorter "
                               "lived, which is the November reference date "
                               "doing what we expected."
                               if smaller else
                               "The missing firms are NOT both smaller and "
                               "shorter lived, so the reference-date story "
                               "does not account for them on its own."))
        # The residual group of Part B, stated here so its weight can be
        # read without opening the CSV. Its employment share, not its
        # firm count, is what decides whether one shared cluster matters.
        nu = s["n_unresolved"]
        if nu == 0:
            L.append("    The cascade leaves no employer uncoded, so Part B "
                     "has no residual group and every cluster is a real "
                     "three-digit industry.")
        elif s.get("share_emp_unresolved") == s.get("share_emp_unresolved"):
            L.append(f"    Part B residual group: {cnt(nu)} employers "
                     f"({nu / max(s['n'], 1):.1%} of the panel) carrying "
                     f"{s['share_emp_unresolved']:.2%} of its employment, "
                     f"median monthly headcount {s['med_unres']:.1f}, "
                     f"{s['months_unres']:.1f} months employing anybody. "
                     f"They share ONE cluster; none is a cluster of its own.")
        else:
            L.append(f"    Part B residual group: fewer than {FLOOR} "
                     f"employers, so it is not described further.")
    return L


def main():
    mc.Tee(OUT / "80_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"80: THE INDUSTRY KEY COMPLETED   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s61, s67, s78, s73, j47, h47 = load_modules()
    book, spec = score_book(h47)
    frame19 = frame_2019(h47)
    expo = s78.exposure(frame19, book, spec, j47, s61)
    del frame19
    gc.collect()
    print(f"  exposure: {len(expo):,} firms")

    key = industry_key(s73)

    counts = None
    if any(p in PARTS for p in "ABC"):
        counts = load_counts("L_counts", s61.PANEL_YEARS)
        if counts is None:
            raise RuntimeError("L_counts_* missing: run 47L first.")
    sexcounts = None
    if any(p in PARTS for p in "AB"):
        sexcounts = load_counts("L_counts_sex", s61.PANEL_YEARS,
                                require=["employer_id", "year_month",
                                         "age_group", "gender", "n_emp"])

    cov_tab, cov_summ, cl_rows, cl_summ, ind_rows = None, [], [], {}, []
    if "A" in PARTS:
        r = opt("Part A", part_a, counts, sexcounts, expo, key, s61, s67,
                s78, s73, j47)
        if r:
            cov_tab, cov_summ = r
    if "B" in PARTS:
        r = opt("Part B", part_b, counts, sexcounts, expo, key, s61, s67,
                s78, s73, j47)
        if r:
            cl_rows, cl_summ = r
    del sexcounts
    gc.collect()
    if "C" in PARTS:
        r = opt("Part C", part_c, counts, expo, key, s61, s78, s73, j47)
        if r:
            ind_rows = r
    del counts
    gc.collect()

    # ---- summary ------------------------------------------------------
    L = ["THE INDUSTRY KEY COMPLETED", "=" * 52, "",
         "Employer x age x month counts, exposure the top quartile of the",
         "2019 education mix, employer-by-month, employer-by-age and",
         "month-by-age effects, Poisson, calendar cycle removed. Nothing",
         "here changes the design. Part A builds the industry key and",
         "reports what it covers, Part B changes only what the standard",
         "errors are clustered on, Part C only which effects absorb the",
         "cell.", "",
         f"The defect being repaired: {PRIMARY_SOURCE} alone left 5,285 of the",
         "111,459 employers on the 22-25 panel and 9,106 of the 128,193 on",
         "the 26-30 panel without a code, and each of those was given a",
         "cluster of its own, so the 5,545 and 9,368 clusters the earlier",
         "runs report are 265 real industry groups and several thousand",
         "singletons.", ""]
    if "A" in PARTS:
        L += ["A. THE CASCADE, PANEL BY PANEL:"]
        if cov_summ and cov_tab is not None:
            L += a_summary_lines(cov_summ)
            step = cov_tab[(cov_tab["block"] == "step")
                           & (cov_tab["item"] == "resolved")]
            for panel in sorted(set(step["panel"])):
                L.append(f"  {panel}, employers resolved at each step:")
                for _, r_ in step[step["panel"] == panel].iterrows():
                    sh, cu = r_["share"], r_["value"]
                    L.append(f"    {str(r_['group']):<12} "
                             f"{cnt(r_['n_employers']):>12}   "
                             + ("       " if sh != sh else f"{sh:7.3%}")
                             + "   cumulative "
                             + ("       " if cu != cu else f"{cu:7.3%}"))
        else:
            L.append("  no panel came back")
        L.append("")
    if "B" in PARTS:
        L += ["B. THE CLUSTERING REDONE ON THE COMPLETE KEY:"]
        if cl_rows:
            B = pd.DataFrame(cl_rows)
            ok = B["coef_match_4dp"].dropna()
            L.append(f"  coefficients equal the employer-clustered run to "
                     f"{MATCH_DP} decimals: "
                     f"{'YES' if len(ok) and bool(ok.all()) else 'NO'}")
            if not (len(ok) and bool(ok.all())):
                L.append("  THE RULE SAYS NOTHING FROM PART B IS QUOTED.")
            for spec_ in ("pooled", "gender"):
                S = B[B["spec"] == spec_]
                if S.empty:
                    continue
                for band in sorted(set(S["young_band"])):
                    g = S[S["young_band"] == band].set_index("term")
                    nh = g["n_clusters_hybrid"].dropna()
                    L.append(f"  {spec_} {band}: "
                             f"{int(g['n_clusters_complete'].iloc[0]):,} "
                             f"clusters on the complete key against "
                             + (f"{int(nh.iloc[0]):,}" if len(nh) else "?")
                             + " before; "
                             f"{cnt(g['n_not_from_2019'].iloc[0])} employers "
                             f"coded from another source, "
                             f"{cnt(g['n_unresolved'].iloc[0])} unresolved")
                    L.append(f"  {'term':<34} {'coef':>9}  {'employer':>10}  "
                             f"{'hybrid':>10}  {'industry':>10}")
                    for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                               "post_x_high_x_young", "post_x_high_x_female",
                               "post_x_high_x_young_x_female"):
                        if t_ not in g.index:
                            continue
                        r_ = g.loc[t_]
                        L.append(f"  {t_:<34} {r_['coef']:+9.4f}  "
                                 f"{r_['se_employer']:10.4f}  "
                                 f"{r_['se_industry_hybrid']:10.4f}  "
                                 f"{r_['se_industry_complete']:10.4f}")
            gs = cl_summ.get("gender", {}).get("steps", {})
            for k, nm in (("male_step", "young men, adoption step"),
                          ("female_differential", "young women minus young men"),
                          ("female_step", "young women, adoption step"),
                          ("male_step_from_2023", "young men, step from 2023"),
                          ("female_step_from_2023", "young women, step from 2023")):
                if k in gs:
                    c_, se_e, se_h, se_i = gs[k]
                    L.append(f"  {nm:<34} {c_:+.4f}  employer {fmt(se_e)}  "
                             f"hybrid {fmt(se_h)}  industry {fmt(se_i)}")
            if gs:
                L += ["  The women's step is the male step plus the "
                      "differential and its",
                      "  standard error comes from the covariance of each fit, "
                      "so it is not",
                      "  the sum of two standard errors. The hybrid column is "
                      "blank for the",
                      "  combinations, whose earlier standard errors were "
                      "never computed."]
        else:
            L.append("  no fit came back")
        L.append("")
    if "C" in PARTS:
        L += ["C. INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS, "
              "BOTH BANDS:"]
        if ind_rows:
            I = pd.DataFrame(ind_rows)
            for band in sorted(set(I["young_band"])):
                Ib = I[I["young_band"] == band]
                for spec_ in ("baseline_same_sample", "industry_age_month"):
                    g = Ib[Ib["spec"] == spec_].set_index("term")
                    for t_ in ("rb_x_high_x_young", "interim_x_high_x_young",
                               "post_x_high_x_young"):
                        if t_ in g.index:
                            L.append(f"  {band} {spec_:<22} {t_:<26} "
                                     f"{float(g.loc[t_,'coef']):+.4f} "
                                     f"({float(g.loc[t_,'se']):.4f})")
                gi = Ib[Ib["spec"] == "industry_age_month"]
                if len(gi):
                    sh = float(gi["retained_share"].iloc[0])
                    L.append(f"  {band}: employers in both fits "
                             f"{cnt(gi['n_firms'].iloc[0])}, of which "
                             f"{cnt(gi['n_not_from_2019'].iloc[0])} "
                             f"({float(gi['share_not_from_2019'].iloc[0]):.1%}) "
                             f"carry a code from a source other than "
                             f"{PRIMARY_SOURCE}")
                    L.append(f"  {band}: retained share of the adoption step "
                             f"against the same-sample baseline: "
                             f"{'n/a' if sh != sh else format(sh, '.0%')}")
            L += ["  Read beside 78's Part F (82 per cent at 22-25) and 79's",
                  "  Part B (46 per cent at 26-30), which are the same fits on",
                  "  the employers Ftg_2019 could place. A carried-forward code",
                  "  is noisier, and noise in the absorbing dimension raises the",
                  "  retained share for a mechanical reason, so the share of",
                  "  non-2019 codes above is part of the reading."]
        else:
            L.append("  no fit came back")
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
        "  1. A is descriptive. It says what the industry variable covers and",
        "     what the 2019 table alone missed; it tests nothing.",
        "  2. B is inference only. The coefficients are 78's and 79's, and the",
        "     paper replaces the hybrid standard errors with these rather than",
        "     reporting both as findings.",
        "  3. C replaces the retained shares of the industry test at both",
        "     bands, with the non-2019 share beside them.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "80_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("80_industry_key", 0, (time.time() - t0) / 60)
    print("\n80 done.")


if __name__ == "__main__":
    main()

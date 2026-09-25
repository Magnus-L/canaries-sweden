#!/usr/bin/env python3
"""
96_payroll_cohorts.py -- the reduced youth payroll contribution as a
                         rival, tested by BIRTH COHORT rather than by age.

======================================================================
  RUNS IN MONA (lane 37b). Output folder CANARIES_96_OUT (default
  output_96); parts with CANARIES_96_PARTS (default CDS). ONE new SQL
  pull: 47L's counts query (with 67's sex column) in which every worker
  born 1994 or later is counted in a cell of his or her own birth year
  and everyone older in the paper's age bands, 2021 to 2025, cached as
  L_counts_cohortsex_YYYY (about three minutes a year, as 67's sex pull).
  Everything else is a cache.
======================================================================

THE STATUTE (verified 25 Sep 2026 in the enacted text, SFS 2021:55 as
amended by SFS 2021:591 and SFS 2022:240, and in Prop. 2020/21:83,
2020/21:202 and 2021/22:97, all at data.riksdagen.se)
Lag (2021:55) om saerskild beraekning av arbetsgivaravgifter och allmaen
loeneavgift foer personer som vid aarets ingaang fyllt 18 men inte 23 aar:
on pay to persons who "vid aarets ingaang har fyllt 18 men inte 23 aar",
only the old-age pension contribution and nine twentieths of the other
contributions are paid (19.73 per cent against 31.42), up to SEK 25,000 a
month. In force 6 February 2021 and applied to pay disbursed AFTER 31
DECEMBER 2020 (retroactively from 1 January 2021, not from 1 April 2021 as
the 2021 budget bill had announced and as Prop. 2025/26:66 later
summarised it); repealed at the end of March 2023, still applying to pay
disbursed 1 January 2021 to 31 March 2023. From 1 June to 31 August 2021
and 1 June to 31 August 2022 only the old-age pension contribution (10.21
per cent) was paid (SFS 2021:591, 2022:240).

WHO WAS COVERED. Turned 18 but not 23 at the start of year Y means born
in Y-23 to Y-19 (the proposition: everyone turning 19 to 23 during the
year). So the covered cohorts are 1998-2002 in 2021, 1999-2003 in 2022
and 2000-2004 in January to March 2023. Every cohort born 1998 to 2004
was covered at some point; born 1997 or earlier, never.

THE PANEL'S AGE. Age is the calendar year minus the birth year
(47L's q_counts: `{year} - FodelseAr`, birth year from Individ 2023, 2021
or 2019), so a worker is 22-25 in 2024 if born 1999-2002 and in 2025 if
born 2000-2003. EVERY WORKER AGED 22-25 IN 2024-25 WAS COVERED AT SOME
POINT. No shift of the young band recovers a never-covered group inside
22-25; the never-covered young adults are born 1994-1997, aged 27-30 in
2024 and 28-31 in 2025. Script 92's premise ("24-25 were never
eligible") is wrong on this definition, and 92 must not be run.

THE DESIGN. Equation (2) exactly (script 78's term set: tightening switch,
interim window, adoption step, three calendar terms, each x High x Young;
employer-by-month, employer-by-cell and month-by-cell effects; Poisson;
clustered by employer; exposure script 82's reported score), with the
young group defined by birth cohort instead of age: a cohort cell is
followed through the panel as it ages, and its national path, entry into
the panel at 22 included, is absorbed by the month-by-cell effect. The
older reference is the paper's four bands 31-34, 35-40, 41-49 and 50-69,
built from workers born 1993 or earlier (so the only change against the
headline reference is that workers born 1994 who turn 31 in 2025 are not
in it). tau = b_L - b_I, the adoption step minus the interim window, with
its standard error from the clustered covariance (V_LL + V_II - 2 V_LI).

THE GATES (hard stop, before anything else)
1. The data. The pull, re-banded by age (a cohort cell's age is the year
   minus its birth year), is compared cell by cell with 47L's L_counts
   (all sexes) and with 67's L_counts_sex (sex 1 and 2); the number of
   cells that differ is reported.
2. The headline. The re-banded pull must reproduce Table 1 at 22-25
   within 0.0005 on coefficient and standard error: adoption step
   -0.0578 (0.0155), tau -0.0399 (0.0102). A miss stops the script.
3. The sex parts additionally require the re-banded sex panel to
   reproduce Table 1's female differential, -0.0858 (0.0142) at adoption
   and tau -0.0714 (0.0109); otherwise Part S is not run.

PART C. THE COHORT CONTRASTS
  (a) never covered: born 1994-1997.
  (b) ever covered, in the panel's young ages: born 1998-2003. The 2002
      and 2003 cohorts reach 22 only in 2024 and 2025, so they enter the
      group in the later window.
  (b2) ever covered and in the panel through both windows: born
      1998-2000 (2000 turns 22 in 2022).
PART D. THE DOSE
  Months of eligibility by cohort, from the statute: 1998 12 (2021 only),
  1999 24 (2021-22), 2000 and 2001 27 (2021, 2022 and January to March
  2023), 2002 27 and 2003 15 (not identified: they enter the panel only in
  the later window, so they have no interim months), 2004 3 (never in the
  panel), 1994-1997 0. (i) One fit with its own Equation (2) terms for
  each class d00 (1994-1997), d12 (1998), d24 (1999) and d27 (2000-2001),
  every class a cell of its own: tau per class. (ii) One fit with common
  terms x High x Young and the same terms x High x Young x dose (years of
  eligibility): the tau of the dose term is the change in tau per year
  of eligibility.
PART S. THE FEMALE DIFFERENTIAL AMONG THE NEVER COVERED
  67's sex specification (every term x High x Young, x High x Female and
  x High x Young x Female; employer-by-cell-and-sex and
  month-by-cell-and-sex effects, still three fixed-effect dimensions) with
  the young cell the 1994-1997 cohorts.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  B1. THE DECLINE IS NOT THE PAYROLL REDUCTION if tau for the
      never-covered cohorts (a) is negative and distinguishable from zero
      at five per cent. THE PAYROLL RIVAL SURVIVES if (b) is negative and
      distinguishable while (a) is not. Anything else is reported as it
      comes.
  B2. The dose carries no verdict. Within the panel eligibility falls
      almost monotonically with age (0 months at 27-31 in the later
      window, 27 months at 23-25), so a dose slope cannot be told from an
      age gradient; the per-class taus are reported so a reader can see
      whether tau steps up where eligibility does.
  B3. The never-covered female differential is reported with its own
      standard error; it is compared with Table 1's only in words.
  A missing row is a missing fit, never a zero.

EXPORT (output_96/)
  payroll_cohorts.csv   every term and every tau, with var_post,
                        var_interim and cov_post_interim, n_obs,
                        n_firms and the young person-months in the
                        interim and later windows (employer counts below
                        five suppressed with their statistic)
  96_summary.txt, 96_log.txt, vcov_s96_*.csv (the last stay on the share)

IN THE PAPER
Replaces the payroll-reduction rebuttal ("24-25 never covered") in the
Results rivals paragraph, OA III.2 and the response letter by the cohort
estimates; the statute's dates in the OA are corrected to 1 January 2021
to 31 March 2023 with the June-August 2021 and 2022 deepening.

    python 96_payroll_cohorts.py
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

OUT = HERE / os.environ.get("CANARIES_96_OUT", "output_96")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_96_PARTS", "CDS").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
SIG5 = 1.959963984540054
PREFIX = "L_counts_cohortsex"
PULL_COLS = ["employer_id", "year_month", "cell", "gender", "n_emp"]
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]
FIRST_COHORT = 1994            # born this year or later: a cell of its own
REF = ["31-34", "35-40", "41-49", "50+"]
BANDS6 = {"22-25": (22, 25), "26-30": (26, 30), "31-34": (31, 34),
          "35-40": (35, 40), "41-49": (41, 49), "50+": (50, 69)}

# The statute: months of eligibility by birth year, Jan 2021 - Mar 2023.
# Covered in year Y if born Y-23 .. Y-19; 2023 counts three months.
def eligible_months(by: int) -> int:
    m = 0
    for y, months in ((2021, 12), (2022, 12), (2023, 3)):
        if y - 23 <= by <= y - 19:
            m += months
    return m


DESIGNS = [("a_never", "never covered, born 1994-1997", range(1994, 1998)),
           ("b_ever", "ever covered, born 1998-2003", range(1998, 2004)),
           ("b2_ever_both_windows",
            "ever covered, in the panel in both windows, born 1998-2000",
            range(1998, 2001))]
DOSE = [("d00", range(1994, 1998)), ("d12", [1998]), ("d24", [1999]),
        ("d27", [2000, 2001])]

GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
WINDOWS = {"interim": ("2022-12", "2023-12"), "later": ("2024-01", "2025-06")}
CELL_CATS = [f"b{b}" for b in range(FIRST_COHORT, 2004)] + list(BANDS6)
MONTH_CATS = [f"{y}-{m:02d}" for y in range(2021, 2026)
              for m in range(1, 13 if y < 2025 else 7)]

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. The pull re-banded by age must reproduce Table 1 at 22-25",
    "  within 0.0005: adoption -0.0578 (0.0155), tau -0.0399 (0.0102);",
    "  a miss stops the script. Part S also needs the female differential",
    "  -0.0858 (0.0142) and tau -0.0714 (0.0109).",
    "  B1. THE DECLINE IS NOT THE PAYROLL REDUCTION if tau for the never-",
    "  covered cohorts (born 1994-1997) is negative and distinguishable",
    "  from zero at five per cent. THE PAYROLL RIVAL SURVIVES if the",
    "  ever-covered cohorts' tau is and the never-covered one is not.",
    "  B2. The dose carries no verdict: eligibility falls almost",
    "  monotonically with age in the panel, so a slope cannot be told",
    "  from an age gradient. Per-class taus are reported.",
    "  B3. The never-covered female differential is reported as it comes.",
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
    if list(j47.INCUMBENT_BANDS) != REF:
        raise RuntimeError(f"47j's incumbent bands are {j47.INCUMBENT_BANDS}"
                           f", not {REF}")
    return s82, s61, s67, s78, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def add(part, spec, term, coef, se, n_obs, n_firms, status="ok", vp=np.nan,
        vi=np.nan, cpi=np.nan, pm_interim=np.nan, pm_later=np.nan,
        cohorts=""):
    ROWS.append({"part": part, "spec": spec, "cohorts": cohorts,
                 "term": term,
                 "coef": float(coef) if coef == coef else np.nan,
                 "se": float(se) if se is not None and se == se else np.nan,
                 "t": tstat(coef, se), "var_post": vp, "var_interim": vi,
                 "cov_post_interim": cpi, "n_obs": n_obs, "n_firms": n_firms,
                 "young_person_months_interim": pm_interim,
                 "young_person_months_later": pm_later, "status": status})


def save() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    if df.empty:
        df.to_csv(OUT / "payroll_cohorts.csv", index=False)
        return df
    had = df["n_firms"].notna()
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = had & df["n_firms"].isna()
    if small.any():
        df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                       "cov_post_interim", "young_person_months_interim",
                       "young_person_months_later"]] = np.nan
    df.to_csv(OUT / "payroll_cohorts.csv", index=False)
    return df


def get(part, spec, term):
    for r in ROWS:
        if (r["part"], r["spec"], r["term"]) == (part, spec, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    """One Poisson fit; (coefficients, clustered vcov or None). A failure
    returns (None, None) and is recorded. R's stderr is written in full by
    mona_common._r_failed, never truncated here."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s96_{tag}", terms=terms,
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
    """(tau, se, V_pp, V_ii, V_pi); tau = post - interim, Var = V_pp + V_ii
    - 2 V_pi from the clustered covariance of the same fit."""
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


def person_months(b: pd.DataFrame, mask=None) -> tuple:
    """Young person-months in the interim and later windows of a fit's own
    panel (or of the rows in `mask`): sums over tens of thousands of
    employers."""
    ym = b["year_month"].astype(str)
    y = (b["young"] == 1) if mask is None else mask
    out = []
    for lo, hi in WINDOWS.values():
        out.append(float(b.loc[y & (ym >= lo) & (ym <= hi), "n_emp"].sum()))
    return tuple(out)


# ----------------------------------------------------------------------
# the cohort pull
# ----------------------------------------------------------------------

def cell_case(year: int) -> str:
    return f"""CASE
             WHEN fodelse >= {FIRST_COHORT}
                  THEN CONCAT('b', CAST(fodelse AS VARCHAR(4)))
             WHEN {year} - fodelse BETWEEN 22 AND 25 THEN '22-25'
             WHEN {year} - fodelse BETWEEN 26 AND 30 THEN '26-30'
             WHEN {year} - fodelse BETWEEN 31 AND 34 THEN '31-34'
             WHEN {year} - fodelse BETWEEN 35 AND 40 THEN '35-40'
             WHEN {year} - fodelse BETWEEN 41 AND 49 THEN '41-49'
             WHEN {year} - fodelse BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""


GENDER_CASE = """CASE WHEN LTRIM(RTRIM(gender)) IN ('1', '2')
             THEN LTRIM(RTRIM(gender)) ELSE '0' END"""


def q_counts_cohortsex(year: int, conn) -> pd.DataFrame:
    """47L's counts query with 67's sex column, and with every worker born
    1994 or later counted in a cell of his or her birth year. Birth year
    and sex from Individ 2023, 2021 or 2019, as in 47L and 67. Sex outside
    1 and 2 (or missing) is kept as '0', so the all-sex sum is 47L's
    count and the 1-and-2 sum is 67's."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               COALESCE(a.Kon, b.Kon, c.Kon) AS gender
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    cc = cell_case(year)
    q = f"""
    WITH base AS ({monthly})
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {cc} AS cell,
           {GENDER_CASE} AS gender,
           COUNT(DISTINCT person_id) AS n_emp
    FROM base
    WHERE fodelse IS NOT NULL AND {year} - fodelse BETWEEN 22 AND 69
    GROUP BY employer_id, period, {cc}, {GENDER_CASE}
    """
    return pd.read_sql(q, conn)


def load_pull(years) -> pd.DataFrame:
    """The cohort-by-sex counts, pulled once per year and cached. The cell
    set of every year is probed; a connection opens only if a year is
    missing, so a rerun needs no SQL."""
    out, conn = [], None
    for y in years:
        cf = CACHE / f"{PREFIX}_{y}.parquet"
        c = mc.read_cache(cf, require=PULL_COLS)
        if c is None:
            if conn is None:
                print("  opening a connection for the cohort pull")
                conn = mc.connect()
            t = time.time()
            c = q_counts_cohortsex(y, conn)
            mc.write_cache(c, cf)
            print(f"  cohort counts {y}: {len(c):,} cells "
                  f"({(time.time() - t) / 60:.1f} min)")
        else:
            print(f"  cohort counts {y}: cached ({len(c):,} cells)")
        cell = c["cell"].astype(str)
        gender = c["gender"].astype(str).str.strip()
        cells = set(cell.unique())
        # born 1994 to year-22 in cells of their own; nobody born 1994 or
        # later can sit in a band cell, so a 22-25 band cell means the
        # cohort rule did not apply
        allowed = {f"b{b}" for b in range(FIRST_COHORT, y - 21)} | set(BANDS6)
        extra = cells - allowed
        if extra or "22-25" in cells or f"b{FIRST_COHORT}" not in cells:
            raise RuntimeError(f"{cf.name}: unexpected cells {sorted(extra)}, "
                               f"a 22-25 band cell, or no b{FIRST_COHORT}; "
                               f"found {sorted(cells)}")
        if not set(gender.unique()) <= {"0", "1", "2"}:
            raise RuntimeError(f"{cf.name}: gender codes "
                               f"{sorted(gender.unique())}")
        # Fixed categories, so the years concatenate as categoricals: the
        # pull is some seventy million rows and three text columns of it
        # would hold over ten gigabytes beside R (failure class 4).
        c["cell"] = pd.Categorical(cell, categories=CELL_CATS)
        c["gender"] = pd.Categorical(gender, categories=["0", "1", "2"])
        c["year_month"] = pd.Categorical(c["year_month"].astype(str),
                                         categories=MONTH_CATS, ordered=True)
        if c["cell"].isna().any() or c["year_month"].isna().any():
            raise RuntimeError(f"{cf.name}: a cell or month outside the "
                               f"expected sets")
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return pd.concat(out, ignore_index=True)


def age_band_of(cell: pd.Series, year: pd.Series) -> pd.Series:
    """A cohort cell re-banded by age in the given year; a band cell stays."""
    out = cell.copy()
    is_c = cell.str.startswith("b")
    age = year[is_c] - cell[is_c].str.slice(1, 5).astype(int)
    lab = pd.Series(pd.NA, index=age.index, dtype="object")
    for k, (lo, hi) in BANDS6.items():
        lab[(age >= lo) & (age <= hi)] = k
    out[is_c] = lab
    return out


def rebanded(pull: pd.DataFrame, sex: bool) -> pd.DataFrame:
    """The pull in 47L's (sex=False, all sexes) or 67's (sex=True, sexes 1
    and 2) layout."""
    p = pull if not sex else pull[pull["gender"].isin(["1", "2"])]
    yr = p["year_month"].str.slice(0, 4).astype(int)
    d = p.assign(age_group=age_band_of(p["cell"], yr))
    keys = ["employer_id", "year_month", "age_group"] + (["gender"] if sex else [])
    return d.groupby(keys, observed=True)["n_emp"].sum().reset_index()


def compare(re_: pd.DataFrame, prefix: str, sex: bool, years) -> None:
    """The re-banded pull against the cache the paper's panels come from,
    cell by cell. Identifier dtype normalised on BOTH sides (failure
    class 3). A diagnostic; the fit gate decides."""
    keys = ["employer_id", "year_month", "age_group"] + (["gender"] if sex else [])
    bad_t, n_t = 0, 0
    for y in years:
        old = mc.read_cache(CACHE / f"{prefix}_{y}.parquet",
                            require=SEX_COLS if sex else COUNT_COLS)
        if old is None:
            NOTES.append(f"data check: {prefix}_{y} not on the share, skipped")
            continue
        new = re_[re_["year_month"].astype(str).str.slice(0, 4) == str(y)]
        # plain text keys on both sides: a categorical key grouped without
        # observed=True expands to every combination of its levels
        a = new.assign(employer_id=pd.to_numeric(new["employer_id"]).astype("int64"),
                       year_month=new["year_month"].astype(str),
                       age_group=new["age_group"].astype(str))
        b = old.assign(employer_id=pd.to_numeric(old["employer_id"]).astype("int64"),
                       year_month=old["year_month"].astype(str),
                       age_group=old["age_group"].astype(str))
        if sex:
            a = a.assign(gender=a["gender"].astype(str))
            b = b.assign(gender=b["gender"].astype(str).str.strip())
            b = b[b["gender"].isin(["1", "2"])]
        j = a.groupby(keys, observed=True)["n_emp"].sum().rename("new") \
            .to_frame().join(b.groupby(keys, observed=True)["n_emp"].sum()
                             .rename("old"), how="outer").fillna(0)
        bad = int((j["new"] != j["old"]).sum())
        bad_t += bad
        n_t += len(j)
        print(f"  data check {prefix} {y}: {len(j):,} cells, {bad:,} differ")
        del a, b, j
        gc.collect()
    msg = (f"data check: the cohort pull re-banded by age differs from "
           f"{prefix} in {bad_t:,} of {n_t:,} cells")
    NOTES.append(msg)
    print(f"  {msg}")


def cohort_frame(pull: pd.DataFrame, groups: dict, sex: bool) -> pd.DataFrame:
    """Cells for one design: every cohort cell mapped to its group label
    (groups: label -> birth years), the four reference bands kept, all
    else dropped; re-summed. The reference bands in the pull hold only
    workers born before 1994."""
    lut = {f"b{by}": lab for lab, bys in groups.items() for by in bys}
    p = pull if not sex else pull[pull["gender"].isin(["1", "2"])]
    lab = p["cell"].map(lambda c: lut.get(c, c if c in REF else None))
    d = p.assign(age_group=lab)
    d = d[d["age_group"].notna()]
    keys = ["employer_id", "year_month", "age_group"] + (["gender"] if sex else [])
    return d.groupby(keys, observed=True)["n_emp"].sum().reset_index()


# ----------------------------------------------------------------------
# skeletons and terms
# ----------------------------------------------------------------------

def skeleton_multi(counts: pd.DataFrame, young: list, j47,
                   from_ym: str) -> pd.DataFrame:
    """78's build_skeleton_bands with SEVERAL young cells: balanced over
    employer x cell x month and zero-filled, all-zero cells dropped,
    employers keeping a young cell and a reference band, integer keys for
    employer x month, employer x cell and month x cell. `young` = 1 on
    every young cell; `cell_k` columns flag each one."""
    bands = list(young) + REF
    p = counts[counts["age_group"].astype(str).isin(bands)]
    p = p[p["year_month"].astype(str) >= from_ym]
    p = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
         ["n_emp"].sum().reset_index())
    p["age_group"] = p["age_group"].astype(str)
    p["year_month"] = p["year_month"].astype(str)
    have = p.groupby("employer_id")["age_group"].agg(set)
    ys, rs = set(young), set(REF)
    keep = have[have.apply(lambda v: bool(v & ys) and bool(v & rs))].index
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
    bal["young"] = bal["age_group"].isin(ys).astype(int)
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"], sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def class_terms(b: pd.DataFrame, s78, classes: list) -> tuple:
    """Equation (2)'s six terms for each young class separately: 78's
    eq2_terms run on a class indicator in place of `young`."""
    terms = []
    for k in classes:
        keep_young = b["young"]
        b["young"] = (b["age_group"] == k).astype(int)
        b, t = s78.eq2_terms(b, "high", f"_{k}")
        b["young"] = keep_young
        terms += t
    return b, terms


def dose_terms(b: pd.DataFrame, s78, dose_years: dict) -> tuple:
    """Common Equation (2) terms x High x Young, plus the same terms x
    High x Young x dose (years of eligibility)."""
    b, t0 = s78.eq2_terms(b)
    b["high_dose"] = b["high"] * b["age_group"].map(dose_years).fillna(0.0)
    b, t1 = s78.eq2_terms(b, "high_dose", "dose")
    return b, t0 + t1


# ----------------------------------------------------------------------
# the gates
# ----------------------------------------------------------------------

def stock_gate(six, expo, s61, s78, j47) -> None:
    skel = s61.build_skeleton(six, "22-25", j47)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b, terms = s78.eq2_terms(b)
    g, v = fit(b, "gate_22_25", terms, j47.FES)
    del b
    gc.collect()
    c, s, vp, vi, cpi = tau(g, v)
    pc = float(g.loc[POST, "coef"]) if g is not None else np.nan
    ps = float(g.loc[POST, "se"]) if g is not None else np.nan
    n_obs = int(g["n_obs"].max()) if g is not None else -1
    add("G", "gate_22_25", "post", pc, ps, n_obs, n, cohorts="age 22-25")
    add("G", "gate_22_25", "tau", c, s, n_obs, n, "derived", vp, vi, cpi,
        cohorts="age 22-25")
    save()
    bad = []
    for key, (cc, ss) in (("post", (pc, ps)), ("tau", (c, s))):
        wc, ws = GATE[key]
        if not (abs(cc - wc) <= GATE_TOL and abs(ss - ws) <= GATE_TOL):
            bad.append(f"{key}: this run {cc:+.4f} ({ss:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f}), |diff| {abs(cc - wc):.4f} and "
                       f"{abs(ss - ws):.4f} against {GATE_TOL}")
    if bad:
        msg = "THE GATE FAILED. Nothing from this run is quotable. " + \
              "; ".join(bad)
        print(f"\n  {msg}")
        FAILURES.append(msg)
        write_summary()
        raise SystemExit("96: the gate failed; stopping before any other fit.")
    print(f"  THE GATE PASSES at 22-25: tau {c:+.4f} ({s:.4f})")


def sex_fit(frame, young_label, expo, s67, s78, j47, spec, cohorts):
    """67's sex specification with `young_label` the young cell; records
    the male step, the female differential and both taus."""
    skel = s67.build_skeleton_sex(frame, young_label, j47, "n_emp")
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append(f"S/{spec}/empty")
        return None, None
    n = int(b["employer_id"].nunique())
    pm = person_months(b)
    b, terms = s78.gender_eq2_terms(b)
    g, v = fit(b, spec, terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return None, None
    n_obs = int(g["n_obs"].max())
    for lab, t_ in (("male_post", POST), ("male_interim", INTERIM),
                    ("female_diff_post", FPOST),
                    ("female_diff_interim", FINTERIM)):
        if t_ in g.index:
            add("S", spec, lab, g.loc[t_, "coef"], g.loc[t_, "se"], n_obs, n,
                cohorts=cohorts)
    c, s, vp, vi, cpi = tau(g, v)
    add("S", spec, "male_tau", c, s, n_obs, n, "derived", vp, vi, cpi,
        *pm, cohorts=cohorts)
    c, s, vp, vi, cpi = tau(g, v, FPOST, FINTERIM)
    add("S", spec, "female_diff_tau", c, s, n_obs, n, "derived", vp, vi, cpi,
        *pm, cohorts=cohorts)
    save()
    return g, v


# ----------------------------------------------------------------------
# the parts
# ----------------------------------------------------------------------

def part_c(pull, expo, s61, s78, j47) -> None:
    for spec, label, bys in DESIGNS:
        print(f"\n  PART C, {spec}: {label}")
        fr = cohort_frame(pull, {"young": bys}, sex=False)
        skel = s78.build_skeleton_bands(fr, ["young"] + REF, "young", j47,
                                        s61.PANEL_FROM)
        del fr
        gc.collect()
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{spec}/empty")
            continue
        n = int(b["employer_id"].nunique())
        pm = person_months(b)
        b, terms = s78.eq2_terms(b)
        g, v = fit(b, spec, terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        n_obs = int(g["n_obs"].max())
        coh = f"born {min(bys)}-{max(bys)}"
        for lab, t_ in (("rb", "rb_x_high_x_young"), ("interim", INTERIM),
                        ("post", POST)):
            add("C", spec, lab, g.loc[t_, "coef"], g.loc[t_, "se"], n_obs, n,
                cohorts=coh)
        c, s, vp, vi, cpi = tau(g, v)
        add("C", spec, "tau", c, s, n_obs, n, "derived", vp, vi, cpi, *pm,
            cohorts=coh)
        save()
        print(f"  {spec}: tau {c:+.4f} ({s:.4f}) on {n:,} employers")


def part_d(pull, expo, s61, s78, j47) -> None:
    groups = {k: list(bys) for k, bys in DOSE}
    months = {k: eligible_months(min(bys)) for k, bys in DOSE}
    for k, bys in DOSE:
        ms = {eligible_months(b) for b in bys}
        if len(ms) != 1:
            raise RuntimeError(f"dose class {k} mixes eligibilities {ms}")
    NOTES.append("dose classes (months eligible): " + ", ".join(
        f"{k} {months[k]}" for k in groups))
    fr = cohort_frame(pull, groups, sex=False)
    b = skeleton_multi(fr, list(groups), j47, s61.PANEL_FROM)
    del fr
    gc.collect()
    b = s78.with_exposure(b, expo)
    if b.empty:
        FAILURES.append("D/empty")
        return
    n = int(b["employer_id"].nunique())
    print(f"\n  PART D, per class: {n:,} employers")
    b, terms = class_terms(b, s78, list(groups))
    g, v = fit(b, "dose_classes", terms, j47.FES)
    b = b.drop(columns=terms)
    if g is not None:
        n_obs = int(g["n_obs"].max())
        for k, bys in DOSE:
            p_ = f"post_x_high_{k}_x_young"
            i_ = f"interim_x_high_{k}_x_young"
            pm = person_months(b, b["age_group"] == k)
            c, s, vp, vi, cpi = tau(g, v, p_, i_)
            add("D", f"class_{k}", "tau", c, s, n_obs, n, "derived", vp, vi,
                cpi, *pm, cohorts=f"born {min(bys)}-{max(bys)}, "
                f"{months[k]} months eligible")
        save()
    dose_years = {k: months[k] / 12.0 for k in groups}
    b, terms = dose_terms(b, s78, dose_years)
    g, v = fit(b, "dose_linear", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return
    n_obs = int(g["n_obs"].max())
    c, s, vp, vi, cpi = tau(g, v)
    add("D", "dose_linear", "tau_at_zero_dose", c, s, n_obs, n, "derived",
        vp, vi, cpi, cohorts="born 1994-2001")
    c, s, vp, vi, cpi = tau(g, v, "post_x_highdose_x_young",
                            "interim_x_highdose_x_young")
    add("D", "dose_linear", "tau_per_year_eligible", c, s, n_obs, n,
        "derived", vp, vi, cpi, cohorts="born 1994-2001")
    save()


def part_s(pull, expo, s67, s78, j47) -> None:
    print("\n  PART S, the sex gate on the re-banded sex panel:")
    six = rebanded(pull, sex=True)
    g, v = sex_fit(six, "22-25", expo, s67, s78, j47, "sex_gate_22_25",
                   "age 22-25")
    del six
    gc.collect()
    if g is None:
        FAILURES.append("S/the sex gate fit did not come back; Part S stops")
        return
    bad = []
    for key, term in (("post", "female_diff_post"), ("tau", "female_diff_tau")):
        c, s = get("S", "sex_gate_22_25", term)
        wc, ws = SEX_GATE[key]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"{key}: this run {c:+.4f} ({s:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f})")
    if bad:
        FAILURES.append("S/the sex gate failed, so the never-covered sex fit "
                        "was not run: " + "; ".join(bad))
        print("  the sex gate FAILED; Part S stops")
        return
    print("  the sex gate passes")
    fr = cohort_frame(pull, {"young": range(1994, 1998)}, sex=True)
    sex_fit(fr, "young", expo, s67, s78, j47, "sex_never_1994_1997",
            "born 1994-1997")


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def sig_neg(c, s) -> bool:
    return bool(s == s and s > 0 and c < 0 and abs(c) >= SIG5 * s)


def cohort_ages() -> list:
    """Which ages each birth cohort occupies in the two windows, from the
    panel's age definition (calendar year minus birth year). No data."""
    L = ["COHORTS, ELIGIBILITY AND THE AGES THEY OCCUPY (age = year - birth"
         " year):"]
    for by in range(1994, 2005):
        a_int = f"{2022 - by}-{2023 - by}"
        a_lat = f"{2024 - by}-{2025 - by}"
        m = eligible_months(by)
        L.append(f"  born {by}: {m:2d} months eligible; interim ages "
                 f"{a_int}, later ages {a_lat}"
                 + ("  (never in the panel)" if 2025 - by < 22 else
                    "  (enters the panel at 22 only in the later window)"
                    if 2023 - by < 22 else ""))
    return L


def write_summary() -> None:
    L = ["THE YOUTH PAYROLL REDUCTION, BY BIRTH COHORT", "=" * 48, "",
         "SFS 2021:55: pay disbursed 1 Jan 2021 - 31 Mar 2023, to those who at",
         "the start of the year had turned 18 but not 23 (born Y-23 to Y-19);",
         "only the old-age pension contribution in Jun-Aug 2021 and 2022.",
         "Covered cohorts: 1998-2002 (2021), 1999-2003 (2022), 2000-2004",
         "(2023). Every worker aged 22-25 in 2024-25 was covered at some",
         "point; the never-covered young adults are born 1994-1997.", ""]
    L += cohort_ages() + [""]
    c, s = get("G", "gate_22_25", "tau")
    if c == c:
        L.append(f"GATE: 22-25 re-banded tau {c:+.4f} ({s:.4f}); Table 1 "
                 f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")
        L.append("")
    if any(r["part"] == "C" for r in ROWS):
        L.append("C. TAU BY COHORT GROUP, against 31-69 born 1993 or earlier:")
        for spec, label, _ in DESIGNS:
            c, s = get("C", spec, "tau")
            n = next((r["n_firms"] for r in ROWS if r["spec"] == spec
                      and r["term"] == "tau"), np.nan)
            pm = next((r for r in ROWS if r["spec"] == spec
                       and r["term"] == "tau"), {})
            if c == c:
                n = int(n) if n == n else -1
                L.append(f"  {label:<60} {c:+.4f} ({s:.4f}) t "
                         f"{tstat(c, s):+.2f}  {n:,} employers, young "
                         f"person-months {pm.get('young_person_months_interim', 0):,.0f}"
                         f" / {pm.get('young_person_months_later', 0):,.0f}")
        L.append("")
    if any(r["part"] == "D" for r in ROWS):
        L.append("D. THE DOSE (no verdict):")
        for r in ROWS:
            if r["part"] == "D":
                L.append(f"  {r['spec']:<14} {r['term']:<22} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  {r['cohorts']}")
        L.append("")
    if any(r["part"] == "S" for r in ROWS):
        L.append("S. THE FEMALE DIFFERENTIAL:")
        for spec in ("sex_gate_22_25", "sex_never_1994_1997"):
            for term in ("female_diff_tau", "female_diff_post", "male_tau"):
                c, s = get("S", spec, term)
                if c == c:
                    L.append(f"  {spec:<22} {term:<18} {c:+.4f} ({s:.4f})")
        L.append("")
    a, sa = get("C", "a_never", "tau")
    b, sb = get("C", "b_ever", "tau")
    if a == a:
        if sig_neg(a, sa):
            v = "B1: THE DECLINE IS NOT THE PAYROLL REDUCTION"
        elif b == b and sig_neg(b, sb):
            v = "B1: THE PAYROLL RIVAL SURVIVES"
        else:
            v = "B1: neither pattern is clean; both reported as they come"
        L += ["VERDICT:", f"  {v}", f"    never covered {a:+.4f} ({sa:.4f});"
              f" ever covered {b:+.4f} ({sb:.4f})", ""]
    L += [f"FITS: {DONE} of {PLANNED} attempted came back. A run far shorter "
          "than the estimate (2 to 3 hours) is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "96_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "96_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"96: THE PAYROLL REDUCTION BY BIRTH COHORT   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        pull = load_pull(s61.PANEL_YEARS)
        lo, hi = str(pull["year_month"].min()), str(pull["year_month"].max())
        print(f"  cohort pull: {len(pull):,} cells, {lo} to {hi}")
        if hi < "2025-06" or lo > s61.PANEL_FROM:
            raise RuntimeError(f"the pull spans {lo} to {hi}, not "
                               f"{s61.PANEL_FROM} to 2025-06")
        six = rebanded(pull, sex=False)
        compare(six, "L_counts", False, s61.PANEL_YEARS)
        if "S" in PARTS:
            compare(rebanded(pull, sex=True), "L_counts_sex", True,
                    s61.PANEL_YEARS)
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers")
        stock_gate(six, expo, s61, s78, j47)
        del six
        gc.collect()
        for part, fn, args in (("C", part_c, (pull, expo, s61, s78, j47)),
                               ("D", part_d, (pull, expo, s61, s78, j47)),
                               ("S", part_s, (pull, expo, s67, s78, j47))):
            if part not in PARTS:
                continue
            try:
                fn(*args)
            except BaseException as ex:
                if isinstance(ex, SystemExit):
                    raise
                print(f"  Part {part} FAILED ({type(ex).__name__}: {ex})")
                traceback.print_exc()
                FAILURES.append(f"{part}/{type(ex).__name__}: {ex}")
        drain(s78, "78")
    except SystemExit:
        mc.runlog("96_payroll_cohorts", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"96 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("96_payroll_cohorts", rc, (time.time() - T0) / 60)
    print("\n96 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

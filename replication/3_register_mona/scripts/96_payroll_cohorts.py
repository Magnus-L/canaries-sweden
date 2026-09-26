#!/usr/bin/env python3
"""
96_payroll_cohorts.py -- the reduced youth payroll contribution, examined
                         on FIXED, DISJOINT birth cohorts: a cohort-specific
                         exposure gradient for cohorts the reduction never
                         covered and for cohorts it covered.

======================================================================
  RUNS IN MONA (lane 37c, second stage, after 95). Output folder
  CANARIES_96_OUT (default output_96); parts CANARIES_96_PARTS (default
  CD; D, the dose, is last and optional). ONE new SQL pull: employer x
  month x cohort group x sex for the cohort groups below, with NO age
  filter, 2021-2025, cached as L_counts_cohortfix_YYYY (~3 min a year).
======================================================================

THE STATUTE (verified 25 Sep 2026 in the enacted text, SFS 2021:55 as
amended by SFS 2021:591 and 2022:240; Prop. 2020/21:83, 2020/21:202,
2021/22:97; data.riksdagen.se). On pay to persons who "vid aarets ingaang
har fyllt 18 men inte 23 aar", 19.73 per cent instead of 31.42 up to SEK
25,000 a month, on pay disbursed 1 January 2021 to 31 March 2023 (in force
6 February 2021, retroactively; the 1 April start was only the budget
bill's announcement, and Prop. 2025/26:66's summary of it is wrong); only
the 10.21 per cent pension contribution on pay disbursed June to August
2021 and 2022. Covered: born Y-23 to Y-19, so 1998-2002 in 2021,
1999-2003 in 2022, 2000-2004 in 2023. ELIGIBILITY HERE IS STATUTORY, by
birth cohort: never months actually employed or subsidised.

WHY FIXED COHORTS AND NOT THE AGE-BAND PANEL. Birth cohorts are not age
bands: those born 1994-1997 are 24-27 in 2021 and 28-31 in 2025, and the
1994 cohort would enter the paper's 31-69 reference in 2025. Filtering the
age-band panel by birth year therefore changes the reference population
during the window. This script builds the groups from birth years alone,
with no age filter, so no worker enters or leaves a group during the
window except by being employed or not (the outcome):
  ref       born 1956-1990: aged 31-65 in 2021 and 35-69 in 2025, the
            reference, fixed throughout
  nc        born 1994-1997: never covered
  ec        born 1998-2003: ever covered (the four classes below)
            e1998 (12 months eligible), e1999 (24), e2000_02 (27),
            e2003 (15)
Born 1991-1993 and outside 1956-2003 are in no group: the gap keeps the
young groups and the reference disjoint.

THE DESIGN. Equation (2)'s terms (script 78: tightening switch, interim
window, adoption step, three calendar-quarter terms), each x High x the
young group; employer-by-month, employer-by-group and month-by-group
effects; Poisson; clustered by employer; exposure script 82's reported
score. The group's national path, including cohorts ageing into
employment, is absorbed by the month-by-group effect. The later-minus-
interim contrast is labelled a COHORT-SPECIFIC EXPOSURE GRADIENT, not
tau: the young group is a cohort, not the paper's age band.

THE GATE (hard stop, before anything is varied): the paper's panel from
47L's L_counts with 82's score reproduces Table 1 at 22-25 within 0.0005:
adoption -0.0578 (0.0155), tau -0.0399 (0.0102). (Lane 37c's first stage,
95, gates as well; each script gates on its own.)

PART C. One fit with nc and ec as separate young cells beside ref, each
with its own terms: the two gradients and their difference (ec - nc),
with standard errors from the fit's covariance.
PART D (optional, LAST). The five classes nc, e1998, e1999, e2000_02,
e2003 as separate cells: a gradient per class; and one fit with common
terms plus the same terms x statutory years of eligibility: the change
in the gradient per year of eligibility. Eligibility falls almost
monotonically with birth year, so a slope cannot be told from a cohort
(age) gradient; it carries no verdict.

READ RULES, FIXED BEFORE THE RUN
  B1. THE PATTERN EXTENDS TO COHORTS THE REDUCTION NEVER COVERED if the
      nc gradient is negative and distinguishable from zero at five per
      cent. THE PAYROLL RIVAL IS NOT EXCLUDED if the ec gradient is and
      the nc gradient is not. Otherwise reported as it comes. This
      qualifies interpretation; it does not separate causes.
  B2. The dose carries no verdict.

EXPORT (output_96/)
  payroll_cohorts.csv  gradients with var_post, var_interim,
                       cov_post_interim, n_obs, n_firms and young
                       person-months; the ec - nc difference with its SE;
                       national person-months by group and month (the
                       groups' own paths, sums over all employers)
  96_summary.txt, 96_log.txt, vcov_s96_*.csv

IN THE PAPER
OA "The payroll reduction for young workers" (appendix_v3 ~l.1043-1060):
the dates corrected and the 22-23/24-25 age split replaced by the cohort
gradients; main_v3 ~l.180; response_v4 ~l.472 and ~l.475.

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
PARTS = os.environ.get("CANARIES_96_PARTS", "CD").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
SIG5 = 1.959963984540054
PREFIX = "L_counts_cohortfix"
PULL_COLS = ["employer_id", "year_month", "cell", "gender", "n_emp"]
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
REF = ["ref"]
CELLS = {"ref": (1956, 1990), "nc": (1994, 1997), "e1998": (1998, 1998),
         "e1999": (1999, 1999), "e2000_02": (2000, 2002),
         "e2003": (2003, 2003)}
EC = ["e1998", "e1999", "e2000_02", "e2003"]
GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
GATE_TOL = 0.0005
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
WINDOWS = {"interim": ("2022-12", "2023-12"), "later": ("2024-01", "2025-06")}

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. Table 1 at 22-25 within 0.0005 on the paper's panel:",
    "  -0.0578 (0.0155), tau -0.0399 (0.0102); a miss stops the script.",
    "  B1. The pattern extends to cohorts the reduction never covered if",
    "  the gradient for those born 1994-1997 is negative and",
    "  distinguishable at five per cent; the payroll rival is not excluded",
    "  if the gradient for 1998-2003 is and 1994-1997's is not.",
    "  B2. The dose (statutory years of eligibility) carries no verdict.",
    "  Labels: a cohort-specific exposure GRADIENT, not tau.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


def eligible_months(by: int) -> int:
    """Statutory months of eligibility by birth year, Jan 2021 - Mar 2023:
    covered in year Y if born Y-23 .. Y-19; 2023 counts three months."""
    m = 0
    for y, months in ((2021, 12), (2022, 12), (2023, 3)):
        if y - 23 <= by <= y - 19:
            m += months
    return m


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
    return s82, s61, s78, l47, l70, j47


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
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def contrast(g, v, post, interim) -> tuple:
    """(post - interim, se, V_pp, V_ii, V_pi) from the fit's covariance."""
    if g is None or post not in g.index or interim not in g.index:
        return (np.nan,) * 5
    c = float(g.loc[post, "coef"]) - float(g.loc[interim, "coef"])
    if v is None or post not in v.index or interim not in v.index:
        return c, np.nan, np.nan, np.nan, np.nan
    vp, vi = float(v.loc[post, post]), float(v.loc[interim, interim])
    cpi = float(v.loc[post, interim])
    var = vp + vi - 2.0 * cpi
    return c, (float(np.sqrt(var)) if var > 0 else np.nan), vp, vi, cpi


def lin_se(v, w: dict) -> float:
    """SE of sum(w_k * term_k) from the covariance."""
    if v is None or any(k not in v.index for k in w):
        return np.nan
    var = sum(w[a] * w[b] * float(v.loc[a, b]) for a in w for b in w)
    return float(np.sqrt(var)) if var > 0 else np.nan


def person_months(b: pd.DataFrame, mask) -> tuple:
    ym = b["year_month"].astype(str)
    return tuple(float(b.loc[mask & (ym >= lo) & (ym <= hi), "n_emp"].sum())
                 for lo, hi in WINDOWS.values())


# ----------------------------------------------------------------------
# the fixed-cohort pull
# ----------------------------------------------------------------------

def cell_case() -> str:
    whens = "\n".join(f"             WHEN fodelse BETWEEN {lo} AND {hi} "
                      f"THEN '{k}'" for k, (lo, hi) in CELLS.items())
    return f"CASE\n{whens}\n             ELSE NULL END"


GENDER_CASE = """CASE WHEN LTRIM(RTRIM(gender)) IN ('1', '2')
             THEN LTRIM(RTRIM(gender)) ELSE '0' END"""


def q_counts_cohortfix(year: int, conn) -> pd.DataFrame:
    """Counts by employer x month x cohort group x sex, the groups defined
    by birth year alone (Individ 2023, 2021, 2019, the panel's linkage)
    and NO age filter, so no worker enters or leaves a group during the
    window."""
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
    cc = cell_case()
    q = f"""
    WITH base AS ({monthly})
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           {cc} AS cell,
           {GENDER_CASE} AS gender,
           COUNT(DISTINCT person_id) AS n_emp
    FROM base
    WHERE fodelse BETWEEN 1956 AND 1990 OR fodelse BETWEEN 1994 AND 2003
    GROUP BY employer_id, period, {cc}, {GENDER_CASE}
    """
    return pd.read_sql(q, conn)


def load_pull(years) -> pd.DataFrame:
    """Pulled once per year and cached; every year's cell set probed. The
    sexes are summed on load (the sex split is not used by this script)."""
    out, conn = [], None
    for y in years:
        cf = CACHE / f"{PREFIX}_{y}.parquet"
        c = mc.read_cache(cf, require=PULL_COLS)
        if c is None:
            if conn is None:
                conn = mc.connect()
            t = time.time()
            c = q_counts_cohortfix(y, conn)
            mc.write_cache(c, cf)
            print(f"  fixed-cohort counts {y}: {len(c):,} cells "
                  f"({(time.time() - t) / 60:.1f} min)")
        else:
            print(f"  fixed-cohort counts {y}: cached ({len(c):,} cells)")
        cells = set(c["cell"].astype(str).str.strip().unique())
        if cells != set(CELLS):
            raise RuntimeError(f"{cf.name}: cells {sorted(cells)}, expected "
                               f"{sorted(CELLS)}")
        c = (c.assign(cell=c["cell"].astype(str).str.strip(),
                      year_month=c["year_month"].astype(str))
             .groupby(["employer_id", "year_month", "cell"], observed=True)
             ["n_emp"].sum().reset_index())
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    return pd.concat(out, ignore_index=True)


def group_paths(pull: pd.DataFrame) -> None:
    """National person-months by group and month: the groups' own paths,
    exported so a reader can see no group jumps (nobody enters or leaves
    by construction; only employment moves them)."""
    t = pull.groupby(["cell", "year_month"])["n_emp"].sum().reset_index()
    for _, r in t.iterrows():
        add("paths", r["cell"], r["year_month"], float(r["n_emp"]), np.nan,
            np.nan, np.nan, "national_person_months")
    save()


def frame(pull: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Cells mapped to the design's labels (label -> list of cells)."""
    lut = {c: lab for lab, cs in groups.items() for c in cs}
    d = pull.assign(age_group=pull["cell"].map(lut))
    d = d[d["age_group"].notna()]
    return (d.groupby(["employer_id", "year_month", "age_group"],
                      observed=True)["n_emp"].sum().reset_index())


def skeleton_multi(counts: pd.DataFrame, young: list, j47,
                   from_ym: str) -> pd.DataFrame:
    """78's build_skeleton_bands with several young cells beside the
    reference: balanced, zero-filled, all-zero cells dropped, employers
    keeping a young cell and the reference, integer keys."""
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
    """Equation (2)'s six terms for each young class separately."""
    terms = []
    for k in classes:
        keep_young = b["young"]
        b["young"] = (b["age_group"] == k).astype(int)
        b, t = s78.eq2_terms(b, "high", f"_{k}")
        b["young"] = keep_young
        terms += t
    return b, terms


def terms_of(k: str) -> tuple:
    return f"post_x_high_{k}_x_young", f"interim_x_high_{k}_x_young"


# ----------------------------------------------------------------------
# gate and parts
# ----------------------------------------------------------------------

def stock_gate(expo, s61, s78, j47) -> None:
    counts = None
    out = []
    for y in s61.PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet", require=COUNT_COLS)
        if c is None:
            raise RuntimeError(f"L_counts_{y} missing; run 47L")
        out.append(c)
    counts = pd.concat(out, ignore_index=True)
    b = s78.with_exposure(s61.build_skeleton(counts, "22-25", j47), expo)
    del counts, out
    gc.collect()
    n = int(b["employer_id"].nunique())
    b, terms = s78.eq2_terms(b)
    g, v = fit(b, "gate_22_25", terms, j47.FES)
    del b
    gc.collect()
    c, s, vp, vi, cpi = contrast(g, v, POST, INTERIM)
    pc = float(g.loc[POST, "coef"]) if g is not None else np.nan
    ps = float(g.loc[POST, "se"]) if g is not None else np.nan
    n_obs = int(g["n_obs"].max()) if g is not None else -1
    add("G", "gate_22_25", "post", pc, ps, n_obs, n, cohorts="age 22-25")
    add("G", "gate_22_25", "tau", c, s, n_obs, n, "derived", vp, vi, cpi,
        cohorts="age 22-25")
    save()
    bad = [f"{k}: this run {x:+.4f} ({y:.4f}), Table 1 {GATE[k][0]:+.4f} "
           f"({GATE[k][1]:.4f})" for k, (x, y) in
           (("post", (pc, ps)), ("tau", (c, s)))
           if not (abs(x - GATE[k][0]) <= GATE_TOL
                   and abs(y - GATE[k][1]) <= GATE_TOL)]
    if bad:
        FAILURES.append("THE GATE FAILED: " + "; ".join(bad))
        write_summary()
        raise SystemExit("96: the gate failed; stopping.")
    print(f"  THE GATE PASSES at 22-25: tau {c:+.4f} ({s:.4f})")


def part_c(pull, expo, s61, s78, j47) -> None:
    print("\n  PART C, never covered and ever covered beside a fixed reference:")
    fr = frame(pull, {"nc": ["nc"], "ec": EC, "ref": ["ref"]})
    b = skeleton_multi(fr, ["nc", "ec"], j47, s61.PANEL_FROM)
    del fr
    gc.collect()
    b = s78.with_exposure(b, expo)
    if b.empty:
        FAILURES.append("C/empty")
        return
    n = int(b["employer_id"].nunique())
    b, terms = class_terms(b, s78, ["nc", "ec"])
    g, v = fit(b, "gradient_nc_ec", terms, j47.FES)
    if g is None:
        return
    n_obs = int(g["n_obs"].max())
    for k, lab in (("nc", "never covered, born 1994-1997"),
                   ("ec", "ever covered, born 1998-2003")):
        p_, i_ = terms_of(k)
        c, s, vp, vi, cpi = contrast(g, v, p_, i_)
        pm = person_months(b, b["age_group"] == k)
        add("C", f"gradient_{k}", "gradient", c, s, n_obs, n, "derived", vp,
            vi, cpi, *pm, cohorts=lab)
    pn, inn = terms_of("nc")
    pe, ie = terms_of("ec")
    cn = get("C", "gradient_nc", "gradient")[0]
    ce = get("C", "gradient_ec", "gradient")[0]
    d = ce - cn
    se = lin_se(v, {pe: 1, ie: -1, pn: -1, inn: 1})
    add("C", "difference", "ec_minus_nc", d, se, n_obs, n, "derived",
        cohorts="1998-2003 minus 1994-1997")
    save()
    del b
    gc.collect()


def part_d(pull, expo, s61, s78, j47) -> None:
    classes = ["nc"] + EC
    months = {"nc": 0, "e1998": eligible_months(1998),
              "e1999": eligible_months(1999),
              "e2000_02": eligible_months(2000), "e2003": eligible_months(2003)}
    for by in (2001, 2002):
        if eligible_months(by) != months["e2000_02"]:
            raise RuntimeError("class e2000_02 mixes eligibilities")
    NOTES.append("D: statutory months eligible " + ", ".join(
        f"{k} {m}" for k, m in months.items()))
    print("\n  PART D (optional), the dose:")
    fr = frame(pull, {k: [k] for k in classes} | {"ref": ["ref"]})
    b = skeleton_multi(fr, classes, j47, s61.PANEL_FROM)
    del fr
    gc.collect()
    b = s78.with_exposure(b, expo)
    if b.empty:
        FAILURES.append("D/empty")
        return
    n = int(b["employer_id"].nunique())
    b, terms = class_terms(b, s78, classes)
    g, v = fit(b, "dose_classes", terms, j47.FES)
    b = b.drop(columns=terms)
    if g is not None:
        n_obs = int(g["n_obs"].max())
        for k in classes:
            c, s, vp, vi, cpi = contrast(g, v, *terms_of(k))
            pm = person_months(b, b["age_group"] == k)
            add("D", f"class_{k}", "gradient", c, s, n_obs, n, "derived", vp,
                vi, cpi, *pm, cohorts=f"{k}, {months[k]} months eligible")
        save()
    b["high_dose"] = b["high"] * b["age_group"].map(
        {k: m / 12.0 for k, m in months.items()}).fillna(0.0)
    b, t0 = s78.eq2_terms(b)
    b, t1 = s78.eq2_terms(b, "high_dose", "dose")
    g, v = fit(b, "dose_linear", t0 + t1, j47.FES)
    del b
    gc.collect()
    if g is None:
        return
    n_obs = int(g["n_obs"].max())
    c, s, vp, vi, cpi = contrast(g, v, POST, INTERIM)
    add("D", "dose_linear", "gradient_at_zero_dose", c, s, n_obs, n,
        "derived", vp, vi, cpi, cohorts="born 1994-2003")
    c, s, vp, vi, cpi = contrast(g, v, "post_x_highdose_x_young",
                                 "interim_x_highdose_x_young")
    add("D", "dose_linear", "gradient_per_year_eligible", c, s, n_obs, n,
        "derived", vp, vi, cpi, cohorts="born 1994-2003")
    save()


# ----------------------------------------------------------------------
# summary and main
# ----------------------------------------------------------------------

def sig_neg(c, s) -> bool:
    return bool(s == s and s > 0 and c < 0 and abs(c) >= SIG5 * s)


def cohort_ages() -> list:
    L = ["COHORTS, STATUTORY ELIGIBILITY AND AGES (age = year - birth year):"]
    for by in range(1994, 2005):
        L.append(f"  born {by}: {eligible_months(by):2d} months eligible; "
                 f"aged {2021 - by} in 2021, {2023 - by} in 2023, "
                 f"{2025 - by} in 2025")
    L.append("  reference born 1956-1990: aged 31-65 in 2021, 35-69 in 2025")
    return L


def write_summary() -> None:
    L = ["THE YOUTH PAYROLL REDUCTION ON FIXED BIRTH COHORTS", "=" * 50, "",
         "SFS 2021:55: pay disbursed 1 Jan 2021 - 31 Mar 2023, to those who at",
         "the start of the year had turned 18 but not 23 (born Y-23 to Y-19);",
         "only the pension contribution in Jun-Aug 2021 and 2022. Groups by",
         "birth year alone, no age filter: nobody enters or leaves a group.",
         ""] + cohort_ages() + [""]
    c, s = get("G", "gate_22_25", "tau")
    if c == c:
        L += [f"GATE: 22-25 tau {c:+.4f} ({s:.4f}); Table 1 "
              f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})", ""]
    if any(r["part"] == "C" for r in ROWS):
        L.append("C. COHORT-SPECIFIC EXPOSURE GRADIENTS (later minus interim),")
        L.append("   against the fixed reference born 1956-1990:")
        for r in ROWS:
            if r["part"] == "C":
                L.append(f"  {r['cohorts']:<34} {r['coef']:+.4f} "
                         f"({r['se']:.4f}) t {tstat(r['coef'], r['se']):+.2f}")
        L.append("")
    if any(r["part"] == "D" for r in ROWS):
        L.append("D. THE DOSE (optional; no verdict):")
        for r in ROWS:
            if r["part"] == "D":
                L.append(f"  {r['spec']:<14} {r['term']:<27} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  {r['cohorts']}")
        L.append("")
    a, sa = get("C", "gradient_nc", "gradient")
    b, sb = get("C", "gradient_ec", "gradient")
    if a == a:
        if sig_neg(a, sa):
            v = "B1: THE PATTERN EXTENDS TO COHORTS THE REDUCTION NEVER COVERED"
        elif b == b and sig_neg(b, sb):
            v = "B1: THE PAYROLL RIVAL IS NOT EXCLUDED"
        else:
            v = "B1: neither pattern is clean; both reported as they come"
        L += ["VERDICT:", f"  {v}", f"    never covered {a:+.4f} ({sa:.4f}); "
              f"ever covered {b:+.4f} ({sb:.4f})", ""]
    L += [f"FITS: {DONE} of {PLANNED} attempted came back."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "96_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def main() -> int:
    global T0
    mc.Tee(OUT / "96_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"96: THE PAYROLL REDUCTION ON FIXED BIRTH COHORTS   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s78, l47, l70, j47 = load_modules()
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        stock_gate(expo, s61, s78, j47)
        pull = load_pull(s61.PANEL_YEARS)
        group_paths(pull)
        for part, fn in (("C", part_c), ("D", part_d)):
            if part not in PARTS:
                continue
            try:
                fn(pull, expo, s61, s78, j47)
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

#!/usr/bin/env python3
"""
100_tipping_point.py -- the demographic-linkage accounting by exposure
                        group and period, and how much differential
                        missingness among young workers at exposed
                        employers it would take to move tau to zero.

======================================================================
  RUNS IN MONA (lane 38a). Output folder CANARIES_100_OUT (default
  output_100). SQL, read only, cached: T_unlinked_YYYY, 2021-2025, one
  query a year (distinct employer x month x person from the monthly
  declarations, the linkage once per PERSON; ~3-5 min a year, the shape
  of 99's raw rebuild). Everything else is a cache (47L, 67, 82).
======================================================================

QUESTION
The response letter's A2e placeholder, from the review of 25 Sep 2026:
"On the headline sample (the 104,217 employers) and contrast (later minus
interim period), by sex: the demographic-linkage accounting by exposure
group and period, and a tipping-point calculation, the differential
missingness among young workers at top-quartile employers that would move
tau to zero, with the extremal allocation of all unlinked person-months to
the young at exposed employers beside it. State the assumptions."

A worker enters the panel only with a birth year (pooled counts, 47L) and
a birth year and a sex coded 1 or 2 (sex counts, 67), read from Individ
2023, 2021 or 2019. A payslip whose worker has neither cannot be put in an
age band; if such workers were disproportionately young at exposed
employers after 2023, the headline would understate young employment
there exactly when it is measured to fall.

THE LINKAGE CATEGORIES (per PERSON, over every joined register row, so a
duplicated register row cannot put one person in two categories)
  no_register        in none of Individ 2023, 2021, 2019
  register_no_birth  in a register, no readable birth year
  birth_no_sex       a birth year, but no sex coded 1 or 2
Unlinked for the POOLED design = no_register + register_no_birth.
Unlinked for the SEX design    = those + birth_no_sex.
The declaration carries no age of its own, so an unlinked person-month
has no age band: its age is exactly what is unknown.

PART A. THE ACCOUNTING (no estimate)
On the headline panel's employers (the 22-25 gate panel, High = the top
quartile of 82's score), person-months of every age, by High vs other and
by period: pre-hike (2021-01 to 2022-03), tightening (2022-04 to
2022-11), interim (2022-12 to 2023-12), later (2024-01 to 2025-06); and
by month. Counts and shares of each category; employers; and, from the
panel itself, young (22-25) and all panel person-months in each group
and period, for scale.

PART T. THE TIPPING POINT (pooled 22-25; the female differential)
Let Y = the young person-months in High employers in the later period
(young women for the differential). The tipping point m* is the
proportional increase in exactly those cells that sets tau to zero, all
else unchanged. Poisson's multiplicative structure makes tau move by
about log(1 + m) (not exactly: the fixed effects re-fit), so:
  fit at k0 = exp(-tau);  secant in log k through (0, tau) and
  (log k0, tau(k0)) to k*;  a verifying fit at k*.
m* = k* - 1, and m* x Y is the number of young person-months that would
have to be missing, ONLY at exposed employers and ONLY in the later
period, beyond whatever is missing symmetrically. It is set against the
unlinked person-months actually present at High employers in the later
period (U_HL), and against two excesses:
  own-rate excess    U_HL - r_HI x A_HL   (r_HI: High's interim unlinked
                     share of all person-months; A_HL: High's later
                     person-months, all ages)
  differential excess U_HL - A_HL x (r_HI + r_OL - r_OI) (other employers'
                     change subtracted: the difference-in-differences in
                     missingness)
The ratio m* Y / U_HL is the share of ALL unlinked later-period
person-months at exposed employers that would have to be young workers
missing there alone. Above one, not even that allocation tips tau.

PART X. EXTREMAL ALLOCATIONS (refits)
  X_all   every unlinked person-month at each High employer-month of the
          interim and later periods, as they occur, added to that
          employer-month's young cell (young women's for the sex
          design); nothing elsewhere. The period difference in
          missingness is what moves tau.
  X_diff  the later-period excess only: at each High employer and later
          month, max(0, U_fm - r_fI x A_fm), r_fI the employer's own
          interim unlinked share; nothing in the interim, nothing at
          other employers.
Unlinked person-months at a High employer whose young cell is not in the
panel cannot be allocated; they are counted and reported, never dropped
silently.

ASSUMPTIONS (printed in the summary)
  1. Unlinked workers belong to the employer that declared them, in the
     month declared; the register failure is about the person.
  2. Missingness enters as young person-months added to High young cells;
     the older bands and other employers are held at their observed
     counts (the most adverse direction for tau).
  3. The tipping point is a counterfactual on the counts, not a model of
     who the unlinked are.

THE GATES (hard stop; each design's gate before its parts)
  Pooled: Table 1 at 22-25 within 0.0005: adoption -0.0578 (0.0155), tau
  -0.0399 (0.0102). Sex: the female differential -0.0858 (0.0142), tau
  -0.0714 (0.0109).

READ RULES, FIXED BEFORE THE RUN
  T1. MISSING DEMOGRAPHICS CANNOT TIP TAU if m* x Y exceeds U_HL, the
      unlinked person-months at exposed employers in the later period.
      Otherwise the summary gives the share of U_HL that would have to be
      young workers missing only there.
  T2. THE EXTREMAL ALLOCATION LEAVES TAU NEGATIVE if tau under X_all
      stays negative and distinguishable from zero at five per cent.
  T3, T4. The same two rules for the female differential.
  A verifying fit whose tau is further than a tenth of the gate's SE from
  zero is reported as such, and m* is then the secant estimate only.

EXPORT (output_100/)
  tipping_point.csv         gates, fits, m*, Y, U_HL, excesses, ratios;
                            employer counts below 5 suppressed with their
                            statistic
  unlinked_accounting.csv   group x period: employers, person-months by
                            category and shares; panel person-months
  unlinked_by_month.csv     group x month: the same
  unlinked_accounting_sex.csv, unlinked_by_month_sex.csv  the same on
                            the sex panel's employers
  100_summary.txt, 100_log.txt; vcov_s100_*.csv
Aggregates over tens of thousands of employers; no identifier leaves.

IN THE PAPER
Response letter A2e (replaces the PENDING box); OA tab:uncounted's
paragraph ("A tipping-point calculation on the headline sample and
contrast is reported in [PENDING ...]").

    python 100_tipping_point.py
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

OUT = HERE / os.environ.get("CANARIES_100_OUT", "output_100")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_100_PARTS", "PS").upper()   # P pooled, S sex
CACHE = mc.CACHE_DIR

FLOOR = 5
SIG5 = 1.959963984540054
BAND = "22-25"
GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005
VERIFY_SE = 0.10                     # a verifying tau within 0.1 SE of zero
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]
UNL_COLS = ["employer_id", "year_month", "n_all", "n_noreg", "n_nobirth",
            "n_nosex"]
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
POST_FROM, INTERIM_FROM = "2024-01", mc.CHATGPT_YM
PERIODS = ("pre_hike", "tightening", "interim", "later")

NOTES: list = []
FAILURES: list = []
EST: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

ASSUMPTIONS = [
    "ASSUMPTIONS:",
    "  1. Unlinked workers belong to the employer that declared them, in the",
    "     month declared.",
    "  2. Missingness enters as young person-months added to exposed young",
    "     cells; older bands and other employers stay as observed (the most",
    "     adverse direction for tau).",
    "  3. The tipping point is a counterfactual on the counts, not a model",
    "     of who the unlinked are. The declaration carries no age.",
]
READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Pooled: Table 1 at 22-25 within 0.0005, -0.0578 (0.0155), tau",
    "  -0.0399 (0.0102). Sex: -0.0858 (0.0142), tau -0.0714 (0.0109). A miss",
    "  stops the design it gates.",
    "  T1. Missing demographics cannot tip tau if m* x Y exceeds the unlinked",
    "  person-months at exposed employers in the later period (U_HL).",
    "  T2. The extremal allocation leaves tau negative if tau under X_all is",
    "  negative and distinguishable from zero at five per cent.",
    "  T3, T4. The same for the female differential.",
    f"  A verifying fit further than {VERIFY_SE} SE from zero is reported as",
    "  such, and m* is then the secant estimate only.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


# ----------------------------------------------------------------------
# plumbing (99's and 97's, unchanged in behaviour)
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
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}")
    return s82, s61, s67, s78, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def est(part, spec, term, coef, se=np.nan, n_obs=np.nan, n_firms=np.nan,
        status="ok", vp=np.nan, vi=np.nan, cpi=np.nan):
    EST.append({"part": part, "spec": spec, "term": term,
                "coef": float(coef) if coef == coef else np.nan,
                "se": float(se) if se == se else np.nan, "t": tstat(coef, se),
                "var_post": vp, "var_interim": vi, "cov_post_interim": cpi,
                "n_obs": n_obs, "n_firms": n_firms, "status": status})
    save_est()


def save_est() -> None:
    df = pd.DataFrame(EST)
    if not df.empty:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
        small = had & df["n_firms"].isna()
        df.loc[small, ["coef", "se", "t"]] = np.nan
    df.to_csv(OUT / "tipping_point.csv", index=False)


def get(part, spec, term):
    for r in EST:
        if (r["part"], r["spec"], r["term"]) == (part, spec, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    """One Poisson fit; (coefficients, clustered vcov or None)."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s100_{tag}", terms=terms,
                                fes=fes, cluster="employer_id")
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty or r["coef"].isna().all():
        FAILURES.append(tag)
        print(f"    {tag}: FAILED, recorded and skipped")
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def tau(g, v, post=POST, interim=INTERIM) -> tuple:
    """(tau, se, V_pp, V_ii, V_pi); Var = V_pp + V_ii - 2 V_pi."""
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


def fit_tau(b, tag, terms, fes, part, spec, n_firms, post, interim) -> tuple:
    """Fit and record post and tau for one pair of terms."""
    g, v = fit(b, tag, terms, fes)
    if g is None:
        return np.nan, np.nan
    n_obs = int(g["n_obs"].max())
    est(part, spec, "post", g.loc[post, "coef"], g.loc[post, "se"],
        n_obs, n_firms)
    c, s, vp, vi, cpi = tau(g, v, post, interim)
    est(part, spec, "tau", c, s, n_obs, n_firms, "derived", vp, vi, cpi)
    return c, s


def load_cache(prefix, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


def emp_int(s: pd.Series) -> pd.Series:
    """Employer ids as int64 on every side of every join (failure class
    3); a missing id becomes -1 and is counted by the caller."""
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64")


def period_of(ym: pd.Series) -> pd.Series:
    ym = ym.astype(str)
    return pd.Series(np.select(
        [ym < mc.RIKSBANK_YM, ym < INTERIM_FROM, ym < POST_FROM],
        ["pre_hike", "tightening", "interim"], "later"), index=ym.index)


# ----------------------------------------------------------------------
# the linkage pull
# ----------------------------------------------------------------------

def agi_union(year: int, cols: str) -> str:
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    return "\nUNION ALL\n".join(
        f"SELECT {cols} FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix}"
        for m in range(1, max_month + 1))


def q_unlinked(year: int, conn) -> pd.DataFrame:
    """Every declared person-month of `year` (distinct employer, month,
    person), by employer and month, with the person's linkage category.
    The linkage is decided once per PERSON over every joined register row
    (MAX), so a duplicated register row cannot split one person across
    two categories. The column names are the ones 47L, 67 and 99 read."""
    u = agi_union(year, "P1207_LOPNR_PEORGNR AS employer_id, PERIOD AS "
                        "period, P1207_LOPNR_PERSONNR AS person_id")
    q = f"""
    WITH trip AS (SELECT DISTINCT employer_id, period, person_id FROM ({u}) x),
    pers AS (SELECT DISTINCT person_id FROM trip),
    pl AS (
        SELECT p.person_id,
               MAX(CASE WHEN a.P1207_LopNr_PersonNr IS NOT NULL
                          OR b.P1207_LopNr_PersonNr IS NOT NULL
                          OR c.P1207_LopNr_PersonNr IS NOT NULL
                        THEN 1 ELSE 0 END) AS in_reg,
               MAX(CASE WHEN COALESCE(TRY_CAST(a.FodelseAr AS INT),
                                      TRY_CAST(b.FodelseAr AS INT),
                                      TRY_CAST(c.FodelseAr AS INT)) IS NOT NULL
                        THEN 1 ELSE 0 END) AS has_birth,
               MAX(CASE WHEN LTRIM(RTRIM(COALESCE(a.Kon, b.Kon, c.Kon)))
                             IN ('1', '2') THEN 1 ELSE 0 END) AS has_sex
        FROM pers p
        LEFT JOIN dbo.Individ_2023 a ON p.person_id = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON p.person_id = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON p.person_id = c.P1207_LopNr_PersonNr
        GROUP BY p.person_id)
    SELECT t.employer_id,
           LEFT(t.period,4) + '-' + SUBSTRING(t.period,5,2) AS year_month,
           COUNT(*) AS n_all,
           SUM(CASE WHEN pl.in_reg = 0 THEN 1 ELSE 0 END) AS n_noreg,
           SUM(CASE WHEN pl.in_reg = 1 AND pl.has_birth = 0 THEN 1 ELSE 0 END)
               AS n_nobirth,
           SUM(CASE WHEN pl.has_birth = 1 AND pl.has_sex = 0 THEN 1 ELSE 0 END)
               AS n_nosex
    FROM trip t JOIN pl ON t.person_id = pl.person_id
    GROUP BY t.employer_id, t.period
    """
    return pd.read_sql(q, conn)


def unlinked(years) -> pd.DataFrame:
    out, conn = [], None
    for y in years:
        cf = CACHE / f"T_unlinked_{y}.parquet"
        c = mc.read_cache(cf, require=UNL_COLS)
        if c is None:
            if conn is None:
                conn = mc.connect()
            t = time.time()
            c = q_unlinked(y, conn)
            mc.write_cache(c, cf)
            print(f"  T_unlinked {y}: {len(c):,} employer-months "
                  f"({(time.time() - t) / 60:.1f} min)")
        else:
            print(f"  T_unlinked {y}: cached ({len(c):,} employer-months)")
        out.append(c)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    u = pd.concat(out, ignore_index=True)
    u["employer_id"] = emp_int(u["employer_id"])
    miss = int((u["employer_id"] < 0).sum())
    if miss:
        NOTES.append(f"linkage: {miss:,} employer-month rows without an "
                     f"employer id (cannot belong to a panel employer)")
    u["year_month"] = u["year_month"].astype(str).str.strip()
    for c in UNL_COLS[2:]:
        u[c] = pd.to_numeric(u[c], errors="coerce").fillna(0).astype("int64")
    u["u_pooled"] = u["n_noreg"] + u["n_nobirth"]
    u["u_sex"] = u["u_pooled"] + u["n_nosex"]
    return u[u["employer_id"] >= 0]


# ----------------------------------------------------------------------
# Part A: the accounting
# ----------------------------------------------------------------------

def accounting(u: pd.DataFrame, b: pd.DataFrame, tag: str) -> dict:
    """Group x period and group x month on the gate panel's employers
    (`tag` "" for the pooled panel, "_sex" for the sex panel, whose
    employer set is its own). Returns the aggregates Part T needs."""
    hmap = (b[["employer_id", "high"]].drop_duplicates("employer_id")
            .assign(employer_id=lambda d: emp_int(d["employer_id"]))
            .set_index("employer_id")["high"])
    d = u[u["employer_id"].isin(hmap.index)].copy()
    d["group"] = np.where(d["employer_id"].map(hmap) == 1, "high", "other")
    d["period"] = period_of(d["year_month"]).to_numpy()
    n_emp_u = d["employer_id"].nunique()
    NOTES.append(f"A{tag}: {n_emp_u:,} of {len(hmap):,} panel employers appear in "
                 f"the linkage pull (every panel employer declares someone; "
                 f"a shortfall is a key problem, not a data one)")
    if n_emp_u < 0.99 * len(hmap):
        FAILURES.append(f"A/only {n_emp_u:,} of {len(hmap):,} panel employers "
                        f"matched the linkage pull (failure class 3)")
    # panel person-months, young and all, for scale
    bp = b.assign(employer_id=emp_int(b["employer_id"]),
                  period=period_of(b["year_month"]).to_numpy(),
                  group=np.where(b["high"] == 1, "high", "other"))
    pan = bp.groupby(["group", "period"], observed=True).agg(
        panel_pm_22_69=("n0", "sum")).join(
        bp[bp["young"] == 1].groupby(["group", "period"], observed=True)
        .agg(panel_pm_young=("n0", "sum")))
    cols = ["n_all", "n_noreg", "n_nobirth", "n_nosex", "u_pooled", "u_sex"]
    rows = []
    for keys, name in ((["group", "period"], f"unlinked_accounting{tag}.csv"),
                       (["group", "year_month"], f"unlinked_by_month{tag}.csv")):
        t = d.groupby(keys, observed=True)[cols].sum()
        t["employers"] = d.groupby(keys, observed=True)["employer_id"].nunique()
        allg = d.groupby(keys[1:], observed=True)[cols].sum()
        allg["employers"] = d.groupby(keys[1:], observed=True)[
            "employer_id"].nunique()
        allg = pd.concat({"all": allg}, names=["group"])
        t = pd.concat([t, allg]).reset_index()
        for c in ("n_noreg", "n_nobirth", "n_nosex", "u_pooled", "u_sex"):
            t[f"share_{c}"] = t[c] / t["n_all"].where(t["n_all"] > 0)
        if keys[1] == "period":
            t = t.merge(pan.reset_index(), on=["group", "period"], how="left")
        t = mc.enforce_min_cell(t, count_col="employers", floor=FLOOR)
        small = t["employers"].isna()
        t.loc[small, [c for c in t.columns if c not in keys]] = np.nan
        t.to_csv(OUT / name, index=False)
        rows.append(t)
    per = rows[0].set_index(["group", "period"])
    return {"per": per, "d": d}


# ----------------------------------------------------------------------
# Part T and Part X on one design
# ----------------------------------------------------------------------

def allocate(b: pd.DataFrame, target: np.ndarray, add: pd.Series) -> tuple:
    """Values of `add` (indexed by employer_id, year_month) for the rows
    in `target`; returns (vector over b, allocated total)."""
    vec = np.zeros(len(b), dtype=float)
    rows = np.flatnonzero(target)
    idx = pd.MultiIndex.from_arrays(
        [emp_int(b["employer_id"].iloc[rows]).to_numpy(),
         b["year_month"].astype(str).iloc[rows].to_numpy()])
    vals = add.reindex(idx).fillna(0.0).to_numpy(float)
    vec[rows] = vals
    return vec, float(vals.sum())


def design(b, terms, fes, j47, part, post, interim, target_young, ucol,
           acct) -> None:
    """Tipping point and the two extremal allocations for one design.
    `target_young` marks the cells that receive (young, or young women);
    `ucol` is the unlinked column that design cannot count."""
    n = int(b["employer_id"].nunique())
    ym = b["year_month"].astype(str)
    hi = (b["high"] == 1).to_numpy()
    later = (ym >= POST_FROM).to_numpy()
    interim_m = ((ym >= INTERIM_FROM) & (ym < POST_FROM)).to_numpy()
    tgt_later = hi & target_young & later
    Y = float(b["n0"].to_numpy()[tgt_later].sum())
    tau0, se0 = get(part, "gate", "tau")

    # ---- the tipping point: secant in log k, then a verifying fit
    k0 = float(np.exp(-tau0))
    b["n_emp"] = b["n0"] * np.where(tgt_later, k0, 1.0)
    t1, s1 = fit_tau(b, f"{part}_k0", terms, fes, part, "scaled_k0", n,
                     post, interim)
    if t1 == t1 and t1 != tau0:
        lk = np.log(k0) - t1 * (np.log(k0) - 0.0) / (t1 - tau0)
        kstar = float(np.exp(lk))
    else:
        kstar = k0
    b["n_emp"] = b["n0"] * np.where(tgt_later, kstar, 1.0)
    t2, s2 = fit_tau(b, f"{part}_kstar", terms, fes, part, "scaled_kstar", n,
                     post, interim)
    mstar = kstar - 1.0
    ok = t2 == t2 and se0 == se0 and abs(t2) <= VERIFY_SE * se0
    NOTES.append(f"T/{part}: k0 {k0:.5f} gives tau {t1:+.5f}; secant k* "
                 f"{kstar:.5f} gives tau {t2:+.5f} "
                 f"({'verified' if ok else 'NOT within ' + str(VERIFY_SE) + ' SE of zero; m* is the secant estimate only'})")
    est(part, "tipping", "m_star", mstar, status="derived")
    est(part, "tipping", "k0", k0, status="derived")
    est(part, "tipping", "tau_at_k0", t1, s1, status="derived")
    est(part, "tipping", "tau_at_kstar", t2, s2, status="derived")
    est(part, "tipping", "young_pm_high_later_Y", Y, status="count")
    need = mstar * Y
    est(part, "tipping", "young_pm_needed", need, status="derived")

    # ---- set against the unlinked
    per = acct["per"]

    def g(group, period, col):
        try:
            return float(per.loc[(group, period), col])
        except KeyError:
            return np.nan
    U_HL, A_HL = g("high", "later", ucol), g("high", "later", "n_all")
    U_HI, A_HI = g("high", "interim", ucol), g("high", "interim", "n_all")
    U_OL, A_OL = g("other", "later", ucol), g("other", "later", "n_all")
    U_OI, A_OI = g("other", "interim", ucol), g("other", "interim", "n_all")
    r_HI, r_OL, r_OI = U_HI / A_HI, U_OL / A_OL, U_OI / A_OI
    exc_own = U_HL - r_HI * A_HL
    exc_diff = U_HL - A_HL * (r_HI + r_OL - r_OI)
    for term, val in (("U_high_later", U_HL), ("A_high_later", A_HL),
                      ("share_unlinked_high_interim", r_HI),
                      ("share_unlinked_other_later", r_OL),
                      ("share_unlinked_other_interim", r_OI),
                      ("excess_own_rate", exc_own),
                      ("excess_differential", exc_diff),
                      ("needed_over_U_high_later", need / U_HL
                       if U_HL else np.nan),
                      ("needed_over_excess_own", need / exc_own
                       if exc_own and exc_own > 0 else np.nan),
                      ("needed_over_excess_diff", need / exc_diff
                       if exc_diff and exc_diff > 0 else np.nan)):
        est(part, "tipping", term, val,
            status="count" if term.startswith(("U_", "A_")) else "derived")

    # ---- X_all: every unlinked person-month, interim and later, High
    d = acct["d"]
    dh = d[d["group"] == "high"]
    add = dh.set_index(["employer_id", "year_month"])[ucol].astype(float)
    tgt = hi & target_young & (later | interim_m)
    vec, got = allocate(b, tgt, add)
    have = float(dh.loc[dh["year_month"] >= INTERIM_FROM, ucol].sum())
    NOTES.append(f"X/{part}/all: {got:,.0f} of {have:,.0f} unlinked "
                 f"person-months at exposed employers (interim and later) "
                 f"found a young cell in the panel; the rest cannot be "
                 f"allocated")
    est(part, "x_all", "allocated_pm", got, status="count")
    est(part, "x_all", "unallocatable_pm", have - got, status="count")
    b["n_emp"] = b["n0"] + vec
    fit_tau(b, f"{part}_xall", terms, fes, part, "x_all", n, post, interim)

    # ---- X_diff: later excess over the employer's own interim share
    di = dh[dh["year_month"].between(INTERIM_FROM, "2023-12")]
    r_f = (di.groupby("employer_id")[ucol].sum()
           / di.groupby("employer_id")["n_all"].sum().replace(0, np.nan)
           ).fillna(0.0)
    dl = dh[dh["year_month"] >= POST_FROM].copy()
    dl["exc"] = (dl[ucol] - dl["employer_id"].map(r_f).fillna(0.0)
                 * dl["n_all"]).clip(lower=0.0)
    add2 = dl.set_index(["employer_id", "year_month"])["exc"]
    vec2, got2 = allocate(b, hi & target_young & later, add2)
    NOTES.append(f"X/{part}/diff: {got2:,.0f} of {float(dl['exc'].sum()):,.0f} "
                 f"excess later-period unlinked person-months allocated")
    est(part, "x_diff", "allocated_pm", got2, status="count")
    b["n_emp"] = b["n0"] + vec2
    fit_tau(b, f"{part}_xdiff", terms, fes, part, "x_diff", n, post, interim)
    b["n_emp"] = b["n0"]


def run_pooled(counts, expo, u, s61, s78, j47) -> None:
    print("\n  POOLED GATE at 22-25:")
    b = s78.with_exposure(s61.build_skeleton(counts, BAND, j47), expo)
    if b.empty:
        raise RuntimeError("the pooled gate's panel is empty")
    b, terms = s78.eq2_terms(b)
    b["n0"] = b["n_emp"].astype(float)
    n = int(b["employer_id"].nunique())
    p, ps = fit_tau(b, "P_gate", terms, j47.FES, "P", "gate", n, POST,
                    INTERIM)
    pp, pps = get("P", "gate", "post")
    bad = [f"{k}: this run {x:+.4f} ({y:.4f}), Table 1 {GATE[k][0]:+.4f} "
           f"({GATE[k][1]:.4f})" for k, (x, y) in
           (("post", (pp, pps)), ("tau", (p, ps)))
           if not (abs(x - GATE[k][0]) <= GATE_TOL
                   and abs(y - GATE[k][1]) <= GATE_TOL)]
    if bad:
        FAILURES.append("THE POOLED GATE FAILED: " + "; ".join(bad))
        return
    print(f"  THE POOLED GATE PASSES: tau {p:+.4f} ({ps:.4f})")
    acct = accounting(u, b, "")
    design(b, terms, j47.FES, j47, "P", POST, INTERIM,
           (b["young"] == 1).to_numpy(), "u_pooled", acct)


def run_sex(sex, expo, u, s67, s78, j47) -> None:
    print("\n  SEX GATE at 22-25:")
    b = s78.with_exposure(s67.build_skeleton_sex(sex, BAND, j47, "n_emp"),
                          expo)
    if b.empty:
        raise RuntimeError("the sex gate's panel is empty")
    b, terms = s78.gender_eq2_terms(b)
    b["n0"] = b["n_emp"].astype(float)
    n = int(b["employer_id"].nunique())
    p, ps = fit_tau(b, "S_gate", terms, j47.FES, "S", "gate", n, FPOST,
                    FINTERIM)
    pp, pps = get("S", "gate", "post")
    bad = [f"{k}: this run {x:+.4f} ({y:.4f}), Table 1 {SEX_GATE[k][0]:+.4f}"
           f" ({SEX_GATE[k][1]:.4f})" for k, (x, y) in
           (("post", (pp, pps)), ("tau", (p, ps)))
           if not (abs(x - SEX_GATE[k][0]) <= GATE_TOL
                   and abs(y - SEX_GATE[k][1]) <= GATE_TOL)]
    if bad:
        FAILURES.append("THE SEX GATE FAILED: " + "; ".join(bad))
        return
    print(f"  THE SEX GATE PASSES: tau {p:+.4f} ({ps:.4f})")
    # the sex panel's employer set is its own, so its accounting is too
    acct = accounting(u, b, "_sex")
    design(b, terms, j47.FES, j47, "S", FPOST, FINTERIM,
           ((b["young"] == 1) & (b["female"] == 1)).to_numpy(), "u_sex",
           acct)


# ----------------------------------------------------------------------
# summary and main
# ----------------------------------------------------------------------

def sig_neg(c, s) -> bool:
    return bool(s == s and s > 0 and c < 0 and abs(c) >= SIG5 * s)


def verdicts() -> list:
    L = []
    for part, lab, r1, r2 in (("P", "tau at 22-25", "T1", "T2"),
                              ("S", "the female differential", "T3", "T4")):
        need, _ = get(part, "tipping", "young_pm_needed")
        U, _ = get(part, "tipping", "U_high_later")
        m, _ = get(part, "tipping", "m_star")
        if need == need and U == U:
            ok = need > U
            L.append(f"  {r1} ({lab}): "
                     f"{'MISSING DEMOGRAPHICS CANNOT TIP IT' if ok else r1 + ' NOT MET'}"
                     f"; m* {m:.2%}, needed {need:,.0f} young person-months "
                     f"against {U:,.0f} unlinked at exposed employers in the "
                     f"later period ({need / U:.2f} times)")
        else:
            L.append(f"  {r1} ({lab}): NO VERDICT, a fit or the accounting "
                     f"is missing")
        c, s = get(part, "x_all", "tau")
        if c == c:
            L.append(f"  {r2} ({lab}): "
                     f"{'THE EXTREMAL ALLOCATION LEAVES IT NEGATIVE' if sig_neg(c, s) else r2 + ' NOT MET'}"
                     f"; tau under X_all {c:+.4f} ({s:.4f})")
        else:
            L.append(f"  {r2} ({lab}): NO VERDICT, the X_all fit is missing")
    return L


def write_summary() -> None:
    L = ["MISSING DEMOGRAPHICS: ACCOUNTING, TIPPING POINT, EXTREMAL",
         "ALLOCATION", "=" * 58, ""]
    for part, lab, gate in (("P", "pooled 22-25 tau", GATE),
                            ("S", "female differential tau", SEX_GATE)):
        c, s = get(part, "gate", "tau")
        if c == c:
            L.append(f"GATE {lab}: {c:+.4f} ({s:.4f}); Table 1 "
                     f"{gate['tau'][0]:+.4f} ({gate['tau'][1]:.4f})")
    for part, lab in (("P", "POOLED (unlinked = no register or no birth year)"),
                      ("S", "SEX (unlinked = those or no sex 1/2)")):
        if not any(r["part"] == part and r["spec"] == "tipping" for r in EST):
            continue
        L += ["", f"{lab}:"]
        for term in ("U_high_later", "A_high_later",
                     "share_unlinked_high_interim",
                     "share_unlinked_other_interim",
                     "share_unlinked_other_later", "excess_own_rate",
                     "excess_differential", "young_pm_high_later_Y", "m_star",
                     "young_pm_needed", "needed_over_U_high_later",
                     "needed_over_excess_own", "needed_over_excess_diff"):
            c, _ = get(part, "tipping", term)
            if c == c:
                fmt = (f"{c:.4%}" if term.startswith(("share", "m_star"))
                       else f"{c:,.2f}" if term.startswith("needed_over")
                       else f"{c:,.0f}")
                L.append(f"  {term:<32} {fmt}")
        for spec in ("scaled_k0", "scaled_kstar", "x_all", "x_diff"):
            c, s = get(part, spec, "tau")
            if c == c:
                L.append(f"  tau {spec:<14} {c:+.5f} ({s:.5f})")
    L += ["", "VERDICTS:"] + verdicts()
    L += ["", f"FITS: {DONE} of {PLANNED} attempted came back. A run far "
          "shorter than the estimate (2.5 to 3 hours) is a failure."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + ASSUMPTIONS + [""] + READ_RULES + [
        "", f"Runtime {(time.time() - T0) / 60:.1f} min. " + mc.mem_line("")]
    (OUT / "100_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def run_part(name, fn, *args) -> None:
    try:
        fn(*args)
    except BaseException as ex:
        if isinstance(ex, SystemExit):
            raise
        print(f"  Part {name} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(f"{name}/{type(ex).__name__}: {ex}")


def main() -> int:
    global T0
    mc.Tee(OUT / "100_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"100: TIPPING POINT   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        u = unlinked(s61.PANEL_YEARS)
        if "P" in PARTS:
            counts = load_cache("L_counts", s61.PANEL_YEARS, COUNT_COLS)
            if counts is None:
                raise RuntimeError("L_counts_2021-2025 missing; run 47L")
            run_part("pooled", run_pooled, counts, expo, u, s61, s78, j47)
            del counts
            gc.collect()
        if "S" in PARTS:
            sex = load_cache("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
            if sex is None:
                raise RuntimeError("L_counts_sex_2021-2025 missing; run 67")
            run_part("sex", run_sex, sex, expo, u, s67, s78, j47)
            del sex
            gc.collect()
        drain(s78, "78")
    except BaseException as ex:
        if isinstance(ex, SystemExit):
            raise
        print(f"100 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save_est()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("100_tipping_point", rc, (time.time() - T0) / 60)
    print("\n100 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

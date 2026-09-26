#!/usr/bin/env python3
"""
102_month_of_year.py -- the deferred month-of-year seasonality check
                        (lane 38c): tau and the female differential with
                        eleven month-of-year terms in place of the three
                        calendar-quarter terms, under both clusterings.

======================================================================
  RUNS IN MONA (lane 38c). Output folder CANARIES_102_OUT (default
  output_102). SQL: none. Every frame is a cache lane 37b left on the
  share: 47L's L_counts_2021-2025, 67's L_counts_sex_2021-2025, 82's
  score caches and 80's I_industry_key. If I_industry_key is gone, 80
  pulls it (about an hour of SQL) and the summary says so.
======================================================================

QUESTION (the response letter, R1.6, box "MONA, deferred")
Equation (2) removes the calendar cycle with three quarter-of-year terms
(fourth quarter omitted). Calendar-quarter terms leave within-quarter
fluctuation, and the interim period, December 2022 to December 2023,
holds an extra December. Does tau move when the cycle is removed month
by month instead? The same for the female differential.

THE OBJECT
tau = b_L - b_I, the later-period coefficient minus the interim one, from
one fit, Var = V_LL + V_II - 2 V_LI from the clustered covariance. The
female differential's tau is the same contrast on the x High x Young x
Female terms of the sex specification.

THE DESIGN
The gate fits Equation (2) exactly as script 78/97 build it (three
quarter terms, Q4 omitted). The month-of-year fit replaces those three
terms by eleven month-of-year indicators, DECEMBER OMITTED, as script
68's monthly shape did; everything else is identical: the same panel,
the same employer x age, employer x month and age x month effects, the
tightening switch, the interim and later terms, Poisson. Each fit is
run clustered by employer and by three-digit industry (80's key,
employers without a code sharing one residual cluster, as 97's Part P).

THE GATES (a miss is a hard stop; nothing from the run is quotable)
  Stock, 22-25: Table 1 within 0.0005 on coefficient and SE, later
  -0.0578 (0.0155), tau -0.0399 (0.0102).
  Sex, 22-25: the female differential -0.0858 (0.0142), tau -0.0714
  (0.0109).

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  M1. tau under month-of-year terms is reported beside the gate's tau,
      with both clusterings' standard errors, and the movement is stated
      in units of the gate's employer-clustered SE ("moves by x SE").
      No verdict beyond that: the two fits share their sample and their
      noise, and the SE of their difference is not available from two
      separate covariances, which the summary says.
  M2. The same for the female differential.
  The eleven month-of-year coefficients are exported, so the cycle the
  quarter terms leave can be seen, not asserted.
  Employer counts below five are suppressed with their statistic.

EXPORT (output_102/)
  month_of_year.csv   every reported term and tau, with var_post,
                      var_interim, cov_post_interim, n_obs and n_firms
  102_summary.txt, 102_log.txt; vcov_s102_*.csv stay on the share

IN THE PAPER
The response letter, R1.6 (the deferred box) and OA III.2, one sentence
on whether monthly seasonality moves tau or the female differential.

    python 102_month_of_year.py
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

OUT = HERE / os.environ.get("CANARIES_102_OUT", "output_102")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
POST_FROM = "2024-01"                       # asserted against 78 below
MONTH_REF = 12                              # December omitted, as 68's monthly shape
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]

GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
CLUSTERS = (("employer_id", ""), ("cl_ind", "_indcl"))

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Stock 22-25: Table 1 within 0.0005, later -0.0578 (0.0155),",
    "  tau -0.0399 (0.0102). Sex 22-25: -0.0858 (0.0142), tau -0.0714",
    "  (0.0109). A miss is a hard stop; nothing from the run is quotable.",
    "  M1. tau with eleven month-of-year terms (December omitted) in place",
    "  of the three quarter terms is reported beside the gate's tau, with",
    "  both clusterings, and the movement is stated in units of the gate's",
    "  employer-clustered SE. No verdict beyond 'moves by x SE'. The SE of",
    "  the difference is not available from two separate covariances, and",
    "  the summary says so.",
    "  M2. The same for the female differential.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


# ----------------------------------------------------------------------
# plumbing (as 97)
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
    s80.OUT, s80.CACHE = OUT, CACHE
    s73 = _mod("73_industry_and_credit.py", "s73")
    s73.OUT = OUT
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}")
    return s82, s61, s67, s73, s78, s80, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def add(part, spec, band, term, coef, se, n_obs, n_firms, status="ok",
        vp=np.nan, vi=np.nan, cpi=np.nan):
    ROWS.append({"part": part, "spec": spec, "young_band": band,
                 "term": term,
                 "coef": float(coef) if coef == coef else np.nan,
                 "se": float(se) if se is not None and se == se else np.nan,
                 "t": tstat(coef, se), "var_post": vp, "var_interim": vi,
                 "cov_post_interim": cpi, "n_obs": n_obs, "n_firms": n_firms,
                 "status": status})


def save() -> pd.DataFrame:
    df = pd.DataFrame(ROWS)
    if df.empty:
        df.to_csv(OUT / "month_of_year.csv", index=False)
        return df
    had = df["n_firms"].notna()
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = had & df["n_firms"].isna()
    if small.any():
        df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                       "cov_post_interim"]] = np.nan
    df.to_csv(OUT / "month_of_year.csv", index=False)
    return df


def get(part, spec, band, term):
    for r in ROWS:
        if (r["part"], r["spec"], r["young_band"], r["term"]) == \
                (part, spec, band, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """One Poisson fit; (coefficients, clustered vcov or None). A failure
    returns (None, None) and is recorded; R's stderr is written in full by
    mona_common._r_failed."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms, cluster {cluster}{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s102_{tag}", terms=terms,
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
    missing = [x for x in terms if x not in g.index]
    if missing:
        NOTES.append(f"{tag}: {len(missing)} terms absent from the fit "
                     f"({', '.join(missing[:4])})")
    DONE += 1
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def tau(g, v, post=POST, interim=INTERIM) -> tuple:
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


def record(g, v, part, spec, band, n_firms, pairs, extra_terms=()) -> None:
    """pairs: (label, post term, interim term) whose tau is recorded with
    the two coefficients; extra_terms are recorded as they are (the
    month-of-year coefficients)."""
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for lab, p_, i_ in pairs:
        for suffix, t_ in (("post", p_), ("interim", i_)):
            if t_ in g.index:
                add(part, spec, band, f"{lab}_{suffix}", g.loc[t_, "coef"],
                    g.loc[t_, "se"], n_obs, n_firms,
                    str(g.loc[t_].get("status", "ok")))
        c, s, vp, vi, cpi = tau(g, v, p_, i_)
        add(part, spec, band, f"{lab}_tau", c, s, n_obs, n_firms, "derived",
            vp, vi, cpi)
    for t_ in extra_terms:
        if t_ in g.index:
            add(part, spec, band, t_, g.loc[t_, "coef"], g.loc[t_, "se"],
                n_obs, n_firms, str(g.loc[t_].get("status", "ok")))
    save()


def check(label: str, got: dict, want: dict) -> list:
    bad = []
    for key in ("post", "tau"):
        c, s = got[key]
        wc, ws = want[key]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"{label} {key}: this run {c:+.4f} ({s:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f}), |diff| {abs(c - wc):.4f} and "
                       f"{abs(s - ws):.4f} against {GATE_TOL}")
    return bad


def stop(bad: list, what: str) -> None:
    msg = f"THE {what} GATE FAILED. Nothing from this run is quotable. " + \
          "; ".join(bad)
    print(f"\n  {msg}")
    FAILURES.append(msg)
    write_summary()
    raise SystemExit(f"102: the {what.lower()} gate failed; stopping.")


def load_counts(prefix: str, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# the month-of-year terms
# ----------------------------------------------------------------------

def month_terms(b: pd.DataFrame, sex: bool = False) -> tuple:
    """Equation (2) with the three quarter terms replaced by eleven
    month-of-year indicators (December omitted): the tightening switch,
    the interim and later terms, each x High x Young (and, on the sex
    panel, x High x Female and x High x Young x Female as in 78's
    gender_eq2_terms). Returns (frame, all terms, the month terms)."""
    ym = b["year_month"].astype(str)
    mo = ym.str.slice(5, 7).astype(int)
    post_any = ym >= mc.CHATGPT_YM
    periods = {"rb": (ym >= mc.RIKSBANK_YM).astype(int),
               "interim": (post_any & (ym < POST_FROM)).astype(int),
               "post": (ym >= POST_FROM).astype(int)}
    for mm in range(1, 13):
        if mm == MONTH_REF:
            continue
        periods[f"m{mm:02d}"] = (mo == mm).astype(int)
    hy = b["high"] * b["young"]
    terms, months = [], []
    if not sex:
        for p, ind in periods.items():
            col = f"{p}_x_high_x_young"
            b[col] = ind * hy
            terms.append(col)
            if p.startswith("m"):
                months.append(col)
        return b, terms, months
    hf = b["high"] * b["female"]
    hyf = hy * b["female"]
    for p, ind in periods.items():
        cols = [f"{p}_x_high_x_young", f"{p}_x_high_x_female",
                f"{p}_x_high_x_young_x_female"]
        b[cols[0]], b[cols[1]], b[cols[2]] = ind * hy, ind * hf, ind * hyf
        terms += cols
        if p.startswith("m"):
            months += cols
    return b, terms, months


def attach_industry(b: pd.DataFrame, s73, s80, tag: str) -> pd.DataFrame:
    """80's three-digit industry as a cluster column cl_ind; employers
    without a code share one residual cluster (97's Part P)."""
    key = s80.industry_key(s73)
    drain(s80, "80")
    kmap, src_map = s80.key_maps(key, s73)
    b, info = s80.attach_cluster(b, kmap, src_map, s73)
    NOTES.append(f"{tag}: industry clusters {info['n_clusters']:,}; "
                 f"{info['n_unresolved']:,} employers without a code share "
                 f"one residual cluster")
    return b


# ----------------------------------------------------------------------
# the stock panel: gate, then the month-of-year fit, both clusterings
# ----------------------------------------------------------------------

def stock(counts, expo, s61, s73, s78, s80, j47) -> None:
    print(f"\n  STOCK GATE at {BAND}:")
    skel = s61.build_skeleton(counts, BAND, j47)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b = attach_industry(b, s73, s80, "stock")
    b, qterms = s78.eq2_terms(b)
    g, v = fit(b, "gate_22_25", qterms, j47.FES)
    record(g, v, "G", "gate", BAND, n, [("hy", POST, INTERIM)],
           extra_terms=[t for t in qterms if t.startswith("q")])
    got = {"post": get("G", "gate", BAND, "hy_post"),
           "tau": get("G", "gate", BAND, "hy_tau")}
    bad = check(BAND, got, GATE)
    if bad:
        stop(bad, "STOCK")
    print(f"  THE STOCK GATE PASSES at {BAND}: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    # the gate's own specification under industry clustering, so both
    # clusterings come from this run
    g, v = fit(b, "gate_indcl_22_25", qterms, j47.FES, cluster="cl_ind")
    record(g, v, "G", "gate_indcl", BAND, n, [("hy", POST, INTERIM)])
    b = b.drop(columns=qterms)
    b, mterms, months = month_terms(b)
    for cl, sfx in CLUSTERS:
        print(f"\n  MONTH OF YEAR, stock, clustered by "
              f"{'employer' if sfx == '' else 'industry'}:")
        g, v = fit(b, f"moy{sfx}_22_25", mterms, j47.FES, cluster=cl)
        record(g, v, "M", f"month_of_year{sfx}", BAND, n,
               [("hy", POST, INTERIM)], extra_terms=months)
    del b
    gc.collect()


# ----------------------------------------------------------------------
# the sex panel: gate, then the month-of-year fit, both clusterings
# ----------------------------------------------------------------------

def sex(sexc, expo, s67, s73, s78, s80, j47) -> None:
    print(f"\n  SEX GATE at {BAND}:")
    skel = s67.build_skeleton_sex(sexc, BAND, j47, "n_emp")
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the sex gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b = attach_industry(b, s73, s80, "sex")
    b, qterms = s78.gender_eq2_terms(b)
    g, v = fit(b, "sex_gate_22_25", qterms, j47.FES)
    record(g, v, "G", "sex_gate", BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)],
           extra_terms=[t for t in qterms if t.startswith("q")])
    got = {"post": get("G", "sex_gate", BAND, "hyf_post"),
           "tau": get("G", "sex_gate", BAND, "hyf_tau")}
    bad = check("female differential", got, SEX_GATE)
    if bad:
        stop(bad, "SEX")
    print(f"  THE SEX GATE PASSES: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    g, v = fit(b, "sex_gate_indcl_22_25", qterms, j47.FES, cluster="cl_ind")
    record(g, v, "G", "sex_gate_indcl", BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    b = b.drop(columns=qterms)
    b, mterms, months = month_terms(b, sex=True)
    for cl, sfx in CLUSTERS:
        print(f"\n  MONTH OF YEAR, sex panel, clustered by "
              f"{'employer' if sfx == '' else 'industry'}:")
        g, v = fit(b, f"sex_moy{sfx}_22_25", mterms, j47.FES, cluster=cl)
        record(g, v, "M", f"sex_month_of_year{sfx}", BAND, n,
               [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)],
               extra_terms=months)
    del b
    gc.collect()


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def movement(label, gate_spec, moy_spec, term) -> list:
    """The comparison M1/M2 asks for: gate tau beside month-of-year tau,
    both clusterings, and the movement in gate SEs."""
    L = []
    c0, s0 = get("G", gate_spec, BAND, term)
    _, s0i = get("G", gate_spec + "_indcl", BAND, term)
    c1, s1 = get("M", moy_spec, BAND, term)
    _, s1i = get("M", moy_spec + "_indcl", BAND, term)
    if not (c0 == c0 and c1 == c1):
        return [f"  {label}: NO COMPARISON, a fit is missing (not a null)"]
    L.append(f"  {label}:")
    L.append(f"    quarter terms (Table 1)  tau {c0:+.4f}  SE {s0:.4f} by "
             f"employer, {s0i:.4f} by industry")
    L.append(f"    month-of-year terms      tau {c1:+.4f}  SE {s1:.4f} by "
             f"employer, {s1i:.4f} by industry")
    d = c1 - c0
    L.append(f"    difference (month minus quarter) {d:+.4f}, which is "
             f"{abs(d) / s0 if s0 else float('nan'):.2f} of the gate's "
             f"employer-clustered SE; its own SE is not available from two "
             f"separate covariances")
    return L


def cycle_lines(spec: str, suffix: str) -> list:
    L = []
    for r in ROWS:
        if r["part"] == "M" and r["spec"] == spec and \
                r["term"].startswith("m") and r["term"].endswith(suffix):
            L.append(f"    {r['term'][:3]}  {r['coef']:+.4f} ({r['se']:.4f})")
    return L


def write_summary() -> None:
    L = ["MONTH-OF-YEAR SEASONALITY: TAU AND THE FEMALE DIFFERENTIAL WITH",
         "ELEVEN MONTH-OF-YEAR TERMS IN PLACE OF THE THREE QUARTER TERMS",
         "=" * 66, "",
         "tau = later-period coefficient minus interim, SE from the clustered",
         "covariance of the same fit (V_LL + V_II - 2 V_LI). Month-of-year",
         f"terms x High x Young (and x Female on the sex panel), month "
         f"{MONTH_REF:02d} (December) omitted, as script 68's monthly shape.", "",
         "GATES (quarter terms, Table 1's specification):"]
    c, s = get("G", "gate", BAND, "hy_tau")
    if c == c:
        L.append(f"  stock {BAND}: tau {c:+.4f} ({s:.4f}); Table 1 "
                 f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")
    c, s = get("G", "sex_gate", BAND, "hyf_tau")
    if c == c:
        L.append(f"  sex {BAND}: female differential tau {c:+.4f} ({s:.4f}); "
                 f"Table 1 {SEX_GATE['tau'][0]:+.4f} ({SEX_GATE['tau'][1]:.4f})")
    L.append("")
    L.append("M. THE MOVEMENT (read rules M1 and M2):")
    L += movement("tau at 22-25 (High x Young)", "gate", "month_of_year",
                  "hy_tau")
    L += movement("the female differential at 22-25 (High x Young x Female)",
                  "sex_gate", "sex_month_of_year", "hyf_tau")
    L.append("")
    L.append("THE CYCLE THE QUARTER TERMS LEAVE: month-of-year coefficients")
    L.append("  x High x Young, stock panel (December = 0 by construction):")
    L += cycle_lines("month_of_year", "_x_high_x_young")
    L.append("  x High x Young x Female, sex panel:")
    L += cycle_lines("sex_month_of_year", "_x_high_x_young_x_female")
    L.append("")
    L += [f"FITS: {DONE} of {PLANNED} attempted came back. A run far shorter "
          "than the estimate (2 to 2.5 hours) is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "102_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

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
    mc.Tee(OUT / "102_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("102: MONTH-OF-YEAR SEASONALITY, TAU AND THE FEMALE DIFFERENTIAL")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s73, s78, s80, l47, l70, j47 = load_modules()
        if not all((CACHE / f"L_counts_sex_{y}.parquet").exists()
                   for y in s61.PANEL_YEARS):
            raise RuntimeError("L_counts_sex_* are not all on the share; this "
                               "script does no SQL for them (run 67)")
        if not (CACHE / "I_industry_key.parquet").exists():
            print("  WARNING: 80's industry key is not on the share; 80 will "
                  "pull it (about an hour of SQL)")
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers")
        # One panel at a time: Python never holds two while R fits.
        counts = load_counts("L_counts", s61.PANEL_YEARS, COUNT_COLS)
        if counts is None:
            raise RuntimeError("L_counts_2021-2025 missing; run 47L")
        last = str(counts["year_month"].max())
        if last < "2025-06":
            raise RuntimeError(f"the counts end at {last}")
        stock(counts, expo, s61, s73, s78, s80, j47)
        del counts
        gc.collect()
        sexc = load_counts("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
        if sexc is None:
            raise RuntimeError(f"L_counts_sex unreadable or lacking {SEX_COLS}")
        sex(sexc, expo, s67, s73, s78, s80, j47)
        del sexc
        gc.collect()
        drain(s78, "78")
        drain(s73, "73")
    except SystemExit:
        mc.runlog("102_month_of_year", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"102 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("102_month_of_year", rc, (time.time() - T0) / 60)
    print("\n102 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

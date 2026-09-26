#!/usr/bin/env python3
"""
105_prepath_placebo.py -- lane 39a: the pre-period from 2019, the backdated
                          placebo, and the flows clustered by industry.

======================================================================
  RUNS IN MONA (lane 39a). Output folder CANARIES_105_OUT (default
  output_105). SQL: none. Every frame is a cache already on the share:
  47L's L_counts_2019 to 2025 (2019 and 2020 were read by lane 25a and
  script 86 drew the 2019Q1 path from them), 54's flows_2021 to 2025,
  82's cascade and baseline caches and 80's I_industry_key. If
  L_counts_2019 or L_counts_2020 is missing the script STOPS and says so;
  it never pulls to extend the window.
======================================================================

QUESTION (the editor's read of 26 September 2026)
Table A (prepath) of the online appendix shows the young-to-older ratio
at exposed employers 0.07 to 0.14 log points above its 2022Q1 level in
2019 and 2020 at 22-25, and flat at 26-30. The paper's drift test runs
from January 2021 only. Two readers asked what the design returns on the
years before the window, and whether the separations result survives
industry clustering. Three parts, each pre-specified.

  G  THE GATE. Table 1's fit at 22-25 (DAIOE, employer clustering) within
     0.0005 on coefficient and SE: later -0.0578 (0.0155), tau -0.0399
     (0.0102). A miss is a hard stop; nothing from the run is quotable.

  D  THE DRIFT TEST ON THE LONGER PRE-PERIOD. 78's drift terms (the three
     calendar-quarter terms, the tightening window April to November 2022,
     and a linear trend per month, each x High x Young) on the pre-launch
     months January 2019 to November 2022 at 22-25 and 26-30, and January
     2020 to November 2022 at 22-25. Reported: the trend and its SE, FLAT
     or NOT FLAT under 78's two-SE rule, and the drift carried over the
     15.5 months between the calendar midpoints of the interim and later
     periods (the scale the appendix uses), beside the paper's own
     January 2021 drift (+0.000383 (0.000773) at 22-25, lane 29b/86).

  B  THE BACKDATED PLACEBO (the device of Facius et al.). Equation (2)
     with every boundary moved back by S months and the panel ending in
     November 2022, so no month after the launch enters and the placebo
     "adoption" falls where generative AI cannot be the cause.
       S = 24: reference Jan 2019-Mar 2020, "tightening" Apr-Nov 2020,
               "interim" Dec 2020-Dec 2021, "later" Jan-Nov 2022.
       S = 36: reference Jan-Mar 2019 (the caches start in 2019),
               "tightening" Apr-Nov 2019, "interim" Dec 2019-Dec 2020,
               "later" Jan 2021-Nov 2022.
     At 22-25 both, at 26-30 S = 24. tau_placebo = later minus interim
     from the covariance of the fit, reported beside tau-hat in units of
     tau-hat's SE. The S = 24 later window contains the real tightening
     months, in which the paper's own tightening coefficient is positive
     (+0.016), so that placebo leans against a spurious decline; the
     S = 36 later window contains all of 2021 and 2022. Both are said in
     the summary.

  F  THE FLOWS CLUSTERED BY INDUSTRY. 82's hires and separations fits at
     22-25 (54's flows, Equation (2)'s terms) refitted with 80's
     three-digit industry as the cluster. The coefficients do not depend
     on the clustering, so their later-minus-interim tau must reproduce
     97's values (+0.0084 hires, +0.0300 separations) within 0.0005: that
     is the flows gate. The employer-clustered SEs (0.0342, 0.0144) are
     quoted from 97 beside the industry-clustered ones.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  D1. The trend is reported with FLAT or NOT FLAT under 78's rule and
      with the drift carried over 15.5 months; no verdict on tau-hat.
  B1. Each placebo tau is reported with its SE and in units of tau-hat's
      SE. A placebo tau of the order of tau-hat, whichever sign, means the
      2021-22 plateau is not established as the counterfactual by these
      data alone; a placebo near zero means the design returns no decline
      where none should be. No other verdict.
  F1. The industry-clustered SE is reported beside the employer-clustered
      one; the coefficient must be the paper's. No verdict.
  Employer counts below five are suppressed with their statistic.

EXPORT (output_105/)
  prepath_placebo.csv  every reported term and tau (part D, B), with
                       var_post, var_interim, cov_post_interim, n_obs,
                       n_firms
  flows_industry.csv   part F, the same columns plus outcome
  105_summary.txt, 105_log.txt; vcov_s105_*.csv stay on the share

IN THE PAPER
Section 3, the pre-launch paragraph (two sentences); OA III.2, the
pre-period paragraph and Table A (prepath), the flows note in Table A
(headline components); letter R1.6 and A2.

    python 105_prepath_placebo.py
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

OUT = HERE / os.environ.get("CANARIES_105_OUT", "output_105")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND, BAND2 = "22-25", "26-30"
POST_FROM = "2024-01"                       # asserted against 78 below
EXTENDED_FROM = "2019-01"                   # the earliest cached month
PRE_LAUNCH_END = mc.CHATGPT_YM              # every pre-period stops here
DRIFT_WINDOWS = [("2019-01", BAND), ("2019-01", BAND2), ("2020-01", BAND)]
SHIFTS = [(24, BAND), (36, BAND), (24, BAND2)]
CARRY_MONTHS = 15.5                         # interim-to-later midpoint gap (OA III.2)
DRIFT_SE = 2.0                              # 78's flat rule
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
FLOW_COLS = ["employer_id", "year_month", "age_group", "n_hire", "n_sep"]

GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
GATE_TOL = 0.0005
PRIOR_DRIFT = {BAND: (0.000383, 0.000773), BAND2: (0.001608, 0.000436)}   # lane 29b/86
FLOW_GATE = {"hires": (+0.0084, 0.0342), "seps": (+0.0300, 0.0144)}       # 97's tau, employer SE
FLOW_TOL = 0.0005

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
TREND = "trend_x_high_x_young"
PLANNED_FITS = 9

NOTES: list = []
FAILURES: list = []
ROWS: list = []
FROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. Stock 22-25: Table 1 within 0.0005, later -0.0578 (0.0155),",
    "  tau -0.0399 (0.0102). A miss is a hard stop; nothing is quotable.",
    "  D1. The trend on the longer pre-period is reported with FLAT / NOT",
    "  FLAT under 78's two-SE rule and with the drift carried over 15.5",
    "  months; no verdict on tau-hat.",
    "  B1. Each placebo tau (boundaries moved back S months, panel ending",
    "  November 2022) is reported with its SE and in units of tau-hat's SE.",
    "  Of the order of tau-hat, whichever sign: the 2021-22 plateau is not",
    "  established as the counterfactual by these data alone. Near zero:",
    "  the design returns no decline where none should be. Nothing else.",
    "  F1. Flows: the industry-clustered SE beside the employer-clustered",
    "  one; the coefficient must reproduce 97's tau within 0.0005 (the",
    "  flows gate). No verdict.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


# ----------------------------------------------------------------------
# plumbing (as 103)
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
    if s78.PRE_LAUNCH_END != PRE_LAUNCH_END or s78.EXTENDED_FROM != EXTENDED_FROM:
        raise RuntimeError("78's pre-launch end or extended start differs "
                           "from this script's; refusing to run.")
    return s82, s61, s73, s78, s80, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def add(part, spec, band, term, coef, se, n_obs, n_firms, status="ok",
        vp=np.nan, vi=np.nan, cpi=np.nan, rows=None, **extra):
    (ROWS if rows is None else rows).append(
        {"part": part, "spec": spec, "young_band": band, "term": term,
         "coef": float(coef) if coef == coef else np.nan,
         "se": float(se) if se is not None and se == se else np.nan,
         "t": tstat(coef, se), "var_post": vp, "var_interim": vi,
         "cov_post_interim": cpi, "n_obs": n_obs, "n_firms": n_firms,
         "status": status, **extra})


def _save_one(rows: list, name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
        small = had & df["n_firms"].isna()
        if small.any():
            df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                           "cov_post_interim"]] = np.nan
    df.to_csv(OUT / name, index=False)
    return df


def save() -> None:
    _save_one(ROWS, "prepath_placebo.csv")
    _save_one(FROWS, "flows_industry.csv")


def get(part, spec, band, term, rows=None):
    for r in (ROWS if rows is None else rows):
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
        r = mc.run_fepois_multi(b, OUT, tag=f"s105_{tag}", terms=terms,
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


def record(g, v, part, spec, band, n_firms, pairs=(), extra_terms=(),
           rows=None, **extra) -> None:
    """pairs: (label, post term, interim term) whose tau is recorded with
    the two coefficients; extra_terms are recorded as they are."""
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for lab, p_, i_ in pairs:
        for suffix, t_ in (("post", p_), ("interim", i_)):
            if t_ in g.index:
                add(part, spec, band, f"{lab}_{suffix}", g.loc[t_, "coef"],
                    g.loc[t_, "se"], n_obs, n_firms,
                    str(g.loc[t_].get("status", "ok")), rows=rows, **extra)
        c, s, vp, vi, cpi = tau(g, v, p_, i_)
        add(part, spec, band, f"{lab}_tau", c, s, n_obs, n_firms, "derived",
            vp, vi, cpi, rows=rows, **extra)
    for t_ in extra_terms:
        if t_ in g.index:
            add(part, spec, band, t_, g.loc[t_, "coef"], g.loc[t_, "se"],
                n_obs, n_firms, str(g.loc[t_].get("status", "ok")),
                rows=rows, **extra)
    save()


def check(label: str, got: dict, want: dict, tol: float = GATE_TOL,
          keys=("post", "tau"), se_too: bool = True) -> list:
    bad = []
    for key in keys:
        c, s = got[key]
        wc, ws = want[key]
        ok = abs(c - wc) <= tol and (not se_too or abs(s - ws) <= tol)
        if not ok:
            bad.append(f"{label} {key}: this run {c:+.4f} ({s:.4f}), expected "
                       f"{wc:+.4f} ({ws:.4f}), |diff| {abs(c - wc):.4f}"
                       + (f" and {abs(s - ws):.4f}" if se_too else "")
                       + f" against {tol}")
    return bad


def stop(bad: list, what: str) -> None:
    msg = f"THE {what} GATE FAILED. Nothing from this run is quotable. " + \
          "; ".join(bad)
    print(f"\n  {msg}")
    FAILURES.append(msg)
    write_summary()
    raise SystemExit(f"105: the {what.lower()} gate failed; stopping.")


def load_counts(prefix: str, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


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


def shift_ym(ym: str, months: int) -> str:
    """`ym` moved back by `months` (YYYY-MM arithmetic)."""
    y, m = int(ym[:4]), int(ym[5:7])
    k = y * 12 + (m - 1) - months
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


# ----------------------------------------------------------------------
# G: the gate
# ----------------------------------------------------------------------

def gate(counts, expo, s61, s78, j47) -> None:
    print(f"\n  G. THE STOCK GATE at {BAND}:")
    skel = s61.build_skeleton(counts, BAND, j47)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError(f"the {BAND} panel is empty")
    n = int(b["employer_id"].nunique())
    b, qterms = s78.eq2_terms(b)
    g, v = fit(b, "gate_22_25", qterms, j47.FES)
    del b
    gc.collect()
    record(g, v, "G", "gate", BAND, n, [("hy", POST, INTERIM)])
    got = {"post": get("G", "gate", BAND, "hy_post"),
           "tau": get("G", "gate", BAND, "hy_tau")}
    bad = check(BAND, got, GATE)
    if bad:
        stop(bad, "STOCK")
    print(f"  THE STOCK GATE PASSES at {BAND}: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")


# ----------------------------------------------------------------------
# D and B: the pre-period panels
# ----------------------------------------------------------------------

def pre_panel(extended, expo, band, from_ym, s78, j47) -> pd.DataFrame:
    """The balanced panel from `from_ym` to the month before the launch,
    with exposure attached: the frame D and B are fitted on."""
    # the counts are cut to the pre-launch months BEFORE the balanced
    # panel is built, so the skeleton never holds the 2023-25 months
    src = extended[extended["year_month"].astype(str) < PRE_LAUNCH_END]
    skel = s78.build_skeleton_bands(src, [band] + j47.INCUMBENT_BANDS,
                                    band, j47, from_ym)
    del src
    if skel.empty:
        raise RuntimeError(f"{band}: the pre-period panel from {from_ym} is empty")
    skel = j47._drop_dead_cells(skel)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError(f"{band}: no exposure on the pre-period panel")
    months = sorted(b["year_month"].astype(str).unique())
    NOTES.append(f"pre-period panel {band} from {from_ym}: months {months[0]} to "
                 f"{months[-1]} ({len(months)}), {b['employer_id'].nunique():,} employers")
    return b


def drift(b: pd.DataFrame, band: str, from_ym: str, s78, j47) -> None:
    """78's drift test on the panel it is given (already cut to the
    pre-launch months): the cycle, the tightening window and the trend."""
    spec = f"drift_from_{from_ym[:4]}"
    print(f"\n  D. THE DRIFT TEST at {band} from {from_ym}:")
    n = int(b["employer_id"].nunique())
    b, dterms = s78.drift_terms(b)
    g, v = fit(b, f"{spec}_{band.replace('-', '_')}", dterms, j47.FES)
    b.drop(columns=dterms, inplace=True)
    record(g, v, "D", spec, band, n, extra_terms=dterms)
    c, s = get("D", spec, band, TREND)
    if c == c:
        add("D", spec, band, "trend_carried_15_5_months", c * CARRY_MONTHS,
            s * CARRY_MONTHS if s == s else np.nan, -1, n, "derived")
        save()
        flat = "FLAT" if abs(c) < DRIFT_SE * s else "NOT FLAT"
        print(f"    trend {c:+.6f} ({s:.6f}) per month, {flat}; carried over "
              f"{CARRY_MONTHS} months {c * CARRY_MONTHS:+.4f}")


def placebo_terms(b: pd.DataFrame, S: int) -> tuple:
    """Equation (2)'s term set with every boundary moved back S months:
    the cumulative tightening switch, the three calendar terms, the
    interim window and the adoption step, each x High x Young."""
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    rb_s = shift_ym(mc.RIKSBANK_YM, S)
    launch_s = shift_ym(mc.CHATGPT_YM, S)
    post_s = shift_ym(POST_FROM, S)
    b["rb_x_high_x_young"] = (ym >= rb_s).astype(int) * hy
    terms = ["rb_x_high_x_young"]
    for qq in (1, 2, 3):
        b[f"q{qq}_x_high_x_young"] = (q == qq).astype(int) * hy
        terms.append(f"q{qq}_x_high_x_young")
    b[INTERIM] = ((ym >= launch_s) & (ym < post_s)).astype(int) * hy
    b[POST] = (ym >= post_s).astype(int) * hy
    terms += [INTERIM, POST]
    return b, terms, (rb_s, launch_s, post_s)


def placebo(b: pd.DataFrame, band: str, S: int, j47) -> None:
    spec = f"placebo_shift_{S}"
    print(f"\n  B. THE BACKDATED PLACEBO at {band}, S = {S} months:")
    n = int(b["employer_id"].nunique())
    b, pterms, (rb_s, launch_s, post_s) = placebo_terms(b, S)
    last = str(b["year_month"].astype(str).max())
    NOTES.append(f"{spec} {band}: reference to {shift_ym(rb_s, 1)}, tightening "
                 f"{rb_s} to {shift_ym(launch_s, 1)}, interim {launch_s} to "
                 f"{shift_ym(post_s, 1)}, later {post_s} to {last}")
    g, v = fit(b, f"{spec}_{band.replace('-', '_')}", pterms, j47.FES)
    b.drop(columns=pterms, inplace=True)
    record(g, v, "B", spec, band, n, [("hy", POST, INTERIM)],
           extra_terms=["rb_x_high_x_young"])
    c, s = get("B", spec, band, "hy_tau")
    if c == c:
        print(f"    placebo tau {c:+.4f} ({s:.4f}); tau-hat {GATE['tau'][0]:+.4f} "
              f"({GATE['tau'][1]:.4f}); placebo is {abs(c) / GATE['tau'][1]:.2f} "
              f"of tau-hat's SE in size")


def pre_period(extended, expo, s78, j47) -> None:
    """Both parts on one skeleton per (band, start): D and B share the
    2019 panel at 22-25; the 2020 drift is the same panel cut."""
    for band in (BAND, BAND2):
        b = pre_panel(extended, expo, band, EXTENDED_FROM, s78, j47)
        for from_ym, bb in DRIFT_WINDOWS:
            if bb == band and from_ym == EXTENDED_FROM:
                drift(b, band, from_ym, s78, j47)
        for S, bb in SHIFTS:
            if bb == band:
                start = shift_ym("2021-01", S)       # where the shifted reference would begin
                if start < EXTENDED_FROM:
                    NOTES.append(f"placebo_shift_{S} {band}: the shifted reference would "
                                 f"start {start}; the caches start {EXTENDED_FROM}, so the "
                                 f"reference is {EXTENDED_FROM} to "
                                 f"{shift_ym(shift_ym(mc.RIKSBANK_YM, S), 1)}")
                placebo(b, band, S, j47)
        for from_ym, bb in DRIFT_WINDOWS:
            if bb == band and from_ym != EXTENDED_FROM:
                cut = b[b["year_month"].astype(str) >= from_ym].copy()
                cut = j47._drop_dead_cells(cut)
                NOTES.append(f"drift_from_{from_ym[:4]} {band}: the {EXTENDED_FROM} "
                             f"panel cut at {from_ym}, {cut['employer_id'].nunique():,} employers")
                drift(cut, band, from_ym, s78, j47)
                del cut
                gc.collect()
        del b
        gc.collect()


# ----------------------------------------------------------------------
# F: the flows clustered by industry
# ----------------------------------------------------------------------

def flows_industry(flows, expo, s61, s73, s78, s80, j47) -> None:
    print("\n  F. THE FLOWS AT 22-25, CLUSTERED BY INDUSTRY:")
    for outcome, col in (("hires", "n_hire"), ("seps", "n_sep")):
        src = flows.rename(columns={col: "n_emp"})[COUNT_COLS]
        skel = s61.build_skeleton(src, BAND, j47)
        del src
        gc.collect()
        if skel.empty:
            FAILURES.append(f"F/{outcome}/empty")
            continue
        b = s78.with_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"F/{outcome}/no exposure")
            continue
        n = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        b = attach_industry(b, s73, s80, f"flows {outcome}")
        g, v = fit(b, f"{outcome}_indcl_22_25", terms, j47.FES, cluster="cl_ind")
        del b
        gc.collect()
        record(g, v, "F", f"{outcome}_indcl", BAND, n, [("hy", POST, INTERIM)],
               rows=FROWS, outcome=outcome)
        c, s = get("F", f"{outcome}_indcl", BAND, "hy_tau", rows=FROWS)
        if c == c:
            wc, ws = FLOW_GATE[outcome]
            bad = check(f"flows {outcome}", {"tau": (c, s)}, {"tau": (wc, ws)},
                        tol=FLOW_TOL, keys=("tau",), se_too=False)
            if bad:
                stop(bad, "FLOWS")
            print(f"    {outcome}: tau {c:+.4f}, SE {s:.4f} by industry against "
                  f"{ws:.4f} by employer (97); coefficient reproduces 97's {wc:+.4f}")


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def write_summary() -> None:
    L = ["THE PRE-PERIOD FROM 2019, THE BACKDATED PLACEBO AND THE FLOWS",
         "CLUSTERED BY INDUSTRY (LANE 39a)", "=" * 66, "",
         "tau = later minus interim from one fit, SE from the clustered",
         "covariance (V_LL + V_II - 2 V_LI). Exposure: 82's occupation-route",
         "score, the paper's (uniform3, backward cascade, floor 5).", "",
         "GATE (DAIOE, Table 1's specification and sample):"]
    c, s = get("G", "gate", BAND, "hy_tau")
    if c == c:
        L.append(f"  stock {BAND}: tau {c:+.4f} ({s:.4f}); Table 1 "
                 f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")
    else:
        L.append("  not reached")
    L += ["", "D. THE DRIFT TEST ON THE LONGER PRE-PERIOD (read rule D1):"]
    for from_ym, band in DRIFT_WINDOWS:
        spec = f"drift_from_{from_ym[:4]}"
        c, s = get("D", spec, band, TREND)
        if c != c:
            L.append(f"  {band} from {from_ym}: NO FIT (not a null)")
            continue
        flat = "FLAT" if abs(c) < DRIFT_SE * s else "NOT FLAT"
        pc, ps = PRIOR_DRIFT.get(band, (np.nan, np.nan))
        L.append(f"  {band} from {from_ym} to {shift_ym(PRE_LAUNCH_END, 1)}: trend "
                 f"{c:+.6f} ({s:.6f}) per month, t {tstat(c, s):+.2f}, {flat}; carried "
                 f"over {CARRY_MONTHS} months {c * CARRY_MONTHS:+.4f} "
                 f"[{(c - 1.96 * s) * CARRY_MONTHS:+.4f}, {(c + 1.96 * s) * CARRY_MONTHS:+.4f}]")
        L.append(f"    the paper's drift from 2021-01 (lane 29b/86): {pc:+.6f} ({ps:.6f})")
    L += ["", "B. THE BACKDATED PLACEBO (read rule B1):"]
    for S, band in SHIFTS:
        spec = f"placebo_shift_{S}"
        c, s = get("B", spec, band, "hy_tau")
        cp, _ = get("B", spec, band, "hy_post")
        ci, _ = get("B", spec, band, "hy_interim")
        win = next((n_ for n_ in NOTES if n_.startswith(f"{spec} {band}: reference")), "")
        if c != c:
            L.append(f"  {band}, S = {S}: NO FIT (not a null)")
            continue
        L.append(f"  {band}, S = {S}: placebo tau {c:+.4f} ({s:.4f}), t {tstat(c, s):+.2f}; "
                 f"later {cp:+.4f}, interim {ci:+.4f}; "
                 f"{abs(c) / GATE['tau'][1]:.2f} of tau-hat's SE ({GATE['tau'][0]:+.4f}, "
                 f"{GATE['tau'][1]:.4f})")
        if win:
            L.append(f"    windows: {win.split(': ', 1)[1]}")
    L += ["  The S = 24 later window holds the real tightening months (April to",
          "  November 2022), in which the paper's tightening coefficient at 22-25",
          "  is +0.016; the S = 36 later window holds all of 2021 and 2022."]
    L += ["", "F. THE FLOWS AT 22-25 CLUSTERED BY THREE-DIGIT INDUSTRY (read rule F1):"]
    for outcome in ("hires", "seps"):
        c, s = get("F", f"{outcome}_indcl", BAND, "hy_tau", rows=FROWS)
        wc, ws = FLOW_GATE[outcome]
        if c != c:
            L.append(f"  {outcome}: NO FIT (not a null)")
            continue
        L.append(f"  {outcome}: tau {c:+.4f}, SE {s:.4f} by industry (t {tstat(c, s):+.2f}); "
                 f"97: {wc:+.4f}, SE {ws:.4f} by employer")
    L.append("")
    L += [f"FITS: {DONE} of {PLANNED} attempted came back ({PLANNED_FITS} planned). "
          "A run far shorter than the estimate (1.5 to 2.5 hours) is a failure, "
          "not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "105_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "105_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("105: THE PRE-PERIOD FROM 2019, THE PLACEBO AND THE FLOWS (LANE 39a)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s73, s78, s80, l47, l70, j47 = load_modules()
        for y in (2019, 2020):
            if not (CACHE / f"L_counts_{y}.parquet").exists():
                raise RuntimeError(f"L_counts_{y} is not on the share; this script "
                                   "does no SQL to extend the window (lane 25a / 86 "
                                   "read it from the share)")
        if not (CACHE / "I_industry_key.parquet").exists():
            print("  WARNING: 80's industry key is not on the share; 80 will "
                  "pull it (about an hour of SQL)")
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        built = s82.build_exposure(l47, l70, j47, audit=False)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers, {int((expo['fq'] == 4).sum()):,} "
              "in the top quartile")
        counts = load_counts("L_counts", s61.PANEL_YEARS, COUNT_COLS)
        if counts is None:
            raise RuntimeError("L_counts_2021-2025 missing; run 47L")
        last = str(counts["year_month"].max())
        if last < "2025-06":
            raise RuntimeError(f"the counts end at {last}")
        gate(counts, expo, s61, s78, j47)
        early = load_counts("L_counts", [2019, 2020], COUNT_COLS)
        if early is None:
            raise RuntimeError("L_counts_2019 or L_counts_2020 unreadable")
        extended = pd.concat([early, counts], ignore_index=True)
        del early, counts
        gc.collect()
        first = str(extended["year_month"].min())
        if first > EXTENDED_FROM:
            raise RuntimeError(f"the extended counts start at {first}, not {EXTENDED_FROM}")
        pre_period(extended, expo, s78, j47)
        del extended
        gc.collect()
        flows = load_counts("flows", s61.PANEL_YEARS, FLOW_COLS)
        if flows is None:
            raise RuntimeError("flows_2021-2025 missing or lacking columns; run 54")
        flows_industry(flows, expo, s61, s73, s78, s80, j47)
        del flows
        gc.collect()
        drain(s78, "78")
        drain(s73, "73")
    except SystemExit:
        mc.runlog("105_prepath_placebo", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"105 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("105_prepath_placebo", rc, (time.time() - T0) / 60)
    print("\n105 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

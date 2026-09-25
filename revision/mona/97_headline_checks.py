#!/usr/bin/env python3
"""
97_headline_checks.py -- the headline, sex and exposure-specification
                         checks of lane 37b (formerly 97_female_diagnostics:
                         the female differential's pre-launch path and
                            its industry test, and the credit test
                            re-estimated on tau.

======================================================================
  RUNS IN MONA (lane 37c, first stage). Output folder CANARIES_97_OUT
  (default output_97); parts with CANARIES_97_PARTS (default KPI). SQL:
  one read of Serrano's balance sheet for 2019 leverage (script 73's
  firm_leverage, as lane 24 read it) and, only if script 80's
  I_industry_key cache is not on the share, 80's industry-key pulls.
  Everything else is a cache (47L, 67, 82).
======================================================================

QUESTION
An external review (25 Sep 2026) asked for three things this script
answers. (1) The female differential is the paper's most precisely
estimated result, but its pre-launch path has only been shown for the
pooled bands: is the differential itself flat before the launch? (2) Is
the differential an industry shock that falls on young women, i.e. does
it survive industry x age x sex x month effects? (3) The robustness checks
of the online appendix report gamma_2, not tau; which of them can be
recovered from saved exports, and which must be refitted?

WHAT IS NOT REFITTED, AND WHY
Three of the four checks named in (3) were fitted with Equation (2)'s
full term set (the tightening switch, the three calendar terms, the
interim window and the adoption step), and their covariances were
exported, so tau and its standard error follow from saved files with no
MONA run: industry x age x month (script 80 part C, lane 29b,
vcov_s80_indseas2_*), the twelvefold floor and the size split (script 83
part D, vcov_s83_size_*), and hires and separations (script 82 part C,
vcov_s82_hires_22_25 and vcov_s82_seps_22_25). They are recovered by
revision/local/l58_tau_recovered.py and the flows by the same arithmetic;
the floor-of-five row reproduces Table 1's tau there (-0.0399 (0.0102)).
Only the credit test cannot be recovered: script 73's credit fits carry
no interim term and no calendar terms. It is refitted here (Part K).

THE OBJECT
tau = b_L - b_I, the adoption step minus the interim window, from one
fit, with Var = V_LL + V_II - 2 V_LI from the clustered covariance.
For the sex specification the object is the female differential's tau:
the same contrast on the x High x Young x Female terms.

THE GATES (each part runs only after its gate; a miss is a hard stop)
  Stock: script 61's panel with script 82's score must reproduce Table 1
  within 0.0005 on coefficient and standard error: 22-25 adoption
  -0.0578 (0.0155), tau -0.0399 (0.0102); 26-30 -0.0482 (0.0104), tau
  -0.0403 (0.0067). Part K runs only after both pass.
  Sex: script 67's panel at 22-25 must reproduce Table 1's female
  differential, -0.0858 (0.0142) at adoption and tau -0.0714 (0.0109).
  Parts P and I run only after it passes.

PART K. THE CREDIT TEST ON TAU (22-25 and 26-30)
Script 73's leverage (one minus equity over assets from the employer's
2019 Serrano balance sheet, clipped to 0-3, split at the median of the
panel's employers that carry one; 73's coverage gate of 500 employers and
30 per cent). On the balance-sheet sample: (base) Equation (2) alone;
(lev) Equation (2) x High x Young plus the same six terms x Young x LevHigh
and x High x Young x LevHigh. Reported: tau of High x Young in each and
the share (lev) keeps of (base), and the tau of High x Young x LevHigh.

PART P. THE PRE-LAUNCH PATH OF THE FEMALE DIFFERENTIAL (22-25)
On the pre-launch months, January 2021 to November 2022 (the window of
the pooled drift test, script 78 part A(ii)), on 67's sex panel:
  (i) the plain quarterly path: one dummy per quarter, 2022Q1 omitted,
      no calendar terms (2022Q4 is October and November only), each x
      High x Young, x High x Female and x High x Young x Female;
  (ii) the drift test: the three calendar terms, the tightening window
      (April to November 2022) and a linear trend in months, each with
      the same three interactions. The trend on High x Young x Female is
      the differential's own drift.
PART I. INDUSTRY x AGE x SEX x MONTH (22-25)
Script 80's complete industry key (Ftg cascade, Serrano, business
register), employers without a code leaving the sample as in 80's part C.
Both fits on the same employers: (base) 67's effects; (ind) the
month-by-age-and-sex effect replaced by three-digit industry x age x sex x
month, in which it is nested (still three fixed-effect dimensions, the
fepois ceiling). Industry x age x sex plus month x age x sex, the
alternative the review named, is absorbed by employer x age x sex, since
industry is fixed within the employer, and is not fitted.

PART Q. THE EXPOSURE SPECIFICATION (22-25; review of 25 Sep 2026, and
Referee 1's "why a discrete measure?")
The gate's own panel, outcome, effects and seasonal terms, with High
replaced (i) by the continuous employer score (82's firm mean `mix`),
standardised to mean zero and one baseline SD across the panel's
employers, weighted by incumbent employment as the quartile cuts are:
tau per SD; (ii) by the four quartiles, Q2, Q3 and Q4 each with its own
Equation (2) terms against Q1, in one fit: a tau for each.
PART P is fitted twice, clustered by employer and by three-digit industry
(80's key; employers without a code share one residual cluster, as in
80's Part B), with the same coefficients.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  K1. CREDIT DOES NOT CARRY TAU if, at 22-25, the High x Young tau with
      the leverage terms in keeps at least half of the same-sample
      baseline tau and stays negative and distinguishable at five per
      cent. Otherwise the summary says which condition failed.
  P1. THE DIFFERENTIAL WAS FLAT BEFORE THE LAUNCH if the monthly trend on
      High x Young x Female lies within two standard errors of zero (78's
      rule for the pooled bands). The path carries no verdict.
  I1. THE DIFFERENTIAL IS NOT AN INDUSTRY SHOCK if its tau under industry
      x age x sex x month keeps at least half of the same-sample baseline
      and stays negative and distinguishable at five per cent.
  A missing row is a missing fit, never a zero.

EXPORT (output_97/)
  headline_checks.csv  every reported term and tau, with
                                 var_post, var_interim, cov_post_interim,
                                 n_obs and n_firms (employer counts below
                                 five suppressed with their statistic)
  97_summary.txt, 97_log.txt; vcov_s97_*.csv stay on the share

IN THE PAPER
K: tab:industry_credit Panel B on tau (replaces the gamma_2 rows and the
"credit test leaves the step whole" sentence). P: tab:prepath gains the
female differential's path and drift; Table 1's drift row for women. I:
OA III.2, the sentence on industry shocks to young women.

    python 97_headline_checks.py
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

OUT = HERE / os.environ.get("CANARIES_97_OUT", "output_97")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_97_PARTS", "QKPI").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
SIG5 = 1.959963984540054
BANDS = ["22-25", "26-30"]
SEX_BAND = "22-25"
PRE_FROM, PRE_TO = "2021-01", "2022-11"      # the pre-launch months
REF_QUARTER = "2022Q1"
DRIFT_SE = 2.0
KEEP = 0.50
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]

GATE = {"22-25": {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)},
        "26-30": {"post": (-0.0482, 0.0104), "tau": (-0.0403, 0.0067)}}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
# The checks recovered from saved exports, not refitted (for the summary).
RECOVERED = [
    "industry x age x month: l58_tau_recovered.py from vcov_s80_indseas2_*",
    "floor of sixty and the size split: l58 from vcov_s83_size_*",
    "hires and separations: vcov_s82_hires_22_25, vcov_s82_seps_22_25 "
    "(lane 28c); tau +0.0084 (0.0342) and +0.0300 (0.0144) by the same "
    "arithmetic",
]

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Stock: Table 1 within 0.0005 at 22-25 (-0.0578 (0.0155), tau",
    "  -0.0399 (0.0102)) and 26-30 (-0.0482 (0.0104), tau -0.0403 (0.0067))",
    "  before Part K. Sex: -0.0858 (0.0142), tau -0.0714 (0.0109) before",
    "  Parts P and I. A miss is a hard stop.",
    "  K1. Credit does not carry tau if at 22-25 the High x Young tau with",
    "  the leverage terms in keeps at least half of the same-sample",
    "  baseline and stays negative and distinguishable at five per cent.",
    "  P1. The differential was flat before the launch if the monthly",
    "  trend on High x Young x Female lies within two SE of zero.",
    "  I1. The differential is not an industry shock if its tau under",
    "  industry x age x sex x month keeps at least half of the same-sample",
    "  baseline and stays negative and distinguishable at five per cent.",
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
    s80.OUT, s80.CACHE = OUT, CACHE
    s73 = _mod("73_industry_and_credit.py", "s73")
    s73.OUT = OUT
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    if s78.POST_FROM != "2024-01":
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
        df.to_csv(OUT / "headline_checks.csv", index=False)
        return df
    had = df["n_firms"].notna()
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = had & df["n_firms"].isna()
    if small.any():
        df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                       "cov_post_interim"]] = np.nan
    df.to_csv(OUT / "headline_checks.csv", index=False)
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
    returns (None, None) and is recorded. R's stderr is written in full by
    mona_common._r_failed, never truncated here."""
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s97_{tag}", terms=terms,
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


def record(g, v, part, spec, band, n_firms, pairs) -> None:
    """pairs: (label, post term, interim term) whose tau is recorded, with
    the two coefficients themselves."""
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
    raise SystemExit(f"97: the {what.lower()} gate failed; stopping.")


def load_counts(prefix: str, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# the stock gate and Part K
# ----------------------------------------------------------------------

def stock_gate(counts, band, expo, s61, s78, j47) -> pd.DataFrame:
    """Table 1's gate at one band; returns the gate's panel without its
    terms, so Part K cuts the same skeleton rather than building another."""
    tag = band.replace("-", "_")
    print(f"\n  STOCK GATE at {band}:")
    skel = s61.build_skeleton(counts, band, j47)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError(f"{band}: the gate's panel is empty")
    b, terms = s78.eq2_terms(b)
    n = int(b["employer_id"].nunique())
    g, v = fit(b, f"gate_{tag}", terms, j47.FES)
    record(g, v, "G", "gate", band, n, [("hy", POST, INTERIM)])
    got = {"post": get("G", "gate", band, "hy_post"),
           "tau": get("G", "gate", band, "hy_tau")}
    bad = check(band, got, GATE[band])
    if bad:
        stop(bad, "STOCK")
    print(f"  THE STOCK GATE PASSES at {band}: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    return b.drop(columns=terms)


def leverage(s73) -> pd.DataFrame:
    """Script 73's 2019 leverage (Serrano, 1 - equity/assets, clipped to
    0-3), read exactly as lane 24 read it."""
    conn = mc.connect()
    try:
        schema = s73.discover(conn)
        if schema.empty:
            raise RuntimeError("the catalogue returned no firm table")
        lev = s73.firm_leverage(conn, schema)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    drain(s73, "73")
    if lev is None or lev.empty:
        raise RuntimeError("no 2019 balance sheet answered; Part K cannot run")
    return lev


def part_k(b0: pd.DataFrame, band: str, lmap: dict, s73, s78, j47) -> None:
    """The credit test at one band, on the gate's own panel."""
    tag = band.replace("-", "_")
    print(f"\n  PART K at {band}:")
    emp = b0["employer_id"].drop_duplicates()
    lv = pd.Series(s73.norm_id(emp).map(lmap).to_numpy(),
                   index=emp.to_numpy())
    n_all, n_lev = len(emp), int(lv.notna().sum())
    rate = n_lev / max(n_all, 1)
    msg = (f"K/{band}: {n_lev:,} of {n_all:,} panel employers carry a "
           f"2019 balance sheet ({rate:.1%})")
    print(f"  {msg}")
    NOTES.append(msg)
    if n_lev < s73.MIN_FIRMS or rate < s73.MIN_MATCH_RATE:
        FAILURES.append(f"K/{band}/below 73's coverage gate")
        return
    lv = lv.dropna()
    cut = float(lv.median())
    hi = (lv >= cut)
    share = float(hi.mean())
    op = ">="
    if not (0.05 < share < 0.95):
        hi, op = (lv > cut), ">"
        share = float(hi.mean())
    if not (0.05 < share < 0.95):
        FAILURES.append(f"K/{band}/the median split does not split "
                        f"({share:.1%} high)")
        return
    NOTES.append(f"K/{band}: leverage split at {cut:.3f} ({op}), "
                 f"{share:.1%} of the balance-sheet employers high")
    b = b0[b0["employer_id"].isin(set(lv.index))].copy()
    b["levhi"] = b["employer_id"].map(hi.astype(int)).astype(int)
    n = int(b["employer_id"].nunique())
    b, t0 = s78.eq2_terms(b)
    g, v = fit(b, f"k_base_{tag}", t0, j47.FES)
    record(g, v, "K", "base_balance_sheet", band, n,
           [("hy", POST, INTERIM)])
    # the leverage terms: every Equation (2) period x Young x LevHigh
    # and x High x Young x LevHigh, so leverage gets the same seasonal
    # and interim terms as exposure
    b["high_lev"] = b["high"] * b["levhi"]
    b, t1 = s78.eq2_terms(b, "levhi", "lev")
    b, t2 = s78.eq2_terms(b, "high_lev", "hlev")
    g, v = fit(b, f"k_lev_{tag}", t0 + t1 + t2, j47.FES)
    del b
    gc.collect()
    record(g, v, "K", "leverage", band, n,
           [("hy", POST, INTERIM),
            ("young_x_lev", "post_x_highlev_x_young",
             "interim_x_highlev_x_young"),
            ("hy_x_lev", "post_x_highhlev_x_young",
             "interim_x_highhlev_x_young")])


# ----------------------------------------------------------------------
# the sex gate, Part P and Part I
# ----------------------------------------------------------------------

def drop_dead_sex(b: pd.DataFrame) -> pd.DataFrame:
    """67's dead-cell rule after a window cut: employer x age x sex cells
    zero in every remaining month go, then employers left with fewer than
    two age bands."""
    alive = b.groupby(["employer_id", "age_group", "gender"],
                      observed=True)["n_emp"].transform("sum") > 0
    b = b[alive]
    nb = b.groupby("employer_id", observed=True)["age_group"].transform("nunique")
    return b[nb >= 2].copy()


def sex_gate(sex, expo, s67, s78, j47) -> pd.DataFrame:
    print(f"\n  SEX GATE at {SEX_BAND}:")
    skel = s67.build_skeleton_sex(sex, SEX_BAND, j47, "n_emp")
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the sex gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b, terms = s78.gender_eq2_terms(b)
    g, v = fit(b, "sex_gate_22_25", terms, j47.FES)
    record(g, v, "G", "sex_gate", SEX_BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    got = {"post": get("G", "sex_gate", SEX_BAND, "hyf_post"),
           "tau": get("G", "sex_gate", SEX_BAND, "hyf_tau")}
    bad = check("female differential", got, SEX_GATE)
    if bad:
        stop(bad, "SEX")
    print(f"  THE SEX GATE PASSES: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    return b.drop(columns=terms)


def quarter_label(ym: pd.Series) -> pd.Series:
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    return ym.str.slice(0, 4) + "Q" + q.astype(str)


def three(b, name: str, ind: pd.Series) -> list:
    """ind x High x Young, x High x Female and x High x Young x Female."""
    hy = b["high"] * b["young"]
    hf = b["high"] * b["female"]
    b[f"{name}_x_hy"] = ind * hy
    b[f"{name}_x_hf"] = ind * hf
    b[f"{name}_x_hyf"] = ind * hy * b["female"]
    return [f"{name}_x_hy", f"{name}_x_hf", f"{name}_x_hyf"]


def part_p(bsex: pd.DataFrame, j47, s73, s80) -> None:
    """The female differential before the launch: the plain quarterly path
    and the drift test, each clustered by employer and by industry."""
    ym = bsex["year_month"].astype(str)
    pre = bsex[(ym >= PRE_FROM) & (ym <= PRE_TO)]
    pre = drop_dead_sex(pre)
    n = int(pre["employer_id"].nunique())
    months = sorted(pre["year_month"].astype(str).unique())
    NOTES.append(f"P: pre-launch window {months[0]} to {months[-1]}, "
                 f"{len(months)} months, {n:,} employers")
    key = s80.industry_key(s73)
    drain(s80, "80")
    kmap, src_map = s80.key_maps(key, s73)
    pre, info = s80.attach_cluster(pre, kmap, src_map, s73)
    NOTES.append(f"P: industry clusters {info['n_clusters']:,}; "
                 f"{info['n_unresolved']:,} employers without a code share "
                 f"one residual cluster")
    ym = pre["year_month"].astype(str)
    lab = quarter_label(ym)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    trend = ((ym.str.slice(0, 4).astype(int) - 2021) * 12
             + ym.str.slice(5, 7).astype(int) - 1).astype(float)
    sets = []
    terms = []
    for qq in sorted(lab.unique()):
        if qq == REF_QUARTER:
            continue
        terms += three(pre, f"pq_{qq}", (lab == qq).astype(int))
    sets.append(("path", terms))
    terms = []
    for qq in (1, 2, 3):
        terms += three(pre, f"q{qq}", (q == qq).astype(int))
    terms += three(pre, "rbw", ((ym >= mc.RIKSBANK_YM)
                                & (ym < mc.CHATGPT_YM)).astype(int))
    terms += three(pre, "trend", trend)
    sets.append(("drift", terms))
    for spec, terms in sets:
        for cl, sfx in (("employer_id", ""), ("cl_ind", "_indcl")):
            print(f"\n  PART P, {spec}, clustered by "
                  f"{'employer' if sfx == '' else 'industry'}:")
            g, v = fit(pre, f"p_{spec}{sfx}_22_25", terms, j47.FES, cluster=cl)
            if g is None:
                continue
            n_obs = int(g["n_obs"].max())
            for t_ in terms:
                if t_ in g.index:
                    add("P", spec + sfx, SEX_BAND, t_, g.loc[t_, "coef"],
                        g.loc[t_, "se"], n_obs, n,
                        str(g.loc[t_].get("status", "ok")))
            if spec == "path":
                add("P", spec + sfx, SEX_BAND, f"pq_{REF_QUARTER}_reference",
                    0.0, 0.0, n_obs, n, "reference")
            save()
    del pre
    gc.collect()


def part_q(b0: pd.DataFrame, expo: pd.DataFrame, s78, j47) -> None:
    """Continuous exposure and all four quartiles on the gate's panel."""
    b = b0.copy()
    n = int(b["employer_id"].nunique())
    e = expo[expo["employer_id"].isin(set(b["employer_id"]))][
        ["employer_id", "mix", "n"]]
    mu = np.average(e["mix"], weights=e["n"])
    sd = float(np.sqrt(np.average((e["mix"] - mu) ** 2, weights=e["n"])))
    NOTES.append(f"Q: continuous score standardised on {len(e):,} employers, "
                 f"incumbent-weighted mean {mu:.2f} and SD {sd:.2f} "
                 f"percentile points")
    z = pd.Series(((e["mix"] - mu) / (sd if sd > 0 else 1.0)).to_numpy(),
                  index=e["employer_id"].to_numpy())
    b["z"] = b["employer_id"].map(z).astype(float)
    b, tz = s78.eq2_terms(b, "z", "z")
    print("\n  PART Q (i), continuous exposure per SD:")
    g, v = fit(b, "q_continuous_22_25", tz, j47.FES)
    record(g, v, "Q", "continuous_per_sd", "22-25", n,
           [("z", "post_x_highz_x_young", "interim_x_highz_x_young")])
    b = b.drop(columns=tz + ["z"])
    tq, pairs = [], []
    for k in (2, 3, 4):
        b[f"fq{k}"] = (b["fq"] == k).astype(int)
        b, t = s78.eq2_terms(b, f"fq{k}", f"q{k}")
        tq += t
        pairs.append((f"q{k}_vs_q1", f"post_x_highq{k}_x_young",
                      f"interim_x_highq{k}_x_young"))
    print("\n  PART Q (ii), all four quartiles:")
    g, v = fit(b, "q_quartiles_22_25", tq, j47.FES)
    del b
    gc.collect()
    record(g, v, "Q", "quartiles_vs_q1", "22-25", n, pairs)


def part_i(bsex: pd.DataFrame, s73, s80, s78, j47) -> None:
    key = s80.industry_key(s73)
    drain(s80, "80")
    kmap, src_map = s80.key_maps(key, s73)
    emp = bsex["employer_id"].drop_duplicates()
    code = s73.norm_id(emp).map(kmap)
    lut = pd.Series(pd.factorize(code)[0], index=emp.to_numpy())
    b = bsex.assign(ind_code=bsex["employer_id"].map(lut).astype("int64"))
    b = b[b["ind_code"] >= 0].copy()
    n_all, n = int(len(emp)), int(b["employer_id"].nunique())
    msg = (f"I: {n:,} of {n_all:,} employers of the sex panel carry a "
           f"three-digit code and form the sample for both fits; "
           f"{int(b['ind_code'].nunique()):,} industry groups")
    print(f"\n  {msg}")
    NOTES.append(msg)
    ic = b["ind_code"].to_numpy(dtype="int64")
    ac = pd.factorize(b["age_group"].astype(str) + "_" + b["gender"].astype(str),
                      sort=False)[0].astype("int64")
    tc = pd.factorize(b["year_month"], sort=False)[0].astype("int64")
    n_a, n_t = int(ac.max()) + 1, int(tc.max()) + 1
    b["fe_ind_agesex_t"] = (ic * n_a + ac) * n_t + tc
    b, terms = s78.gender_eq2_terms(b)
    g, v = fit(b, "i_base_22_25", terms, j47.FES)
    record(g, v, "I", "base_industry_sample", SEX_BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    # month x age x sex (fe_t_age on 67's panel) is nested in industry x
    # age x sex x month and is dropped: the same model, one effect fewer
    fes = tuple(f for f in j47.FES if f != "fe_t_age") + ("fe_ind_agesex_t",)
    g, v = fit(b, "i_ind_22_25", terms, fes)
    del b
    gc.collect()
    record(g, v, "I", "industry_age_sex_month", SEX_BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def sig_neg(c, s) -> bool:
    return bool(s == s and s > 0 and c < 0 and abs(c) >= SIG5 * s)


def verdicts() -> list:
    L = []
    b0, _ = get("K", "base_balance_sheet", "22-25", "hy_tau")
    b1, s1 = get("K", "leverage", "22-25", "hy_tau")
    if b0 == b0 and b1 == b1:
        keep = b1 / b0 if b0 else np.nan
        ok = keep >= KEEP and sig_neg(b1, s1)
        L.append(f"  K1: {'CREDIT DOES NOT CARRY TAU' if ok else 'K1 NOT MET'}"
                 f"; 22-25 tau {b1:+.4f} ({s1:.4f}) with leverage in, "
                 f"{keep:.0%} of the same-sample {b0:+.4f}")
    else:
        L.append("  K1: NO VERDICT, a credit fit is missing (not a null)")
    c, s = get("P", "drift", SEX_BAND, "trend_x_hyf")
    if c == c:
        flat = s == s and abs(c) <= DRIFT_SE * s
        L.append(f"  P1: {'THE DIFFERENTIAL WAS FLAT BEFORE THE LAUNCH' if flat else 'P1 NOT MET: the differential drifted before the launch'}"
                 f"; trend {c:+.5f} ({s:.5f}) per month, over 23 months "
                 f"{23 * c:+.4f}")
    else:
        L.append("  P1: NO VERDICT, the drift fit is missing")
    i0, _ = get("I", "base_industry_sample", SEX_BAND, "hyf_tau")
    i1, si = get("I", "industry_age_sex_month", SEX_BAND, "hyf_tau")
    if i0 == i0 and i1 == i1:
        keep = i1 / i0 if i0 else np.nan
        ok = keep >= KEEP and sig_neg(i1, si)
        L.append(f"  I1: {'THE DIFFERENTIAL IS NOT AN INDUSTRY SHOCK' if ok else 'I1 NOT MET'}"
                 f"; tau {i1:+.4f} ({si:.4f}), {keep:.0%} of the "
                 f"same-sample {i0:+.4f}")
    else:
        L.append("  I1: NO VERDICT, an industry fit is missing")
    return L


def write_summary() -> None:
    L = ["THE FEMALE DIFFERENTIAL'S PRE-PATH AND INDUSTRY TEST, AND THE",
         "CREDIT TEST ON TAU", "=" * 62, "",
         "tau = adoption step minus interim, SE from the clustered covariance",
         "of the same fit (V_LL + V_II - 2 V_LI).", "", "GATES:"]
    for band in BANDS:
        c, s = get("G", "gate", band, "hy_tau")
        if c == c:
            L.append(f"  stock {band}: tau {c:+.4f} ({s:.4f}); Table 1 "
                     f"{GATE[band]['tau'][0]:+.4f} ({GATE[band]['tau'][1]:.4f})")
    c, s = get("G", "sex_gate", SEX_BAND, "hyf_tau")
    if c == c:
        L.append(f"  sex 22-25: female differential tau {c:+.4f} ({s:.4f}); "
                 f"Table 1 {SEX_GATE['tau'][0]:+.4f} ({SEX_GATE['tau'][1]:.4f})")
    L.append("")
    if any(r["part"] == "K" for r in ROWS):
        L.append("K. THE CREDIT TEST ON TAU (balance-sheet sample):")
        for band in BANDS:
            for spec, term in (("base_balance_sheet", "hy_tau"),
                               ("leverage", "hy_tau"),
                               ("leverage", "hy_x_lev_tau"),
                               ("leverage", "young_x_lev_tau")):
                c, s = get("K", spec, band, term)
                if c == c:
                    L.append(f"  {band} {spec:<20} {term:<16} {c:+.4f} "
                             f"({s:.4f}) t {tstat(c, s):+.2f}")
        L.append("")
    if any(r["part"] == "P" for r in ROWS):
        L.append(f"P. THE FEMALE DIFFERENTIAL BEFORE THE LAUNCH ({PRE_FROM} to "
                 f"{PRE_TO}), x High x Young x Female:")
        for r in ROWS:
            if r["part"] == "P" and r["spec"] == "path" \
                    and r["term"].endswith("_x_hyf"):
                L.append(f"  {r['term'][3:9]} {r['coef']:+.4f} ({r['se']:.4f})")
        c, s = get("P", "drift", SEX_BAND, "trend_x_hyf")
        ci, si = get("P", "drift_indcl", SEX_BAND, "trend_x_hyf")
        if c == c:
            L.append(f"  drift: trend {c:+.5f} per month, SE {s:.5f} by "
                     f"employer, {si:.5f} by industry")
        L.append("")
    if any(r["part"] == "Q" for r in ROWS):
        L.append("Q. THE EXPOSURE SPECIFICATION, 22-25 (tau):")
        c, s_ = get("Q", "continuous_per_sd", "22-25", "z_tau")
        if c == c:
            L.append(f"  continuous score, per baseline SD  {c:+.4f} ({s_:.4f})")
        for k in (2, 3, 4):
            c, s_ = get("Q", "quartiles_vs_q1", "22-25", f"q{k}_vs_q1_tau")
            if c == c:
                L.append(f"  Q{k} against Q1                    {c:+.4f} "
                         f"({s_:.4f})")
        L.append("")
    if any(r["part"] == "I" for r in ROWS):
        L.append("I. INDUSTRY x AGE x SEX x MONTH, same employers:")
        for spec in ("base_industry_sample", "industry_age_sex_month"):
            for term in ("hyf_tau", "hy_tau"):
                c, s = get("I", spec, SEX_BAND, term)
                if c == c:
                    L.append(f"  {spec:<24} {term:<8} {c:+.4f} ({s:.4f})")
        L.append("")
    L += ["VERDICTS:"] + verdicts() + [""]
    L += ["NOT REFITTED (tau recovered from saved exports, see the "
          "docstring):"] + [f"  {x}" for x in RECOVERED] + [""]
    L += [f"FITS: {DONE} of {PLANNED} attempted came back. A run far shorter "
          "than the estimate (2 to 2.5 hours) is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "97_summary.txt").write_text("\n".join(L), encoding="utf-8")
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
    mc.Tee(OUT / "97_log.txt")
    T0 = time.time()
    print("=" * 70)
    print(f"97: FEMALE DIFFERENTIAL DIAGNOSTICS AND CREDIT ON TAU   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s73, s78, s80, l47, l70, j47 = load_modules()
        need_sex = any(p in PARTS for p in "PI")
        if need_sex and not all((CACHE / f"L_counts_sex_{y}.parquet").exists()
                                for y in s61.PANEL_YEARS):
            raise RuntimeError("L_counts_sex_* are not all on the share; this "
                               "script does no SQL for them (run 67)")
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        built = s82.build_exposure(l47, l70, j47)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers")
        if "K" in PARTS or "Q" in PARTS:
            counts = load_counts("L_counts", s61.PANEL_YEARS, COUNT_COLS)
            if counts is None:
                raise RuntimeError("L_counts_2021-2025 missing; run 47L")
            last = str(counts["year_month"].max())
            if last < "2025-06":
                raise RuntimeError(f"the counts end at {last}")
            # The balance sheet is read once, before any fit; it is data,
            # not an estimate. A failed read costs Part K and nothing else.
            lmap = None
            try:
                if "K" in PARTS:
                    lev = leverage(s73)
                    lmap = dict(zip(s73.norm_id(lev["employer_id"]),
                                    lev["lev"].astype(float)))
                    del lev
            except BaseException as ex:
                if isinstance(ex, SystemExit):
                    raise
                print(f"  leverage FAILED ({type(ex).__name__}: {ex})")
                traceback.print_exc()
                FAILURES.append(f"K/leverage/{type(ex).__name__}: {ex}")
            # One band at a time, gate first: Python never holds two
            # panels while R fits (failure class 4).
            for band in BANDS:
                b0 = stock_gate(counts, band, expo, s61, s78, j47)
                if band == "22-25" and "Q" in PARTS:
                    run_part("Q", part_q, b0, expo, s78, j47)
                if lmap is not None:
                    run_part("K", part_k, b0, band, lmap, s73, s78, j47)
                del b0
                gc.collect()
            del counts
            gc.collect()
        if need_sex:
            sex = load_counts("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
            if sex is None:
                raise RuntimeError("L_counts_sex unreadable or lacking "
                                   f"{SEX_COLS}")
            bsex = sex_gate(sex, expo, s67, s78, j47)
            del sex
            gc.collect()
            if "P" in PARTS:
                run_part("P", part_p, bsex, j47, s73, s80)
            if "I" in PARTS:
                run_part("I", part_i, bsex, s73, s80, s78, j47)
            del bsex
            gc.collect()
        drain(s78, "78")
        drain(s73, "73")
    except SystemExit:
        mc.runlog("97_headline_checks", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"97 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("97_headline_checks", rc, (time.time() - T0) / 60)
    print("\n97 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

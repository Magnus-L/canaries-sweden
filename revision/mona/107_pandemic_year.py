#!/usr/bin/env python3
"""
107_pandemic_year.py -- lane 39c: the pandemic year. Was the female
                        differential present in 2020, which group moved, and
                        does the pre-2021 slope survive a pandemic-year term?

======================================================================
  RUNS IN MONA (lane 39c). Output folder CANARIES_107_OUT (default
  output_107). SQL: 67's sex-counts query for 2019 and 2020 ONLY IF
  L_counts_sex_2019 / _2020 are not on the share; the pull is cached
  under those names (two new caches, the only thing this lane writes to
  the share). Everything else is cached: 47L's L_counts_2019-2025, 67's
  L_counts_sex_2021-2025, 82's cascade and baseline caches.
======================================================================

QUESTION (lane 39a, 26 September 2026)
Backdating the paper's design by 36 months returns tau = -0.038 (0.008),
the size of tau-hat: the young-to-older ratio at exposed employers fell
relative to less exposed employers between 2020 and 2021-22, in the
pandemic year, by as much as in 2024-25 (Table A19). Before the paper can
lead with the female differential as the finding a generic retreat from
junior employment would not produce, three things must be known.

  F  THE FEMALE DIFFERENTIAL BEFORE 2021. On the sex panel from January
     2019 to November 2022: (F1) the backdated placebo of the sex
     specification, boundaries moved back 36 and 24 months, tau for
     High x Young x Female (and for High x Young); (F2) 78's drift test
     from January 2019 on the same three term sets; (F3) the raw
     quarterly path of the female differential 2019Q1 to 2022Q4, 2022Q1
     omitted (97's Part P, extended), fitted last because it carries the
     most terms.
  P  THE POOLED SLOPE NET OF THE PANDEMIC YEAR. 78's drift test from
     January 2019 at 22-25 with one more term, an indicator for March
     2020 to February 2021 x High x Young: if the negative slope of lane
     39a is that year's movement, the indicator takes it and the trend
     returns towards zero.
  R  WHICH GROUP MOVED (descriptive, no fit). From the caches, the
     quarterly head counts of ages 22-25 and 31-69 at top-quartile and at
     other employers of the headline panel, 2019Q1 to 2025Q2, and the same
     for young women and young men: sums over thousands of employers, no
     identifier, so the reader can see whether the 2020 movement ran
     through the young at exposed employers or through the recovery of the
     young elsewhere.

THE GATE (a miss is a hard stop; nothing from the run is quotable)
  The sex specification on the paper's panel reproduces Table 1's female
  differential within 0.0005: -0.0858 (0.0142), tau -0.0714 (0.0109).

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  F1. Each female placebo tau is reported with its SE and in units of the
      differential's SE (0.0109). Of the order of the differential,
      whichever sign: the differential also moved in the pandemic year and
      cannot be presented as specific to 2024-25. Near zero: it did not.
      Nothing else.
  F2. The female drift from 2019 with FLAT / NOT FLAT under 78's rule.
  P1. The trend and the pandemic-year term are reported side by side; no
      verdict beyond the numbers.
  R1. Counts are reported as they are; a cell resting on fewer than five
      employers is suppressed.

EXPORT (output_107/)
  pandemic_year.csv          every reported term and tau (parts G, F, P)
  female_prepath.csv         the quarterly path of the three terms (F3)
  quarterly_counts.csv       part R: quarter x band x group x sex head counts
  107_summary.txt, 107_log.txt; vcov_s107_*.csv stay on the share

IN THE PAPER
Section 3 (the pre-launch paragraph and the female paragraph), OA III.2
(the pre-period paragraphs, Figure A5, Table A19), letter R1.6.

    python 107_pandemic_year.py
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

OUT = HERE / os.environ.get("CANARIES_107_OUT", "output_107")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
POST_FROM = "2024-01"                       # asserted against 78 below
EXTENDED_FROM = "2019-01"
PRE_LAUNCH_END = mc.CHATGPT_YM
PANDEMIC = ("2020-03", "2021-02")           # the pandemic-year indicator, inclusive
REF_QUARTER = "2022Q1"
CARRY_MONTHS = 15.5
DRIFT_SE = 2.0
SHIFTS = (36, 24)
EARLY_YEARS = (2019, 2020)
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]
OLDER = ["31-34", "35-40", "41-49", "50+"]

SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005
POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
TREND, FTREND = "trend_x_high_x_young", "trend_x_high_x_young_x_female"
PANT = "pandemic_x_high_x_young"
PLANNED_FITS = 6

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PATH: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. The sex specification on the paper's panel reproduces Table 1's",
    "  female differential within 0.0005 (-0.0858 (0.0142), tau -0.0714",
    "  (0.0109)); a miss is a hard stop.",
    "  F1. Each female placebo tau (boundaries moved back S months, panel",
    "  ending November 2022) with its SE and in units of the differential's",
    "  SE. Of the order of the differential, whichever sign: the differential",
    "  also moved in the pandemic year. Near zero: it did not. Nothing else.",
    "  F2. The female drift from January 2019, FLAT / NOT FLAT under 78's rule.",
    "  P1. The pooled trend from 2019 and the pandemic-year term side by side.",
    f"  R1. Head counts as they are; cells under {FLOOR} employers suppressed.",
]


# ----------------------------------------------------------------------
# plumbing (as 103 and 105)
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


def open_conn():
    """Separated so the local dry run can replace it."""
    return mc.connect()


def pull_sex(year: int, conn, s67) -> pd.DataFrame:
    """67's sex-counts query for one year; separated so the local dry run
    can replace it."""
    return s67.q_counts_sex(year, conn)


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def add(part, spec, band, term, coef, se, n_obs, n_firms, status="ok",
        vp=np.nan, vi=np.nan, cpi=np.nan):
    ROWS.append({"part": part, "spec": spec, "young_band": band, "term": term,
                 "coef": float(coef) if coef == coef else np.nan,
                 "se": float(se) if se is not None and se == se else np.nan,
                 "t": tstat(coef, se), "var_post": vp, "var_interim": vi,
                 "cov_post_interim": cpi, "n_obs": n_obs, "n_firms": n_firms,
                 "status": status})


def _floor(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "n_firms" not in df.columns:
        return df
    had = df["n_firms"].notna()
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = had & df["n_firms"].isna()
    if small.any():
        df.loc[small, [c for c in ("coef", "se", "t", "var_post", "var_interim",
                                   "cov_post_interim") if c in df.columns]] = np.nan
    return df


def save() -> None:
    _floor(pd.DataFrame(ROWS)).to_csv(OUT / "pandemic_year.csv", index=False)
    _floor(pd.DataFrame(PATH)).to_csv(OUT / "female_prepath.csv", index=False)


def get(part, spec, band, term):
    for r in ROWS:
        if (r["part"], r["spec"], r["young_band"], r["term"]) == (part, spec, band, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    global DONE, PLANNED
    PLANNED += 1
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms, cluster {cluster}{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s107_{tag}", terms=terms,
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


def tau(g, v, post, interim) -> tuple:
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


def record(g, v, part, spec, band, n_firms, pairs=(), extra_terms=()) -> None:
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for lab, p_, i_ in pairs:
        for suffix, t_ in (("post", p_), ("interim", i_)):
            if t_ in g.index:
                add(part, spec, band, f"{lab}_{suffix}", g.loc[t_, "coef"],
                    g.loc[t_, "se"], n_obs, n_firms, str(g.loc[t_].get("status", "ok")))
        c, s, vp, vi, cpi = tau(g, v, p_, i_)
        add(part, spec, band, f"{lab}_tau", c, s, n_obs, n_firms, "derived", vp, vi, cpi)
    for t_ in extra_terms:
        if t_ in g.index:
            add(part, spec, band, t_, g.loc[t_, "coef"], g.loc[t_, "se"], n_obs,
                n_firms, str(g.loc[t_].get("status", "ok")))
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
    msg = f"THE {what} GATE FAILED. Nothing from this run is quotable. " + "; ".join(bad)
    print(f"\n  {msg}")
    FAILURES.append(msg)
    write_summary()
    raise SystemExit(f"107: the {what.lower()} gate failed; stopping.")


def load_counts(prefix: str, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


def shift_ym(ym: str, months: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    k = y * 12 + (m - 1) - months
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


def quarter_label(ym: pd.Series) -> pd.Series:
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    return ym.str.slice(0, 4) + "Q" + q.astype(str)


def month_index(ym: pd.Series) -> pd.Series:
    """Months since January 2021 (78's convention), negative before."""
    return (ym.str.slice(0, 4).astype(int) - 2021) * 12 + ym.str.slice(5, 7).astype(int) - 1


# ----------------------------------------------------------------------
# the early sex counts (67's query, cached)
# ----------------------------------------------------------------------

def early_sex_counts(s67) -> pd.DataFrame:
    """L_counts_sex_2019 and _2020: read from the share if there, pulled
    through 67's own query and cached if not. The only share write."""
    out = []
    for y in EARLY_YEARS:
        cf = CACHE / f"L_counts_sex_{y}.parquet"
        c = mc.read_cache(cf, require=SEX_COLS)
        if c is None:
            print(f"  L_counts_sex_{y} is not on the share: pulling through 67's "
                  f"q_counts_sex ({12} monthly tables)")
            t = time.time()
            conn = open_conn()
            try:
                c = pull_sex(y, conn, s67)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            c = c[SEX_COLS]
            mc.write_cache(c, cf)
            NOTES.append(f"pulled and cached L_counts_sex_{y}: {len(c):,} cells, "
                         f"{(time.time() - t) / 60:.1f} min")
            print(f"    cached {cf.name}: {len(c):,} cells in {(time.time() - t) / 60:.1f} min")
        else:
            NOTES.append(f"L_counts_sex_{y} read from the share ({len(c):,} cells)")
        out.append(c)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# term builders on the sex panel
# ----------------------------------------------------------------------

def three(b: pd.DataFrame, name: str, ind: pd.Series) -> list:
    """ind x High x Young, x High x Female and x High x Young x Female
    (97's convention, with 78's term names for the three headline terms)."""
    hy = b["high"] * b["young"]
    hf = b["high"] * b["female"]
    b[f"{name}_x_high_x_young"] = ind * hy
    b[f"{name}_x_high_x_female"] = ind * hf
    b[f"{name}_x_high_x_young_x_female"] = ind * hy * b["female"]
    return [f"{name}_x_high_x_young", f"{name}_x_high_x_female",
            f"{name}_x_high_x_young_x_female"]


def sex_placebo_terms(b: pd.DataFrame, S: int) -> tuple:
    """78's gender term set with every boundary moved back S months."""
    ym = b["year_month"].astype(str)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    rb_s, launch_s, post_s = (shift_ym(mc.RIKSBANK_YM, S), shift_ym(mc.CHATGPT_YM, S),
                              shift_ym(POST_FROM, S))
    terms = three(b, "rb", (ym >= rb_s).astype(int))
    for qq in (1, 2, 3):
        terms += three(b, f"q{qq}", (q == qq).astype(int))
    terms += three(b, "interim", ((ym >= launch_s) & (ym < post_s)).astype(int))
    terms += three(b, "post", (ym >= post_s).astype(int))
    return b, terms, (rb_s, launch_s, post_s)


def sex_drift_terms(b: pd.DataFrame) -> tuple:
    """78's drift terms on the sex panel: the cycle, the tightening
    window and a linear trend, each on the three term sets."""
    ym = b["year_month"].astype(str)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    terms = []
    for qq in (1, 2, 3):
        terms += three(b, f"q{qq}", (q == qq).astype(int))
    terms += three(b, "rbw", ((ym >= mc.RIKSBANK_YM) & (ym < mc.CHATGPT_YM)).astype(int))
    terms += three(b, "trend", month_index(ym).astype(float))
    return b, terms


def sex_path_terms(b: pd.DataFrame) -> tuple:
    """One term per quarter of the pre-period, REF_QUARTER omitted, on
    the three term sets; no calendar terms."""
    lab = quarter_label(b["year_month"].astype(str))
    terms = []
    for qq in sorted(lab.unique()):
        if qq == REF_QUARTER:
            continue
        terms += three(b, f"pq_{qq}", (lab == qq).astype(int))
    return b, terms


# ----------------------------------------------------------------------
# G: the sex gate
# ----------------------------------------------------------------------

def sex_gate(sexc, expo, s67, s78, j47) -> None:
    print(f"\n  G. THE SEX GATE at {BAND}:")
    skel = s67.build_skeleton_sex(sexc, BAND, j47, "n_emp")
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the sex gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b, qterms = s78.gender_eq2_terms(b)
    g, v = fit(b, "sex_gate_22_25", qterms, j47.FES)
    del b
    gc.collect()
    record(g, v, "G", "sex_gate", BAND, n, [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    got = {"post": get("G", "sex_gate", BAND, "hyf_post"), "tau": get("G", "sex_gate", BAND, "hyf_tau")}
    bad = check("female differential", got, SEX_GATE)
    if bad:
        stop(bad, "SEX")
    print(f"  THE SEX GATE PASSES: tau {got['tau'][0]:+.4f} ({got['tau'][1]:.4f})")


# ----------------------------------------------------------------------
# F: the female differential before 2021
# ----------------------------------------------------------------------

def sex_pre_panel(sex_ext, expo, s67, s78, j47) -> pd.DataFrame:
    src = sex_ext[sex_ext["year_month"].astype(str) < PRE_LAUNCH_END]
    skel = s67.build_skeleton_sex(src, BAND, j47, "n_emp")
    del src
    gc.collect()
    if skel.empty:
        raise RuntimeError("the sex pre-period panel is empty")
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("no exposure on the sex pre-period panel")
    months = sorted(b["year_month"].astype(str).unique())
    NOTES.append(f"sex pre-period panel from {EXTENDED_FROM}: months {months[0]} to {months[-1]} "
                 f"({len(months)}), {b['employer_id'].nunique():,} employers")
    return b


def female_placebos(b: pd.DataFrame, j47) -> None:
    n = int(b["employer_id"].nunique())
    for S in SHIFTS:
        spec = f"placebo_shift_{S}"
        print(f"\n  F1. THE FEMALE PLACEBO, S = {S} months:")
        b, terms, (rb_s, launch_s, post_s) = sex_placebo_terms(b, S)
        last = str(b["year_month"].astype(str).max())
        NOTES.append(f"{spec} sex: reference to {shift_ym(rb_s, 1)}, tightening {rb_s} to "
                     f"{shift_ym(launch_s, 1)}, interim {launch_s} to {shift_ym(post_s, 1)}, "
                     f"later {post_s} to {last}")
        g, v = fit(b, f"sex_{spec}_22_25", terms, j47.FES)
        b.drop(columns=terms, inplace=True)
        record(g, v, "F", spec, BAND, n, [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
        c, s = get("F", spec, BAND, "hyf_tau")
        if c == c:
            print(f"    female placebo tau {c:+.4f} ({s:.4f}); the differential's SE "
                  f"{SEX_GATE['tau'][1]:.4f}; placebo is {abs(c) / SEX_GATE['tau'][1]:.2f} of it")


def female_drift(b: pd.DataFrame, j47) -> None:
    print("\n  F2. THE FEMALE DRIFT from January 2019:")
    n = int(b["employer_id"].nunique())
    b, terms = sex_drift_terms(b)
    g, v = fit(b, "sex_drift_from_2019_22_25", terms, j47.FES)
    b.drop(columns=terms, inplace=True)
    record(g, v, "F", "drift_from_2019", BAND, n, extra_terms=terms)
    for t_, lab in ((FTREND, "female differential"), (TREND, "young, both sexes")):
        c, s = get("F", "drift_from_2019", BAND, t_)
        if c == c:
            add("F", "drift_from_2019", BAND, t_ + "_carried_15_5_months", c * CARRY_MONTHS,
                s * CARRY_MONTHS if s == s else np.nan, -1, n, "derived")
            print(f"    {lab}: trend {c:+.6f} ({s:.6f}), {'FLAT' if abs(c) < DRIFT_SE * s else 'NOT FLAT'}")
    save()


def female_path(b: pd.DataFrame, j47) -> None:
    print("\n  F3. THE FEMALE QUARTERLY PATH from 2019Q1 (fitted last):")
    n = int(b["employer_id"].nunique())
    b, terms = sex_path_terms(b)
    g, v = fit(b, "sex_prepath_22_25", terms, j47.FES)
    b.drop(columns=terms, inplace=True)
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for t_ in terms:
        if t_ in g.index:
            which = ("hyf" if t_.endswith("_x_female") and "_x_young_" in t_ else
                     "hf" if t_.endswith("_x_female") else "hy")
            PATH.append({"quarter": t_.removeprefix("pq_").split("_x_high")[0], "term": which,
                         "coef": float(g.loc[t_, "coef"]), "se": float(g.loc[t_, "se"]),
                         "n_obs": n_obs, "n_firms": n, "status": str(g.loc[t_].get("status", "ok"))})
    for which in ("hy", "hf", "hyf"):
        PATH.append({"quarter": REF_QUARTER, "term": which, "coef": 0.0, "se": 0.0,
                     "n_obs": 0, "n_firms": n, "status": "reference"})
    save()


# ----------------------------------------------------------------------
# P: the pooled slope net of the pandemic year
# ----------------------------------------------------------------------

def pooled_pandemic(extended, expo, s78, j47) -> None:
    print("\n  P. THE POOLED DRIFT FROM 2019 WITH A PANDEMIC-YEAR TERM:")
    src = extended[extended["year_month"].astype(str) < PRE_LAUNCH_END]
    skel = s78.build_skeleton_bands(src, [BAND] + j47.INCUMBENT_BANDS, BAND, j47, EXTENDED_FROM)
    del src
    gc.collect()
    skel = j47._drop_dead_cells(skel)
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the pooled pre-period panel is empty")
    n = int(b["employer_id"].nunique())
    b, dterms = s78.drift_terms(b)
    ym = b["year_month"].astype(str)
    b[PANT] = ((ym >= PANDEMIC[0]) & (ym <= PANDEMIC[1])).astype(int) * b["high"] * b["young"]
    terms = dterms + [PANT]
    g, v = fit(b, "pooled_drift_pandemic_22_25", terms, j47.FES)
    del b
    gc.collect()
    record(g, v, "P", "drift_from_2019_pandemic", BAND, n, extra_terms=terms)
    c, s = get("P", "drift_from_2019_pandemic", BAND, TREND)
    d, e = get("P", "drift_from_2019_pandemic", BAND, PANT)
    if c == c:
        add("P", "drift_from_2019_pandemic", BAND, TREND + "_carried_15_5_months", c * CARRY_MONTHS,
            s * CARRY_MONTHS if s == s else np.nan, -1, n, "derived")
        save()
        print(f"    trend {c:+.6f} ({s:.6f}), {'FLAT' if abs(c) < DRIFT_SE * s else 'NOT FLAT'}; "
              f"pandemic-year term {d:+.4f} ({e:.4f})")


# ----------------------------------------------------------------------
# R: which group moved (descriptive)
# ----------------------------------------------------------------------

def quarterly_counts(extended, sex_ext, expo, employers) -> pd.DataFrame:
    """Quarterly head counts by band (22-25, 31-69) and exposure group on
    the headline panel's employers, both sexes and by sex."""
    e = expo[["employer_id", "fq"]]
    rows = []
    for src, sexed in ((extended, False), (sex_ext, True)):
        d = src[src["employer_id"].isin(employers)].merge(e, on="employer_id", how="inner")
        d = d[d["age_group"].astype(str).isin([BAND] + OLDER)].copy()
        d["band"] = np.where(d["age_group"].astype(str) == BAND, BAND, "31-69")
        d["group"] = np.where(d["fq"] == 4, "top", "rest")
        d["quarter"] = quarter_label(d["year_month"].astype(str))
        keys = ["quarter", "band", "group"] + (["gender"] if sexed else [])
        g = d.groupby(keys, observed=True).agg(n_emp=("n_emp", "sum"),
                                                n_employers=("employer_id", "nunique")).reset_index()
        if not sexed:
            g["gender"] = "all"
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["gender"] = out["gender"].astype(str).map({"1": "men", "2": "women", "all": "all"}).fillna(out["gender"].astype(str))
    small = out["n_employers"] < FLOOR
    if small.any():
        out.loc[small, ["n_emp"]] = np.nan
        NOTES.append(f"R: {int(small.sum())} quarterly cells under the floor suppressed")
    out = out.sort_values(["gender", "band", "group", "quarter"]).reset_index(drop=True)
    out.to_csv(OUT / "quarterly_counts.csv", index=False)
    print(f"  R: {len(out)} quarter x band x group x sex cells written")
    return out


def group_lines(q: pd.DataFrame) -> list:
    """The young-to-older ratio by group and the first quarter of each
    year, both sexes, as the summary prints it."""
    L = []
    for gender in ("all", "women", "men"):
        s = q[q["gender"] == gender]
        if s.empty:
            continue
        L.append(f"  {gender}: young (22-25) per 100 older (31-69), first quarters, top | rest")
        for yr in range(2019, 2026):
            qq = f"{yr}Q1"
            vals = []
            for grp in ("top", "rest"):
                y = s[(s.quarter == qq) & (s.band == BAND) & (s.group == grp)]["n_emp"]
                o = s[(s.quarter == qq) & (s.band == "31-69") & (s.group == grp)]["n_emp"]
                vals.append(f"{100 * float(y.iloc[0]) / float(o.iloc[0]):.2f}" if len(y) == 1 and len(o) == 1
                            and np.isfinite(y.iloc[0]) and np.isfinite(o.iloc[0]) and float(o.iloc[0]) > 0 else "n/a")
            L.append(f"    {qq}: {vals[0]} | {vals[1]}")
    return L


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def write_summary(q: pd.DataFrame = None) -> None:
    L = ["THE PANDEMIC YEAR: THE FEMALE DIFFERENTIAL BEFORE 2021, THE POOLED",
         "SLOPE NET OF 2020, AND WHICH GROUP MOVED (LANE 39c)", "=" * 66, "",
         "tau = later minus interim from one fit, SE from the clustered",
         "covariance; employer clustering. Exposure: 82's occupation-route score.", "",
         "GATE (the sex specification, Table 1's sample):"]
    c, s = get("G", "sex_gate", BAND, "hyf_tau")
    L.append(f"  female differential tau {c:+.4f} ({s:.4f}); Table 1 {SEX_GATE['tau'][0]:+.4f} "
             f"({SEX_GATE['tau'][1]:.4f})" if c == c else "  not reached")
    L += ["", "F1. THE FEMALE PLACEBO (read rule F1):"]
    for S in SHIFTS:
        spec = f"placebo_shift_{S}"
        cf, sf = get("F", spec, BAND, "hyf_tau")
        cy, sy = get("F", spec, BAND, "hy_tau")
        win = next((n_ for n_ in NOTES if n_.startswith(f"{spec} sex: reference")), "")
        if cf != cf:
            L.append(f"  S = {S}: NO FIT (not a null)")
            continue
        L.append(f"  S = {S}: female differential tau {cf:+.4f} ({sf:.4f}), t {tstat(cf, sf):+.2f}, "
                 f"{abs(cf) / SEX_GATE['tau'][1]:.2f} of the differential's SE; young men's tau "
                 f"{cy:+.4f} ({sy:.4f})")
        if win:
            L.append(f"    windows: {win.split(': ', 1)[1]}")
    L += ["", "F2. THE FEMALE DRIFT from January 2019 (read rule F2):"]
    for t_, lab in ((FTREND, "female differential"), (TREND, "young, both sexes")):
        c, s = get("F", "drift_from_2019", BAND, t_)
        L.append(f"  {lab}: trend {c:+.6f} ({s:.6f}) per month, t {tstat(c, s):+.2f}, "
                 f"{'FLAT' if abs(c) < DRIFT_SE * s else 'NOT FLAT'}; carried over 15.5 months {c * CARRY_MONTHS:+.4f}"
                 if c == c else f"  {lab}: NO FIT (not a null)")
    L += ["", "F3. THE FEMALE QUARTERLY PATH 2019Q1 to 2022Q4 (High x Young x Female, 2022Q1 omitted):"]
    if PATH:
        P = pd.DataFrame(PATH)
        for _, r in P[P.term == "hyf"].sort_values("quarter").iterrows():
            L.append(f"    {r['quarter']:<8} " + ("reference" if r["status"] == "reference"
                                                   else f"{r['coef']:+.4f} ({r['se']:.4f})"))
    else:
        L.append("  NO FIT (not a null)")
    L += ["", "P. THE POOLED DRIFT FROM 2019 WITH A PANDEMIC-YEAR TERM (read rule P1):"]
    c, s = get("P", "drift_from_2019_pandemic", BAND, TREND)
    d, e = get("P", "drift_from_2019_pandemic", BAND, PANT)
    if c == c:
        L.append(f"  trend {c:+.6f} ({s:.6f}) per month, t {tstat(c, s):+.2f}, "
                 f"{'FLAT' if abs(c) < DRIFT_SE * s else 'NOT FLAT'}; carried over 15.5 months {c * CARRY_MONTHS:+.4f}")
        L.append(f"  pandemic-year term (March 2020 to February 2021 x High x Young) {d:+.4f} ({e:.4f}), t {tstat(d, e):+.2f}")
        L.append("  lane 39a's trend without the term: -0.003209 (0.000430)")
    else:
        L.append("  NO FIT (not a null)")
    L += ["", "R. WHICH GROUP MOVED (read rule R1; quarterly_counts.csv):"]
    L += group_lines(q) if q is not None and not q.empty else ["  not computed"]
    L.append("")
    L += [f"FITS: {DONE} of {PLANNED} attempted came back ({PLANNED_FITS} planned). "
          "A run far shorter than the estimate (2 to 3 hours) is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES), "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} min. " + mc.mem_line("")]
    (OUT / "107_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "107_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("107: THE PANDEMIC YEAR (LANE 39c)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    q = None
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        for y in (2019, 2020):
            if not (CACHE / f"L_counts_{y}.parquet").exists():
                raise RuntimeError(f"L_counts_{y} is not on the share")
        if not all((CACHE / f"L_counts_sex_{y}.parquet").exists() for y in s61.PANEL_YEARS):
            raise RuntimeError("L_counts_sex_2021-2025 are not all on the share (run 67)")
        built = s82.build_exposure(l47, l70, j47, audit=False)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers, {int((expo['fq'] == 4).sum()):,} in the top quartile")
        # the early sex counts first: SQL, if any, happens before any R fit
        early_sex = early_sex_counts(s67)
        sexc = load_counts("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
        if sexc is None:
            raise RuntimeError("L_counts_sex_2021-2025 unreadable")
        sex_gate(sexc, expo, s67, s78, j47)
        sex_ext = pd.concat([early_sex, sexc], ignore_index=True)
        del early_sex
        gc.collect()
        b = sex_pre_panel(sex_ext, expo, s67, s78, j47)
        female_placebos(b, j47)
        female_drift(b, j47)
        female_path(b, j47)
        del b
        gc.collect()
        counts = load_counts("L_counts", list(EARLY_YEARS) + s61.PANEL_YEARS, COUNT_COLS)
        if counts is None:
            raise RuntimeError("L_counts_2019-2025 missing")
        pooled_pandemic(counts, expo, s78, j47)
        # the headline panel's employers, for the descriptive counts
        skel = s61.build_skeleton(counts[counts["year_month"].astype(str) >= s61.PANEL_FROM], BAND, j47)
        employers = set(s78.with_exposure(skel, expo)["employer_id"].unique())
        del skel
        gc.collect()
        NOTES.append(f"R: counts on the {len(employers):,} employers of the headline panel at {BAND}")
        q = quarterly_counts(counts, sex_ext, expo, employers)
        del counts, sex_ext, sexc
        gc.collect()
        drain(s78, "78")
    except SystemExit:
        mc.runlog("107_pandemic_year", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"107 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary(q)
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("107_pandemic_year", rc, (time.time() - T0) / 60)
    print("\n107 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

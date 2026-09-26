#!/usr/bin/env python3
"""
103_eloundou_classification.py -- lane 38d: the headline design with
                                  employers classified by the GPT-exposure
                                  rating of Eloundou et al. (2024) in place
                                  of DAIOE.

======================================================================
  RUNS IN MONA (lane 38d). Output folder CANARIES_103_OUT (default
  output_103). SQL: none. Every frame is a cache lane 37b left on the
  share: 47L's L_counts_2021-2025, 67's L_counts_sex_2021-2025, 82's
  cascade and baseline caches and 80's I_industry_key. Input file:
  eloundou_ssyk4.dta at the project root (the file script 63 read).
======================================================================

QUESTION (the paper, Section 3; OA III.2)
The employment result ranks employers by the DAIOE exposure of the 2019
occupations of their incumbents aged 31 to 69 and treats the top quartile
as exposed. Is the estimate specific to that index? Script 63 answered on
an occupation-scaled route (each band scored from its own occupations);
this script answers on the HEADLINE design itself: the same cascade of
2019 codes, the same three-digit book, the same incumbent floor and the
same employment-weighted quartile cut, with the Eloundou rating as the
occupation score, and then Equation (2) exactly as Table 1 fits it.

THE OBJECT
tau = b_L - b_I from one fit, Var = V_LL + V_II - 2 V_LI from the clustered
covariance. The female differential's tau is the same contrast on the
x High x Young x Female terms of the sex specification.

THE DESIGN
  1. The gate: Table 1's fit at 22-25 on DAIOE (employer clustering).
  2. The common sample: employers scored on BOTH indices. An employer is
     unscored on the Eloundou rating only if none of its coded incumbents
     holds a code that rating covers (394 of DAIOE's 423 four-digit keys),
     so the loss is small and is counted.
  3. On the common sample, three fits of Equation (2): DAIOE
     classification (employer clustering); Eloundou classification,
     clustered by employer and by three-digit industry (80's key).
  4. The same on the sex panel (female differential), and the DAIOE and
     Eloundou fits at 26-30 (employer clustering).
  5. How the two classifications agree, on every scored employer and on
     the estimation panel: employers and incumbent employment in the same
     quartile, the overlap of the two top quartiles, the rank correlation
     of the two employer scores, and the 4 x 4 quartile cross-tabulation.
Each index keeps its OWN quartile cut, formed over the employers it
scores, as the design defines exposure; the fits then run on the
employers both indices score. Ten fits.

THE GATES (a miss is a hard stop; nothing from the run is quotable)
  Stock, 22-25: Table 1 within 0.0005 on coefficient and SE, later
  -0.0578 (0.0155), tau -0.0399 (0.0102).
  Sex, 22-25: the female differential -0.0858 (0.0142), tau -0.0714
  (0.0109).

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  E1. tau at 22-25 on the Eloundou classification is reported beside the
      DAIOE tau ON THE SAME EMPLOYERS and beside the gate, with both
      clusterings for the Eloundou fit, and the movement (Eloundou minus
      DAIOE, same employers) is stated in units of the gate's
      employer-clustered SE. No verdict beyond that: the fits share their
      sample and their noise, and the SE of the difference is not
      available from two separate covariances, which the summary says.
  E2. The same for the female differential.
  E3. The same for 26-30, employer clustering only.
  E4. The agreement of the two classifications is reported as shares of
      employers and of incumbent employment, never as a verdict.
  Employer counts below five are suppressed with their statistic; the
  cross-tabulation cells likewise.

EXPORT (output_103/)
  eloundou_classification.csv   every reported term and tau, with
                                var_post, var_interim, cov_post_interim,
                                n_obs and n_firms
  classification_agreement.csv  the agreement statistics, one row each
  103_summary.txt, 103_log.txt; vcov_s103_*.csv stay on the share

IN THE PAPER
OA III.2 (one or two sentences beside Table A25 Panel F and Table A18),
the paper's final results paragraph (one clause), the letter's C5.

    python 103_eloundou_classification.py
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

OUT = HERE / os.environ.get("CANARIES_103_OUT", "output_103")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
BAND2 = "26-30"
POST_FROM = "2024-01"                       # asserted against 78 below
ELOUNDOU_FILE = "eloundou_ssyk4.dta"        # at the project root, as 63 read it
ELOUNDOU_COL = "eloundou_score"
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]

GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
TABLE1_26_30 = (-0.0399, 0.0068)            # printed -0.040 (0.007); reported beside, no gate
GATE_TOL = 0.0005

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
CLUSTERS = (("employer_id", ""), ("cl_ind", "_indcl"))
PLANNED_FITS = 10

NOTES: list = []
FAILURES: list = []
ROWS: list = []
AGREE: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Stock 22-25: Table 1 within 0.0005, later -0.0578 (0.0155),",
    "  tau -0.0399 (0.0102). Sex 22-25: -0.0858 (0.0142), tau -0.0714",
    "  (0.0109). A miss is a hard stop; nothing from the run is quotable.",
    "  E1. tau at 22-25 on the Eloundou classification is reported beside",
    "  the DAIOE tau on the same employers and beside the gate, both",
    "  clusterings for the Eloundou fit; the movement (Eloundou minus DAIOE,",
    "  same employers) is stated in units of the gate's employer-clustered",
    "  SE. No verdict beyond 'moves by x SE'. The SE of the difference is",
    "  not available from two separate covariances, and the summary says so.",
    "  E2. The same for the female differential.",
    "  E3. The same for 26-30, employer clustering only.",
    "  E4. Agreement of the two classifications as shares of employers and",
    "  of incumbent employment; no verdict.",
    f"  Employer counts below {FLOOR} are suppressed with their statistic.",
]


# ----------------------------------------------------------------------
# plumbing (as 102)
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
        df.to_csv(OUT / "eloundou_classification.csv", index=False)
    else:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
        small = had & df["n_firms"].isna()
        if small.any():
            df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                           "cov_post_interim"]] = np.nan
        df.to_csv(OUT / "eloundou_classification.csv", index=False)
    a = pd.DataFrame(AGREE, columns=["population", "statistic", "value"])
    a.to_csv(OUT / "classification_agreement.csv", index=False)
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
        r = mc.run_fepois_multi(b, OUT, tag=f"s103_{tag}", terms=terms,
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
    the two coefficients; extra_terms are recorded as they are."""
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
    raise SystemExit(f"103: the {what.lower()} gate failed; stopping.")


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


# ----------------------------------------------------------------------
# the two scores
# ----------------------------------------------------------------------

def eloundou_scores() -> pd.DataFrame:
    """The Eloundou frame as 82's scoring chain needs it: ssyk4 (four
    characters) and `score`. The rating is the beta share of tasks on the
    occupation's own scale; the quartile cut is on the employer mix's
    distribution, so the scale does not enter the classification."""
    p = Path(mc.SHARE) / ELOUNDOU_FILE
    if not p.exists():
        raise RuntimeError(f"{ELOUNDOU_FILE} is not at the project root ({p})")
    d = pd.read_stata(str(p))
    if ELOUNDOU_COL not in d.columns:
        raise RuntimeError(f"{ELOUNDOU_FILE} has no column {ELOUNDOU_COL}: "
                           f"{list(d.columns)}")
    d["ssyk4"] = d["ssyk4"].astype(int).astype(str).str.zfill(4)
    d = d.rename(columns={ELOUNDOU_COL: "score"})[["ssyk4", "score"]]
    d = d[d["score"].notna()].drop_duplicates("ssyk4")
    if not 300 <= len(d) <= 500 or d["ssyk4"].str.len().ne(4).any():
        raise RuntimeError(f"{ELOUNDOU_FILE}: {len(d)} scored codes, expected "
                           "a few hundred four-digit keys")
    print(f"  Eloundou rating: {len(d)} four-digit codes, score "
          f"{d['score'].min():.3f} to {d['score'].max():.3f}")
    return d


def build_both(s82, l47, l70, j47) -> tuple:
    """The two employer classifications from one scoring chain: 82's
    build_exposure with its own DAIOE frame, then with the Eloundou
    frame in its place. Nothing is written to the share: the chain's
    caches are read, not rebuilt, and the score itself is never cached."""
    print("\n  SCORE 1: DAIOE (the paper's classification)")
    built = s82.build_exposure(l47, l70, j47)
    drain(s82, "82/daioe")
    expo_d = built["exposure"]
    daioe_codes = set(built["daioe"]["ssyk4"].astype(str))
    del built
    gc.collect()
    print("\n  SCORE 2: the Eloundou rating, same chain")
    print("  (the within-group benchmark printed below compares to DAIOE's "
          "unweighted benchmark and is not read for this score)")
    el = eloundou_scores()
    both = len(daioe_codes & set(el["ssyk4"]))
    NOTES.append(f"scores: DAIOE {len(daioe_codes)} codes, Eloundou "
                 f"{len(el)} codes, {both} in common")
    built = s82.build_exposure(l47, l70, j47, daioe=el, audit=False)
    drain(s82, "82/eloundou")
    expo_e = built["exposure"]
    del built
    gc.collect()
    for name, e in (("DAIOE", expo_d), ("Eloundou", expo_e)):
        print(f"  {name}: {len(e):,} employers scored, "
              f"{int((e['fq'] == 4).sum()):,} in the top quartile")
    return expo_d, expo_e


# ----------------------------------------------------------------------
# how the two classifications agree (E4)
# ----------------------------------------------------------------------

def agree_stat(pop: str, stat: str, value) -> None:
    AGREE.append({"population": pop, "statistic": stat,
                  "value": float(value) if value == value else np.nan})


def agreement(expo_d: pd.DataFrame, expo_e: pd.DataFrame, pop: str,
              employers=None) -> None:
    """Shares of employers and of incumbent employment (the DAIOE
    frame's floor quantity n) in the same quartile, the overlap of the two
    top quartiles, the rank correlation of the two employer scores, and
    the quartile cross-tabulation, on `employers` (None = every employer
    scored on either index)."""
    d = expo_d[["employer_id", "fq", "mix", "n"]]
    e = expo_e[["employer_id", "fq", "mix"]].rename(
        columns={"fq": "fq_e", "mix": "mix_e"})
    if employers is not None:
        keep = set(employers)
        d = d[d["employer_id"].isin(keep)]
        e = e[e["employer_id"].isin(keep)]
    m = d.merge(e, on="employer_id", how="inner")
    agree_stat(pop, "n_scored_daioe", len(d))
    agree_stat(pop, "n_scored_eloundou", len(e))
    agree_stat(pop, "n_scored_both", len(m))
    if m.empty:
        NOTES.append(f"agreement/{pop}: no employer scored on both")
        return
    w = m["n"].astype(float)
    same = m["fq"] == m["fq_e"]
    td, te = m["fq"] == 4, m["fq_e"] == 4
    agree_stat(pop, "share_employers_same_quartile", same.mean())
    agree_stat(pop, "share_employment_same_quartile", w[same].sum() / w.sum())
    agree_stat(pop, "n_top_daioe", td.sum())
    agree_stat(pop, "n_top_eloundou", te.sum())
    agree_stat(pop, "n_top_both", (td & te).sum())
    agree_stat(pop, "share_top_daioe_also_top_eloundou",
               (td & te).sum() / max(td.sum(), 1))
    agree_stat(pop, "share_top_eloundou_also_top_daioe",
               (td & te).sum() / max(te.sum(), 1))
    agree_stat(pop, "employment_share_top_daioe_also_top_eloundou",
               w[td & te].sum() / max(w[td].sum(), 1e-9))
    agree_stat(pop, "jaccard_top_quartiles",
               (td & te).sum() / max((td | te).sum(), 1))
    agree_stat(pop, "spearman_employer_scores",
               m["mix"].corr(m["mix_e"], method="spearman"))
    agree_stat(pop, "pearson_employer_scores", m["mix"].corr(m["mix_e"]))
    for qd in (1, 2, 3, 4):
        for qe in (1, 2, 3, 4):
            n = int(((m["fq"] == qd) & (m["fq_e"] == qe)).sum())
            agree_stat(pop, f"n_daioe_q{qd}_eloundou_q{qe}",
                       n if (n == 0 or n >= FLOOR) else np.nan)
    print(f"  agreement ({pop}): {len(m):,} employers scored on both; "
          f"{same.mean():.1%} of employers and {w[same].sum() / w.sum():.1%} "
          f"of incumbent employment in the same quartile; of DAIOE's top "
          f"quartile {(td & te).sum() / max(td.sum(), 1):.1%} is Eloundou's "
          f"too; Spearman {m['mix'].corr(m['mix_e'], method='spearman'):.3f}")
    save()


# ----------------------------------------------------------------------
# the fits
# ----------------------------------------------------------------------

def common_panel(b: pd.DataFrame, expo_e: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The gate's panel restricted to employers the Eloundou rating also
    scores, with both quartiles attached. `high` is set by the caller."""
    n0 = b["employer_id"].nunique()
    bc = b.merge(expo_e[["employer_id", "fq"]].rename(columns={"fq": "fq_e"}),
                 on="employer_id", how="inner")
    n1 = bc["employer_id"].nunique()
    NOTES.append(f"{tag}: {n0:,} employers in the DAIOE panel, {n1:,} also "
                 f"scored on Eloundou ({n0 - n1:,} lost)")
    print(f"  common sample: {n1:,} of {n0:,} employers ({n0 - n1:,} not "
          f"scored on the Eloundou rating)")
    if bc.empty:
        raise RuntimeError(f"{tag}: no employer is scored on both indices")
    return bc


def stock(counts, expo_d, expo_e, band, s61, s73, s78, s80, j47,
          gate: bool) -> None:
    print(f"\n  STOCK at {band}:")
    skel = s61.build_skeleton(counts, band, j47)
    b = s78.with_exposure(skel, expo_d)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError(f"the {band} panel is empty")
    n = int(b["employer_id"].nunique())
    if gate:
        b, qterms = s78.eq2_terms(b)
        g, v = fit(b, f"gate_{band.replace('-', '_')}", qterms, j47.FES)
        record(g, v, "G", "gate", band, n, [("hy", POST, INTERIM)])
        got = {"post": get("G", "gate", band, "hy_post"),
               "tau": get("G", "gate", band, "hy_tau")}
        bad = check(band, got, GATE)
        if bad:
            stop(bad, "STOCK")
        print(f"  THE STOCK GATE PASSES at {band}: tau {got['tau'][0]:+.4f} "
              f"({got['tau'][1]:.4f})")
        b = b.drop(columns=qterms)
        agreement(expo_d, expo_e, "estimation_panel_22_25",
                  employers=b["employer_id"].unique())
    bc = common_panel(b, expo_e, f"stock {band}")
    del b
    gc.collect()
    nc = int(bc["employer_id"].nunique())
    # DAIOE on the common employers
    bc["high"] = (bc["fq"] == 4).astype(int)
    bc, qterms = s78.eq2_terms(bc)
    g, v = fit(bc, f"daioe_common_{band.replace('-', '_')}", qterms, j47.FES)
    record(g, v, "D", "daioe_common", band, nc, [("hy", POST, INTERIM)])
    bc = bc.drop(columns=qterms)
    # Eloundou on the same employers
    bc["high"] = (bc["fq_e"] == 4).astype(int)
    bc, qterms = s78.eq2_terms(bc)
    clusters = CLUSTERS if band == BAND else CLUSTERS[:1]
    if band == BAND:
        bc = attach_industry(bc, s73, s80, "stock")
    for cl, sfx in clusters:
        print(f"\n  ELOUNDOU CLASSIFICATION, stock {band}, clustered by "
              f"{'employer' if sfx == '' else 'industry'}:")
        g, v = fit(bc, f"eloundou{sfx}_{band.replace('-', '_')}", qterms,
                   j47.FES, cluster=cl)
        record(g, v, "E", f"eloundou{sfx}", band, nc, [("hy", POST, INTERIM)])
    del bc
    gc.collect()


def sex(sexc, expo_d, expo_e, s67, s73, s78, s80, j47) -> None:
    print(f"\n  SEX GATE at {BAND}:")
    skel = s67.build_skeleton_sex(sexc, BAND, j47, "n_emp")
    b = s78.with_exposure(skel, expo_d)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError("the sex gate's panel is empty")
    n = int(b["employer_id"].nunique())
    b, qterms = s78.gender_eq2_terms(b)
    g, v = fit(b, "sex_gate_22_25", qterms, j47.FES)
    record(g, v, "G", "sex_gate", BAND, n,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    got = {"post": get("G", "sex_gate", BAND, "hyf_post"),
           "tau": get("G", "sex_gate", BAND, "hyf_tau")}
    bad = check("female differential", got, SEX_GATE)
    if bad:
        stop(bad, "SEX")
    print(f"  THE SEX GATE PASSES: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    b = b.drop(columns=qterms)
    bc = common_panel(b, expo_e, "sex")
    del b
    gc.collect()
    nc = int(bc["employer_id"].nunique())
    bc["high"] = (bc["fq"] == 4).astype(int)
    bc, qterms = s78.gender_eq2_terms(bc)
    g, v = fit(bc, "sex_daioe_common_22_25", qterms, j47.FES)
    record(g, v, "D", "sex_daioe_common", BAND, nc,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    bc = bc.drop(columns=qterms)
    bc["high"] = (bc["fq_e"] == 4).astype(int)
    bc, qterms = s78.gender_eq2_terms(bc)
    bc = attach_industry(bc, s73, s80, "sex")
    for cl, sfx in CLUSTERS:
        print(f"\n  ELOUNDOU CLASSIFICATION, sex panel, clustered by "
              f"{'employer' if sfx == '' else 'industry'}:")
        g, v = fit(bc, f"sex_eloundou{sfx}_22_25", qterms, j47.FES, cluster=cl)
        record(g, v, "E", f"sex_eloundou{sfx}", BAND, nc,
               [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    del bc
    gc.collect()


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def movement(label, band, gate_spec, d_spec, e_spec, term, table1=None) -> list:
    """E1 to E3: the gate, DAIOE and Eloundou on the same employers, the
    Eloundou fit's second clustering where it exists, and the movement in
    gate SEs."""
    L = [f"  {label}:"]
    c0, s0 = get("G", gate_spec, band, term) if gate_spec else (np.nan, np.nan)
    if gate_spec and c0 == c0:
        L.append(f"    gate, Table 1's sample     tau {c0:+.4f}  SE {s0:.4f} by employer")
    elif table1 is not None:
        c0, s0 = table1
        L.append(f"    Table 1 (printed)          tau {c0:+.4f}  SE {s0:.4f} by employer")
    cd, sd = get("D", d_spec, band, term)
    ce, se = get("E", e_spec, band, term)
    _, sei = get("E", e_spec + "_indcl", band, term)
    if not (cd == cd and ce == ce):
        return L + ["    NO COMPARISON, a fit is missing (not a null)"]
    L.append(f"    DAIOE, common employers    tau {cd:+.4f}  SE {sd:.4f} by employer")
    L.append(f"    Eloundou, same employers   tau {ce:+.4f}  SE {se:.4f} by employer"
             + (f", {sei:.4f} by industry" if sei == sei else ""))
    d = ce - cd
    ref = s0 if s0 == s0 and s0 > 0 else sd
    L.append(f"    difference (Eloundou minus DAIOE, same employers) {d:+.4f}, "
             f"which is {abs(d) / ref if ref else float('nan'):.2f} of the "
             f"gate's employer-clustered SE; its own SE is not available from "
             f"two separate covariances")
    return L


def agreement_lines(pop: str) -> list:
    a = {r["statistic"]: r["value"] for r in AGREE if r["population"] == pop}
    if not a:
        return [f"  {pop}: not computed"]
    f = lambda k, fmt="{:.3f}": (fmt.format(a[k]) if k in a and a[k] == a[k]  # noqa: E731
                                 else "n/a")
    return [f"  {pop}: {f('n_scored_both', '{:,.0f}')} employers scored on both "
            f"(DAIOE {f('n_scored_daioe', '{:,.0f}')}, Eloundou "
            f"{f('n_scored_eloundou', '{:,.0f}')})",
            f"    same quartile: {f('share_employers_same_quartile')} of employers, "
            f"{f('share_employment_same_quartile')} of incumbent employment",
            f"    top quartiles: {f('n_top_both', '{:,.0f}')} in both; of DAIOE's top "
            f"{f('share_top_daioe_also_top_eloundou')} also Eloundou's (by "
            f"employment {f('employment_share_top_daioe_also_top_eloundou')}); "
            f"of Eloundou's top {f('share_top_eloundou_also_top_daioe')} also "
            f"DAIOE's; Jaccard {f('jaccard_top_quartiles')}",
            f"    employer scores: Spearman {f('spearman_employer_scores')}, "
            f"Pearson {f('pearson_employer_scores')}"]


def write_summary() -> None:
    L = ["THE HEADLINE DESIGN WITH EMPLOYERS CLASSIFIED BY THE ELOUNDOU",
         "RATING IN PLACE OF DAIOE (LANE 38d)",
         "=" * 66, "",
         "tau = later-period coefficient minus interim, SE from the clustered",
         "covariance of the same fit (V_LL + V_II - 2 V_LI). Same cascade of",
         "2019 codes, same three-digit book, same incumbent floor and the same",
         "employment-weighted quartile cut; each index keeps its own cut and",
         "the fits run on the employers both indices score.", "",
         "GATES (DAIOE, Table 1's specification and sample):"]
    c, s = get("G", "gate", BAND, "hy_tau")
    if c == c:
        L.append(f"  stock {BAND}: tau {c:+.4f} ({s:.4f}); Table 1 "
                 f"{GATE['tau'][0]:+.4f} ({GATE['tau'][1]:.4f})")
    c, s = get("G", "sex_gate", BAND, "hyf_tau")
    if c == c:
        L.append(f"  sex {BAND}: female differential tau {c:+.4f} ({s:.4f}); "
                 f"Table 1 {SEX_GATE['tau'][0]:+.4f} ({SEX_GATE['tau'][1]:.4f})")
    L += ["", "E. THE ESTIMATES (read rules E1 to E3):"]
    L += movement("tau at 22-25 (High x Young)", BAND, "gate", "daioe_common",
                  "eloundou", "hy_tau")
    L += movement("the female differential at 22-25 (High x Young x Female)",
                  BAND, "sex_gate", "sex_daioe_common", "sex_eloundou", "hyf_tau")
    L += movement("tau at 26-30 (High x Young), employer clustering only",
                  BAND2, None, "daioe_common", "eloundou", "hy_tau",
                  table1=TABLE1_26_30)
    L += ["", "A. HOW THE TWO CLASSIFICATIONS AGREE (read rule E4):"]
    L += agreement_lines("all_scored_employers")
    L += agreement_lines("estimation_panel_22_25")
    L += ["  The 4 x 4 quartile cross-tabulation is in classification_agreement.csv",
          "  (cells under the floor suppressed)."]
    L.append("")
    L += [f"FITS: {DONE} of {PLANNED} attempted came back ({PLANNED_FITS} planned). "
          "A run far shorter than the estimate (2.5 to 3 hours) is a failure, "
          "not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "103_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "103_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("103: THE HEADLINE DESIGN ON THE ELOUNDOU CLASSIFICATION (LANE 38d)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s73, s78, s80, l47, l70, j47 = load_modules()
        if not (Path(mc.SHARE) / ELOUNDOU_FILE).exists():
            raise RuntimeError(f"{ELOUNDOU_FILE} is not at the project root; "
                               "nothing to classify with")
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
        expo_d, expo_e = build_both(s82, l47, l70, j47)
        agreement(expo_d, expo_e, "all_scored_employers")
        # One panel at a time: Python never holds two while R fits.
        counts = load_counts("L_counts", s61.PANEL_YEARS, COUNT_COLS)
        if counts is None:
            raise RuntimeError("L_counts_2021-2025 missing; run 47L")
        last = str(counts["year_month"].max())
        if last < "2025-06":
            raise RuntimeError(f"the counts end at {last}")
        stock(counts, expo_d, expo_e, BAND, s61, s73, s78, s80, j47, gate=True)
        stock(counts, expo_d, expo_e, BAND2, s61, s73, s78, s80, j47, gate=False)
        del counts
        gc.collect()
        sexc = load_counts("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
        if sexc is None:
            raise RuntimeError(f"L_counts_sex unreadable or lacking {SEX_COLS}")
        sex(sexc, expo_d, expo_e, s67, s73, s78, s80, j47)
        del sexc
        gc.collect()
        drain(s78, "78")
        drain(s73, "73")
    except SystemExit:
        mc.runlog("103_eloundou_classification", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"103 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("103_eloundou_classification", rc, (time.time() - T0) / 60)
    print("\n103 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

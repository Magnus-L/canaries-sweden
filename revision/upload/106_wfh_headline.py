#!/usr/bin/env python3
"""
106_wfh_headline.py -- lane 39b: AI exposure against working from home on
                       the headline design itself.

======================================================================
  RUNS IN MONA (lane 39b). Output folder CANARIES_106_OUT (default
  output_106). SQL: none. Every frame is a cache on the share: 47L's
  L_counts_2021-2025, 67's L_counts_sex_2021-2025, 82's cascade and
  baseline caches. Input file: dingel_neiman_ssyk4 (.dta/.csv/.txt) at
  the project root, the file script 89 read.
======================================================================

QUESTION (the editor's read of 26 September 2026, on referee point R1.7)
The letter says the employer-level AI and teleworkability scores correlate
at 0.88 and offers an occupation-scaled check in place of a joint
specification on the headline model. The editor asks for the headline
employer exposure indicator kept, with a predetermined employer
teleworkability score interacted with the young indicator and the same
period and seasonal terms, against a same-sample baseline; wide intervals
to be reported as such, and the female specification extended the same
way. Script 89 (23 Sep) ran the split-sample analogue on the step from
the tightening months; this script runs the joint model and the
split-sample cells on tau = later minus interim.

THE TWO SCORES
Both come from 82's build_exposure() on the same 2019 incumbents aged 31
to 69, the same three-digit book, the same floor and the same
employment-weighted quartile cut: the DAIOE generative-AI percentile for
one, the Dingel and Neiman teleworkable share for the other. The
continuous teleworkability score z_wfh is the employer mean, standardised
over the employers that carry both scores, weighted by incumbent
employment (97's Part Q convention).

THE DESIGN
  G  the stock gate at 22-25 (Table 1 within 0.0005; hard stop) and the
     sex gate (the female differential within 0.0005; hard stop).
  J  on the employers both indices score (the common sample):
     J1 AI only, Equation (2) as Table 1 (the same-sample baseline);
     J2 joint: Equation (2)'s terms x High x Young PLUS the same terms
        x z_wfh x Young, in one fit;
     J3 teleworkability only, the same terms x z_wfh x Young.
     tau_AI and tau_WFH (per SD of the teleworkability score) from each.
  O  the split-sample cells, on tau: the AI step among employers in the
     LOW-teleworkability half and in the HIGH half; the teleworkability
     step (top quartile of that score) among employers in the LOW-AI half
     and in the HIGH-AI half, halves cut at the medians of the two
     employer scores. Fits 1 and 3 discriminate, as in 89; a half in
     which the indicator does not vary is reported as not estimable.
  S  the female differential: S1 the sex specification on the common
     sample (baseline); S2 joint, the sex terms for High and for z_wfh
     in one fit. tau for High x Young x Female and z_wfh x Young x Female.
Eleven fits, employer clustering throughout.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the summary)
  J1. tau_AI from the joint fit is reported beside the AI-only tau on the
      same employers, with both SEs, and tau_WFH beside it per SD; the
      correlation of the two scores is printed with them. A wide
      interval is reported as wide; it is neither a zero nor a defence.
  O1. 89's rule: AI is called the operative score only if the young
      decline appears among employers in the LOW-teleworkability half
      and not among employers in the LOW-AI half; otherwise the cells
      are reported without a verdict. A cell that is imprecise is
      imprecise, never a zero.
  S1. As J1 for the female differential.
  Employer counts below five are suppressed with their statistic.

EXPORT (output_106/)
  wfh_headline.csv         every reported term and tau (parts G, J, O, S)
  wfh_scores_overlap.csv   the two scores: correlations, medians, cells
  106_summary.txt, 106_log.txt; vcov_s106_*.csv stay on the share

IN THE PAPER
OA III.2 (one paragraph beside the remote-work reading of the oldest
band's gain), letter R1.7, one clause in Section 3.

    python 106_wfh_headline.py
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

OUT = HERE / os.environ.get("CANARIES_106_OUT", "output_106")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
BAND = "22-25"
POST_FROM = "2024-01"                       # asserted against 78 below
COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]
WFH_FILES = ("dingel_neiman_ssyk4.dta", "dingel_neiman_ssyk4.csv",
             "dingel_neiman_ssyk4.txt")
WFH_COLS = ("teleworkable", "telework", "wfh", "teleworkable_share")

GATE = {"post": (-0.0578, 0.0155), "tau": (-0.0399, 0.0102)}
SEX_GATE = {"post": (-0.0858, 0.0142), "tau": (-0.0714, 0.0109)}
GATE_TOL = 0.0005

POST, INTERIM = "post_x_high_x_young", "interim_x_high_x_young"
WPOST, WINTERIM = "post_x_highwfh_x_young", "interim_x_highwfh_x_young"
FPOST, FINTERIM = POST + "_x_female", INTERIM + "_x_female"
WFPOST, WFINTERIM = WPOST + "_x_female", WINTERIM + "_x_female"
PLANNED_FITS = 11

NOTES: list = []
FAILURES: list = []
ROWS: list = []
OVERLAP: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATES. Stock 22-25: Table 1 within 0.0005, later -0.0578 (0.0155),",
    "  tau -0.0399 (0.0102). Sex 22-25: -0.0858 (0.0142), tau -0.0714",
    "  (0.0109). A miss is a hard stop; nothing from the run is quotable.",
    "  J1. tau_AI from the joint fit beside the AI-only tau on the same",
    "  employers, both SEs, and tau_WFH per SD beside it, with the",
    "  correlation of the two scores. A wide interval is reported as wide;",
    "  it is neither a zero nor a defence.",
    "  O1. 89's rule: AI is the operative score only if the young decline",
    "  appears among employers in the LOW-teleworkability half and not",
    "  among employers in the LOW-AI half; otherwise no verdict. An",
    "  imprecise cell is imprecise, never a zero.",
    "  S1. As J1 for the female differential.",
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
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}")
    return s82, s61, s67, s78, l47, l70, j47


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
    if not df.empty:
        had = df["n_firms"].notna()
        df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
        small = had & df["n_firms"].isna()
        if small.any():
            df.loc[small, ["coef", "se", "t", "var_post", "var_interim",
                           "cov_post_interim"]] = np.nan
    df.to_csv(OUT / "wfh_headline.csv", index=False)
    pd.DataFrame(OVERLAP, columns=["item", "value"]).to_csv(
        OUT / "wfh_scores_overlap.csv", index=False)
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
        r = mc.run_fepois_multi(b, OUT, tag=f"s106_{tag}", terms=terms,
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


def record(g, v, part, spec, band, n_firms, pairs) -> None:
    """pairs: (label, post term, interim term) whose tau is recorded with
    the two coefficients."""
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
    raise SystemExit(f"106: the {what.lower()} gate failed; stopping.")


def load_counts(prefix: str, years, require):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# the two scores (89's book, 82's chain)
# ----------------------------------------------------------------------

def wfh_book() -> pd.DataFrame:
    """The teleworkability score in the shape build_exposure() expects:
    ssyk4 (four characters) and `score`. Dingel and Neiman's share runs 0
    to 1 where DAIOE's percentile runs 0 to 100; the scale does not enter
    the quartile, and the continuous use standardises."""
    for name in WFH_FILES:
        p = Path(mc.SHARE) / name
        if not p.exists():
            continue
        d = pd.read_stata(str(p)) if p.suffix == ".dta" else pd.read_csv(p)
        col = next((c for c in d.columns if c.lower() in WFH_COLS), None)
        if col is None:
            raise RuntimeError(f"{name} carries no teleworkable column; "
                               f"found {list(d.columns)}")
        d["ssyk4"] = d["ssyk4"].astype(int).astype(str).str.zfill(4)
        d = d.rename(columns={col: "score"})[["ssyk4", "score"]]
        d = d[d["score"].notna()].drop_duplicates("ssyk4")
        print(f"  teleworkability: {len(d)} four-digit codes from {name}, "
              f"score {d['score'].min():.2f} to {d['score'].max():.2f}")
        return d
    raise RuntimeError("dingel_neiman_ssyk4 is not at the project root")


def build_both(s82, l47, l70, j47) -> tuple:
    """DAIOE and teleworkability exposure from one chain; nothing is
    written to the share."""
    print("\n  SCORE 1: DAIOE (the paper's classification)")
    built = s82.build_exposure(l47, l70, j47, audit=False)
    drain(s82, "82/daioe")
    expo_a = built["exposure"]
    del built
    gc.collect()
    print("\n  SCORE 2: teleworkability, same chain")
    built = s82.build_exposure(l47, l70, j47, daioe=wfh_book(), audit=False)
    drain(s82, "82/wfh")
    expo_w = built["exposure"]
    del built
    gc.collect()
    for name, e in (("DAIOE", expo_a), ("teleworkability", expo_w)):
        print(f"  {name}: {len(e):,} employers scored, "
              f"{int((e['fq'] == 4).sum()):,} in the top quartile")
    return expo_a, expo_w


def overlap(expo_a: pd.DataFrame, expo_w: pd.DataFrame) -> pd.DataFrame:
    """The employers both indices score, with the WFH quartile, the two
    median halves and z_wfh (incumbent-weighted standardisation). The
    overlap statistics go to wfh_scores_overlap.csv before any fit."""
    m = expo_a[["employer_id", "mix", "fq", "n"]].merge(
        expo_w[["employer_id", "mix", "fq"]], on="employer_id",
        suffixes=("_ai", "_wfh"))
    if m.empty:
        raise RuntimeError("the two scores share no employer")
    w = m["n"].astype(float)
    mu = float(np.average(m["mix_wfh"], weights=w))
    sd = float(np.sqrt(np.average((m["mix_wfh"] - mu) ** 2, weights=w)))
    m["z_wfh"] = (m["mix_wfh"] - mu) / (sd if sd > 0 else 1.0)
    med_a, med_w = float(m["mix_ai"].median()), float(m["mix_wfh"].median())
    m["hi_ai"] = (m["mix_ai"] > med_a).astype(int)
    m["hi_wfh"] = (m["mix_wfh"] > med_w).astype(int)
    m["high_w"] = (m["fq_wfh"] == 4).astype(int)
    rho = float(m["mix_ai"].corr(m["mix_wfh"], method="spearman"))
    pear = float(m["mix_ai"].corr(m["mix_wfh"]))
    same_q = float((m["fq_ai"] == m["fq_wfh"]).mean())
    top_both = int(((m["fq_ai"] == 4) & (m["fq_wfh"] == 4)).sum())
    OVERLAP[:] = [{"item": "n_employers_both_scores", "value": float(len(m))},
                  {"item": "spearman", "value": rho}, {"item": "pearson", "value": pear},
                  {"item": "median_ai", "value": med_a}, {"item": "median_wfh", "value": med_w},
                  {"item": "wfh_mean_weighted", "value": mu}, {"item": "wfh_sd_weighted", "value": sd},
                  {"item": "share_same_quartile", "value": same_q},
                  {"item": "n_top_ai", "value": float((m["fq_ai"] == 4).sum())},
                  {"item": "n_top_wfh", "value": float(m["high_w"].sum())},
                  {"item": "n_top_both", "value": float(top_both)}]
    for ha in (0, 1):
        for hw in (0, 1):
            n = int(((m["hi_ai"] == ha) & (m["hi_wfh"] == hw)).sum())
            OVERLAP.append({"item": f"cell_ai{ha}_wfh{hw}",
                            "value": float(n if (n == 0 or n >= FLOOR) else np.nan)})
    NOTES.append(f"the two employer scores correlate at Spearman {rho:+.3f} (Pearson "
                 f"{pear:+.3f}) on {len(m):,} employers; {same_q:.1%} share a quartile; "
                 f"z_wfh standardised with incumbent weights, mean {mu:.3f}, SD {sd:.3f}")
    print(f"  {NOTES[-1]}")
    save()
    return m[["employer_id", "z_wfh", "hi_ai", "hi_wfh", "high_w"]]


def common_panel(b: pd.DataFrame, both: pd.DataFrame, tag: str) -> pd.DataFrame:
    n0 = b["employer_id"].nunique()
    bc = b.merge(both, on="employer_id", how="inner")
    n1 = bc["employer_id"].nunique()
    NOTES.append(f"{tag}: {n0:,} employers in the DAIOE panel, {n1:,} also scored "
                 f"on teleworkability ({n0 - n1:,} lost)")
    print(f"  common sample: {n1:,} of {n0:,} employers")
    if bc.empty:
        raise RuntimeError(f"{tag}: no employer carries both scores")
    return bc


def gender_terms(b: pd.DataFrame, high_col: str = "high", suffix: str = "") -> tuple:
    """78's gender_eq2_terms for any exposure column: every period term
    x Exposure x Young, x Exposure x Female and x Exposure x Young x
    Female. With high_col='high' and no suffix it is 78's term set."""
    ym = b["year_month"].astype(str)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    post_any = ym >= mc.CHATGPT_YM
    periods = {"rb": (ym >= mc.RIKSBANK_YM).astype(int),
               "interim": (post_any & (ym < POST_FROM)).astype(int),
               "post": (ym >= POST_FROM).astype(int)}
    for qq in (1, 2, 3):
        periods[f"q{qq}"] = (q == qq).astype(int)
    h = b[high_col]
    hy, hf = h * b["young"], h * b["female"]
    hyf = hy * b["female"]
    terms = []
    for p, ind in periods.items():
        b[f"{p}_x_high{suffix}_x_young"] = ind * hy
        b[f"{p}_x_high{suffix}_x_female"] = ind * hf
        b[f"{p}_x_high{suffix}_x_young_x_female"] = ind * hyf
        terms += [f"{p}_x_high{suffix}_x_young", f"{p}_x_high{suffix}_x_female",
                  f"{p}_x_high{suffix}_x_young_x_female"]
    return b, terms


# ----------------------------------------------------------------------
# G and J: the gate and the joint fits on the stock panel
# ----------------------------------------------------------------------

def stock_gate(counts, expo_a, s61, s78, j47) -> pd.DataFrame:
    print(f"\n  G. THE STOCK GATE at {BAND}:")
    skel = s61.build_skeleton(counts, BAND, j47)
    b = s78.with_exposure(skel, expo_a)
    del skel
    gc.collect()
    if b.empty:
        raise RuntimeError(f"the {BAND} panel is empty")
    n = int(b["employer_id"].nunique())
    b, qterms = s78.eq2_terms(b)
    g, v = fit(b, "gate_22_25", qterms, j47.FES)
    record(g, v, "G", "gate", BAND, n, [("hy", POST, INTERIM)])
    got = {"post": get("G", "gate", BAND, "hy_post"),
           "tau": get("G", "gate", BAND, "hy_tau")}
    bad = check(BAND, got, GATE)
    if bad:
        stop(bad, "STOCK")
    print(f"  THE STOCK GATE PASSES at {BAND}: tau {got['tau'][0]:+.4f} "
          f"({got['tau'][1]:.4f})")
    return b.drop(columns=qterms)


def joint(b0: pd.DataFrame, both: pd.DataFrame, j47, s78) -> None:
    """J1 AI only, J2 joint, J3 teleworkability only, on the common
    employers of the gate's panel."""
    print("\n  J. THE JOINT SPECIFICATION on the common sample:")
    bc = common_panel(b0, both, "stock")
    n = int(bc["employer_id"].nunique())
    bc, ta = s78.eq2_terms(bc, "high", "")
    g, v = fit(bc, "ai_only_22_25", ta, j47.FES)
    record(g, v, "J", "ai_only", BAND, n, [("hy", POST, INTERIM)])
    bc, tw = s78.eq2_terms(bc, "z_wfh", "wfh")
    g, v = fit(bc, "joint_22_25", ta + tw, j47.FES)
    record(g, v, "J", "joint", BAND, n,
           [("hy", POST, INTERIM), ("wfh", WPOST, WINTERIM)])
    g, v = fit(bc, "wfh_only_22_25", tw, j47.FES)
    record(g, v, "J", "wfh_only", BAND, n, [("wfh", WPOST, WINTERIM)])
    del bc
    gc.collect()


# ----------------------------------------------------------------------
# O: the split-sample cells on tau
# ----------------------------------------------------------------------

def cell(b0: pd.DataFrame, both: pd.DataFrame, tag: str, high_col: str,
         half_col: str, half_val: int, j47, s78) -> None:
    """Equation (2) on the employers of one half, with `high_col` as the
    exposure indicator. A half in which the indicator does not vary is
    recorded as not estimable and not fitted."""
    ids = set(both.loc[both[half_col] == half_val, "employer_id"])
    b = b0[b0["employer_id"].isin(ids)]
    if b.empty:
        add("O", tag, BAND, "hy_tau", np.nan, np.nan, 0, 0, "empty_half")
        NOTES.append(f"O/{tag}: the half is empty on this panel")
        save()
        return
    b = b.merge(both[["employer_id", "high_w"]], on="employer_id", how="inner")
    n = int(b["employer_id"].nunique())
    per_emp = b.groupby("employer_id")[high_col].first()
    n_high = int(per_emp.sum())
    if n_high == 0 or n_high == len(per_emp):
        add("O", tag, BAND, "hy_tau", np.nan, np.nan, len(b), n, "not_estimable")
        NOTES.append(f"O/{tag}: {high_col} does not vary among the {n:,} employers "
                     f"of the half ({n_high:,} high); not fitted")
        print(f"    {tag}: not estimable ({n_high:,} of {n:,} employers high)")
        save()
        return
    b, terms = s78.eq2_terms(b, high_col, "")
    g, v = fit(b, f"{tag}_22_25", terms, j47.FES)
    del b
    gc.collect()
    record(g, v, "O", tag, BAND, n, [("hy", POST, INTERIM)])
    c, s = get("O", tag, BAND, "hy_tau")
    if c == c:
        print(f"    {tag:<16} tau {c:+.4f} ({s:.4f}) t {tstat(c, s):+.2f}   "
              f"employers {n:,} ({n_high:,} high)")


CELLS = [("ai_in_low_wfh", "high", "hi_wfh", 0, True),
         ("ai_in_high_wfh", "high", "hi_wfh", 1, False),
         ("wfh_in_low_ai", "high_w", "hi_ai", 0, True),
         ("wfh_in_high_ai", "high_w", "hi_ai", 1, False)]


def offdiagonal(b0: pd.DataFrame, both: pd.DataFrame, j47, s78) -> None:
    print("\n  O. THE SPLIT-SAMPLE CELLS on tau (fits 1 and 3 discriminate):")
    for tag, high_col, half_col, half_val, _ in CELLS:
        cell(b0, both, tag, high_col, half_col, half_val, j47, s78)


# ----------------------------------------------------------------------
# S: the female differential
# ----------------------------------------------------------------------

def sex(sexc, expo_a, both, s67, s78, j47) -> None:
    print(f"\n  S. SEX GATE at {BAND}:")
    skel = s67.build_skeleton_sex(sexc, BAND, j47, "n_emp")
    b = s78.with_exposure(skel, expo_a)
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
    print(f"  THE SEX GATE PASSES: tau {got['tau'][0]:+.4f} ({got['tau'][1]:.4f})")
    b = b.drop(columns=qterms)
    bc = common_panel(b, both, "sex")
    del b
    gc.collect()
    nc = int(bc["employer_id"].nunique())
    bc, ta = gender_terms(bc, "high", "")
    g, v = fit(bc, "sex_ai_only_22_25", ta, j47.FES)
    record(g, v, "S", "sex_ai_only", BAND, nc,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM)])
    bc, tw = gender_terms(bc, "z_wfh", "wfh")
    g, v = fit(bc, "sex_joint_22_25", ta + tw, j47.FES)
    record(g, v, "S", "sex_joint", BAND, nc,
           [("hy", POST, INTERIM), ("hyf", FPOST, FINTERIM),
            ("wfh", WPOST, WINTERIM), ("wfhf", WFPOST, WFINTERIM)])
    del bc
    gc.collect()


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def line(label, part, spec, term) -> str:
    c, s = get(part, spec, BAND, term)
    if c != c:
        return f"  {label:<44} NO FIT (not a null)"
    return f"  {label:<44} {c:+.4f} ({s:.4f})  t {tstat(c, s):+.2f}  [{c - 1.96 * s:+.3f}, {c + 1.96 * s:+.3f}]"


def write_summary() -> None:
    ov = {r["item"]: r["value"] for r in OVERLAP}
    L = ["AI EXPOSURE AGAINST WORKING FROM HOME ON THE HEADLINE DESIGN",
         "(LANE 39b)", "=" * 66, "",
         "tau = later minus interim from one fit, SE from the clustered",
         "covariance (V_LL + V_II - 2 V_LI), employer clustering. Both scores",
         "from 82's chain on the same 2019 incumbents; z_wfh is the employer",
         "teleworkability mean standardised with incumbent weights.", "",
         "GATES (DAIOE, Table 1's specification and sample):"]
    c, s = get("G", "gate", BAND, "hy_tau")
    L.append(f"  stock {BAND}: tau {c:+.4f} ({s:.4f}); Table 1 {GATE['tau'][0]:+.4f} "
             f"({GATE['tau'][1]:.4f})" if c == c else "  stock: not reached")
    c, s = get("G", "sex_gate", BAND, "hyf_tau")
    L.append(f"  sex {BAND}: female differential tau {c:+.4f} ({s:.4f}); Table 1 "
             f"{SEX_GATE['tau'][0]:+.4f} ({SEX_GATE['tau'][1]:.4f})" if c == c
             else "  sex: not reached")
    if ov:
        L += ["", "THE TWO SCORES:",
              f"  {ov.get('n_employers_both_scores', float('nan')):,.0f} employers carry both; "
              f"Spearman {ov.get('spearman', float('nan')):+.3f}, Pearson "
              f"{ov.get('pearson', float('nan')):+.3f}; {ov.get('share_same_quartile', float('nan')):.1%} "
              f"in the same quartile; top quartiles: {ov.get('n_top_ai', float('nan')):,.0f} AI, "
              f"{ov.get('n_top_wfh', float('nan')):,.0f} teleworkability, "
              f"{ov.get('n_top_both', float('nan')):,.0f} both",
              "  median halves (AI x teleworkability): "
              + ", ".join(f"ai{a}_wfh{w} {ov.get(f'cell_ai{a}_wfh{w}', float('nan')):,.0f}"
                          for a in (0, 1) for w in (0, 1))]
    L += ["", "J. THE JOINT SPECIFICATION, common employers (read rule J1):",
          line("AI only (same-sample baseline), tau_AI", "J", "ai_only", "hy_tau"),
          line("joint, tau_AI (High x Young)", "J", "joint", "hy_tau"),
          line("joint, tau_WFH per SD (z_wfh x Young)", "J", "joint", "wfh_tau"),
          line("teleworkability only, tau_WFH per SD", "J", "wfh_only", "wfh_tau")]
    L += ["", "O. THE SPLIT-SAMPLE CELLS on tau (read rule O1; 1 and 3 discriminate):"]
    for tag, high_col, half_col, half_val, disc in CELLS:
        c, s = get("O", tag, BAND, "hy_tau")
        st = next((r["status"] for r in ROWS if r["part"] == "O" and r["spec"] == tag
                   and r["term"] == "hy_tau"), "missing")
        mark = "  <- discriminating" if disc else ""
        if c != c:
            L.append(f"  {tag:<16} {st.upper().replace('_', ' ')}{mark}")
        else:
            nf = next((r["n_firms"] for r in ROWS if r["part"] == "O" and r["spec"] == tag
                       and r["term"] == "hy_tau"), np.nan)
            L.append(f"  {tag:<16} tau {c:+.4f} ({s:.4f}) t {tstat(c, s):+.2f}   employers "
                     f"{nf:,.0f}{mark}")
    c1, s1 = get("O", "ai_in_low_wfh", BAND, "hy_tau")
    c3, s3 = get("O", "wfh_in_low_ai", BAND, "hy_tau")
    if c1 == c1 and c3 == c3:
        ai_shows = c1 < 0 and abs(c1) > 2 * s1
        wfh_shows = c3 < 0 and abs(c3) > 2 * s3
        L.append("  89's rule: " + ("AI is the operative score (decline in the low-WFH half, "
                                    "none in the low-AI half)" if ai_shows and not wfh_shows
                                    else "NO VERDICT (the two discriminating cells do not "
                                         "separate the readings)"))
    L += ["", "S. THE FEMALE DIFFERENTIAL, common employers (read rule S1):",
          line("sex, AI only, tau_F (High x Young x Female)", "S", "sex_ai_only", "hyf_tau"),
          line("sex, joint, tau_F,AI", "S", "sex_joint", "hyf_tau"),
          line("sex, joint, tau_F,WFH per SD", "S", "sex_joint", "wfhf_tau"),
          line("sex, joint, tau_AI (High x Young)", "S", "sex_joint", "hy_tau"),
          line("sex, joint, tau_WFH per SD (z x Young)", "S", "sex_joint", "wfh_tau")]
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
    (OUT / "106_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "106_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("106: AI EXPOSURE AGAINST WORKING FROM HOME, HEADLINE DESIGN (LANE 39b)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        if not any((Path(mc.SHARE) / f).exists() for f in WFH_FILES):
            raise RuntimeError("dingel_neiman_ssyk4 is not at the project root; "
                               "nothing to score with")
        if not all((CACHE / f"L_counts_sex_{y}.parquet").exists()
                   for y in s61.PANEL_YEARS):
            raise RuntimeError("L_counts_sex_* are not all on the share; this "
                               "script does no SQL for them (run 67)")
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        expo_a, expo_w = build_both(s82, l47, l70, j47)
        both = overlap(expo_a, expo_w)
        counts = load_counts("L_counts", s61.PANEL_YEARS, COUNT_COLS)
        if counts is None:
            raise RuntimeError("L_counts_2021-2025 missing; run 47L")
        last = str(counts["year_month"].max())
        if last < "2025-06":
            raise RuntimeError(f"the counts end at {last}")
        b0 = stock_gate(counts, expo_a, s61, s78, j47)
        del counts
        gc.collect()
        joint(b0, both, j47, s78)
        offdiagonal(b0, both, j47, s78)
        del b0
        gc.collect()
        sexc = load_counts("L_counts_sex", s61.PANEL_YEARS, SEX_COLS)
        if sexc is None:
            raise RuntimeError(f"L_counts_sex unreadable or lacking {SEX_COLS}")
        sex(sexc, expo_a, both, s67, s78, j47)
        del sexc
        gc.collect()
        drain(s78, "78")
    except SystemExit:
        mc.runlog("106_wfh_headline", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"106 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    save()
    write_summary()
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("106_wfh_headline", rc, (time.time() - T0) / 60)
    print("\n106 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

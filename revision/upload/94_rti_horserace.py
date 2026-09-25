#!/usr/bin/env python3
"""
94_rti_horserace.py -- generative-AI exposure against routine-task
                       intensity, on the paper's headline design.

======================================================================
  RUNS IN MONA. Output folder CANARIES_94_OUT (default output_94); parts
  with CANARIES_94_PARTS (default ABC). No SQL when the caches of 47L,
  67 and 82 are on the share: every register input is a cache.
======================================================================

QUESTION
Is the young workers' decline specific to what generative AI can do, or
is it the continuation of routine-biased change, or a cycle that hits
routine jobs? The task literature measures routine work by the
routine-task intensity of Autor and Dorn (2013), RTI = ln(routine) -
ln(manual) - ln(abstract) task input (Acemoglu and Autor 2011 on the
task framework). If the young decline is routine-biased change, an
employer ranked by the RTI of its incumbents' occupations should show it,
and should absorb the DAIOE ranking when the two are entered together.

WHY THIS IS A HORSE RACE AND 89 WAS NOT
Script 89 declined to run a joint regression on teleworkability because
the two firm scores correlate at +0.88 there. RTI is a different case:
across the 417 SSYK occupations both measures score, RTI and the DAIOE
generative-AI percentile correlate at +0.05 (employment-weighted +0.05,
Spearman -0.01; revision/local/l54_rti_ssyk4.py), because routine MANUAL
work (machine operators, assemblers, drivers) is high in RTI and low in
generative-AI exposure, while professional work is the reverse. Clerical
support is high in both. With the scores nearly orthogonal, the joint
regression is informative and is the test. Part A measures the
separation at the employer level before any fit.

THE TWO SCORES
Built by 82's build_exposure(), the one function every occupation-route
script calls: the same employers, the 2019 freeze, the incumbents aged 31
to 69, the backward cascade, the uniform three-digit book, the floor of
five person-months, the quartile cut points weighted by incumbent
employment. Only the occupation score book differs: the DAIOE genAI
percentile (daioe_quartiles.dta) for one and the Autor-Dorn RTI
(rti_ssyk4.dta) for the other. RTI enters the same book-building step, so
an employer's RTI is the employment-weighted mean over its incumbents of
the 2019 national employment-weighted RTI of their three-digit groups.

PART A. THE SEPARATION, NO FIT. The employer-level correlation of the two
firm means (Pearson and Spearman, unweighted and weighted by incumbent
employment), the four cells of a median cut on both with the share of
employers off the diagonal, the share in the same quartile and the
employers in the top quartile of both.

PART B. THE HORSE RACE, AT 22-25 AND 26-30. Equation (2) exactly as in
Table 1 (82's panel and 78's term set: tightening switch from April 2022,
interim window December 2022 to December 2023, adoption step from
January 2024, three calendar-quarter terms, each x High x Young;
employer-by-month, employer-by-age and month-by-age effects; Poisson;
clustered by employer). Four specifications per band:
  (a) High on DAIOE only, on the paper's own sample: THE GATE. It must
      reproduce Table 1 (22-25: adoption step -0.0578 (0.0155), step from
      the 2023 level -0.0399 (0.0102); 26-30: -0.0482 (0.0104) and
      -0.0403 (0.0067)) within GATE_TOL, or the script stops before any
      other fit and nothing from it is quoted.
  (b) High on RTI only (top quartile of employer RTI, same weighting).
  (c) both entered jointly, each with its full set of Equation (2) terms.
  (d) (c) with the two continuous firm means standardised (mean zero, SD
      one across the employers of the fit, weighted by incumbent
      employment) in place of the two quartile indicators.
(b) to (d) run on the employers both scores cover. If that is not the
whole of (a)'s sample, (a) is refitted on it as (a2), so that (a2), (b)
and (c) differ in the regressors and nothing else.

PART C. THE SEX SPLIT UNDER (c), AT 22-25. 82's sex panel (67's skeleton,
employer by age-and-sex and month by age-and-sex effects: still three
fixed-effect dimensions, which is what the fepois ceiling counts). First
the DAIOE-only fit on the paper's sample, which must reproduce Table 1's
female differential (-0.0858 (0.0142) at adoption, -0.0714 (0.0109) from
the 2023 level) within GATE_TOL; if it does not, the joint sex fit is not
run. Then every term entered for both indicators, x High x Young,
x High x Female and x High x Young x Female.

READ RULES, FIXED BEFORE THE RUN (printed at the start and in the
summary). The rule is read on (c) at 22-25, on the step from the 2023
level, the number the paper reports; every other cell is reported and
settles nothing.
  1. AI-SPECIFIC if the DAIOE step is negative and distinguishable from
     zero at five per cent AND the RTI step is not negative and
     distinguishable from zero at five per cent.
  2. RTI ABSORBS if the RTI step is negative and distinguishable at five
     per cent AND the DAIOE step is not.
  3. BOTH if both are negative and distinguishable: the decline loads on
     both rankings and the paper says so.
  4. NEITHER otherwise: reported as it comes, never read as a zero.
  The share of (a2)'s DAIOE step that (c) retains is printed beside the
  verdict. An imprecise cell is imprecise, not an absence of effect.

INPUTS AND OUTPUTS
Reads rti_ssyk4.dta and daioe_quartiles.dta from the input folder
(mc.SHARE); the caches L_baseline_2019_cascade, L_baseline_2019 and
L_counts_2019 (82, 47L), L_counts_2021 to 2025 (47L) and L_counts_sex_2021
to 2025 (67). Writes to output_94/: rti_horserace.csv (every coefficient,
standard error, observation count and employer count, and Part A's
statistics, with the export floor applied), 94_summary.txt, 94_log.txt
and the vcov_s94_*.csv files (these stay on the share; the step standard
errors they give are already in the CSV).

IN THE PAPER
Online Appendix III.2, the rival explanations; Section 3, one clause in
the rivals paragraph.

    python 94_rti_horserace.py
"""

import gc
import hashlib
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

OUT = HERE / os.environ.get("CANARIES_94_OUT", "output_94")
OUT.mkdir(exist_ok=True)
# 82 is imported for its score builder; its own OUT is pointed here so the
# run leaves no stray output_82 behind (as 89 and 92 do).
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
PARTS = os.environ.get("CANARIES_94_PARTS", "ABC").upper()
CACHE = mc.CACHE_DIR

FLOOR = 5
BANDS = ["22-25", "26-30"]
SEX_BAND = "22-25"
SIG5 = 1.959963984540054

# The gate: what Table 1 prints, from export 2026-09-23_0655
# (occ_route_headline.csv, occ_route_gender.csv and their vcov files).
GATE = {
    "22-25": {"post": (-0.0578, 0.0155), "step": (-0.0399, 0.0102)},
    "26-30": {"post": (-0.0482, 0.0104), "step": (-0.0403, 0.0067)},
}
SEX_GATE = {"post": (-0.0858, 0.0142), "step": (-0.0714, 0.0109)}
GATE_TOL = 0.0005

# The RTI score file, built by revision/local/l54_rti_ssyk4.py and pinned
# here so that "the file that was built" is proved on arrival, as the
# DAIOE file is.
RTI_NAMES = ("rti_ssyk4.dta",)
RTI_SHA256 = "175269848a8e44066f1808b9e531e02265d8e2822717fa6949c3cad07f7751af"
RTI_MIN_CODES = 400

COUNT_COLS = ["employer_id", "year_month", "age_group", "n_emp"]
SEX_COLS = ["employer_id", "year_month", "age_group", "gender", "n_emp"]

RB = "rb_x_high{s}_x_young"
INTERIM = "interim_x_high{s}_x_young"
POST = "post_x_high{s}_x_young"

NOTES: list = []
FAILURES: list = []
ROWS: list = []
PLANNED = 0
DONE = 0
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  Read on (c), both indicators entered jointly, at 22-25, on the step",
    "  from the 2023 level (adoption minus interim), the number the paper",
    "  reports. Every other cell is reported and settles nothing.",
    "  1. AI-SPECIFIC if the DAIOE step is negative and distinguishable",
    "     from zero at five per cent AND the RTI step is not negative and",
    "     distinguishable at five per cent.",
    "  2. RTI ABSORBS if the RTI step is negative and distinguishable at",
    "     five per cent AND the DAIOE step is not.",
    "  3. BOTH if both are negative and distinguishable.",
    "  4. NEITHER otherwise, reported as it comes and never read as a zero.",
    "  THE GATE: (a) must reproduce Table 1 within "
    f"{GATE_TOL} at both bands",
    "  (22-25 -0.0578 (0.0155) and -0.0399 (0.0102); 26-30 -0.0482",
    "  (0.0104) and -0.0403 (0.0067)), or the script stops and nothing is",
    "  quoted. The sex split runs only if the DAIOE-only sex fit reproduces",
    "  -0.0858 (0.0142) and -0.0714 (0.0109).",
    f"  Employer counts below {FLOOR} are suppressed before anything leaves",
    "  MONA, and a statistic is suppressed with its count.",
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
    if hasattr(mod, "NOTES"):
        mod.NOTES.clear()
    if hasattr(mod, "FAILURES"):
        mod.FAILURES.clear()


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
        raise RuntimeError(f"78's adoption date is {s78.POST_FROM}, not "
                           f"2024-01; refusing to run.")
    return s82, s61, s67, s78, l47, l70, j47


def tstat(c, s) -> float:
    return float(c / s) if s and s == s and s > 0 else float("nan")


def save() -> pd.DataFrame:
    """The one export. Every write passes through the floor: an employer
    count below five is suppressed, and so is the statistic beside it."""
    df = pd.DataFrame(ROWS)
    if df.empty:
        df.to_csv(OUT / "rti_horserace.csv", index=False)
        return df
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    small = df["n_firms"].isna()
    if small.any():
        df.loc[small, ["coef", "se", "t"]] = np.nan
    df.to_csv(OUT / "rti_horserace.csv", index=False)
    return df


def add(block, spec, band, indicator, term, coef, se, n_obs, n_firms,
        status="ok"):
    ROWS.append({"block": block, "spec": spec, "young_band": band,
                 "indicator": indicator, "term": term,
                 "coef": float(coef) if coef == coef else np.nan,
                 "se": float(se) if se is not None and se == se else np.nan,
                 "t": tstat(coef, se), "n_obs": n_obs, "n_firms": n_firms,
                 "status": status})


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple):
    """One Poisson fit; (coefficients by term, clustered vcov or None).
    A failure is recorded and returns (None, None): a missing row is a
    missing fit, never a zero. R's own stderr is written in full beside
    the log by mona_common._r_failed, never truncated here."""
    global DONE
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms, "
          f"{len(terms)} terms{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s94_{tag}", terms=terms,
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


def step(g, v, suffix="", tail="") -> tuple:
    """Adoption minus interim for one indicator, with the standard error
    from the covariance: Var(a-b) = Vaa + Vbb - 2Vab."""
    p = POST.format(s=suffix) + tail
    i = INTERIM.format(s=suffix) + tail
    if g is None or p not in g.index or i not in g.index:
        return np.nan, np.nan
    c = float(g.loc[p, "coef"]) - float(g.loc[i, "coef"])
    if v is None or p not in v.index or i not in v.index:
        return c, np.nan
    var = (float(v.loc[p, p]) + float(v.loc[i, i]) - 2.0 * float(v.loc[p, i]))
    return c, float(np.sqrt(var)) if var > 0 else np.nan


def record_fit(g, v, spec, band, inds, n_firms, tail="", block="fit"):
    """Every tightening, interim and adoption term of each indicator, and
    the derived step from the 2023 level, into the export."""
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for ind, s in inds:
        for lab, pat in (("rb", RB), ("interim", INTERIM), ("post", POST)):
            t_ = pat.format(s=s) + tail
            if t_ in g.index:
                add(block, spec, band, ind, lab, g.loc[t_, "coef"],
                    g.loc[t_, "se"], n_obs, n_firms,
                    str(g.loc[t_].get("status", "ok")))
        c, se = step(g, v, s, tail)
        add(block, spec, band, ind, "step_from_2023", c, se, n_obs, n_firms,
            "derived")
    save()


def get(spec, band, ind, term, block="fit"):
    for r in ROWS:
        if (r["block"], r["spec"], r["young_band"], r["indicator"],
                r["term"]) == (block, spec, band, ind, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


# ----------------------------------------------------------------------
# the score
# ----------------------------------------------------------------------

def rti_book() -> pd.DataFrame:
    """
    The RTI file in the shape build_exposure() expects, ssyk4 and
    `score`, after a probe of what actually arrived: the hash, the
    columns, the dtypes, the code set. A file that fails any of these
    stops the script before anything else is read.
    """
    for name in RTI_NAMES:
        p = Path(mc.SHARE) / name
        if not p.exists():
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != RTI_SHA256:
            raise RuntimeError(f"{name}: sha256 {got[:12]} is not the pinned "
                               f"{RTI_SHA256[:12]}; the file that arrived is "
                               f"not the file that was built")
        d = pd.read_stata(str(p))
        print(f"  RTI file: {name}, columns {list(d.columns)}, dtypes "
              f"{[str(x) for x in d.dtypes]}, {len(d)} rows, sha256 matches")
        if list(d.columns) != ["ssyk4", "rti"]:
            raise RuntimeError(f"{name}: expected columns ['ssyk4', 'rti'], "
                               f"found {list(d.columns)}")
        d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
        d["rti"] = pd.to_numeric(d["rti"], errors="coerce")
        if d["rti"].isna().any() or not d["ssyk4"].is_unique \
                or len(d) < RTI_MIN_CODES:
            raise RuntimeError(f"{name}: {int(d['rti'].isna().sum())} missing "
                               f"scores, unique keys {d['ssyk4'].is_unique}, "
                               f"{len(d)} codes (at least {RTI_MIN_CODES} "
                               f"required)")
        return d.rename(columns={"rti": "score"})[["ssyk4", "score"]]
    raise RuntimeError(f"none of {RTI_NAMES} found in {mc.SHARE}")


def build_scores(s82, l47, l70, j47) -> tuple:
    """Both firm scores from one builder on one incumbent frame. The RTI
    code set must be a subset of DAIOE's, so no employer is scored on
    occupations the AI score cannot see."""
    daioe = l70.daioe_scores()
    rti = rti_book()
    extra = set(rti["ssyk4"]) - set(daioe["ssyk4"])
    if extra:
        raise RuntimeError(f"{len(extra)} RTI codes are not DAIOE codes")
    NOTES.append(f"RTI scores {len(rti)} of the {len(daioe)} DAIOE "
                 f"occupations")
    ai = s82.build_exposure(l47, l70, j47, daioe=daioe)
    drain(s82, "82/ai")
    rt = s82.build_exposure(l47, l70, j47, daioe=rti, audit=False)
    drain(s82, "82/rti")
    # Keep the firm frames and nothing else: each build also returns the
    # cascade and the incumbent frame, millions of rows each, which the
    # fits do not need and which would sit in memory beside R.
    keep = ("exposure", "arm", "floor", "basis")
    ai = {k: ai[k] for k in keep}
    rt = {k: rt[k] for k in keep}
    gc.collect()
    for nm, x in (("AI", ai), ("RTI", rt)):
        e = x["exposure"]
        print(f"  {nm} score: {len(e):,} employers, {x['arm']} arm, floor "
              f"{x['floor']} {x['basis']}")
    return ai, rt


# ----------------------------------------------------------------------
# Part A: the separation
# ----------------------------------------------------------------------

def wcorr(x, y, w) -> float:
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    c = np.average((x - mx) * (y - my), weights=w)
    d = np.sqrt(np.average((x - mx) ** 2, weights=w)
                * np.average((y - my) ** 2, weights=w))
    return float(c / d) if d > 0 else float("nan")


def part_a(ai: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    m = ai[["employer_id", "mix", "fq", "n"]].merge(
        rt[["employer_id", "mix", "fq"]], on="employer_id",
        suffixes=("_ai", "_rti"))
    if m.empty:
        raise RuntimeError("the two scores share no employer")
    n = len(m)
    ra, rr = m["mix_ai"].rank(), m["mix_rti"].rank()
    stats = {
        "pearson": float(m["mix_ai"].corr(m["mix_rti"])),
        "spearman": float(ra.corr(rr)),
        "pearson_weighted": wcorr(m["mix_ai"], m["mix_rti"], m["n"]),
        "spearman_weighted": wcorr(ra, rr, m["n"]),
    }
    for k, v in stats.items():
        add("overlap", "A", "all", "both", k, v, np.nan, -1, n)
    med_a, med_r = m["mix_ai"].median(), m["mix_rti"].median()
    hi_a = (m["mix_ai"] > med_a).astype(int)
    hi_r = (m["mix_rti"] > med_r).astype(int)
    for a_ in (0, 1):
        for r_ in (0, 1):
            k = int(((hi_a == a_) & (hi_r == r_)).sum())
            add("overlap", "A", "all", "both", f"median_cell_ai{a_}_rti{r_}",
                k / n, np.nan, -1, k)
    off = int((hi_a != hi_r).sum())
    add("overlap", "A", "all", "both", "share_off_diagonal_median", off / n,
        np.nan, -1, off)
    same = int((m["fq_ai"] == m["fq_rti"]).sum())
    add("overlap", "A", "all", "both", "share_same_quartile", same / n,
        np.nan, -1, same)
    top = int(((m["fq_ai"] == 4) & (m["fq_rti"] == 4)).sum())
    add("overlap", "A", "all", "both", "top_quartile_both", top / n,
        np.nan, -1, top)
    for nm, e in (("ai", ai), ("rti", rt)):
        add("overlap", "A", "all", nm, "employers_scored", float(len(e)),
            np.nan, -1, len(e))
    add("overlap", "A", "all", "both", "employers_scored_both", float(n),
        np.nan, -1, n)
    save()
    msg = (f"employer-level correlation of the two firm means: Pearson "
           f"{stats['pearson']:+.3f} (weighted {stats['pearson_weighted']:+.3f}), "
           f"Spearman {stats['spearman']:+.3f} (weighted "
           f"{stats['spearman_weighted']:+.3f}); {off / n:.1%} of employers "
           f"off the diagonal of the median cut; {same / n:.1%} in the same "
           f"quartile; {n:,} employers carry both scores")
    print(f"  {msg}")
    NOTES.append(msg)
    return m


# ----------------------------------------------------------------------
# Part B: the horse race
# ----------------------------------------------------------------------

def check_gate(band: str) -> None:
    """(a) must reproduce Table 1. Stops the script; does not warn. The
    arithmetic that refused is printed and written to the summary."""
    want = GATE[band]
    bad = []
    for key, term in (("post", "post"), ("step", "step_from_2023")):
        c, s = get("a", band, "ai", term, block="gate")
        wc, ws = want[key]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"{band} {key}: this run {c:+.4f} ({s:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f}), |diff| {abs(c - wc):.4f} and "
                       f"{abs(s - ws):.4f} against a tolerance of {GATE_TOL}")
    if bad:
        msg = "THE GATE FAILED. Nothing from this run is quotable. " + \
              "; ".join(bad)
        print(f"\n  {msg}")
        FAILURES.append(msg)
        write_summary(T0)
        raise SystemExit("94: the gate failed; stopping before any other fit.")
    c, s = get("a", band, "ai", "step_from_2023", block="gate")
    print(f"  the gate passes at {band}: step from 2023 {c:+.4f} ({s:.4f}), "
          f"Table 1 {want['step'][0]:+.4f} ({want['step'][1]:.4f})")


def standardise(b: pd.DataFrame, ai, rt) -> pd.DataFrame:
    """z_ai and z_rti: the continuous firm means, mean zero and SD one
    across the employers of the fit, weighted by incumbent employment (the
    weight the quartile cuts use)."""
    e = ai[["employer_id", "mix", "n"]].rename(columns={"mix": "mix_ai"}) \
        .merge(rt[["employer_id", "mix"]].rename(columns={"mix": "mix_rti"}),
               on="employer_id")
    e = e[e["employer_id"].isin(set(b["employer_id"]))]
    for col, src in (("z_ai", "mix_ai"), ("z_rti", "mix_rti")):
        mu = np.average(e[src], weights=e["n"])
        sd = np.sqrt(np.average((e[src] - mu) ** 2, weights=e["n"]))
        e[col] = (e[src] - mu) / (sd if sd > 0 else 1.0)
    return b.merge(e[["employer_id", "z_ai", "z_rti"]], on="employer_id",
                   how="inner")


def part_b(counts, ai, rt, s61, s78, j47) -> None:
    global PLANNED
    ai_e, rt_e = ai["exposure"], rt["exposure"]
    both = set(ai_e["employer_id"]) & set(rt_e["employer_id"])
    hi_rti = rt_e[["employer_id", "fq"]].rename(columns={"fq": "fq_rti"})
    for band in BANDS:
        print(f"\n  PART B at {band}:")
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"B/{band}/empty skeleton")
            continue
        b = s78.with_exposure(skel, ai_e)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"B/{band}/no exposure")
            continue
        n_all = int(b["employer_id"].nunique())
        tag = band.replace("-", "_")
        # (a) THE GATE, on the paper's own sample and nothing else changed
        b, t_ai = s78.eq2_terms(b)
        g, v = fit(b, f"a_{tag}", t_ai, j47.FES)
        record_fit(g, v, "a", band, [("ai", "")], n_all, block="gate")
        check_gate(band)
        b = b.drop(columns=t_ai)
        # the common sample
        b = b[b["employer_id"].isin(both)]
        b = b.merge(hi_rti, on="employer_id", how="inner")
        b["high_rti"] = (b["fq_rti"] == 4).astype(int)
        n_c = int(b["employer_id"].nunique())
        lost = n_all - n_c
        NOTES.append(f"{band}: the common sample keeps {n_c:,} of the "
                     f"{n_all:,} employers of (a)")
        if lost:
            PLANNED += 1
            # (a2) the DAIOE step on the common sample, so that (a2), (b)
            # and (c) differ in the regressors and nothing else
            b, t_ai = s78.eq2_terms(b)
            g, v = fit(b, f"a2_{tag}", t_ai, j47.FES)
            record_fit(g, v, "a2", band, [("ai", "")], n_c)
            b = b.drop(columns=t_ai)
        else:
            for r in [dict(r) for r in ROWS if r["block"] == "gate"
                      and r["young_band"] == band]:
                r.update(block="fit", spec="a2", status=r["status"])
                ROWS.append(r)
            save()
            print(f"    (a2) is (a): RTI scores every employer (a) uses")
        # (b) RTI only
        b, t_r = s78.eq2_terms(b, "high_rti", "rti")
        g, v = fit(b, f"b_{tag}", t_r, j47.FES)
        record_fit(g, v, "b", band, [("rti", "rti")], n_c)
        b = b.drop(columns=t_r)
        # (c) both
        b, t_ai = s78.eq2_terms(b)
        b, t_r = s78.eq2_terms(b, "high_rti", "rti")
        g, v = fit(b, f"c_{tag}", t_ai + t_r, j47.FES)
        record_fit(g, v, "c", band, [("ai", ""), ("rti", "rti")], n_c)
        b = b.drop(columns=t_ai + t_r)
        # (d) both, continuous and standardised
        b = standardise(b, ai_e, rt_e)
        b, t_za = s78.eq2_terms(b, "z_ai", "zai")
        b, t_zr = s78.eq2_terms(b, "z_rti", "zrti")
        g, v = fit(b, f"d_{tag}", t_za + t_zr, j47.FES)
        record_fit(g, v, "d", band, [("z_ai", "zai"), ("z_rti", "zrti")],
                   int(b["employer_id"].nunique()))
        del b
        gc.collect()


# ----------------------------------------------------------------------
# Part C: the sex split
# ----------------------------------------------------------------------

def gender_terms(b: pd.DataFrame, col: str, suffix: str) -> tuple:
    """78's gender_eq2_terms for any indicator column, line for line: every
    Equation (2) period x High x Young, x High x Female and x High x Young
    x Female. 78's own function hard-codes `high`; this one takes the
    column, so two indicators can sit in one model."""
    ym = b["year_month"].astype(str)
    q = ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1
    post_any = ym >= mc.CHATGPT_YM
    periods = {"rb": (ym >= mc.RIKSBANK_YM).astype(int),
               "interim": (post_any & (ym < "2024-01")).astype(int),
               "post": (ym >= "2024-01").astype(int)}
    for qq in (1, 2, 3):
        periods[f"q{qq}"] = (q == qq).astype(int)
    hy = b[col] * b["young"]
    hf = b[col] * b["female"]
    hyf = hy * b["female"]
    s = suffix
    terms = []
    for p, ind in periods.items():
        b[f"{p}_x_high{s}_x_young"] = ind * hy
        b[f"{p}_x_high{s}_x_female"] = ind * hf
        b[f"{p}_x_high{s}_x_young_x_female"] = ind * hyf
        terms += [f"{p}_x_high{s}_x_young", f"{p}_x_high{s}_x_female",
                  f"{p}_x_high{s}_x_young_x_female"]
    return b, terms


def record_sex(g, v, spec, inds, n_firms, block="fit"):
    """The male step and the female differential, at adoption and from the
    2023 level, for each indicator."""
    if g is None:
        return
    n_obs = int(g["n_obs"].max()) if "n_obs" in g.columns else -1
    for ind, s in inds:
        for lab, tail in (("male", ""), ("female_diff", "_x_female")):
            p = POST.format(s=s) + tail
            if p in g.index:
                add(block, spec, SEX_BAND, ind, f"{lab}_post",
                    g.loc[p, "coef"], g.loc[p, "se"], n_obs, n_firms)
            c, se = step(g, v, s, tail)
            add(block, spec, SEX_BAND, ind, f"{lab}_step_from_2023", c, se,
                n_obs, n_firms, "derived")
    save()


def part_c(sexcounts, ai, rt, s67, s78, j47) -> None:
    print(f"\n  PART C, the sex split at {SEX_BAND}:")
    skel = s67.build_skeleton_sex(sexcounts, SEX_BAND, j47, "n_emp")
    if skel.empty:
        FAILURES.append("C/empty skeleton")
        return
    b = s78.with_exposure(skel, ai["exposure"])
    del skel
    gc.collect()
    n_all = int(b["employer_id"].nunique())
    b, t_ai = s78.gender_eq2_terms(b)
    g, v = fit(b, "sex_a_22_25", t_ai, j47.FES)
    record_sex(g, v, "sex_a", [("ai", "")], n_all, block="gate")
    if g is None:
        return
    bad = []
    for key, term in (("post", "female_diff_post"),
                      ("step", "female_diff_step_from_2023")):
        c, s = get("sex_a", SEX_BAND, "ai", term, block="gate")
        wc, ws = SEX_GATE[key]
        if not (abs(c - wc) <= GATE_TOL and abs(s - ws) <= GATE_TOL):
            bad.append(f"{key}: this run {c:+.4f} ({s:.4f}), Table 1 "
                       f"{wc:+.4f} ({ws:.4f})")
    if bad:
        FAILURES.append("C/the sex gate failed, so the joint sex fit was not "
                        "run: " + "; ".join(bad))
        print("  the sex gate FAILED; the joint sex fit is not run")
        return
    print("  the sex gate passes")
    b = b.drop(columns=t_ai)
    rt_e = rt["exposure"][["employer_id", "fq"]].rename(
        columns={"fq": "fq_rti"})
    b = b.merge(rt_e, on="employer_id", how="inner")
    b["high_rti"] = (b["fq_rti"] == 4).astype(int)
    n_c = int(b["employer_id"].nunique())
    b, t_ai = s78.gender_eq2_terms(b)
    b, t_r = gender_terms(b, "high_rti", "rti")
    g, v = fit(b, "sex_c_22_25", t_ai + t_r, j47.FES)
    record_sex(g, v, "sex_c", [("ai", ""), ("rti", "rti")], n_c)
    del b
    gc.collect()


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def verdict() -> tuple:
    ca, sa = get("c", "22-25", "ai", "step_from_2023")
    cr, sr = get("c", "22-25", "rti", "step_from_2023")
    if ca != ca or cr != cr:
        return "NO VERDICT", ["  NO VERDICT: the joint fit at 22-25 did not "
                              "come back, and a missing fit is not a null."]
    neg = lambda c, s: bool(s == s and s > 0 and c < 0 and abs(c) >= SIG5 * s)  # noqa: E731
    a_, r_ = neg(ca, sa), neg(cr, sr)
    v = ("1 AI-SPECIFIC" if a_ and not r_ else
         "2 RTI ABSORBS" if r_ and not a_ else
         "3 BOTH" if a_ and r_ else "4 NEITHER")
    L = [f"  THE VERDICT, read rule {v}",
         f"    (c) at 22-25, step from the 2023 level: DAIOE {ca:+.4f} "
         f"({sa:.4f}) t {tstat(ca, sa):+.2f}; RTI {cr:+.4f} ({sr:.4f}) "
         f"t {tstat(cr, sr):+.2f}"]
    c2, s2 = get("a2", "22-25", "ai", "step_from_2023")
    if c2 == c2 and c2 != 0:
        L.append(f"    (c) retains {ca / c2:.0%} of (a2)'s DAIOE step "
                 f"({c2:+.4f} ({s2:.4f}))")
    return v, L


def write_summary(t0: float) -> None:
    L = ["GENERATIVE-AI EXPOSURE AGAINST ROUTINE-TASK INTENSITY", "=" * 54,
         "", "Both firm scores from 82's build_exposure() on the same",
         "employers, freeze year, incumbents and floor; only the book",
         "differs: the DAIOE genAI percentile and the Autor-Dorn (2013) RTI.",
         ""]
    L += ["A. THE SEPARATION:"] + [f"  {n}" for n in NOTES
                                    if n.startswith("employer-level")] + [""]
    if any(r["block"] in ("gate", "fit") for r in ROWS):
        L += ["B. THE HORSE RACE (step from the 2023 level; adoption step in",
              "   brackets after it):"]
        for band in BANDS:
            for spec, ind in (("a", "ai"), ("a2", "ai"), ("b", "rti"),
                              ("c", "ai"), ("c", "rti"), ("d", "z_ai"),
                              ("d", "z_rti")):
                blk = "gate" if spec == "a" else "fit"
                c, s = get(spec, band, ind, "step_from_2023", block=blk)
                pc, ps = get(spec, band, ind, "post", block=blk)
                if c != c:
                    continue
                n = next((r["n_firms"] for r in ROWS if r["spec"] == spec
                          and r["young_band"] == band), -1)
                L.append(f"  {band} ({spec:<2}) {ind:<6} {c:+.4f} ({s:.4f}) "
                         f"t {tstat(c, s):+6.2f}   [{pc:+.4f} ({ps:.4f})]   "
                         f"{int(n):,} employers")
        L.append("")
        v, lines = verdict()
        L += lines + [""]
    if any(r["spec"].startswith("sex") for r in ROWS):
        L += [f"C. THE SEX SPLIT AT {SEX_BAND}:"]
        for spec, ind in (("sex_a", "ai"), ("sex_c", "ai"), ("sex_c", "rti")):
            blk = "gate" if spec == "sex_a" else "fit"
            for term in ("male_step_from_2023", "female_diff_step_from_2023",
                         "female_diff_post"):
                c, s = get(spec, SEX_BAND, ind, term, block=blk)
                if c == c:
                    L.append(f"  ({spec}) {ind:<4} {term:<28} {c:+.4f} "
                             f"({s:.4f}) t {tstat(c, s):+.2f}")
        L.append("")
    L += [f"FITS: {DONE} of {PLANNED} planned came back. A run far shorter "
          "than the estimate with fits missing is a failure, not a result."]
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "WHAT FAILED: " + " | ".join(FAILURES),
              "A missing row is a missing fit, never a zero."]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - t0) / 60:.1f} "
                              "min. " + mc.mem_line("")]
    (OUT / "94_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global PLANNED, T0
    mc.Tee(OUT / "94_log.txt")
    t0 = T0 = time.time()
    print("=" * 70)
    print(f"94: GENAI EXPOSURE AGAINST ROUTINE-TASK INTENSITY   parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    try:
        s82, s61, s67, s78, l47, l70, j47 = load_modules()
        # Schema probes on every cache before anything is built from it.
        counts = s82.load_counts("L_counts", s61.PANEL_YEARS,
                                 require=COUNT_COLS) if "B" in PARTS else None
        if "B" in PARTS and counts is None:
            raise RuntimeError("L_counts_2021-2025 missing or lacking "
                               f"{COUNT_COLS}; this script does no SQL for "
                               "them. Run 47L first.")
        if counts is not None:
            last = str(counts["year_month"].max())
            print(f"  counts: {len(counts):,} employer-age-months to {last}")
            if last < "2024-01":
                raise RuntimeError(f"the counts end at {last}")
        # The sex counts are only checked for here and read after Part B,
        # so that the Python side does not hold them while R fits the
        # stock panel (failure class 4: the two share one job budget).
        sex_ok = "C" in PARTS and all(
            (CACHE / f"L_counts_sex_{y}.parquet").exists()
            for y in s61.PANEL_YEARS)
        if "C" in PARTS and not sex_ok:
            NOTES.append("L_counts_sex_* missing: Part C is skipped")
        if not s82.CASC_CACHE.exists():
            print("  WARNING: the cascade cache is not on the share; 82 will "
                  "pull it (SQL)")
        PLANNED = (4 * len(BANDS) if "B" in PARTS else 0) + \
                  (2 if sex_ok else 0)

        ai, rt = build_scores(s82, l47, l70, j47)
        # Part A needs no fit and is always run: it is the separation the
        # rest of the script is read against.
        part_a(ai["exposure"], rt["exposure"])
        if "B" in PARTS:
            part_b(counts, ai, rt, s61, s78, j47)
        del counts
        gc.collect()
        sexcounts = s82.load_counts("L_counts_sex", s61.PANEL_YEARS,
                                    require=SEX_COLS) if sex_ok else None
        if sex_ok and sexcounts is None:
            NOTES.append("L_counts_sex_* unreadable or lacking "
                         f"{SEX_COLS}: Part C is skipped")
        if sexcounts is not None:
            try:
                part_c(sexcounts, ai, rt, s67, s78, j47)
            except BaseException as ex:
                if isinstance(ex, SystemExit):
                    raise
                print(f"  Part C FAILED ({type(ex).__name__}: {ex})")
                traceback.print_exc()
                FAILURES.append(f"C/{type(ex).__name__}")
        drain(s78, "78")
    except SystemExit:
        mc.runlog("94_rti_horserace", 2, (time.time() - t0) / 60)
        raise
    except BaseException as ex:
        print(f"94 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    write_summary(t0)
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("94_rti_horserace", rc, (time.time() - t0) / 60)
    print("\n94 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

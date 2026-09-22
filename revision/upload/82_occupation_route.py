#!/usr/bin/env python3
"""
82_occupation_route.py -- the paper's findings re-estimated on a firm
                          score that uses no education record at all.

======================================================================
  RUNS IN MONA. Part A is counting only and needs no SQL if 47L's
  baseline cache is on the share. Part B and Part C fit Poisson models
  on the panels scripts 68, 74, 67 and 54 already build. Only Part C's
  vintage check pulls: one read of the November 2019 declarations
  joined to two Individ tables. Parts are chosen with the environment
  variable CANARIES_82_PARTS (default ABC) and the folder with
  CANARIES_82_OUT (default output_82); the lane runners set both.
======================================================================

QUESTION
The paper's firm exposure is routed through education. Each education
group is given the employment-weighted mean DAIOE generative-AI
percentile of the occupations its holders worked in during 2019, and an
employer is ranked by the mean of that score over its incumbents aged 31
to 69. Education therefore carries the exposure from the occupation to
the firm, and a reader may reasonably ask how much of the result is the
education register rather than the work. This script answers it by
deleting the intermediate step: an employer is scored directly by the
employment-weighted mean DAIOE percentile of the four-digit occupations
its own incumbents aged 31 to 69 held in November 2019. The freeze year,
the incumbent restriction, the person floor and the quartile logic are
the paper's; no education record enters at any point.

DESIGN
Exposure (script 65's occupation_exposure, imported rather than
rewritten): the worker-weighted mean DAIOE percentile of the 2019
four-digit occupations of the employer's incumbents aged 31 to 69, from
script 47L's baseline pull; an employer with fewer than five coded
incumbents is not scored; the quartile cut points are weighted by
incumbent employment, so the top quartile holds a quarter of incumbent
employment rather than a quarter of employers. That is script 47j's
incumbent_exposure line for line with the score taken from the
occupation register instead of the education register, which is what
makes the two routes comparable.

One unit differs and is stated rather than hidden. The education route
weights an employer's incumbents by person-months summed over 2019, so
its floor of five is five person-months; the occupation register has a
single November reference, so here the weight is the November head count
and the floor is five incumbents. The floor binds on the same object in
both routes (an employer too thin to classify) and on a different scale.

Part A (part_a). No fit. The coverage of the 2019 occupation code among
incumbent person-months by age band, both as a coded share and as a
scored share, since a code outside the DAIOE file cannot carry an
exposure; the employers each route can score, alone and on the two young
panels; the occupation quartile against the education quartile among the
employers both routes score, with the share on the diagonal and the
Spearman rank correlation of the underlying continuous scores; and the
size and longevity of the employers one route scores and the other does
not, read off the panel the fits themselves run on.

Part B (part_b). Equation (2) on the employment stock at 22-25 and at
26-30: the cumulative tightening switch from April 2022, the interim
window from the launch to December 2023, the adoption step from January
2024, and the three calendar-quarter terms with the fourth quarter
omitted, each interacted with High x Young, under employer-by-month,
employer-by-age and month-by-age effects, Poisson pseudo-maximum
likelihood, standard errors clustered by employer. The term set is
script 78's, which is script 68's. Then the six-band profile of script
74's seasonal arm: the same terms per band with 41-49 omitted, on
script 70's six-band skeleton, so every coefficient is a difference from
the prime-aged band.

Part C (part_c). The sex specification of Equation (2) at 22-25 on
script 67's panel, every term entered as High x Young, High x Female and
High x Young x Female, with employer-by-age-and-sex and
month-by-age-and-sex effects; hires and separations at 22-25 on script
54's flows with the same terms as the stock; and a vintage check, the
analogue of the education re-scoring the paper reports, in which the
same November 2019 incumbents are re-scored from the occupation code the
Individ register of 2021 holds for them. The birth year and therefore
the population come from the 2019 register in both arms, so a person
absent from the later register loses his code rather than leaving the
sample, and the two arms differ in the code and in nothing else.

READ RULES
Fixed before the run and printed at the start and in the summary. There
is no coefficient gate: a different measure gives different estimates.
Three questions are settled in advance and answered explicitly whichever
way they fall, with every point estimate beside the education-route one.

INPUTS AND OUTPUTS
Reads the caches L_baseline_2019, L_counts_2021 to 2025 (script 47L),
flows_2021 to 2025 (script 54), L_counts_sex_2021 to 2025 (script 67)
and edu_hr_weights_2019 to 2021 and edu_hr_2019 (script 47h, for the
education route Part A compares against), and the input file
daioe_quartiles.dta. Caches L_baseline_2019_asof2021.parquet. Writes to
output_82/: occ_route_coverage.csv, occ_route_headline.csv,
occ_route_profile.csv, occ_route_gender.csv, occ_route_flows.csv,
occ_route_vintage.csv, the vcov_s82_*.csv files and 82_summary.txt.

IN THE PAPER
Online Appendix III.2, the robustness of the exposure route: whether the
findings of Section 3 and Table 1 survive a firm score built without the
education register. Nothing here replaces a number in Table 1.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_82_OUT", "output_82")
PARTS = os.environ.get("CANARIES_82_PARTS", "ABC").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BASE_YEAR = 2019                 # the year the score is frozen at
VINTAGE = 2021                   # the later register of the vintage check
POST_FROM = "2024-01"            # adoption, as in 68 and 78
BANDS = ["22-25", "26-30"]       # the two young bands the paper reports
SEX_BAND = "22-25"               # the band the paper's sex result is on
FLOW_BAND = "22-25"              # and the band the margins are read at
PROFILE_REF = "41-49"            # the omitted band of the profile
FLOOR = 5                        # the export floor, as in mona_common
SIG5 = 1.959963984540054         # two-sided five per cent
SIG1 = 2.5758293035489004        # two-sided one per cent
BASE_CACHE = CACHE / f"L_baseline_{BASE_YEAR}.parquet"
VINT_CACHE = CACHE / f"L_baseline_{BASE_YEAR}_asof{VINTAGE}.parquet"
BASE_COLS = ["employer_id", "age_group", "ssyk4", "n"]

# What the education route prints, taken from the exports the paper
# quotes: script 68's seasonal_pooled.csv (the stock and flow steps),
# script 74's contrast_seasonal.csv (the seasonal arm of the profile) and
# script 78's gender_eq2.csv (the full sex specification). Every table
# this script writes carries the matching pair, so no comparison depends
# on a reader holding two files open.
EDU_STOCK = {"22-25": (-0.0408, 0.0150), "26-30": (-0.0394, 0.0102)}
EDU_FLOW = {"hires": (-0.0032, 0.0363), "seps": (+0.0787, 0.0203)}
EDU_PROFILE = {"22-25": (-0.0099, 0.0121), "26-30": (-0.0096, 0.0084),
               "31-34": (+0.0159, 0.0063), "35-40": (+0.0090, 0.0051),
               "41-49": (0.0, 0.0), "50+": (+0.0589, 0.0062)}
EDU_FEMALE = (-0.0746, 0.0142)

NOTES = []
FAILURES = []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  There is NO coefficient gate. This is a different measure of the",
    "  same object, so the estimates will differ from the education route",
    "  and that is expected. What is judged is whether the paper's",
    "  findings survive the change of route, on three questions settled",
    "  before any of this was estimated:",
    "    1. REPRODUCES if the adoption step at 22-25 is negative and",
    "       distinguishable from zero at the five per cent level on",
    "       employer clustering.",
    "    2. THE PROFILE REPRODUCES if the 50-and-over band gains against",
    "       41-49 and the young band is the lowest or the second lowest",
    "       of the six.",
    "    3. THE SEX RESULT REPRODUCES if the female differential is",
    "       negative and distinguishable from zero at the one per cent",
    "       level.",
    "  Each verdict is reported explicitly and whatever the numbers are,",
    "  and every point estimate is reported beside the education-route",
    "  one in every table.",
    "  The education route prints: the adoption step -0.0408 (0.0150) at",
    "  22-25 and -0.0394 (0.0102) at 26-30; 22-25 against 41-49 -0.0099",
    "  (0.0121) and 50 and over +0.0589 (0.0062); the female differential",
    "  -0.0746 (0.0142); separations +0.0787 (0.0203) and hires -0.0032",
    "  (0.0363).",
    f"  Employer counts below {FLOOR} are suppressed before anything",
    "  leaves MONA, and a share is suppressed with its own numerator.",
]


def opt(label, fn, *a, **kw):
    """Run one part. A part that dies is recorded and the others still
    run; nothing partial is silently treated as a result."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(label)
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_modules():
    """
    The scripts this one reuses rather than reimplements.

    61 builds the balanced employer by band by month skeleton, 67 the
    same skeleton with sex as a fourth dimension, 70 the six-band
    skeleton and the education exposure, 74 the per-band profile terms,
    78 the term sets of Equation (2) and of its sex split, 65 the
    occupation score, 80 the description of a panel's employers, 47L the
    baseline pull, 47j the fixed-effect list and the incumbent bands.
    Importing them rather than copying is the point: this script must be
    68, 74, 67 and 54 with one column changed, and a copied term list
    could drift away from the estimates the paper quotes.
    """
    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    s74 = _mod("74_contrast_seasonal.py", "s74")
    s78 = _mod("78_final_checks.py", "s78")
    s80 = _mod("80_industry_key.py", "s80")
    l47 = _mod("47L_age_baseline_exposure.py", "l47")
    l65 = _mod("65_occupation_arm.py", "l65")
    l70 = _mod("70_respecifications.py", "l70")
    j47 = s61._j47()
    # 78's module-level OUT and CACHE are its own. Point them here so that
    # anything reached through it lands with this script's exports.
    s78.OUT, s78.CACHE = OUT, CACHE
    # Four hard guards. Each of them is a place where this script's
    # docstring and read rules would describe a model it is not fitting.
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(
            f"78's adoption date is {s78.POST_FROM} and this script says "
            f"{POST_FROM}; the terms come from 78, so settle it there first")
    if l65.MIN_FIRM_INCUMBENTS != j47.MIN_FIRM_INCUMBENTS:
        raise RuntimeError(
            f"65's incumbent floor is {l65.MIN_FIRM_INCUMBENTS} and 47j's is "
            f"{j47.MIN_FIRM_INCUMBENTS}; the two routes would then differ in "
            f"the floor as well as in the register, and nothing here would "
            f"be a comparison")
    if list(s74.BANDS) != list(l70.CONTRAST_BANDS):
        raise RuntimeError(
            f"74 has {s74.BANDS} and 70 builds the skeleton from "
            f"{l70.CONTRAST_BANDS}; the panel and the profile terms would "
            f"not agree")
    if l70.REF_BAND != PROFILE_REF:
        raise RuntimeError(
            f"70's reference band is {l70.REF_BAND} and this script says "
            f"{PROFILE_REF}; read rule 2 names the band explicitly")
    return s61, s67, s74, s78, s80, l47, l65, l70, j47


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """
    One Poisson fit. Returns (coefficient table indexed by term, the
    clustered covariance as a DataFrame or None). A failure is recorded
    and returns (None, None), because a missing row must never be read as
    a zero.
    """
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s82_{tag}", terms=terms,
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
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return g, v


def cnt(v) -> str:
    """A count as the summary may print it: the export floor applies to
    the text that leaves MONA as much as to the CSV beside it."""
    if v is None or v != v:
        return "(suppressed)"
    v = int(v)
    return "(suppressed)" if 0 < v < FLOOR else f"{v:,}"


def save(rows: list, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    """
    Write one export, with the floor applied on the way out.

    Every table this script writes passes through here, so no export can
    reach the share without the suppression having run on it, and the
    rule is enforced in one place rather than remembered at six call
    sites.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def tstat(coef, se) -> float:
    return float(coef) / se if se and se == se and se > 0 else np.nan


# ----------------------------------------------------------------------
# The score: the occupation route, and the education route beside it
# ----------------------------------------------------------------------

def daioe_scores(l70) -> pd.DataFrame:
    """ssyk4 and the generative-AI percentile, through 70's loader."""
    return l70.daioe_scores()


def baseline(l47) -> pd.DataFrame:
    """
    47L's 2019 baseline: employer by age band by four-digit occupation by
    head count in November 2019, with '____' for a worker the register
    leaves uncoded. Read from the cache when it is there; pulled through
    47L's own query when it is not, so there is one pull of this frame in
    the project and not two.
    """
    b = mc.read_cache(BASE_CACHE, require=BASE_COLS)
    if b is not None:
        print(f"  baseline {BASE_YEAR}: cached ({len(b):,} rows)")
        return b
    t = time.time()
    conn = mc.connect()
    try:
        b = l47.q_baseline(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    mc.write_cache(b, BASE_CACHE)
    print(f"  baseline {BASE_YEAR}: {len(b):,} rows ({time.time()-t:.0f}s)")
    return b


def baseline_vintage_sql(vintage: int) -> str:
    """
    47L's baseline query with the register that supplies the occupation
    code separated from the register that supplies the birth year.

    The population must not move. 47L reads the band and the code from
    one Individ table, so simply joining a later one would drop every
    worker the later register does not hold, and the vintage check would
    then measure attrition as well as re-coding. Here the birth year, and
    therefore the age band and the sample filter, come from the 2019
    register in both arms, and the code comes from the vintage register;
    a worker the later register does not hold keeps his place and loses
    his code to the '____' convention, which is what the coverage column
    then counts.

    At vintage = BASE_YEAR the two joins are the same table and this is
    47L's q_baseline.
    """
    age_case = "\n".join(
        f"             WHEN {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) "
        f"BETWEEN {lo} AND {hi} THEN '{lab}'"
        for lab, (lo, hi) in mc.AGE_GROUPS.items())
    age_case = f"CASE\n{age_case}\n             ELSE NULL END"
    code = ("""CASE WHEN v.Ssyk4_2012_J16 IS NULL
                      OR LTRIM(v.Ssyk4_2012_J16) = ''
                      OR LEFT(LTRIM(v.Ssyk4_2012_J16), 1) = '*'
                 THEN '____'
                 ELSE RIGHT('0000' + CAST(v.Ssyk4_2012_J16 AS VARCHAR(4)), 4)
                 END""")
    return f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           {age_case} AS age_group,
           {code} AS ssyk4,
           LTRIM(RTRIM(v.SsykStatus_J16)) AS ssyk_status,
           COUNT(DISTINCT agi.P1207_LOPNR_PERSONNR) AS n
    FROM dbo.Arb_AGIIndivid{BASE_YEAR}11_def agi
    LEFT JOIN dbo.Individ_{BASE_YEAR} b
      ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
    LEFT JOIN dbo.Individ_{vintage} v
      ON agi.P1207_LOPNR_PERSONNR = v.P1207_LopNr_PersonNr
    WHERE {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY agi.P1207_LOPNR_PEORGNR, {age_case}, {code},
             LTRIM(RTRIM(v.SsykStatus_J16))
    """


def baseline_vintage(vintage: int) -> pd.DataFrame:
    """The same incumbents, coded as the register of `vintage` has them."""
    b = mc.read_cache(VINT_CACHE, require=BASE_COLS)
    if b is not None:
        print(f"  baseline {BASE_YEAR} as of {vintage}: cached "
              f"({len(b):,} rows)")
        return b
    t = time.time()
    conn = mc.connect()
    try:
        b = pd.read_sql(baseline_vintage_sql(vintage), conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    mc.write_cache(b, VINT_CACHE)
    print(f"  baseline {BASE_YEAR} as of {vintage}: {len(b):,} rows "
          f"({time.time()-t:.0f}s)")
    return b


def occ_exposure(base: pd.DataFrame, daioe: pd.DataFrame, l65,
                 j47) -> pd.DataFrame:
    """
    The occupation-route firm score and quartile, through 65's own
    builder, so the two scripts cannot hold two versions of it.
    """
    return l65.occupation_exposure(base, daioe, j47.INCUMBENT_BANDS)


# ----------------------------------------------------------------------
# Part A: the score and what it covers. No fit.
# ----------------------------------------------------------------------

def coverage_by_band(base: pd.DataFrame, daioe: pd.DataFrame, j47) -> list:
    """
    The share of incumbent head count carrying a usable 2019 occupation
    code, by age band.

    Two shares, not one. `coded` is the share the register gives a code
    at all, which is the '____' convention counted from the other side.
    `scored` is the share whose code is also in the DAIOE file, which is
    the share that can actually carry an exposure; a code outside the
    file is as useless to this route as no code, and reporting only the
    first would overstate what the route can see.
    """
    b = base.copy()
    b["ssyk4"] = b["ssyk4"].astype(str).str.zfill(4)
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    b = b[b["age_group"].notna() & (b["n"] > 0)]
    scored_codes = set(daioe["ssyk4"].astype(str))
    b["is_coded"] = (b["ssyk4"] != "____").astype(int)
    b["is_scored"] = b["is_coded"] * b["ssyk4"].isin(scored_codes).astype(int)
    rows = []
    # The incumbent total is taken from 47j's band list, not from a list
    # written here, so the coverage figure describes the same workers the
    # score is built from.
    for band, d in list(b.groupby("age_group", observed=True)) + \
            [("31-69 incumbents",
              b[b["age_group"].isin(j47.INCUMBENT_BANDS)])]:
        tot = int(d["n"].sum())
        cod = int((d["n"] * d["is_coded"]).sum())
        sco = int((d["n"] * d["is_scored"]).sum())
        firms = int(d["employer_id"].nunique())
        rows.append({"panel": "baseline 2019", "block": "coverage",
                     "group": str(band), "item": "coded_share",
                     "n_employers": firms, "n_obs": tot,
                     "share": cod / max(tot, 1), "value": float(cod)})
        rows.append({"panel": "baseline 2019", "block": "coverage",
                     "group": str(band), "item": "scored_share",
                     "n_employers": firms, "n_obs": tot,
                     "share": sco / max(tot, 1), "value": float(sco)})
    return rows


def route_rows(occ: pd.DataFrame, edu: pd.DataFrame) -> tuple:
    """How many employers each route can score, and the overlap."""
    o = set(occ["employer_id"])
    e = set(edu["employer_id"])
    both, oo, eo = o & e, o - e, e - o
    rows = []
    for name, s in (("occupation_route", o), ("education_route", e),
                    ("both_routes", both), ("occupation_only", oo),
                    ("education_only", eo)):
        rows.append({"panel": "all scored employers", "block": "route",
                     "group": name, "item": "n_employers",
                     "n_employers": len(s), "n_obs": len(s),
                     "share": len(s) / max(len(o | e), 1),
                     "value": float(len(s))})
    return rows, both


def crosstab_rows(occ: pd.DataFrame, edu: pd.DataFrame,
                  both: set) -> tuple:
    """
    The occupation quartile against the education quartile, among the
    employers both routes score.

    The diagonal share says how often the two routes put a firm in the
    same quartile, and the rank correlation says whether they agree about
    the ordering when they disagree about the cut. Both are reported: a
    high diagonal with a low correlation would mean the agreement is an
    artefact of where the cuts fall, and the converse would mean the two
    orderings agree and the quartile boundaries do not.
    """
    j = (occ[occ["employer_id"].isin(both)][["employer_id", "fq", "mix"]]
         .rename(columns={"fq": "fq_occ", "mix": "mix_occ"})
         .merge(edu[edu["employer_id"].isin(both)][["employer_id", "fq",
                                                    "mix"]]
                .rename(columns={"fq": "fq_edu", "mix": "mix_edu"}),
                on="employer_id", how="inner"))
    rows = []
    if j.empty:
        return rows, {}
    n = len(j)
    for (qo, qe), d in j.groupby(["fq_occ", "fq_edu"], observed=True):
        rows.append({"panel": "both routes", "block": "crosstab",
                     "group": f"occ_Q{int(qo)}_edu_Q{int(qe)}",
                     "item": "n_employers", "n_employers": int(len(d)),
                     "n_obs": int(len(d)), "share": len(d) / n,
                     "value": float(len(d))})
    diag = float((j["fq_occ"] == j["fq_edu"]).mean())
    top = float(((j["fq_occ"] == 4) & (j["fq_edu"] == 4)).sum())
    n_top_occ = int((j["fq_occ"] == 4).sum())
    n_top_edu = int((j["fq_edu"] == 4).sum())
    rho = float(j["mix_occ"].corr(j["mix_edu"], method="spearman"))
    pear = float(j["mix_occ"].corr(j["mix_edu"], method="pearson"))
    for item, val, nfirm in (("share_on_diagonal", diag, n),
                             ("spearman_rank_correlation", rho, n),
                             ("pearson_correlation", pear, n),
                             ("n_top_quartile_both", top, int(top)),
                             ("n_top_quartile_occupation", n_top_occ,
                              n_top_occ),
                             ("n_top_quartile_education", n_top_edu,
                              n_top_edu)):
        rows.append({"panel": "both routes", "block": "crosstab",
                     "group": "summary", "item": item,
                     "n_employers": nfirm, "n_obs": n,
                     "share": np.nan, "value": val})
    return rows, {"n": n, "diag": diag, "rho": rho,
                  "top_both": int(top), "top_occ": n_top_occ,
                  "top_edu": n_top_edu}


def panel_rows(band: str, counts, occ, edu, s61, s80, j47) -> tuple:
    """
    The two routes on the panel the fits themselves run on, and the size
    and longevity of the employers one route places and the other does
    not.

    The panel is rebuilt rather than approximated by an employer list,
    because the question is how many of THOSE employers each route can
    score, and any other population answers a different question.
    """
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        return [], {}
    emp = s80.panel_employers(skel)
    del skel
    gc.collect()
    ids = emp["employer_id"]
    in_o = ids.isin(set(occ["employer_id"]))
    in_e = ids.isin(set(edu["employer_id"]))
    n = int(len(emp))
    groups = {"scored_by_both": emp[(in_o & in_e).to_numpy()],
              "occupation_only": emp[(in_o & ~in_e).to_numpy()],
              "education_only": emp[(~in_o & in_e).to_numpy()],
              "scored_by_neither": emp[(~in_o & ~in_e).to_numpy()]}
    panel = f"{band} stock"
    rows = [{"panel": panel, "block": "panel", "group": "all",
             "item": "employers_on_panel", "n_employers": n, "n_obs": n,
             "share": 1.0, "value": float(n)},
            {"panel": panel, "block": "panel", "group": "occupation_route",
             "item": "n_employers", "n_employers": int(in_o.sum()),
             "n_obs": n, "share": float(in_o.mean()),
             "value": float(in_o.sum())},
            {"panel": panel, "block": "panel", "group": "education_route",
             "item": "n_employers", "n_employers": int(in_e.sum()),
             "n_obs": n, "share": float(in_e.mean()),
             "value": float(in_e.sum())}]
    for name, d in groups.items():
        rows.append({"panel": panel, "block": "panel", "group": name,
                     "item": "n_employers", "n_employers": int(len(d)),
                     "n_obs": n, "share": len(d) / max(n, 1),
                     "value": float(len(d))})
        for r in s80.size_rows(panel, name, d):
            r["n_obs"] = int(len(d))
            rows.append(r)
    summ = {"panel": panel, "n": n,
            "n_occ": int(in_o.sum()), "n_edu": int(in_e.sum()),
            "n_both": int(len(groups["scored_by_both"])),
            "n_occ_only": int(len(groups["occupation_only"])),
            "n_edu_only": int(len(groups["education_only"]))}
    for name, key in (("occupation_only", "occ_only"),
                      ("education_only", "edu_only"),
                      ("scored_by_both", "both")):
        d = groups[name]
        summ[f"med_{key}"] = (float(d["mean_headcount"].median())
                              if len(d) >= FLOOR else np.nan)
        summ[f"months_{key}"] = (float(d["months_active"].mean())
                                 if len(d) >= FLOOR else np.nan)
    del emp
    gc.collect()
    return rows, summ


def part_a(counts, occ, edu, base, daioe, s61, s80, j47) -> tuple:
    """What the occupation route scores, and how it lines up with the
    education route. No fit runs here."""
    rows = coverage_by_band(base, daioe, j47)
    r, both = route_rows(occ, edu)
    rows += r
    r, cross = crosstab_rows(occ, edu, both)
    rows += r
    summaries = []
    for band in BANDS:
        r, s = panel_rows(band, counts, occ, edu, s61, s80, j47)
        if not r:
            FAILURES.append(f"A/{band}/empty")
            continue
        rows += r
        summaries.append(s)
        print(f"  A: {band} stock, {cnt(s['n'])} employers, "
              f"{cnt(s['n_occ'])} scorable on the occupation route, "
              f"{cnt(s['n_edu'])} on the education route, "
              f"{cnt(s['n_both'])} on both")
    tab = mc.enforce_min_cell(pd.DataFrame(rows), count_col="n_obs",
                              floor=FLOOR)
    tab = mc.enforce_min_cell(tab, count_col="n_employers", floor=FLOOR)
    # A share and a published denominator reproduce a suppressed count, so
    # the share and the value go with their own numerator.
    gone = tab["n_employers"].isna()
    tab.loc[gone, ["share", "value"]] = np.nan
    tab = save(tab, "occ_route_coverage.csv", count_col="n_employers")
    return tab, summaries, cross


# ----------------------------------------------------------------------
# Part B: the headline and the profile
# ----------------------------------------------------------------------

def edu_pair(d: dict, key: str) -> tuple:
    c, s = d.get(key, (np.nan, np.nan))
    return c, s


def part_b(counts, occ, s61, s74, s78, l70, j47) -> tuple:
    """
    Equation (2) on the stock at both young bands, and the six-band
    profile against 41-49, on the occupation-route quartile.

    The terms are 78's, which are 68's; the profile terms are 74's
    seasonal arm on 70's six-band skeleton. Only the column that says
    which employers are highly exposed differs from the paper's route.
    """
    head, prof = [], []
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"B/{band}/empty")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"B/{band}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, _ = fit(b, f"stock_{band.replace('-', '_')}", terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        ec, es = edu_pair(EDU_STOCK, band)
        for r in s78.rows_of(g, terms, young_band=band, outcome="stock",
                             n_firms=n_firms):
            r["t"] = tstat(r["coef"], r["se"])
            post = r["term"] == "post_x_high_x_young"
            r["edu_coef"] = ec if post else np.nan
            r["edu_se"] = es if post else np.nan
            head.append(r)
        save(head, "occ_route_headline.csv")
        p = [h for h in head if h["young_band"] == band
             and h["term"] == "post_x_high_x_young"]
        if p:
            print(f"  B: {band} adoption step {p[0]['coef']:+.4f} "
                  f"({p[0]['se']:.4f}) t {p[0]['t']:+.2f}   education route "
                  f"{ec:+.4f} ({es:.4f})")
    # ---- the six-band profile ----------------------------------------
    skel = l70.all_band_skeleton(counts)
    if skel.empty:
        FAILURES.append("B/profile/empty")
        return head, prof
    b = s78.with_exposure(skel, occ)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("B/profile/no exposure")
        return head, prof
    n_firms = int(b["employer_id"].nunique())
    b, terms = s74.build_terms(b, l70, seasonal=True)
    g, _ = fit(b, "profile_six_band", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return head, prof
    for band in s74.BANDS:
        ec, es = edu_pair(EDU_PROFILE, band)
        if band == PROFILE_REF:
            prof.append({"band": band, "coef": 0.0, "se": 0.0, "t": np.nan,
                         "n_firms": n_firms, "n_obs": int(g["n_obs"].max()),
                         "status": "reference", "edu_coef": ec,
                         "edu_se": es})
            continue
        t_ = l70.band_col("gpt_x_high", band)
        if t_ not in g.index:
            continue
        prof.append({"band": band, "coef": float(g.loc[t_, "coef"]),
                     "se": float(g.loc[t_, "se"]),
                     "t": tstat(float(g.loc[t_, "coef"]),
                                float(g.loc[t_, "se"])),
                     "n_firms": n_firms,
                     "n_obs": int(g.loc[t_, "n_obs"]),
                     "status": str(g.loc[t_].get("status", "ok")),
                     "edu_coef": ec, "edu_se": es})
    if prof:
        save(prof, "occ_route_profile.csv")
        for r in prof:
            print(f"  B: profile {r['band']:<6} {r['coef']:+.4f} "
                  f"({r['se']:.4f})   education route "
                  f"{r['edu_coef']:+.4f} ({r['edu_se']:.4f})")
    return head, prof


# ----------------------------------------------------------------------
# Part C: the sex split, the margins and the vintage check
# ----------------------------------------------------------------------

def part_c_gender(sexcounts, occ, s67, s78, j47) -> tuple:
    """The sex specification of Equation (2) at 22-25."""
    if sexcounts is None:
        FAILURES.append("C/gender/no L_counts_sex cache")
        print("  C: L_counts_sex_* missing, the sex specification is skipped")
        return [], {}
    skel = s67.build_skeleton_sex(sexcounts, SEX_BAND, j47, "n_emp")
    if skel.empty:
        FAILURES.append("C/gender/empty")
        return [], {}
    b = s78.with_exposure(skel, occ)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("C/gender/no exposure")
        return [], {}
    n_firms = int(b["employer_id"].nunique())
    b, terms = s78.gender_eq2_terms(b)
    g, v = fit(b, f"gender_{SEX_BAND.replace('-', '_')}", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return [], {}
    rows = []
    for r in s78.rows_of(g, terms, block="term", young_band=SEX_BAND,
                         n_firms=n_firms):
        r["t"] = tstat(r["coef"], r["se"])
        diff = r["term"] == "post_x_high_x_young_x_female"
        r["edu_coef"] = EDU_FEMALE[0] if diff else np.nan
        r["edu_se"] = EDU_FEMALE[1] if diff else np.nan
        rows.append(r)
    # The three numbers a reader wants. The women's step is the male step
    # plus the differential, so its standard error comes from the
    # covariance of the fit and not from adding two standard errors.
    m, d = "post_x_high_x_young", "post_x_high_x_young_x_female"
    c = {t_: float(g.loc[t_, "coef"]) for t_ in terms if t_ in g.index}
    e = {t_: float(g.loc[t_, "se"]) for t_ in terms if t_ in g.index}
    steps = {}
    if m in c and d in c:
        steps["male_step"] = (c[m], e[m])
        steps["female_differential"] = (c[d], e[d])
        steps["female_step"] = (c[m] + c[d], s78.lincomb(v, {m: 1, d: 1}))
    for k, (cc, ss) in steps.items():
        rows.append({"block": "step", "young_band": SEX_BAND, "term": k,
                     "coef": cc, "se": (np.nan if ss is None else ss),
                     "t": tstat(cc, ss), "n_obs": int(g["n_obs"].max()),
                     "n_firms": n_firms, "status": "derived",
                     "edu_coef": (EDU_FEMALE[0]
                                  if k == "female_differential" else np.nan),
                     "edu_se": (EDU_FEMALE[1]
                                if k == "female_differential" else np.nan)})
    save(rows, "occ_route_gender.csv")
    if "female_differential" in steps:
        cc, ss = steps["female_differential"]
        print(f"  C: female differential {cc:+.4f} ({ss:.4f}) t "
              f"{tstat(cc, ss):+.2f}   education route "
              f"{EDU_FEMALE[0]:+.4f} ({EDU_FEMALE[1]:.4f})")
    return rows, steps


def part_c_flows(flows, occ, s61, s78, j47) -> list:
    """Hires and separations at 22-25, on the terms the stock uses."""
    if flows is None:
        FAILURES.append("C/flows/no flows cache")
        print("  C: flows_* missing, the margins are skipped")
        return []
    rows = []
    for outcome, col in (("hires", "n_hire"), ("seps", "n_sep")):
        src = flows.rename(columns={col: "n_emp"})
        skel = s61.build_skeleton(src, FLOW_BAND, j47)
        del src
        gc.collect()
        if skel.empty:
            FAILURES.append(f"C/{outcome}/empty")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{outcome}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, _ = fit(b, f"{outcome}_{FLOW_BAND.replace('-', '_')}", terms,
                   j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        ec, es = edu_pair(EDU_FLOW, outcome)
        for r in s78.rows_of(g, terms, outcome=outcome,
                             young_band=FLOW_BAND, n_firms=n_firms):
            r["t"] = tstat(r["coef"], r["se"])
            post = r["term"] == "post_x_high_x_young"
            r["edu_coef"] = ec if post else np.nan
            r["edu_se"] = es if post else np.nan
            rows.append(r)
        p = [r for r in rows if r["outcome"] == outcome
             and r["term"] == "post_x_high_x_young"]
        if p:
            print(f"  C: {outcome} {p[0]['coef']:+.4f} ({p[0]['se']:.4f}) "
                  f"t {p[0]['t']:+.2f}   education route {ec:+.4f} ({es:.4f})")
    if rows:
        save(rows, "occ_route_flows.csv")
    return rows


def vintage_stability(true_e: pd.DataFrame, asof_e: pd.DataFrame) -> dict:
    """How far the later register moves the 2019 classifier."""
    j = true_e.merge(asof_e, on="employer_id", suffixes=("_t", "_a"))
    if j.empty:
        return {}
    keep = float((j["fq_t"] == j["fq_a"]).mean())
    top = float(((j["fq_t"] == 4) == (j["fq_a"] == 4)).mean())
    shift = float(np.mean(np.abs(j["mix_a"] - j["mix_t"])
                          / j["mix_t"].abs().clip(lower=1e-9)))
    rho = float(j["mix_t"].corr(j["mix_a"], method="spearman"))
    return {"n_both": int(len(j)), "share_keeping_quartile": keep,
            "share_keeping_top": top, "mean_relative_mix_shift": shift,
            "spearman": rho,
            "n_true_only": int(len(set(true_e["employer_id"])
                                   - set(asof_e["employer_id"]))),
            "n_asof_only": int(len(set(asof_e["employer_id"])
                                   - set(true_e["employer_id"])))}


def part_c_vintage(counts, occ, daioe, l65, s61, s78, j47) -> tuple:
    """
    The same 2019 incumbents, re-scored from the register of a later
    year, and the distance that moves the adoption step.

    Both arms are fitted here rather than one being read from Part B, so
    the artefact comes from one panel and two scores. That is how script
    47j measures the education artefact and it is the only way the
    difference is the re-scoring and not the sample.
    """
    base_v = baseline_vintage(VINTAGE)
    occ_v = occ_exposure(base_v, daioe, l65, j47)
    del base_v
    gc.collect()
    if occ_v.empty:
        FAILURES.append("C/vintage/no exposure")
        return [], {}
    stab = vintage_stability(occ, occ_v)
    rows = [{"block": "stability", "arm": f"{BASE_YEAR} vs {VINTAGE}",
             "item": k, "coef": np.nan, "se": np.nan, "t": np.nan,
             "value": v, "n_firms": stab.get("n_both", np.nan),
             "n_obs": stab.get("n_both", np.nan), "status": "descriptive"}
            for k, v in stab.items()]
    if stab:
        print(f"  C: vintage stability, "
              f"{stab['share_keeping_quartile']:.1%} of employers keep their "
              f"quartile, {stab['share_keeping_top']:.1%} keep their place in "
              f"or out of the top one, mean relative shift in the score "
              f"{stab['mean_relative_mix_shift']:.2%}")
    skel = s61.build_skeleton(counts, SEX_BAND, j47)
    if skel.empty:
        FAILURES.append("C/vintage/empty panel")
        return rows, stab
    got = {}
    for arm, e in ((f"true_{BASE_YEAR}", occ), (f"asof_{VINTAGE}", occ_v)):
        b = s78.with_exposure(skel, e)
        if b.empty:
            FAILURES.append(f"C/vintage/{arm}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, _ = fit(b, f"vintage_{arm}", terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        ec, es = edu_pair(EDU_STOCK, SEX_BAND)
        for r in s78.rows_of(g, terms, block="fit", arm=arm,
                             n_firms=n_firms):
            r["item"] = r.pop("term")
            r["t"] = tstat(r["coef"], r["se"])
            r["value"] = np.nan
            rows.append(r)
        post = "post_x_high_x_young"
        if post in g.index:
            got[arm] = float(g.loc[post, "coef"])
            print(f"  C: vintage {arm:<12} adoption step "
                  f"{got[arm]:+.4f} ({float(g.loc[post, 'se']):.4f})"
                  f"   education route {ec:+.4f} ({es:.4f})")
    del skel
    gc.collect()
    if len(got) == 2:
        a = got[f"asof_{VINTAGE}"] - got[f"true_{BASE_YEAR}"]
        rows.append({"block": "artefact", "arm": "asof minus true",
                     "item": "post_x_high_x_young", "coef": a, "se": np.nan,
                     "t": np.nan, "value": a, "n_firms": np.nan,
                     "n_obs": np.nan, "status": "derived"})
        stab["artefact"] = a
        print(f"  C: vintage artefact on the adoption step {a:+.4f}")
    save(rows, "occ_route_vintage.csv")
    return rows, stab


def part_c(counts, sexcounts, flows, occ, daioe, s61, s67, s78, l65,
           j47) -> tuple:
    g_rows, steps = part_c_gender(sexcounts, occ, s67, s78, j47)
    f_rows = part_c_flows(flows, occ, s61, s78, j47)
    v_rows, stab = part_c_vintage(counts, occ, daioe, l65, s61, s78, j47)
    return g_rows, steps, f_rows, v_rows, stab


# ----------------------------------------------------------------------
# The three verdicts
# ----------------------------------------------------------------------

def verdict_headline(head: list) -> tuple:
    """Read rule 1, at 22-25, with the 26-30 band reported beside it."""
    L, verdict = [], "NO VERDICT"
    p = [r for r in head if r["young_band"] == "22-25"
         and r["term"] == "post_x_high_x_young"]
    if not p:
        return verdict, ["  NO VERDICT on rule 1: the 22-25 stock fit did "
                         "not come back, and a missing fit is not a null."]
    c, s = float(p[0]["coef"]), float(p[0]["se"])
    t = tstat(c, s)
    ec, es = EDU_STOCK["22-25"]
    verdict = ("REPRODUCES" if (c < 0 and abs(t) >= SIG5)
               else "DOES NOT REPRODUCE")
    L.append(f"  1. THE ADOPTION STEP AT 22-25: {verdict}")
    L.append(f"     occupation route {c:+.4f} ({s:.4f}) t {t:+.2f}; "
             f"education route {ec:+.4f} ({es:.4f})")
    if verdict == "DOES NOT REPRODUCE":
        L.append("     The sign, the size and the t are reported above "
                 "whichever way they fall; a step that is negative but not "
                 "distinguishable from zero is a failure of this rule and "
                 "not a zero.")
    for band in BANDS[1:]:
        q = [r for r in head if r["young_band"] == band
             and r["term"] == "post_x_high_x_young"]
        if q:
            bc, bs = float(q[0]["coef"]), float(q[0]["se"])
            e2 = EDU_STOCK.get(band, (np.nan, np.nan))
            L.append(f"     {band}, not part of the rule: {bc:+.4f} "
                     f"({bs:.4f}) t {tstat(bc, bs):+.2f}; education route "
                     f"{e2[0]:+.4f} ({e2[1]:.4f})")
    return verdict, L


def verdict_profile(prof: list) -> tuple:
    """Read rule 2, on the six-band profile against 41-49."""
    if not prof:
        return "NO VERDICT", ["  NO VERDICT on rule 2: the profile fit did "
                              "not come back."]
    d = {r["band"]: float(r["coef"]) for r in prof}
    if "50+" not in d or "22-25" not in d or len(d) < 6:
        return "NO VERDICT", ["  NO VERDICT on rule 2: the profile is "
                              "incomplete, so the ranking cannot be read."]
    order = sorted(d, key=lambda b: d[b])
    rank = order.index("22-25") + 1
    gains = d["50+"] > 0
    verdict = ("THE PROFILE REPRODUCES" if (gains and rank <= 2)
               else "THE PROFILE DOES NOT REPRODUCE")
    L = [f"  2. THE AGE PROFILE: {verdict}",
         f"     50 and over against 41-49 {d['50+']:+.4f} "
         f"(education route {EDU_PROFILE['50+'][0]:+.4f}), so the older band "
         f"{'gains' if gains else 'does NOT gain'}",
         f"     22-25 is ranked {rank} of six from the bottom "
         f"{'(lowest or second lowest, as the rule requires)' if rank <= 2 else '(the rule requires first or second)'}",
         "     the whole profile, lowest first:"]
    for b in order:
        e = EDU_PROFILE.get(b, (np.nan, np.nan))
        r = next(x for x in prof if x["band"] == b)
        L.append(f"       {b:<6} {d[b]:+.4f} ({float(r['se']):.4f})   "
                 f"education route {e[0]:+.4f} ({e[1]:.4f})")
    return verdict, L


def verdict_gender(steps: dict) -> tuple:
    """Read rule 3, on the female differential."""
    if "female_differential" not in steps:
        return "NO VERDICT", ["  NO VERDICT on rule 3: the sex fit did not "
                              "come back."]
    c, s = steps["female_differential"]
    t = tstat(c, s)
    verdict = ("THE SEX RESULT REPRODUCES" if (c < 0 and abs(t) >= SIG1)
               else "THE SEX RESULT DOES NOT REPRODUCE")
    L = [f"  3. THE FEMALE DIFFERENTIAL: {verdict}",
         f"     occupation route {c:+.4f} ({s:.4f}) t {t:+.2f}; "
         f"education route {EDU_FEMALE[0]:+.4f} ({EDU_FEMALE[1]:.4f})",
         "     the rule is the one per cent level, so the threshold is "
         f"|t| >= {SIG1:.2f}"]
    for k, nm in (("male_step", "young men, adoption step"),
                  ("female_step", "young women, adoption step")):
        if k in steps:
            cc, ss = steps[k]
            L.append(f"     {nm:<28} {cc:+.4f} "
                     + ("(no SE)" if ss is None or ss != ss
                        else f"({ss:.4f})"))
    return verdict, L


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    mc.Tee(OUT / "82_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"82: THE OCCUPATION ROUTE, WITH NO EDUCATION ANYWHERE   "
          f"parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s61, s67, s74, s78, s80, l47, l65, l70, j47 = load_modules()
    daioe = daioe_scores(l70)
    base = baseline(l47)
    occ = occ_exposure(base, daioe, l65, j47)
    if occ.empty:
        raise RuntimeError("no employer could be scored on the occupation "
                           "route; there is nothing to estimate")
    shares = (occ.groupby("fq")["n"].sum() / occ["n"].sum())
    msg = (f"occupation route: {len(occ):,} employers scored, quartile "
           f"shares of incumbent employment "
           + " ".join(f"Q{int(k)} {v:.2f}" for k, v in shares.items()))
    print(f"  {msg}")
    NOTES.append(msg)

    edu = None
    if "A" in PARTS:
        edu = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
        print(f"  education route: {len(edu):,} employers scored")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    last = str(counts["year_month"].max())
    if last < POST_FROM:
        raise RuntimeError(f"the counts end at {last} and the adoption "
                           f"window opens at {POST_FROM}; refusing to run.")
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")
    sexcounts = load_counts("L_counts_sex", s61.PANEL_YEARS,
                            require=["employer_id", "year_month",
                                     "age_group", "gender", "n_emp"]) \
        if "C" in PARTS else None
    flows = load_counts("flows", s61.PANEL_YEARS,
                        require=["employer_id", "year_month", "age_group",
                                 "n_hire", "n_sep"]) if "C" in PARTS else None

    cov_tab, cov_summ, cross = None, [], {}
    head, prof, g_rows, steps, f_rows, v_rows, stab = [], [], [], {}, [], [], {}
    if "A" in PARTS:
        r = opt("Part A", part_a, counts, occ, edu, base, daioe, s61, s80,
                j47)
        if r:
            cov_tab, cov_summ, cross = r
    del base, edu
    gc.collect()
    if "B" in PARTS:
        r = opt("Part B", part_b, counts, occ, s61, s74, s78, l70, j47)
        if r:
            head, prof = r
    if "C" in PARTS:
        r = opt("Part C", part_c, counts, sexcounts, flows, occ, daioe, s61,
                s67, s78, l65, j47)
        if r:
            g_rows, steps, f_rows, v_rows, stab = r
    del counts, sexcounts, flows
    gc.collect()

    # ---- summary ------------------------------------------------------
    L = ["THE OCCUPATION ROUTE, WITH NO EDUCATION ANYWHERE", "=" * 52, "",
         "The paper routes exposure through education: an education group",
         "carries the mean DAIOE percentile of the occupations its holders",
         "worked in during 2019, and an employer is ranked by the mean over",
         "its incumbents aged 31 to 69. Here the intermediate step is",
         "deleted: an employer is ranked by the mean DAIOE percentile of",
         "the 2019 four-digit occupations of its OWN incumbents aged 31 to",
         "69. Same freeze year, same incumbent restriction, same floor,",
         "same quartile weighting, no education record anywhere.", ""]
    if "A" in PARTS:
        L += ["A. THE SCORE AND WHAT IT COVERS (no fit):"]
        if cov_tab is not None:
            cv = cov_tab[(cov_tab["block"] == "coverage")]
            L.append("  share of incumbent head count carrying a 2019 code,")
            L.append("  and the share whose code also carries a DAIOE score:")
            order = [a for a in mc.AGE_GROUPS if a in set(cv["group"])] \
                + [g_ for g_ in sorted(set(cv["group"]))
                   if g_ not in mc.AGE_GROUPS]
            for grp in order:
                d = cv[cv["group"] == grp].set_index("item")
                cs = d.loc["coded_share", "share"] \
                    if "coded_share" in d.index else np.nan
                ss = d.loc["scored_share", "share"] \
                    if "scored_share" in d.index else np.nan
                L.append(f"    {grp:<18} coded "
                         + ("      " if cs != cs else f"{cs:6.1%}")
                         + "   scored "
                         + ("      " if ss != ss else f"{ss:6.1%}"))
            rt = cov_tab[cov_tab["block"] == "route"].set_index("group")
            if len(rt):
                L.append("  employers each route can score:")
                for g_ in ("occupation_route", "education_route",
                           "both_routes", "occupation_only",
                           "education_only"):
                    if g_ in rt.index:
                        L.append(f"    {g_:<18} "
                                 f"{cnt(rt.loc[g_, 'n_employers']):>12}")
            if cross:
                L.append(f"  the two quartiles agree for "
                         f"{cross['diag']:.1%} of the "
                         f"{cnt(cross['n'])} employers both routes score; "
                         f"Spearman rank correlation of the underlying "
                         f"scores {cross['rho']:+.3f}")
                L.append(f"  top quartile: {cnt(cross['top_occ'])} on the "
                         f"occupation route, {cnt(cross['top_edu'])} on the "
                         f"education route, {cnt(cross['top_both'])} on both")
            for s in cov_summ:
                L.append(f"  {s['panel']}: {cnt(s['n'])} employers, "
                         f"{cnt(s['n_occ'])} scorable on the occupation "
                         f"route ({s['n_occ'] / max(s['n'], 1):.1%}), "
                         f"{cnt(s['n_edu'])} on the education route "
                         f"({s['n_edu'] / max(s['n'], 1):.1%}), "
                         f"{cnt(s['n_both'])} on both")
                if s["med_occ_only"] == s["med_occ_only"] \
                        and s["med_edu_only"] == s["med_edu_only"]:
                    L.append(f"    median monthly headcount "
                             f"{s['med_occ_only']:.1f} in the employers only "
                             f"the occupation route scores, "
                             f"{s['med_edu_only']:.1f} in those only the "
                             f"education route scores, {s['med_both']:.1f} in "
                             f"those both score; months employing anybody "
                             f"{s['months_occ_only']:.1f}, "
                             f"{s['months_edu_only']:.1f} and "
                             f"{s['months_both']:.1f}")
        else:
            L.append("  Part A produced nothing")
        L.append("")
    verdicts = []
    if "B" in PARTS:
        L += ["B. THE HEADLINE:"]
        if head:
            L.append(f"  {'band':<6} {'term':<28} {'coef':>9} {'se':>9} "
                     f"{'t':>7}")
            for r in head:
                if r["term"] not in ("rb_x_high_x_young",
                                     "interim_x_high_x_young",
                                     "post_x_high_x_young"):
                    continue
                L.append(f"  {r['young_band']:<6} {r['term']:<28} "
                         f"{r['coef']:+9.4f} {r['se']:9.4f} {r['t']:+7.2f}")
        else:
            L.append("  no stock fit came back")
        v, lines = verdict_headline(head)
        verdicts.append(("1 adoption step at 22-25", v))
        L += [""] + lines
        v, lines = verdict_profile(prof)
        verdicts.append(("2 age profile", v))
        L += [""] + lines + [""]
    if "C" in PARTS:
        L += ["C. THE SEX SPLIT, THE MARGINS AND THE VINTAGE CHECK:"]
        v, lines = verdict_gender(steps)
        verdicts.append(("3 female differential", v))
        L += lines
        if f_rows:
            L.append("  the margins at 22-25, adoption step:")
            for outcome in ("hires", "seps"):
                q = [r for r in f_rows if r["outcome"] == outcome
                     and r["term"] == "post_x_high_x_young"]
                if q:
                    e = EDU_FLOW[outcome]
                    L.append(f"    {outcome:<6} {q[0]['coef']:+.4f} "
                             f"({q[0]['se']:.4f}) t {q[0]['t']:+.2f}   "
                             f"education route {e[0]:+.4f} ({e[1]:.4f})")
        else:
            L.append("  no flow fit came back")
        if stab:
            L.append(f"  the vintage check: the same {BASE_YEAR} incumbents "
                     f"re-scored from the {VINTAGE} register.")
            if "share_keeping_quartile" in stab:
                L.append(f"    {stab['share_keeping_quartile']:.1%} of "
                         f"employers keep their quartile, "
                         f"{stab['share_keeping_top']:.1%} keep their place "
                         f"in or out of the top one, mean relative shift in "
                         f"the score {stab['mean_relative_mix_shift']:.2%}, "
                         f"Spearman {stab['spearman']:+.3f}")
            if "artefact" in stab:
                L.append(f"    the adoption step at 22-25 moves "
                         f"{stab['artefact']:+.4f} when the later codes are "
                         f"used. The education route's own re-scoring moves "
                         f"it +0.0113 (-0.0408 to -0.0295).")
        else:
            L.append("  the vintage check produced nothing")
        L.append("")
    if verdicts:
        L += ["THE THREE VERDICTS:"]
        L += [f"  {k:<28} {v}" for k, v in verdicts]
        L.append("")
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += ["WHAT FAILED: " + "; ".join(FAILURES),
              "A missing row is a missing fit or a missing pull, never a "
              "zero.", ""]
    L += READ_RULES + [
        "",
        "WHAT TO EXPECT, SO IT IS NOT READ AS A BUG:",
        "  1. This route scores far fewer employers. The occupation",
        "     register samples about half the workforce and imputes the",
        "     rest, so an employer needs five CODED incumbents rather than",
        "     five classified ones. Script 70's ladder put the education",
        "     route at 311,227 employers and this one at 65,146. Standard",
        "     errors here will therefore be larger, and read rule 1 asks",
        "     for significance on that smaller sample.",
        "  2. About a third of 2019 occupation codes were assigned in an",
        "     earlier year. That is measurement error in the regressor, it",
        "     attenuates towards zero, and it cannot manufacture a result.",
        "  3. The two routes are standardised on their own distributions",
        "     and rank employers differently, so agreement in SIZE is not",
        "     expected and is not what any of the three rules asks for.",
        "  4. The weight and the floor are November head counts here and",
        "     person-months on the education route. The floor binds on the",
        "     same object, an employer too thin to classify, on a",
        "     different scale.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "82_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("82_occupation_route", 0, (time.time() - t0) / 60)
    print("\n82 done.")


if __name__ == "__main__":
    main()

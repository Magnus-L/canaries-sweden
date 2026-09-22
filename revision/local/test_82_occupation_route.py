#!/usr/bin/env python3
"""
test_82_occupation_route.py -- the occupation route must score the firm
                               from its INCUMBENTS' 2019 work and from
                               nothing else, and the three read rules
                               must be able to fail.

The checks are on mechanisms, not on outputs. One synthetic world on 140
employers, with a 2019 occupation baseline built so that every rule the
score claims to follow has something to catch:

  1-45      their INCUMBENTS hold high-DAIOE occupations and their YOUNG
            hold low-DAIOE ones. Every other firm is the mirror image. A
            score that let the young in would therefore rank the firms
            backwards, so "the young never enter" is checked against a
            prediction and not against a comment.
  101-106   exactly four coded incumbents. The floor is five, so they must
            not be scored, and they must still appear on the panel and be
            reported as unscorable rather than vanish.
  138-140   absent from the education frame altogether, so exactly three
            employers are scorable on the occupation route alone. Three is
            below the export floor, so that row must leave MONA
            suppressed; without it the suppression check would be vacuous.
  every firm carries an uncoded '____' cell in every band, so the coverage
            denominators have something to cover and a '____' can be seen
            not to be scored.

The employment is drawn with the planted steps on the employers the
OCCUPATION route puts in its top quartile, which is computed first and not
assumed: a Q4 seasonal bump and a tightening rise at 22-25 so the calendar
terms and the Riksbank switch have work to do, an adoption step at 22-25
with an extra step for young women, a gain at 50 and over, and more
separations at 22-25 with hiring left flat. The three read rules should
then all be met, and each verdict function is also fed a planted null, so
the test establishes that they can fail and not only that they can pass.

Also tested: that the quartile cuts are weighted by incumbent employment
rather than by employers, on a frame where a few large firms make the two
answers visibly different; that re-scoring the same incumbents from a
later register moves some of them and that the movement is reported; that
the vintage query takes the birth year from the 2019 register and the code
from the later one, so the population cannot move with the coding; that
counts of one to four never leave MONA; and main() end to end off the
caches with a summary that states every read rule and every verdict.

    CANARIES_DRYRUN=1 python3 revision/local/test_82_occupation_route.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
os.environ["CANARIES_82_PARTS"] = "ABC"
# The script's Tee caps the terminal echo at 2 KB, which is right under
# BatchClient's blocking pipe and wrong here: it would swallow the check
# lines printed after main(). The lane runner lifts it the same way.
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries82_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
# 82's OUT is HERE / the environment variable, and pathlib lets an
# absolute value win, so the sandbox gets every export and the repository
# gets no output_82 folder from a test run.
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
mc.connect = lambda: object()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s82 = load("82_occupation_route.py", "s82")
s82.CACHE = mc.CACHE_DIR
s82.BASE_CACHE = mc.CACHE_DIR / "L_baseline_2019.parquet"
s82.VINT_CACHE = mc.CACHE_DIR / "L_baseline_2019_asof2021.parquet"
s61, s67, s74, s78, s80, l47, l65, l70, j47 = s82.load_modules()
h47 = j47._h47()
for m_ in (s61, s67, s74, s78, s80, l47, l65, l70, j47):
    m_.OUT = s82.OUT
    m_.CACHE = mc.CACHE_DIR

from _fixtures import Fixture  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================
# the 2019 occupation baseline: which firm's incumbents did what
# ======================================================================
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
YOUNG = ("22-25", "26-30")
N_FIRMS = 140
FIRMS = list(range(1, N_FIRMS + 1))
EXPOSED = set(range(1, 46))          # their INCUMBENTS do high-DAIOE work
FLOOR_FIRMS = list(range(101, 107))  # exactly four coded incumbents
N_EDU_FIRMS = 137                    # so 138-140 are occupation-only
OCC_ONLY = [138, 139, 140]

DAIOE = pd.read_stata(_LOCAL)
DAIOE["ssyk4"] = DAIOE["ssyk4"].astype(str).str.zfill(4)
SCORES = DAIOE.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                             "score"]]
HI = sorted(DAIOE.loc[DAIOE.high_exposure == 1, "ssyk4"])
LO = sorted(DAIOE.loc[DAIOE.high_exposure == 0, "ssyk4"])
# One four-digit code the DAIOE file does not carry, so the difference
# between "coded" and "scored" is a real gap and not a rounding.
UNSCORED_CODE = "9999"
assert UNSCORED_CODE not in set(DAIOE["ssyk4"])


def occ_baseline(swap_young: bool = False, floor_n: int = 4,
                 recoded=frozenset()) -> pd.DataFrame:
    """
    47L's baseline frame, drawn deterministically so that two arms differ
    in exactly the thing being varied.

    `swap_young` flips the young cells' occupations and leaves the
    incumbents alone, which is the invariance the design claims.
    `recoded` is the set of firms whose INCUMBENTS the later register
    files under low-exposure work; the head counts are untouched, so the
    population cannot move with the coding.
    """
    rows = []
    for emp in FIRMS:
        hi = (emp in EXPOSED) and (emp not in recoded)
        for age in AGES:
            young = age in YOUNG
            want_hi = (not hi) if young else hi
            if swap_young and young:
                want_hi = not want_hi
            pool = HI if want_hi else LO
            if emp in FLOOR_FIRMS and not young:
                # one coded cell in one band only, so the whole firm has
                # `floor_n` coded incumbents and the floor decides it
                if age == "31-34":
                    rows.append((emp, age, pool[(emp * 7) % len(pool)], "1",
                                 floor_n))
            else:
                for i in range(3):
                    rows.append((emp, age, pool[(emp * 5 + i * 37) % len(pool)],
                                 "1", 4 + (emp + i) % 8))
            # a code the DAIOE file does not hold, and a worker the
            # register leaves uncoded: both count in the denominator and
            # neither can carry an exposure
            rows.append((emp, age, UNSCORED_CODE, "1", 2))
            rows.append((emp, age, "____", "", 3))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "ssyk_status", "n"])


BASE = occ_baseline()
BASE.to_parquet(s82.BASE_CACHE, index=False)
OCC = s82.occ_exposure(BASE, SCORES, l65, j47)
Q4F = set(OCC.loc[OCC["fq"] == 4, "employer_id"].astype(int))

check("the occupation route scores most firms and puts some of them in "
      "the top quartile",
      0 < len(Q4F) < len(OCC) and len(OCC) == N_FIRMS - len(FLOOR_FIRMS),
      f"{len(OCC)} scored, {len(Q4F)} in Q4")
check("only firms whose INCUMBENTS do high-exposure work reach the top "
      "quartile, so the young never enter the score",
      Q4F <= EXPOSED and len(Q4F) > 10,
      f"{len(Q4F)} in Q4, {len(Q4F - EXPOSED)} of them not incumbent-exposed")
SWAPPED = s82.occ_exposure(occ_baseline(swap_young=True), SCORES, l65, j47)
check("and changing ONLY the young cells leaves the score identical",
      SWAPPED[["employer_id", "fq", "mix"]].equals(
          OCC[["employer_id", "fq", "mix"]]),
      f"{int((SWAPPED['fq'].to_numpy() != OCC['fq'].to_numpy()).sum())} "
      f"quartiles moved")
check("an employer with four coded incumbents is below the floor and is "
      "not scored",
      not (set(OCC["employer_id"].astype(int)) & set(FLOOR_FIRMS)),
      f"{len(set(OCC['employer_id'].astype(int)) & set(FLOOR_FIRMS))} of "
      f"{len(FLOOR_FIRMS)} scored")
FIVE = s82.occ_exposure(occ_baseline(floor_n=5), SCORES, l65, j47)
check("and the same employer with five is scored, so the floor is the "
      "thing deciding it and not the firm",
      set(FLOOR_FIRMS) <= set(FIVE["employer_id"].astype(int)),
      f"{len(set(FIVE['employer_id'].astype(int)) & set(FLOOR_FIRMS))} of "
      f"{len(FLOOR_FIRMS)} scored at a floor of five")
# The '____' convention: it must not be scored, and it must not be
# silently dropped from the denominator either.
cov = s82.coverage_by_band(BASE, SCORES, j47)
c31 = {r["item"]: r for r in cov if r["group"] == "31-69 incumbents"}
check("the uncoded '____' cells are not scored and still count in the "
      "denominator",
      0 < c31["coded_share"]["share"] < 1
      and c31["scored_share"]["share"] < c31["coded_share"]["share"],
      f"coded {c31['coded_share']['share']:.3f}, "
      f"scored {c31['scored_share']['share']:.3f}")
check("and a code the DAIOE file does not hold is coded but not scored, "
      "so the two shares are not the same number",
      abs(c31["coded_share"]["share"] - c31["scored_share"]["share"]) > 1e-6)


# ---- the quartile cuts are weighted by employment, not by employers ----
# A few large firms at the top of the score make the two answers visibly
# different: a quarter of INCUMBENT EMPLOYMENT is not a quarter of
# employers, and the design says the first.
def lopsided() -> pd.DataFrame:
    rows = []
    for i in range(100):
        emp = 1000 + i
        big = i >= 90
        code = (HI if big else LO)[(i * 3) % (len(HI) if big else len(LO))]
        rows.append((emp, "41-49", code, "1", 100 if big else 5))
    return pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "ssyk_status", "n"])


LOP = s82.occ_exposure(lopsided(), SCORES, l65, j47)
w_top = float(LOP.loc[LOP["fq"] == 4, "n"].sum() / LOP["n"].sum())
f_top = float((LOP["fq"] == 4).mean())
check("the quartile cut points are weighted by incumbent employment: the "
      "top quartile holds about a quarter of employment and far fewer "
      "than a quarter of employers",
      0.18 <= w_top <= 0.35 and f_top < 0.12,
      f"{w_top:.1%} of incumbent employment, {f_top:.1%} of employers")


# ======================================================================
# the education route, and the panel with the planted steps
# ======================================================================
FIX = Fixture(mc, h47, n_firms=N_EDU_FIRMS, n_exposed=45)
FIX.install_edu([2019], cache=mc.CACHE_DIR)
EDU = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
check("the education route scores the firms the fixture gives it, and "
      "not the three the occupation route has to itself",
      len(EDU) > 100 and not (set(EDU["employer_id"].astype(int))
                              & set(OCC_ONLY)),
      f"{len(EDU)} firms on the education route")

SEAS = float(np.log(1.22))     # the Q4 bump in every year, 22-25
STEP_RB = float(np.log(1.05))  # the tightening rise, 22-25
STEP_22 = float(np.log(0.80))  # young men's adoption step, 22-25
DIFF = float(np.log(0.85))     # young women's extra step, 22-25
STEP_50 = float(np.log(1.10))  # the gain at 50 and over
STEP_SEP = float(np.log(1.25))  # separations at 22-25
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]


def counts_by_sex() -> pd.DataFrame:
    rng = np.random.default_rng(82)
    lam0 = {"22-25": 26, "26-30": 20, "31-34": 14, "35-40": 16,
            "41-49": 20, "50+": 22}
    rows = []
    for emp in FIRMS:
        hit = emp in Q4F
        for ym in MONTHS:
            q = (int(ym[5:7]) - 1) // 3 + 1
            for age, lam in lam0.items():
                for g in ("1", "2"):
                    lam_ = lam / 2
                    if hit and age == "22-25":
                        if q == 4:
                            lam_ *= np.exp(SEAS)
                        if ym >= mc.RIKSBANK_YM:
                            lam_ *= np.exp(STEP_RB)
                        if ym >= s82.POST_FROM:
                            lam_ *= np.exp(STEP_22)
                            if g == "2":
                                lam_ *= np.exp(DIFF)
                    if hit and age == "50+" and ym >= s82.POST_FROM:
                        lam_ *= np.exp(STEP_50)
                    rows.append((emp, ym, age, g, int(rng.poisson(lam_)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "gender", "n_emp"])


def flows_frame() -> pd.DataFrame:
    """54's flow cache: hires flat, separations up at 22-25 after adoption."""
    rng = np.random.default_rng(8200)
    rows = []
    for emp in FIRMS:
        hit = emp in Q4F
        for ym in MONTHS:
            for age in AGES:
                h, s = 5.0, 5.0
                if hit and age == "22-25" and ym >= s82.POST_FROM:
                    s *= np.exp(STEP_SEP)
                rows.append((emp, ym, age, int(rng.poisson(h)) + 1,
                             int(rng.poisson(s)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_hire", "n_sep"])


SEX = counts_by_sex()
COUNTS = (SEX.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
FLOWS = flows_frame()
for y in s61.PANEL_YEARS:
    ys = str(y)
    COUNTS[COUNTS["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    SEX[SEX["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)
    FLOWS[FLOWS["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"flows_{y}.parquet", index=False)

# Count the fits, so "Part A runs none" is checked by the fit that does
# not happen and not by a log line.
FITS = []
_fit = s82.fit


def counting_fit(b, tag, *a, **kw):
    FITS.append(tag)
    return _fit(b, tag, *a, **kw)


s82.fit = counting_fit


# ======================================================================
# A. the score and what it covers
# ======================================================================
s82.NOTES.clear(); s82.FAILURES.clear(); FITS.clear()
COV, CSUM, CROSS = s82.part_a(COUNTS, OCC, EDU, BASE, SCORES, s61, s80, j47)
check("A: no fit runs in Part A", not FITS, str(FITS))
check("A: the coverage of the 2019 code is reported for every age band "
      "and for the incumbents as a whole",
      {"coded_share", "scored_share"}
      == set(COV[COV["block"] == "coverage"]["item"])
      and set(AGES + ["31-69 incumbents"])
      <= set(COV[COV["block"] == "coverage"]["group"]),
      str(sorted(set(COV[COV['block'] == 'coverage']['group']))))
rt = COV[COV["block"] == "route"].set_index("group")
check("A: the employers each route can score are counted, and the three "
      "the occupation route has to itself are SUPPRESSED, being fewer "
      "than five",
      pd.isna(rt.loc["occupation_only", "n_employers"])
      and int(rt.loc["both_routes", "n_employers"])
      == len(set(OCC["employer_id"].astype(int))
             & set(EDU["employer_id"].astype(int))),
      f"occupation_only {rt.loc['occupation_only', 'n_employers']}, "
      f"planted {len(OCC_ONLY)}")
check("A: a suppressed count takes its own share with it",
      pd.isna(rt.loc["occupation_only", "share"])
      and pd.isna(rt.loc["occupation_only", "value"]))
check("A: the employers below the incumbent floor are reported as "
      "scorable on the education route alone, not dropped",
      int(rt.loc["education_only", "n_employers"]) >= len(FLOOR_FIRMS),
      f"{int(rt.loc['education_only', 'n_employers'])} education-only "
      f"against {len(FLOOR_FIRMS)} below the floor")
check("A: the crosstab of the two quartiles is exported with the "
      "diagonal share and the rank correlation",
      CROSS and 0.0 <= CROSS["diag"] <= 1.0 and -1.0 <= CROSS["rho"] <= 1.0
      and {"share_on_diagonal", "spearman_rank_correlation"}
      <= set(COV[COV["block"] == "crosstab"]["item"]),
      f"diagonal {CROSS.get('diag', float('nan')):.1%}, "
      f"Spearman {CROSS.get('rho', float('nan')):+.3f}")
check("A: and the two routes agree about most firms, as two measures of "
      "the same thing should",
      CROSS["rho"] > 0.3 and CROSS["diag"] > 0.25,
      f"Spearman {CROSS['rho']:+.3f}, diagonal {CROSS['diag']:.1%}")
pan = COV[(COV["block"] == "panel") & (COV["panel"] == "22-25 stock")]
g = pan.set_index(["group", "item"])
check("A: both young panels are described",
      {"22-25 stock", "26-30 stock"} <= set(COV["panel"]),
      str(sorted(set(COV["panel"]))))
GRP = {k: float(g.loc[(k, "n_employers"), "n_employers"])
       for k in ("scored_by_both", "occupation_only", "education_only",
                 "scored_by_neither")}
check("A: the panel is the one the fits run on, and every employer on it "
      "is placed in exactly one of the four groups, the suppressed one "
      "included",
      int(g.loc[("all", "employers_on_panel"), "n_employers"]) == N_FIRMS
      and float(np.nansum(list(GRP.values()))) + len(OCC_ONLY) == N_FIRMS
      and pd.isna(GRP["occupation_only"]),
      f"{int(g.loc[('all', 'employers_on_panel'), 'n_employers'])} on the "
      f"panel, groups {GRP}")
SZ = COV[(COV["block"] == "size") & (COV["panel"] == "22-25 stock")]
check("A: the size and longevity of the employers one route places and "
      "the other does not are exported",
      {"mean_headcount_p50", "months_active_mean"}
      <= set(SZ[SZ["group"] == "education_only"]["item"])
      and {"scored_by_both", "occupation_only", "education_only"}
      <= set(SZ["group"]),
      str(sorted(set(SZ[SZ['group'] == 'education_only']['item'])))[:90])
check("A: and the group too small to describe gets its statistics "
      "suppressed rather than published",
      SZ[(SZ["group"] == "occupation_only")
         & (SZ["item"] == "mean_headcount_p50")]["value"].isna().all(),
      f"{len(SZ[SZ['group'] == 'occupation_only'])} rows for the three "
      f"occupation-only employers")
small = COV["n_employers"]
check("A: nothing between one and four leaves MONA",
      not ((small > 0) & (small < s82.FLOOR)).any()
      and int(small.isna().sum()) >= 1,
      f"{int(small.isna().sum())} suppressed cells")
check("A: the export is on disk",
      (s82.OUT / "occ_route_coverage.csv").exists())


# ======================================================================
# B. the headline and the profile
# ======================================================================
s82.NOTES.clear(); FITS.clear()
HEAD, PROF = s82.part_b(COUNTS, OCC, s61, s74, s78, l70, j47)
check("B: three fits run, the two stock bands and the profile",
      len(FITS) == 3, str(FITS))
H = pd.DataFrame(HEAD)
check("B: both young bands are exported with every term of Equation (2)",
      set(H["young_band"]) == {"22-25", "26-30"}
      and {"rb_x_high_x_young", "q1_x_high_x_young", "q2_x_high_x_young",
           "q3_x_high_x_young", "interim_x_high_x_young",
           "post_x_high_x_young"} <= set(H["term"]),
      str(sorted(set(H["term"])))[:110])
p22 = H[(H.young_band == "22-25") & (H.term == "post_x_high_x_young")].iloc[0]
check("B: the planted adoption step at 22-25 comes back",
      abs(float(p22["coef"]) - (STEP_22 + float(np.log(
          (1.0 + np.exp(DIFF)) / 2.0)))) < 0.08,
      f"{float(p22['coef']):+.4f} against "
      f"{STEP_22 + float(np.log((1.0 + np.exp(DIFF)) / 2.0)):+.4f}")
check("B: and the planted calendar bump is in the Q4 normalisation, so "
      "the three quarter terms are negative",
      (H[(H.young_band == "22-25")
         & (H.term.isin(["q1_x_high_x_young", "q2_x_high_x_young",
                         "q3_x_high_x_young"]))]["coef"] < 0).all(),
      str([round(float(x), 3) for x in
           H[(H.young_band == "22-25")
             & (H.term.str.startswith("q"))]["coef"]]))
check("B: the education-route number travels with the adoption step and "
      "with no other term",
      abs(float(p22["edu_coef"]) - s82.EDU_STOCK["22-25"][0]) < 1e-9
      and H[H.term != "post_x_high_x_young"]["edu_coef"].isna().all(),
      f"{float(p22['edu_coef']):+.4f}")
check("B: the employer count and the number of observations are on every "
      "row",
      H["n_firms"].notna().all() and H["n_obs"].notna().all()
      and int(H["n_firms"].iloc[0]) > 0)
P = pd.DataFrame(PROF)
check("B: the profile is exported for all six bands against 41-49",
      set(P["band"]) == set(s74.BANDS)
      and float(P[P.band == "41-49"]["coef"].iloc[0]) == 0.0,
      str(sorted(set(P["band"]))))
check("B: the planted gain at 50 and over comes back against 41-49",
      abs(float(P[P.band == "50+"]["coef"].iloc[0]) - STEP_50) < 0.08,
      f"{float(P[P.band == '50+']['coef'].iloc[0]):+.4f} against "
      f"{STEP_50:+.4f}")
check("B: the education-route profile travels beside it",
      abs(float(P[P.band == "50+"]["edu_coef"].iloc[0])
          - s82.EDU_PROFILE["50+"][0]) < 1e-9)
V1, L1 = s82.verdict_headline(HEAD)
V2, L2 = s82.verdict_profile(PROF)
check("B: read rule 1 is met on the planted world", V1 == "REPRODUCES",
      V1 + " | " + L1[1].strip())
check("B: read rule 2 is met on the planted world",
      V2 == "THE PROFILE REPRODUCES", V2 + " | " + L2[1].strip())
check("B: the exports are on disk and carry the covariance",
      (s82.OUT / "occ_route_headline.csv").exists()
      and (s82.OUT / "occ_route_profile.csv").exists()
      and len(list(s82.OUT.glob("vcov_s82_stock_*.csv"))) == 2,
      f"{len(list(s82.OUT.glob('vcov_s82_*.csv')))} covariance files")

# ---- the rules must be able to FAIL, or they are decoration -----------
NULL_HEAD = [{"young_band": "22-25", "term": "post_x_high_x_young",
              "coef": -0.0007, "se": 0.0200, "n_obs": 1, "n_firms": 1}]
POS_HEAD = [{"young_band": "22-25", "term": "post_x_high_x_young",
             "coef": +0.0600, "se": 0.0100, "n_obs": 1, "n_firms": 1}]
check("rule 1 fails on a step that is negative but not distinguishable "
      "from zero",
      s82.verdict_headline(NULL_HEAD)[0] == "DOES NOT REPRODUCE")
check("rule 1 fails on a step that is significant and POSITIVE",
      s82.verdict_headline(POS_HEAD)[0] == "DOES NOT REPRODUCE")


def prof_of(d):
    return [{"band": b, "coef": c, "se": 0.01} for b, c in d.items()]


FLAT = prof_of({"22-25": -0.05, "26-30": -0.01, "31-34": 0.01,
                "35-40": 0.02, "41-49": 0.0, "50+": -0.02})
MID = prof_of({"22-25": -0.01, "26-30": -0.05, "31-34": -0.04,
               "35-40": 0.02, "41-49": 0.0, "50+": 0.05})
check("rule 2 fails when the older band does NOT gain",
      s82.verdict_profile(FLAT)[0] == "THE PROFILE DOES NOT REPRODUCE")
check("rule 2 fails when the young band is only the third lowest",
      s82.verdict_profile(MID)[0] == "THE PROFILE DOES NOT REPRODUCE")
check("rule 2 gives NO VERDICT rather than a pass when the profile is "
      "incomplete",
      s82.verdict_profile(prof_of({"22-25": -0.05, "50+": 0.05}))[0]
      == "NO VERDICT")


# ======================================================================
# C. the sex split, the margins and the vintage check
# ======================================================================
# The later register files a third of the exposed firms' incumbents under
# low-exposure work; the head counts are untouched, so anything that moves
# is the coding and not the sample.
RECODED = set(sorted(Q4F)[::3])
VINT = occ_baseline(recoded=RECODED)
check("the vintage frame re-codes without moving anybody: the head count "
      "per employer and band is identical",
      VINT.groupby(["employer_id", "age_group"])["n"].sum().equals(
          BASE.groupby(["employer_id", "age_group"])["n"].sum()),
      f"{len(RECODED)} firms re-coded")

# First with no cache on the share, so the pull path and the cache write
# are exercised rather than assumed.
SQL_CALLS = []


def fake_read_sql(q, conn=None, *a, **kw):
    SQL_CALLS.append(str(q))
    return VINT.copy()


_real_read_sql = pd.read_sql
pd.read_sql = fake_read_sql
s82.VINT_CACHE.unlink(missing_ok=True)
s82.NOTES.clear(); FITS.clear()
VROWS, STAB = s82.part_c_vintage(COUNTS, OCC, SCORES, l65, s61, s78, j47)
pd.read_sql = _real_read_sql
check("C: the vintage arm pulls once when the cache is absent and caches "
      "what it pulled",
      len(SQL_CALLS) == 1 and s82.VINT_CACHE.exists(),
      f"{len(SQL_CALLS)} SQL calls")
q = SQL_CALLS[0] if SQL_CALLS else ""
check("C: the query takes the BIRTH YEAR from the 2019 register and the "
      "CODE from the later one, so a worker the later register does not "
      "hold loses his code and not his place",
      "Individ_2019 b" in q and f"Individ_{s82.VINTAGE} v" in q
      and "b.FodelseAr" in q and "v.Ssyk4_2012_J16" in q
      and "b.Ssyk4_2012_J16" not in q,
      q.strip().splitlines()[0][:60] if q else "no query")
check("C: and the sample filter is on the 2019 record, so the population "
      "is the same in both arms",
      "WHERE 2019 - TRY_CAST(b.FodelseAr AS INT) BETWEEN 22 AND 69" in q)
check("C: at the base year the query is 47L's own, the two joins being "
      "the same table",
      f"Individ_{s82.BASE_YEAR} v"
      in s82.baseline_vintage_sql(s82.BASE_YEAR))
check("C: the re-scoring moves some employers out of their quartile and "
      "not all of them",
      0.0 < STAB["share_keeping_quartile"] < 1.0
      and STAB["mean_relative_mix_shift"] > 0,
      f"{STAB['share_keeping_quartile']:.1%} keep their quartile, mean "
      f"relative shift {STAB['mean_relative_mix_shift']:.2%}")
check("C: both arms are fitted on one panel, so the artefact is the "
      "re-scoring and not the sample",
      len(FITS) == 2 and "artefact" in STAB
      and len({r["n_obs"] for r in VROWS if r["block"] == "fit"}) == 1,
      f"{FITS}, artefact {STAB.get('artefact', float('nan')):+.4f}")
check("C: and the artefact points towards zero, since the later codes "
      "misclassify firms that were correctly placed in 2019",
      STAB["artefact"] > 0,
      f"{STAB['artefact']:+.4f} on a true step of "
      f"{[r['coef'] for r in VROWS if r.get('item') == 'post_x_high_x_young' and r['block'] == 'fit'][0]:+.4f}")

s82.NOTES.clear(); FITS.clear()
GROWS, STEPS = s82.part_c_gender(SEX, OCC, s67, s78, j47)
G = pd.DataFrame(GROWS)
check("C: the sex specification runs one fit and exports every term",
      len(FITS) == 1
      and {"post_x_high_x_young", "post_x_high_x_female",
           "post_x_high_x_young_x_female"} <= set(G["term"]),
      str(FITS))
check("C: the planted female differential comes back",
      abs(STEPS["female_differential"][0] - DIFF) < 0.08,
      f"{STEPS['female_differential'][0]:+.4f} against {DIFF:+.4f}")
check("C: young men's step comes back too",
      abs(STEPS["male_step"][0] - STEP_22) < 0.08,
      f"{STEPS['male_step'][0]:+.4f} against {STEP_22:+.4f}")
check("C: the women's step is the sum and its standard error comes from "
      "the covariance of the fit, not from adding two standard errors",
      STEPS["female_step"][1] is not None
      and abs(STEPS["female_step"][0] - (STEP_22 + DIFF)) < 0.09
      and STEPS["female_step"][1] < (STEPS["male_step"][1]
                                     + STEPS["female_differential"][1]),
      f"{STEPS['female_step'][0]:+.4f} ({STEPS['female_step'][1]:.4f})")
check("C: the education-route differential travels beside ours",
      abs(float(G[G.term == "post_x_high_x_young_x_female"]["edu_coef"]
                .iloc[0]) - s82.EDU_FEMALE[0]) < 1e-9)
V3, L3 = s82.verdict_gender(STEPS)
check("C: read rule 3 is met on the planted world",
      V3 == "THE SEX RESULT REPRODUCES", V3 + " | " + L3[1].strip())
check("rule 3 fails at the one per cent level on a differential that "
      "would pass at five",
      s82.verdict_gender({"female_differential": (-0.045, 0.021)})[0]
      == "THE SEX RESULT DOES NOT REPRODUCE",
      "t = -2.14, which clears five per cent and not one")
check("rule 3 fails on a POSITIVE differential however precise",
      s82.verdict_gender({"female_differential": (0.080, 0.005)})[0]
      == "THE SEX RESULT DOES NOT REPRODUCE")

s82.NOTES.clear(); FITS.clear()
FROWS = s82.part_c_flows(FLOWS, OCC, s61, s78, j47)
F = pd.DataFrame(FROWS)
check("C: hires and separations are both fitted at 22-25",
      len(FITS) == 2 and set(F["outcome"]) == {"hires", "seps"}, str(FITS))
fs = F[(F.outcome == "seps") & (F.term == "post_x_high_x_young")].iloc[0]
fh = F[(F.outcome == "hires") & (F.term == "post_x_high_x_young")].iloc[0]
check("C: the planted rise in separations comes back and hiring stays flat",
      abs(float(fs["coef"]) - STEP_SEP) < 0.08
      and abs(float(fh["coef"])) < 0.05,
      f"separations {float(fs['coef']):+.4f} against {STEP_SEP:+.4f}, "
      f"hires {float(fh['coef']):+.4f}")
check("C: the education-route margins travel beside them",
      abs(float(fs["edu_coef"]) - s82.EDU_FLOW["seps"][0]) < 1e-9
      and abs(float(fh["edu_coef"]) - s82.EDU_FLOW["hires"][0]) < 1e-9)
for nm in ("occ_route_gender.csv", "occ_route_flows.csv",
           "occ_route_vintage.csv"):
    check(f"C: {nm} is on disk", (s82.OUT / nm).exists())


# ======================================================================
# end to end
# ======================================================================
for f in s82.OUT.glob("*.csv"):
    f.unlink()
s82.NOTES.clear(); s82.FAILURES.clear(); FITS.clear()
SQL_CALLS.clear()
pd.read_sql = fake_read_sql
s82.main()
pd.read_sql = _real_read_sql
for nm in ("occ_route_coverage.csv", "occ_route_headline.csv",
           "occ_route_profile.csv", "occ_route_gender.csv",
           "occ_route_flows.csv", "occ_route_vintage.csv",
           "82_summary.txt"):
    check(f"main() writes {nm}", (s82.OUT / nm).exists())
check("main() made no SQL call, since both baselines are cached",
      not SQL_CALLS, f"{len(SQL_CALLS)} calls")
check("main() ran eight fits: three for B and five for C",
      len(FITS) == 8, str(FITS))
check("no part failed in the end-to-end run", not s82.FAILURES,
      str(s82.FAILURES))
check("the covariance of every fit left with the exports",
      len(list(s82.OUT.glob("vcov_s82_*.csv"))) >= 8,
      f"{len(list(s82.OUT.glob('vcov_s82_*.csv')))} files")
summ = (s82.OUT / "82_summary.txt").read_text(encoding="utf-8")
for must in ("READ RULES, FIXED BEFORE THE RUN",
             "There is NO coefficient gate",
             "1. REPRODUCES if the adoption step at 22-25",
             "2. THE PROFILE REPRODUCES if the 50-and-over band",
             "3. THE SEX RESULT REPRODUCES if the female differential",
             "-0.0408 (0.0150)", "-0.0746 (0.0142)", "+0.0787 (0.0203)",
             "A. THE SCORE AND WHAT IT COVERS",
             "share of incumbent head count carrying a 2019 code",
             "employers each route can score",
             "Spearman rank correlation",
             "B. THE HEADLINE",
             "1. THE ADOPTION STEP AT 22-25: REPRODUCES",
             "2. THE AGE PROFILE: THE PROFILE REPRODUCES",
             "3. THE FEMALE DIFFERENTIAL: THE SEX RESULT REPRODUCES",
             "the margins at 22-25",
             "the vintage check",
             "THE THREE VERDICTS",
             "WHAT TO EXPECT, SO IT IS NOT READ AS A BUG"):
    check(f"the summary states {must!r}", must in summ)
check("the summary names no failure, since none happened",
      "WHAT FAILED" not in summ)
cov2 = pd.read_csv(s82.OUT / "occ_route_coverage.csv")
for col in ("n_employers", "n_obs"):
    v = cov2[col]
    check(f"nothing between one and four leaves MONA in {col}",
          not ((v > 0) & (v < s82.FLOOR)).any(),
          f"{int(v.isna().sum())} suppressed cells")
for nm in ("occ_route_headline.csv", "occ_route_profile.csv",
           "occ_route_gender.csv", "occ_route_flows.csv",
           "occ_route_vintage.csv"):
    d = pd.read_csv(s82.OUT / nm)
    v = d["n_firms"]
    check(f"and in {nm}", not ((v > 0) & (v < s82.FLOOR)).any(),
          f"{len(d)} rows")

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""
test_83_occupation_route_rest.py -- the rest of the paper must run on
                                    LANE 28'S score and on no other, the
                                    first stage must be able to fail, and
                                    the one gate must be able to catch a
                                    moved panel.

The checks are on mechanisms, not on outputs. One synthetic world, drawn
so that every rule the script claims to follow has something to catch:

  1000 employers carry a 2019 occupation mix and a 2019 person-month
       count, so the score is built over a population large enough for
       script 71's OWN sample gates (800 firms with an outcome and 100 of
       them exposed; 600 respondents and 80 exposed) to be met rather
       than lowered for the test. Nothing in 71 or 73 is relaxed here.
  the first 600 of them also carry a monthly panel, so the fits run on a
       subset of the scored employers, which is the real shape: every
       panel in this lane is a subset of the education route's.
  each employer's incumbents hold ONE occupation, drawn from four codes
       in four different three-digit groups at four separated points of
       the DAIOE distribution, so the employment-weighted quartile is the
       planted tier and the test can name which firms must be exposed
       rather than read it off the result.
  every employer also holds an uncoded cell and a cell whose code is not
       in the DAIOE file, so coverage is below one and the coded and the
       scored share are different numbers.

  REPORTED AI USE is planted on the OCCUPATION quartile and on nothing
       else, in three firm waves and one individual wave. The education
       route is built from the same fixture and is correlated with it but
       not equal to it, so a first stage computed on the wrong quartile
       returns the wrong number and the test can tell.
  EMPLOYMENT is planted with an adoption fall at 22-25 in the exposed
       firms, a rise during the tightening window, a FLAT pre-launch
       path, and growth in every band after adoption that is largest at
       50 and over, which is the shape the education route reports and
       the descriptive arm has to recover.
  INDUSTRY is planted so that ten employers carry no code at all, which
       must leave them out of BOTH fits of the industry test and in the
       one residual cluster of the clustering arm.
  A BALANCE SHEET is planted for most but not all employers, with
       leverage spread so a median split actually splits.

Also tested: that the score is lane 28's own, bit for bit, and is not
rebuilt; that 80's prior-export search path is emptied, so the
four-decimal gate checks this lane's own employer-clustered run and not
the education route's; that the drift window is January 2021 to November
2022; that the reference window's tightening term covers April to
November 2022 only; that the industry test's two fits run on identical
firm sets; that the credit test's baseline is re-estimated on the
balance-sheet sample and not taken from the full panel; that every
verdict can fail as well as pass, each fed a planted null; and main()
end to end.

    CANARIES_DRYRUN=1 python3 revision/local/test_83_occupation_route_rest.py
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
os.environ["CANARIES_83_PARTS"] = "ABCD"
# The script's Tee caps the terminal echo at 2 KB, which is right under
# BatchClient's blocking pipe and wrong here: it would swallow the check
# lines printed after main(). The lane runner lifts it the same way.
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries83_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
# 83's OUT is HERE / the environment variable, and pathlib lets an
# absolute value win, so the sandbox gets every export and the repository
# gets no output_83 folder from a test run. 82's OUT is pointed at the
# same place for the same reason.
os.environ["CANARIES_83_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
mc.connect = lambda: object()

FITS = []
_real_multi = mc.run_fepois_multi


def counting_multi(panel, workdir, tag, *a, **kw):
    """Every Poisson fit in this lane goes through one function, whichever
    script asked for it, so the count is of fits and not of call sites."""
    FITS.append(tag)
    return _real_multi(panel, workdir, tag, *a, **kw)


mc.run_fepois_multi = counting_multi


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s83 = load("83_occupation_route_rest.py", "s83")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================
# the world
# ======================================================================
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
INC = ["31-34", "35-40", "41-49", "50+"]
N_SCORE = 1000            # employers the score is built over
# Six hundred, not a token handful: script 73's credit arm refuses an
# estimate on fewer than 500 panel employers carrying the covariate, and
# that gate is left at its real value rather than lowered for the test,
# so the panel has to be big enough to pass it.
N_PANEL = 600             # of those, the ones that also carry a panel
SCORED = list(range(1, N_SCORE + 1))
PANEL = list(range(1, N_PANEL + 1))
NO_INDUSTRY = list(range(591, 601))       # ten panel firms with no code
NO_BALANCE = list(range(571, 591))        # twenty with no balance sheet
UNSCORED_CODE = "9999"

DAIOE = pd.read_stata(_LOCAL)
DAIOE["ssyk4"] = DAIOE["ssyk4"].astype(str).str.zfill(4)
SCORES = DAIOE.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                             "score"]]
assert UNSCORED_CODE not in set(DAIOE["ssyk4"])

# Four codes, one per tier, each in a three-digit group of its own among
# the codes this fixture plants, so the three-digit book value of a tier
# is that code's own score and the employment-weighted quartile is the
# planted tier rather than something to be read off the result.
_d = DAIOE.sort_values("pctl_rank_genai").reset_index(drop=True)
TIER_CODE, _seen = [], set()
for q in (0.10, 0.40, 0.65, 0.92):
    for i in range(int(q * len(_d)), len(_d)):
        c = _d.loc[i, "ssyk4"]
        if c[:3] not in _seen:
            TIER_CODE.append(c)
            _seen.add(c[:3])
            break
assert len(TIER_CODE) == 4, TIER_CODE
TIER_SCORE = [float(DAIOE.loc[DAIOE["ssyk4"] == c,
                              "pctl_rank_genai"].iloc[0]) for c in TIER_CODE]


# The four tiers are deliberately UNEQUAL, 20, 20, 20 and 40 per cent of
# the employers. The quartile cut points are weighted by incumbent
# employment, so with four equal tiers the cut at the twenty-fifth
# percentile lands exactly ON the lowest tier's own score and that whole
# tier is pushed above its own cut: the bottom quartile comes back empty
# and two tiers share the top one. Uneven tiers put each cut strictly
# between two planted scores, which makes the top tier exactly the top
# quartile and lets the test name the exposed employers in advance.
TIER_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3}


# A hundred employers whose score rests on ONE incumbent, and whose one
# occupation is the most or the least exposed code in the book. They are
# the mechanism Part D exists to measure: a mean over one worker is the
# noisiest score there is, its sampling error puts it further out than
# any well-measured firm's mean can go, and so small employers crowd
# BOTH tails of the score distribution. They clear the reported floor of
# five person-months comfortably and fail a floor of sixty.
TINY = list(range(501, 601))


def tier(emp: int) -> int:
    """0, 1, 2, 3; tier 3 is the one the score must put in the top
    quartile, and it is a property of the employer number, so the test
    can name the exposed firms before anything is estimated."""
    if emp in TINY:
        return 3 if emp % 2 else 0
    return TIER_OF[emp % 10]


def size_mult(emp: int) -> int:
    """How many workers stand behind the mix, so employers differ in size
    and a size tercile is not degenerate. It does not touch the ratio the
    mix is formed from, so two employers of the same tier and different
    sizes have the SAME score and differ only in how well it is
    measured."""
    return 1 + (emp % 9)


EXPOSED = {e for e in SCORED if tier(e) == 3}
EXPOSED_PANEL = sorted(e for e in PANEL if tier(e) == 3)


def cascade_frame() -> pd.DataFrame:
    """
    The 2019 occupation mix, as script 82's cascade pull returns it.

    Six workers on the employer's own tier code and one on a neighbouring
    tier's, so there is real variation WITHIN a firm and the reliability
    decomposition has something to decompose; the ratio is the same for
    every employer of a tier, so the tier's score is the same whatever
    the employer's size, and size and score are independent by
    construction. An uncoded cell and a cell the DAIOE file does not
    score, so the coded share and the scored share are different numbers
    and coverage is below one.

    A TINY employer holds ONE coded incumbent, in one band, on an extreme
    code, and nothing else.
    """
    rows = []
    for emp in SCORED:
        t = tier(emp)
        c = TIER_CODE[t]
        if emp in TINY:
            rows.append((emp, "41-49", c, c[:3], "2019", 1))
            continue
        # Six workers on the employer's own tier code and one on the
        # code at the OPPOSITE end of the distribution. The WITHIN-firm
        # spread is then much wider than the between-firm spread of the
        # firm means, which is the shape the real register has and the
        # shape that makes a one-worker mean unreliable; the odd worker
        # is placed opposite rather than next door for exactly that
        # reason. The tier still orders the means, and the ratio does not
        # depend on size, so two employers of one tier have the same
        # score and differ only in how well it is measured.
        far = TIER_CODE[3 - t]
        m = size_mult(emp)
        for age in AGES:
            rows.append((emp, age, c, c[:3], "2019", 6 * m))
            rows.append((emp, age, far, far[:3], "2019", 1 * m))
            rows.append((emp, age, UNSCORED_CODE, UNSCORED_CODE[:3],
                         "2019", 2))
            rows.append((emp, age, "____", "___", "none", 2))
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "ssyk3", "source_year", "n"])
    d["ssyk_ar"] = "2019"
    d["ssyk_status"] = np.where(d["ssyk4"] == "____", "9", "1")
    return d


def counts_2019() -> pd.DataFrame:
    """
    47L's monthly counts for the base year, which the floor sums.

    An ordinary employer carries its size multiplier in every band and
    every month, so it holds 48 times that in incumbent person-months:
    the smallest of them fails a floor of sixty and the rest clear it. A
    TINY employer carries one worker in one band, twelve person-months,
    which clears the reported floor of five and fails sixty.
    """
    rows = []
    for emp in SCORED:
        if emp in TINY:
            rows += [(emp, f"2019-{m:02d}", "41-49", 1)
                     for m in range(1, 13)]
            continue
        k = size_mult(emp)
        rows += [(emp, f"2019-{m:02d}", age, k)
                 for m in range(1, 13) for age in AGES]
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


CASC = cascade_frame()
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(
     mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
counts_2019().to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

# ---- the monthly panel, with every planted step in it -----------------
STEP_22 = float(np.log(0.75))    # the adoption fall at 22-25, exposed
STEP_RB = float(np.log(1.08))    # the tightening rise at 22-25, exposed
DIFF_F = float(np.log(0.88))     # young women's extra step at 22-25
# Post-period growth in the exposed firms, by band. Every band grows and
# the oldest grows fastest, which is the shape the education route
# reports and the descriptive arm has to recover; the youngest band's
# growth is set ABOVE the adoption fall, so the raw level rises while the
# within-firm contrast against the incumbents falls. Both things are true
# of the paper's own data, and a fixture in which the young band merely
# shrank would not test that the two are different statistics.
GROW = {"22-25": 0.50, "26-30": 0.55, "31-34": 0.60, "35-40": 0.65,
        "41-49": 0.70, "50+": 0.85}


def month_list(s61):
    return [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
            for m in range(1, 13 if y < 2025 else 7)]


# ======================================================================
# the modules, and the fixtures that need them
# ======================================================================
(s82, s61, s66, s67, s71, s73, s74, s75, s78, s80, l47, l70, j47,
 h47) = s83.load_modules()
for m_ in (s61, s66, s67, s71, s73, s74, s75, s78, s80, l47, l70, j47,
           h47, s82):
    m_.OUT = s83.OUT
    m_.CACHE = mc.CACHE_DIR
s83.OUT.mkdir(parents=True, exist_ok=True)

from _fixtures import Fixture  # noqa: E402

MONTHS = month_list(s61)
POST = s83.POST_FROM
RB_FROM, RB_TO = mc.RIKSBANK_YM, mc.CHATGPT_YM


def counts_by_sex() -> pd.DataFrame:
    """
    Monthly employment by employer, band, sex and month, with every
    planted step in it: the adoption fall at 22-25 in the exposed firms,
    an extra fall for young women, a rise during the tightening window,
    growth in every band after adoption that is largest at 50 and over,
    and NO pre-launch trend, so the drift arm should come back flat.
    """
    rng = np.random.default_rng(83)
    lam0 = {"22-25": 14, "26-30": 12, "31-34": 10, "35-40": 10,
            "41-49": 12, "50+": 14}
    rows = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age, lam in lam0.items():
                for g in ("1", "2"):
                    x = lam / 2
                    if hit:
                        if ym >= POST:
                            x *= np.exp(GROW[age])
                            if age == "22-25":
                                x *= np.exp(STEP_22)
                                if g == "2":
                                    x *= np.exp(DIFF_F)
                        if age == "22-25" and RB_FROM <= ym < RB_TO:
                            x *= np.exp(STEP_RB)
                    rows.append((emp, ym, age, g, int(rng.poisson(x)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "gender", "n_emp"])


SEX = counts_by_sex()
COUNTS = (SEX.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
          .sum().reset_index())
for y in s61.PANEL_YEARS:
    ys = str(y)
    COUNTS[COUNTS["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    SEX[SEX["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)

# ---- the industry key, as script 80 caches it ------------------------
KEY = pd.DataFrame([
    {"employer_id": e, "ind3": f"{100 + (e % 12)}",
     "source": "Ftg_2019" if e % 7 else "Ftg_2018"}
    for e in SCORED if e not in NO_INDUSTRY])
mc.write_cache(KEY, s80.KEY_CACHE)

# ---- the education route ---------------------------------------------
FIX = Fixture(mc, h47, n_firms=N_SCORE, n_exposed=250)
FIX.install_edu([2019], cache=mc.CACHE_DIR)
EDU = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
EDU_HI = set(EDU.loc[EDU["fq"] == 4, "employer_id"].astype(int))


# ======================================================================
# the survey tables and the balance sheet, behind one read_sql stub
# ======================================================================
SURVEY_SCHEMA = []
for t, cols in (("ITFtg_Stora_2023", ["PeOrgNr", "E_AI_TML", "E_AI_TNLG"]),
                ("ITFtg_Stora_2021", ["PeOrgNr", "E_AI_TML", "E_AI_TNLG"]),
                ("ai_itftg_2019", ["PeOrgNr", "AI_USE"]),
                ("BITA_2024", ["P1207_LopNr_PersonNr", "CH1", "CH2b",
                               "vikt_ind_SE"])):
    SURVEY_SCHEMA += [(t, c, "varchar") for c in cols]
SURVEY_SCHEMA = pd.DataFrame(SURVEY_SCHEMA, columns=["TABLE_NAME",
                                                     "COLUMN_NAME",
                                                     "DATA_TYPE"])
FIRM_SCHEMA = pd.DataFrame(
    [("FE_2019", c, "float") for c in ("PeOrgNrHE", "SummaTillgangar",
                                       "SummaEgetKapital")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])

# Reported AI use is planted on the OCCUPATION tier, with a smaller
# education-route component so that BOTH routes return a positive gap on
# the same table and the read rule's "at least half the size" has
# something to compare. A first stage computed on the wrong quartile
# returns the wrong number, which is what makes this a test.
OCC_EFFECT, EDU_EFFECT, AI_BASE = 0.45, 0.15, 0.15


def ai_flag(rng, emp: int) -> int:
    p = AI_BASE + OCC_EFFECT * (emp in EXPOSED) + EDU_EFFECT * (emp <= 250)
    return int(rng.random() < p)


def itftg_table(seed: int, twocol: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for emp in SCORED:
        a = ai_flag(rng, emp)
        if twocol:
            # the technology block: any-AI is the row-wise maximum, so a
            # firm ticking either column counts as a user
            rows.append((str(emp), str(a), str(a if rng.random() < 0.6
                                               else 0)))
        else:
            rows.append((str(emp), str(a)))
    return pd.DataFrame(rows, columns=(["PeOrgNr", "E_AI_TML", "E_AI_TNLG"]
                                       if twocol else ["PeOrgNr", "AI_USE"]))


def bita_table() -> pd.DataFrame:
    rng = np.random.default_rng(8324)
    rows = []
    for i in range(3000):
        emp = SCORED[i % len(SCORED)]
        rows.append((f"P{i}", str(ai_flag(rng, emp)),
                     str(int(rng.random() < 0.3)), 1.0))
    return pd.DataFrame(rows, columns=["P1207_LopNr_PersonNr", "CH1",
                                       "CH2b", "vikt_ind_SE"])


BITA = bita_table()
LINK = pd.DataFrame([{"person_id": f"P{i}",
                      "employer_id": str(SCORED[i % len(SCORED)])}
                     for i in range(3000)])


def fe_table() -> pd.DataFrame:
    rng = np.random.default_rng(8325)
    rows = []
    for emp in SCORED:
        if emp in NO_BALANCE:
            continue
        assets = 1000.0
        # leverage spread over the unit interval and INDEPENDENT of
        # exposure, so a median split splits and the credit term is not
        # the exposure term wearing another name
        rows.append((float(emp), assets,
                     assets * float(rng.uniform(0.1, 0.9))))
    return pd.DataFrame(rows, columns=["PeOrgNrHE", "SummaTillgangar",
                                       "SummaEgetKapital"])


FE = fe_table()
SQL_CALLS = []
_real_read_sql = pd.read_sql


def fake_read_sql(q, conn=None, *a, **kw):
    s = str(q)
    SQL_CALLS.append(s)
    if "INFORMATION_SCHEMA" in s:
        return (SURVEY_SCHEMA.copy() if "'ITFtg%'" in s
                else FIRM_SCHEMA.copy())
    if "Arb_AGIIndivid" in s:
        return LINK.copy()
    if "FE_2019" in s:
        return FE.rename(columns={"PeOrgNrHE": "employer_id",
                                  "SummaTillgangar": "assets",
                                  "SummaEgetKapital": "equity"}).copy()
    if "ITFtg_Stora_2023" in s:
        return itftg_table(101, True)
    if "ITFtg_Stora_2021" in s:
        return itftg_table(102, True)
    if "ai_itftg_2019" in s:
        return itftg_table(103, False)
    if "BITA_2024" in s:
        return BITA.copy()
    raise AssertionError(f"unexpected query: {s[:200]}")


pd.read_sql = fake_read_sql


# ======================================================================
# THE SCORE: lane 28's, and not rebuilt
# ======================================================================
_real_build = s82.build_exposure
BUILD_CALLS = []


def counting_build(*a, **kw):
    BUILD_CALLS.append(1)
    return _real_build(*a, **kw)


s82.build_exposure = counting_build
BUILT = s82.build_exposure(l47, l70, j47)
OCC = BUILT["exposure"]
Q4 = set(OCC.loc[OCC["fq"] == 4, "employer_id"].astype(int))

check("the score comes from lane 28's own builder and is not rebuilt "
      "here: 83 defines no exposure builder of its own",
      not any(n for n in dir(s83)
              if "exposure" in n.lower() and callable(getattr(s83, n))),
      ", ".join(n for n in dir(s83) if "exposure" in n.lower()) or "none")
DIRECT = s82.occ_route_exposure(
    s82.incumbent_frame(CASC, SCORES,
                        s82.build_book3(CASC, SCORES)[0], j47),
    s82.incumbent_floor_series(l47, CASC, j47),
    s82.FLOOR_MAIN, s82.ARM_YEARS[s82.MAIN_ARM])
check("and what the builder returns is bit for bit lane 28's primary "
      "arm, so the two lanes cannot drift apart",
      OCC.reset_index(drop=True).equals(DIRECT.reset_index(drop=True)),
      f"{len(OCC)} employers against {len(DIRECT)}")
check("the primary arm is the uniform three-digit one at a floor of "
      "five on the backward cascade, as lane 28 names it",
      BUILT["arm"] == "uniform3" and BUILT["floor"] == 5
      and s82.MAIN_ARM == "backward",
      f"{BUILT['arm']}, floor {BUILT['floor']}, {s82.MAIN_ARM}")
check("the planted top tier is exactly the top quartile, so the test "
      "names the exposed firms before anything is estimated",
      Q4 == EXPOSED, f"{len(Q4)} in Q4 against {len(EXPOSED)} planted")
check("the three-digit book scores the four planted tiers in the "
      "planted order",
      all(TIER_SCORE[i] < TIER_SCORE[i + 1] for i in range(3)),
      " < ".join(f"{s:.1f}" for s in TIER_SCORE))
check("coverage is below one, since every ordinary employer also holds "
      "an uncoded worker and one the DAIOE file cannot score",
      0.5 < float(OCC["coverage"].median()) < 1.0
      and float(OCC["coverage"].max()) <= 1.0,
      f"median coverage {float(OCC['coverage'].median()):.3f}")
check("the occupation route and the education route are different "
      "classifications, so a first stage on the wrong one is visible",
      len(Q4 & EDU_HI) < 0.8 * len(Q4),
      f"{len(Q4 & EDU_HI)} of {len(Q4)} occupation-exposed firms are also "
      f"education-exposed")
check("80's prior-export search path is emptied, so the four-decimal "
      "gate checks THIS lane's employer-clustered run and never the "
      "education route's exports",
      tuple(s80.PRIOR_DIRS) == () and s80.prior("cluster_industry.csv")
      is None)


# ======================================================================
# A. the first stage and the descriptive counterpart
# ======================================================================
s83.NOTES.clear(); s83.FAILURES.clear(); FITS.clear(); SQL_CALLS.clear()
FS, OV, WAVES, DROWS = s83.part_a(COUNTS, OCC, EDU, s66, s71, s73, l47, j47)

check("A: no Poisson fit runs in Part A, which is why it is first",
      not FITS, str(FITS))
check("A: the catalogue is probed before any survey table is read",
      SQL_CALLS and "INFORMATION_SCHEMA" in SQL_CALLS[0],
      f"{len(SQL_CALLS)} queries")
check("A: no part of Part A failed", not s83.FAILURES, str(s83.FAILURES))

O23 = s83.headline_gap(FS, "occupation", "itftg")
E23 = s83.headline_gap(FS, "education", "itftg")
O24 = s83.headline_gap(FS, "occupation", "bita")
E24 = s83.headline_gap(FS, "education", "bita")
check("A: the first stage is run for BOTH routes on the SAME table, so "
      "the comparison is within one regression sample and not between "
      "two runs",
      bool(O23 and E23 and O24 and E24)
      and O23["source"] == E23["source"] and O24["source"] == E24["source"],
      f"firm {O23.get('source')}, individual {O24.get('source')}")
check("A: reported AI use was planted on the OCCUPATION quartile, and "
      "the occupation route is the one that recovers it",
      O23["points"] > E23["points"] > 0
      and abs(O23["points"] - 100 * OCC_EFFECT) < 6,
      f"occupation {O23['points']:+.1f} points, education "
      f"{E23['points']:+.1f}, planted {100 * OCC_EFFECT:+.1f}")
check("A: and the same at the individual level, where the respondent is "
      "joined to the employer through the November declarations",
      O24["points"] > E24["points"] > 0,
      f"occupation {O24['points']:+.1f} points, education "
      f"{E24['points']:+.1f} on {cnt(O24['n']) if False else O24['n']:,} "
      f"respondents")
check("A: 71's OWN sample gates are the ones that ran, and were met "
      "rather than lowered for the test",
      s71.MIN_ITFTG_FIRMS == 800 and s71.MIN_ITFTG_HIGH == 100
      and s71.MIN_BITA_PERSONS == 600 and s71.MIN_BITA_HIGH == 80
      and O23["n"] >= 800 and O24["n"] >= 600,
      f"{O23['n']:,} firms, {O24['n']:,} respondents")

V1, L1 = s83.verdict_first_stage(FS)
check("A: read rule 1 is met on the planted world",
      V1 == "THE FIRST STAGE REPRODUCES", V1)
check("A: and the verdict names the education-route figure it was read "
      "against, and the recorded one beside it",
      any("recorded" in l for l in L1) and any("MET" in l for l in L1))


def planted(occ_points, edu_points, both=True):
    """Two routes, one firm table and one individual table, at the sizes
    a test wants to feed the rule."""
    out = []
    for route, pts in (("occupation", occ_points), ("education",
                                                    edu_points)):
        rows = [("ITFtg_Stora_2023", "2023", "ai_any"),
                ("BITA_2024", "2024", "genai")]
        for src, wave, outc in (rows if both else rows[:1]):
            out.append({"route": route, "outcome": outc, "source": src,
                        "wave": wave, "coef_points": pts, "se_points": 1.0,
                        "t": pts, "n": 1000})
    return out


check("rule 1 FAILS on a score that does not predict AI use at all",
      s83.verdict_first_stage(planted(0.4, 21.5))[0]
      == "THE FIRST STAGE DOES NOT REPRODUCE")
check("rule 1 FAILS on a score whose association has the OPPOSITE sign, "
      "however large",
      s83.verdict_first_stage(planted(-40.0, 21.5))[0]
      == "THE FIRST STAGE DOES NOT REPRODUCE")
check("rule 1 FAILS just below half the education route's size, and "
      "passes just above it, so the threshold is the stated one",
      s83.verdict_first_stage(planted(10.0, 21.5))[0]
      == "THE FIRST STAGE DOES NOT REPRODUCE"
      and s83.verdict_first_stage(planted(11.0, 21.5))[0]
      == "THE FIRST STAGE REPRODUCES")
_mixed = [r for r in planted(30.0, 21.5) if r["outcome"] == "ai_any"] \
    + [dict(r, coef_points=0.2, t=0.2) for r in planted(30.0, 21.5)
       if r["outcome"] == "genai" and r["route"] == "occupation"] \
    + [r for r in planted(30.0, 21.5)
       if r["outcome"] == "genai" and r["route"] == "education"]
check("rule 1 says PARTLY when one of the two waves holds and the other "
      "does not, and names which",
      s83.verdict_first_stage(_mixed)[0]
      == "THE FIRST STAGE PARTLY REPRODUCES"
      and any("genai" in l or "individual" in l
              for l in s83.verdict_first_stage(_mixed)[1]))
check("rule 1 gives NO VERDICT, and explicitly not a pass, when 71 made "
      "no estimate at all",
      s83.verdict_first_stage([])[0] == "NO VERDICT"
      and any("not a pass" in l for l in s83.verdict_first_stage([])[1]))
check("a wave the education route itself could not estimate is NO "
      "VERDICT rather than a pass on a zero benchmark",
      s83.verdict_first_stage(planted(30.0, 0.0))[0] == "NO VERDICT")

check("A: the pre-ChatGPT any-AI gap is reported for every firm wave "
      "the delivery holds, on both routes",
      {w["wave"] for w in WAVES if w["route"] == "occupation"}
      == {"2019", "2021", "2023"},
      str(sorted({w["wave"] for w in WAVES if w["route"] == "occupation"})))
_gaps = {w["wave"]: w["points"] for w in WAVES
         if w["route"] == "occupation"}
check("A: and it comes back flat across the three waves, as it was "
      "planted, which is the diagnostic the paper reports",
      max(_gaps.values()) - min(_gaps.values()) < 10,
      " ".join(f"{k} {v:+.1f}" for k, v in sorted(_gaps.items())))
check("A: the diagnostic is stated to settle nothing, and carries the "
      "education route's recorded figures",
      any("settles nothing" in l for l in s83.verdict_waves(WAVES))
      and any("20.3" in l for l in s83.verdict_waves(WAVES)))

check("A: the descriptive counterpart is built for both routes",
      {r["route"] for r in DROWS} == {"occupation", "education"})
_occ4 = {r["age_group"]: r["log_change"] for r in DROWS
         if r["route"] == "occupation" and r["fq"] == 4}
check("A: the planted shape comes back: the exposed employers grew in "
      "every band and grew the oldest band fastest",
      all(v > 0 for v in _occ4.values())
      and max(_occ4, key=_occ4.get) == "50+",
      " ".join(f"{k} {v:+.3f}" for k, v in sorted(_occ4.items())))
check("A: and the sentence the summary prints says exactly that",
      "EVERY band" in s83.describe_shape(DROWS, "occupation")
      and "50+ fastest" in s83.describe_shape(DROWS, "occupation"),
      s83.describe_shape(DROWS, "occupation")[:110])
_bad = [dict(r, log_change=(-0.2 if r["age_group"] == "26-30"
                            else r["log_change"]))
        for r in DROWS]
check("and it says so differently when a band does NOT grow, so the "
      "sentence is read off the numbers and not asserted",
      "EVERY" not in s83.describe_shape(_bad, "occupation"),
      s83.describe_shape(_bad, "occupation")[:110])
check("A: the descriptive carries no verdict, and says why",
      any("no verdict" in l.lower() for l in s83.verdict_descriptive(DROWS))
      and any("raw means" in l.lower()
              for l in s83.verdict_descriptive(DROWS)))

D = pd.read_csv(s83.OUT / "occ_rest_descriptive.csv")
check("A: the descriptive export carries the mean in both windows, the "
      "change and the education route's recorded shape",
      {"route", "fq", "age_group", "mean_pre", "mean_post", "log_change",
       "n_firms", "edu_route_shape"} <= set(D.columns), str(list(D.columns)))
check("A: nothing between one and four leaves MONA in the descriptive",
      not ((D["n_firms"] > 0) & (D["n_firms"] < s83.FLOOR)).any())
F = pd.read_csv(s83.OUT / "occ_rest_firststage.csv")
check("A: the first-stage export carries the route, the source, the gap "
      "in points and the recorded education-route figure",
      {"route", "source", "outcome", "coef_points", "se_points", "t", "n",
       "edu_recorded_points"} <= set(F.columns), str(list(F.columns)))
check("A: and nothing between one and four leaves MONA in it",
      not ((F["n"] > 0) & (F["n"] < s83.FLOOR)).any())


# ======================================================================
# the term builders, checked directly, before anything is fitted
# ======================================================================
_tiny = pd.DataFrame({
    "year_month": ["2021-06", "2022-03", "2022-04", "2022-11", "2022-12",
                   "2023-06", "2024-01", "2025-01"],
    "high": 1, "young": 1, "n_emp": 1})
_w, _wt = s75.add_window_terms(_tiny.copy())
check("B: the tightening term of the reference window is a WINDOW, one "
      "in April to November 2022 and zero everywhere else, so the "
      "adoption term reads against the months before the rate rise",
      list(_w["rbw_x_high_x_young"]) == [0, 0, 1, 1, 0, 0, 0, 0],
      str(list(_w["rbw_x_high_x_young"])))
check("B: and the adoption term opens in January 2024, as it does in 68, "
      "75, 78 and 80",
      list(_w["post_x_high_x_young"]) == [0, 0, 0, 0, 0, 0, 1, 1]
      and s83.POST_FROM == "2024-01")
check("B: the drift is estimated on January 2021 to November 2022, the "
      "months before the launch",
      s78.PRE_TREND_FROM == "2021-01" and s78.PRE_LAUNCH_END == "2022-12"
      and mc.CHATGPT_YM == "2022-12",
      f"{s78.PRE_TREND_FROM} to before {s78.PRE_LAUNCH_END}")
_d2 = pd.DataFrame({"year_month": ["2021-01", "2021-02", "2022-01"],
                    "high": 1, "young": 1, "n_emp": 1})
_d2, _dt = s78.drift_terms(_d2)
check("B: and the trend is in MONTHS since January 2021, so its "
      "coefficient is a monthly drift",
      list(_d2["trend_x_high_x_young"]) == [0.0, 1.0, 12.0]
      and "trend_x_high_x_young" in _dt)
check("C: 73's OWN coverage gate is the one the credit test runs under",
      s73.MIN_FIRMS == 500 and s73.MIN_MATCH_RATE == 0.30,
      f"{s73.MIN_FIRMS} firms, {s73.MIN_MATCH_RATE:.0%}")
_gb = pd.DataFrame({"employer_id": [str(i) for i in range(1, 2001)]})
s73.NOTES.clear()
check("and it REFUSES rather than reporting a thin estimate when too "
      "few of the panel's employers carry a balance sheet",
      s73.gate(_gb, pd.DataFrame({"employer_id": [str(i) for i in
                                                  range(1, 100)]}),
               "leverage/test") is False
      and any("BELOW THRESHOLD" in n for n in s73.NOTES))
check("and it passes when enough of them do, so the gate is a threshold "
      "and not a refusal",
      s73.gate(_gb, pd.DataFrame({"employer_id": [str(i) for i in
                                                  range(1, 1800)]}),
               "leverage/test") is True)
s73.NOTES.clear()


# ======================================================================
# main(), end to end
# ======================================================================
mc.runlog = lambda *a, **kw: None      # the UNC root is not here anyway
FITS.clear(); SQL_CALLS.clear()
s83.NOTES.clear(); s83.FAILURES.clear()
s83.main()

check("main() ran thirty fits: two window, two drift and six clustered "
      "in B, four industry and six credit in C, and four floor and six "
      "tercile in D",
      len(FITS) == 30, f"{len(FITS)}: {sorted(FITS)}")
check("main() got its score from lane 28's builder, which is the only "
      "thing that prints that line",
      (s83.OUT / "83_log.txt").read_text(encoding="utf-8",
                                         errors="replace").count(
          "occupation route:") >= 1)
check("no part failed in the end-to-end run", not s83.FAILURES,
      str(s83.FAILURES))
check("and the first stage settled, so the rest may be read",
      s83.FIRST_STAGE == "THE FIRST STAGE REPRODUCES", s83.FIRST_STAGE)

W = pd.read_csv(s83.OUT / "occ_rest_window.csv")
check("B: the reference window is exported for both bands with every "
      "term of the specification",
      set(W["young_band"]) == {"22-25", "26-30"}
      and {"rbw_x_high_x_young", "interim_x_high_x_young",
           "post_x_high_x_young"} <= set(W["term"]))
_wp = W[(W["term"] == "post_x_high_x_young")].set_index("young_band")
check("B: the planted fall comes back as a level after adoption at both "
      "bands",
      (_wp["coef"] < 0).all(),
      " ".join(f"{b} {v:+.4f}" for b, v in _wp["coef"].items()))
check("B: and the education route's own figure travels on the same row",
      abs(float(_wp.loc["22-25", "edu_coef"]) + 0.0194) < 1e-9
      and abs(float(_wp.loc["26-30", "edu_coef"]) + 0.0172) < 1e-9)
# The tightening rise was planted at 22-25 alone, so that is where the
# window term has to find it; 26-30 is printed beside it and is expected
# to be near zero.
_rb = W[W["term"] == "rbw_x_high_x_young"].set_index("young_band")
check("B: the planted tightening rise comes back at the band it was "
      "planted in, so the window term is measuring the months it says "
      "it is",
      float(_rb.loc["22-25", "coef"]) > 0
      and abs(float(_rb.loc["26-30", "coef"])) < float(_rb.loc["22-25",
                                                               "coef"]),
      " ".join(f"{b} {v:+.4f}" for b, v in _rb["coef"].items()))

DR = pd.read_csv(s83.OUT / "occ_rest_drift.csv")
_dt2 = DR[DR["term"] == "trend_x_high_x_young"].set_index("young_band")
check("B: the drift is exported for both bands, with the window it was "
      "estimated on beside it",
      set(_dt2.index) == {"22-25", "26-30"}
      and set(_dt2["first_month"]) == {"2021-01"}
      and set(_dt2["last_month"]) == {"2022-11"},
      f"{sorted(set(_dt2['first_month']))} to "
      f"{sorted(set(_dt2['last_month']))}")
check("B: the planted world has no pre-launch trend and the drift comes "
      "back FLAT at both bands",
      bool(_dt2["flat_within_2se"].all()),
      " ".join(f"{b} {c:+.4f} ({s:.4f})" for b, c, s
               in zip(_dt2.index, _dt2["coef"], _dt2["se"])))
V5, L5 = s83.verdict_drift(DR.to_dict("records"))
check("B: read rule 5 reads that as flat at both bands", V5
      == "FLAT AT BOTH BANDS", V5)
_drift = [dict(r, coef=0.02, se=0.001) if r["term"]
          == "trend_x_high_x_young" else r for r in DR.to_dict("records")]
check("and it reads a planted drift as NOT flat, so the rule can fail",
      s83.verdict_drift(_drift)[0] == "NOT FLAT AT EITHER BAND")
check("and it reports the education route's own drift beside ours, "
      "including that the education route is itself not flat at 26-30",
      any("education route" in l and "26-30" in l and "NOT FLAT" in l
          for l in L5),
      next((l.strip() for l in L5
            if "education route" in l and "26-30" in l), "")[:100])

CL = pd.read_csv(s83.OUT / "occ_rest_cluster.csv")
check("B: the clustering is exported for both pooled bands and for the "
      "sex specification",
      set(CL["spec"]) == {"pooled", "gender"}
      and set(CL[CL["spec"] == "pooled"]["young_band"]) == {"22-25",
                                                            "26-30"})
check("B: THE GATE. Every coefficient reproduces this lane's own "
      "employer-clustered run to four decimals",
      bool(CL["coef_match_4dp"].dropna().all())
      and CL["coef_match_4dp"].notna().all(),
      f"{int(CL['coef_match_4dp'].sum())} of {len(CL)} terms")
check("B: and the check is against a run REFITTED here, not against an "
      "education-route export, which is what emptying 80's search path "
      "buys",
      any("refitted in this run" in n or "PRIOR" in n or "prior" in n
          for n in s83.NOTES),
      next((n for n in s83.NOTES if "prior" in n), "")[:90])
check("B: only the covariance moves: the industry standard error is a "
      "different number from the employer one",
      (CL["se_industry_complete"] != CL["se_employer"]).all())
check("B: the employers with no industry code share ONE residual "
      "cluster rather than getting one each",
      set(CL["n_unresolved"]) == {len(NO_INDUSTRY)},
      f"{sorted(set(CL['n_unresolved']))} unresolved of "
      f"{sorted(set(CL['n_firms']))} on the panel")
check("B: the education route's industry standard errors travel beside "
      "ours",
      abs(float(CL[(CL["spec"] == "pooled")
                   & (CL["young_band"] == "22-25")
                   & (CL["term"] == "post_x_high_x_young")]
                ["edu_se_industry"].iloc[0]) - 0.0302) < 1e-9)
check("B: 80's own file name does not survive beside this lane's",
      not (s83.OUT / "cluster_industry_v2.csv").exists())
_moved = CL.to_dict("records")
_moved[0] = dict(_moved[0], coef_match_4dp=False)
check("the clustering verdict FAILS when a coefficient has moved, and "
      "says nothing from the arm is quoted",
      s83.verdict_cluster(_moved, {})[0] == "THE GATE FAILS"
      and any("NOTHING FROM THE CLUSTERING ARM IS QUOTED" in l
              for l in s83.verdict_cluster(_moved, {})[1]))

IN = pd.read_csv(s83.OUT / "occ_rest_industry.csv")
check("C: the industry test is exported at both bands, each with a "
      "baseline on the same firms",
      set(IN["spec"]) == {"baseline_same_sample", "industry_age_month"}
      and set(IN["young_band"]) == {"22-25", "26-30"})
check("C: the two fits run on IDENTICAL firm sets, so the difference "
      "between them is the specification and not the sample",
      IN.groupby("young_band")["n_firms"].nunique().eq(1).all(),
      str(sorted(set(IN["n_firms"]))))
check("C: an employer with no industry code is in NEITHER fit, rather "
      "than in a residual group of its own",
      set(IN["n_firms"]) == {N_PANEL - len(NO_INDUSTRY)},
      f"{sorted(set(IN['n_firms']))} of {N_PANEL} panel employers")
check("C: the retained share is exported with the education route's "
      "beside it and with the share coded from a later source, which "
      "flatters the test",
      {"retained_share", "edu_retained_share",
       "share_not_from_2019"} <= set(IN.columns)
      and abs(float(IN[IN["young_band"] == "22-25"]
                    ["edu_retained_share"].iloc[0]) - 0.85) < 1e-9
      and (IN["share_not_from_2019"] > 0).any())
check("C: 80's own file name does not survive beside this lane's",
      not (s83.OUT / "industry_seasonal_v2.csv").exists())
V7, L7 = s83.verdict_industry(IN.to_dict("records"))
check("C: read rule 7 is met on the planted world, where industry is "
      "independent of exposure and so absorbs nothing",
      V7 == "THE STEP SURVIVES AT BOTH BANDS", V7)
_ind = [dict(r, coef=(r["coef"] * 0.2 if r["spec"] == "industry_age_month"
                      else r["coef"]),
             retained_share=(0.2 if r["spec"] == "industry_age_month"
                             else r["retained_share"]))
        for r in IN.to_dict("records")]
check("and rule 7 FAILS when industry absorbs most of the step, so the "
      "rule can fail",
      s83.verdict_industry(_ind)[0] == "THE STEP SURVIVES AT NEITHER BAND")
check("and it says in the summary that the education route itself would "
      "not pass at 26-30, rather than leaving that to be discovered",
      any("would not pass" in l for l in L7))

CR = pd.read_csv(s83.OUT / "occ_rest_credit.csv")
check("C: the credit test exports a baseline RE-ESTIMATED on the "
      "balance-sheet sample, not the full-panel one",
      {"baseline_on_balance_sheet_sample", "post_x_high_x_young",
       "post_x_high_x_young_x_lev", "full_panel_baseline"}
      <= set(CR["term"]))
_bs = CR[CR["term"] == "baseline_on_balance_sheet_sample"].set_index("band")
_fp = CR[CR["term"] == "full_panel_baseline"].set_index("band")
check("C: and the two are different numbers, which is what makes the "
      "comparison a specification change rather than a sample change",
      all(abs(float(_bs.loc[b, "coef"]) - float(_fp.loc[b, "coef"])) > 0
          for b in ("22-25", "26-30")),
      " ".join(f"{b} {float(_bs.loc[b, 'coef']):+.4f} against "
               f"{float(_fp.loc[b, 'coef']):+.4f}" for b in ("22-25",
                                                             "26-30")))
check("C: the leverage split is reported, with the cut and the share it "
      "put on each side",
      any("split at" in n for n in s83.NOTES),
      next((n for n in s83.NOTES if "split at" in n), "")[:80])
check("C: the balance sheet reaches the employers that have one and no "
      "others",
      any(f"{N_PANEL - len(NO_BALANCE)} of {N_PANEL}" in n
          for n in s83.NOTES),
      next((n for n in s83.NOTES if "carry the covariate" in n), "")[:90])
check("C: the education route's own figures travel on the row",
      abs(float(CR[(CR["term"] == "post_x_high_x_young")
                   & (CR["band"] == "22-25")]
                ["edu_share_of_baseline"].iloc[0]) - 1.02) < 1e-9)
V8, L8 = s83.verdict_credit(CR.to_dict("records"))
check("C: read rule 8 is met on the planted world, where leverage is "
      "independent of exposure",
      V8 == "THE STEP IS NOT A CREDIT EFFECT", V8)
_cr = [dict(r, coef=(r["coef"] * 0.3 if r.get("term")
                     == "post_x_high_x_young" and r.get("spec") == "credit"
                     else r["coef"])) for r in CR.to_dict("records")]
check("and rule 8 FAILS when the step collapses once leverage is in, so "
      "the rule can fail",
      s83.verdict_credit(_cr)[0]
      == "THE STEP DOES NOT SURVIVE THE CREDIT TEST AT BOTH BANDS")


# ======================================================================
# D. the firm-size robustness
# ======================================================================
REL = pd.read_csv(s83.OUT / "occ_rest_reliability.csv")
_v = REL[REL["block"] == "variance"].set_index("item")["value"]
check("D: the worker-level score is decomposed into a between-firm and "
      "a within-firm variance, and the fixture has real variation inside "
      "a firm for it to find",
      float(_v["variance_within_firm"]) > 0
      and float(_v["variance_between_firms_observed"]) > 0,
      f"within {float(_v['variance_within_firm']):.2f}, between "
      f"{float(_v['variance_between_firms_observed']):.2f}")
check("D: the between component is netted of the sampling noise it "
      "contains, so the reliability is not overstated",
      0 <= float(_v["variance_between_firms_net"])
      <= float(_v["variance_between_firms_observed"])
      and float(_v["sampling_noise_removed"]) > 0,
      f"{float(_v['variance_between_firms_observed']):.2f} - "
      f"{float(_v['sampling_noise_removed']):.2f} = "
      f"{float(_v['variance_between_firms_net']):.2f}")
_at = REL[REL["block"] == "reliability_at_n"].set_index("item")
check("D: the reliability RISES with the number of incumbents behind "
      "the score, which is the whole point of the table",
      float(_at.loc["n_1", "value"]) < float(_at.loc["n_10", "value"])
      < float(_at.loc["n_100", "value"]) <= 1.0,
      " ".join(f"{i} {float(_at.loc[i, 'value']):.2f}"
               for i in ("n_1", "n_10", "n_100")))
check("D: and the hundred one-incumbent employers are counted as such, "
      "with the employment they hold",
      int(_at.loc["n_1", "n_firms"]) == len(TINY),
      f"{int(_at.loc['n_1', 'n_firms'])} employers on one incumbent")
_thin = REL[REL["block"] == "thin"].iloc[0]
check("D: the share of employers whose score falls below a reliability "
      "of one half is reported, with the share of incumbent employment "
      "sitting there",
      float(_thin["share_firms"]) > 0
      and 0 <= float(_thin["share_employment"]) < 1,
      f"{float(_thin['share_firms']):.1%} of employers, "
      f"{float(_thin['share_employment']):.2%} of employment")
check("D: nothing between one and four leaves MONA in the reliability "
      "table",
      not ((REL["n_firms"] > 0) & (REL["n_firms"] < s83.FLOOR)).any())

# the planted mechanism: a score resting on one worker lands further out
# than any well-measured firm's score can
_tm = OCC[OCC["employer_id"].isin(TINY)]["mix"]
_nm = OCC[~OCC["employer_id"].isin(TINY)]["mix"]
check("D: the one-incumbent employers sit at BOTH extremes of the score "
      "distribution, outside every well-measured employer's mean, which "
      "is the composition distortion Part D exists to measure",
      float(_tm.min()) < float(_nm.min())
      and float(_tm.max()) > float(_nm.max()),
      f"one-incumbent {float(_tm.min()):.1f} to {float(_tm.max()):.1f}, "
      f"the rest {float(_nm.min()):.1f} to {float(_nm.max()):.1f}")

SZ = pd.read_csv(s83.OUT / "occ_rest_size.csv")
check("D: the size arms are exported at both bands: the reported floor, "
      "a floor of sixty person-months and three size terciles",
      set(SZ["spec"]) == {"floor_5", "floor_60", "tercile_1", "tercile_2",
                          "tercile_3"}
      and SZ.groupby("spec")["young_band"].nunique().eq(2).all(),
      str(sorted(set(SZ["spec"]))))
_b5 = set(OCC.loc[OCC["n"] >= 60, "employer_id"].astype(int))
check("D: a floor of sixty person-months drops every one-incumbent "
      "employer and every employer of one worker a band",
      not (_b5 & set(TINY))
      and not (_b5 & {e for e in SCORED
                      if e not in TINY and size_mult(e) == 1}),
      f"{len(_b5)} of {len(OCC)} employers clear it")
_q4 = set(OCC.loc[OCC["fq"] == 4, "employer_id"].astype(int))
_re = s82.occ_route_exposure(s82.incumbent_frame(
        CASC, SCORES, s82.build_book3(CASC, SCORES)[0], j47),
    s82.incumbent_floor_series(l47, CASC, j47), 60,
    s82.ARM_YEARS[s82.MAIN_ARM])
check("D: THE QUARTILE IS NOT RECUT. Every employer that clears the "
      "higher floor keeps exactly the quartile the national cut gave "
      "it, which is what makes the arm a change of sample and not a "
      "change of treatment",
      set(OCC[OCC["n"] >= 60].loc[lambda d: d["fq"] == 4, "employer_id"]
          .astype(int)) == (_q4 & _b5))


def mini(floor: int):
    """
    A purpose-built world in which dropping the small employers DOES move
    the weighted cut points, so the difference between recutting and
    restricting is a real one and the check above is not vacuous.

    Six hundred small employers hold a hundred person-months each and all
    sit at the top of the score distribution; two hundred large ones hold
    three hundred each and are spread evenly over it. The two groups hold
    the same total weight, so removing the small ones takes the top of
    the distribution away and every cut point has to move down.
    """
    rows, fl = [], {}
    for i in range(800):
        emp = 90000 + i
        small = i < 600
        t = 3 if small else (i % 4)
        code = TIER_CODE[t]
        rows.append((emp, "41-49", code, code[:3], "2019", 10))
        fl[emp] = 100 if small else 300
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "ssyk3", "source_year", "n"])
    inc_ = s82.incumbent_frame(d, SCORES, s82.build_book3(CASC, SCORES)[0],
                               j47)
    nf = pd.Series(fl).rename_axis("employer_id").rename("n_floor")
    return s82.occ_route_exposure(inc_, nf, floor,
                                  s82.ARM_YEARS[s82.MAIN_ARM])


_full, _cut = mini(1), mini(200)
_j = _full[["employer_id", "fq"]].merge(_cut[["employer_id", "fq"]],
                                        on="employer_id",
                                        suffixes=("_nat", "_recut"))
check("D: and that matters, because recutting on the survivors DOES "
      "move employers between quartiles when the dropped employers hold "
      "weight: the fixed cut is a choice, not a coincidence of the data",
      len(_j) and float((_j["fq_nat"] != _j["fq_recut"]).mean()) > 0,
      f"{float((_j['fq_nat'] != _j['fq_recut']).mean()):.0%} of the "
      f"{len(_j)} survivors would move under a recut")

_terc = SZ[SZ["spec"].str.startswith("tercile")]
check("D: the terciles partition the panel's employers, so no employer "
      "is counted twice and none is left out",
      int(_terc[(_terc["young_band"] == "22-25")
                & (_terc["term"] == "post_x_high_x_young")]
          ["n_firms"].sum())
      == int(SZ[(SZ["spec"] == "floor_5")
                & (SZ["young_band"] == "22-25")
                & (SZ["term"] == "post_x_high_x_young")]
             ["n_firms"].iloc[0]),
      str(sorted(_terc["n_firms"].unique())))
check("D: and so do the exposed employers within them, which is what "
      "shows the NATIONAL quartile was carried in rather than recut",
      int(_terc[(_terc["young_band"] == "22-25")
                & (_terc["term"] == "post_x_high_x_young")]
          ["n_exposed_firms"].sum())
      == int(SZ[(SZ["spec"] == "floor_5")
                & (SZ["young_band"] == "22-25")
                & (SZ["term"] == "post_x_high_x_young")]
             ["n_exposed_firms"].iloc[0]))
_t1 = SZ[(SZ["spec"] == "tercile_1") & (SZ["young_band"] == "22-25")
         & (SZ["term"] == "post_x_high_x_young")].iloc[0]
_t3 = SZ[(SZ["spec"] == "tercile_3") & (SZ["young_band"] == "22-25")
         & (SZ["term"] == "post_x_high_x_young")].iloc[0]
check("D: the smallest tercile holds a HIGHER share of exposed "
      "employers than the largest, because its noisy scores crowd the "
      "tails, which is the distortion the part reports",
      (_t1["n_exposed_firms"] / _t1["n_firms"])
      > (_t3["n_exposed_firms"] / _t3["n_firms"]),
      f"{_t1['n_exposed_firms'] / _t1['n_firms']:.1%} against "
      f"{_t3['n_exposed_firms'] / _t3['n_firms']:.1%}")

V9, L9 = s83.verdict_size(SZ.to_dict("records"), {})
check("D: read rule 9 passes on the planted world, where the step is "
      "the same in every employer whatever its size",
      V9 == "THE SIZE ROBUSTNESS PASSES", V9)
_far = [dict(r, coef=(r["coef"] * 0.2 if r["spec"] == "floor_60"
                      else r["coef"])) for r in SZ.to_dict("records")]
check("and rule 9 FAILS when the higher floor moves the step by more "
      "than one reported standard error",
      s83.verdict_size(_far, {})[0] == "THE SIZE ROBUSTNESS DOES NOT PASS")
_carry = [dict(r, coef=(r["coef"] * 0.01
                        if r["spec"] in ("tercile_2", "tercile_3")
                        else r["coef"])) for r in SZ.to_dict("records")]
V9c, L9c = s83.verdict_size(_carry, {})
check("and rule 9 FAILS, and says the concern is NOT ANSWERED, when the "
      "SMALLEST tercile carries the whole step",
      V9c == "THE SIZE ROBUSTNESS DOES NOT PASS"
      and any("SMALLEST TERCILE" in l for l in L9c)
      and any("NOT ANSWERED" in l for l in L9c))
check("and it gives NO VERDICT rather than a pass when a tercile did "
      "not come back",
      s83.verdict_size([r for r in SZ.to_dict("records")
                        if r["spec"] != "tercile_2"], {})[0]
      == "NO VERDICT")


# ======================================================================
# what leaves MONA, and what the summary says
# ======================================================================
for nm, col in (("occ_rest_firststage.csv", "n"),
                ("occ_rest_descriptive.csv", "n_firms"),
                ("occ_rest_window.csv", "n_firms"),
                ("occ_rest_drift.csv", "n_firms"),
                ("occ_rest_cluster.csv", "n_firms"),
                ("occ_rest_industry.csv", "n_firms"),
                ("occ_rest_size.csv", "n_firms"),
                ("occ_rest_reliability.csv", "n_firms")):
    d = pd.read_csv(s83.OUT / nm)
    v = d[col]
    check(f"nothing between one and four leaves MONA in {nm}",
          not ((v > 0) & (v < s83.FLOOR)).any(), f"{len(d)} rows")
check("every export the upload list names is on disk, with the "
      "covariance of every clustered fit beside it",
      all((s83.OUT / f).exists() for f in
          ("occ_rest_firststage.csv", "occ_rest_descriptive.csv",
           "occ_rest_window.csv", "occ_rest_drift.csv",
           "occ_rest_cluster.csv", "occ_rest_industry.csv",
           "occ_rest_credit.csv", "occ_rest_size.csv",
           "occ_rest_reliability.csv", "83_summary.txt"))
      and len(list(s83.OUT.glob("vcov_*.csv"))) >= 26,
      f"{len(list(s83.OUT.glob('vcov_*.csv')))} covariance files")

SUMM = (s83.OUT / "83_summary.txt").read_text(encoding="utf-8")
for must in ("READ RULES, FIXED BEFORE THE RUN",
             "There is NO coefficient gate anywhere in this lane EXCEPT",
             "1. THE FIRST STAGE, and it decides whether anything in "
             "this lane",
             "NOTHING IN LANE",
             "THE FIRST STAGE REPRODUCES",
             "THE SCORE. Every fit here uses lane 28's firm score",
             "build_exposure()",
             "+21.5", "+23.5", "20.3", "19.8", "21.5",
             "-0.0194 (0.0184)", "-0.0172 (0.0121)",
             "+0.0006 (0.0008)", "+0.0019 (0.0004)",
             "0.0302", "0.0200", "0.0353", "0.0179", "0.0285",
             "85 per cent at 22-25", "47 per cent at 26-30",
             "102 and 105 per cent of -0.0695",
             "exposed employers grew in every band",
             "a diagnostic", "settles nothing",
             "2. THE PRE-CHATGPT ANY-AI GAP",
             "3. THE DESCRIPTIVE COUNTERPART",
             "4. THE REFERENCE WINDOW",
             "5. THE PRE-LAUNCH DRIFT",
             "6. THE INDUSTRY CLUSTERING",
             "7. THE INDUSTRY TEST",
             "8. THE CREDIT TEST",
             "9. THE FIRM-SIZE ROBUSTNESS",
             "D. THE FIRM-SIZE ROBUSTNESS:",
             "271,047", "262,089",
             "NEVER used to correct an estimate",
             "never recut inside a subsample",
             "reproduce the employer-clustered run to 4 decimals",
             "THE VERDICTS:",
             "WHAT TO EXPECT, SO IT IS NOT READ AS A BUG"):
    check(f"the summary states {must!r}", must in SUMM)
check("the summary names no failure, since none happened",
      "WHAT FAILED" not in SUMM)
check("and it opens with the first stage, before any estimate",
      SUMM.index("THE FIRST STAGE REPRODUCES")
      < SUMM.index("THE REFERENCE WINDOW"))

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
pd.read_sql = _real_read_sql
sys.exit(1 if FAILS else 0)

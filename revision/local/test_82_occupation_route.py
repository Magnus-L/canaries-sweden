#!/usr/bin/env python3
"""
test_82_occupation_route.py -- the score must qualify a firm on its
                               INCUMBENTS, complete the code backwards
                               and never forwards, and the three read
                               rules must be able to fail.

The checks are on mechanisms, not on outputs. One synthetic world on 150
employers, with a 2019 cascade frame built so that every rule the score
claims to follow has something to catch, and every group planted six
strong or more unless it is planted to be suppressed:

  1-50      their INCUMBENTS hold high-DAIOE occupations and their YOUNG
            hold low-DAIOE ones. Every other firm is the mirror image. A
            score that let the young in would rank the firms backwards,
            so "the young never enter" is checked against a prediction.
  101-106   three coded incumbents in November and sixty incumbent
            person-months. Script 65's rule, which counts CODED November
            persons against a floor of five, loses them; the floor on the
            firm's incumbents keeps them. They must come back as
            RECOVERED BY THE FLOOR.
  107-110   coded in 2021 and in no earlier year. The reported score is
            backward only, so they must NOT be scored on it and must be
            scored on the forward arm. Four firms, so the row is also
            below the export floor and must leave MONA suppressed.
  111-118   coded in 2018 and in no later year. The cascade must recover
            them, and they must come back as RECOVERED BY THE CASCADE
            with every coded incumbent from a year before 2019.
  121-126   incumbents, sixty person-months, and no code in any year.
            LOST TO MISSING CODES, whatever the floor does.
  131-135   one incumbent person-month, coded. LOST TO THE FLOOR.
  136-140   one incumbent person-month, no code. LOST TO BOTH.
  148-150   absent from the education frame, so three employers are
            scorable on the occupation route alone: below the floor, and
            suppressed, which is what keeps the suppression check from
            being vacuous.
  every firm carries an uncoded '____' cell and a cell whose code is not
            in the DAIOE file, so coverage has something to cover and the
            coded and scored shares are not the same number.

The employment is drawn with the planted steps on the employers the
OCCUPATION route puts in its top quartile, which is computed first and
not assumed: a Q4 seasonal bump and a tightening rise at 22-25, an
adoption step at 22-25 with an extra step for young women, a gain at 50
and over, and more separations at 22-25 with hiring left flat. The three
read rules should then all be met, and each verdict function is also fed
a planted null, so the test establishes that they can fail and not only
that they can pass.

Also tested: that the floor is applied to incumbent person-months, the
education route's own unit, when 47L's 2019 counts are on the share and
to the November head count when they are not; that the quartile cuts are
weighted by incumbent employment rather than by employers; that the
cascade query cleans each vintage BEFORE the COALESCE, takes the birth
year from the 2019 register and never joins a year the catalogue does not
hold; that the head-count audit catches a frame that disagrees with 47L's
own; that the verdicts read the reported arm and ignore the forward one;
that counts of one to four never leave MONA; and main() end to end.

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
s82.CASC_CACHE = mc.CACHE_DIR / "L_baseline_2019_cascade.parquet"
s82.VINT_CACHE = mc.CACHE_DIR / "L_baseline_2019_asof2021.parquet"
s82.BASE_CACHE = mc.CACHE_DIR / "L_baseline_2019.parquet"
s82.COUNTS_2019_CACHE = mc.CACHE_DIR / "L_counts_2019.parquet"
s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
h47 = j47._h47()
for m_ in (s61, s67, s74, s78, s80, l47, l70, j47):
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
# the cascade frame: which firm's incumbents were coded, and when
# ======================================================================
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
INC = ["31-34", "35-40", "41-49", "50+"]
YOUNG = ("22-25", "26-30")
N_FIRMS = 150
FIRMS = list(range(1, N_FIRMS + 1))
EXPOSED = set(range(1, 51))              # their INCUMBENTS do high work
THIN_CODED = list(range(101, 107))       # 3 coded, plenty of person-months
FORWARD_ONLY = list(range(107, 111))     # coded in 2021 alone
CASCADE_ONLY = list(range(111, 119))     # coded in 2018 alone
NO_CODE = list(range(121, 127))          # never coded
TINY = list(range(131, 136))             # one person-month, coded
TINY_NO_CODE = list(range(136, 141))     # one person-month, never coded
N_EDU_FIRMS = 147                        # so 148-150 are occupation-only
OCC_ONLY = [148, 149, 150]
SMALL = set(TINY + TINY_NO_CODE)
UNCODED = set(NO_CODE + TINY_NO_CODE)

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


def code_for(emp: int, i: int, want_hi: bool) -> str:
    pool = HI if want_hi else LO
    return pool[(emp * 5 + i * 37) % len(pool)]


def cascade_frame(swap_young: bool = False, recoded=frozenset(),
                  vintage_only: bool = False) -> pd.DataFrame:
    """
    The cascade pull's frame, drawn deterministically so that two arms
    differ in exactly the thing being varied.

    `swap_young` flips the young cells' occupations and leaves the
    incumbents alone, which is the invariance the design claims.
    `recoded` is the set of firms whose INCUMBENTS a later register files
    under low-exposure work; the head counts are untouched, so the
    population cannot move with the coding. `vintage_only` builds the
    single-vintage frame the as-of arm pulls, which carries no source
    year of its own.
    """
    rows = []
    for emp in FIRMS:
        hi = (emp in EXPOSED) and (emp not in recoded)
        # which vintage answers for this firm's INCUMBENTS
        if emp in UNCODED:
            src = None
        elif emp in FORWARD_ONLY:
            src = "2021"
        elif emp in CASCADE_ONLY:
            src = "2018"
        else:
            src = "2019"
        for age in AGES:
            young = age in YOUNG
            want_hi = (not hi) if young else hi
            if swap_young and young:
                want_hi = not want_hi
            s_ = ("2019" if emp not in UNCODED else None) if young else src
            if emp in SMALL:
                # one incumbent in one band, so the floor decides them and
                # nothing else does
                if age == "41-49":
                    if s_ is not None:
                        rows.append((emp, age, code_for(emp, 0, want_hi),
                                     s_, 1))
                    rows.append((emp, age, "____", "none", 1))
                continue
            if emp in THIN_CODED and not young:
                # FOUR coded incumbents in ONE band, so the firm has fewer
                # than five coded November persons in total and script 65's
                # rule loses it, while its sixty incumbent person-months
                # clear the floor the education route uses
                if age == "31-34" and s_ is not None:
                    rows.append((emp, age, code_for(emp, 0, want_hi), s_, 4))
            elif s_ is not None:
                for i in range(3):
                    rows.append((emp, age, code_for(emp, i, want_hi), s_,
                                 4 + emp % 6))
            # a code the DAIOE file does not hold, and a worker the
            # register leaves uncoded: both count in the denominator and
            # neither can carry an exposure
            rows.append((emp, age, UNSCORED_CODE, "2019", 2))
            rows.append((emp, age, "____", "none", 3))
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "source_year", "n"])
    if vintage_only:
        # the as-of pull groups on the code alone and carries no source
        d = (d.groupby(["employer_id", "age_group", "ssyk4"],
                       observed=True)["n"].sum().reset_index())
    return d


CASC = cascade_frame()
CASC.to_parquet(s82.CASC_CACHE, index=False)
# 47L's own baseline, for the head-count audit: the same frame summed
# over the source year, which is exactly what 47L's pull returns.
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(s82.BASE_CACHE, index=False)


def counts_2019() -> pd.DataFrame:
    """47L's monthly counts for the base year, which the floor sums."""
    rows = []
    for emp in FIRMS:
        tiny = emp in SMALL
        months = ["2019-01"] if tiny else [f"2019-{m:02d}"
                                           for m in range(1, 13)]
        for ym in months:
            for age in AGES:
                if tiny and age != "41-49":
                    continue
                rows.append((emp, ym, age, 1 if tiny else 5))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


counts_2019().to_parquet(s82.COUNTS_2019_CACHE, index=False)

INCF = s82.incumbent_frame(CASC, SCORES, j47)
NFLOOR = s82.incumbent_floor_series(l47, CASC, j47)
check("the floor is on incumbent PERSON-MONTHS, the education route's "
      "own unit, when 47L's 2019 counts are on the share",
      s82.BASIS == "person-months", s82.BASIS)
check("and the tiny firms have one incumbent person-month while the "
      "ordinary ones have sixty",
      int(NFLOOR.loc[TINY[0]]) == 1 and int(NFLOOR.loc[1]) == 240,
      f"tiny {int(NFLOOR.loc[TINY[0]])}, ordinary {int(NFLOOR.loc[1])}")

OCC = s82.occ_route_exposure(INCF, SCORES, NFLOOR, s82.FLOOR_MAIN,
                             s82.ARM_YEARS[s82.MAIN_ARM])
OLD = s82.old_rule_scored(INCF)
Q4F = set(OCC.loc[OCC["fq"] == 4, "employer_id"].astype(int))
SCORED = set(OCC["employer_id"].astype(int))

check("the occupation route scores every firm with incumbents enough and "
      "a code, and no other",
      SCORED == set(FIRMS) - set(FORWARD_ONLY) - UNCODED - set(TINY),
      f"{len(SCORED)} scored of {N_FIRMS}")
check("only firms whose INCUMBENTS do high-exposure work reach the top "
      "quartile, so the young never enter the score",
      Q4F <= EXPOSED and len(Q4F) > 10,
      f"{len(Q4F)} in Q4, {len(Q4F - EXPOSED)} of them not "
      f"incumbent-exposed")
SW = s82.occ_route_exposure(
    s82.incumbent_frame(cascade_frame(swap_young=True), SCORES, j47),
    SCORES, NFLOOR, s82.FLOOR_MAIN, s82.ARM_YEARS[s82.MAIN_ARM])
check("and changing ONLY the young cells leaves the score identical",
      SW[["employer_id", "fq", "mix"]].equals(
          OCC[["employer_id", "fq", "mix"]]),
      f"{int((SW['fq'].to_numpy() != OCC['fq'].to_numpy()).sum())} "
      f"quartiles moved")

# ---- THE FLOOR: on the firm's incumbents, not on its coded ones -------
check("the OLD rule loses a firm with three coded incumbents, because it "
      "counts CODED November persons against a floor of five",
      not (set(THIN_CODED) & OLD),
      f"{len(set(THIN_CODED) & OLD)} of {len(THIN_CODED)} kept by the old "
      f"rule")
check("and the floor on the firm's INCUMBENTS keeps them, the mix being "
      "formed over whatever share carries a code",
      set(THIN_CODED) <= SCORED,
      f"{len(set(THIN_CODED) & SCORED)} of {len(THIN_CODED)} scored")
thin_cov = float(OCC.set_index("employer_id").loc[THIN_CODED[0], "coverage"])
check("and their coverage is reported rather than assumed away",
      0 < thin_cov < 0.5, f"coverage {thin_cov:.3f}")
check("a firm with one incumbent person-month is below the floor",
      not (set(TINY) & SCORED))
check("a firm with incumbents and no code in any year cannot be scored, "
      "which is a different loss from the floor",
      not (UNCODED & SCORED))

# ---- THE CASCADE: backward only ---------------------------------------
check("a firm coded in 2018 and in no later year is recovered by the "
      "cascade",
      set(CASCADE_ONLY) <= SCORED,
      f"{len(set(CASCADE_ONLY) & SCORED)} of {len(CASCADE_ONLY)}")
check("and every one of its coded incumbents is reported as coming from "
      "a year before the freeze year",
      float(OCC.set_index("employer_id").loc[CASCADE_ONLY[0],
                                             "share_not_2019"]) == 1.0
      and float(OCC.set_index("employer_id").loc[1, "share_not_2019"]) == 0.0,
      f"{float(OCC.set_index('employer_id').loc[CASCADE_ONLY[0], 'share_not_2019']):.2f}")
check("the reported cascade reaches no year after the freeze year",
      max(s82.ARM_YEARS[s82.MAIN_ARM]) <= s82.BASE_YEAR
      and s82.MAIN_ARM == "backward",
      str(s82.ARM_YEARS[s82.MAIN_ARM]))
check("so a firm coded in 2021 alone is NOT on the reported score",
      not (set(FORWARD_ONLY) & SCORED))
FWD = s82.occ_route_exposure(INCF, SCORES, NFLOOR, s82.FLOOR_MAIN,
                             s82.ARM_YEARS["forward"])
check("and IS on the forward arm, which is reported and is never the "
      "score",
      set(FORWARD_ONLY) <= set(FWD["employer_id"].astype(int))
      and len(FWD) == len(OCC) + len(FORWARD_ONLY),
      f"{len(FWD)} on the forward arm against {len(OCC)} on the reported "
      f"one")

# ---- the quartile cuts are weighted by employment, not by employers ---
def lopsided():
    rows, floor = [], {}
    for i in range(100):
        emp = 1000 + i
        big = i >= 90
        code = (HI if big else LO)[(i * 3) % (len(HI) if big else len(LO))]
        rows.append((emp, "41-49", code, "2019", 100 if big else 5))
        floor[emp] = 100 if big else 5
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "source_year", "n"])
    return (s82.incumbent_frame(d, SCORES, j47),
            pd.Series(floor).rename_axis("employer_id").rename("n_floor"))


li, lf = lopsided()
LOP = s82.occ_route_exposure(li, SCORES, lf, 1, s82.ARM_YEARS[s82.MAIN_ARM])
w_top = float(LOP.loc[LOP["fq"] == 4, "n"].sum() / LOP["n"].sum())
f_top = float((LOP["fq"] == 4).mean())
check("the quartile cut points are weighted by incumbent employment: the "
      "top quartile holds about a quarter of employment and far fewer "
      "than a quarter of employers",
      0.18 <= w_top <= 0.35 and f_top < 0.12,
      f"{w_top:.1%} of incumbent employment, {f_top:.1%} of employers")


# ======================================================================
# the cascade query, and the catalogue probe
# ======================================================================
CATALOGUE = pd.DataFrame(
    [(f"Individ_{y}", c)
     for y in (2015, 2016, 2018, 2019, 2020, 2021)     # 2017 is ABSENT
     for c in ("P1207_LopNr_PersonNr", "FodelseAr", "Ssyk4_2012_J16")],
    columns=["TABLE_NAME", "COLUMN_NAME"])
SQL_CALLS = []


def fake_read_sql(q, conn=None, *a, **kw):
    SQL_CALLS.append(str(q))
    if "INFORMATION_SCHEMA" in str(q):
        return CATALOGUE.copy()
    if "Individ_2021 v" in str(q):
        return cascade_frame(recoded=RECODED, vintage_only=True)
    return CASC.copy()


_real_read_sql = pd.read_sql
RECODED = set(sorted(Q4F)[::3])
pd.read_sql = fake_read_sql
s82.CASC_CACHE.unlink()
s82.NOTES.clear()
PULLED = s82.baseline_cascade()
pd.read_sql = _real_read_sql
check("the cascade probes the catalogue before it pulls",
      len(SQL_CALLS) == 2 and "INFORMATION_SCHEMA" in SQL_CALLS[0],
      f"{len(SQL_CALLS)} calls")
q = SQL_CALLS[1]
check("a year the catalogue does not hold is skipped and said so",
      "Individ_2017" not in q
      and any("2017" in n and "absent" in n for n in s82.NOTES),
      next((n for n in s82.NOTES if "absent" in n), "")[:100])
check("every vintage the catalogue does hold is joined, in the cascade's "
      "own order",
      all(f"Individ_{y} i{y}" in q for y in (2019, 2018, 2016, 2015,
                                             2020, 2021))
      and q.index("Individ_2019 i2019") < q.index("Individ_2018 i2018")
      < q.index("Individ_2016 i2016"))
check("each vintage is cleaned of the missing conventions BEFORE the "
      "COALESCE, so a '****' in 2019 cannot block the 2018 code",
      q.count("LEFT(LTRIM(i") >= 6 and "COALESCE(CASE WHEN i2019" in
      q.replace("\n", "").replace("  ", ""),
      f"{q.count('LEFT(LTRIM(i')} cleaned vintages")
check("the birth year and the sample filter come from the 2019 register, "
      "so the population cannot move with the coding",
      "WHERE 2019 - TRY_CAST(b.FodelseAr AS INT) BETWEEN 22 AND 69" in q
      and "Individ_2019 b" in q)
check("the source year travels with the code",
      "AS source_year" in q and "THEN '2018'" in q)
check("and the frame is cached, so the other parts do not pull it again",
      s82.CASC_CACHE.exists() and len(PULLED) == len(CASC))

s82.NOTES.clear()
s82.cascade_audit(CASC)
check("the head-count audit passes on a frame that agrees with 47L's own",
      any("0 disagree" in n for n in s82.NOTES),
      next((n for n in s82.NOTES if "disagree" in n), "")[:90])
s82.NOTES.clear()
BAD = CASC.copy()
BAD.loc[BAD.index[:20], "n"] = BAD.loc[BAD.index[:20], "n"] + 1
s82.cascade_audit(BAD)
check("and catches one that does not, since a vintage with two rows for "
      "one person would inflate the head count",
      any("disagree" in n and "NOT the head count" in n for n in s82.NOTES),
      next((n for n in s82.NOTES if "disagree" in n), "")[:90])

# the November head-count fallback, when the 2019 counts cannot be read
_keep = s82.COUNTS_2019_CACHE.read_bytes()
s82.COUNTS_2019_CACHE.unlink()
_pull = l47.q_counts
l47.q_counts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no SQL"))
s82.NOTES.clear()
NF2 = s82.incumbent_floor_series(l47, CASC, j47)
l47.q_counts = _pull
s82.COUNTS_2019_CACHE.write_bytes(_keep)
check("when the 2019 counts cannot be read the floor falls back to the "
      "November head count and SAYS SO",
      s82.BASIS == "November head count"
      and any("NOT commensurable" in n for n in s82.NOTES)
      and int(NF2.loc[1]) < int(NFLOOR.loc[1]),
      f"{s82.BASIS}: {int(NF2.loc[1])} against {int(NFLOOR.loc[1])}")
s82.incumbent_floor_series(l47, CASC, j47)      # back to person-months
check("and the person-month basis is restored once they can be",
      s82.BASIS == "person-months")


# ======================================================================
# the panel with the planted steps
# ======================================================================
FIX = Fixture(mc, h47, n_firms=N_EDU_FIRMS, n_exposed=50)
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

FITS = []
_fit = s82.fit


def counting_fit(b, tag, *a, **kw):
    FITS.append(tag)
    return _fit(b, tag, *a, **kw)


s82.fit = counting_fit


# ======================================================================
# A. the score, the decomposition and the coverage
# ======================================================================
s82.NOTES.clear(); s82.FAILURES.clear(); FITS.clear()
COV, CSUM, CROSS, LOSS, FSUM = s82.part_a(
    COUNTS, OCC, EDU, CASC, INCF, NFLOOR, SCORES, s61, s80, j47)
check("A: no fit runs in Part A", not FITS, str(FITS))
st = COV[(COV["block"] == "step") & (COV["item"] == "incumbents_resolved")]
check("A: where the cascade resolved each incumbent is reported, step by "
      "step",
      {"2019", "2018", "none"} <= set(st["group"])
      and abs(float(st["share"].sum()) - 1.0) < 1e-9,
      f"{sorted(set(st['group']))}, shares sum to "
      f"{float(st['share'].sum()):.6f}")
cv = COV[COV["block"] == "coverage"].set_index(["group", "item"])
c19 = float(cv.loc[("31-69 incumbents", "coded_share_2019_only"), "share"])
cbk = float(cv.loc[("31-69 incumbents", "coded_share_backward"), "share"])
s19 = float(cv.loc[("31-69 incumbents", "scored_share_2019_only"), "share"])
sbk = float(cv.loc[("31-69 incumbents", "scored_share_backward"), "share"])
check("A: coverage is reported before AND after the cascade, and the "
      "cascade raises it",
      cbk > c19 and sbk > s19,
      f"coded {c19:.3f} -> {cbk:.3f}, scored {s19:.3f} -> {sbk:.3f}")
check("A: and the scored share is below the coded share, since a code "
      "outside the DAIOE file carries no exposure",
      sbk < cbk, f"{sbk:.3f} against {cbk:.3f}")
check("A: the employer loss is decomposed, and the groups partition the "
      "employers",
      sum(LOSS[k] for k in
          ("scored_by_both_rules", "recovered_by_the_floor",
           "recovered_by_the_cascade", "lost_to_the_floor_only",
           "lost_to_missing_codes_only", "lost_to_both",
           "scored_old_but_not_new")) == LOSS["n"],
      f"{LOSS['n']} employers")
check("A: no employer the old rule scored is lost by the new one, so the "
      "new rule is a relaxation and not a different sample",
      LOSS["scored_old_but_not_new"] == 0,
      f"{LOSS['scored_old_but_not_new']}")
check("A: the firms the FLOOR lost are counted as such",
      LOSS["recovered_by_the_floor"] == len(THIN_CODED),
      f"{LOSS['recovered_by_the_floor']} against {len(THIN_CODED)} planted")
check("A: the firms the CASCADE recovers are counted separately",
      LOSS["recovered_by_the_cascade"] == len(CASCADE_ONLY),
      f"{LOSS['recovered_by_the_cascade']} against {len(CASCADE_ONLY)}")
check("A: the firms still lost to missing codes after the full backward "
      "cascade are counted separately again",
      LOSS["lost_to_missing_codes_only"] == len(NO_CODE) + len(FORWARD_ONLY),
      f"{LOSS['lost_to_missing_codes_only']} against "
      f"{len(NO_CODE) + len(FORWARD_ONLY)} planted (never coded, plus the "
      f"four coded only after the freeze year)")
check("A: the firms lost to the floor alone and to both are counted "
      "separately",
      LOSS["lost_to_the_floor_only"] == len(TINY)
      and LOSS["lost_to_both"] == len(TINY_NO_CODE),
      f"floor {LOSS['lost_to_the_floor_only']} of {len(TINY)}, both "
      f"{LOSS['lost_to_both']} of {len(TINY_NO_CODE)}")
check("A: the floor sensitivity is reported at 1, 3 and 5, and a lower "
      "floor scores more employers",
      set(FSUM) == set(s82.FLOORS) and FSUM[1] >= FSUM[3] >= FSUM[5],
      str(FSUM))
rt = COV[COV["block"] == "route"].set_index("group")
check("A: the employers each route can score are counted, and the three "
      "the occupation route has to itself are SUPPRESSED, being fewer "
      "than five",
      pd.isna(rt.loc["occupation_only", "n_employers"])
      and int(rt.loc["both_routes", "n_employers"])
      == len(SCORED & set(EDU["employer_id"].astype(int))),
      f"occupation_only {rt.loc['occupation_only', 'n_employers']}, "
      f"planted {len(OCC_ONLY)}")
check("A: a suppressed count takes its own share with it",
      pd.isna(rt.loc["occupation_only", "share"])
      and pd.isna(rt.loc["occupation_only", "value"]))
check("A: the crosstab of the two quartiles is exported with the "
      "diagonal share and the rank correlation",
      CROSS and 0.0 <= CROSS["diag"] <= 1.0
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
         & (SZ["item"] == "mean_headcount_p50")]["value"].isna().all())
small = COV["n_employers"]
check("A: nothing between one and four leaves MONA",
      not ((small > 0) & (small < s82.FLOOR)).any()
      and int(small.isna().sum()) >= 1,
      f"{int(small.isna().sum())} suppressed cells")
check("A: the export is on disk",
      (s82.OUT / "occ_route_coverage.csv").exists())


# ======================================================================
# B. the headline, the profile, the floor variants and the forward arm
# ======================================================================
s82.NOTES.clear(); FITS.clear()
HEAD, PROF = s82.part_b(COUNTS, INCF, SCORES, NFLOOR, s61, s74, s78, l70,
                        j47)
check("B: six fits run: both bands, the profile, two floor variants and "
      "the forward arm",
      len(FITS) == 6 and sum("forward" in t for t in FITS) == 1
      and sum("_f1" in t or "_f3" in t for t in FITS) == 2, str(FITS))
H = pd.DataFrame(HEAD)
M = H[(H.arm == s82.MAIN_ARM) & (H.floor == s82.FLOOR_MAIN)]
check("B: both young bands are exported with every term of Equation (2)",
      set(M["young_band"]) == {"22-25", "26-30"}
      and {"rb_x_high_x_young", "q1_x_high_x_young", "interim_x_high_x_young",
           "post_x_high_x_young"} <= set(M["term"]),
      str(sorted(set(M["term"])))[:110])
p22 = M[(M.young_band == "22-25")
        & (M.term == "post_x_high_x_young")].iloc[0]
want22 = STEP_22 + float(np.log((1.0 + np.exp(DIFF)) / 2.0))
check("B: the planted adoption step at 22-25 comes back",
      abs(float(p22["coef"]) - want22) < 0.08,
      f"{float(p22['coef']):+.4f} against {want22:+.4f}")
check("B: every row carries the arm, the floor, the employer count, the "
      "number of observations and the share of coded incumbents from "
      "before the freeze year",
      {"arm", "floor", "n_firms", "n_obs", "share_not_2019"}
      <= set(H.columns) and H["n_firms"].notna().all()
      and H["share_not_2019"].notna().all(),
      f"share_not_2019 {float(p22['share_not_2019']):.4f}")
check("B: the floor variants and the forward arm are exported beside the "
      "reported score and marked as what they are",
      set(H["floor"]) == set(s82.FLOORS)
      and set(H["arm"]) == {s82.MAIN_ARM, "forward"},
      f"floors {sorted(set(H['floor']))}, arms {sorted(set(H['arm']))}")
f1 = H[(H.floor == 1) & (H.term == "post_x_high_x_young")].iloc[0]
check("B: a lower floor scores more employers, which is what the floor "
      "sensitivity is for",
      int(f1["n_firms"]) >= int(p22["n_firms"]),
      f"{int(f1['n_firms'])} at floor 1 against {int(p22['n_firms'])} at "
      f"floor {s82.FLOOR_MAIN}")
P = pd.DataFrame(PROF)
check("B: the profile is exported for all six bands against 41-49",
      set(P["band"]) == set(s74.BANDS)
      and float(P[P.band == "41-49"]["coef"].iloc[0]) == 0.0,
      str(sorted(set(P["band"]))))
check("B: the planted gain at 50 and over comes back against 41-49",
      abs(float(P[P.band == "50+"]["coef"].iloc[0]) - STEP_50) < 0.08,
      f"{float(P[P.band == '50+']['coef'].iloc[0]):+.4f} against "
      f"{STEP_50:+.4f}")
V1, L1 = s82.verdict_headline(HEAD)
V2, L2 = s82.verdict_profile(PROF)
check("B: read rule 1 is met on the planted world", V1 == "REPRODUCES",
      V1 + " | " + L1[1].strip())
check("B: read rule 2 is met on the planted world",
      V2 == "THE PROFILE REPRODUCES", V2 + " | " + L2[1].strip())
POISON = [dict(r) for r in HEAD] + [
    {"young_band": "22-25", "term": "post_x_high_x_young", "arm": "forward",
     "floor": s82.FLOOR_MAIN, "coef": +0.9, "se": 0.001, "t": 900.0,
     "n_firms": 9, "n_obs": 9}]
check("B: the verdict reads the REPORTED arm and ignores the forward one, "
      "however loud it is",
      s82.verdict_headline(POISON)[0] == "REPRODUCES"
      and abs(float(s82.verdict_headline(POISON)[1][1].split()[2])
              - float(p22["coef"])) < 1e-4,
      s82.verdict_headline(POISON)[1][1].strip()[:70])
check("B: the exports are on disk and carry the covariance",
      (s82.OUT / "occ_route_headline.csv").exists()
      and (s82.OUT / "occ_route_profile.csv").exists()
      and len(list(s82.OUT.glob("vcov_s82_stock_*.csv"))) == 5,
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
s82.NOTES.clear(); FITS.clear()
GROWS, STEPS = s82.part_c_gender(SEX, OCC, s67, s78, j47)
G = pd.DataFrame(GROWS)
check("C: the sex specification runs one fit and exports every term",
      len(FITS) == 1
      and {"post_x_high_x_young", "post_x_high_x_female",
           "post_x_high_x_young_x_female"} <= set(G["term"]), str(FITS))
check("C: the planted female differential comes back",
      abs(STEPS["female_differential"][0] - DIFF) < 0.08,
      f"{STEPS['female_differential'][0]:+.4f} against {DIFF:+.4f}")
check("C: the women's step is the sum and its standard error comes from "
      "the covariance of the fit, not from adding two standard errors",
      STEPS["female_step"][1] is not None
      and abs(STEPS["female_step"][0] - (STEP_22 + DIFF)) < 0.09
      and STEPS["female_step"][1] < (STEPS["male_step"][1]
                                     + STEPS["female_differential"][1]),
      f"{STEPS['female_step'][0]:+.4f} ({STEPS['female_step'][1]:.4f})")
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
fs = F[(F.outcome == "seps") & (F.term == "post_x_high_x_young")].iloc[0]
fh = F[(F.outcome == "hires") & (F.term == "post_x_high_x_young")].iloc[0]
check("C: hires and separations are both fitted at 22-25, the planted "
      "rise in separations comes back and hiring stays flat",
      len(FITS) == 2 and abs(float(fs["coef"]) - STEP_SEP) < 0.08
      and abs(float(fh["coef"])) < 0.05,
      f"separations {float(fs['coef']):+.4f} against {STEP_SEP:+.4f}, "
      f"hires {float(fh['coef']):+.4f}")
check("C: the education-route margins travel beside them",
      abs(float(fs["edu_coef"]) - s82.EDU_FLOW["seps"][0]) < 1e-9
      and abs(float(fh["edu_coef"]) - s82.EDU_FLOW["hires"][0]) < 1e-9)

s82.VINT_CACHE.unlink(missing_ok=True)
s82.NOTES.clear(); FITS.clear(); SQL_CALLS.clear()
pd.read_sql = fake_read_sql
VROWS, STAB = s82.part_c_vintage(COUNTS, INCF, SCORES, NFLOOR, j47, s61, s78)
pd.read_sql = _real_read_sql
check("C: the vintage arm pulls once when the cache is absent and caches "
      "what it pulled",
      len(SQL_CALLS) == 1 and s82.VINT_CACHE.exists(),
      f"{len(SQL_CALLS)} SQL calls")
q = SQL_CALLS[0] if SQL_CALLS else ""
check("C: the as-of query takes the birth year from the 2019 register "
      "and the code from the later one, so a worker the later register "
      "does not hold loses his code and not his place",
      "Individ_2019 b" in q and f"Individ_{s82.VINTAGE} v" in q
      and "b.FodelseAr" in q and "v.Ssyk4_2012_J16" in q
      and "b.Ssyk4_2012_J16" not in q)
check("C: three arms are fitted on one panel AND one set of employers, "
      "so the difference between them is the score and not the sample",
      len(FITS) == 3
      and len({r["n_obs"] for r in VROWS if r["block"] == "fit"}) == 1
      and len({r["n_firms"] for r in VROWS if r["block"] == "fit"}) == 1,
      f"{FITS}, "
      f"{sorted({r['n_firms'] for r in VROWS if r['block'] == 'fit'})} firms")
art = {r["item"]: r["value"] for r in VROWS if r["block"] == "artefact"}
check("C: both the re-coding artefact and the cascade's own contribution "
      "are reported, each against the 2019 code alone",
      {"recoding_artefact", "cascade_effect"} == set(art),
      ", ".join(f"{k} {v:+.4f}" for k, v in art.items()))
check("C: the re-scoring moves some employers out of their quartile and "
      "not all of them",
      0.0 < STAB[f"asof_{s82.VINTAGE}"]["share_keeping_quartile"] < 1.0,
      f"{STAB[f'asof_{s82.VINTAGE}']['share_keeping_quartile']:.1%} keep "
      f"their quartile")
for nm in ("occ_route_gender.csv", "occ_route_flows.csv",
           "occ_route_vintage.csv"):
    check(f"C: {nm} is on disk", (s82.OUT / nm).exists())


# ======================================================================
# end to end
# ======================================================================
for f in s82.OUT.glob("*.csv"):
    f.unlink()
s82.NOTES.clear(); s82.FAILURES.clear(); FITS.clear(); SQL_CALLS.clear()
pd.read_sql = fake_read_sql
s82.main()
pd.read_sql = _real_read_sql
for nm in ("occ_route_coverage.csv", "occ_route_headline.csv",
           "occ_route_profile.csv", "occ_route_gender.csv",
           "occ_route_flows.csv", "occ_route_vintage.csv",
           "82_summary.txt"):
    check(f"main() writes {nm}", (s82.OUT / nm).exists())
check("main() made no SQL call, since every frame is cached",
      not SQL_CALLS, f"{len(SQL_CALLS)} calls")
check("main() ran twelve fits: six for B and six for C",
      len(FITS) == 12, str(FITS))
check("no part failed in the end-to-end run", not s82.FAILURES,
      str(s82.FAILURES))
summ = (s82.OUT / "82_summary.txt").read_text(encoding="utf-8")
for must in ("READ RULES, FIXED BEFORE THE RUN",
             "There is NO coefficient gate",
             "1. REPRODUCES if the adoption step at 22-25",
             "2. THE PROFILE REPRODUCES if the 50-and-over band",
             "3. THE SEX RESULT REPRODUCES if the female differential",
             "-0.0408 (0.0150)", "-0.0746 (0.0142)", "+0.0787 (0.0203)",
             "THE FLOOR AND ITS UNIT",
             "person-months",
             "commensurable",
             "THE CASCADE",
             "Backward only",
             "where the cascade resolves each incumbent",
             "coverage of the code among incumbents, before and",
             "where the employers the OLD rule lost went",
             "recovered_by_the_floor", "recovered_by_the_cascade",
             "lost_to_the_floor_only", "lost_to_missing_codes_only",
             "lost_to_both",
             "the floor sensitivity, employers scored",
             "Spearman rank correlation",
             "1. THE ADOPTION STEP AT 22-25: REPRODUCES",
             "2. THE AGE PROFILE: THE PROFILE REPRODUCES",
             "3. THE FEMALE DIFFERENTIAL: THE SEX RESULT REPRODUCES",
             "the vintage check, every arm against the 2019 code",
             "THE THREE VERDICTS",
             "WHAT TO EXPECT, SO IT IS NOT READ AS A BUG"):
    check(f"the summary states {must!r}", must in summ)
check("the summary names the defect it repairs and the ladder that is "
      "not this comparison",
      "172,396" in summ and "60,704" in summ and "CODED November" in summ)
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

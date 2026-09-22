#!/usr/bin/env python3
"""
test_80_industry_key.py -- the cascade must resolve each firm from the
                           source it was planted in, and the two exercises
                           must change their standard errors and nothing
                           else.

The checks are on mechanisms, not on outputs. One synthetic world on the
fixture's 140 employers (the top DAIOE quartile Q4 about a third of them),
with a Q4 seasonal bump in exposed firms' young cells so the calendar terms
have work to do, a known adoption step at each young band and a known extra
step for young women, and a stub register in which the industry code is
planted one firm at a time:

  111-116   absent from Ftg_2019 altogether, present in Ftg_2018 with one
            code and in Ftg_2020 with another. They must be resolved from
            2018, counted as resolved from 2018, and must carry the 2018
            code, because the cascade prefers the nearer year and the
            earlier of two equally near ones.
  117-122   present in Ftg_2019 with the value "****". That is a missing
            convention, not a code: they must be resolved from Ftg_2018,
            and no employer anywhere may end up with an empty code.
  123-128   only in Serrano, whose column is a FLOAT. Firm 123's raw value
            is 11.0 and must come back as 011, the growing of crops, and
            not as 110, the manufacture of beverages, which is what
            stripping the non-digits out of "11.0" would give.
  129-134   only in the business register, the last step of the cascade.
  135-140   in no source at all. They must be REPORTED as unresolved:
            not dropped from the panel, and not given a cluster each,
            since a cluster each is the defect the script exists to
            remove.

Groups of six, because a count of one to four does not leave MONA and a
suppressed group could not be read back.

Also tested: that the employers Ftg_2019 missed are described beside the
ones it found; that Part B returns the SAME COEFFICIENTS under employer
and industry clustering with different standard errors, which is the whole
mechanism, since clustering moves the covariance and nothing else; that
Part B reads lane 25's employer-clustered export rather than refitting when
it is on the share, checked by the fit that does not happen; that Part C
fits its baseline and its industry specification on identical firm sets;
that counts of one to four never leave MONA; and main() end to end off the
caches with a summary that states every read rule.

    CANARIES_DRYRUN=1 python3 revision/local/test_80_industry_key.py
"""
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
os.environ["CANARIES_80_PARTS"] = "ABC"
# The script's Tee caps the terminal echo at 2 KB, which is right under
# BatchClient's blocking pipe and wrong here: it would swallow the check
# lines printed after main(). The lane runner lifts it the same way.
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries80_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


s80 = load("80_industry_key.py", "s80")
s80.OUT = TMP / "out"; s80.OUT.mkdir()
s80.CACHE = mc.CACHE_DIR
s80.KEY_CACHE = mc.CACHE_DIR / "I_industry_key.parquet"
s80.PRIOR_ROOT = TMP                  # lane 25's folders live in the sandbox
s61, s67, s78, s73, j47, h47 = s80.load_modules()
s61.OUT = s67.OUT = s78.OUT = s80.OUT

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo = s78.exposure(frames[2019], book, spec, j47, s61)
Q4F = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
STEP1 = float(np.log(1.12))      # the tightening rise, 22-25
STEP2 = float(np.log(0.78))      # young men's adoption step, 22-25
DIFF = float(np.log(0.80))       # young women's extra step, 22-25
STEP_B = float(np.log(0.85))     # the adoption step at 26-30, both sexes
SEAS = float(np.log(1.25))       # the Q4 bump in every year
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


check("the fixture puts some employers in the top DAIOE quartile",
      0 < len(Q4F) < FIX.n_firms, f"{len(Q4F)} of {FIX.n_firms} firms")


# ======================================================================
# the stub register: which firm's code is planted where
# ======================================================================
FIRMS = list(range(1, FIX.n_firms + 1))
# Six firms in each planted group, so that no group is suppressed by the
# five-employer export floor and the coverage table can be read back.
ONLY_2018 = list(range(111, 117))            # and in 2020, with another code
STARRED = list(range(117, 123))              # in Ftg_2019 as "****"
ONLY_SERRANO = list(range(123, 129))
ONLY_FDB = list(range(129, 135))
NOWHERE = list(range(135, 141))
MISSED = ONLY_2018 + STARRED + ONLY_SERRANO + ONLY_FDB + NOWHERE
SHORTLIVED = [111, 112, 113]                 # absent from the last 18 months
IN_FTG2019 = [f for f in FIRMS if f not in
              (ONLY_2018 + ONLY_SERRANO + ONLY_FDB + NOWHERE)]
# Every firm Ftg_2019 can place gets one of five three-digit groups.
IND19 = {f: f"{100 + (f % 5) * 7:03d}" for f in FIRMS}
CODE_2018 = "201"        # what ONLY_2018 carries in 2018
CODE_2020 = "301"        # and in 2020, so the cascade's choice is visible
CODE_STARRED = "205"     # what the "****" firms carry in 2018
# Serrano's column is a float, so 011 arrives as 11.0 and 047 as 47.0. The
# text route would make those 110 and 470.
SERRANO_RAW = dict(zip(ONLY_SERRANO,
                       [11.0, 620.0, 47.0, 9.0, 452.0, 683.0]))
SERRANO_WANT = dict(zip(ONLY_SERRANO,
                        ["011", "620", "047", "009", "452", "683"]))
FDB_CODE = dict(zip(ONLY_FDB,
                    ["310", "452", "861", "478", "162", "521"]))

FTG_COLS = ["P1207_LopNr_PeOrgNr", "Org_Sni2007", "Ar"]
CATALOGUE = pd.DataFrame(
    [(f"Ftg_{y}", c, "varchar")
     for y in s80.FTG_YEARS for c in FTG_COLS]
    + [("Serrano_Serrano_20230614", c, t) for c, t in
       (("P1207_Lopnr_ORGNR", "int"), ("bransch_sni3", "float"),
        ("ser_year", "float"))]
    + [("FDB_JE_2014_2021", c, t) for c, t in
       (("P1207_Lopnr_peorgnr", "int"), ("ng3", "char"), ("ar", "varchar"))],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])


def ftg_rows(year: int) -> pd.DataFrame:
    """One LISA firm table as the stub server would return it."""
    if year == 2019:
        emp = [f for f in IN_FTG2019]
        ind = ["****" if f in STARRED else IND19[f] + "42" for f in emp]
    elif year == 2018:
        emp = ONLY_2018 + STARRED + IN_FTG2019[:20]
        ind = ([CODE_2018 + "10"] * len(ONLY_2018)
               + [CODE_STARRED + "10"] * len(STARRED)
               + [IND19[f] + "42" for f in IN_FTG2019[:20]])
    elif year == 2020:
        emp = ONLY_2018 + IN_FTG2019[:20]
        ind = ([CODE_2020 + "10"] * len(ONLY_2018)
               + [IND19[f] + "42" for f in IN_FTG2019[:20]])
    else:
        # The far years exist and hold only firms already placed, so the
        # cascade reaches them, adds nobody, and says so.
        emp = IN_FTG2019[:10]
        ind = [IND19[f] + "42" for f in emp]
    return pd.DataFrame({"employer_id": emp, "ind": ind})


SQL_CALLS = []


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    SQL_CALLS.append(ql[:400])
    if "information_schema" in ql:
        return CATALOGUE.copy()
    m = re.search(r"ftg_(\d{4})", ql)
    if m:
        return ftg_rows(int(m.group(1)))
    if "serrano_serrano" in ql:
        emp = ONLY_SERRANO + IN_FTG2019[:5]
        ind = ([SERRANO_RAW[f] for f in ONLY_SERRANO]
               + [float(IND19[f]) for f in IN_FTG2019[:5]])
        return pd.DataFrame({"employer_id": emp, "ind": ind})
    if "fdb_je" in ql:
        emp = ONLY_FDB + IN_FTG2019[:5]
        ind = ([FDB_CODE[f] for f in ONLY_FDB]
               + [IND19[f] for f in IN_FTG2019[:5]])
        return pd.DataFrame({"employer_id": emp, "ind": ind})
    raise AssertionError(f"unexpected query: {str(q)[:120]}")


pd.read_sql = fake_read_sql
mc.connect = lambda: object()


# ======================================================================
# the panel: employer x band x sex x month counts with the planted steps
# ======================================================================

def counts_by_sex() -> pd.DataFrame:
    """
    The fixture's employment, with three plants.

    The employers Ftg_2019 misses are drawn smaller, and three of them stop
    employing anybody eighteen months before the panel ends, so the size
    distribution Part A exports has something to find. None of them is in
    the exposed set, so the planted steps are not disturbed.
    """
    rng = np.random.default_rng(80)
    lam0 = {"22-25": 24, "26-30": 20, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    months = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
              for m in range(1, 13 if y < 2025 else 7)]
    stop = months[-18]
    rows = []
    for emp in FIRMS:
        hit = emp in Q4F
        small = emp in MISSED
        for ym in months:
            if emp in SHORTLIVED and ym >= stop:
                continue                   # employs nobody from here on
            q = (int(ym[5:7]) - 1) // 3 + 1
            for age, lam in lam0.items():
                for g in ("1", "2"):
                    lam_ = (lam / 6 if small else lam) / 2
                    if hit and age == "22-25":
                        if q == 4:
                            lam_ *= np.exp(SEAS)
                        if ym >= mc.RIKSBANK_YM:
                            lam_ *= np.exp(STEP1)
                        if ym >= s80.POST_FROM:
                            lam_ *= np.exp(STEP2)
                            if g == "2":
                                lam_ *= np.exp(DIFF)
                    if hit and age == "26-30":
                        if q == 4:
                            lam_ *= np.exp(SEAS)
                        if ym >= s80.POST_FROM:
                            lam_ *= np.exp(STEP_B)
                    rows.append((emp, ym, age, g, int(rng.poisson(lam_)) + 1))
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

# Count the fits, so "read lane 25's export rather than refit" is checked by
# the fit that does not happen and not by a log line.
FITS = []
_fit = s80.fit


def counting_fit(b, tag, *a, **kw):
    FITS.append(tag)
    return _fit(b, tag, *a, **kw)


s80.fit = counting_fit


# ======================================================================
# the key itself
# ======================================================================
s80.NOTES.clear(); s80.FAILURES.clear()
KEY = s80.industry_key(s73)
K = KEY.set_index(KEY["employer_id"].astype(int))

check("the cascade resolves every firm that any source can place",
      len(K) == FIX.n_firms - len(NOWHERE),
      f"{len(K)} firms of {FIX.n_firms - len(NOWHERE)} placeable")
check("a firm absent from the 2019 table but present in 2018 is resolved "
      "from 2018 and counted as such",
      all(K.loc[f, "source"] == "Ftg_2018" for f in ONLY_2018)
      and all(K.loc[f, "ind3"] == CODE_2018 for f in ONLY_2018),
      f"{K.loc[ONLY_2018[0], 'source']} / {K.loc[ONLY_2018[0], 'ind3']}")
check("and 2018 beats 2020, which also holds those firms under another code",
      not any(K.loc[f, "ind3"] == CODE_2020 for f in ONLY_2018))
check("a value of '****' is treated as missing and not as a code",
      all(K.loc[f, "source"] == "Ftg_2018" for f in STARRED)
      and all(K.loc[f, "ind3"] == CODE_STARRED for f in STARRED),
      f"{K.loc[STARRED[0], 'source']} / {K.loc[STARRED[0], 'ind3']}")
check("no employer anywhere carries an empty or unparseable code",
      not KEY["ind3"].isin(["", "nan", "None", "<NA>"]).any()
      and KEY["ind3"].str.fullmatch(r"\d{2,3}").all(),
      str(sorted(set(KEY["ind3"]))[:6]))
check("the missing conventions are counted by name in the notes",
      any(f"asterisks {len(STARRED)}" in n for n in s80.NOTES),
      next((n for n in s80.NOTES if "asterisks" in n), "")[:120])
check("Serrano's float column is cast and padded, not stripped of its dot",
      all(K.loc[f, "ind3"] == SERRANO_WANT[f] for f in ONLY_SERRANO)
      and all(K.loc[f, "source"] == "Serrano" for f in ONLY_SERRANO),
      f"firm {ONLY_SERRANO[0]} planted 11.0, resolved "
      f"{K.loc[ONLY_SERRANO[0], 'ind3']!r}, "
      f"which the text route would have made '110'")
check("the business register is the last step and still resolves its firms",
      all(K.loc[f, "source"] == "FDB_JE" for f in ONLY_FDB)
      and all(K.loc[f, "ind3"] == FDB_CODE[f] for f in ONLY_FDB))
check("a firm present nowhere is not invented a code",
      not any(f in set(K.index) for f in NOWHERE))
check("the far years are read, add nobody, and say so",
      any("added no firm" in n for n in s80.NOTES),
      next((n for n in s80.NOTES if "added no firm" in n), "")[:90])
check("the key is cached, so the other parts do not pull it again",
      s80.KEY_CACHE.exists())


# ======================================================================
# A. coverage, and what Ftg_2019 missed
# ======================================================================
s80.NOTES.clear(); FITS.clear()
n_sql = len(SQL_CALLS)
COV, CSUM = s80.part_a(COUNTS, SEX, expo, KEY, s61, s67, s78, s73, j47)
check("A: no fit runs in Part A", not FITS, str(FITS))
check("A: the cached key is read rather than pulled again",
      len(SQL_CALLS) == n_sql, f"{len(SQL_CALLS) - n_sql} calls")
check("A: every panel is reported",
      {"22-25 stock", "26-30 stock", "22-25 sex"} == set(COV["panel"]),
      str(sorted(set(COV["panel"]))))
step = COV[(COV["block"] == "step") & (COV["item"] == "resolved")]
p22 = step[step["panel"] == "22-25 stock"].set_index("group")
check("A: the 2019 table places the firms it holds and no others",
      int(p22.loc["Ftg_2019", "n_employers"]) == len(IN_FTG2019) - len(STARRED),
      f"{int(p22.loc['Ftg_2019', 'n_employers'])} of "
      f"{len(IN_FTG2019) - len(STARRED)}")
check("A: the later steps place exactly the firms planted in them",
      int(p22.loc["Ftg_2018", "n_employers"]) == len(ONLY_2018) + len(STARRED)
      and int(p22.loc["Serrano", "n_employers"]) == len(ONLY_SERRANO)
      and int(p22.loc["FDB_JE", "n_employers"]) == len(ONLY_FDB),
      f"2018 {int(p22.loc['Ftg_2018','n_employers'])}, "
      f"Serrano {int(p22.loc['Serrano','n_employers'])}, "
      f"FDB {int(p22.loc['FDB_JE','n_employers'])}")
check("A: a firm present nowhere is REPORTED unresolved, not dropped",
      "unresolved" in p22.index
      and int(p22.loc["unresolved", "n_employers"]) == len(NOWHERE),
      f"{int(p22.loc['unresolved', 'n_employers'])} unresolved")
check("A: the shares are on the record and the cascade reaches every firm",
      abs(float(p22["share"].sum()) - 1.0) < 1e-9,
      f"shares sum to {float(p22['share'].sum()):.6f}")
size = COV[COV["block"] == "size"]
m50 = size[(size["panel"] == "22-25 stock")
           & (size["group"] == "missed_by_Ftg_2019")
           & (size["item"] == "mean_headcount_p50")]["value"].iloc[0]
f50 = size[(size["panel"] == "22-25 stock")
           & (size["group"] == "in_Ftg_2019")
           & (size["item"] == "mean_headcount_p50")]["value"].iloc[0]
check("A: the employers the 2019 table missed are described beside the ones "
      "it found, and here they are the smaller ones",
      float(m50) < float(f50),
      f"median monthly headcount {float(m50):.1f} against {float(f50):.1f}")
mm = size[(size["panel"] == "22-25 stock")
          & (size["group"] == "missed_by_Ftg_2019")
          & (size["item"] == "months_active_mean")]["value"].iloc[0]
fm = size[(size["panel"] == "22-25 stock")
          & (size["group"] == "in_Ftg_2019")
          & (size["item"] == "months_active_mean")]["value"].iloc[0]
check("A: and the shorter lived ones", float(mm) < float(fm),
      f"months employing anybody {float(mm):.1f} against {float(fm):.1f}")
check("A: the export is on disk",
      (s80.OUT / "industry_key_coverage.csv").exists())


# ======================================================================
# B. the clustering redone, first with no lane 25 export on the share
# ======================================================================
s80.NOTES.clear(); FITS.clear()
rows_b, bsum = s80.part_b(COUNTS, SEX, expo, KEY, s61, s67, s78, s73, j47)
B = pd.DataFrame(rows_b)
check("B: with no lane 25 export on the share every fit is repeated here",
      len(FITS) == 6 and sum("clemp" in t for t in FITS) == 3, str(FITS))
check("B: both pooled bands and the sex specification are exported",
      set(B["spec"]) == {"pooled", "gender"}
      and set(B[B["spec"] == "pooled"]["young_band"]) == {"22-25", "26-30"},
      str(sorted(set(B["young_band"]))))
check("B: clustering moves no coefficient (identical to four decimals)",
      len(B) and B["coef_match_4dp"].all(),
      f"{int(B['coef_match_4dp'].sum())} of {len(B)} terms")
diff = (B["se_industry_complete"] - B["se_employer"]).abs()
check("B: and the standard errors DO move, which is the whole exercise",
      (B["se_industry_complete"] > 0).all() and (diff > 1e-9).all(),
      f"smallest absolute change {float(diff.min()):.6f}")
want_clusters = KEY["ind3"].nunique() + 1      # the industries, plus the
                                               # ONE residual group
check("B: the firms no source places share ONE residual group rather than "
      "a cluster each",
      int(B["n_clusters_complete"].iloc[0]) == want_clusters,
      f"{int(B['n_clusters_complete'].iloc[0])} clusters against "
      f"{want_clusters} wanted; a cluster each would give "
      f"{KEY['ind3'].nunique() + len(NOWHERE)}")
check("B: and they stay in the panel rather than being dropped",
      int(B["n_firms"].iloc[0]) == FIX.n_firms
      and int(B["n_unresolved"].iloc[0]) == len(NOWHERE),
      f"{int(B['n_firms'].iloc[0])} firms, "
      f"{int(B['n_unresolved'].iloc[0])} unresolved")
check("B: the employers coded from a source other than 2019 are counted",
      int(B["n_not_from_2019"].iloc[0])
      == len(ONLY_2018) + len(STARRED) + len(ONLY_SERRANO) + len(ONLY_FDB),
      f"{int(B['n_not_from_2019'].iloc[0])} of {len(MISSED) - len(NOWHERE)}")
gs = bsum["gender"]["steps"]
check("B: young men's adoption step is the planted step",
      "male_step" in gs and abs(gs["male_step"][0] - STEP2) < 0.07,
      f"{gs.get('male_step', ('?',))[0]:+.4f} against {STEP2:+.4f}")
check("B: the female differential is the planted extra step",
      "female_differential" in gs
      and abs(gs["female_differential"][0] - DIFF) < 0.07,
      f"{gs.get('female_differential', ('?',))[0]:+.4f} against {DIFF:+.4f}")
check("B: the women's step is the sum and carries a standard error under "
      "BOTH clusterings, from the covariance and not from adding two",
      "female_step" in gs and gs["female_step"][1] is not None
      and gs["female_step"][3] is not None
      and abs(gs["female_step"][0] - (STEP2 + DIFF)) < 0.09,
      f"{gs.get('female_step', ('?',))[0]:+.4f} employer "
      f"{gs['female_step'][1]} industry {gs['female_step'][3]}")
check("B: the covariance of every fit left with the exports",
      len(list(s80.OUT.glob("vcov_s80_*clind2*.csv"))) >= 3
      and (s80.OUT / "cluster_industry_v2.csv").exists(),
      f"{len(list(s80.OUT.glob('vcov_s80_*.csv')))} covariance files")

# ---- the same part with lane 25's and lane 26's exports in place ---------
L25 = TMP / "output_78b"; L25.mkdir()
L26 = TMP / "output_79a"; L26.mkdir()
P = B[B["spec"] == "pooled"].copy()
P["se_industry"] = P["se_industry_complete"] * 1.10   # the earlier hybrid
P["n_clusters_industry"] = 999
P[["young_band", "term", "coef", "se_employer", "se_industry",
   "coef_employer_run", "coef_match_4dp", "n_clusters_industry"]].to_csv(
    L25 / "cluster_industry.csv", index=False)
G = B[B["spec"] == "gender"].copy()
(G.rename(columns={"coef_employer_run": "coef_e", "se_employer": "se"})
 .assign(coef=lambda d: d["coef_e"])
 [["young_band", "term", "coef", "se", "n_obs", "status"]]
 .to_csv(L25 / "gender_eq2.csv", index=False))
shutil.copy(s80.OUT / "vcov_s80_gender_clemp_22_25.csv",
            L25 / "vcov_s78_gender_eq2_22_25.csv")
G2 = G.copy()
G2["se_industry"] = G2["se_industry_complete"] * 1.20
G2["n_clusters_industry"] = 5545
G2[["young_band", "term", "coef", "se_employer", "se_industry",
    "coef_employer_run", "coef_match_4dp", "n_clusters_industry"]].to_csv(
    L26 / "gender_cluster_industry.csv", index=False)

s80.NOTES.clear(); FITS.clear()
rows_b2, bsum2 = s80.part_b(COUNTS, SEX, expo, KEY, s61, s67, s78, s73, j47)
B2 = pd.DataFrame(rows_b2)
check("B: with the earlier exports on the share only the three new fits run",
      len(FITS) == 3 and not any("clemp" in t for t in FITS), str(FITS))
check("B: the read numbers reproduce the refitted ones to four decimals",
      B2["coef_match_4dp"].all()
      and np.allclose(B2["se_employer"], B["se_employer"], atol=1e-9))
check("B: the earlier hybrid standard errors are carried through, so the "
      "change is visible",
      np.allclose(B2[B2["spec"] == "pooled"]["se_industry_hybrid"],
                  B2[B2["spec"] == "pooled"]["se_industry_complete"] * 1.10,
                  atol=1e-9)
      and np.allclose(B2[B2["spec"] == "gender"]["se_industry_hybrid"],
                      B2[B2["spec"] == "gender"]["se_industry_complete"] * 1.20,
                      atol=1e-9))
check("B: and the cluster count the earlier run used is on the record beside "
      "the new one",
      set(B2[B2["spec"] == "gender"]["n_clusters_hybrid"]) == {5545.0}
      and int(B2["n_clusters_complete"].iloc[0]) == want_clusters)
check("B: the women's step keeps its employer-clustered standard error from "
      "the covariance beside the export",
      bsum2["gender"]["steps"]["female_step"][1] is not None
      and abs(bsum2["gender"]["steps"]["female_step"][1]
              - gs["female_step"][1]) < 1e-9)


# ======================================================================
# C. the industry effects redone
# ======================================================================
s80.NOTES.clear(); FITS.clear()
rows_c = s80.part_c(COUNTS, expo, KEY, s61, s78, s73, j47)
C = pd.DataFrame(rows_c)
check("C: both specifications are fitted at both bands",
      set(C["spec"]) == {"baseline_same_sample", "industry_age_month"}
      and set(C["young_band"]) == {"22-25", "26-30"},
      str(sorted(set(C["young_band"]))))
check("C: on identical firm sets, the employers the cascade could place",
      C["n_firms"].nunique() == 1
      and int(C["n_firms"].iloc[0]) == FIX.n_firms - len(NOWHERE),
      f"n_firms {sorted(C['n_firms'].unique())}")
for band in ("22-25", "26-30"):
    Cb = C[C["young_band"] == band]
    check(f"C: and on identical panels at {band} (the same cells in both fits)",
          Cb.groupby("spec")["n_obs"].nunique().eq(1).all()
          and Cb["n_obs"].nunique() == 1,
          f"n_obs {sorted(Cb['n_obs'].unique())}")
check("C: the firms coded from a source other than 2019 are counted and "
      "their share reported, since a carried-forward code absorbs less",
      int(C["n_not_from_2019"].iloc[0])
      == len(ONLY_2018) + len(STARRED) + len(ONLY_SERRANO) + len(ONLY_FDB)
      and abs(float(C["share_not_from_2019"].iloc[0])
              - int(C["n_not_from_2019"].iloc[0]) / (FIX.n_firms - len(NOWHERE)))
      < 1e-9,
      f"{int(C['n_not_from_2019'].iloc[0])} firms, "
      f"{float(C['share_not_from_2019'].iloc[0]):.1%}")
for band, want in (("22-25", STEP2 + float(np.log((1.0 + np.exp(DIFF)) / 2.0))),
                   ("26-30", STEP_B)):
    cb = C[(C["young_band"] == band) & (C["spec"] == "baseline_same_sample")
           & (C["term"] == "post_x_high_x_young")].iloc[0]
    check(f"C: the same-sample baseline returns the planted {band} step",
          abs(float(cb["coef"]) - want) < 0.08,
          f"{float(cb['coef']):+.4f} against {want:+.4f}")
ci = C[(C["young_band"] == "26-30") & (C["spec"] == "industry_age_month")
       & (C["term"] == "post_x_high_x_young")].iloc[0]
cb = C[(C["young_band"] == "26-30") & (C["spec"] == "baseline_same_sample")
       & (C["term"] == "post_x_high_x_young")].iloc[0]
check("C: the retained share is the ratio of the two adoption steps, and it "
      "is 1 on the baseline row",
      abs(float(ci["retained_share"]) - float(ci["coef"]) / float(cb["coef"]))
      < 1e-9 and float(cb["retained_share"]) == 1.0,
      f"retained {float(ci['retained_share']):.3f}")
check("C: the exports are on disk",
      (s80.OUT / "industry_seasonal_v2.csv").exists()
      and (s80.OUT / "vcov_s80_indseas2_ind_26_30.csv").exists())


# ======================================================================
# end to end
# ======================================================================
for f in s80.OUT.glob("*.csv"):
    f.unlink()
s80.NOTES.clear(); s80.FAILURES.clear(); FITS.clear()
n_sql = len(SQL_CALLS)
s80.main()
for nm in ("industry_key_coverage.csv", "cluster_industry_v2.csv",
           "industry_seasonal_v2.csv", "80_summary.txt"):
    check(f"main() writes {nm}", (s80.OUT / nm).exists())
check("main() made no SQL call at all, since the key is cached",
      len(SQL_CALLS) == n_sql, f"{len(SQL_CALLS) - n_sql} calls")
check("main() ran seven fits: three for B, which read the earlier exports, "
      "and four for C", len(FITS) == 7, str(FITS))
summ = (s80.OUT / "80_summary.txt").read_text(encoding="utf-8")
for must in ("READ RULES, FIXED BEFORE THE RUN",
             "THE CASCADE, PANEL BY PANEL",
             "employers resolved at each step",
             "unresolved",
             f"coefficients equal the employer-clustered run to "
             f"{s80.MATCH_DP} decimals: YES",
             "young women minus young men", "young women, adoption step",
             "hybrid", "industry",
             "INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR TERMS",
             "retained share of the adoption step",
             "carry a code from a source other than Ftg_2019",
             "AND WHEN QUOTING"):
    check(f"the summary states {must!r}", must in summ)
check("the summary names the defect it repairs",
      "5,545" in summ and "9,368" in summ and "265" in summ)
check("no part failed in the end-to-end run", not s80.FAILURES,
      str(s80.FAILURES))
check("the covariance files left with the outputs",
      len(list(s80.OUT.glob("vcov_s80_*.csv"))) >= 7,
      f"{len(list(s80.OUT.glob('vcov_s80_*.csv')))} files")
cov = pd.read_csv(s80.OUT / "industry_key_coverage.csv")
small = cov["n_employers"]
check("nothing between one and four leaves MONA",
      not ((small > 0) & (small < s80.FLOOR)).any(),
      f"{int(small.isna().sum())} suppressed cells")

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

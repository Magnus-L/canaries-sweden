#!/usr/bin/env python3
"""
test_78_final_checks.py -- each of the seven checks must find what was
                           planted, and nothing where nothing was.

Three synthetic worlds on the same employer population (the fixture's
140 employers, the top DAIOE quartile Q4 of about a third of them, all of
Q4 inside the 60 employers whose incumbents hold post-secondary
education), each with a Q4 seasonal bump in exposed firms' 22-25 cells so
the calendar terms have work to do, and with the oldest band drawn as
50-64 and 65-69 so the split profile has something to split:

  STEP world   exposed (Q4) 22-25 employment rises by a known step from
               April 2022 and falls by a known step from January 2024;
               young women in those cells fall by a further known amount.
               Part A(ii) must read FLAT; B must return the male step, the
               female differential and their sum; C must return the same
               coefficients under both clusterings; D must find that the
               DAIOE step survives the skill cut, because the step was
               planted on Q4 and Q4 is only part of the post-secondary
               quartile; F must fit its baseline and its industry
               specification on the same firms; G must drop April 2022 and
               still return the rise and the step.
  SENIOR world nothing on the young; the 65-69 cell of exposed firms rises
               by a known step from January 2024 while 50-64 does not. Part
               E must put the gain in 65-69 and not in 50-64. (Planted in
               its own world because a gain in the older bands moves the
               reference every youth contrast reads against.)
  DRIFT world  the same panel drawn from 2019, plus a linear pre-launch
               drift in exposed 22-25 cells. Part A(ii) must read NOT FLAT
               and return the slope; A(i) must run from 2019Q1 when the
               2019 and 2020 caches are present.
  SKILL world  the adoption step is planted on EVERY post-secondary
               employer, not on Q4. Part D must find that the skill cut
               reproduces the step and absorbs the DAIOE one.

Also tested: the industry loader's own-cluster rule for employers with
no 2019 code, the SQL stub is the only SQL (E reads its cache and never
pulls), main() runs end to end off the caches and its summary states
every read rule.

    CANARIES_DRYRUN=1 python3 revision/local/test_78_final_checks.py
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
os.environ["CANARIES_78_PARTS"] = "ABCDEFG"
# The script's Tee caps what reaches the terminal at 2 KB, which is right
# under BatchClient's blocking pipe and wrong here: it would swallow the
# check lines printed after main(). The lane runner lifts it the same way.
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries78_"))
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


s78 = load("78_final_checks.py", "s78")
s78.OUT = TMP / "out"; s78.OUT.mkdir(); s78.CACHE = mc.CACHE_DIR
s78.S68_EXPORT = TMP / "no_such_export.csv"     # C falls back to its own fit
s61, s67, j47, h47 = s78.load_modules()
s61.OUT = s78.OUT; s67.OUT = s78.OUT

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo = s78.exposure(frames[2019], book, spec, j47, s61)
Q4F = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
EDU, CUT = s78.skill_cut(frames[2019], j47, h47)
HIEDU = set(EDU.loc[EDU["high_edu"] == 1, "employer_id"].astype(int))
STEP1 = float(np.log(1.12))          # the tightening rise
STEP2 = float(np.log(0.78))          # the adoption step, from that level
DIFF = float(np.log(0.80))           # young women's extra step
SEAS = float(np.log(1.25))           # Q4 bump in every year
DRIFT = 0.012                        # per month, pre-launch, DRIFT world
SENIOR = float(np.log(1.35))         # the 65-69 gain, SENIOR world
# The stock pools the sexes, so its planted step is the men's step plus
# the women's extra fall averaged over an equal split: what the 6-band
# fits (D, F, G) must return. B reads the sexes apart and sees STEP2.
STEP2_POOLED = STEP2 + float(np.log((1.0 + np.exp(DIFF)) / 2.0))
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


check("the fixture's DAIOE top quartile sits inside its post-secondary quartile",
      Q4F and Q4F <= HIEDU and len(HIEDU) > len(Q4F),
      f"Q4 {len(Q4F)} firms, High_edu {len(HIEDU)} firms, cut {CUT:.2f}")


def counts_by_sex(world: str, years=None) -> pd.DataFrame:
    """Employer x month x band x sex counts, seven bands (the oldest split
    at 65), with the world's planted effects."""
    rng = np.random.default_rng({"step": 78, "drift": 79, "skill": 80,
                                 "senior": 81}[world])
    # Larger young cells than the other tests use: every draw carries a
    # +1 floor so no cell is dead, and on a cell of four or five that floor
    # shrinks a proportional effect by a fifth. At twelve per sex it is
    # under a tenth, which is what the tolerances below assume.
    lam0 = {"22-25": 24, "26-30": 20, "31-34": 7, "35-40": 9, "41-49": 12,
            "50-64": 14, "65-69": 8}
    hit_set = HIEDU if world == "skill" else Q4F
    young_hit = world != "senior"
    years = years or s61.PANEL_YEARS
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        hit = emp in hit_set
        for y in years:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                q = (m - 1) // 3 + 1
                t = (y - 2021) * 12 + (m - 1)
                for age, lam in lam0.items():
                    for g in ("1", "2"):
                        lam_ = lam / 2
                        if hit and young_hit and age == "22-25":
                            if q == 4:
                                lam_ *= np.exp(SEAS)
                            if ym >= mc.RIKSBANK_YM:
                                lam_ *= np.exp(STEP1)
                            if ym >= s78.POST_FROM:
                                lam_ *= np.exp(STEP2)
                                if g == "2":
                                    lam_ *= np.exp(DIFF)
                            if world == "drift":
                                lam_ *= np.exp(DRIFT * min(t, 22))
                        if (hit and world == "senior" and age == "65-69"
                                and ym >= s78.POST_FROM):
                            lam_ *= np.exp(SENIOR)
                        rows.append((emp, ym, age, g, int(rng.poisson(lam_)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month", "age_group",
                                       "gender", "n_emp"])


def six_bands(df: pd.DataFrame) -> pd.DataFrame:
    """47L's six bands: 50-64 and 65-69 summed into 50+."""
    d = df.copy()
    d["age_group"] = d["age_group"].replace({"50-64": "50+", "65-69": "50+"})
    return d


def collapse(sex: pd.DataFrame) -> pd.DataFrame:
    return (sex.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
            .sum().reset_index())


def install(sex7: pd.DataFrame, years, split: bool):
    """Write the caches a run reads: L_counts (six bands), L_counts_sex
    (six bands by sex) and, if asked, L_counts_split (seven bands)."""
    for f in mc.CACHE_DIR.glob("L_counts*.parquet"):
        f.unlink()
    sex6 = six_bands(sex7)
    c6, c7 = collapse(sex6), collapse(sex7)
    for y in years:
        ys = str(y)
        c6[c6["year_month"].str.slice(0, 4) == ys].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
        sex6[sex6["year_month"].str.slice(0, 4) == ys].to_parquet(
            mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)
        if split:
            c7[c7["year_month"].str.slice(0, 4) == ys].to_parquet(
                mc.CACHE_DIR / f"L_counts_split_{y}.parquet", index=False)
    return c6


# ---- the industry stub: LISA's firm table, ten employers without a code -
FIRMS = list(range(1, FIX.n_firms + 1))
NO_CODE = set(FIRMS[-10:])
IND = {f: f"{100 + (f % 5) * 7:03d}" for f in FIRMS}
CATALOGUE = pd.DataFrame(
    [("Ftg_2019", c, "varchar") for c in ("LopNr_PeOrgNr", "Org_Sni2007")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
SQL_CALLS = []


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    SQL_CALLS.append(ql[:400])
    if "information_schema" in ql:
        return CATALOGUE.copy()
    if "ftg_2019" in ql:
        keep = [f for f in FIRMS if f not in NO_CODE]
        return pd.DataFrame({"employer_id": keep,
                             "ind": [IND[f] + "42" for f in keep]})
    raise AssertionError(f"unexpected query: {q[:100]}")


pd.read_sql = fake_read_sql
mc.connect = lambda: object()

# ======================================================================
# DRIFT world, drawn from 2019: A(ii) must see the slope, A(i) must reach
# back to 2019Q1 when the early caches are there
# ======================================================================
YEARS_D = [2019, 2020] + list(s61.PANEL_YEARS)
SEX_D = counts_by_sex("drift", YEARS_D)
C_D = install(SEX_D, YEARS_D, split=False)
early = s78.load_counts("L_counts", [2019, 2020])
s78.YOUNG_BANDS = ["22-25"]
s78.NOTES.clear()
path_d, drift_d = s78.part_a(C_D[C_D["year_month"] >= "2021-01"], expo, s61, j47,
                             extended=C_D)
D = pd.DataFrame(drift_d).set_index("term")
c, s = float(D.loc["trend_x_high_x_young", "coef"]), float(D.loc["trend_x_high_x_young", "se"])
check("DRIFT world: the pre-launch trend is recovered",
      abs(c - DRIFT) < max(0.004, 2 * s), f"trend {c:+.4f} ({s:.4f}) against planted {DRIFT:+.4f}")
check("DRIFT world: the rule reads NOT FLAT", abs(c) > s78.DRIFT_RULE_SE * s,
      f"t {c/max(s,1e-12):+.1f}")
P = pd.DataFrame(path_d)
check("A(i) runs from 2019Q1 when the 2019 and 2020 counts are cached",
      {"2019Q1", "2020Q4", "2021Q1", "2025Q2"} <= set(P["quarter"])
      and (P.loc[P["quarter"] == s78.REF_QUARTER, "status"] == "reference").all()
      and any("2019 and 2020 counts cached" in n for n in s78.NOTES),
      f"{len(P)} quarters")
check("the early caches were read by the same loader main() uses",
      early is not None and set(early["year_month"].str.slice(0, 4)) == {"2019", "2020"})

# ======================================================================
# STEP world: B, C, D, E, F, G and the end-to-end run
# ======================================================================
SEX_S = counts_by_sex("step")
C_S = install(SEX_S, s61.PANEL_YEARS, split=True)

# ---- A(ii) reads FLAT when nothing drifts, and A(i) falls back to 2021 --
s78.NOTES.clear()
path_s, drift_s = s78.part_a(C_S, expo, s61, j47, extended=None)
Ds = pd.DataFrame(drift_s).set_index("term")
c, s = float(Ds.loc["trend_x_high_x_young", "coef"]), float(Ds.loc["trend_x_high_x_young", "se"])
check("STEP world: the pre-launch trend is a null and reads FLAT",
      abs(c) <= s78.DRIFT_RULE_SE * s, f"trend {c:+.4f} ({s:.4f})")
check("A(ii) also returns the tightening window",
      "rbw_x_high_x_young" in Ds.index
      and abs(float(Ds.loc["rbw_x_high_x_young", "coef"]) - STEP1) < 0.08,
      f"window {float(Ds.loc['rbw_x_high_x_young','coef']):+.4f} against {STEP1:+.4f}")
Ps = pd.DataFrame(path_s)
check("A(i) without the early caches runs from 2021Q1 and says what extending it costs",
      min(Ps["quarter"]) == "2021Q1"
      and any("two more year pulls" in n for n in s78.NOTES))

# ---- B: the sex split on Equation (2) -------------------------------------
rows_b, gs = s78.part_b(expo, s61, s67, j47)
check("B: young men's adoption step is the planted step",
      "male_step" in gs and abs(gs["male_step"][0] - STEP2) < 0.07,
      f"{gs.get('male_step', ('?',))[0]:+.4f} against {STEP2:+.4f}")
check("B: the female differential is the planted extra step",
      "female_differential" in gs and abs(gs["female_differential"][0] - DIFF) < 0.07,
      f"{gs.get('female_differential', ('?',))[0]:+.4f} against {DIFF:+.4f}")
check("B: the female step is the sum and carries a standard error from the covariance",
      "female_step" in gs and gs["female_step"][1] is not None
      and abs(gs["female_step"][0] - (STEP2 + DIFF)) < 0.09,
      f"{gs.get('female_step', ('?', None))[0]:+.4f} ({gs.get('female_step', (0, 0))[1]})")
check("B: the steps from the 2023 level are on the record for both sexes",
      all(k in gs and gs[k][1] is not None
          for k in ("male_step_from_2023", "female_step_from_2023")))
check("B: every term of Equation (2) is exported three ways",
      {"rb_x_high_x_young", "rb_x_high_x_female", "rb_x_high_x_young_x_female",
       "q3_x_high_x_young_x_female", "post_x_high_x_female"}
      <= {r["term"] for r in rows_b})

# ---- C: clustering by industry changes no coefficient ---------------------
s78.NOTES.clear()
rows_c = s78.part_c(C_S, expo, s61, j47)
Cc = pd.DataFrame(rows_c)
check("C: the coefficients are identical under both clusterings to four decimals",
      len(Cc) and Cc["coef_match_4dp"].all(),
      f"{int(Cc['coef_match_4dp'].sum())} of {len(Cc)} terms")
post = Cc[Cc["term"] == "post_x_high_x_young"].iloc[0]
check("C: both standard errors are on the record and differ",
      post["se_industry"] > 0 and post["se_employer"] > 0
      and abs(post["se_industry"] - post["se_employer"]) > 1e-6,
      f"employer {post['se_employer']:.4f}, industry {post['se_industry']:.4f}")
check("C: employers without a 2019 code are their own cluster, and the count is said",
      any("10 without one are their own cluster" in n for n in s78.NOTES),
      next((n for n in s78.NOTES if "own cluster" in n), "")[:90])

# ---- D: the skill cut, where the step was planted on Q4 -------------------
rows_d, vd = s78.part_d(C_S, expo, frames[2019], s61, j47, h47)
v = vd.get("22-25", {})
check("D (STEP world): the DAIOE step alone is the planted pooled step",
      "ai_alone" in v and abs(v["ai_alone"][0] - STEP2_POOLED) < 0.07,
      f"{v.get('ai_alone', ('?',))[0]:+.4f} against {STEP2_POOLED:+.4f}")
check("D (STEP world): the skill cut alone does not reproduce a step planted on Q4",
      v.get("reproduces") is False,
      f"skill alone {v.get('skill_alone', ('?',))[0]:+.4f}")
check("D (STEP world): AI survives the skill cut", v.get("survives") is True,
      f"both in: DAIOE {v.get('ai_both', ('?',))[0]:+.4f}, skill {v.get('skill_both', ('?',))[0]:+.4f}")

# ---- F: baseline and industry fit on identical firm sets ------------------
s78.NOTES.clear()
rows_f = s78.part_f(C_S, expo, s61, j47)
F = pd.DataFrame(rows_f)
check("F: both specifications are fitted",
      set(F["spec"]) == {"baseline_same_sample", "industry_age_month"})
check("F: on identical firm sets (the employers with a 2019 code)",
      F["n_firms"].nunique() == 1 and int(F["n_firms"].iloc[0]) == FIX.n_firms - 10,
      f"n_firms {sorted(F['n_firms'].unique())}")
fb = F[(F["spec"] == "baseline_same_sample") & (F["term"] == "post_x_high_x_young")].iloc[0]
check("F: the same-sample baseline returns the planted pooled step",
      abs(fb["coef"] - STEP2_POOLED) < 0.07, f"{fb['coef']:+.4f} against {STEP2_POOLED:+.4f}")

# ---- G: April dropped, indicator from May ---------------------------------
s78.NOTES.clear()
rows_g = s78.part_g(C_S, expo, s61, j47)
G = pd.DataFrame(rows_g).set_index("term")
full_n = int(Cc["n_clusters_industry"].iloc[0]) and int(F[F["spec"] == "baseline_same_sample"]["n_obs"].iloc[0])
skel_full = s61.build_skeleton(C_S, "22-25", j47)
n_april = int((skel_full.merge(expo[["employer_id", "fq"]], on="employer_id")["year_month"] == mc.RIKSBANK_YM).sum())
check("G: April 2022 is dropped from the panel, and the note says how many cells",
      int(G.loc["post_x_high_x_young", "n_obs"]) == len(skel_full.merge(expo[["employer_id", "fq"]], on="employer_id")) - n_april
      and any(f"{n_april:,} cells of {mc.RIKSBANK_YM} dropped" in n for n in s78.NOTES),
      f"n_obs {int(G.loc['post_x_high_x_young','n_obs']):,}, April cells {n_april:,}")
check("G: the tightening rise and the step survive the May boundary",
      abs(float(G.loc["rb_x_high_x_young", "coef"]) - STEP1) < 0.08
      and abs(float(G.loc["post_x_high_x_young", "coef"]) - STEP2_POOLED) < 0.07,
      f"rb {float(G.loc['rb_x_high_x_young','coef']):+.4f}, post {float(G.loc['post_x_high_x_young','coef']):+.4f}")
check("G: the boundary is recorded on every row",
      all(r["boundary"] == s78.MAY_BOUNDARY for r in rows_g))

# ======================================================================
# SENIOR world: the split profile puts the senior gain in 65-69
# ======================================================================
install(counts_by_sex("senior"), s61.PANEL_YEARS, split=True)
n_sql = len(SQL_CALLS)
rows_e = s78.part_e(expo, s61, j47)
E = pd.DataFrame(rows_e).set_index("band")
check("E: the seven bands are on the record with 41-49 as the reference",
      set(E.index) == set(s78.SPLIT_BANDS) and E.loc["41-49", "status"] == "reference")
check("E: the planted 65-69 gain is recovered in 65-69",
      "65-69" in E.index and abs(float(E.loc["65-69", "coef"]) - SENIOR) < 0.08,
      f"65-69 {float(E.loc['65-69','coef']):+.4f} against {SENIOR:+.4f}")
check("E: and not in 50-64",
      "50-64" in E.index and abs(float(E.loc["50-64", "coef"])) < 2 * float(E.loc["50-64", "se"]) + 0.03,
      f"50-64 {float(E.loc['50-64','coef']):+.4f} ({float(E.loc['50-64','se']):.4f})")
check("E: nothing was planted on the young and none is found",
      abs(float(E.loc["22-25", "coef"])) < 2 * float(E.loc["22-25", "se"]) + 0.03,
      f"22-25 {float(E.loc['22-25','coef']):+.4f} ({float(E.loc['22-25','se']):.4f})")
check("E: the split counts came from the cache, no pull was made",
      len(SQL_CALLS) == n_sql)

# ======================================================================
# SKILL world: the step planted on every post-secondary employer
# ======================================================================
C_K = collapse(six_bands(counts_by_sex("skill")))
_, vk = s78.part_d(C_K, expo, frames[2019], s61, j47, h47)
v = vk.get("22-25", {})
check("D (SKILL world): the skill cut alone reproduces the step",
      v.get("reproduces") is True,
      f"DAIOE alone {v.get('ai_alone', ('?',))[0]:+.4f}, skill alone {v.get('skill_alone', ('?',))[0]:+.4f}")
check("D (SKILL world): the skill cut absorbs the DAIOE step",
      v.get("survives") is False,
      f"both in: DAIOE {v.get('ai_both', ('?',))[0]:+.4f}, skill {v.get('skill_both', ('?',))[0]:+.4f}")

# ======================================================================
# end to end, off the STEP world's caches (2021 to 2025, split cached)
# ======================================================================
install(SEX_S, s61.PANEL_YEARS, split=True)
for f in s78.OUT.glob("*.csv"):
    f.unlink()
s78.NOTES.clear(); s78.FAILURES.clear()
n_sql_before = len(SQL_CALLS)
s78.main()
for nm in ("prepath_plain.csv", "predrift.csv", "gender_eq2.csv",
           "cluster_industry.csv", "skill_placebo.csv", "prof_split.csv",
           "industry_seasonal.csv", "boundary_may.csv", "78_summary.txt"):
    check(f"main() writes {nm}", (s78.OUT / nm).exists())
check("main() made no SQL call beyond the industry read",
      all("information_schema" in q or "ftg_2019" in q for q in SQL_CALLS[n_sql_before:]),
      f"{len(SQL_CALLS) - n_sql_before} calls")
summ = (s78.OUT / "78_summary.txt").read_text(encoding="utf-8")
for must in ("FLAT", "not identified from two years", "young women minus young men",
             "step from the 2023 level", "four decimals: YES",
             "AI SURVIVES THE SKILL CUT", "50 AND OVER SPLIT AT 65",
             "retained share of the adoption step", "TIGHTENING BOUNDARY AT 2022-05",
             "two more year pulls", "READ THIS BEFORE QUOTING"):
    check(f"the summary states {must!r}", must in summ)
check("no fit failed in the end-to-end run", not s78.FAILURES, str(s78.FAILURES))
check("the covariance files left with the outputs",
      len(list(s78.OUT.glob("vcov_s78_*.csv"))) >= 8,
      f"{len(list(s78.OUT.glob('vcov_s78_*.csv')))} files")

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

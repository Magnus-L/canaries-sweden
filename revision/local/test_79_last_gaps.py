#!/usr/bin/env python3
"""
test_79_last_gaps.py -- each of the three parts must find what was planted,
                        in the cell it was planted in.

One synthetic world on the fixture's 140 employers (the top DAIOE quartile
Q4 about a third of them), with a Q4 seasonal bump in exposed firms' young
cells so the calendar terms have work to do, and three separate plants:

  22-25, by sex   exposed employment rises by a known step from April 2022
                  and falls by a known step from January 2024; young women
                  in those cells fall by a further known amount. Part A
                  must return the male step, the female differential and
                  their sum, and must return THE SAME COEFFICIENTS under
                  employer and industry clustering with two different
                  standard errors. That is the mechanism: clustering moves
                  the covariance and nothing else.
  26-30, pooled   exposed employment falls by a known step from January
                  2024. Part B must recover it on the same-sample baseline
                  and must fit the baseline and the industry specification
                  on identical firm sets, since a retained share is only a
                  retained share if the samples are the same.
  declarations    a planted set of workers who reach none of the three
                  individual registers, concentrated in exposed firms at
                  22-25 in 2024 and 2025. Part C must place them in the
                  right band and the right exposure quartile, must put
                  employers the score book never scored in their own group,
                  and must suppress a cell of three person-months.

The age branch is tested both ways: with a birth-year column in the
declaration catalogue, when the bands must come back, and without one,
when every row must fall in one band called 'all' and the summary must say
that the share is unbroken by age. The delivery has no such column, which
is why the script probes for one rather than assuming either.

Also tested: the industry loader's own-cluster rule for employers with no
2019 code; the employer-clustered comparison read from lane 25's export
rather than refitted, and the one fit that then does not run; the export
floor; and main() end to end off the caches, with a summary that states
every read rule.

    CANARIES_DRYRUN=1 python3 revision/local/test_79_last_gaps.py
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
os.environ["CANARIES_79_PARTS"] = "ABC"
# The script's Tee caps the terminal echo at 2 KB, which is right under
# BatchClient's blocking pipe and wrong here: it would swallow the check
# lines printed after main(). The lane runner lifts it the same way.
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries79_"))
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


s79 = load("79_last_gaps.py", "s79")
s79.OUT = TMP / "out"; s79.OUT.mkdir(); s79.CACHE = mc.CACHE_DIR
s79.LANE25_ROOT = TMP                 # lane 25's folders live in the sandbox
s61, s67, s78, j47, h47 = s79.load_modules()
s61.OUT = s67.OUT = s79.OUT

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
BANDS6 = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


check("the fixture puts some employers in the top DAIOE quartile",
      0 < len(Q4F) < FIX.n_firms, f"{len(Q4F)} of {FIX.n_firms} firms")


# ======================================================================
# the panel: employer x band x sex x month counts with the three plants
# ======================================================================

def counts_by_sex() -> pd.DataFrame:
    rng = np.random.default_rng(79)
    # Cells large enough that the +1 floor on every draw does not shrink a
    # proportional effect: at twelve per sex the floor is under a tenth.
    lam0 = {"22-25": 24, "26-30": 20, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        hit = emp in Q4F
        for y in s61.PANEL_YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                q = (m - 1) // 3 + 1
                for age, lam in lam0.items():
                    for g in ("1", "2"):
                        lam_ = lam / 2
                        if hit and age == "22-25":
                            if q == 4:
                                lam_ *= np.exp(SEAS)
                            if ym >= mc.RIKSBANK_YM:
                                lam_ *= np.exp(STEP1)
                            if ym >= s79.POST_FROM:
                                lam_ *= np.exp(STEP2)
                                if g == "2":
                                    lam_ *= np.exp(DIFF)
                        if hit and age == "26-30":
                            if q == 4:
                                lam_ *= np.exp(SEAS)
                            if ym >= s79.POST_FROM:
                                lam_ *= np.exp(STEP_B)
                        rows.append((emp, ym, age, g, int(rng.poisson(lam_)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month", "age_group",
                                       "gender", "n_emp"])


def collapse(sex: pd.DataFrame) -> pd.DataFrame:
    return (sex.groupby(["employer_id", "year_month", "age_group"])["n_emp"]
            .sum().reset_index())


SEX = counts_by_sex()
COUNTS = collapse(SEX)
for y in s61.PANEL_YEARS:
    ys = str(y)
    COUNTS[COUNTS["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
    SEX[SEX["year_month"].str.slice(0, 4) == ys].to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)


# ======================================================================
# the stub database: LISA's firm table, the declaration catalogue and the
# declarations themselves
# ======================================================================
FIRMS = list(range(1, FIX.n_firms + 1))
NO_CODE = set(FIRMS[-10:])                 # ten employers with no 2019 code
IND = {f: f"{100 + (f % 5) * 7:03d}" for f in FIRMS}
UNSCORED = [901, 902, 903]                 # in the declarations, never scored
LISA_CATALOGUE = pd.DataFrame(
    [("Ftg_2019", c, "varchar") for c in ("LopNr_PeOrgNr", "Org_Sni2007")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
AGI_COLS_PLAIN = ["P1207_LOPNR_PEORGNR", "P1207_LOPNR_PERSONNR", "PERIOD",
                  "KONTANT_ERSATTNING_ULAG_AG", "ASTNR", "FORSTA_ANSTALLD"]


def agi_catalogue(with_age: bool) -> pd.DataFrame:
    cols = AGI_COLS_PLAIN + (["FodelseAr"] if with_age else [])
    rows = [(s79.agi_table(y, 1), c, "varchar")
            for y in s79.UNCOUNTED_YEARS for c in cols]
    return pd.DataFrame(rows, columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])


WITH_AGE = [True]          # flipped by the second Part C run
GHOST_FLOOR_BAND = "50+"   # where the three-person-month cell is planted


def declarations() -> pd.DataFrame:
    """
    Person-months in the employer declarations, by employer, by what the
    registers can say about the worker, and by the band the worker is in.

    The plant: workers who reach no register are concentrated in exposed
    firms at 22-25 in 2024 and 2025, which is the editor's mechanism drawn
    as data. Employers 901 to 903 never appear in the 2019 education mix,
    so they are the unscored group; three of their 2019 person-months in
    the oldest band reach no register, which is the cell the export floor
    has to suppress.
    """
    rng = np.random.default_rng(790)
    rows = []
    for y in s79.UNCOUNTED_YEARS:
        months = 12 if y < 2025 else 6
        new = y in s79.NEW_YEARS
        for emp in FIRMS + UNSCORED:
            q4 = emp in Q4F
            for band in BANDS6:
                rows.append((y, emp, "counted", band,
                             int(rng.poisson(18)) + 6))
                rows.append((y, emp, "register_no_birth_or_sex", band,
                             int(rng.poisson(1))))
                rows.append((y, emp, "outside_age_range", band,
                             int(rng.poisson(2))))
                if emp in UNSCORED:
                    gh = 0                      # kept clean for the floor cell
                elif new:
                    gh = {(True, True): 6.0, (True, False): 1.0,
                          (False, True): 2.0,
                          (False, False): 0.5}[(q4, band == "22-25")]
                else:
                    gh = 0.2
                rows.append((y, emp, "no_register", band,
                             int(rng.poisson(gh * months / 12))))
    d = pd.DataFrame(rows, columns=["year", "employer_id", "status", "band", "n"])
    # the floor cell: exactly three person-months, one per unscored employer
    for emp in UNSCORED:
        d.loc[len(d)] = (2019, emp, "no_register", GHOST_FLOOR_BAND, 1)
    return d[d["n"] > 0].reset_index(drop=True)


DECL = declarations()
SQL_CALLS = []


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    SQL_CALLS.append(ql[:400])
    if "information_schema.columns" in ql and "table_name in (" in ql:
        return agi_catalogue(WITH_AGE[0])
    if "information_schema" in ql:
        return LISA_CATALOGUE.copy()
    if "ftg_2019" in ql:
        keep = [f for f in FIRMS if f not in NO_CODE]
        return pd.DataFrame({"employer_id": keep,
                             "ind": [IND[f] + "42" for f in keep]})
    if "arb_agiindivid" in ql:
        # The server answers the query as written: grouped by band when the
        # query builds a band, and not when it does not.
        year = int(re.search(r"arb_agiindivid(\d{4})", ql).group(1))
        d = DECL[DECL["year"] == year]
        if "decl_age between" in ql:
            g = (d.groupby(["employer_id", "status", "band"], as_index=False)
                 ["n"].sum().rename(columns={"band": "decl_band"}))
        else:
            g = d.groupby(["employer_id", "status"], as_index=False)["n"].sum()
            g["decl_band"] = "all"
        return g.rename(columns={"n": "n_personmonths"})[
            ["employer_id", "status", "decl_band", "n_personmonths"]]
    raise AssertionError(f"unexpected query: {str(q)[:120]}")


pd.read_sql = fake_read_sql
mc.connect = lambda: object()

# Count the fits, so "read lane 25's export rather than refit" is checked by
# the fit that does not happen and not by a log line.
FITS = []
_fit = s79.fit


def counting_fit(b, tag, *a, **kw):
    FITS.append(tag)
    return _fit(b, tag, *a, **kw)


s79.fit = counting_fit

# ======================================================================
# A. the sexes, clustered by industry, first with no lane 25 export
# ======================================================================
s79.NOTES.clear(); s79.FAILURES.clear(); FITS.clear()
rows_a, ga = s79.part_a(expo, s61, s67, s78, j47)
A = pd.DataFrame(rows_a)
check("A: both fits ran when lane 25's export is not on the share",
      len(FITS) == 2 and any("clemp" in t for t in FITS), str(FITS))
check("A: young men's adoption step is the planted step",
      "male_step" in ga and abs(ga["male_step"][0] - STEP2) < 0.07,
      f"{ga.get('male_step', ('?',))[0]:+.4f} against {STEP2:+.4f}")
check("A: the female differential is the planted extra step",
      "female_differential" in ga and abs(ga["female_differential"][0] - DIFF) < 0.07,
      f"{ga.get('female_differential', ('?',))[0]:+.4f} against {DIFF:+.4f}")
check("A: the women's step is the sum and carries a standard error under BOTH "
      "clusterings",
      "female_step" in ga and ga["female_step"][1] is not None
      and ga["female_step"][2] is not None
      and abs(ga["female_step"][0] - (STEP2 + DIFF)) < 0.09,
      f"{ga.get('female_step', ('?',))[0]:+.4f} employer "
      f"{ga.get('female_step', (0, 0, 0))[1]} industry "
      f"{ga.get('female_step', (0, 0, 0))[2]}")
check("A: clustering moves no coefficient (identical to four decimals)",
      len(A) and A["coef_match_4dp"].all(),
      f"{int(A['coef_match_4dp'].sum())} of {len(A)} terms")
check("A: the reproduction rule is recorded as met", ga.get("reproduces") is True)
post = A[A["term"] == "post_x_high_x_young_x_female"].iloc[0]
check("A: both standard errors are on the record and differ",
      post["se_industry"] > 0 and post["se_employer"] > 0
      and abs(post["se_industry"] - post["se_employer"]) > 1e-6,
      f"employer {post['se_employer']:.4f}, industry {post['se_industry']:.4f}")
check("A: every term of Equation (2) is exported three ways",
      {"rb_x_high_x_young", "rb_x_high_x_female", "rb_x_high_x_young_x_female",
       "q3_x_high_x_young_x_female", "interim_x_high_x_female"}
      <= set(A["term"]))
check("A: employers without a 2019 code are their own cluster, and it is said",
      any("10 without one are their own cluster" in n for n in s79.NOTES),
      next((n for n in s79.NOTES if "own cluster" in n), "")[:100])
check("A: the cluster count is the industries plus those employers",
      int(A["n_clusters_industry"].iloc[0]) == len(set(IND.values())) + 10,
      f"{int(A['n_clusters_industry'].iloc[0])} clusters")
check("A: the covariance left with the exports",
      (s79.OUT / "vcov_s79_gender_clind_22_25.csv").exists()
      and (s79.OUT / "gender_cluster_industry.csv").exists())

# ---- the same part with lane 25's export in place -------------------------
L25 = TMP / "output_78b"; L25.mkdir()
(A.assign(young_band=s79.YOUNG_SEX)
 .rename(columns={"coef_employer_run": "coef", "se_employer": "se"})
 [["young_band", "term", "coef", "se", "n_obs", "status"]]
 .to_csv(L25 / "gender_eq2.csv", index=False))
shutil.copy(s79.OUT / "vcov_s79_gender_clemp_22_25.csv",
            L25 / "vcov_s78_gender_eq2_22_25.csv")
s79.NOTES.clear(); FITS.clear()
rows_a2, ga2 = s79.part_a(expo, s61, s67, s78, j47)
A2 = pd.DataFrame(rows_a2)
check("A: with lane 25's export on the share only one fit runs",
      len(FITS) == 1 and "clind" in FITS[0], str(FITS))
check("A: and the note names the file it read",
      any("output_78b/gender_eq2.csv" in n for n in s79.NOTES),
      next((n for n in s79.NOTES if "read from" in n), "")[:100])
check("A: the read numbers reproduce the refitted ones to four decimals",
      A2["coef_match_4dp"].all()
      and np.allclose(A2["se_employer"], A["se_employer"], atol=1e-9))
check("A: the women's step keeps its employer-clustered standard error from "
      "the covariance beside the export",
      ga2["female_step"][1] is not None
      and abs(ga2["female_step"][1] - ga["female_step"][1]) < 1e-9)

# ======================================================================
# B. industry x age x month with the calendar terms, at 26-30
# ======================================================================
s79.NOTES.clear(); FITS.clear()
rows_b = s79.part_b(COUNTS, expo, s61, s78, j47)
B = pd.DataFrame(rows_b)
check("B: both specifications are fitted at 26-30",
      set(B["spec"]) == {"baseline_same_sample", "industry_age_month"}
      and set(B["young_band"]) == {"26-30"})
check("B: on identical firm sets (the employers with a 2019 code)",
      B["n_firms"].nunique() == 1 and int(B["n_firms"].iloc[0]) == FIX.n_firms - 10,
      f"n_firms {sorted(B['n_firms'].unique())}")
check("B: and on identical panels (the same number of cells in both fits)",
      B.groupby("spec")["n_obs"].nunique().eq(1).all()
      and B["n_obs"].nunique() == 1,
      f"n_obs {sorted(B['n_obs'].unique())}")
bb = B[(B["spec"] == "baseline_same_sample")
       & (B["term"] == "post_x_high_x_young")].iloc[0]
bi = B[(B["spec"] == "industry_age_month")
       & (B["term"] == "post_x_high_x_young")].iloc[0]
check("B: the same-sample baseline returns the planted 26-30 step",
      abs(bb["coef"] - STEP_B) < 0.07, f"{bb['coef']:+.4f} against {STEP_B:+.4f}")
check("B: the industry specification returns it too, since exposure and "
      "industry are unrelated in the fixture",
      abs(bi["coef"] - STEP_B) < 0.09, f"{bi['coef']:+.4f} against {STEP_B:+.4f}")
check("B: the retained share is exported, 1 for the baseline and the ratio "
      "for the industry fit",
      float(B[B["spec"] == "baseline_same_sample"]["retained_share"].iloc[0]) == 1.0
      and abs(float(bi["retained_share"]) - bi["coef"] / bb["coef"]) < 1e-9,
      f"retained {float(bi['retained_share']):.3f}")
check("B: the exports are on disk",
      (s79.OUT / "industry_seasonal_2630.csv").exists()
      and (s79.OUT / "vcov_s79_indseas_ind_26_30.csv").exists())

# ======================================================================
# C. the uncounted payslips, with the declaration carrying a birth year
# ======================================================================
s79.NOTES.clear()
tab, info = s79.part_c(expo)
check("C: the age probe found the declaration's birth year",
      info["age_col"] == "FodelseAr" and info["age_kind"] == "birth",
      str(info["age_col"]))
check("C: the bands come back and the six panel bands are among them",
      set(BANDS6) <= set(tab["decl_band"]), str(sorted(set(tab["decl_band"]))))
check("C: the quartile groups include the top quartile, the pooled lower "
      "three and the employers the score book never scored",
      {"Q4", "Q1-Q3", "unscored", "all"} <= set(tab["quartile_group"]),
      str(sorted(set(tab["quartile_group"]))))


def planted_share(year, emps, band=None) -> float:
    d = DECL[(DECL["year"] == year) & (DECL["employer_id"].isin(emps))]
    if band:
        d = d[d["band"] == band]
    tot = d["n"].sum()
    return float(d.loc[d["status"] == "no_register", "n"].sum()) / float(tot)


def got(year, group, band) -> float:
    r = tab[(tab["year"] == year) & (tab["quartile_group"] == group)
            & (tab["decl_band"] == band)]
    return float(r["share_no_register"].iloc[0]) if len(r) else float("nan")


want = planted_share(2024, Q4F, "22-25")
check("C: the planted ghosts land in the right band and the right quartile",
      abs(got(2024, "Q4", "22-25") - want) < 1e-9,
      f"exported {got(2024, 'Q4', '22-25'):.4%} against planted {want:.4%}")
low = set(FIRMS) - Q4F
check("C: and the lower quartiles carry the smaller planted share",
      abs(got(2024, "Q1-Q3", "22-25") - planted_share(2024, low, "22-25")) < 1e-9
      and got(2024, "Q4", "22-25") > got(2024, "Q1-Q3", "22-25"),
      f"Q4 {got(2024, 'Q4', '22-25'):.3%} against Q1-Q3 "
      f"{got(2024, 'Q1-Q3', '22-25'):.3%}")
check("C: the unscored employers are their own group, not dropped",
      abs(got(2024, "unscored", "22-25")
          - planted_share(2024, UNSCORED, "22-25")) < 1e-9)
check("C: 2019 to 2023 carry the comparison and a smaller share",
      all(y in set(tab["year"]) for y in range(2019, 2024))
      and got(2023, "all", "all") < got(2024, "all", "all"),
      f"2023 {got(2023, 'all', 'all'):.3%}, 2024 {got(2024, 'all', 'all'):.3%}")
floor_cell = tab[(tab["year"] == 2019) & (tab["quartile_group"] == "unscored")
                 & (tab["decl_band"] == GHOST_FLOOR_BAND)]
check("C: a cell of three person-months is suppressed, and its share with it",
      len(floor_cell) == 1 and pd.isna(floor_cell["n_no_register"].iloc[0])
      and pd.isna(floor_cell["share_no_register"].iloc[0]),
      f"planted 3, exported {floor_cell['n_no_register'].iloc[0] if len(floor_cell) else 'no row'}")
full = tab[(tab["year"] == 2024) & (tab["quartile_group"] == "all")
           & (tab["decl_band"] == "all")].iloc[0]
check("C: the four statuses exhaust the declarations",
      int(full["n_total"]) == int(full[["n_counted", "n_no_register",
                                        "n_register_no_birth_or_sex",
                                        "n_outside_age_range"]].sum()),
      f"{int(full['n_total']):,} person-months in 2024")
check("C: and the total is the planted total",
      int(full["n_total"]) == int(DECL[DECL["year"] == 2024]["n"].sum()))

# ---- the same part when the declaration carries no age ------------------
WITH_AGE[0] = False
s79.NOTES.clear()
tab2, info2 = s79.part_c(expo)
check("C: with no age column in the declaration, none is used",
      info2["age_col"] is None and "unbroken by age" in info2["note"],
      info2["note"][:100])
check("C: and every row falls in one band called 'all'",
      set(tab2["decl_band"]) == {"all"}, str(sorted(set(tab2["decl_band"]))))
check("C: the quartile shares are unchanged by losing the band dimension",
      abs(float(tab2[(tab2["year"] == 2024) & (tab2["quartile_group"] == "Q4")]
                ["share_no_register"].iloc[0])
          - planted_share(2024, Q4F)) < 1e-9)
check("C: the two branches cached separately, so neither reads the other",
      len(list(mc.CACHE_DIR.glob("U_uncounted_2024_*.parquet"))) == 2,
      str(sorted(p.name for p in mc.CACHE_DIR.glob("U_uncounted_2024_*"))))

# ======================================================================
# end to end
# ======================================================================
WITH_AGE[0] = True
for f in s79.OUT.glob("*.csv"):
    f.unlink()
s79.NOTES.clear(); s79.FAILURES.clear(); FITS.clear()
n_sql = len(SQL_CALLS)
s79.main()
for nm in ("gender_cluster_industry.csv", "industry_seasonal_2630.csv",
           "uncounted_share.csv", "79_summary.txt"):
    check(f"main() writes {nm}", (s79.OUT / nm).exists())
check("main() made no SQL call beyond the industry read, the catalogue probe "
      "and the declarations",
      all("information_schema" in q or "ftg_2019" in q or "arb_agiindivid" in q
          for q in SQL_CALLS[n_sql:]), f"{len(SQL_CALLS) - n_sql} calls")
check("main() read the cached declarations rather than pulling them again",
      not any("arb_agiindivid" in q and "percell" in q
              for q in SQL_CALLS[n_sql:]),
      f"{sum('percell' in q for q in SQL_CALLS[n_sql:])} pulls")
check("main() ran three fits: two for B and one for A, which read lane 25",
      len(FITS) == 3, str(FITS))
summ = (s79.OUT / "79_summary.txt").read_text(encoding="utf-8")
for must in ("READ RULES, FIXED BEFORE THE RUN",
             "coefficients equal the employer-clustered run to 4 decimals: YES",
             "young women minus young men", "young women, adoption step",
             "industry SE", "INDUSTRY x AGE x MONTH EFFECTS WITH THE CALENDAR "
             "TERMS, 26-30 stock",
             "retained share of the adoption step",
             "THE PAYSLIPS THE PANEL NEVER COUNTS",
             "no register", "STATED PLAINLY", "AND WHEN QUOTING"):
    check(f"the summary states {must!r}", must in summ)
check("the summary names the band the declaration puts the worker in",
      "22-25" in summ and "Q1-Q3" in summ)
check("no part failed in the end-to-end run", not s79.FAILURES, str(s79.FAILURES))
check("the covariance files left with the outputs",
      len(list(s79.OUT.glob("vcov_s79_*.csv"))) >= 3,
      f"{len(list(s79.OUT.glob('vcov_s79_*.csv')))} files")
floored = pd.read_csv(s79.OUT / "uncounted_share.csv")
small = floored[[c for c in floored.columns if c.startswith("n_")]]
check("nothing between one and four leaves MONA",
      not ((small > 0) & (small < s79.FLOOR)).any().any(),
      f"{int(small.isna().sum().sum())} suppressed cells")

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

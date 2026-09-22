#!/usr/bin/env python3
"""
test_73_industry_credit.py -- both parts claim to RULE SOMETHING OUT, so
                              both must be shown to rule it in when it is
                              there.

A specification that always returns "survives" rules out nothing. So each
part is run against two planted worlds:

  industry  CONFOUNDED: the young decline hits every firm in the
            industries that happen to be exposure-heavy, so the baseline
            sees an effect and industry x age x month must kill it.
            CLEAN: the decline follows exposure across industries, and it
            must survive.

  leverage  MONETARY: the decline follows leverage, which is correlated
            with exposure, so the baseline sees an effect and the credit
            term must take it. AI: the decline follows exposure and the
            exposure term must keep it.

    CANARIES_DRYRUN=1 python3 revision/local/test_73_industry_credit.py
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
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries73_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)
sys.path.insert(0, str(HERE))
from _fixtures import Fixture, AGES  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s73 = load("73_industry_and_credit.py", "s73")
s73.OUT = TMP / "out"; s73.OUT.mkdir(); s73.CACHE = mc.CACHE_DIR
j47 = s73._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


YEARS = [2021, 2022, 2023, 2024, 2025]
fx = Fixture(mc, h47, n_firms=900, n_exposed=380)
fx.install_edu([2019], cache=mc.CACHE_DIR)
EXPO = s73.edu_exposure(j47)
EXPO["employer_id"] = s73.norm_id(EXPO["employer_id"])
HIGH = set(EXPO[EXPO["fq"] == 4]["employer_id"])
FIRMS = list(EXPO["employer_id"])
check("the fixture classifies enough firms to clear the gate",
      len(FIRMS) >= s73.MIN_FIRMS and 0 < len(HIGH) < len(FIRMS),
      f"{len(FIRMS)} firms, {len(HIGH)} high")

rng = np.random.default_rng(31)
# Eight industries. Exposed firms CONCENTRATE in the first three, so the
# industries differ sharply in exposure share and an industry-wide age
# shock will show up as an exposure effect in the baseline. But the
# concentration is only 80 per cent, so exposure still varies WITHIN
# every industry and industry x age x month is not collinear with the
# treatment. A cleanly separated fixture would make the test vacuous in
# the other direction: the FE would absorb the treatment itself.
HOT = {"100", "101", "102"}
COLD = {"103", "104", "105", "106", "107"}


def _ind(f):
    hot = (f in HIGH) == (rng.random() < 0.8)
    pool = sorted(HOT) if hot else sorted(COLD)
    return pool[int(rng.integers(0, len(pool)))]


IND = {f: _ind(f) for f in FIRMS}
# Leverage is correlated with exposure but far from collinear with it: at
# 80 per cent alignment the two post x young interactions are the same
# column to numerical precision and fixest drops one, which is a fixture
# defect that reads as a failed test.
# Continuous, as real leverage is, and correlated with exposure without
# being collinear with it. A two-point distribution made the median split
# degenerate on 21 September, which surfaced a real hole in the script.
LEV = {f: float(np.clip(rng.normal(0.55 if f in HIGH else 0.40, 0.18),
                        0.01, 1.5)) for f in FIRMS}


def counts(world, beta=-0.35, seed=4):
    """One monthly panel; only WHO declines changes between worlds."""
    r = np.random.default_rng(seed)
    lam0 = {"22-25": 9, "26-30": 10, "31-34": 9, "35-40": 11,
            "41-49": 13, "50+": 15}
    rows = []
    for f in FIRMS:
        if world == "clean":
            hit = f in HIGH
        elif world == "confounded":
            hit = IND[f] in HOT              # industry, not exposure
        elif world == "monetary":
            hit = LEV[f] >= 0.5              # leverage, not exposure
        else:
            hit = f in HIGH
        for y in YEARS:
            for m in (range(1, 13) if y < 2025 else range(1, 7)):
                ym = f"{y}-{m:02d}"
                for age in AGES:
                    lam = float(lam0[age])
                    if hit and age == "22-25" and ym >= s73.POOLED_FROM:
                        lam *= float(np.exp(beta))
                    rows.append((f, ym, age, int(r.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


# The REAL P1207 catalogue, as the 21 September probe returned it, with
# every trap that broke the guessed version:
#   * FDB_JE tables are year RANGES with an `ar` column, and the
#     alphabetically first one is 1990-1993 carrying sni69ng1, the 1969
#     industry classification;
#   * no FE_ table exists at all, so the FEK path must fall through;
#   * Serrano_bokslut names carry no `skuld` or `tillgang` anywhere;
#   * and FDB_JE, though keyed on the same identifier as AGI, is the
#     BUSINESS register rather than the employer population: at ar=2019
#     it matched 12.3% of the panel. Every Swedish employer has an
#     industry code, so the fixture gives FDB a sliver and gives the
#     LISA firm table Ftg_2019 the whole panel, which is what the real
#     registers do.
CATALOGUE = pd.DataFrame(
    [("Ftg_2019", c, "varchar") for c in
     ("LopNr_PeOrgNr", "Org_SateKommun", "Org_Sni2007")]
    # Arbst_2019 as the database ACTUALLY has it: no industry column
    # at all from 2011, whatever the mona-dictionary page says. Only the
    # pre-2011 vintages carry one.
    + [("Arbst_2019", c, "varchar") for c in
       ("P1207_LopNr_PeOrgNr", "P1207_LopNr_CfarNr", "P1207_LopNr_ArbstId",
        "AstNr", "Ast_SektorKod", "Ast_Kommun")]
    + [("Arbst_2010", c, "varchar") for c in
       ("P1207_LopNr_PeOrgNr", "P1207_LopNr_CfarNr", "P1207_LopNr_ArbstId",
        "AstNr", "Ast_SektorKod", "Ast_Kommun", "Ast_Sni2002")]
    + [("FDB_JE_1990_1993", c, "varchar") for c in
     ("P1207_Lopnr_peorgnr", "ar", "sni69ng1", "anst")]
    + [("FDB_JE_2014_2021", c, "varchar") for c in
       ("P1207_Lopnr_peorgnr", "ar", "ng1", "ng2", "ng3", "ng5", "ngs1",
        "anst", "sektor")]
    + [("FDB_JE_ALL_YEARS", c, "varchar") for c in
       ("P1207_Lopnr_peorgnr", "ar", "ng3")]
    + [("Serrano_bokslut_20230614", c, "varchar")
       for c in ("P1207_Lopnr_ORGNR", "NTOMS", "RORRESUL", "TILLGSU",
                 "EKSU", "LSKSU", "KSKSU", "EKSKSU", "RTEKOEXT",
                 "BSLSTART", "BSLSLUT")]
    + [("Serrano_bol_20230614", c, "varchar")
       for c in ("P1207_Lopnr_ORGNR", "P1207_Lopnr_WORGNR", "wstatdat",
                 "FUPLAN", "FUTYP", "status")]
    + [("Serrano_Serrano_20230614", c, "varchar")
       for c in ("P1207_Lopnr_ORGNR", "P1207_Lopnr_knc_orgnrk", "ser_year",
                 "bol_kkfall", "bol_konkurs", "ser_aktiv", "ny_skuldgrd",
                 "ny_solid", "bransch_sni073")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
FAILED = set(FIRMS[:40])


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    if "information_schema" in ql:
        return CATALOGUE.copy()
    if "fdb_je_1990_1993" in ql:
        raise AssertionError("queried the 1990-1993 table, which carries "
                             "the 1969 industry classification")
    if "ftg_2019" in ql:
        # LISA's firm table: every employer, full five-digit SNI2007
        return pd.DataFrame({"employer_id": FIRMS,
                             "ind": [IND[f] + "42" for f in FIRMS]})
    if "arbst_2019" in ql or "arbst_2010" in ql:
        # workplace file: two workplaces for the first firm, the larger
        # one in the industry the firm should end up with
        ids = list(FIRMS) + [FIRMS[0]]
        return pd.DataFrame(
            {"employer_id": ids,
             "ind": [IND[f] + "42" for f in FIRMS] + ["99999"],
             "n": [50] * len(FIRMS) + [1]})
    if "fdb_je" in ql:
        # the business register covers a sliver of the employers
        sliver = FIRMS[: max(1, len(FIRMS) // 8)]
        return pd.DataFrame({"employer_id": sliver,
                             "ind": [IND[f] for f in sliver],
                             "yr": [2019] * len(sliver)})
    if "bokslut" in ql:
        # Serrano holds one row per firm and accounting year, and BSLSLUT
        # is a SQL date. The 22 September code review found that the
        # loader's year filter silently matched nothing on a date column
        # and kept whichever row came first. So the fixture now does what
        # the register does: a 2018 close listed FIRST with a different
        # leverage, then the 2019 close. Only a loader that parses the
        # date and keeps 2019 reproduces LEV.
        import datetime as _dt
        rows = []
        for f in FIRMS:
            rows.append((f, 1.0, 1.0 - min(LEV[f] + 0.3, 0.95), _dt.date(2018, 12, 31)))
            rows.append((f, 1.0, 1.0 - LEV[f], _dt.date(2019, 12, 31)))
        d = pd.DataFrame(rows, columns=["employer_id", "assets", "equity", "yr"])
        d["dlong"] = (1.0 - d["equity"]) * 0.6
        d["dshort"] = (1.0 - d["equity"]) * 0.4
        return d
    if "serrano_serrano" in ql:
        # the purpose-built flags: 1 for the failed firms, 0 otherwise
        return pd.DataFrame(
            {"employer_id": FIRMS,
             "bol_kkfall": [0] * len(FIRMS),
             "bol_konkurs": [1 if f in FAILED else 0 for f in FIRMS]})
    if "serrano_bol" in ql:
        # THE DEFECT: `status` does not contain the word "Konkurs", so
        # the old loader returned nobody out of 989,707 firms.
        return pd.DataFrame({"employer_id": FIRMS,
                             "status": ["A"] * len(FIRMS)})
    raise AssertionError(f"unexpected query: {q[:100]}")


s73.pd.read_sql = fake_read_sql
mc.connect = lambda: object()
SCHEMA = s73.discover(None)


# ---- discovery -------------------------------------------------------
ind = s73.firm_industry(None, SCHEMA)
lev = s73.firm_leverage(None, SCHEMA)
failed = s73.firm_failed(None, SCHEMA)
LOADER_NOTES = list(s73.NOTES)
check("industry is found and reduced to three digits",
      len(ind) and set(ind["ind3"].str.len()) == {3}, f"{len(ind)} firms")
check("leverage is found", len(lev) and lev["lev"].between(0, 3).all(),
      f"{len(lev)} firms")
check("with no FEK table present, Serrano's real columns are used",
      any("EKSU" in n and "TILLGSU" in n for n in s73.NOTES),
      next((n for n in s73.NOTES if "leverage built" in n), "")[:70])
check("the loader says which accounting year it kept and how many rows",
      any("leverage year filter" in n and "2019" in n for n in s73.NOTES),
      next((n for n in s73.NOTES if "year filter" in n), "")[:90])
check("industry comes from the LISA firm table, not the business "
      "register, and covers the whole panel",
      any("Ftg_2019" in n for n in s73.NOTES) and len(ind) == len(FIRMS),
      next((n for n in s73.NOTES if "industry:" in n), "")[:90])
check("the five-digit SNI2007 is cut to the three-digit group",
      ind.set_index("employer_id")["ind3"].to_dict()
      == {f: IND[f] for f in FIRMS})

# With Ftg absent the workplace file answers, largest workplace winning;
# with both absent FDB_JE is the last resort and its thin coverage shows.
_full = SCHEMA
s73.NOTES.clear()
alt = s73.firm_industry(None, _full[_full["TABLE_NAME"] != "Ftg_2019"])
check("with Ftg_2019 absent the workplace file CANNOT answer for 2019, "
      "because Arbst carries no industry column from 2011, so FDB_JE "
      "takes it",
      any("FDB_JE_2014_2021" in n for n in s73.NOTES)
      and not any("Arbst_2019" in n for n in s73.NOTES),
      next((n for n in s73.NOTES if "industry:" in n), "")[:90])
check("and FDB_JE's thin coverage is visible, not silent",
      len(alt) < len(FIRMS) // 4, f"{len(alt)} of {len(FIRMS)}")

# an early base year is the case the workplace branch exists for
s73.NOTES.clear()
_by = s73.BASE_YEAR
s73.BASE_YEAR = 2010
try:
    early = s73.firm_industry(
        None, _full[~_full["TABLE_NAME"].isin(["Ftg_2019", "Ftg_2010"])])
    check("at an early base year the workplace file DOES answer, on the "
          "vintage column the delivery actually has",
          any("Arbst_2010" in n and "Ast_Sni2002" in n for n in s73.NOTES),
          next((n for n in s73.NOTES if "industry:" in n), "")[:90])
    check("and it says 'first workplace', not 'largest', because this "
          "delivery's Arbst carries no size column to rank by",
          any("first workplace" in n for n in s73.NOTES)
          and len(early) == len(FIRMS), f"{len(early)} firms")
finally:
    s73.BASE_YEAR = _by
s73.NOTES.clear()
ind = s73.firm_industry(None, _full)
check("leverage reproduces 1 - equity/assets FROM THE 2019 CLOSE, not the 2018 row listed first",
      abs(float(lev.set_index("employer_id").loc[FIRMS[0], "lev"])
          - LEV[FIRMS[0]]) < 1e-9)
check("failed firms come from Serrano_Serrano's bol_konkurs, not from "
      "Serrano_bol's status, which matches nobody",
      len(failed) == len(FAILED)
      and any("Serrano_Serrano" in n for n in LOADER_NOTES),
      next((n for n in LOADER_NOTES if "corporate events" in n), "")[:90])
s73.NOTES.clear()


# THE 21 SEPTEMBER FATAL BUG. L_counts supplies employer_id as float
# while main() had normalised exposure to string, so every band died on
# "merge on float64 and object columns" and the lane produced nothing.
_probe = counts("clean").copy()
_probe["employer_id"] = _probe["employer_id"].astype(float)
_sk = {k: [] for k in ("ind", "indq", "lev", "bank")}
s73.run_band(_probe, EXPO, ind, lev, failed, "22-25", j47, _sk)
check("a float employer_id in the counts still merges against string "
      "exposure, which is what killed lane 19",
      len(_sk["ind"]) > 0, f"{len(_sk['ind'])} rows")
s73.NOTES.clear(); s73.FAILURES.clear()


def run(world):
    s73.NOTES.clear(); s73.FAILURES.clear()
    cnt = counts(world)
    # mirror main()'s sinks exactly; a fixture that builds its own dict
    # silently diverges the moment a new arm is added
    sinks = {k: [] for k in ("ind", "indq", "lev", "bank")}
    s73.run_band(cnt, EXPO, ind, lev, failed, "22-25", j47, sinks)
    return (pd.DataFrame(sinks["ind"]), pd.DataFrame(sinks["lev"]),
            pd.DataFrame(sinks["bank"]), pd.DataFrame(sinks["indq"]))


def coef(df, **kw):
    d = df
    for k, v in kw.items():
        d = d[d[k] == v]
    return float(d.iloc[0]["coef"]) if len(d) else None


# ---- the quarterly industry path, added 21 Sep ------------------------
# The pooled industry fit cannot see a crossing INSIDE the post period,
# and the spreading claim is exactly that: 26-30 overtakes 22-25 during
# 2025. This arm exists to settle it, so it has to actually produce a
# path, and the verdict has to be able to go both ways.
def _pathrows(q):
    return sorted(q["period"].unique()) if len(q) else []


# ---- industry: must kill a confound and spare a real effect ----------
i_c, _, _, q_c = run("clean")
i_f, _, _, q_f = run("confounded")
b_c, a_c = coef(i_c, spec="baseline"), coef(i_c, spec="industry_age_t")
b_f, a_f = coef(i_f, spec="baseline"), coef(i_f, spec="industry_age_t")
check("CLEAN world: the baseline sees the planted decline",
      b_c is not None and b_c < -0.15, f"{b_c}")
check("CLEAN world: it SURVIVES industry x age x month",
      a_c is not None and abs(a_c) >= s73.ATTEN_MAX * abs(b_c),
      f"baseline {b_c:.4f} -> {a_c:.4f}" if None not in (b_c, a_c) else "")
check("CONFOUNDED world: the baseline is fooled",
      b_f is not None and b_f < -0.05, f"{b_f}")
check("CONFOUNDED world: industry x age x month KILLS it, so the "
      "specification rules something out",
      None not in (b_f, a_f) and abs(a_f) < s73.ATTEN_MAX * abs(b_f),
      f"baseline {b_f:.4f} -> {a_f:.4f}" if None not in (b_f, a_f) else "")


# ---- leverage: must take a credit story and spare an AI one ----------
_, l_ai, _, _ = run("clean")
_, l_mo, _, _ = run("monetary")


def term(df, t):
    d = df[df.term == t] if len(df) and "term" in df else pd.DataFrame()
    return (float(d.iloc[0]["coef"]), float(d.iloc[0]["se"])) if len(d) \
        else (None, None)


e_ai, _ = term(l_ai, "post_x_high_x_young")
e_mo, _ = term(l_mo, "post_x_high_x_young")
m_mo, sm = term(l_mo, "post_x_young_x_lev")
check("AI world: the exposure term holds up with leverage in",
      e_ai is not None and e_ai < -0.15, f"{e_ai}")
check("MONETARY world: the credit term is negative and significant",
      None not in (m_mo, sm) and m_mo / sm <= -1.96,
      f"{m_mo:.4f} (t {m_mo/sm:.1f})" if None not in (m_mo, sm) else "")
check("MONETARY world: the exposure term does NOT hold up, so the test "
      "can actually lose",
      None not in (e_ai, e_mo) and abs(e_mo) < abs(e_ai) * 0.6,
      f"AI {e_ai:.4f} vs monetary {e_mo:.4f}"
      if None not in (e_ai, e_mo) else "")


# ---- the read rule reaches the right words ---------------------------
def words(ind_rows, lev_rows):
    return " ".join(s73.verdict(pd.DataFrame(ind_rows),
                                pd.DataFrame(lev_rows), pd.DataFrame()))


surv = words([{"band": "22-25", "spec": "baseline", "coef": -0.20,
               "se": 0.02},
              {"band": "22-25", "spec": "industry_age_t", "coef": -0.18,
               "se": 0.02}], [])
dead = words([{"band": "22-25", "spec": "baseline", "coef": -0.20,
               "se": 0.02},
              {"band": "22-25", "spec": "industry_age_t", "coef": -0.02,
               "se": 0.02}], [])
check("a surviving industry test says SURVIVES", "SURVIVES" in surv)
check("a failing one says DOES NOT SURVIVE", "DOES NOT SURVIVE" in dead)

mon = words([{"band": "22-25", "spec": "baseline", "coef": -0.20,
              "se": 0.02}],
            [{"band": "22-25", "term": "post_x_high_x_young", "coef": -0.02,
              "se": 0.02},
             {"band": "22-25", "term": "post_x_young_x_lev", "coef": -0.20,
              "se": 0.02}])
ai = words([{"band": "22-25", "spec": "baseline", "coef": -0.20,
             "se": 0.02}],
           [{"band": "22-25", "term": "post_x_high_x_young", "coef": -0.19,
             "se": 0.02},
            {"band": "22-25", "term": "post_x_young_x_lev", "coef": -0.01,
             "se": 0.02}])
check("a credit-driven result reads MONETARY", "MONETARY" in mon)
check("a surviving one reads AI SURVIVES", "AI SURVIVES" in ai)
check("the verdict survives empty frames",
      isinstance(s73.verdict(pd.DataFrame(), pd.DataFrame(),
                             pd.DataFrame()), list))


# ---- a degenerate leverage split must be refused, not returned NaN ---
flat = pd.DataFrame({"employer_id": FIRMS, "lev": [0.5] * len(FIRMS)})
s73.NOTES.clear()
sk = {"ind": [], "lev": [], "bank": []}
s73.run_band(counts("clean"), EXPO, pd.DataFrame(), flat, set(), "22-25",
             j47, sk)
check("a leverage variable that cannot be split is SKIPPED and said so, "
      "rather than silently returning an unidentified term",
      not len(sk["lev"]) and any("would not be identified" in n
                                 for n in s73.NOTES),
      next((n for n in s73.NOTES if "not be identified" in n), "")[:70])
s73.NOTES.clear()


# ---- the gate must be able to refuse ---------------------------------
s73.NOTES.clear()
tiny = ind.head(3)
b = pd.DataFrame({"employer_id": FIRMS})
check("a thin covariate match is refused",
      not s73.gate(b, tiny, "probe")
      and any("BELOW THRESHOLD" in n for n in s73.NOTES))
check("a full match passes", s73.gate(b, ind, "probe2"))


# ---- end to end -------------------------------------------------------
CNT = counts("clean")
for y in YEARS:
    CNT[CNT["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
s73.PANEL_YEARS = YEARS
s73.YOUNG_BANDS = ["22-25"]
_out = sys.stdout
try:
    s73.main()
    ran = True
except BaseException as ex:  # noqa: BLE001
    ran = False; sys.stdout = _out
    print(f"      main() raised: {type(ex).__name__}: {ex}")
sys.stdout = _out
check("main() runs end to end", ran)
check("a summary is written", (s73.OUT / "73_summary.txt").exists())
if (s73.OUT / "73_summary.txt").exists():
    txt = (s73.OUT / "73_summary.txt").read_text()
    check("the summary says both covariates are frozen pre-shock",
          "frozen" in txt and "2019" in txt)
    check("the summary states what a credit test cannot do",
          "not the same as identifying an AI effect" in txt)

check("the quarterly industry path is produced, not just the pooled fit",
      len(q_c) > 0, f"{len(q_c)} quarter rows")
check("and its periods are post-ChatGPT quarters",
      all(p[:4].isdigit() and "Q" in p for p in _pathrows(q_c)),
      ", ".join(_pathrows(q_c)[:4]))

# the verdict must be able to say BOTH things, or it is not a test
import pandas as _pd
_cross = [{"band": "22-25", "period": "2025Q2", "coef": -0.05, "se": 0.01},
          {"band": "26-30", "period": "2025Q2", "coef": -0.08, "se": 0.01}]
_flat = [{"band": "22-25", "period": "2025Q2", "coef": -0.05, "se": 0.01},
         {"band": "26-30", "period": "2025Q2", "coef": -0.02, "se": 0.01}]
check("verdict detects a surviving crossing",
      any("CROSSING SURVIVES" in v for v in s73.path_verdict(_cross)))
check("verdict detects a crossing that does NOT survive",
      any("DOES NOT SURVIVE" in v for v in s73.path_verdict(_flat)))
check("and it says so rather than crashing when nothing was estimated",
      s73.path_verdict([]) == ["industry path: not estimated"])

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

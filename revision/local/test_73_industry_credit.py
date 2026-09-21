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


CATALOGUE = pd.DataFrame(
    [("FDB_JE_2019", c, "varchar") for c in ("PeOrgNr", "NgS", "Anst")]
    + [("FE_2019", c, "varchar") for c in
       ("LopNr_PeOrgNrHE", "SummaTillgangar", "SummaEgetKapital",
        "Nettoomsattning")]
    + [("Serrano_bokslut_20230614", c, "varchar")
       for c in ("ORGNR", "summa_skulder", "summa_tillgangar", "ar")]
    + [("Serrano_bol_20230614", c, "varchar") for c in ("ORGNR", "status")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])
FAILED = set(FIRMS[:40])


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    if "information_schema" in ql:
        return CATALOGUE.copy()
    if "fdb_je" in ql:
        return pd.DataFrame({"employer_id": FIRMS,
                             "ind": [IND[f] for f in FIRMS]})
    if "fe_2019" in ql:
        # equity/assets chosen so that 1 - equity/assets reproduces LEV
        return pd.DataFrame({"employer_id": FIRMS,
                             "assets": [1.0] * len(FIRMS),
                             "equity": [1.0 - LEV[f] for f in FIRMS]})
    if "bokslut" in ql:
        return pd.DataFrame({"employer_id": FIRMS,
                             "debt": [LEV[f] for f in FIRMS],
                             "assets": [1.0] * len(FIRMS),
                             "yr": [2019] * len(FIRMS)})
    if "serrano_bol" in ql:
        return pd.DataFrame({"employer_id": list(FAILED),
                             "status": ["Konkurs"] * len(FAILED)})
    raise AssertionError(f"unexpected query: {q[:100]}")


s73.pd.read_sql = fake_read_sql
mc.connect = lambda: object()
SCHEMA = s73.discover(None)


# ---- discovery -------------------------------------------------------
ind = s73.firm_industry(None, SCHEMA)
lev = s73.firm_leverage(None, SCHEMA)
failed = s73.firm_failed(None, SCHEMA)
check("industry is found and reduced to three digits",
      len(ind) and set(ind["ind3"].str.len()) == {3}, f"{len(ind)} firms")
check("leverage is found", len(lev) and lev["lev"].between(0, 3).all(),
      f"{len(lev)} firms")
check("FEK is PREFERRED over Serrano, being SCB's own and population",
      any("FE_2019" in n for n in s73.NOTES),
      next((n for n in s73.NOTES if "leverage built" in n), "")[:60])
check("and the consolidation and coverage traps are recorded, not buried",
      any("CONSOLIDATED" in n and "non-profit" in n for n in s73.NOTES))
check("leverage reproduces 1 - equity/assets",
      abs(float(lev.set_index("employer_id").loc[FIRMS[0], "lev"])
          - LEV[FIRMS[0]]) < 1e-9)
check("failed firms are found", len(failed) == len(FAILED))
s73.NOTES.clear()


def run(world):
    s73.NOTES.clear(); s73.FAILURES.clear()
    cnt = counts(world)
    sinks = {"ind": [], "lev": [], "bank": []}
    s73.run_band(cnt, EXPO, ind, lev, failed, "22-25", j47, sinks)
    return (pd.DataFrame(sinks["ind"]), pd.DataFrame(sinks["lev"]),
            pd.DataFrame(sinks["bank"]))


def coef(df, **kw):
    d = df
    for k, v in kw.items():
        d = d[d[k] == v]
    return float(d.iloc[0]["coef"]) if len(d) else None


# ---- industry: must kill a confound and spare a real effect ----------
i_c, _, _ = run("clean")
i_f, _, _ = run("confounded")
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
_, l_ai, _ = run("clean")
_, l_mo, _ = run("monetary")


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

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

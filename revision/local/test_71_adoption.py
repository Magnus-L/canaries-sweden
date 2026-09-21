#!/usr/bin/env python3
"""
test_71_adoption.py -- the gate must hold, and the survey parser must not
                       read an unknown code as "no".

71 touches three tables this project has never read, so most of what can
go wrong is parsing and gating rather than econometrics:

  * "****" is this delivery's missing marker, not a value. If to01 mapped
    it to 0 the first stage would silently be estimated against a pile of
    fabricated noes, and it would still produce a publishable-looking
    number.
  * the thresholds exist so a thin overlap cannot be talked into a result.
    A test that only ran the happy path would leave the gate unverified,
    which is the one part that protects the paper.

So both are tested directly, along with a planted first stage the
estimator must recover, and an empty catalogue it must survive.

    CANARIES_DRYRUN=1 python3 revision/local/test_71_adoption.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries71_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta",
          "eloundou_ssyk4.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL_DAIOE = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL_DAIOE
_load_daioe = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL_DAIOE: _load_daioe(path)
sys.path.insert(0, str(HERE))
from _fixtures import Fixture  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s71 = load("71_adoption_validation.py", "s71")
s71.OUT = TMP / "out"; s71.OUT.mkdir(); s71.CACHE = mc.CACHE_DIR
j47 = s71._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- 1. the survey parser --------------------------------------------
v = s71.to01(pd.Series(["1", "0", "J", "N", "Ja", "Nej", "****", "", "9",
                        "TRUE", "FALSE", None]))
check("'1' and 'J' and 'Ja' read as yes",
      list(v[[0, 2, 4]]) == [1.0, 1.0, 1.0])
check("'0' and 'N' and 'Nej' read as no",
      list(v[[1, 3, 5]]) == [0.0, 0.0, 0.0])
check("'****' is missing, NOT a no", bool(np.isnan(v[6])))
check("an empty string is missing, NOT a no", bool(np.isnan(v[7])))
check("an unrecognised code is missing, NOT a no", bool(np.isnan(v[8])))
check("a NULL is missing", bool(np.isnan(v[11])))
check("case does not matter", s71.to01(pd.Series(["ja"]))[0] == 1.0)
# THE 21 SEPTEMBER MONA FAILURE. pyodbc returns a numeric survey flag that
# has NULLs as float, so it stringifies as "1.0". Every ITFtg value parsed
# as missing, the gate read that as a thin sample and refused, and the run
# came back in 2.4 minutes looking like a clean exit.
fl = s71.to01(pd.Series([1.0, 0.0, np.nan, 2.0]))
check("a float column parses, which it did not on MONA",
      list(fl[[0, 1, 3]]) == [1.0, 0.0, 0.0] and bool(np.isnan(fl[2])),
      str(list(fl)))
check("'1.0' as text parses too", s71.to01(pd.Series(["1.0"]))[0] == 1.0)
# and if a column still yields nothing, the codes must be reported rather
# than left to another guess and another MONA round
s71.NOTES.clear()
s71.parse_report(pd.Series(["Q", "Z"]), pd.Series([np.nan, np.nan]), "T.C")
check("an unparseable column reports its raw codes",
      any("distinct raw codes" in n.lower() for n in s71.NOTES),
      "; ".join(s71.NOTES)[:70])
s71.NOTES.clear()

cols = ["P1207_LopNr_PeOrgNr", "E_AI_TNLG", "foo"]
check("the firm key is found by pattern",
      s71.pick(cols, r"PeOrgNr") == "P1207_LopNr_PeOrgNr")
check("a missing key returns None, rather than guessing",
      s71.pick(["a", "b"], r"PeOrgNr") is None)


# ---- 2. a synthetic delivery ------------------------------------------
YEARS = [2021, 2022, 2023, 2024, 2025]
# Big enough to clear MIN_ITFTG_FIRMS at its REAL value. Lowering
# the threshold for the test would leave the number that protects
# the paper unexercised, which is the one thing worth testing here.
fx = Fixture(mc, h47, n_firms=1200, n_exposed=520)
fx.install_edu([2019], cache=mc.CACHE_DIR)
fx.baseline_frame().to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet",
                               index=False)
EXPO = s71.edu_exposure(j47, s71.DESIGN, s71.ARM)
HIGH = set(EXPO[EXPO["fq"] == 4]["employer_id"])
ALLF = list(EXPO["employer_id"])
check("the fixture classifies enough firms to exercise the gate",
      len(ALLF) > s71.MIN_ITFTG_FIRMS and len(HIGH) > s71.MIN_ITFTG_HIGH,
      f"{len(ALLF)} firms, {len(HIGH)} high")

TRUE_GAP = 0.25          # the first stage the generator plants
COVER = {"wide": 1.0, "thin": 0.08}


def survey(cover: str, seed=3):
    """ITFtg-shaped rows for a share of our firms, with a planted gap."""
    rng = np.random.default_rng(seed)
    firms = [f for f in ALLF if rng.random() < COVER[cover]]
    rows = []
    for f in firms:
        p = 0.20 + (TRUE_GAP if f in HIGH else 0.0)
        val = "1" if rng.random() < p else "0"
        # a realistic sprinkling of the delivery's missing marker
        if rng.random() < 0.05:
            val = "****"
        rows.append((f, val, val))
    return pd.DataFrame(rows, columns=["P1207_LopNr_PeOrgNr",
                                       "E_AI_TML", "E_AI_TNLG"])


CATALOGUE = pd.DataFrame(
    [("ITFtg_Stora_2023", c, "varchar") for c in
     ("P1207_LopNr_PeOrgNr", "E_AI_TML", "E_AI_TNLG")]
    + [("ai_fufi_2019", c, "varchar") for c in
       ("P1207_LopNr_PeOrgNr", "AI_COST_T", "AI_COST_GOODS")]
    + [("BITA_2024", c, "varchar") for c in
       ("P1207_LopNr_PersonNr", "CH1", "CH2b", "vikt_ind_SE")],
    columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"])


def spend(cover: str, seed=8):
    """ai_fufi-shaped rows: a CONTINUOUS AI expenditure outcome."""
    rng = np.random.default_rng(seed)
    firms = [f for f in ALLF if rng.random() < COVER[cover]]
    rows = []
    for f in firms:
        base = 8.0 + (1.2 if f in HIGH else 0.0)
        v = float(np.exp(base + rng.normal(0, 0.5))) if rng.random() < 0.6 \
            else 0.0
        rows.append((f, v, v * 0.4))
    return pd.DataFrame(rows, columns=["P1207_LopNr_PeOrgNr",
                                       "AI_COST_T", "AI_COST_GOODS"])


def bita(seed=4, multi=30):
    rng = np.random.default_rng(seed)
    rows = []
    for i, f in enumerate(ALLF):
        p = 0.30 + (TRUE_GAP if f in HIGH else 0.0)
        for k in range(3):
            pid = i * 10 + k
            rows.append((pid, "1" if rng.random() < p else "0",
                         "1" if rng.random() < p else "0", 1.0 + rng.random()))
    return pd.DataFrame(rows, columns=["P1207_LopNr_PersonNr", "CH1", "CH2b",
                                       "vikt_ind_SE"])


def link(multi=30):
    rows = []
    for i, f in enumerate(ALLF):
        for k in range(3):
            rows.append((i * 10 + k, f))
    # a handful of respondents hold two employers and must be DROPPED, not
    # assigned to whichever row happens to come first
    for i in range(multi):
        rows.append((i * 10, ALLF[(i + 1) % len(ALLF)]))
    return pd.DataFrame(rows, columns=["person_id", "employer_id"])


STATE = {"cover": "wide", "catalogue": CATALOGUE}


def fake_read_sql(q, conn=None, *a, **kw):
    ql = str(q).lower()
    if "information_schema" in ql:
        return STATE["catalogue"].copy()
    if "ai_fufi" in ql:
        return spend(STATE["cover"]).copy()
    if "itftg" in ql:
        return survey(STATE["cover"]).copy()
    if "bita" in ql:
        return bita().copy()
    if "arb_agiindivid" in ql:
        return link().copy()
    raise AssertionError(f"unexpected query: {q[:120]}")


s71.pd.read_sql = fake_read_sql
mc.connect = lambda: object()


# ---- 3. the gate, both ways -------------------------------------------
def run(cover):
    STATE["cover"] = cover
    s71.NOTES.clear()
    for f in s71.OUT.glob("*.csv"):
        f.unlink()
    _out = sys.stdout
    try:
        s71.main()
    finally:
        sys.stdout = _out
    return (s71.OUT / "71_summary.txt").read_text()


txt_thin = run("thin")
check("a thin overlap produces NO first-stage estimate",
      not (s71.OUT / "itftg_firststage.csv").exists())
check("and says why, naming the rule fixed before the run",
      any("BELOW THRESHOLD" in n for n in s71.NOTES),
      "; ".join(n for n in s71.NOTES if "THRESHOLD" in n)[:90])

txt_wide = run("wide")
check("a wide overlap DOES produce a first stage",
      (s71.OUT / "itftg_firststage.csv").exists())
if (s71.OUT / "itftg_firststage.csv").exists():
    fs = pd.read_csv(s71.OUT / "itftg_firststage.csv")
    hi = fs[(fs.term == "high") & (fs.outcome == "ai_any")]
    check("the first stage recovers the planted gap",
          len(hi) and abs(float(hi.iloc[0]["coef"]) - TRUE_GAP) < 0.08,
          f"{float(hi.iloc[0]['coef']):.3f} vs planted {TRUE_GAP}"
          if len(hi) else "no row")
    check("it is estimated on both routes where both classify",
          set(fs["route"]) >= {"education"}, str(sorted(set(fs["route"]))))
    check("the genAI-specific outcome is reported separately",
          "ai_genai" in set(fs["outcome"]))
    check("the CONTINUOUS expenditure outcome is estimated too, which is "
          "where the power is",
          "ai_spend_log" in set(fs["outcome"]),
          str(sorted(set(fs["outcome"]))))
    sp = fs[(fs.term == "high") & (fs.outcome == "ai_spend_log")]
    check("and it recovers a positive exposure gradient in AI spending",
          len(sp) and float(sp.iloc[0]["coef"]) > 0,
          f"{float(sp.iloc[0]['coef']):+.3f}" if len(sp) else "no row")
    check("the 2019 expenditure table is used, not only the 2023 survey",
          any("fufi" in str(x).lower() for x in fs["source"]),
          str(sorted(set(fs["source"]))))

check("the overlap counts are written", (s71.OUT / "overlap_counts.csv").exists())
check("the schema it found is written down",
      (s71.OUT / "schema_found.csv").exists())
check("the summary states the sample-survey limit",
      "stratified sample surveys" in txt_wide)
check("the summary refuses to call a first stage a causal claim",
      "does not make adoption exogenous" in txt_wide)

# BITA respondents with two employers must be dropped, not assigned
if (s71.OUT / "overlap_counts.csv").exists():
    oc = pd.read_csv(s71.OUT / "overlap_counts.csv")
    b = oc[oc["source"].str.startswith("BITA")]
    check("BITA respondents are linked to at most one employer",
          (not len(b)) or bool((b["matched"] <= 3 * len(ALLF)).all()),
          f"matched {list(b['matched'])[:2]}")


# ---- 4. disclosure: dropped, never blanked ----------------------------
if (s71.OUT / "overlap_counts.csv").exists():
    oc = pd.read_csv(s71.OUT / "overlap_counts.csv")
    check("no surviving count row sits under the export floor",
          bool((oc["with_outcome"] >= s71.FLOOR).all()
               and (oc["high_with_outcome"] >= s71.FLOOR).all()))
    check("suppression drops rows rather than blanking them",
          not oc.isna().any().any())


# ---- 5. an empty catalogue must not crash -----------------------------
STATE["catalogue"] = CATALOGUE.iloc[0:0]
s71.NOTES.clear()
(s71.OUT / "71_summary.txt").unlink(missing_ok=True)
_out = sys.stdout
crashed = False
try:
    s71.main()
except BaseException as ex:  # noqa: BLE001
    crashed = True
    sys.stdout = _out
    print(f"      raised {type(ex).__name__}: {ex}")
sys.stdout = _out
check("an empty catalogue is survived, not crashed on", not crashed)
check("and it says plainly that nothing was visible",
      (s71.OUT / "71_summary.txt").exists()
      and "No ITFtg" in (s71.OUT / "71_summary.txt").read_text())

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

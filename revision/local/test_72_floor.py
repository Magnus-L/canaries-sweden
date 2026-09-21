#!/usr/bin/env python3
"""
test_72_floor.py -- the reliability measure must separate a firm whose
                    young do the same work as its incumbents from one
                    whose young do not, and the attenuation premise the
                    read rule rests on must actually hold.

The fixture already has both worlds. year_frame draws every age band in a
firm from the same education pool, so young and incumbent mixes AGREE.
year_frame(corrupt_young=True) gives the young gymnasium cells regardless
of the firm, so they DISAGREE. If Part A cannot tell those apart it is
measuring nothing.

Read rule 2 says a weak proxy means the headline is attenuated and the
leave-one-out estimate should be larger. That is an assumption about
measurement error, so it is tested rather than believed: an estimate on a
deliberately scrambled exposure must shrink toward zero against the same
estimate on the true one.

    CANARIES_DRYRUN=1 python3 revision/local/test_72_floor.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries72_"))
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
from _fixtures import Fixture, AGES  # noqa: E402


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s72 = load("72_incumbent_floor.py", "s72")
s72.OUT = TMP / "out"; s72.OUT.mkdir(); s72.CACHE = mc.CACHE_DIR
j47 = s72._mod("47j_within_employer_triple.py", "j47")
h47 = j47._h47()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


YEARS = [2021, 2022, 2023, 2024, 2025]
fx = Fixture(mc, h47, n_firms=160, n_exposed=70)


EDU_COLS = ["niva_t", "inr_t", "expb_t"]


def edu_world(disagree: bool):
    """
    Install the education caches for one world and return its inputs.

    NOT the fixture's `corrupt_young`: that rewrites the AS-OF columns,
    simulating stale codes for the young, which is what the backtest
    needs and not what this script measures. Checked on 21 September
    2026, when using it produced an identical correlation in both worlds
    because the true columns never moved.

    The disagreement world instead permutes the young rows' TRUE
    education cells across firms. Each firm keeps a young workforce and
    the national mix is unchanged; only the link between a firm's young
    and its own incumbents is destroyed, which is exactly the world ML
    asked about, where the young are trainees and assistants rather than
    junior versions of their older colleagues.
    """
    for y in (2019, 2020, 2021):
        fx.weights_frame(y).to_parquet(
            mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet", index=False)
    fr = fx.year_frame(2019)
    if disagree:
        yng = fr["age_group"].astype(str).isin(["22-25", "26-30"])
        blk = fr.loc[yng, EDU_COLS]
        perm = np.random.default_rng(404).permutation(len(blk))
        fr.loc[yng, EDU_COLS] = blk.to_numpy()[perm]
    fr.to_parquet(mc.CACHE_DIR / "edu_hr_2019.parquet", index=False)
    return s72.edu_inputs(j47)


# ---- 1. the band-set swap must be safe -------------------------------
book, spec, frame19 = edu_world(disagree=False)
before = list(j47.INCUMBENT_BANDS)
e_inc, _ = s72.exposure_on(j47, frame19, book, spec, s72.INCUMBENT_BANDS)
check("the incumbent band set is restored after the call",
      j47.INCUMBENT_BANDS == before)
try:
    s72.exposure_on(j47, None, book, spec, ["22-25"])
except BaseException:
    pass
check("it is restored even when the call raises",
      j47.INCUMBENT_BANDS == before, str(j47.INCUMBENT_BANDS))

e_y, _ = s72.exposure_on(j47, frame19, book, spec, ["22-25"])
check("a different band set gives a different measure",
      e_y is not None and len(e_y) and not e_inc["mix"].equals(
          e_inc["employer_id"].map(
              e_y.set_index("employer_id")["mix"])))


# ---- 2. Part A must separate the two worlds --------------------------
def reliability(disagree):
    book, spec, frame19 = edu_world(disagree)
    sink = []
    s72.part_a(j47, frame19, book, spec, sink)
    d = pd.DataFrame(sink)
    d = d[(d.young_band == "22-25") & (d.group == "all")]
    return (float(d.iloc[0]["pearson"]), float(d.iloc[0]["same_quartile"])) \
        if len(d) else (None, None)


r_agree, q_agree = reliability(False)
r_diff, q_diff = reliability(True)
check("AGREE world: the young mix tracks the incumbent mix",
      r_agree is not None and r_agree > 0.5, f"r {r_agree}")
check("DISAGREE world: it does not",
      r_diff is not None and r_diff < r_agree - 0.2,
      f"agree {r_agree} vs disagree {r_diff}")
check("quartile agreement moves the same way",
      None not in (q_agree, q_diff) and q_agree > q_diff,
      f"{q_agree} vs {q_diff}")


# ---- 3. the read rule maps those onto the right words ----------------
def verdict_for(r, q):
    rel = pd.DataFrame([{"young_band": "22-25", "group": "all",
                         "pearson": r, "same_quartile": q}])
    return " ".join(s72.verdict(rel, pd.DataFrame(), pd.DataFrame()))


check("a good proxy reads GOOD", "PROXY GOOD" in verdict_for(0.85, 0.75))
check("a weak proxy reads WEAK", "PROXY WEAK" in verdict_for(0.20, 0.30))
check("in between reads PARTIAL", "PARTIAL" in verdict_for(0.55, 0.50))

art = pd.DataFrame([{"young_band": "22-25", "true": -0.05, "asof": -0.04,
                     "artefact": 0.01}])
rel = pd.DataFrame([{"young_band": "22-25", "group": "all",
                     "pearson": 0.85, "same_quartile": 0.75}])
check("a small artefact reads USABLE",
      "USABLE." in " ".join(s72.verdict(rel, pd.DataFrame(), art)))
art.loc[0, "artefact"] = 0.20
check("a large artefact reads NOT USABLE",
      "NOT USABLE" in " ".join(s72.verdict(rel, pd.DataFrame(), art)))

loo = pd.DataFrame([
    {"young_band": "22-25", "measure": "incumbent31", "coef": -0.20,
     "se": 0.02, "n_firms": 100},
    {"young_band": "22-25", "measure": "leave_one_out", "coef": -0.10,
     "se": 0.02, "n_firms": 100}])
weak = pd.DataFrame([{"young_band": "22-25", "group": "all",
                      "pearson": 0.20, "same_quartile": 0.30}])
check("a weak proxy whose leave-one-out is NOT larger is flagged, "
      "not written around",
      "needs chasing" in " ".join(s72.verdict(weak, loo, pd.DataFrame())))


# ---- 4. the attenuation premise itself -------------------------------
book, spec, frame19 = edu_world(disagree=False)
EXPO, _ = s72.exposure_on(j47, frame19, book, spec, s72.INCUMBENT_BANDS)
HIGH = set(EXPO[EXPO["fq"] == 4]["employer_id"])


def counts_with_decline(beta=-0.35, seed=5):
    rng = np.random.default_rng(seed)
    lam0 = {"22-25": 9, "26-30": 10, "31-34": 9, "35-40": 11,
            "41-49": 13, "50+": 15}
    rows = []
    for emp in EXPO["employer_id"]:
        hi = emp in HIGH
        for y in YEARS:
            for m in (range(1, 13) if y < 2025 else range(1, 7)):
                ym = f"{y}-{m:02d}"
                for age in AGES:
                    lam = float(lam0[age])
                    if hi and age == "22-25" and ym >= s72.POOLED_FROM:
                        lam *= float(np.exp(beta))
                    rows.append((emp, ym, age, int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "n_emp"])


CNT = counts_with_decline()
clean = s72.fit_headline(CNT, EXPO, "22-25", j47, "clean")
noisy_expo = EXPO.copy()
rng = np.random.default_rng(99)
scramble = rng.random(len(noisy_expo)) < 0.45
noisy_expo.loc[scramble, "fq"] = rng.integers(1, 5, scramble.sum())
noisy = s72.fit_headline(CNT, noisy_expo, "22-25", j47, "noisy")
check("the clean measure recovers the planted decline",
      clean is not None and clean["coef"] < -0.20,
      f"{clean['coef']:.4f}" if clean else "no fit")
check("a noisier measure ATTENUATES it, which is what read rule 2 assumes",
      None not in (clean, noisy) and abs(noisy["coef"]) < abs(clean["coef"]),
      f"clean {clean['coef']:.4f} vs noisy {noisy['coef']:.4f}"
      if None not in (clean, noisy) else "")


# ---- 5. the leave-one-out band set ------------------------------------
loo_bands = [b for b in s72.YOUNG_BANDS + s72.INCUMBENT_BANDS
             if b != "22-25"]
check("leave-one-out excludes the band being estimated",
      "22-25" not in loo_bands)
check("but keeps the other young band, which is the point",
      "26-30" in loo_bands, "+".join(loo_bands))


# ---- 6. end to end -----------------------------------------------------
for y in YEARS:
    CNT[CNT["year_month"].str[:4] == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
s72.PANEL_YEARS = YEARS
_real = sys.stdout
try:
    s72.main()
    ran = True
except BaseException as ex:  # noqa: BLE001
    ran = False
    sys.stdout = _real
    print(f"      main() raised: {type(ex).__name__}: {ex}")
sys.stdout = _real
check("main() runs end to end", ran)
check("a summary is written", (s72.OUT / "72_summary.txt").exists())
if (s72.OUT / "72_summary.txt").exists():
    txt = (s72.OUT / "72_summary.txt").read_text()
    check("the summary says the floor is forced rather than chosen",
          "forced, not chosen" in txt)
    check("the summary refuses attenuation correction explicitly",
          "No attenuation correction" in txt)

(mc.CACHE_DIR / f"L_counts_{YEARS[0]}.parquet").unlink()
refused = False
try:
    s72.main()
except SystemExit as ex:
    refused = "performs no SQL" in str(ex)
except BaseException:
    refused = False
sys.stdout = _real
check("a missing counts cache is refused, not pulled", refused)

print("\n" + "=" * 62)
print(f"{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all checks passed'}")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

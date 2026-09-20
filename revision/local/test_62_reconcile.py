#!/usr/bin/env python3
"""
test_62_reconcile.py -- the decomposition must attribute a disagreement to
                        the ingredient that actually caused it.

The fixture builds a world where the UNIT is what matters. Young workers
in the exposed firms hold high-DAIOE occupations in 2019; those same
firms' incumbents do not. A decline is then planted on the AGE-SPECIFIC
exposure of the 22-25 band. So:

  variant A, occupation measured on the age group itself, must find it,
  variant B, the same occupations measured on the firm's incumbents, must
    not, because by construction those incumbents are unexposed.

If the script reported the same gradient for A and B it would be blind to
the one distinction it exists to draw. Also tested: the firm-level measure
ignores the young entirely, the education measure collapses the twelve
monthly rows of 2019 correctly, the quartile form is a quartile, and
main() runs end to end off the caches with SQL forbidden.

    CANARIES_DRYRUN=1 python3 revision/local/test_62_reconcile.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries62_"))
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


s62 = load("62_reconcile_gradients.py", "s62")
s62.OUT = TMP / "out"; s62.OUT.mkdir(); s62.CACHE = mc.CACHE_DIR
l47 = s62._mod("47L_age_baseline_exposure.py", "l47")
h47 = s62._mod("47h_edu_horserace.py", "h47")
l47.OUT = s62.OUT

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
BASE = FIX.baseline_frame()
BETA = -0.35
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- what the two units actually measure -----------------------------
A = s62.occ_exposure(BASE, FIX.daioe, True)
B = s62.occ_exposure(BASE, FIX.daioe, False)
check("the age-specific measure varies across bands inside a firm",
      A.groupby("employer_id")["expo"].nunique().min() > 1)
check("the firm measure is one number repeated across the firm's bands",
      (B.groupby("employer_id")["expo"].nunique() == 1).all())
check("the exposed firms' YOUNG are the exposed ones",
      A[(A.employer_id.isin(FIX.exposed)) & (A.age_group == "22-25")]["expo"].mean()
      > A[(~A.employer_id.isin(FIX.exposed)) & (A.age_group == "22-25")]["expo"].mean() + 5,
      "which is the world the decomposition has to see through")

corrupt = BASE.copy()
young = corrupt["age_group"].isin(["22-25", "26-30"])
corrupt.loc[young, "ssyk4"] = "9999"        # a code DAIOE does not score
Bc = s62.occ_exposure(corrupt, FIX.daioe, False)
j = B.merge(Bc, on=["employer_id", "age_group"], suffixes=("", "_c"))
check("the firm measure ignores the young entirely",
      len(j) > 100 and np.allclose(j["expo"], j["expo_c"]),
      f"{len(j)} cells identical after corrupting every young occupation")

# ---- the education side ----------------------------------------------
book, spec, frames = FIX.install_edu([2019])
f19 = frames[2019]
C = s62.edu_exposure(f19, book, "OL_daioe", spec, True, False)
D = s62.edu_exposure(f19, book, "OL_daioe", spec, False, True)
check("the quartile form really is a quartile",
      set(np.unique(D["expo"])) <= {1.0, 2.0, 3.0, 4.0},
      " ".join(f"{v:.0f}" for v in np.unique(D["expo"])))
# 47j cuts the quartiles on WORKERS, not on firms, so variant D must too
# or it is not 47j's exposure. Check the share of employment, not the
# share of employers.
w19 = (f19.groupby("employer_id", observed=True)["n_emp"].sum()
       .rename("w").reset_index())
dw = D.drop_duplicates("employer_id").merge(w19, on="employer_id")
shr = dw.groupby("expo")["w"].sum() / dw["w"].sum()
check("and it is cut on workers, so the four groups carry ~25% each",
      shr.min() > 0.15 and shr.max() < 0.35,
      " ".join(f"Q{int(k)} {v:.0%}" for k, v in shr.items()))
# The script collapses the twelve monthly rows of 2019 before scoring,
# because the uncollapsed frame is tens of millions of rows in MONA. That
# is only safe if it gives the same answer as the obvious way, so here is
# the obvious way, computed independently row by row.
cols = ["niva_t", "inr_t", "expb_t"]
ref = f19[["employer_id", "age_group", "n_emp"] + cols].copy()
combo = ref[cols + ["age_group"]].drop_duplicates().reset_index(drop=True)
combo["_s"] = book.score_frame("OL_daioe", spec, combo[cols[0]], combo[cols[1]],
                               combo[cols[2]], None, combo["age_group"])
ref = ref.merge(combo, on=cols + ["age_group"], how="left")
ref = ref[ref["_s"].notna()]
ref["_ws"] = ref["_s"] * ref["n_emp"]
ref = (ref.groupby(["employer_id", "age_group"], observed=True)
       .agg(ws=("_ws", "sum"), n=("n_emp", "sum")).reset_index())
ref["ref"] = ref["ws"] / ref["n"]
m = C.merge(ref[["employer_id", "age_group", "ref"]],
            on=["employer_id", "age_group"])
check("collapsing before scoring gives the row-by-row answer exactly",
      len(m) > 100 and np.allclose(m["expo"], m["ref"], atol=1e-9),
      f"{len(m)} cells agree to 1e-9")

# ---- the mechanism: a unit-driven disagreement ------------------------
counts = FIX.install_occ(l47.YEARS, zmap=FIX.zmap(l47), beta=BETA,
                         bands=("22-25",), from_ym=mc.CHATGPT_YM)


def gradient(expo, tag):
    bal = l47.build_panel(counts, expo)
    g = l47.fit_gradient(bal, tag)
    return g.set_index("age_group")["coef"] if not g.empty else None


gA = gradient(A, "t62_A")
gB = gradient(B, "t62_B")
check("variant A recovers the planted age-specific decline",
      gA is not None and abs(gA["22-25"] - BETA) < 0.10,
      f"{gA['22-25']:+.4f} against a planted {BETA:+.4f}")
check("variant B, the same occupations read off the incumbents, does not",
      gB is not None and abs(gB["22-25"]) < abs(gA["22-25"]) / 2,
      f"A {gA['22-25']:+.4f} vs B {gB['22-25']:+.4f}")
check("so the decomposition can see a UNIT-driven disagreement",
      abs(gB["22-25"] - gA["22-25"]) > 0.10)

# ---- end to end -------------------------------------------------------
def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s62.main()
G = pd.read_csv(s62.OUT / "gradient_by_variant.csv")
check("all four variants were estimated",
      set(G["variant"]) == {k for k, _ in s62.VARIANTS},
      " ".join(sorted(set(G["variant"]))))
piv = G.pivot_table(index="age_group", columns="variant", values="coef")
check("main() finds the decline through the age-specific occupation route",
      abs(piv.loc["22-25", "A_occ_age_cont"] - BETA) < 0.10,
      f"{piv.loc['22-25', 'A_occ_age_cont']:+.4f}")
check("and main() does NOT find it through the firm-incumbent route",
      abs(piv.loc["22-25", "B_occ_firm_cont"])
      < abs(piv.loc["22-25", "A_occ_age_cont"]) / 2,
      f"{piv.loc['22-25', 'B_occ_firm_cont']:+.4f}")
agree = pd.read_csv(s62.OUT / "exposure_agreement.csv")
check("the agreement table covers every pair of variants",
      len(agree) == 6, f"{len(agree)} pairs")
summ = (s62.OUT / "62_summary.txt").read_text()
for must in ("UNIT alone (A to B)", "SOURCE alone (A to C)", "READ THIS"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""
test_65_occupation.py -- the occupational arm must classify the same firms
                         the same way when the two registers agree, and
                         must use incumbents only.

Four claims, each tested:
  1 the exposure uses INCUMBENTS ONLY, so corrupting or deleting every
    young worker's occupation leaves every firm's quartile untouched,
  2 the quartile cutoffs are WORKER-weighted, as 47j's are, so the four
    groups carry about a quarter of incumbent employment each,
  3 a decline planted on the firms this classifier calls Q4 is recovered
    with the right sign and rough size,
  4 main() runs end to end off the caches with SQL forbidden, and writes
    the two registers side by side when 61's output is present.

    CANARIES_DRYRUN=1 python3 revision/local/test_65_occupation.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries65_"))
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


s65 = load("65_occupation_arm.py", "s65")
s65.OUT = TMP / "out"; s65.OUT.mkdir(); s65.CACHE = mc.CACHE_DIR
s61 = s65._mod("61_redated_triple.py", "s61"); s61.OUT = s65.OUT
j47 = s61._j47()
h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
BASE = FIX.baseline_frame()
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


E = s65.occupation_exposure(BASE, FIX.daioe, j47.INCUMBENT_BANDS)
check("every firm with enough incumbents is classified",
      len(E) > 100 and set(E["fq"]) <= {1, 2, 3, 4}, f"{len(E)} firms")
shares = E.groupby("fq")["n"].sum() / E["n"].sum()
check("the cutoffs are worker-weighted, so each quartile carries ~25% of "
      "incumbent employment",
      shares.min() > 0.15 and shares.max() < 0.35,
      " ".join(f"Q{int(k)} {v:.0%}" for k, v in shares.items()))

corrupt = BASE.copy()
young = corrupt["age_group"].isin(["22-25", "26-30"])
corrupt.loc[young, "ssyk4"] = "9999"
Ec = s65.occupation_exposure(corrupt, FIX.daioe, j47.INCUMBENT_BANDS)
j = E.merge(Ec, on="employer_id", suffixes=("", "_c"))
check("the classification ignores the young entirely",
      len(j) == len(E) and (j["fq"] == j["fq_c"]).all()
      and np.allclose(j["mix"], j["mix_c"]),
      f"{len(j)} firms identical after corrupting every young occupation")
dropped = BASE[~young]
Ed = s65.occupation_exposure(dropped, FIX.daioe, j47.INCUMBENT_BANDS)
check("and deleting the young changes nothing either",
      len(Ed) == len(E) and (E.set_index("employer_id")["fq"]
                             == Ed.set_index("employer_id")["fq"]).all())

# ---- a planted decline on the firms THIS classifier calls Q4 ---------
Q4 = set(E.loc[E["fq"] == 4, "employer_id"].astype(int))
BETA = float(np.log(0.75))
counts = pd.concat([FIX.counts_frame(y, zmap={(e, "22-25"): 1.0 for e in Q4},
                                     beta=BETA, bands=("22-25",),
                                     from_ym="2024-01")
                    for y in s61.PANEL_YEARS], ignore_index=True)
skel = s61.build_skeleton(counts, "22-25", j47)
b, step_terms, pool_terms = s61.attach_exposure(skel, E)
r = mc.run_fepois_multi(b, s65.OUT, tag="t65_pool", terms=pool_terms,
                        fes=j47.FES)
got = float(r.set_index("term").loc["post2024_x_high_x_young", "coef"])
check("a decline planted on this classifier's Q4 is recovered",
      abs(got - BETA) < 0.15, f"{got:+.4f} against a planted {BETA:+.4f}")

# ---- end to end -------------------------------------------------------
FIX.install_occ(s61.PANEL_YEARS, zmap={(e, "22-25"): 1.0 for e in Q4},
                beta=BETA, bands=("22-25",), from_ym="2024-01")
# 61's own answer, so the side-by-side block is exercised
(s65.HERE / "output_61").mkdir(exist_ok=True)
prior = s65.HERE / "output_61" / "redated_pooled.csv"
had_prior = prior.exists()
if not had_prior:
    pd.DataFrame([{"design": "OL_daioe", "arm": "true", "young_band": "22-25",
                   "term": "post2024_x_high_x_young", "coef": -0.0509,
                   "se": 0.0122, "n_obs": 10, "status": "ok"}]).to_csv(
        prior, index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
j47.YOUNG_BANDS = ["22-25"]
try:
    s65.main()
finally:
    if not had_prior:
        prior.unlink(missing_ok=True)
P = pd.read_csv(s65.OUT / "occ_pooled.csv")
check("main() writes the pooled coefficient",
      (P["term"] == "post2024_x_high_x_young").any()
      and (P["status"] == "ok").all())
summ = (s65.OUT / "65_summary.txt").read_text()
check("the summary sets the two registers side by side",
      "THE TWO REGISTERS SIDE BY SIDE" in summ)
for must in ("READ THIS", "Education stays primary", "2019 is final"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

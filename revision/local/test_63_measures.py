#!/usr/bin/env python3
"""
test_63_measures.py -- the placebo has to be able to fire.

A robustness check that cannot come out badly is not a check. This test
therefore builds the world twice.

  WORLD ONE, the one we hope we are in: the decline is planted on DAIOE
  exposure. The daioe column must find it, and in the horse race the
  teleworkability coefficient must be the weaker of the two.

  WORLD TWO, the one that would sink the paper: the same decline is
  planted on teleworkability instead. The horse race must then say so,
  with the two coefficients the other way round. If it could not, the
  test in world one would be worthless.

Also tested: each score is standardised before it is used, so the three
measures are on a comparable scale; the correlation table is produced;
and main() runs end to end off the caches with SQL forbidden.

    CANARIES_DRYRUN=1 python3 revision/local/test_63_measures.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries63_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "dingel_neiman_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


s63 = load("63_measure_robustness.py", "s63")
s63.OUT = TMP / "out"; s63.OUT.mkdir(); s63.CACHE = mc.CACHE_DIR
l47 = s63._mod("47L_age_baseline_exposure.py", "l47")
s54 = s63._mod("54_hiring_flows.py", "s54")
h47 = s63._mod("47h_edu_horserace.py", "h47")
l47.OUT = s54.OUT = s63.OUT

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
BASE = FIX.baseline_frame()
BETA = -0.35
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---- the measures themselves -----------------------------------------
SC = {k: s63.load_measure(f, c) for k, f, c, _ in s63.MEASURES}
for k, d in SC.items():
    check(f"{k} is standardised before use",
          abs(d["score"].mean()) < 1e-9 and abs(d["score"].std(ddof=0) - 1) < 1e-9,
          f"mean {d['score'].mean():+.1e} sd {d['score'].std(ddof=0):.6f}")
check("no occupation carries a missing score",
      all(d["score"].notna().all() for d in SC.values()))

EXPO = {k: l47.build_exposure(BASE, d) for k, d in SC.items()}
cellcorr = float(np.corrcoef(
    *EXPO["daioe"].merge(EXPO["telework"], on=["employer_id", "age_group"],
                         suffixes=("_d", "_t"))[["expo_d", "expo_t"]]
    .to_numpy().T)[0, 1])
print(f"     (fixture cell-level corr daioe vs telework {cellcorr:+.3f})")


def zmap(key):
    e = EXPO[key]
    mu, sd = e["expo"].mean(), e["expo"].std(ddof=0)
    return {(r.employer_id, r.age_group): (r.expo - mu) / (sd or 1.0)
            for r in e.itertuples()}


def world(planted_on: str):
    """Counts in which the 22-25 decline loads on ONE measure's exposure."""
    z = zmap(planted_on)
    return pd.concat([FIX.counts_frame(y, zmap=z, beta=BETA, bands=("22-25",),
                                       from_ym=mc.CHATGPT_YM)
                      for y in l47.YEARS], ignore_index=True)


def race(counts, tag):
    """(daioe, telework) at 22-25, from the by-age horse race."""
    bal = l47.build_panel(counts, EXPO["daioe"])
    r, n = s63.horserace(bal, EXPO["telework"], l47, tag)
    by = r[r["spec"] == "by_age"].set_index("term")["coef"]
    pool = r[r["spec"] == "pooled"].set_index("term")["coef"]
    return (float(by[l47.age_term("22-25")]), float(by[s63.tele_term("22-25")]),
            float(pool["post_gpt_x_expo"]))


# ---- world one: the decline is genuinely about AI exposure ------------
d1, t1, p1 = race(world("daioe"), "t63_world_ai")
check("planted on AI exposure, the horse race puts the weight on daioe",
      d1 < t1 - 0.05, f"daioe {d1:+.4f} vs telework {t1:+.4f} at 22-25")
check("and the AI coefficient has the planted sign and rough size",
      abs(d1 - BETA) < 0.15, f"{d1:+.4f} against a planted {BETA:+.4f}")
check("the POOLED row is diluted, as the summary warns it is",
      abs(p1) < abs(d1) / 2,
      f"pooled {p1:+.4f} against by-age {d1:+.4f}: one band in six")

# ---- world two: the same decline is really about office work ---------
d2, t2, _ = race(world("telework"), "t63_world_tele")
check("planted on teleworkability instead, the horse race says SO",
      t2 < d2, f"daioe {d2:+.4f} vs telework {t2:+.4f} at 22-25")
check("so this test can fail, which is the only reason to run it",
      (d1 < t1) and (t2 < d2),
      f"AI world {d1:+.3f}/{t1:+.3f}, telework world {d2:+.3f}/{t2:+.3f}")

# ---- end to end -------------------------------------------------------
FIX.install_occ(l47.YEARS, zmap=zmap("daioe"), beta=BETA, bands=("22-25",),
                from_ym=mc.CHATGPT_YM)
rng = np.random.default_rng(63)
for y in s54.YEARS:
    c = pd.read_parquet(mc.CACHE_DIR / f"L_counts_{y}.parquet")
    f = c.rename(columns={"n_emp": "n_hire"})
    f["n_hire"] = rng.poisson(2.0, len(f))      # no shock on the flow
    f["n_sep"] = rng.poisson(2.0, len(f))
    f[s54.FLOW_COLS].to_parquet(mc.CACHE_DIR / f"flows_{y}.parquet",
                                index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s63.main()
G = pd.read_csv(s63.OUT / "robustness_gradient.csv")
check("every measure was estimated on both outcomes",
      set(zip(G["measure"], G["outcome"])) ==
      {(m, o) for m in SC for o in ("stock", "hires")},
      f"{len(set(zip(G['measure'], G['outcome'])))} of 6 combinations")
st = G[G.outcome == "stock"].pivot_table(index="age_group", columns="measure",
                                         values="coef")
check("main() recovers the planted decline on the stock, on daioe",
      abs(st.loc["22-25", "daioe"] - BETA) < 0.15,
      f"{st.loc['22-25', 'daioe']:+.4f}")
hi = G[G.outcome == "hires"].pivot_table(index="age_group", columns="measure",
                                         values="coef")
check("and finds nothing on the flow, where nothing was planted",
      abs(hi.loc["22-25", "daioe"]) < 0.10,
      f"{hi.loc['22-25', 'daioe']:+.4f}")
H = pd.read_csv(s63.OUT / "horserace.csv")
check("the horse race ran on both outcomes and both specifications",
      set(H["outcome"]) == {"stock", "hires"}
      and set(H["spec"]) == {"pooled", "by_age"})
cor = pd.read_csv(s63.OUT / "measure_correlation.csv")
check("the correlation table covers every pair of measures",
      len(cor) == 3, f"{len(cor)} pairs")
check("and it reports the telework correlation the reader needs",
      float(cor.loc[(cor.a == "daioe") & (cor.b == "telework"), "corr"].iloc[0])
      > 0.5, "a 0.74 placebo is a weak placebo and the summary must say so")
summ = (s63.OUT / "63_summary.txt").read_text()
for must in ("PLACEBO TEST", "HORSE RACE", "READ THIS"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""
test_67_gender.py -- the gender differential must be a test, not two
                     numbers side by side.

Three worlds:
  BOTH    the same decline planted on young men and young women. The
          female differential must be near zero, because there is no
          difference to find, and the pooled young term must carry it.
  WOMEN   the decline planted on young women only. The differential must
          be negative and beyond two standard errors, and men near zero.
  NEITHER nothing planted. Everything near zero.

Also tested: the fixed effects are built on age AND sex, so the national
path of young women is absorbed; post x high alone is correctly absent
from the term list because the employer-by-month effect takes it; and
the SCB character coding of Kon is normalised.

    CANARIES_DRYRUN=1 python3 revision/local/test_67_gender.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries67_"))
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


s67 = load("67_gender_on_the_new_design.py", "s67")
s67.OUT = TMP / "out"; s67.OUT.mkdir(); s67.CACHE = mc.CACHE_DIR
s61 = s67._mod("61_redated_triple.py", "s61"); s61.OUT = s67.OUT
j47 = s61._j47(); h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s61.TRUNC)
Q4 = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
BETA = float(np.log(0.75))
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def _base():
    """One draw, thinned per world, so variants differ only where meant."""
    rng = np.random.default_rng(67)
    lam0 = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9, "41-49": 12,
            "50+": 15}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        for y in s67.YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                for age, lam in lam0.items():
                    for g in ("1", "2"):
                        rows.append((emp, ym, age, g,
                                     int(rng.poisson(lam / 2))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "gender", "n_emp"])


BASE = _base()


def counts(world):
    rng = np.random.default_rng(1967)
    c = BASE.copy()
    hit = (c["employer_id"].isin(Q4) & (c["age_group"] == "22-25")
           & (c["year_month"] >= "2024-01"))
    if world == "women":
        hit &= (c["gender"] == "2")
    if world != "neither":
        c.loc[hit, "n_emp"] = rng.binomial(c.loc[hit, "n_emp"], np.exp(BETA))
    c["n_emp"] = c["n_emp"].astype(int) + 1
    return c


def fit(world):
    skel = s67.build_skeleton_sex(counts(world), "22-25", j47, "n_emp")
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    b["high"] = (b["fq"] == 4).astype(int)
    b, terms = s67.add_gender_terms(b, "2024-01")
    r = mc.run_fepois_multi(b, s67.OUT, tag=f"t67_{world}", terms=terms,
                            fes=j47.FES)
    g = r.set_index("term")
    return (float(g.loc["post_x_high_x_young", "coef"]),
            float(g.loc["post_x_high_x_young_x_female", "coef"]),
            float(g.loc["post_x_high_x_young_x_female", "se"]), terms)


# ---- the term list --------------------------------------------------
_, _, _, terms = fit("neither")
check("post x high alone is absent, the employer-by-month effect takes it",
      not any(t in ("post_x_high", "rb_x_high") for t in terms),
      " ".join(terms))
check("the differential is in the term list",
      "post_x_high_x_young_x_female" in terms)
probe = s67.build_skeleton_sex(counts("both"), "22-25", j47, "n_emp")
check("the fixed effects are built on age AND sex",
      probe["fe_t_age"].nunique()
      == len(probe[["year_month", "age_group", "gender"]].drop_duplicates()),
      f"{probe['fe_t_age'].nunique()} codes")
check("the SCB character coding is normalised into a female indicator",
      set(probe["female"].unique()) == {0, 1}
      and (probe.loc[probe.gender == "2", "female"] == 1).all())

# ---- the three worlds ------------------------------------------------
m_b, d_b, s_b, _ = fit("both")
check("with the same decline on both sexes, the differential is near zero",
      abs(d_b) < 2 * s_b, f"differential {d_b:+.4f} (SE {s_b:.4f})")
check("and the pooled young term carries the decline",
      m_b < -0.10, f"{m_b:+.4f}")

m_w, d_w, s_w, _ = fit("women")
check("with the decline on women only, the differential is negative and "
      "significant",
      d_w < -2 * s_w, f"differential {d_w:+.4f} (SE {s_w:.4f}) "
                      f"t {d_w/max(s_w,1e-12):+.2f}")
check("and the male coefficient is near zero, as planted",
      abs(m_w) < 0.06, f"men {m_w:+.4f}")
check("so the design distinguishes a shared decline from a female one, "
      "which two separate regressions cannot",
      abs(d_b) < abs(d_w) / 2,
      f"both-sexes world {d_b:+.4f}, women-only world {d_w:+.4f}")

m_n, d_n, s_n, _ = fit("neither")
check("with nothing planted, neither term moves",
      abs(m_n) < 0.06 and abs(d_n) < 2 * s_n,
      f"young {m_n:+.4f}, differential {d_n:+.4f}")

# ---- end to end on the real entry point ------------------------------
# main() reaches SQL only when a cache misses, so writing the caches it
# expects exercises the whole entry point with the database forbidden.
# Script 58 died eighteen seconds into a MONA run in a block only main()
# reached, which is the reason this test exists at all.
c = counts("women")
for y in s67.YEARS:
    sub = c[c["year_month"].str.slice(0, 4) == str(y)]
    sub.to_parquet(mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)
    fl = sub.rename(columns={"n_emp": "n_hire"}).copy()
    fl["n_sep"] = fl["n_hire"]
    fl.to_parquet(mc.CACHE_DIR / f"flows_sex_{y}.parquet", index=False)


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s67._mod = lambda name, alias: s61 if name.startswith("61") else load(name, alias)
# main() asks 61 for a FRESH 47j instance, so narrowing the bands on our
# own copy does nothing unless 61 is told to hand ours back. The first
# version of this test set YOUNG_BANDS and then found both bands in the
# output, which is the same class of mistake as patching a module you did
# not import.
s61._j47 = lambda: j47
j47.YOUNG_BANDS = ["22-25"]
s67.main()
I = pd.read_csv(s67.OUT / "gender_interaction.csv")
check("main() writes the differential at both datings and all three margins",
      set(I["dating"]) == {"launch", "adoption"}
      and set(I["outcome"]) == {"stock", "hires", "seps"},
      f"{len(I)} rows")
check("and every fit came back ok", (I["status"] == "ok").all())
adopt = I[(I.dating == "adoption") & (I.arm == "true")
          & (I.outcome == "stock")
          & (I.term == "post_x_high_x_young_x_female")]
check("main() reproduces the women-only world it was fed",
      len(adopt) == 1 and float(adopt["coef"].iloc[0]) < -0.10,
      f"{float(adopt['coef'].iloc[0]):+.4f}" if len(adopt) else "missing")
check("the as-of arm ran, so the artefact is measured not assumed",
      "asof" in set(I["arm"]))
S = pd.read_csv(s67.OUT / "gender_by_sex.csv")
check("and the per-sex regressions are reported beside it",
      set(S["sex"]) == {"men", "women"}, f"{len(S)} rows")
summ = (s67.OUT / "67_summary.txt").read_text()
for must in ("FEMALE DIFFERENTIAL", "READ THIS", "not significant"):
    check(f"the summary states {must!r}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

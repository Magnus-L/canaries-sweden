#!/usr/bin/env python3
"""
test_76_77_tracks.py -- the decomposition must find composition where
                        composition was planted, and the track cut must
                        find a gradient only where one was planted.

The fixture gives every firm x band x sex cell three education tracks
(business 0.5, ict 0.3, other 0.2 of the cell's workers, women tilted
toward business), and plants, from January 2024 in exposed firms:

  * a decline on young WOMEN in the business track only, so the pooled
    female differential is negative, the business-track differential is
    larger than the pooled one, ict and other are near zero, and the
    within-track number (weighted by women's shares) is close to the
    pooled one: the rule must NOT return COMPOSITION when women are
    genuinely hit within their track;
  * a decline on BOTH sexes at 22-25 in the ict track only, so 77's
    ict contrast against 41-49 is negative and significant while the
    other tracks are nulls.

Also tested: the SQL is never reached when the caches are warm; the mix
table is floored and carries a mean exposure score; both summaries state
their gate; the tracks map from the delivered key's field codes.

    CANARIES_DRYRUN=1 python3 revision/local/test_76_77_tracks.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries7677_"))
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


s76 = load("76_gender_decomposition.py", "s76")
s76.OUT = TMP / "out76"; s76.OUT.mkdir(); s76.CACHE = mc.CACHE_DIR
s77 = load("77_contrast_by_track.py", "s77")
s77.OUT = TMP / "out77"; s77.OUT.mkdir(); s77.CACHE = mc.CACHE_DIR
s61 = s76._mod("61_redated_triple.py", "s61"); s61.OUT = s76.OUT
s67 = s76._mod("67_gender_on_the_new_design.py", "s67"); s67.OUT = s76.OUT
j47 = s61._j47(); h47 = j47._h47()

from _fixtures import Fixture  # noqa: E402

FIX = Fixture(mc, h47)
book, spec, frames = FIX.install_edu(j47.YEARS)
expo, _ = j47.incumbent_exposure(frames[2019], book, "OL_daioe", spec, "true",
                                 s61.TRUNC)
Q4 = set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))
KEY = h47.load_key()
BETA = float(np.log(0.70))
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def code_for(track):
    """A real (niva, inr) pair from the delivered key, in the given track."""
    two = {"ict": "48", "business_law_social": "34", "other": "21"}[track]
    k = KEY[KEY["inr"].str.slice(0, 2) == two]
    k = k[k["niva"].str.slice(0, 1).isin(["3", "5"])]
    r = k.iloc[0]
    return str(r["niva"]), str(r["inr"])


CODES = {t: code_for(t) for t in ("ict", "business_law_social", "other")}
SHARES = {"1": {"business_law_social": 0.40, "ict": 0.40, "other": 0.20},
          "2": {"business_law_social": 0.60, "ict": 0.20, "other": 0.20}}


def counts():
    rng = np.random.default_rng(7677)
    lam0 = {"22-25": 14, "26-30": 12, "31-34": 8, "35-40": 10, "41-49": 14,
            "50+": 16}
    rows = []
    for emp in range(1, FIX.n_firms + 1):
        hit = emp in Q4
        for y in s76.YEARS:
            for m in range(1, 13 if y < 2025 else 7):
                ym = f"{y}-{m:02d}"
                post = ym >= s76.POST_FROM
                for age, lam in lam0.items():
                    for g in ("1", "2"):
                        for track, sh in SHARES[g].items():
                            lam_ = lam * sh
                            if hit and post and age == "22-25":
                                if g == "2" and track == "business_law_social":
                                    lam_ *= np.exp(BETA)
                                if track == "ict":
                                    lam_ *= np.exp(BETA)
                            niva, inr = CODES[track]
                            rows.append((emp, ym, age, g, niva, inr,
                                         int(rng.poisson(lam_)) + 1))
    return pd.DataFrame(rows, columns=s76.EDU_COLS + ["n_emp"])


C = counts()
for y in s76.YEARS:
    sub = C[C["year_month"].str.slice(0, 4) == str(y)]
    s76.compact(sub.copy()).to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_edu_{y}.parquet", index=False)

# ---- the mapping --------------------------------------------------------
T = s76.track_of(pd.Series([CODES["ict"][1], CODES["business_law_social"][1],
                            CODES["other"][1], None, "5210"]))
check("tracks map from the two-digit field of the delivered key",
      list(T) == ["ict", "business_law_social", "other", "na", "engineering"],
      " ".join(map(str, T)))
L = s76.level_of(pd.Series(["310", "520", "100", None]))
check("levels map from the first digit of the SUN level",
      list(L) == ["upper_secondary", "post_secondary",
                  "below_upper_secondary", "na"], " ".join(map(str, L)))


def boom(*a, **k):
    raise AssertionError("SQL attempted although the cache is warm")


mc.connect = boom
s76._mod = lambda name, alias: (s61 if name.startswith("61")
                                else s67 if name.startswith("67")
                                else load(name, alias))
s77._mod = lambda name, alias: (s61 if name.startswith("61")
                                else s76 if name.startswith("76")
                                else load(name, alias))
s61._j47 = lambda: j47

# ---- 76 ---------------------------------------------------------------
s76.S68_DIFF = (float("nan"), 0.0131)   # no real 68 to reproduce here; gate reports
s76.main()
M = pd.read_csv(s76.OUT / "education_mix_by_sex.csv")
check("the mix table has both dimensions, both sexes, both firm groups",
      set(M["dimension"]) == {"track", "level"} and set(M["gender"]) == {"men", "women"}
      and set(M["exposed"]) == {0, 1})
check("and it is floored", (M["persons_avg"] >= s76.EXPORT_FLOOR).all())
# The synthetic score book scores only the 42 key cells the fixture drew,
# so a track cell built from codes outside them has nothing to average.
# What the code guarantees is weaker and is what is checked: the scored
# share is reported for every cell, and the mean exists wherever any
# worker in the cell was scored.
trk = M[M["dimension"] == "track"]
check("every track cell reports its scored share, and a mean wherever one exists",
      trk["scored_share"].between(0, 1).all()
      and trk.loc[trk["scored_share"] > 0, "mean_score"].notna().all()
      and trk.loc[trk["scored_share"] == 0, "mean_score"].isna().all(),
      f"scored shares {sorted(set(trk['scored_share'].round(2)))}")
wom = M[(M.dimension == "track") & (M.exposed == 1) & (M.gender == "women")]
wshare = dict(zip(wom["cell"], wom["share"]))
check("young women's business share in exposed firms is the planted 0.6",
      abs(wshare.get("business_law_social", 0) - 0.60) < 0.05,
      f"{wshare.get('business_law_social', float('nan')):.3f}")
G = pd.read_csv(s76.OUT / "gender_by_track.csv").set_index("track")
check("the pooled and every planted track were fitted",
      {"all", "ict", "business_law_social", "other"} <= set(G.index),
      " ".join(G.index))
d_all, s_all = float(G.loc["all", "diff"]), float(G.loc["all", "diff_se"])
d_bus, s_bus = (float(G.loc["business_law_social", "diff"]),
                float(G.loc["business_law_social", "diff_se"]))
d_ict = float(G.loc["ict", "diff"])
check("the pooled female differential is negative and significant",
      d_all < -2 * s_all, f"{d_all:+.4f} ({s_all:.4f})")
check("the business-track differential is larger than the pooled one",
      d_bus < d_all and d_bus < -2 * s_bus, f"business {d_bus:+.4f}, pooled {d_all:+.4f}")
check("and the ict differential is near zero (only both-sex decline planted there)",
      abs(d_ict) < 3 * float(G.loc["ict", "diff_se"]), f"{d_ict:+.4f}")
S = pd.read_csv(s76.OUT / "gender_split.csv").iloc[0]
check("the split is written with within, composition and the ratio",
      all(k in S.index for k in ("within", "composition", "ratio_within", "verdict")))
check("women genuinely hit within their track is NOT read as composition",
      S["verdict"] != "COMPOSITION",
      f"ratio {S['ratio_within']:.3f}, verdict {S['verdict']}")
summ = (s76.OUT / "76_summary.txt").read_text()
for must in ("REPRODUCTION GATE", "THE SPLIT", "carried forward", "READ THIS"):
    check(f"76's summary states {must!r}", must in summ)
check("76 wrote a covariance per fit",
      len(list(s76.OUT.glob("vcov_s76_*.csv"))) >= 4)

# ---- 77 ---------------------------------------------------------------
s77.S74_2225 = (float("nan"), 0.0126)
s77.main()
K = pd.read_csv(s77.OUT / "contrast_by_track.csv")
k22 = K[K["band_vs_ref"] == "22-25"].set_index("track")
check("77 reports 22-25 against 41-49 for the pooled panel and every track",
      {"all", "ict", "business_law_social", "other"} <= set(k22.index),
      " ".join(k22.index))
c_ict, s_ict = float(k22.loc["ict", "coef"]), float(k22.loc["ict", "se"])
c_oth, s_oth = float(k22.loc["other", "coef"]), float(k22.loc["other", "se"])
check("the ict contrast is negative and significant, as planted",
      c_ict < -2 * s_ict, f"{c_ict:+.4f} ({s_ict:.4f})")
check("and the other-track contrast is a null, as planted",
      abs(c_oth) < 2 * s_oth, f"{c_oth:+.4f} ({s_oth:.4f})")
check("the pooled contrast sits between them",
      c_ict < float(k22.loc["all", "coef"]) < c_oth + 2 * s_oth,
      f"all {float(k22.loc['all', 'coef']):+.4f}")
summ7 = (s77.OUT / "77_summary.txt").read_text()
for must in ("REPRODUCTION GATE", "heterogeneity", "READ THIS"):
    check(f"77's summary states {must!r}", must in summ7)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

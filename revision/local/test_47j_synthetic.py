#!/usr/bin/env python3
"""
test_47j_synthetic.py -- 47j end to end locally, real key/DAIOE/Eloundou and
real R + fixest, synthetic year frames in a fake 47h cache. Cases:
  1 exposure uses INCUMBENTS ONLY: adding or removing young workers, or
    corrupting every young education record, leaves it bit-identical
  2 the three fixed-effect sets absorb what they must, so the triple
    interaction is the only surviving treatment term
  3 a planted young-in-exposed-firms decline is recovered with the right sign
  4 end to end with SQL forbidden; exports floored; the summary states the
    estimand and what it cannot see
    CANARIES_DRYRUN=1 python3 revision/local/test_47j_synthetic.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries47j_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
spec_j = importlib.util.spec_from_file_location("s47j", MONA / "47j_within_employer_triple.py")
mod = importlib.util.module_from_spec(spec_j); spec_j.loader.exec_module(mod)
mod.OUT = TMP / "output_47j"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR
h47 = mod._h47()

RNG = np.random.default_rng(147)
KEY = h47.load_key()
D = pd.read_stata(SHARE / "daioe_quartiles.dta")
HI = D.loc[D.high_exposure == 1, "ssyk4"].astype(str).str.zfill(4).to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].astype(str).str.zfill(4).to_numpy()
ter = KEY[KEY["niva"].str[:1].isin(["4", "5", "6"])].sample(30, random_state=1)
gym = KEY[KEY["niva"].str[:1] == "3"].sample(12, random_state=2)
CELLS = pd.concat([ter, gym]).reset_index(drop=True)
AGE_W = {"22-25": 0.10, "26-30": 0.13, "31-34": 0.10, "35-40": 0.14,
         "41-49": 0.23, "50+": 0.30}
EXPOSED = set(range(1, 61))          # firms 1-60 are the exposed ones
SHOCK = 0.65                          # young employment in exposed firms after GPT


def weights_frame(year):
    RNG = np.random.default_rng(1000 + year)      # deterministic per year
    rows = []
    for _, c in CELLS.iterrows():
        hi = c["niva"][:1] in "456"
        for code in np.concatenate([RNG.choice(HI, 4), RNG.choice(LO, 4)]):
            base = 500 if ((code in set(HI)) == hi) else 120
            for band in h47.EXP_BANDS:
                rows.append((c["niva"], c["inr"], code, 1, band, 1,
                             int(RNG.poisson(base)) + 1))
    return pd.DataFrame(rows, columns=["niva", "inr", "ssyk4", "fresh",
                                       "expband", "young", "n"])


def year_frame(year, corrupt_young=False, drop_young=False, shock=True,
               shock_firms=None):
    # One seed per YEAR only: two variants of the same year therefore differ
    # in exactly the thing being varied and in nothing else, which is what
    # makes the incumbent-invariance test meaningful.
    RNG = np.random.default_rng(2000 + year)
    ters = CELLS[CELLS["niva"].str[:1].isin(list("456"))]
    gyms = CELLS[CELLS["niva"].str[:1] == "3"]
    rows = []
    for emp in range(1, 141):
        pool = ters if emp in EXPOSED else gyms
        mix = pool.sample(min(4, len(pool)), random_state=emp % 97)
        for m in range(1, 13):
            ym = f"{year}-{m:02d}"
            post = ym >= "2022-12"
            for _, c in mix.iterrows():
                for age, w in AGE_W.items():
                    # Draw for EVERY age, then skip: dropping the young with a
                    # `continue` before the draw shifts the RNG sequence, so the
                    # incumbents differ too and the invariance test measures the
                    # fixture rather than the design.
                    lam = 40 * w
                    tgt = EXPOSED if shock_firms is None else shock_firms
                    if shock and post and age == "22-25" and emp in tgt:
                        lam *= SHOCK
                    n_emp = int(RNG.poisson(lam)) + 1
                    if drop_young and age in ("22-25", "26-30"):
                        continue
                    rec = dict(employer_id=emp, year_month=ym, age_group=age,
                               niva_t=c["niva"], inr_t=c["inr"], expb_t="3-5",
                               n_emp=n_emp)
                    for T in (2021, 2022):
                        bad = corrupt_young and age in ("22-25", "26-30")
                        g = gyms.iloc[(emp + m) % len(gyms)] if bad else c
                        rec[f"niva_{T%100}"], rec[f"inr_{T%100}"] = g["niva"], g["inr"]
                        rec[f"expb_{T%100}"] = "3-5"
                        rec[f"enr_{T%100}"] = None
                    # 47h's pull now also returns the legacy (47b) cascade
                    # columns. This script never uses that arm, so they alias
                    # the corrected ones; the gate in 47h is where they differ.
                    rec["niva_21g"], rec["inr_21g"] = rec["niva_21"], rec["inr_21"]
                    rows.append(rec)
    return h47.compact(pd.DataFrame(rows)[h47.YEAR_COLS + ["n_emp"]])


def q4_firms():
    """The firms the classifier actually calls Q4. The shock has to be planted
    on THOSE, not on a set that merely correlates with them: a shock spread
    over Q3 and Q4 is diluted by the controls and the recovered coefficient
    would understate the truth for a reason that has nothing to do with 47j."""
    book, sp, frames = setup(shock=False)
    expo, _ = mod.incumbent_exposure(frames[2019], book, "OL_daioe", sp, "true", 2021)
    return set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))


def setup(**kw):
    for y in (2019, 2020, 2021):
        weights_frame(y).to_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet", index=False)
    for y in mod.YEARS:
        year_frame(y, **kw).to_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet", index=False)
    counts = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet")
              for y in (2019, 2020, 2021)}
    h47.MIN_CELL = 10
    book = h47.ScoreBook(counts, KEY, h47.load_scores())
    sp = dict(h47.DESIGNS["OL_daioe"]); book.build("OL_daioe", sp)
    frames = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet") for y in mod.YEARS}
    return book, sp, frames


def test_incumbents_only():
    book, sp, frames = setup()
    base, _ = mod.incumbent_exposure(frames[2019], book, "OL_daioe", sp, "true", 2021)
    _, _, f_corrupt = setup(corrupt_young=True)
    corrupt, _ = mod.incumbent_exposure(f_corrupt[2019], book, "OL_daioe", sp, "asof", 2021)
    j = base.merge(corrupt, on="employer_id", suffixes=("_b", "_c"))
    assert len(j) > 50
    assert (j["fq_b"] == j["fq_c"]).all(), j[j.fq_b != j.fq_c].head()
    assert np.allclose(j["mix_b"], j["mix_c"]), "incumbent mix moved"
    _, _, f_nokids = setup(drop_young=True)
    nokids, _ = mod.incumbent_exposure(f_nokids[2019], book, "OL_daioe", sp, "true", 2021)
    k = base.merge(nokids, on="employer_id", suffixes=("_b", "_n"))
    assert (k["fq_b"] == k["fq_n"]).all()
    print(f"PASS exposure uses incumbents only: corrupting EVERY young education "
          f"record leaves all {len(j)} firms' quartiles and mixes identical, and so "
          f"does deleting the young entirely")


def test_absorption_and_sign():
    tgt = q4_firms()
    book, sp, frames = setup(shock_firms=tgt)
    expo, _ = mod.incumbent_exposure(frames[2019], book, "OL_daioe", sp, "true", 2021)
    bal = mod.build_panel(frames, expo, "22-25")
    assert not bal.empty
    for fe, term in (("fe_emp_t", "post_gpt"), ("fe_emp_age", "high"),
                     ("fe_t_age", "young")):
        assert fe in bal.columns
    # the two-way pieces are collinear with the FE sets; the triple is not
    g = bal.groupby("fe_emp_t")["post_gpt_x_high_x_young"].nunique()
    assert (g > 1).any(), "the triple must vary within employer x month"
    r = mod.fit(bal, "t2_sign")
    assert r["status"] == "ok", r
    assert r["gamma3"] < 0, r
    assert abs(r["gamma3"] - np.log(SHOCK)) < 0.10, (r["gamma3"], np.log(SHOCK))
    print(f"PASS the planted decline is recovered: gamma3 {r['gamma3']:+.4f} "
          f"against a planted log({SHOCK}) = {np.log(SHOCK):+.4f} on the "
          f"{len(tgt)} firms the classifier calls Q4, and it varies within "
          f"employer x month as a triple difference must")


def test_end_to_end():
    setup()
    def boom(*a, **k):
        raise AssertionError("SQL attempted although the cache is warm")
    mc.connect = boom
    mod.YOUNG_BANDS = ["22-25"]
    mod.main()
    est = pd.read_csv(mod.OUT / "triple_estimates.csv")
    assert len(est) == 2 * 2 * 2 * 1, len(est)
    assert est["gamma3"].notna().all() and (est["status"] == "ok").all()
    summ = (mod.OUT / "47j_summary.txt").read_text()
    for must in ("INCUMBENTS", "WITHIN one employer", "would NOT appear here", "READ RULE"):
        assert must in summ, must
    q = pd.read_csv(mod.OUT / "triple_quartile_sizes.csv")
    v = q["firms"].dropna()
    assert ((v == 0) | (v >= 5)).all()
    print(f"PASS end to end with no SQL: {len(est)} fits, all ok, exports floored, "
          f"summary states the estimand and its blind spot")


def test_dead_cells_are_free():
    """
    On 20 Sep every 22-25 fit crashed R with an access violation on a
    39-million-row panel, while the LARGER 26-30 panel at 45 million
    succeeded. Zero-filling five age bands over seven years creates a
    great many cells that are empty for the life of the panel, and the
    youngest band is where employers most often have none.

    fixest discards such cells internally, because under an employer x
    age fixed effect a cell that is zero throughout has that effect at
    minus infinity and contributes nothing to any other parameter. So
    removing them before they reach R is free. This checks that it really
    is free rather than merely plausible.
    """
    book, sp, frames = setup()
    expo, _ = mod.incumbent_exposure(frames[2019], book, "OL_daioe", sp,
                                     "true", 2021)
    bal = mod.build_panel(frames, expo, "22-25")
    assert not bal.empty, "the fixture produced no panel"
    tot = bal.groupby(["employer_id", "age_group"], observed=True)["n_emp"].sum()
    assert (tot > 0).all(), "a cell that is zero throughout survived"
    bands = bal.groupby("employer_id", observed=True)["age_group"].nunique()
    assert (bands >= 2).all(), "an employer with one band survived"

    real = mod._drop_dead_cells
    mod._drop_dead_cells = lambda b: b
    try:
        full = mod.build_panel(frames, expo, "22-25")
    finally:
        mod._drop_dead_cells = real

    # The fixture as built has no dead cells, so comparing it with itself
    # would prove nothing. Create some: empty the 41-49 band entirely for
    # a third of employers, exactly the pattern that inflates the real
    # panel. The drop must remove them and the estimate must not move.
    emp = sorted(full["employer_id"].unique())[::3]
    killed = full["employer_id"].isin(emp) & (full["age_group"] == "41-49")
    full = full.copy()
    full.loc[killed, "n_emp"] = 0
    trimmed = mod._drop_dead_cells(full)
    assert len(trimmed) < len(full), "the drop removed nothing on a panel "\
                                     "built to contain dead cells"
    a = mod.fit(trimmed, "dead_trimmed")
    b = mod.fit(full, "dead_kept")
    assert np.isfinite(a["gamma3"]) and np.isfinite(b["gamma3"]), (a, b)
    assert abs(a["gamma3"] - b["gamma3"]) < 1e-6, (a["gamma3"], b["gamma3"])
    print(f"PASS dropping dead cells changes nothing: {len(full):,} rows -> "
          f"{len(trimmed):,} ({1-len(trimmed)/len(full):.0%} removed), "
          f"gamma3 {b['gamma3']:+.6f} -> {a['gamma3']:+.6f}")


if __name__ == "__main__":
    test_incumbents_only()
    test_absorption_and_sign()
    test_dead_cells_are_free()
    test_end_to_end()
    print(f"\nALL PASS  (tmp: {TMP})")

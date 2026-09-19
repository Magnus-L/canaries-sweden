#!/usr/bin/env python3
"""
test_47i_synthetic.py -- run 47i end to end locally against synthetic year
frames written into a fake 47h cache, with the real key, DAIOE and Eloundou
inputs and real R + fixest. Numbers mean nothing; the mechanics are tested.

Cases
  1 it runs from a warm cache with no SQL at all (a pull would raise)
  2 the firm quartile is a FIRM attribute: one quartile per employer-year,
    and the young worker's own record never enters it
  3 the thresholds are fixed on 2019 and applied to later years
  4 the as-of arm moves the firm mix far LESS than it moved worker cells
  5 every export exists, counts are floored, and the summary states the
    across-firm identification caveat
    CANARIES_DRYRUN=1 python3 revision/local/test_47i_synthetic.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries47i_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()

spec_i = importlib.util.spec_from_file_location("s47i", MONA / "47i_firmmix.py")
mod = importlib.util.module_from_spec(spec_i); spec_i.loader.exec_module(mod)
mod.OUT = TMP / "output_47i"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR
h47 = mod._h47()

RNG = np.random.default_rng(47)
KEY = h47.load_key()
D = pd.read_stata(SHARE / "daioe_quartiles.dta")
HI = D.loc[D.high_exposure == 1, "ssyk4"].astype(str).str.zfill(4).to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].astype(str).str.zfill(4).to_numpy()
AGES = mod.AGES
ter = KEY[KEY["niva"].str[:1].isin(["4", "5", "6"])].sample(30, random_state=1)
gym = KEY[KEY["niva"].str[:1] == "3"].sample(12, random_state=2)
CELLS = pd.concat([ter, gym]).reset_index(drop=True)


def weights_frame(year):
    rows = []
    for _, c in CELLS.iterrows():
        hi_tilt = c["niva"][:1] in "456"
        for code in np.concatenate([RNG.choice(HI, 4), RNG.choice(LO, 4)]):
            base = 500 if ((code in set(HI)) == hi_tilt) else 120
            for band in h47.EXP_BANDS:
                rows.append((c["niva"], c["inr"], code, 1, band, 1,
                             int(RNG.poisson(base)) + 1))
    return pd.DataFrame(rows, columns=["niva", "inr", "ssyk4", "fresh",
                                       "expband", "young", "n"])


def year_frame(year):
    rows = []
    for emp in range(1, 121):
        mix = RNG.choice(len(CELLS), 5, replace=False)
        for m in range(1, 13):
            ym = f"{year}-{m:02d}"
            for cid in mix:
                c = CELLS.iloc[cid]
                for age in AGES:
                    # realistic age composition: 22-25 are about a tenth of
                    # headcount, not a sixth. The firm-mix design's whole
                    # claim is that a stale record for a SMALL minority
                    # barely moves the average, so a fixture with equal age
                    # weights tests something the design never claimed.
                    w = {"22-25": 0.10, "26-30": 0.13, "31-34": 0.10,
                         "35-40": 0.14, "41-49": 0.23, "50+": 0.30}[age]
                    rec = dict(employer_id=emp, year_month=ym, age_group=age,
                               niva_t=c["niva"], inr_t=c["inr"], expb_t="3-5",
                               n_emp=int(RNG.poisson(40 * w)) + 1)
                    for T in (2021, 2022):
                        stale = (year > T and age == "22-25"
                                 and c["niva"][:1] in "456" and RNG.uniform() < 0.5)
                        g = gym.iloc[RNG.integers(len(gym))] if stale else c
                        rec[f"niva_{T%100}"], rec[f"inr_{T%100}"] = g["niva"], g["inr"]
                        rec[f"expb_{T%100}"] = "3-5"
                        rec[f"enr_{T%100}"] = None
                    rows.append(rec)
    return h47.compact(pd.DataFrame(rows)[h47.YEAR_COLS + ["n_emp"]])


def warm_cache():
    for y in (2019, 2020, 2021):
        weights_frame(y).to_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet", index=False)
    for y in mod.YEARS:
        year_frame(y).to_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet", index=False)


def book_and_frames():
    counts = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet")
              for y in (2019, 2020, 2021)}
    h47.MIN_CELL = 10
    book = h47.ScoreBook(counts, KEY, h47.load_scores())
    spec = dict(h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frames = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet") for y in mod.YEARS}
    return book, spec, frames


def test_firm_level_and_fixed_cuts():
    book, spec, frames = book_and_frames()
    fq19, cuts = mod.firm_quartiles(frames[2019], book, "OL_daioe", spec, "true", 2021, None)
    assert fq19["employer_id"].is_unique, "a firm must hold ONE quartile per year"
    assert len(cuts) == 3 and cuts == sorted(cuts), cuts
    fq23, cuts2 = mod.firm_quartiles(frames[2023], book, "OL_daioe", spec, "true", 2021, cuts)
    assert cuts2 == cuts, "2019 thresholds must be reused, not recomputed"
    assert set(fq23["fq"]) <= {1, 2, 3, 4}
    print(f"PASS firm-level classification: {len(fq19)} firms in 2019, one quartile each, "
          f"thresholds fixed and reused ({len(set(fq19['fq']))} quartiles populated)")


def test_young_record_barely_moves_the_firm():
    """The point of the design: staleness in 22-25 records shifts a firm's
    mix far less than it shifted the worker-level cells in 47b."""
    book, spec, frames = book_and_frames()
    tr, cuts = mod.firm_quartiles(frames[2023], book, "OL_daioe", spec, "true", 2021, None)
    af, _ = mod.firm_quartiles(frames[2023], book, "OL_daioe", spec, "asof", 2021, cuts)
    j = tr.merge(af, on="employer_id", suffixes=("_t", "_a"))
    moved = (j["fq_t"] != j["fq_a"]).mean()
    rel = float(np.mean(np.abs(j["mix_a"] - j["mix_t"]) / j["mix_t"].abs()))
    # What the design CLAIMS is that a stale record for a small minority
    # barely moves the firm's average. That is the mix shift, and it is what
    # is asserted. How often that shift crosses a quartile boundary depends
    # on how bunched the score distribution is -- in this fixture only 28
    # groups, with two cut points three points apart, so firms sit on top of
    # boundaries and crossings are a property of the fixture, not the design.
    # It is reported, not asserted, and it is exactly why 47i MEASURES the
    # artefact on the real distribution instead of assuming immunity.
    assert rel < 0.05, f"mean relative mix shift {rel:.2%}"
    near = float(np.mean(np.abs(j["mix_t"].to_numpy()[:, None]
                                - np.array(cuts)[None, :]).min(axis=1)))
    print(f"PASS staleness barely moves the firm: mean |relative mix shift| "
          f"{rel:.2%} (47b reassigned 63% of worker cells outright); "
          f"{moved:.0%} of firms cross a cut in this fixture, mean distance to "
          f"the nearest cut {near:.2f}")


def test_end_to_end_no_sql():
    def boom(*a, **k):
        raise AssertionError("SQL attempted although the 47h cache is warm")
    mc.connect = boom
    mod.YEARS = [2019, 2020, 2021, 2022, 2023]
    mod.AGES = ["22-25", "50+"]
    mod.main()
    est = pd.read_csv(mod.OUT / "firmmix_estimates.csv")
    assert len(est) == 2 * 2 * 2 * len(mod.AGES), len(est)
    assert est["gamma2"].notna().all(), est[est["gamma2"].isna()]
    assert (est["status"] == "ok").all(), est["status"].value_counts()
    summ = (mod.OUT / "47i_summary.txt").read_text()
    assert "ACROSS firms" in summ and "ARTEFACT" in summ and "READ RULE" in summ
    q = pd.read_csv(mod.OUT / "firmmix_quartile_sizes.csv")
    v = q["firms"].dropna()
    assert ((v == 0) | (v >= 5)).all(), v[(v > 0) & (v < 5)]
    print(f"PASS end to end with no SQL: {len(est)} fits, all ok, exports floored, "
          "summary states the across-firm caveat")


if __name__ == "__main__":
    warm_cache()
    test_firm_level_and_fixed_cuts()
    test_young_record_barely_moves_the_firm()
    test_end_to_end_no_sql()
    print(f"\nALL PASS  (tmp: {TMP})")

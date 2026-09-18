#!/usr/bin/env python3
"""
test_50_synthetic.py -- run 50_sim_moments end to end locally, no MONA.
Synthetic frames at the query boundary; real key and DAIOE inputs; asserts
every export exists, every exported count is 0 or >= 5, the two pandas
collapses are right on hand-built frames, and one failing query costs that
moment, never the job.
    CANARIES_DRYRUN=1 python3 revision/local/test_50_synthetic.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries50_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
spec = importlib.util.spec_from_file_location("s50", MONA / "50_sim_moments.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

RNG = np.random.default_rng(50)
KEY = mod.load_key()
DAIOE = pd.read_stata(SHARE / "daioe_quartiles.dta")
SSYK = DAIOE["ssyk4"].astype(str).str.zfill(4).to_numpy()
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
BANDS = ["0-2", "3-5", "6-10", "11-20", "21+", "na"]
CELLS = KEY.sample(50, random_state=1)


def counts(n_rows, lo=0, hi=40):
    return RNG.integers(lo, hi, n_rows)


def f_completion(y, conn):
    lv = RNG.choice(list("23456"), 300); age = RNG.integers(16, 60, 300)
    return pd.DataFrame({"level": lv, "exam_age": age, "n": counts(300)})


def f_level_change(t, conn):
    return pd.DataFrame({"age_group": RNG.choice(AGES[:3], 200),
                         "level_t": RNG.choice(list("3456"), 200),
                         "level_t1": RNG.choice(list("3456"), 200), "n": counts(200)})


def f_occ_change(t, conn):
    n = 3000
    return pd.DataFrame({"age_group": RNG.choice(AGES, n), "expband": RNG.choice(BANDS, n),
                         "ssyk4_t": RNG.choice(SSYK, n), "ssyk4_t1": RNG.choice(SSYK, n),
                         "n": counts(n)})


def f_stale(y, conn):
    return pd.DataFrame({"age_group": RNG.choice(AGES, 300), "stale_years": RNG.integers(0, 8, 300),
                         "ssyk_status": RNG.choice(["1", "2", ""], 300), "n": counts(300)})


def f_enrol(y, conn):
    return pd.DataFrame({"age_group": RNG.choice(AGES[:2], 60), "tertiary": RNG.integers(0, 2, 60),
                         "registered": RNG.integers(0, 2, 60), "n": counts(60)})


def f_switch(conn):
    return pd.DataFrame({"exam_year": RNG.integers(2019, 2024, 40),
                         "match": RNG.choice(["same", "same3", "different", "none"], 40), "n": counts(40)})


def f_size(conn):
    b = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-999", "1000+"]
    return pd.DataFrame({"size_band": b, "n": counts(8, 5, 500), "persons": counts(8, 100, 9000)})


def f_matrix(y, conn):
    n = 4000
    c = CELLS.sample(n, replace=True, random_state=y)
    return pd.DataFrame({"niva": c["niva"].to_numpy(), "inr": c["inr"].to_numpy(),
                         "ssyk4": RNG.choice(SSYK, n), "fresh": RNG.integers(0, 2, n),
                         "expband": RNG.choice(BANDS, n), "n": counts(n)})


def f_validation(t, k, conn):
    n = 5000
    c = CELLS.sample(n, replace=True, random_state=t * 10 + k)
    ter = c["niva"].str[:1].isin(["4", "5", "6"]).to_numpy()
    enr = np.where(ter, None, RNG.choice(list(CELLS["inr"].unique()) + [None], n))
    return pd.DataFrame({"age_group": RNG.choice(AGES, n), "niva_lag": c["niva"].to_numpy(),
                         "inr_lag": c["inr"].to_numpy(), "expband_lag": RNG.choice(BANDS, n),
                         "tertiary_lag": ter.astype(int), "enr_inr": enr,
                         "ssyk4_t": RNG.choice(SSYK, n), "n": counts(n)})


def wire(tmp, fail=None):
    sys.stdout = sys.__stdout__
    out = tmp / "output_50"; out.mkdir(); mod.OUT = out
    mod.CACHE = tmp / "cache"; mod.CACHE.mkdir()
    mc.connect = lambda: object()
    mod.q_completion_age, mod.q_level_change, mod.q_occ_change = f_completion, f_level_change, f_occ_change
    mod.q_staleness, mod.q_enrolment, mod.q_field_switch = f_stale, f_enrol, f_switch
    mod.q_employer_size, mod.q_matrix = f_size, f_matrix
    mod.q_validation = f_validation
    if fail:
        def boom(*a, **k):
            raise RuntimeError("synthetic query failure")
        setattr(mod, fail, boom)
    return out


def test_collapses():
    daioe = mod.load_daioe()
    raw = pd.DataFrame({"age_group": ["22-25"] * 3, "expband": ["0-2"] * 3,
                        "ssyk4_t": ["2512", "2512", "4110"], "ssyk4_t1": ["2512", "1120", "4110"],
                        "n": [10, 6, 7]})
    m2 = mod.m2_collapse(raw, daioe)
    assert m2["n"].sum() == 23
    assert m2.loc[m2["enters_mgr"] == 1, "n"].sum() == 6, m2
    assert m2.loc[m2["same_code"] == 1, "n"].sum() == 17, m2
    k = KEY.iloc[0]
    raw6 = pd.DataFrame({"niva": [k["niva"].upper() + " ", "999"], "inr": [k["inr"], "zzzz"],
                         "ssyk4": ["2512", "2512"], "fresh": [1, 1], "expband": ["0-2", "0-2"], "n": [9, 8]})
    m6 = mod.m6_collapse(raw6, KEY)
    assert set(m6["grp"]) == {k["grp"], "unmatched"}, m6
    m6b = mod.m6b_collapse(pd.DataFrame({"niva": ["536", "310"], "inr": ["3440", "3440"],
                                         "ssyk4": ["2512", "2512"], "fresh": [1, 1],
                                         "expband": ["0-2", "0-2"], "n": [5, 7]}))
    assert m6b["n"].sum() == 5, m6b            # only the tertiary row survives
    raw7 = pd.DataFrame({"age_group": ["22-25"] * 2, "niva_lag": [k["niva"], None],
                         "inr_lag": [k["inr"], None], "expband_lag": ["0-2", None],
                         "tertiary_lag": [1, 0], "enr_inr": [None, "3440"],
                         "ssyk4_t": ["2512", "2512"], "n": [6, 9]})
    m7 = mod.m7_collapse(raw7, KEY)
    assert set(m7["grp_lag"]) == {k["grp"], "unmatched"} and set(m7["enr_inr"]) == {"none", "3440"}, m7
    print("PASS collapses: m2 flags, m6 key join, m6b tertiary filter, m7 unmatched/none handling")


def test_happy():
    tmp = TMP / "happy"; tmp.mkdir(); out = wire(tmp)
    mod.main()
    expected = ["m1a_completion_age.csv", "m1b_level_change.csv", "m2_occ_change.csv",
                "m3_staleness.csv", "m4a_enrolment_prevalence.csv", "m4b_field_switch.csv",
                "m5a_employer_size.csv"] + [f"m6_matrix_{y}.csv" for y in mod.YEARS] \
               + [f"m6b_inr_tertiary_{y}.csv" for y in mod.YEARS] \
               + [f"m7_validation_t{t}_k{k}.csv" for t in (2021, 2022, 2023) for k in (0, 2)]
    for f in expected:
        assert (out / f).exists(), f
        df = pd.read_csv(out / f)
        col = "n"
        vals = df[col].dropna()
        assert ((vals == 0) | (vals >= 5)).all(), (f, vals[(vals > 0) & (vals < 5)].head())
    assert not (out / "m5b_retention_22_25.csv").exists()   # no 47h cache here
    assert (out / "50_summary.txt").exists() and (out / "50_log.txt").exists()
    print(f"PASS happy path: {len(expected)} exports, every count 0 or >= 5")


def test_one_failure_costs_one_moment():
    tmp = TMP / "fail"; tmp.mkdir(); out = wire(tmp, fail="q_field_switch")
    mod.main()
    assert not (out / "m4b_field_switch.csv").exists()
    assert (out / "m6_matrix_2023.csv").exists() and (out / "50_summary.txt").exists()
    assert "M4b: FAILED" in (out / "50_log.txt").read_text()
    print("PASS failure path: one bad query, every other moment written")


if __name__ == "__main__":
    test_collapses(); test_happy(); test_one_failure_costs_one_moment()
    print(f"\nALL PASS  (tmp: {TMP})")
    print(f"HAPPY_OUT={TMP / 'happy' / 'output_50'}")

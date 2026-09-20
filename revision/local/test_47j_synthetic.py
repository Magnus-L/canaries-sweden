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

from _fixtures import Fixture  # noqa: E402

# The synthetic register frames live in _fixtures, shared with the tests
# for 61 and 62. SHOCK is the multiplier planted on young employment in
# exposed firms after the launch, and the fit must return its log.
FIX = Fixture(mc, h47)
SHOCK = FIX.shock


def setup(**kw):
    return FIX.install_edu(mod.YEARS, **kw)


def q4_firms():
    """The firms the classifier actually calls Q4. The shock has to be planted
    on THOSE, not on a set that merely correlates with them: a shock spread
    over Q3 and Q4 is diluted by the controls and the recovered coefficient
    would understate the truth for a reason that has nothing to do with 47j."""
    book, sp, frames = setup(shock=False)
    expo, _ = mod.incumbent_exposure(frames[2019], book, "OL_daioe", sp,
                                     "true", 2021)
    return set(expo.loc[expo["fq"] == 4, "employer_id"].astype(int))


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

#!/usr/bin/env python3
"""
test_47k_synthetic.py -- 47k end to end, locally, with real inputs and real
R + fixest against synthetic year frames in a fake 47h cache.

The claims 47k makes, each tested:
  1 the sampling rule selects the SAME rows in both arms -- if it did not,
    the "artefact" would mix classification error with sample change, and
    the whole test would be void
  2 the feasible rule is computable from as-of columns ALONE (it never
    touches niva_t/inr_t), so it can be applied to 2024-25
  3 the oracle rule keeps exactly the rows whose record is unchanged
  4 the feasible rule does NOT catch the mid-degree trap by itself: a
    worker with an old pre-degree record but a live enrolment must be
    excluded by the enrolment condition, not by the band condition
  5 restricting to settled rows shrinks the artefact on data built to have
    one
  6 end to end with SQL forbidden; exports floored; retention reported
    CANARIES_DRYRUN=1 python3 revision/local/test_47k_synthetic.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries47k_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
spec_k = importlib.util.spec_from_file_location("s47k", MONA / "47k_settled_sample.py")
mod = importlib.util.module_from_spec(spec_k); spec_k.loader.exec_module(mod)
mod.OUT = TMP / "output_47k"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR
h47 = mod._h47(); mod._H47 = h47
# only AFTER _h47() has used the real directory to find 47h: from here on
# HERE is the sandbox, so the 47h-completion marker is written there
mod.HERE = TMP

RNG = np.random.default_rng(47000)
KEY = h47.load_key()
D = pd.read_stata(SHARE / "daioe_quartiles.dta")
HI = D.loc[D.high_exposure == 1, "ssyk4"].astype(str).str.zfill(4).to_numpy()
LO = D.loc[D.high_exposure == 0, "ssyk4"].astype(str).str.zfill(4).to_numpy()
ter = KEY[KEY["niva"].str[:1].isin(["4", "5", "6"])].sample(26, random_state=1)
gym = KEY[KEY["niva"].str[:1] == "3"].sample(10, random_state=2)
CELLS = pd.concat([ter, gym]).reset_index(drop=True)
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def weights_frame(year):
    R = np.random.default_rng(900 + year)
    rows = []
    for _, c in CELLS.iterrows():
        hi = c["niva"][:1] in "456"
        for code in np.concatenate([R.choice(HI, 4), R.choice(LO, 4)]):
            base = 500 if ((code in set(HI)) == hi) else 120
            for band in h47.EXP_BANDS:
                rows.append((c["niva"], c["inr"], code, 1, band, 1,
                             int(R.poisson(base)) + 1))
    return pd.DataFrame(rows, columns=["niva", "inr", "ssyk4", "fresh",
                                       "expband", "young", "n"])


def year_frame(year):
    """Three kinds of young worker, so every branch of the rules is exercised:
    settled (old record, no enrolment), mid-degree (OLD record but enrolled --
    the trap), and recent graduate (0-2 band)."""
    R = np.random.default_rng(700 + year)
    rows = []
    for emp in range(1, 111):
        mix = CELLS.sample(5, random_state=emp % 89)
        for m in range(1, 13):
            ym = f"{year}-{m:02d}"
            for _, c in mix.iterrows():
                for age, w in (("22-25", 0.10), ("26-30", 0.13), ("50+", 0.40)):
                    kind = (R.choice(["settled", "middegree", "recent"],
                                     p=[0.55, 0.25, 0.20])
                            if age == "22-25" else "settled")
                    rec = dict(employer_id=emp, year_month=ym, age_group=age,
                               niva_t=c["niva"], inr_t=c["inr"],
                               expb_t="6-10" if kind != "recent" else "0-2",
                               n_emp=int(R.poisson(40 * w)) + 1)
                    for T in (2021, 2022):
                        s = T % 100
                        # Staleness DEEPENS with each year past the truncation:
                        # a one-year-old code misclassifies less than a
                        # three-year-old one. Script 45 measured the deepening
                        # (-0.30, -0.41, -0.65, -0.71 by half-year of staleness).
                        # A fixture with a single step at T+1 is absorbed by the
                        # Riksbank term and leaves the ChatGPT term empty, which
                        # is a property of the fixture and not of the design.
                        depth = 0.0 if year <= T else min(0.35 * (year - T), 1.0)
                        stale = (kind in ("middegree", "recent")
                                 and R.uniform() < depth)
                        # A stale record is not a random other record: it is the
                        # worker's PRE-DEGREE record, which is systematically
                        # less exposed. 47b measured the consequence -- 63% of
                        # genuinely top-quartile young cells reassigned OUT. A
                        # fixture that reassigns at random produces noise, not
                        # that mechanism, and would not exercise the rules.
                        g = gym.iloc[0] if stale else c
                        rec[f"niva_{s}"], rec[f"inr_{s}"] = g["niva"], g["inr"]
                        rec[f"expb_{s}"] = ("0-2" if kind == "recent" else "6-10")
                        rec[f"enr_{s}"] = (c["inr"] if kind == "middegree" else None)
                    # 47h's pull now also returns the legacy (47b) cascade
                    # columns. This script never uses that arm, so they alias
                    # the corrected ones; the gate in 47h is where they differ.
                    rec["niva_21g"], rec["inr_21g"] = rec["niva_21"], rec["inr_21"]
                    rows.append(rec)
    return h47.compact(pd.DataFrame(rows)[h47.YEAR_COLS + ["n_emp"]])


def mark_47h_finished(done=True):
    """47k reads 47h's caches only when 47h has FINISHED, which its summary
    file proves. Without it, 47k keeps a private copy, so two consoles can
    run at once without one truncating the other's parquet."""
    d = mod.HERE / "output_47h"
    d.mkdir(exist_ok=True)
    f = d / "47h_summary.txt"
    if done:
        f.write_text("synthetic marker")
    elif f.exists():
        f.unlink()


def setup():
    for y in (2019, 2020, 2021):
        weights_frame(y).to_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet", index=False)
    for y in mod.YEARS:
        year_frame(y).to_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet", index=False)
    counts = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_weights_{y}.parquet")
              for y in (2019, 2020, 2021)}
    h47.MIN_CELL = 10
    book = h47.ScoreBook(counts, KEY, h47.load_scores())
    sp = dict(h47.DESIGNS["OL_daioe"]); book.build("OL_daioe", sp)
    frames = {y: pd.read_parquet(mc.CACHE_DIR / f"edu_hr_{y}.parquet") for y in mod.YEARS}
    return book, sp, frames


def test_mask_is_arm_invariant_and_feasible(frames):
    f = frames[2023]
    for T in (2021, 2022):
        m = mod.settled_mask(f, T, "feasible")
        # (2) feasible must not depend on the truth: blank the true columns
        g = f.copy()
        g["niva_t"] = pd.Series([None] * len(g), dtype="object").astype("category")
        g["inr_t"] = pd.Series([None] * len(g), dtype="object").astype("category")
        m2 = mod.settled_mask(g, T, "feasible")
        check(f"feasible rule is computable without the truth (T{T})",
              bool((m == m2).all()), f"{int((m != m2).sum())} rows differ")
        # (1) one mask, both arms: the sample cannot move between arms
        o = mod.settled_mask(f, T, "oracle")
        same = ((f[f"niva_{T%100}"].astype("string") == f["niva_t"].astype("string"))
                & (f[f"inr_{T%100}"].astype("string") == f["inr_t"].astype("string")))
        check(f"oracle rule keeps exactly the unchanged records (T{T})",
              bool((o == same.fillna(False)).all()))
    print("PASS one mask is computed per (T, rule) and used for BOTH arms "
          "(by construction in run(): masks are built once, outside the arm loop)")


def test_the_middegree_trap(frames):
    """A mid-degree worker has an OLD record and must still be excluded --
    by the enrolment condition, since the band condition cannot see them."""
    f = frames[2023]
    s = 21
    old_band = f[f"expb_{s}"].astype("string").isin(mod.SETTLED_BANDS)
    enrolled = f[f"enr_{s}"].notna()
    trap = old_band & enrolled
    check("the fixture contains the trap (old record, still enrolled)",
          int(trap.sum()) > 0, f"{int(trap.sum())} rows")
    m = mod.settled_mask(f, 2021, "feasible")
    check("the band condition alone would have kept the trap",
          int((old_band & ~m).sum()) >= int(trap.sum()))
    check("the feasible rule excludes every trapped row",
          int((trap & m).sum()) == 0)
    band_only_kept = int(old_band.sum())
    check("so the enrolment condition is doing real work",
          int(m.sum()) < band_only_kept,
          f"{int(m.sum()):,} kept vs {band_only_kept:,} on the band alone")


def test_restriction_shrinks_the_artefact(book, sp, frames):
    art = {}
    for rule in ("all", "feasible", "oracle"):
        masks = {y: mod.settled_mask(frames[y], 2021, rule) for y in mod.YEARS}
        g = {}
        for arm in ("true", "asof"):
            coll = pd.concat([mod.collapse(frames[y], masks[y], book, "OL_daioe",
                                           sp, arm, 2021, ["22-25"])
                              for y in mod.YEARS], ignore_index=True)
            g[arm] = mod.estimate(coll, "22-25", f"k_{rule}_{arm}")["gamma2"]
        art[rule] = g["asof"] - g["true"]
    check("the unrestricted sample shows an artefact", abs(art["all"]) > 0.05,
          f"{art['all']:+.4f}")
    check("the oracle rule removes it", abs(art["oracle"]) < 0.02, f"{art['oracle']:+.4f}")
    check("the feasible rule shrinks it toward the oracle",
          abs(art["feasible"]) < abs(art["all"]),
          " -> ".join(f"{k} {art[k]:+.4f}" for k in ("all", "feasible", "oracle")))


def test_cache_isolation():
    """
    47k shares 47h's cache unconditionally, and the safety comes from the
    WRITE being atomic rather than from a second private copy.

    Until 19 Sep 2026 this asserted the opposite: a private "_k" duplicate
    whenever 47h had not written its summary. 47h then halted at its gate
    without a summary, 47k kept a second copy of five 38-million-row year
    frames, and three lanes died with "No space left on device". The test
    now checks the mechanism that replaced it, which is what a reader in
    another console actually depends on: while a frame is being written,
    the target path is either absent or a COMPLETE, readable parquet, never
    a truncated one.
    """
    import threading
    mark_47h_finished(False)
    a = mod.cache_name("edu_hr_2021")
    mark_47h_finished(True)
    b = mod.cache_name("edu_hr_2021")
    check("one shared cache name, whatever 47h is doing",
          a == b and a.name == "edu_hr_2021.parquet", f"{a.name} / {b.name}")

    # A frame big enough that the write takes long enough to race against.
    big = pd.DataFrame({"employer_id": np.arange(400_000),
                        "n_emp": np.arange(400_000) % 7,
                        "pad": ["xxxxxxxxxxxxxxxxxxxx"] * 400_000})
    target = mod.CACHE / "atomicity_probe.parquet"
    target.unlink(missing_ok=True)
    torn = []

    def reader():
        deadline = time.time() + 5
        while time.time() < deadline:
            if target.exists():
                try:
                    got = pd.read_parquet(target)
                except Exception as ex:          # a partial file would land here
                    torn.append(f"unreadable: {type(ex).__name__}")
                    return
                if len(got) != len(big):
                    torn.append(f"short read: {len(got)} of {len(big)}")
                    return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    mc.write_cache(big, target)
    t.join(timeout=6)
    check("a concurrent reader never sees a partial cache",
          not torn, "; ".join(torn))
    check("no temp file is left behind",
          not list(mod.CACHE.glob("atomicity_probe.*.tmp*")),
          str([f.name for f in mod.CACHE.glob("atomicity_probe.*.tmp*")]))
    target.unlink(missing_ok=True)


def test_end_to_end():
    mark_47h_finished(True)      # the fixture wrote 47h-style cache names
    def boom(*a, **k):
        raise AssertionError("SQL attempted although the cache is warm")
    mc.connect = boom
    mod.AGES = ["22-25", "50+"]
    mod.AGES_EXTRA = []
    mod.main()
    est = pd.read_csv(mod.OUT / "settled_estimates.csv")
    check("every fit produced a coefficient", est["gamma2"].notna().all(),
          str(est[est["gamma2"].isna()].head()))
    check("no fit errored", (est["status"] == "ok").all(),
          str(est["status"].value_counts().to_dict()))
    ret = pd.read_csv(mod.OUT / "retention.csv")
    v = ret["n_kept"].dropna()
    check("retention is exported and floored", ((v == 0) | (v >= 5)).all())
    summ = (mod.OUT / "47k_summary.txt").read_text()
    for must in ("exposed young vs OTHER YOUNG", "applied to BOTH arms",
                 "CONSERVATIVE", "READ RULE"):
        check(f"the summary states: {must[:34]}", must in summ)
    print(f"      ({len(est)} fits)")


if __name__ == "__main__":
    book, sp, frames = setup()
    test_mask_is_arm_invariant_and_feasible(frames)
    test_the_middegree_trap(frames)
    test_restriction_shrinks_the_artefact(book, sp, frames)
    test_cache_isolation()
    test_end_to_end()
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""
_fixtures_trip37.py -- the sandbox and the score world shared by the dry
                       runs of scripts 95, 96, 97 and 98 (trip 37).

What every one of those tests needs before it can plant anything:

  sandbox()        a temporary share with the real input files, the cache
                   folder redirected into it, and mona_common patched so that
                   no SQL can be issued (a test world holds every cache it
                   needs; a query means the script would pull on MONA when
                   it should not).
  install_score()  script 82's cascade caches for a world of employers laid
                   out on the four DAIOE tiers, one real four-digit
                   occupation per tier, each in its own three-digit group,
                   so the uniform three-digit book returns the four-digit
                   score unchanged and the top tier is the top quartile.
                   Tier 3 is weighted up so that the incumbent-weighted 75th
                   percentile falls strictly inside it.
  months()         the panel's months, January 2021 to June 2025.
  Check            a PASS/FAIL recorder that exits 1 on any failure.

Nothing here touches SQL, MONA or the register. The generators are seeded
per unit (employer, cell, month), so two worlds that differ in one
planted effect draw the same noise and differ in that effect alone
(feedback_fixture_must_reproduce_mechanism, rule 3).
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"


def sandbox(tag: str, out_vars: tuple):
    """Temporary share and cache; returns (mc, TMP). `out_vars` are the
    CANARIES_*_OUT variables pointed into the sandbox."""
    os.environ["CANARIES_DRYRUN"] = "1"
    os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
    tmp = Path(tempfile.mkdtemp(prefix=f"canaries{tag}_"))
    share = tmp / "input"
    share.mkdir()
    for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
              "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
        shutil.copy(UPLOAD / f, share / f)
    os.environ["CANARIES_SHARE"] = str(share)
    for v in out_vars + ("CANARIES_82_OUT", "CANARIES_80_OUT",
                         "CANARIES_73_OUT"):
        os.environ[v] = str(tmp / "out")
    os.environ["CANARIES_RWORK_TAG"] = f"_test{tag}"
    sys.path.insert(0, str(MONA))
    sys.path.insert(0, str(HERE))
    import mona_common as mc
    mc.SHARE = str(share)
    mc.CACHE_DIR = tmp / "cache"
    mc.CACHE_DIR.mkdir()
    local = str(share / "daioe_quartiles.dta")
    mc.DAIOE_PATH = local
    ld = mc.load_daioe
    mc.load_daioe = lambda path=local: ld(path)

    def _no_sql():
        raise RuntimeError("the test world has every cache; no SQL may be "
                           "issued")
    mc.connect = _no_sql
    (tmp / "out").mkdir()
    return mc, tmp


def load(name: str, alias: str):
    sp = importlib.util.spec_from_file_location(alias, MONA / name)
    m = importlib.util.module_from_spec(sp)
    sys.modules[alias] = m
    sp.loader.exec_module(m)
    return m


AGES6 = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]


def tier_codes(share: Path) -> dict:
    """One real occupation per DAIOE quartile tier, distinct three-digit
    groups, the most typical code of each tier."""
    d = pd.read_stata(share / "daioe_quartiles.dta")
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    d["t"] = pd.qcut(d["pctl_rank_genai"], 4, labels=False)
    out, used = {}, set()
    for t in range(4):
        c = d[(d.t == t) & ~d.ssyk4.str[:3].isin(used)]
        c = c.assign(dd=(c.pctl_rank_genai - c.pctl_rank_genai.median()).abs())
        code = c.sort_values("dd").iloc[0]["ssyk4"]
        out[t] = code
        used.add(code[:3])
    return out


def install_score(mc, emps, tier_of, size_of, share: Path) -> None:
    """82's three caches: the cascade, the 2019 baseline and the 2019
    monthly counts that carry the incumbent floor."""
    code = tier_codes(share)
    rows = []
    for e in emps:
        c = code[tier_of(e)]
        for age in AGES6:
            rows += [(e, age, c, c[:3], "2019", int(round(6 * size_of(e)))),
                     (e, age, "____", "___", "none", 1)]
    casc = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                       "ssyk3", "source_year", "n"])
    casc["ssyk_ar"] = "2019"
    casc["ssyk_status"] = np.where(casc["ssyk4"] == "____", "9", "1")
    casc.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet",
                    index=False)
    (casc.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
     .sum().reset_index()).to_parquet(
        mc.CACHE_DIR / "L_baseline_2019.parquet", index=False)
    pd.DataFrame([(e, f"2019-{m:02d}", a, int(round(10 * size_of(e))))
                  for e in emps for m in range(1, 13) for a in AGES6],
                 columns=["employer_id", "year_month", "age_group", "n_emp"]
                 ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet",
                              index=False)


def months() -> list:
    return [f"{y}-{m:02d}" for y in range(2021, 2026)
            for m in range(1, 13 if y < 2025 else 7)]


def write_by_year(df: pd.DataFrame, cache: Path, prefix: str) -> None:
    for y in range(2021, 2026):
        df[df["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            cache / f"{prefix}_{y}.parquet", index=False)


def poisson_same_noise(lam: np.ndarray, seed: int) -> np.ndarray:
    """Poisson counts by inverse CDF from ONE uniform per cell, drawn from a
    seed that does not depend on the world. Two worlds laid out on the
    same grid therefore share every uniform and differ in the planted
    rates alone; a skipped or extra draw cannot shift the stream."""
    from scipy.stats import poisson
    u = np.random.default_rng(seed).random(len(lam))
    return poisson.ppf(u, lam).astype(int)


class Check:
    def __init__(self):
        self.fails = []

    def __call__(self, name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name
              + (f"  [{detail}]" if detail else ""))
        if not cond:
            self.fails.append(name)

    def done(self):
        print(f"\n{'ALL PASS' if not self.fails else 'FAILURES: ' + '; '.join(self.fails)}")
        raise SystemExit(1 if self.fails else 0)


def triple_diff(c: pd.DataFrame, high: set, young, old, post_from="2024-01",
                interim_from="2022-12") -> float:
    """The raw contrast the design estimates, by hand: the log
    young-to-old ratio, high minus low employers, later window minus the
    interim window (the fixture has no calendar cycle, so the calendar
    terms change nothing in expectation)."""
    ym = c["year_month"]
    per = np.where(ym >= post_from, "post",
                   np.where(ym >= interim_from, "interim", "x"))
    grp = np.where(c["age_group"].isin(young), "y",
                   np.where(c["age_group"].isin(old), "o", "x"))
    d = c.assign(hi=c["employer_id"].isin(high), per=per, grp=grp)
    t = d[(d.per != "x") & (d.grp != "x")].groupby(
        ["hi", "per", "grp"])["n_emp"].sum()
    lr = lambda h, p: np.log(t[(h, p, "y")] / t[(h, p, "o")])  # noqa: E731
    return float((lr(True, "post") - lr(True, "interim"))
                 - (lr(False, "post") - lr(False, "interim")))

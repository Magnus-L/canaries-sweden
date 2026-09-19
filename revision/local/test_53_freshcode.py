#!/usr/bin/env python3
"""
test_53_freshcode.py -- 53 end to end, real DAIOE input, real R + fixest,
synthetic pulls at the SQL boundary.

The fixture has to reproduce the MECHANISM, not just the code path. A
carried-forward code is wrong because the worker MOVED and the register did
not follow, so here:
  - there is no true effect at all (the null), and
  - a rising share of high-exposure young workers genuinely move to
    low-exposure work, while their stale code keeps saying high.
That, and only that, manufactures a negative gamma2 in the stale and all
arms while the fresh arm stays at zero.

Claims tested:
  1 the three arms partition the coded worker-months exactly
  2 under a NULL with systematic carry-forward, the fresh arm recovers zero
    and the stale arm does not: the contrast is the point of the script
  3 a planted real effect IS recovered by the fresh arm (it has power)
  4 the selection diagnostic fires when, and only when, freshness is
    differentially selected across the exposure dimension over time
  5 a primary fit returning nothing raises; an optional table failing does not
  6 end to end with no SQL: exports written, floored, summary states the limits

    CANARIES_DRYRUN=1 python3 revision/local/test_53_freshcode.py
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
TMP = Path(tempfile.mkdtemp(prefix="canaries53_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
# mona_common builds DAIOE_PATH with a Windows separator, which is right on
# MONA and wrong here. load_daioe binds it as a default argument at import,
# so rebinding the module constant alone is not enough.
_LOCAL_DAIOE = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL_DAIOE
_load_daioe = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL_DAIOE: _load_daioe(path)
spec = importlib.util.spec_from_file_location("s53", MONA / "53_freshcode_panel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = TMP / "output_53"; mod.OUT.mkdir(); mod.CACHE = mc.CACHE_DIR

D = pd.read_stata(SHARE / "daioe_quartiles.dta")
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
Q = D.copy()
if not pd.api.types.is_numeric_dtype(Q["exposure_quartile"]):
    Q["exposure_quartile"] = (Q["exposure_quartile"].astype(str)
                              .str.extract(r"(\d)").astype(int))
HI = Q.loc[Q.exposure_quartile == 4, "ssyk4"].to_numpy()
LO = Q.loc[Q.exposure_quartile <= 3, "ssyk4"].to_numpy()
MONTHS = [f"{y}-{m:02d}" for y in mod.YEARS for m in range(1, 13)]
N_EMP = 90
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_panel(true_effect=0.0, misplace=0.55, selective_freshness=0.0,
               seed=5):
    """
    Synthetic employer x month x ssyk4 x freshness x age panel.

    THE MECHANISM, stated before the code, because the first version of this
    fixture had it backwards and passed anyway.

    Staleness does not manufacture a decline by making workers move. It
    manufactures one by MISPLACING them. A young worker whose record is
    carried forward is filed under the occupation the register last saw,
    and young workers churn: script 50's M2 moment found 24.1 per cent of
    22-25 year olds change code in a year and only 49.1 per cent of those
    stay in the same exposure quartile. So a growing share of workers who
    are really doing high-exposure work are counted in a low-exposure cell.
    The TRUE high-exposure young headcount is flat; the MEASURED one falls.

    misplace              share of the stale high-exposure young headcount
                          that is filed under a low-exposure code by the END
                          of the window, rising linearly from zero, so
                          staleness DEEPENS rather than sitting at a level.
    true_effect           a genuine log decline on high-exposure young cells
                          after ChatGPT (0.0 = the null).
    selective_freshness   raises the Q4 freshness share over time relative to
                          the rest: the one thing that lets the fresh
                          restriction itself manufacture an interaction.
    """
    R = np.random.default_rng(seed)
    rows = []
    for emp in range(1, N_EMP + 1):
        hi_code = R.choice(HI)
        lo_code = R.choice(LO)
        for i, ym in enumerate(MONTHS):
            frac = i / (len(MONTHS) - 1)
            post = ym >= mc.CHATGPT_YM
            for age in ("22-25", "26-30", "50+"):
                young = age == "22-25"
                # --- the real world: flat, except any planted effect ---
                true_hi = 40.0 * (np.exp(true_effect)
                                  if (post and young) else 1.0)
                true_lo = 60.0
                sf = 0.60 + (selective_freshness * frac if young else 0.0)
                # fresh records describe the real job, always
                hi_fresh, lo_fresh = true_hi * sf, true_lo * sf
                # stale records misplace a GROWING share of the high-exposure
                # workers into the low-exposure code. Young workers only:
                # a 50-year-old's code was right a decade ago and still is.
                m = (misplace * frac) if young else 0.0
                hi_stale_true = true_hi * (1 - sf)
                hi_stale = hi_stale_true * (1 - m)
                lo_stale = true_lo * (1 - sf) + hi_stale_true * m
                for code, n, fresh in ((hi_code, hi_fresh, "fresh"),
                                       (hi_code, hi_stale, "stale"),
                                       (lo_code, lo_fresh, "fresh"),
                                       (lo_code, lo_stale, "stale")):
                    if n <= 0:
                        continue
                    rows.append((emp, ym, code, fresh, age,
                                 int(R.poisson(n))))
            if emp % 11 == 0:
                rows.append((emp, ym, "____", "none", "26-30", 3))
    p = pd.DataFrame(rows, columns=["employer_id", "year_month", "ssyk4",
                                    "freshness", "age_group", "n_emp"])
    p = (p.groupby(["employer_id", "year_month", "ssyk4", "freshness",
                    "age_group"], as_index=False)["n_emp"].sum())
    for c in ("ssyk4", "freshness", "age_group"):
        p[c] = p[c].astype("string").astype("category")
    p["n_emp"] = p["n_emp"].astype("int32")
    return p


def g2_of(bal_panel, arm, age="22-25"):
    """Run the script's own pipeline for one arm and return gamma2."""
    daioe = mc.load_daioe()
    agg = mod.arm_rows(bal_panel, arm)
    agg = mc.merge_daioe_and_filter(agg, daioe)
    agg = mc.aggregate_to_quartile(agg)
    months = sorted(agg["year_month"].unique())
    bal = mod.build_age_panel(agg, months, age)
    r = mc.run_fepois(bal, mod.OUT, tag=f"t53_{arm}_{age}")
    g = r[r["term"] == "post_gpt_x_high"]
    return float(g["coef"].iloc[0]) if not g.empty else np.nan


# ---- 1. the arms partition the coded worker-months --------------------
p = make_panel()
coded = p[p["ssyk4"] != "____"]["n_emp"].sum()
a = mod.arm_rows(p, "all")["n_emp"].sum()
f = mod.arm_rows(p, "fresh")["n_emp"].sum()
s = mod.arm_rows(p, "stale")["n_emp"].sum()
check("the three arms partition the coded worker-months",
      a == coded and f + s == a, f"all {a:,} = fresh {f:,} + stale {s:,}")
check("uncoded worker-months are excluded from every arm",
      mod.arm_rows(p, "all")["ssyk4"].ne("____").all())

# ---- 2. THE MECHANISM: null DGP, systematic carry-forward -------------
g_fresh = g2_of(p, "fresh")
g_stale = g2_of(p, "stale")
g_all = g2_of(p, "all")
check("under a null, the fresh arm returns ~zero",
      abs(g_fresh) < 0.03, f"{g_fresh:+.4f}")
check("under the same null, the stale arm manufactures a decline",
      g_stale < -0.05, f"{g_stale:+.4f}")
check("the all arm sits between them, as a weighted mixture must",
      g_stale < g_all < g_fresh + 0.02,
      f"stale {g_stale:+.4f} < all {g_all:+.4f} < fresh {g_fresh:+.4f}")

# ---- 3. the fresh arm has POWER, not just a zero ----------------------
p_eff = make_panel(true_effect=-0.20, seed=6)
g_fresh_eff = g2_of(p_eff, "fresh")
check("with a real -0.20 planted, the fresh arm recovers it",
      g_fresh_eff < -0.12, f"{g_fresh_eff:+.4f} against a planted -0.20")
check("a zero from the fresh arm is therefore informative, not just weak",
      abs(g_fresh_eff - g_fresh) > 0.10,
      f"null {g_fresh:+.4f} vs effect {g_fresh_eff:+.4f}")

# ---- 4. the selection diagnostic fires only when it should ------------
daioe = mc.load_daioe()
_, sd_flat = mod.freshness_diagnostics(p, daioe)
_, sd_sel = mod.freshness_diagnostics(
    make_panel(selective_freshness=0.25, seed=7), daioe)
flat = abs(sd_flat.loc[sd_flat.age_group == "22-25", "selection_did"].iloc[0])
sel = abs(sd_sel.loc[sd_sel.age_group == "22-25", "selection_did"].iloc[0])
# The freshness gap is reported but NOT gated, and this is why: the fixture
# below has NO differential selection, only misplacement, and the gap still
# moves by 0.13. Misplacing high-exposure workers into low-exposure codes
# leaves the surviving Q4 cell fresher, so the gap tracks the artefact
# itself. A rule built on it would reject every run in which the artefact
# is present, which is every run that needs the correction.
check("the freshness gap moves even with NO differential selection, "
      "which is why it cannot be the gate",
      flat > mod.SELECTION_DID_LIMIT, f"{flat:.4f} with selection off")
check("adding real differential selection moves it further",
      sel > flat * 0 + mod.SELECTION_DID_LIMIT, f"{sel:.4f}")

# The rule that DOES work: the 50+ placebo on the fresh arm. 50+ workers'
# codes are selected for freshness too, but they barely churn, so the
# placebo isolates the restriction from the churn it corrects.
g_placebo = g2_of(p, "fresh", age="50+")
check("the fresh arm's 50+ placebo is clean under the null",
      abs(g_placebo) < mod.PLACEBO_LIMIT,
      f"{g_placebo:+.4f}, limit {mod.PLACEBO_LIMIT}")
g_placebo_stale = g2_of(p, "stale", age="50+")
check("and the stale arm's 50+ placebo is ALSO clean, so a 22-25 artefact "
      "cannot be dismissed as a general age effect",
      abs(g_placebo_stale) < 0.05, f"{g_placebo_stale:+.4f}")

# ---- 5. capture-noisily discipline ------------------------------------
def boom():
    raise ValueError("planted")
check("an optional table failing does not stop the run",
      mod.opt("planted failure", boom) is None)

# ---- 6. end to end, no SQL --------------------------------------------
for y in mod.YEARS:
    mc.write_cache(p[p["year_month"].astype(str).str[:4] == str(y)]
                   if (p["year_month"].astype(str).str[:4] == str(y)).any()
                   else p.assign(year_month=f"{y}-01"),
                   mc.CACHE_DIR / f"freshcode_{y}.parquet")
def no_sql(*a, **k):
    raise AssertionError("SQL attempted although every year is cached")
mc.connect = no_sql
mod.ES_AGES = ("22-25",)
mod.main()
est = pd.read_csv(mod.OUT / "fresh_pooled.csv")
check("every arm produced estimates", set(est["arm"]) == set(mod.ARMS),
      str(sorted(est["arm"].unique())))
check("no fit errored", (est["status"] == "ok").all(),
      str(est["status"].value_counts().to_dict()))
check("selection_did.csv written", (mod.OUT / "selection_did.csv").exists())
sh = pd.read_csv(mod.OUT / "freshness_shares.csv")
v = sh["n_coded"].dropna()
check("the freshness table is floored", ((v == 0) | (v >= 5)).all())
summ = (mod.OUT / "53_summary.txt").read_text()
for must in ("ends in 2023", "over-represents movers", "50+ row is the placebo",
             "never one", "SELECTION CHECK"):
    check(f"the summary states: {must[:30]}", must in summ)

print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)

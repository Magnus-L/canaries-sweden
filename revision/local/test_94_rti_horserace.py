#!/usr/bin/env python3
"""
test_94_rti_horserace.py -- the horse race must attribute a planted
                            decline to the score that carries it, the
                            gate must be able to fail, and main() must
                            run end to end and write its one export.

TWO synthetic worlds, identical except for which score the decline
follows, because the whole question is which one the Swedish data are:

  AI world    employment at 22-25 and 26-30 in employers in the top
              quartile of DAIOE falls by log(0.8) from January 2024;
              young women there fall by a further log(0.9).
  RTI world   the same decline, in the top quartile of RTI instead.

Employers are laid out on a four-by-four grid of DAIOE tier by RTI tier,
each holding one four-digit occupation drawn from the real score files
so that its DAIOE and RTI quartiles are the grid cell; the two firm
scores are therefore independent by construction, as the occupation-level
correlation of +0.05 says they nearly are. A binary decline is planted
because (a) to (c) fit binary indicators; (d) fits standardised
continuous scores and is checked for sign and for the null score only.

The seed is shared, so the two worlds draw the same noise and differ in
the planted cell alone. Before any assertion on the estimator, the
planted step is recomputed by hand as a raw triple difference, so that a
failure can be placed in the fixture or in the code.

    python3 revision/local/test_94_rti_horserace.py
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
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries94_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta", "rti_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_94_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)


def _no_sql():
    raise RuntimeError("the test world has every cache; no SQL may be issued")


mc.connect = _no_sql


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s94 = load("94_rti_horserace.py", "s94")
s94.OUT = TMP / "out"; s94.OUT.mkdir(parents=True, exist_ok=True)
s94.CACHE = mc.CACHE_DIR
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================
# the grid: one real occupation per (DAIOE tier, RTI tier) cell, each in
# its own three-digit group, so the uniform three-digit book returns the
# four-digit score unchanged
# ======================================================================
D = pd.read_stata(SHARE / "daioe_quartiles.dta")
R = pd.read_stata(SHARE / "rti_ssyk4.dta")
G = D.merge(R, on="ssyk4")
G["ssyk4"] = G["ssyk4"].astype(str).str.zfill(4)
G["ta"] = pd.qcut(G["pctl_rank_genai"], 4, labels=False)
G["tr"] = pd.qcut(G["rti"], 4, labels=False)
CODE, used3 = {}, set()
for a in range(4):
    for r in range(4):
        cand = G[(G.ta == a) & (G.tr == r) & ~G.ssyk4.str[:3].isin(used3)]
        # the most typical code of the cell, so tiers do not overlap
        cand = cand.assign(d=(cand.pctl_rank_genai
                              - cand.pctl_rank_genai.median()).abs())
        c = cand.sort_values("d").iloc[0]["ssyk4"]
        CODE[(a, r)] = c; used3.add(c[:3])
# every employer's tiers: DAIOE from e % 4, RTI from (e // 4) % 4, size
# from e // 16, so the tiers are independent and each carries the same
# size distribution. Tier 3 is weighted up so the weighted 75th
# percentile falls strictly inside it on both scores.
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_EMP = 16 * 20
EMPS = list(range(1, N_EMP + 1))
ta = lambda e: e % 4                                          # noqa: E731
tr = lambda e: (e // 4) % 4                                   # noqa: E731
size = lambda e: (1 + (e // 16) % 5) * (1.4 if ta(e) == 3 else 1.0) \
    * (1.4 if tr(e) == 3 else 1.0)                            # noqa: E731
UNSCORED = "9999"


def cascade_frame() -> pd.DataFrame:
    rows = []
    for e in EMPS:
        c = CODE[(ta(e), tr(e))]
        for age in AGES:
            rows += [(e, age, c, c[:3], "2019", int(round(6 * size(e)))),
                     (e, age, UNSCORED, UNSCORED[:3], "2019", 1),
                     (e, age, "____", "___", "none", 1)]
    d = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                    "ssyk3", "source_year", "n"])
    d["ssyk_ar"] = "2019"
    d["ssyk_status"] = np.where(d["ssyk4"] == "____", "9", "1")
    return d


CASC = cascade_frame()
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"]
 .sum().reset_index()).to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet",
                                  index=False)
pd.DataFrame([(e, f"2019-{m:02d}", a, int(round(10 * size(e))))
              for e in EMPS for m in range(1, 13) for a in AGES],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

s82, s61, s67, s78, l47, l70, j47 = s94.load_modules()
for m_ in (s82, s61, s78, l47, l70, j47):
    m_.OUT, m_.CACHE = s94.OUT, mc.CACHE_DIR
MONTHS = [f"{y}-{m:02d}" for y in s61.PANEL_YEARS
          for m in range(1, 13 if y < 2025 else 7)]
FALL = float(np.log(0.80))
FALL_F = float(np.log(0.90))
RISE_RB = float(np.log(1.06))


def sex_counts(world: str) -> pd.DataFrame:
    rng = np.random.default_rng(94)       # the same draws in both worlds
    lam0 = {"22-25": 8, "26-30": 8, "31-34": 6, "35-40": 6, "41-49": 7,
            "50+": 7}
    rows = []
    for e in EMPS:
        hit = (ta(e) == 3) if world == "ai" else (tr(e) == 3)
        for ym in MONTHS:
            for age, lam in lam0.items():
                for g in ("1", "2"):
                    x = float(lam)
                    if hit and age in ("22-25", "26-30"):
                        if ym >= mc.RIKSBANK_YM:
                            x *= np.exp(RISE_RB)
                        if ym >= "2024-01":
                            x *= np.exp(FALL)
                            if g == "2" and age == "22-25":
                                x *= np.exp(FALL_F)
                    rows.append((e, ym, age, g, int(rng.poisson(x)) + 1))
    return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                       "age_group", "gender", "n_emp"])


def install(world: str) -> pd.DataFrame:
    for p in mc.CACHE_DIR.glob("L_counts_20[2]*.parquet"):
        p.unlink()
    for p in mc.CACHE_DIR.glob("L_counts_sex_*.parquet"):
        p.unlink()
    sx = sex_counts(world)
    c = (sx.groupby(["employer_id", "year_month", "age_group"],
                    as_index=False)["n_emp"].sum())
    for y in s61.PANEL_YEARS:
        c[c["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_{y}.parquet", index=False)
        sx[sx["year_month"].str.slice(0, 4) == str(y)].to_parquet(
            mc.CACHE_DIR / f"L_counts_sex_{y}.parquet", index=False)
    return c


def by_hand(c: pd.DataFrame, which: str) -> float:
    """The raw triple difference the design estimates, on the planted
    score: log young-to-old ratio, high minus low, 2024 on minus the
    interim window."""
    c = c.copy()
    hi = c["employer_id"].map(lambda e: (ta(e) if which == "ai" else tr(e)) == 3)
    young = c["age_group"] == "22-25"
    old = c["age_group"].isin(j47.INCUMBENT_BANDS)
    per = np.where(c["year_month"] >= "2024-01", "post",
                   np.where(c["year_month"] >= mc.CHATGPT_YM, "interim", "x"))
    c = c.assign(hi=hi, per=per, grp=np.where(young, "y",
                                              np.where(old, "o", "x")))
    t = c[(c.per != "x") & (c.grp != "x")].groupby(
        ["hi", "per", "grp"])["n_emp"].sum()
    lr = lambda h, p: np.log(t[(h, p, "y")] / t[(h, p, "o")])  # noqa: E731
    return float((lr(True, "post") - lr(True, "interim"))
                 - (lr(False, "post") - lr(False, "interim")))


def run_world(world: str) -> dict:
    """Parts A to C through the script's own functions, the gate bypassed
    because a synthetic world cannot reproduce Table 1."""
    c = install(world)
    hand_ai, hand_rti = by_hand(c, "ai"), by_hand(c, "rti")
    s94.ROWS.clear(); s94.NOTES.clear(); s94.FAILURES.clear()
    keep = s94.check_gate
    s94.check_gate = lambda band: None
    try:
        ai, rt = s94.build_scores(s82, l47, l70, j47)
        m = s94.part_a(ai["exposure"], rt["exposure"])
        counts = s82.load_counts("L_counts", s61.PANEL_YEARS,
                                 require=s94.COUNT_COLS)
        s94.part_b(counts, ai, rt, s61, s78, j47)
        sx = s82.load_counts("L_counts_sex", s61.PANEL_YEARS,
                             require=s94.SEX_COLS)
        # the sex gate cannot pass either; point it at this world's own
        # DAIOE-only fit by running that fit first through part_c's path
        s94.SEX_GATE_KEEP = dict(s94.SEX_GATE)
        s94.SEX_GATE.update(_sex_gate_for(sx, ai))
        s94.part_c(sx, ai, rt, s67, s78, j47)
        s94.SEX_GATE.update(s94.SEX_GATE_KEEP)
    finally:
        s94.check_gate = keep
    return {"hand_ai": hand_ai, "hand_rti": hand_rti, "m": m,
            "rows": list(s94.ROWS), "fail": list(s94.FAILURES),
            "ai": ai, "rt": rt}


def _sex_gate_for(sx, ai) -> dict:
    """This world's own DAIOE-only sex estimates, so the gate passes here
    exactly when the script's fit reproduces itself."""
    skel = s67.build_skeleton_sex(sx, s94.SEX_BAND, j47, "n_emp")
    b = s78.with_exposure(skel, ai["exposure"])
    b, t = s78.gender_eq2_terms(b)
    g, v = s94.fit(b, "probe_sex", t, j47.FES)
    p = "post_x_high_x_young_x_female"
    return {"post": (float(g.loc[p, "coef"]), float(g.loc[p, "se"])),
            "step": s94.step(g, v, "", "_x_female")}


def val(rows, spec, band, ind, term, block="fit"):
    for r in rows:
        if (r["block"], r["spec"], r["young_band"], r["indicator"],
                r["term"]) == (block, spec, band, ind, term):
            return r["coef"], r["se"]
    return np.nan, np.nan


# ======================================================================
print("\n--- the RTI file: pinned hash, schema, code set ---")
book = s94.rti_book()
check("the RTI file loads through the probe", len(book) >= s94.RTI_MIN_CODES
      and list(book.columns) == ["ssyk4", "score"], f"{len(book)} codes")
_bad = SHARE / "rti_ssyk4.dta"
_orig = _bad.read_bytes()
d_ = pd.read_stata(_bad); d_.loc[0, "rti"] += 0.001
d_.to_stata(_bad, write_index=False, version=118)
try:
    s94.rti_book(); _stopped = False
except RuntimeError:
    _stopped = True
_bad.write_bytes(_orig)
check("a file that is not the pinned one is refused", _stopped)

print("\n--- the step SE comes from the covariance ---")
_g = pd.DataFrame({"coef": [-0.06, -0.02]},
                  index=["post_x_high_x_young", "interim_x_high_x_young"])
_v = pd.DataFrame([[4e-4, 3e-4], [3e-4, 9e-4]], index=_g.index,
                  columns=_g.index)
_c, _s = s94.step(_g, _v)
check("step = post - interim, SE with -2Cov",
      abs(_c + 0.04) < 1e-12 and abs(_s - np.sqrt(7e-4)) < 1e-12,
      f"{_c:+.4f} ({_s:.5f})")

print("\n--- the gate can fail as well as pass ---")
s94.ROWS.clear()
for band, w in s94.GATE.items():
    s94.add("gate", "a", band, "ai", "post", *w["post"], 1, 100)
    s94.add("gate", "a", band, "ai", "step_from_2023", *w["step"], 1, 100)
try:
    for band in s94.BANDS:
        s94.check_gate(band)
    _passed = True
except SystemExit:
    _passed = False
check("the gate passes on Table 1's numbers", _passed)
s94.ROWS[1]["coef"] += 0.0006
try:
    s94.check_gate("22-25"); _stopped = False
except SystemExit:
    _stopped = True
check("the gate STOPS when the step is 0.0006 away", _stopped)
s94.ROWS[1]["coef"] -= 0.0002
try:
    s94.check_gate("22-25"); _ok = True
except SystemExit:
    _ok = False
check("and passes at 0.0004 away (the stated tolerance is 0.0005)", _ok)

for world in ("ai", "rti"):
    print(f"\n--- the {world.upper()} world ---")
    w = run_world(world)
    planted = FALL
    hand = w["hand_ai"] if world == "ai" else w["hand_rti"]
    other = w["hand_rti"] if world == "ai" else w["hand_ai"]
    check("by hand, the planted score carries the planted step",
          abs(hand - planted) < 0.05, f"{hand:+.4f} against {planted:+.4f}")
    check("by hand, the other score carries almost none of it",
          abs(other) < 0.06, f"{other:+.4f}")
    m = w["m"]
    r_emp = float(m["mix_ai"].corr(m["mix_rti"], method="spearman"))
    check("the two firm scores are near-independent, as designed",
          abs(r_emp) < 0.2, f"Spearman {r_emp:+.3f}")
    check("Part A exported the off-diagonal share",
          any(r["term"] == "share_off_diagonal_median" for r in w["rows"]))
    check("no fit failed", not w["fail"], "; ".join(w["fail"]))
    for band in s94.BANDS:
        ca, sa = val(w["rows"], "c", band, "ai", "step_from_2023")
        cr, sr = val(w["rows"], "c", band, "rti", "step_from_2023")
        win, lose = ((ca, sa), (cr, sr)) if world == "ai" else ((cr, sr), (ca, sa))
        check(f"{band} (c): the planted score recovers the step",
              abs(win[0] - planted) < 0.06 and win[0] < -1.96 * win[1],
              f"{win[0]:+.4f} ({win[1]:.4f})")
        check(f"{band} (c): the other score is not significantly negative",
              not (lose[0] < -1.96 * lose[1]), f"{lose[0]:+.4f} ({lose[1]:.4f})")
        ga, _ = val(w["rows"], "a2", band, "ai", "step_from_2023")
        gb, _ = val(w["rows"], "b", band, "rti", "step_from_2023")
        alone = ga if world == "ai" else gb
        check(f"{band} alone: the planted score's own fit finds it",
              abs(alone - planted) < 0.06, f"{alone:+.4f}")
        za, sza = val(w["rows"], "d", band, "z_ai", "step_from_2023")
        zr, szr = val(w["rows"], "d", band, "z_rti", "step_from_2023")
        zw, zl = ((za, sza), (zr, szr)) if world == "ai" else ((zr, szr), (za, sza))
        check(f"{band} (d): the planted score is negative, the other is not "
              f"significantly so", zw[0] < -1.96 * zw[1]
              and not (zl[0] < -1.96 * zl[1]),
              f"planted {zw[0]:+.4f} ({zw[1]:.4f}), other {zl[0]:+.4f} "
              f"({zl[1]:.4f})")
    fa, sfa = val(w["rows"], "sex_c", "22-25", "ai", "female_diff_step_from_2023")
    fr, sfr = val(w["rows"], "sex_c", "22-25", "rti", "female_diff_step_from_2023")
    fw, fl = ((fa, sfa), (fr, sfr)) if world == "ai" else ((fr, sfr), (fa, sfa))
    check("sex (c): the planted score carries the female differential",
          abs(fw[0] - FALL_F) < 0.06 and fw[0] < -1.96 * fw[1],
          f"{fw[0]:+.4f} ({fw[1]:.4f}) against {FALL_F:+.4f}")
    check("sex (c): the other score does not", not (fl[0] < -1.96 * fl[1]),
          f"{fl[0]:+.4f} ({fl[1]:.4f})")
    s94.ROWS[:] = w["rows"]
    v, _ = s94.verdict()
    want = "1 AI-SPECIFIC" if world == "ai" else "2 RTI ABSORBS"
    check("the read rule gives the planted verdict", v == want, v)

print("\n--- (a2): refitted when RTI scores fewer employers than (a) ---")
w = run_world("ai")
_rt = dict(w["rt"])
_drop = set(_rt["exposure"]["employer_id"].iloc[:20])
_rt["exposure"] = _rt["exposure"][~_rt["exposure"]["employer_id"].isin(_drop)]
s94.ROWS.clear(); s94.FAILURES.clear()
_keep = s94.check_gate
s94.check_gate = lambda band: None
try:
    s94.BANDS = ["22-25"]
    s94.part_b(s82.load_counts("L_counts", s61.PANEL_YEARS), w["ai"], _rt,
               s61, s78, j47)
finally:
    s94.check_gate = _keep
    s94.BANDS = ["22-25", "26-30"]
_a2 = [r for r in s94.ROWS if r["spec"] == "a2"]
_a = [r for r in s94.ROWS if r["spec"] == "a"]
check("(a2) is a fit of its own on the smaller sample",
      bool(_a2) and _a2[0]["status"] != "derived" and _a2[0]["block"] == "fit"
      and _a2[0]["n_firms"] < _a[0]["n_firms"],
      f"(a) {_a[0]['n_firms'] if _a else '?'} employers, (a2) "
      f"{_a2[0]['n_firms'] if _a2 else '?'}")

print("\n--- main(), end to end, in the AI world, gate pointed at it ---")
for band in s94.BANDS:
    s94.GATE[band] = {
        "post": val(w["rows"], "a", band, "ai", "post", block="gate"),
        "step": val(w["rows"], "a", band, "ai", "step_from_2023", block="gate")}
s94.SEX_GATE.update(_sex_gate_for(
    s82.load_counts("L_counts_sex", s61.PANEL_YEARS), w["ai"]))
s94.ROWS.clear(); s94.NOTES.clear(); s94.FAILURES.clear()
s94.DONE = 0
_stdout = sys.stdout
rc = s94.main()
sys.stdout = _stdout
out = pd.read_csv(s94.OUT / "rti_horserace.csv")
summ = (s94.OUT / "94_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s94.FAILURES}")
check("the one CSV carries every spec",
      set(out["spec"]) >= {"A", "a", "a2", "b", "c", "d", "sex_a", "sex_c"},
      str(sorted(set(out["spec"]))))
check("every fit came back", s94.DONE == s94.PLANNED,
      f"{s94.DONE} of {s94.PLANNED}")
check("the summary states the verdict", "1 AI-SPECIFIC" in summ)
check("no employer count under the floor survives",
      bool(((out["n_firms"].isna()) | (out["n_firms"] >= 5)).all()))
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ssyk4", "fe_emp_t")))
check("the log exists", (s94.OUT / "94_log.txt").exists())

print("\n--- main() stops on a failed gate before any other fit ---")
s94.GATE["22-25"]["step"] = (s94.GATE["22-25"]["step"][0] + 0.01,
                             s94.GATE["22-25"]["step"][1])
s94.ROWS.clear(); s94.NOTES.clear(); s94.FAILURES.clear()
s94.DONE = 0
try:
    s94.main(); _stopped = False
except SystemExit:
    _stopped = True
sys.stdout = _stdout
check("a gate failure stops the script", _stopped)
check("and only the gate fit ran", s94.DONE == 1, f"{s94.DONE} fits")
check("the summary records the failed gate",
      "THE GATE FAILED" in (s94.OUT / "94_summary.txt").read_text())

print("\n" + ("all checks passed" if not FAILS else f"FAILED: {FAILS}"))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

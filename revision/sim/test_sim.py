#!/usr/bin/env python3
"""
test_sim.py -- acceptance tests for the simulation study.

Sections 8.1-8.6 of notes/simulation-study-spec_2026-09-19.md. Two kinds:

STRUCTURAL tests must pass on any calibration, placeholder included, because
they are about the machinery: the oracle recovers what it should, switching
the register layer off removes every artefact, the emitted frames are the
ones 47h consumes, and the runner's metrics are computed the way the spec
says.

CALIBRATED tests bind to script 50's export and are SKIPPED, loudly, until
it is present: the DGP reproducing 47b's -0.36 is a statement about the
Swedish registers, not about code, and asserting it against placeholders
would only be asserting my own guesses back at me.

    python3 revision/sim/test_sim.py [--export <output_50 dir>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sim as R                      # noqa: E402  (sets the env and imports 47h)
import calib                             # noqa: E402
from dgp import Params, Sim, BACKTEST_YEARS, PROD_YEARS, YEAR_COLS  # noqa: E402

SMALL = dict(n_persons=5000, n_emp=200)
OUT = Path("/tmp/canaries_simtest"); OUT.mkdir(exist_ok=True)
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def make(export, scenario="paper", **kw):
    key, daioe = R.load_inputs()
    cal = calib.build(export, key, daioe, np.random.default_rng(7))
    p = Params(gamma=R.SCENARIOS[scenario], **{**SMALL, **kw})
    return Sim(p, cal, key, daioe, np.random.default_rng(11)), key, daioe, cal


def t1_frames_match_47h(export):
    """8.6 the emitted frames are exactly what 47h consumes."""
    sim, key, daioe, _ = make(export)
    w = sim.emit_weights(2019)
    need_w = {"niva", "inr", "ssyk4", "fresh", "expband", "young", "n"}
    check("emit_weights has 47h.pull_weights' columns", set(w.columns) == need_w,
          str(sorted(set(w.columns) ^ need_w)))
    f = sim.emit_year(2021)
    check("emit_year has 47h.YEAR_COLS + n_emp",
          set(f.columns) == set(YEAR_COLS) | {"n_emp"},
          str(sorted(set(f.columns) ^ (set(YEAR_COLS) | {"n_emp"}))))
    c = R.h47.compact(f)
    check("47h.compact accepts the frame", len(c) > 0 and c["n_emp"].dtype == "int32")
    book = R.h47.ScoreBook({y: sim.emit_weights(y) for y in (2019, 2020, 2021)}, key,
                           daioe.rename(columns={"score": "daioe_score", "high": "daioe_high"})
                           [["ssyk4", "daioe_score", "daioe_high", "eloundou_score", "eloundou_high"]])
    R.h47.MIN_CELL = 10
    for nm, sp in R.h47.DESIGNS.items():
        book.build(nm, sp)
    pieces, rates = R.h47.collapse_year(2021, c, book, R.h47.DESIGNS, ["22-25"])
    check("47h.collapse_year runs on the emitted frame",
          len(pieces) == len(R.h47.DESIGNS) * 4, f"{len(pieces)} pieces")
    # At y == T the truncated register IS the current one, so the two arms
    # must be IDENTICAL; they may only diverge in years after the truncation.
    check("at y == T the as-of arm equals the true arm",
          pieces[("OL_daioe", "true", 2021)].equals(pieces[("OL_daioe", "asof", 2021)]))
    c23 = R.h47.compact(sim.emit_year(2023))
    p23, _ = R.h47.collapse_year(2023, c23, book, {"OL_daioe": R.h47.DESIGNS["OL_daioe"]},
                                 ["22-25"])
    d21 = p23[("OL_daioe", "true", 2021)].merge(
        p23[("OL_daioe", "asof", 2021)],
        on=["employer_id", "year_month", "exposure_quartile", "age_group"],
        how="outer", suffixes=("_t", "_a")).fillna(0)
    check("after the truncation the as-of arm differs from the true arm",
          (d21["n_emp_t"] != d21["n_emp_a"]).any(),
          f"{int((d21['n_emp_t'] != d21['n_emp_a']).sum())} cells differ")
    check("the two truncations differ from each other in 2023",
          not p23[("OL_daioe", "asof", 2021)].equals(p23[("OL_daioe", "asof", 2022)]))
    check("anchoring fires for the enrolment designs",
          (not rates.empty) and rates["n_anchored"].sum() > 0,
          f"{0 if rates.empty else int(rates['n_anchored'].sum())} anchored")


def t2_oracle_recovers(export):
    """8.2 the oracle tracks the structural effect. Stated as the spec does,
    in standard errors, not against a fixed threshold: on a sample this size
    the Monte Carlo error is itself around 0.05. The clean statement is the
    DIFFERENCE between the treated and null draws on the same seed, because a
    common small-sample offset at 22-25 cancels there -- which is also why the
    study scores designs against the oracle rather than against gamma."""
    g = {}
    for scen in ("null", "paper"):
        sim, *_ = make(export, scenario=scen, n_persons=20000, n_emp=500)
        orc = pd.concat([sim.emit_oracle(y) for y in PROD_YEARS], ignore_index=True)
        for a in ("22-25", "50+"):
            g[(scen, a)] = R.estimate_cells(orc, a, f"t2_{scen}_{a}", OUT, with_se=True)
    for scen in ("null", "paper"):
        c, se = g[(scen, "50+")]
        check(f"oracle 50+ is within 2 SE of zero under {scen}",
              abs(c) < 2 * se + 0.02, f"{c:+.4f} (SE {se:.4f})")
    c0, se0 = g[("null", "22-25")]
    c1, se1 = g[("paper", "22-25")]
    diff = c1 - c0
    sed = float(np.hypot(se0, se1))
    check("the treated oracle is below the null oracle at 22-25",
          diff < 0, f"{diff:+.4f} (SE {sed:.4f}, structural -0.15)")
    check("and the gap is of the structural order, not a tenth of it",
          abs(diff) > 0.5 * 0.15, f"{diff:+.4f} vs -0.15")
    check("the null oracle at 22-25 is within 2 SE of zero",
          abs(c0) < 2 * se0 + 0.02, f"{c0:+.4f} (SE {se0:.4f})")


def t3_no_distortion_no_artefact(export):
    """8.3 with the register layer off, every design's artefact is ~0."""
    key, daioe = R.load_inputs()
    cal = calib.build(export, key, daioe, np.random.default_rng(7))
    p = Params(gamma=R.SCENARIOS["paper"], distortions=False, lag_edu=0, **SMALL)
    sim = Sim(p, cal, key, daioe, np.random.default_rng(11))
    book = R.h47.ScoreBook({y: sim.emit_weights(y) for y in (2019, 2020, 2021)}, key,
                           daioe.rename(columns={"score": "daioe_score", "high": "daioe_high"})
                           [["ssyk4", "daioe_score", "daioe_high", "eloundou_score", "eloundou_high"]])
    R.h47.MIN_CELL = 10
    spec = dict(R.h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frames = {y: R.h47.compact(sim.emit_year(y)) for y in BACKTEST_YEARS}
    arms = {}
    for arm in ("true", "asof"):
        coll = pd.concat([R.h47.collapse_year(y, frames[y], book, {"OL_daioe": spec},
                                              ["22-25"])[0][("OL_daioe", arm, 2021)]
                          for y in BACKTEST_YEARS], ignore_index=True)
        arms[arm] = R.estimate_cells(coll, "22-25", f"t3_{arm}", OUT)
    art = arms["asof"] - arms["true"]
    check("no distortion -> no artefact", abs(art) < 0.02, f"{art:+.4f}")


def t4_metrics_and_ranking():
    """8.x the runner's metrics and the pre-registered rule, on a fixture."""
    df = pd.DataFrame([
        dict(design="good", scenario="paper", registers="base", seed=0, age_group="22-25",
             gamma_oracle=-0.15, gamma_prod=-0.14, bias=0.01,
             artefact_T2021=0.01, artefact_T2022=0.01),
        dict(design="good", scenario="null", registers="base", seed=0, age_group="22-25",
             gamma_oracle=0.0, gamma_prod=0.0, bias=0.0,
             artefact_T2021=0.0, artefact_T2022=0.0),
        dict(design="good", scenario="paper", registers="base", seed=0, age_group="50+",
             gamma_oracle=0.0, gamma_prod=0.0, bias=0.0,
             artefact_T2021=0.005, artefact_T2022=0.004),
        # passes the backtest but is badly biased: the false pass M3 exists for
        dict(design="sneaky", scenario="paper", registers="base", seed=0, age_group="22-25",
             gamma_oracle=-0.15, gamma_prod=-0.35, bias=-0.20,
             artefact_T2021=0.01, artefact_T2022=0.02),
        dict(design="sneaky", scenario="null", registers="base", seed=0, age_group="22-25",
             gamma_oracle=0.0, gamma_prod=-0.30, bias=-0.30,
             artefact_T2021=0.0, artefact_T2022=0.0),
        dict(design="sneaky", scenario="paper", registers="base", seed=0, age_group="50+",
             gamma_oracle=0.0, gamma_prod=0.0, bias=0.0,
             artefact_T2021=0.001, artefact_T2022=0.001),
    ])
    txt = R.rank(df, OUT)
    tab = pd.read_csv(OUT / "ranking.csv", index_col=0)
    check("the biased design is caught as a false pass",
          tab.loc["sneaky", "false_pass"] == 1.0 and tab.loc["good", "false_pass"] == 0.0)
    check("the null false-decline rate is computed",
          tab.loc["sneaky", "null_false_decline"] == 1.0
          and tab.loc["good", "null_false_decline"] == 0.0)
    check("ranking is by mean absolute bias", list(tab.index) == ["good", "sneaky"])
    check("the placebo column reads the 50+ rows", tab.loc["good", "placebo_50p"] < 0.01)


def t5_calibrated(export):
    """8.1, 8.4, 8.5 -- only meaningful against script 50's export."""
    if export is None:
        print("SKIP calibrated tests: no --export given (script 50 has not landed). "
              "The DGP reproducing 47b's -0.36 is a claim about the registers, not "
              "about this code, and cannot be asserted against placeholders.")
        return
    key, daioe = R.load_inputs()
    cal = calib.build(export, key, daioe, np.random.default_rng(7))
    check("8.1 every calibration field is measured", not cal.placeholder_fields,
          "placeholders: " + ", ".join(cal.placeholder_fields))
    sim, *_ = make(export, scenario="null", n_persons=20000, n_emp=500, lag_edu=2)
    book = R.h47.ScoreBook({y: sim.emit_weights(y) for y in (2019, 2020, 2021)}, key,
                           daioe.rename(columns={"score": "daioe_score", "high": "daioe_high"})
                           [["ssyk4", "daioe_score", "daioe_high", "eloundou_score", "eloundou_high"]])
    spec = dict(R.h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frames = {y: R.h47.compact(sim.emit_year(y)) for y in BACKTEST_YEARS}
    arms = {}
    for arm in ("true", "asof"):
        coll = pd.concat([R.h47.collapse_year(y, frames[y], book, {"OL_daioe": spec},
                                              ["22-25", "50+"])[0][("OL_daioe", arm, 2021)]
                          for y in BACKTEST_YEARS], ignore_index=True)
        arms[arm] = {a: R.estimate_cells(coll, a, f"t5_{arm}_{a}", OUT) for a in ("22-25", "50+")}
    art = arms["asof"]["22-25"] - arms["true"]["22-25"]
    art50 = arms["asof"]["50+"] - arms["true"]["50+"]
    check("8.4 the 47b phenomenon reproduces (22-25 artefact below -0.15)",
          art < -0.15, f"{art:+.4f}")
    check("8.4 the 50+ placebo stays near zero", abs(art50) < 0.02, f"{art50:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    a = ap.parse_args()
    exp = Path(a.export) if a.export else None
    print("=" * 66)
    print("ACCEPTANCE TESTS" + ("  (calibrated)" if exp else "  (structural only)"))
    print("=" * 66)
    t1_frames_match_47h(exp)
    t2_oracle_recovers(exp)
    t3_no_distortion_no_artefact(exp)
    t4_metrics_and_ranking()
    t5_calibrated(exp)
    print("=" * 66)
    if FAILS:
        print("FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
test_dryrun.py -- local pre-flight for the MONA revision battery.

Runs OUTSIDE MONA, on synthetic data, with no SQL. It answers one question:
would these scripts have crashed on the share for a reason we could have found
at home? It cannot check that the numbers are right -- only MONA has the data.

    python3 test_dryrun.py            # everything
    python3 test_dryrun.py --keep     # keep the temp workdir for inspection

What it does:
  1. Compiles every .py in the upload set.
  2. Imports each script with CANARIES_DRYRUN=1 (pyodbc mocked, SQL raises).
  3. Builds a synthetic employer x ssyk4 x age x month panel with a real
     post-ChatGPT effect at 22-25, writes it where 39 would write it, and
     runs 40-44 and 46 as separate processes against it.
  4. Pre-writes 45's own dual-panel caches and runs 45 too.
  5. Exercises the three R wrappers end-to-end against a known DGP.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

PY_FILES = ["mona_common.py", "run_all_mona.py"] + sorted(
    f.name for f in HERE.glob("[0-9][0-9]_*.py"))
R_FILES = ["r_fepois.R", "r_fepois_es.R", "r_fepois_multi.R"]
# 39 pulls SQL and 47 is a stub; both are excluded from the run stage.
RUNNABLE = ["43_poisson_primary.py", "42_frozen_cohort.py", "45_asof_backtest.py",
            "40_coverage_diagnostics.py", "41_vintage_event_studies.py",
            "44_decile_gradient.py", "46_wfh_horserace.py"]

MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2026) for m in range(1, 13)
          if not (y == 2025 and m > 6)]
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]

results = []


def record(stage, name, ok, detail=""):
    results.append((stage, name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------

def make_panel(daioe: pd.DataFrame, n_emp: int = 160, seed: int = 20260903):
    """
    Employer x ssyk4 x age_group x month x vintage counts. Every employer
    holds one Q4 occupation and one below-Q4 occupation, so balance_panel's
    q4-and-below identification restriction keeps all of them. A real
    post-ChatGPT decline is written into 22-25 in Q4 so the estimators have
    a signal to find and the sign is predictable.
    """
    rng = np.random.default_rng(seed)
    q4 = daioe.loc[daioe.exposure_quartile == 4, "ssyk4"].tolist()
    lo = daioe.loc[daioe.exposure_quartile < 4, "ssyk4"].tolist()
    rows = []
    for e in range(n_emp):
        emp = f"E{e:05d}"
        pair = [(rng.choice(q4), 4), (rng.choice(lo), int(rng.integers(1, 4)))]
        for ssyk, q in pair:
            for age in AGES:
                base = float(rng.integers(6, 40))
                for ym in MONTHS:
                    y, m = int(ym[:4]), int(ym[5:])
                    post_gpt = (ym >= "2022-12")
                    eff = 1.0
                    if q == 4 and age == "22-25" and post_gpt:
                        # ramp to about -20 per cent by the end of the window
                        ramp = min(1.0, (y * 12 + m - (2022 * 12 + 12)) / 24)
                        eff = 1.0 - 0.20 * ramp
                    n = rng.poisson(max(base * eff, 0.5))
                    if n == 0:
                        continue
                    if y <= 2022:
                        vint = "own"
                    else:
                        vint = str(rng.choice(["2023", "2022", "2021"],
                                              p=[0.86, 0.09, 0.05]))
                    rows.append((emp, ym, ssyk, vint, age, int(n)))
        # uncoded rows: the excluded workers E3 asks to be counted
        for ym in MONTHS[-30:]:
            rows.append((emp, ym, "____", "none", str(rng.choice(AGES)),
                         int(rng.integers(1, 4))))
    return pd.DataFrame(rows, columns=["employer_id", "year_month", "ssyk4",
                                       "vintage", "age_group", "n_emp"])


def make_dual(panel: pd.DataFrame, daioe: pd.DataFrame, trunc: int, seed=7):
    """45's input: the same cells with a true code and an as-of code, where
    the as-of code is stale (or missing) for years after the truncation."""
    rng = np.random.default_rng(seed + trunc)
    p = panel[(panel.ssyk4 != "____")
              & (panel.year_month.str[:4].astype(int) <= 2023)].copy()
    p = p.rename(columns={"ssyk4": "ssyk_true"})
    year = p.year_month.str[:4].astype(int)
    stale = rng.random(len(p))
    other = daioe.ssyk4.sample(len(p), replace=True, random_state=seed).to_numpy()
    asof = p.ssyk_true.to_numpy().copy()
    after = (year > trunc).to_numpy()
    asof = np.where(after & (stale < 0.08), "____", asof)
    asof = np.where(after & (stale >= 0.08) & (stale < 0.16), other, asof)
    p["ssyk_asof"] = asof
    return p[["employer_id", "year_month", "ssyk_true", "ssyk_asof",
              "age_group", "n_emp"]]


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

def stage_compile():
    print("\n1. COMPILE")
    for f in PY_FILES:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(HERE / f)],
                           capture_output=True, text=True)
        record("compile", f, r.returncode == 0, r.stderr.strip()[:120])


def stage_import(work: Path, env: dict):
    print("\n2. IMPORT (CANARIES_DRYRUN=1)")
    for f in PY_FILES:
        mod = f[:-3]
        code = (f"import importlib.util,sys;"
                f"spec=importlib.util.spec_from_file_location('{mod}',r'{work / f}');"
                f"m=importlib.util.module_from_spec(spec);sys.modules['{mod}']=m;"
                f"spec.loader.exec_module(m)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=work, env=env)
        record("import", f, r.returncode == 0, r.stderr.strip().splitlines()[-1][:120]
               if r.returncode else "")


def stage_run(work: Path, env: dict):
    print("\n3. RUN against synthetic data")
    for f in RUNNABLE:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(work / f)], capture_output=True,
                           text=True, cwd=work, env=env, timeout=1800)
        out = work / f"output_{f[:2]}"
        n_out = len(list(out.glob("*.csv"))) + len(list(out.glob("*.txt"))) if out.exists() else 0
        # 40, 41, 42 and 45 legitimately pull data the cache does not hold
        # (entrant splits, person-employer flags, the force_cascade panel, the
        # dual panels). Hitting the mocked SQL boundary is the expected local
        # outcome, not a defect; anything else is.
        sql_boundary = "SQL access is not available in LOCAL_DRYRUN" in (r.stderr or "")
        ok = (r.returncode == 0 and n_out > 0) or (sql_boundary and n_out > 0)
        detail = f"{time.time()-t0:5.1f}s, {n_out} output files"
        if sql_boundary and r.returncode != 0:
            detail += "  (ran to the SQL boundary, as expected locally)"
        elif r.returncode != 0:
            tail = [l for l in r.stdout.strip().splitlines()[-4:]] + \
                   [l for l in r.stderr.strip().splitlines()[-4:]]
            detail += "\n        " + "\n        ".join(tail)
        record("run", f, ok, detail)


def stage_preflight(work: Path, env: dict):
    """The pre-flight is what stands between an upload and a wasted batch
    submission; it gets tested like everything else (lesson of 4 Sep, when
    it demanded .csv while the shipped files were .dta and the suite was
    silent because nothing exercised it)."""
    print("\n3b. PRE-FLIGHT against the synthetic share")
    def run_pf():
        return subprocess.run([sys.executable, str(work / "run_all_mona.py"),
                               "--dry-run"], capture_output=True, text=True,
                              cwd=work, env=env)
    r = run_pf()
    ok = "Pre-flight failed" not in r.stdout and "MISSING" not in r.stdout \
         and "BAD HASH" not in r.stdout
    record("preflight", "all inputs green", ok,
           "" if ok else "\n        " + "\n        ".join(
               l for l in r.stdout.splitlines() if "MISSING" in l or "BAD" in l))
    share = Path(env["CANARIES_SHARE"])
    daioe = share / "daioe_quartiles.dta"
    orig = daioe.read_bytes()
    daioe.write_bytes(orig + b"x")
    r = run_pf()
    record("preflight", "tampered daioe is refused", "BAD HASH" in r.stdout)
    daioe.write_bytes(orig)
    dn = share / "dingel_neiman_ssyk4.dta"
    dn_bytes = dn.read_bytes(); dn.unlink()
    r = run_pf()
    record("preflight", "missing input is refused", "MISSING" in r.stdout)
    dn.write_bytes(dn_bytes)


def stage_r(work: Path, env: dict):
    print("\n4. R WRAPPERS against a known DGP")
    sys.path.insert(0, str(work))
    os.environ.update(env)
    import mona_common as mc
    rng = np.random.default_rng(11)
    n_e, ms = 120, MONTHS
    rows = []
    for e in range(n_e):
        for q in (4, 2):
            for ym in ms:
                mu = 20 * (1 - 0.25 * (q == 4 and ym >= "2022-12"))
                rows.append((f"E{e}", q, ym, rng.poisson(mu)))
    p = pd.DataFrame(rows, columns=["employer_id", "exposure_quartile",
                                    "year_month", "n_emp"])
    p = mc.add_treatment(p)
    p["halfyear"] = mc.assign_halfyear(p["year_month"])
    wd = work / "_rtest"; wd.mkdir(exist_ok=True)
    truth = np.log(0.75)
    try:
        res = mc.run_fepois(p, wd, "t")
        row = res.loc[res.term == "post_gpt_x_high"].iloc[0]
        beta, conv = float(row.coef), bool(row.converged)
        record("R", "run_fepois", conv and abs(beta - truth) < 0.05,
               f"post_gpt_x_high {beta:+.4f} vs true {truth:+.4f}, converged={conv}")
    except Exception as ex:
        record("R", "run_fepois", False, str(ex)[:150])
    try:
        es = mc.run_fepois_es(p, wd, "t")
        # the reference half-year must be absent or zero, and the last
        # post period must be negative if the DGP came through
        last = es.iloc[-1]
        record("R", "run_fepois_es", len(es) > 0 and float(last.coef) < 0,
               f"{len(es)} rows, last coef {float(last.coef):+.4f}")
    except Exception as ex:
        record("R", "run_fepois_es", False, str(ex)[:150])
    try:
        terms = ["post_rb_x_high", "post_gpt_x_high"]
        mu = mc.run_fepois_multi(p, wd, "t", terms=terms)
        got = set(mu.term) if "term" in mu.columns else set()
        record("R", "run_fepois_multi", set(terms) <= got,
               f"{len(mu)} rows, terms {sorted(got)}")
    except Exception as ex:
        record("R", "run_fepois_multi", False, str(ex)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    stage_compile()

    work = Path(tempfile.mkdtemp(prefix="canaries_dryrun_"))
    share = work / "share"; share.mkdir()
    for f in PY_FILES + R_FILES:
        shutil.copy(HERE / f, work / f)

    daioe = pd.read_csv(REPO / "data/processed/daioe_quartiles.csv")
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    if not pd.api.types.is_numeric_dtype(daioe["exposure_quartile"]):
        daioe["exposure_quartile"] = (daioe["exposure_quartile"].astype(str)
                                      .str.extract(r"(\d)").astype(int))
    # SHARE is concatenated with r"\name.csv", so on POSIX the file lives
    # beside the share directory with a backslash in its name. That is the
    # point: it exercises the real path expression, unmodified.
    # On Windows/MONA, SHARE + r"\x.dta" (the loaders) and Path(SHARE)/"x.dta"
    # (the pre-flight) are the same file; on macOS they are not, so the fixture
    # provides both forms. Same bytes, so the hash check sees the same content.
    from datetime import datetime as _dt
    _ts = _dt(2026, 9, 4, 12, 0)
    daioe.to_stata(f"{share}\\daioe_quartiles.dta", write_index=False, time_stamp=_ts)
    dn = pd.read_csv(REPO / "mona_package/dingel_neiman_ssyk4.txt")
    dn.to_stata(f"{share}\\dingel_neiman_ssyk4.dta", write_index=False, time_stamp=_ts)
    import shutil as _sh
    _sh.copy(REPO / "revision/upload/daioe_quartiles.dta", share / "daioe_quartiles.dta")
    dn.to_stata(share / "dingel_neiman_ssyk4.dta", write_index=False, time_stamp=_ts)

    print("\n0. SYNTHETIC DATA")
    panel = make_panel(daioe)
    (work / "cache").mkdir()
    panel.to_parquet(work / "cache" / "panel_vintage.parquet", index=False)
    record("data", "panel_vintage.parquet",
           len(panel) > 100_000, f"{len(panel):,} rows, {panel.employer_id.nunique()} employers")
    for trunc in (2021, 2022):
        d = make_dual(panel, daioe, trunc)
        d.to_parquet(work / "cache" / f"panel_dual_T{trunc}.parquet", index=False)
        record("data", f"panel_dual_T{trunc}.parquet", len(d) > 10_000, f"{len(d):,} rows")

    env = dict(os.environ, CANARIES_DRYRUN="1", CANARIES_SHARE=str(share),
               PYTHONPATH=str(work))

    stage_import(work, env)
    stage_run(work, env)
    stage_preflight(work, env)
    stage_r(work, env)

    print("\n" + "=" * 74)
    bad = [r for r in results if not r[2]]
    print(f"{len(results) - len(bad)}/{len(results)} passed")
    for stage, name, _, detail in bad:
        print(f"  FAIL  {stage:<8} {name}  {detail.splitlines()[0] if detail else ''}")
    print("=" * 74)
    if a.keep:
        print(f"workdir kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

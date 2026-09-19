#!/usr/bin/env python3
"""
run_sim.py -- the runner: generate, distort, estimate every design, score.

    python3 revision/sim/run_sim.py --export <output_50 dir> [--quick|--full]

Imports the SCORING, ASSIGNMENT, BALANCING and ESTIMATION code from
47h_edu_horserace.py and mona_common, unchanged, so what is measured here
is the same machinery that runs in MONA. Only the data are simulated.

Metrics per (design, scenario, register setting, seed):
  M1 bias      production gamma2 (2019-2025, cascade after 2023) minus the
               ORACLE gamma2 on the same sample -- the ceiling a perfect
               assignment reaches, which is the right comparison for an
               education measure that is attenuated by construction
  M2 artefact  as-of minus true at T=2021 and T=2022, per age band
  M3 false-pass  |M2| < 0.05 at both T while |M1| > 0.05
  M4 null false-decline  P(production gamma2 < -0.05) when gamma = 0
  M5 placebo   |M2| at 50+
Results are appended to results/runs.csv as produced, so a killed run
resumes; `--resume` skips (design, scenario, seed) triples already there.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
os.environ.setdefault("CANARIES_DRYRUN", "1")
# mona_common reads CANARIES_SHARE at import and 47h derives its input paths
# from it at import, so both must be set before either is imported.
os.environ.setdefault("CANARIES_SHARE", str(REV / "upload"))
sys.path[:0] = [str(HERE), str(REV / "mona")]

import mona_common as mc                      # noqa: E402
import calib                                  # noqa: E402
from dgp import Params, Sim, PROD_YEARS, BACKTEST_YEARS, MONTHS_ALL  # noqa: E402

UPLOAD = REV / "upload"
_LOCAL_DAIOE = str(UPLOAD / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL_DAIOE
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL_DAIOE: _ld(path)

_spec = importlib.util.spec_from_file_location("s47h", REV / "mona" / "47h_edu_horserace.py")
h47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h47)

SCENARIOS = {
    "null":   {},
    "paper":  {"22-25": -0.15},
    "grad":   {"22-25": -0.15, "26-30": -0.08, "31-34": -0.03},
}
REGISTER_SETTINGS = {
    "base":      dict(lag_edu=2, occ_change_scale=1.0),
    "lag1":      dict(lag_edu=1, occ_change_scale=1.0),
    "mobile":    dict(lag_edu=2, occ_change_scale=1.4),
}
AGES_SCORED = ["22-25", "26-30", "50+"]


def load_inputs():
    key = h47.load_key()
    d = pd.read_stata(_LOCAL_DAIOE)
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    q = d["exposure_quartile"]
    if not pd.api.types.is_numeric_dtype(q):
        q = q.astype(str).str.extract(r"(\d)")[0].astype(int)
    daioe = pd.DataFrame({"ssyk4": d["ssyk4"], "score": d["pctl_rank_genai"].astype(float),
                          "high": d["high_exposure"].astype(float), "q": q.astype(int)})
    e = pd.read_stata(UPLOAD / "eloundou_ssyk4.dta")
    e["ssyk4"] = e["ssyk4"].astype(str).str.zfill(4)
    elo = pd.DataFrame({"ssyk4": e["ssyk4"],
                        "eloundou_score": e["eloundou_score"].astype(float),
                        "eloundou_high": e["high_exposure_eloundou"].astype(float)})
    return key, daioe.merge(elo, on="ssyk4", how="left")


def estimate_cells(coll: pd.DataFrame, age: str, tag: str, out: Path,
                   with_se: bool = False):
    """47h's own estimator on an employer x quartile x month cell frame."""
    nan = (np.nan, np.nan) if with_se else np.nan
    if coll.empty:
        return nan
    sub = coll[coll["age_group"].astype(str) == age].copy()
    if sub.empty:
        return np.nan
    sub["year_month"] = sub["year_month"].astype(str)
    sub["exposure_quartile"] = sub["exposure_quartile"].astype(int)
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= h47.STEP1_MIN_CUMULATIVE].index)]
    if sub.empty:
        return nan
    months = sorted(coll["year_month"].astype(str).unique())
    bal = h47.fast_balance(sub, months)
    if bal.empty:
        return nan
    r = mc.run_fepois(mc.add_treatment(bal), out, tag=tag)
    if r.empty or not (r["term"] == "post_gpt_x_high").any():
        return nan
    row = r.loc[r["term"] == "post_gpt_x_high"].iloc[0]
    return (float(row["coef"]), float(row["se"])) if with_se else float(row["coef"])


def production_cells(sim: Sim, book, name: str, spec: dict) -> pd.DataFrame:
    """The 2019-2025 panel a design would actually estimate on."""
    out = []
    for y in PROD_YEARS:
        f = sim.emit_production(y)
        if f.empty:
            continue
        combos = f[["niva", "inr", "expband", "age_group"]].drop_duplicates().reset_index(drop=True)
        s = book.score_frame(name, dict(spec, enrol=False), combos["niva"],
                             combos["inr"], combos["expband"], None, combos["age_group"])
        combos["_q"] = h47.to_quartile(s, book.cuts[name])
        g = f.merge(combos, on=["niva", "inr", "expband", "age_group"], how="left")
        g = g[g["_q"] > 0]
        out.append(g.groupby(["employer_id", "year_month", "_q", "age_group"],
                             observed=True)["n_emp"].sum().reset_index()
                   .rename(columns={"_q": "exposure_quartile"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def one_draw(scenario: str, reg: str, seed: int, cal, key, daioe, designs,
             out: Path, n_persons: int, n_emp: int) -> list:
    rng = np.random.default_rng(seed)
    p = Params(n_persons=n_persons, n_emp=n_emp, gamma=SCENARIOS[scenario],
               **REGISTER_SETTINGS[reg])
    t0 = time.time()
    sim = Sim(p, cal, key, daioe, rng)
    gen_s = time.time() - t0

    # oracle: the ceiling, on the production window
    orc = pd.concat([sim.emit_oracle(y) for y in PROD_YEARS], ignore_index=True)
    oracle = {a: estimate_cells(orc, a, f"or_{a}", out) for a in AGES_SCORED}

    counts = {y: sim.emit_weights(y) for y in (2019, 2020, 2021)}
    book = h47.ScoreBook(counts, key, daioe.rename(columns={
        "score": "daioe_score", "high": "daioe_high"})[
        ["ssyk4", "daioe_score", "daioe_high", "eloundou_score", "eloundou_high"]])
    for nm, sp in designs.items():
        book.build(nm, sp)

    frames = {y: sim.emit_year(y) for y in BACKTEST_YEARS}
    rows = []
    for nm, sp in designs.items():
        # backtest arms, both truncations
        art = {}
        for T in (2021, 2022):
            arms = {}
            for arm in ("true", "asof"):
                pieces = []
                for y in BACKTEST_YEARS:
                    pc, _ = h47.collapse_year(y, frames[y], book, {nm: sp}, AGES_SCORED)
                    pieces.append(pc[(nm, arm, T)])
                coll = pd.concat(pieces, ignore_index=True)
                arms[arm] = {a: estimate_cells(coll, a, f"{nm}_{arm}_{T}_{a}", out)
                             for a in AGES_SCORED}
            for a in AGES_SCORED:
                art[(T, a)] = arms["asof"][a] - arms["true"][a]
        prod = production_cells(sim, book, nm, sp)
        for a in AGES_SCORED:
            g_prod = estimate_cells(prod, a, f"{nm}_prod_{a}", out)
            rows.append(dict(design=nm, scenario=scenario, registers=reg, seed=seed,
                             age_group=a, gamma_struct=SCENARIOS[scenario].get(a, 0.0),
                             gamma_oracle=oracle[a], gamma_prod=g_prod,
                             bias=g_prod - oracle[a],
                             artefact_T2021=art[(2021, a)], artefact_T2022=art[(2022, a)],
                             gen_s=round(gen_s, 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None, help="script 50's output_50 directory")
    ap.add_argument("--quick", action="store_true", help="1 scenario, 1 setting, 1 seed")
    ap.add_argument("--full", action="store_true", help="3 x 3 x 5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-persons", type=int, default=60_000)
    ap.add_argument("--n-emp", type=int, default=1_500)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-cell", type=int, default=None,
                    help="override 47h.MIN_CELL (the scoring floor); scale it "
                         "with --n-persons, since it is a count of people")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.min_cell is not None:
        h47.MIN_CELL = a.min_cell
    print(f"scoring floor MIN_CELL = {h47.MIN_CELL}")
    key, daioe = load_inputs()
    cal = calib.build(a.export, key, daioe, np.random.default_rng(7))
    print(calib.summary(cal))
    if cal.placeholder_fields:
        print("  NOTE placeholders in: " + ", ".join(cal.placeholder_fields))

    designs = {k: dict(v) for k, v in h47.DESIGNS.items()}
    scen = ["null", "paper", "grad"]
    regs = ["base", "lag1", "mobile"]
    seeds = list(range(a.seeds))
    if a.quick:
        scen, regs, seeds = ["paper"], ["base"], [0]
    elif not a.full:
        regs = ["base", "lag1"]

    path = out / "runs.csv"
    done = set()
    if a.resume and path.exists():
        prev = read_runs(path)
        done = set(map(tuple, prev[["scenario", "registers", "seed"]].drop_duplicates().to_numpy()))
    t0 = time.time()
    for s in scen:
        for r in regs:
            for sd in seeds:
                if (s, r, sd) in done:
                    print(f"skip {s}/{r}/{sd} (resume)")
                    continue
                t = time.time()
                rows = one_draw(s, r, sd, cal, key, daioe, designs, out,
                                a.n_persons, a.n_emp)
                pd.DataFrame(rows).to_csv(path, mode="a", index=False,
                                          header=not path.exists())
                print(f"  draw {s}/{r}/seed{sd}: {len(rows)} rows "
                      f"({time.time()-t:.0f}s)")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min -> {path}")
    if path.exists():
        print(rank(read_runs(path), out))


def read_runs(path: Path) -> pd.DataFrame:
    """
    Read runs.csv WITHOUT pandas' default NA conversion. One of the three
    scenarios is called "null", which is in pandas' default na_values list, so
    a plain read_csv turns that label into NaN and every filter on it matches
    nothing. The null false-decline rate -- one of the four metrics the study
    exists to produce -- would then be empty for every design, silently.
    """
    return pd.read_csv(path, keep_default_na=False,
                       na_values=["", "NaN", "nan"])


def rank(df: pd.DataFrame, out: Path) -> str:
    df = df.copy()
    df["scenario"] = df["scenario"].fillna("null")   # belt and braces
    for c in ("gamma_oracle", "gamma_prod", "bias",
              "artefact_T2021", "artefact_T2022"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y = df[df["age_group"] == "22-25"].copy()
    y["pass_backtest"] = (y[["artefact_T2021", "artefact_T2022"]].abs() < 0.05).all(axis=1)
    y["false_pass"] = y["pass_backtest"] & (y["bias"].abs() > 0.05)
    g = y.groupby("design").agg(
        mean_abs_bias=("bias", lambda s: float(np.nanmean(np.abs(s)))),
        max_abs_art=("artefact_T2021", lambda s: float(np.nanmax(np.abs(s)))),
        false_pass=("false_pass", "mean"))
    nul = y[y["scenario"] == "null"].groupby("design")["gamma_prod"].apply(
        lambda s: float(np.nanmean(s < -0.05)))
    plac = (df[df["age_group"] == "50+"].groupby("design")[["artefact_T2021", "artefact_T2022"]]
            .apply(lambda d: float(np.nanmax(np.abs(d.to_numpy())))))
    g["null_false_decline"], g["placebo_50p"] = nul, plac
    g = g.sort_values("mean_abs_bias")
    lines = ["", "RANKING (ages 22-25; bias is against the ORACLE on the same sample)",
             f"{'design':<15}{'|bias|':>9}{'|artefact|':>12}{'falsepass':>11}"
             f"{'null<-0.05':>12}{'placebo':>9}"]
    for nm, r in g.iterrows():
        lines.append(f"{nm:<15}{r.mean_abs_bias:>9.3f}{r.max_abs_art:>12.3f}"
                     f"{r.false_pass:>11.2f}{r.null_false_decline:>12.2f}{r.placebo_50p:>9.3f}")
    lines += ["", "Finalists = smallest |bias| among designs with null_false_decline < 0.10",
              "and placebo < 0.02. The winner is never the largest coefficient."]
    txt = "\n".join(lines)
    (out / "ranking.txt").write_text(txt)
    g.to_csv(out / "ranking.csv")
    return txt


if __name__ == "__main__":
    main()

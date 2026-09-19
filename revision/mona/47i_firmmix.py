#!/usr/bin/env python3
"""
47i_firmmix.py -- the register route that survives: exposure on the FIRM,
not on the young worker.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_47i/. Reads 47h's cached year frames, so it needs
  NO SQL of its own once 47h has pulled (about 10 minutes if the cache
  is warm; it rebuilds the pulls itself if it is not).
  Local end-to-end test: revision/local/test_47i_synthetic.py
======================================================================

WHY (19 Sep 2026, ML's ruling). Script 47b showed that assigning exposure
to a 22-25 year old from their own education record fails the as-of
backtest worse than the occupation design: artefact -0.36, because at that
age people are still completing the education the register reports. All
eight designs in 47h assign at the worker level, so all eight inherit that
exposure.

Nordstrom Skans and Sokolow Romin (2026) find the young-worker pattern on
the same registers and do NOT hit this, because their exposure is a
property of the FIRM: the worker-weighted mean over the educational
composition of its whole workforce, binned into quartiles fixed in 2019.
A firm's mix is dominated by incumbents whose education settled years ago,
so a stale record for one 23-year-old moves it by almost nothing. The young
worker is then counted as an outcome, never classified.

THIS SCRIPT applies that to our outcome and our estimator, and puts it
through the same backtest, because "their design does not have the problem"
is an argument and the artefact is a measurement.

DESIGN
  exposure   firm f's worker-weighted mean education-group DAIOE genAI
             score in year y, over ALL its workers of every age; quartile
             cutoffs fixed on the 2019 worker-weighted distribution and
             applied to every later year (their rule: firms may move
             between quartiles, the thresholds may not)
  cells      employer x month, employment of one age band
  spec       Poisson, employer FE + month FE, PostRB x High and
             PostGPT x High, clustered on employer. High = firm quartile 4.
             NOTE this is ACROSS-firm identification: a firm has one
             quartile, so the paper's within-employer comparison is not
             available here. That is the price of the fix and it is stated
             in the output, not buried.
  industry   the same estimate on firms reweighted to the 3-digit industry
             mix of Q4 is NOT computed here (no industry column in the
             cached frames); flagged as the obvious extension.
  backtest   the firm mix is computed twice, from the true education record
             and from the register truncated at T in {2021, 2022}, and the
             artefact is reported for every age band, as in 47b and 47h.

READ RULE, PRE-COMMITTED, the same bar as 47b: an artefact below 0.05 in
absolute value at both truncations at 22-25 means this design can carry
register evidence; between that and half the occupation artefact
(0.153 / 0.081) it is usable with the artefact stated beside every
estimate; at or above half it cannot.
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_47i"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YEARS = [2019, 2020, 2021, 2022, 2023]
TRUNCATIONS = (2021, 2022)
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
OCC_ARTEFACT = {2021: -0.3068, 2022: -0.1627}
MIN_FIRM_WORKERS = 5          # as the paper's employer floor
TERMS = ["post_rb_x_high", "post_gpt_x_high"]
FES = ("employer_id", "year_month")


def _h47():
    """Import 47h for its key, scores and score machinery. Kept behind a
    function so an import failure lands in this script's log, not in a
    silent stderr BatchClient discards."""
    import importlib.util
    _here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "h47", _here / "47h_edu_horserace.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def firm_quartiles(frame: pd.DataFrame, book, name: str, spec: dict,
                   arm: str, T: int, cuts=None):
    """
    Firm-year exposure from the education mix of the WHOLE workforce.
    Returns (employer_id, year_month, quartile) plus the cutoffs used.
    `cuts` is None for the 2019 base year (they are computed and returned)
    and passed in for every later year, which is the fixed-threshold rule.
    """
    if arm == "true":
        cols = ["niva_t", "inr_t", "expb_t"]
        enr = None
    else:
        cols = [f"niva_{T % 100}", f"inr_{T % 100}", f"expb_{T % 100}"]
        enr = f"enr_{T % 100}" if spec.get("enrol") else None
    keycols = cols + ([enr] if enr else []) + ["age_group"]
    combos = frame[keycols].drop_duplicates().reset_index(drop=True)
    s = book.score_frame(name, spec, combos[cols[0]], combos[cols[1]],
                         combos[cols[2]], combos[enr] if enr else None,
                         combos["age_group"])
    combos["_s"] = s
    g = frame.merge(combos, on=keycols, how="left")
    g = g[g["_s"].notna()]
    if g.empty:
        return pd.DataFrame(columns=["employer_id", "year_month", "fq"]), cuts
    g["_ws"] = g["_s"] * g["n_emp"]
    # one exposure per employer-YEAR (not month): a monthly mean would move
    # with hiring, which is the outcome, and that would be endogenous
    fy = (g.groupby("employer_id", observed=True)
          .agg(ws=("_ws", "sum"), n=("n_emp", "sum")).reset_index())
    fy = fy[fy["n"] >= MIN_FIRM_WORKERS]
    fy["mix"] = fy["ws"] / fy["n"]
    if cuts is None:
        o = np.argsort(fy["mix"].to_numpy(), kind="stable")
        v, w = fy["mix"].to_numpy()[o], fy["n"].to_numpy()[o]
        cum = np.cumsum(w) / w.sum()
        cuts = [float(v[np.searchsorted(cum, q, side="left")]) for q in (0.25, 0.5, 0.75)]
    fy["fq"] = np.searchsorted(np.asarray(cuts), fy["mix"].to_numpy(), side="right") + 1
    return fy[["employer_id", "fq", "mix", "n"]], cuts


def build_panel(frames: dict, book, name: str, spec: dict, arm: str, T: int,
                age: str):
    """employer x month employment of one age band, with the firm's quartile."""
    cuts = None
    pieces, mixes = [], []
    for y in YEARS:
        f = frames[y]
        fq, cuts = firm_quartiles(f, book, name, spec, arm, T, cuts)
        if fq.empty:
            continue
        mixes.append(fq.assign(year=y))
        sub = f[f["age_group"].astype(str) == age]
        cell = (sub.groupby(["employer_id", "year_month"], observed=True)["n_emp"]
                .sum().reset_index())
        pieces.append(cell.merge(fq[["employer_id", "fq"]], on="employer_id", how="inner"))
    if not pieces:
        return pd.DataFrame(), pd.DataFrame()
    panel = pd.concat(pieces, ignore_index=True)
    months = sorted(panel["year_month"].astype(str).unique())
    # balance: every surviving employer in every month, zero-filled
    emp = panel[["employer_id", "fq"]].drop_duplicates("employer_id")
    full = pd.MultiIndex.from_product(
        [emp["employer_id"].to_numpy(), months], names=["employer_id", "year_month"])
    bal = (panel.groupby(["employer_id", "year_month"], observed=True)["n_emp"].sum()
           .reindex(full, fill_value=0).reset_index()
           .merge(emp, on="employer_id", how="left"))
    bal["n_emp"] = bal["n_emp"].astype(int)
    bal["year_month"] = bal["year_month"].astype(str)
    bal["post_rb"] = (bal["year_month"] >= mc.RIKSBANK_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= mc.CHATGPT_YM).astype(int)
    bal["high"] = (bal["fq"] == 4).astype(int)
    bal["post_rb_x_high"] = bal["post_rb"] * bal["high"]
    bal["post_gpt_x_high"] = bal["post_gpt"] * bal["high"]
    return bal, pd.concat(mixes, ignore_index=True)


def fit(bal: pd.DataFrame, tag: str) -> dict:
    if bal.empty:
        return {"gamma2": np.nan, "se": np.nan, "n_obs": 0, "status": "empty"}
    r = mc.run_fepois_multi(bal, OUT, tag=tag, terms=TERMS, fes=FES)
    row = {"gamma2": np.nan, "se": np.nan, "n_obs": len(bal), "status": "no_output"}
    if not r.empty and (r["term"] == "post_gpt_x_high").any():
        g = r[r["term"] == "post_gpt_x_high"].iloc[0]
        row.update(gamma2=float(g["coef"]), se=float(g["se"]),
                   status=str(g.get("status", "ok")))
    return row


def main():
    mc.Tee(OUT / "47i_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("47i: FIRM-MIX EXPOSURE -- the education route that does not")
    print("     classify the young worker (Nordstrom Skans & Sokolow Romin)")
    print("=" * 70)
    print(mc.mem_line("  "))
    h47 = _h47()

    key = h47.load_key()
    scores = h47.load_scores()
    conn = None
    counts = {}
    for y in (2019, 2020, 2021):
        cf = CACHE / f"edu_hr_weights_{y}.parquet"
        w = mc.read_cache(cf, require=h47.WEIGHT_COLS)
        if w is None:
            conn = conn or mc.connect()
            w = h47.pull_weights(y, conn)
            mc.write_cache(w, cf)
            print(f"  weights {y}: pulled {len(w):,} cells")
        else:
            print(f"  weights {y}: cached")
        counts[y] = w
    book = h47.ScoreBook(counts, key, scores)
    # Two scorings only: the published mapping (OL_daioe) and the entrant
    # variant. The firm mix averages over a whole workforce, so the finer
    # worker-level refinements have little room to matter here.
    designs = {k: dict(h47.DESIGNS[k]) for k in ("OL_daioe", "entrant")}
    for nm, sp in designs.items():
        book.build(nm, sp)

    frames = {}
    for y in YEARS:
        cf = CACHE / f"edu_hr_{y}.parquet"
        f = mc.read_cache(cf, require=h47.YEAR_COLS + ["n_emp"])
        if f is None:
            conn = conn or mc.connect()
            print(f"  {y}: no 47h cache, pulling")
            f = h47.pull_year(y, conn, True)
            mc.write_cache(f, cf)
        else:
            print(f"  {y}: cached ({len(f):,} cells)")
        frames[y] = f

    rows, mixrows = [], []
    for nm, sp in designs.items():
        for T in TRUNCATIONS:
            for arm in ("true", "asof"):
                for age in AGES:
                    bal, mix = build_panel(frames, book, nm, sp, arm, T, age)
                    r = fit(bal, f"fm_{nm}_{arm}_{T}_{age.replace('-', '_').replace('+', 'p')}")
                    r.update(design=nm, arm=arm, trunc=T, age_group=age)
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(OUT / "firmmix_estimates.csv", index=False)
                    print(f"  {nm:<10} {arm:<4} T{T} {age:<5} gamma2 {r['gamma2']:+.4f} "
                          f"(SE {r['se']:.4f}) n {r['n_obs']:,} {r['status']}")
                    if arm == "true" and age == "22-25" and not mix.empty:
                        q = (mix.groupby(["year", "fq"], observed=True)
                             .agg(firms=("employer_id", "nunique"),
                                  workers=("n", "sum")).reset_index())
                        q["design"], q["trunc"] = nm, T
                        mixrows.append(q)
                    del bal, mix
                    gc.collect()
    est = pd.DataFrame(rows)
    if mixrows:
        mc.enforce_min_cell(pd.concat(mixrows, ignore_index=True), count_col="firms") \
            .to_csv(OUT / "firmmix_quartile_sizes.csv", index=False)

    lines = ["FIRM-MIX EXPOSURE: ARTEFACT BY AGE BAND", "=" * 58,
             "Exposure is the firm's workforce education mix, so a young",
             "worker's own (stale) record never enters the classification.",
             "Identification is ACROSS firms: a firm holds one quartile, so the",
             "paper's within-employer comparison is not available in this design.",
             "", "Occupation design (45), 22-25: "
             + "  ".join(f"T{T} {v:+.4f}" for T, v in OCC_ARTEFACT.items()),
             "Worker-level education design (47b), 22-25: T2021 -0.3596  T2022 -0.2941",
             ""]
    for nm in designs:
        lines.append(f"{nm}:")
        for age in AGES:
            for T in TRUNCATIONS:
                s = est[(est.design == nm) & (est.age_group == age) & (est.trunc == T)]
                if len(s) == 2:
                    tr = float(s[s.arm == "true"]["gamma2"].iloc[0])
                    af = float(s[s.arm == "asof"]["gamma2"].iloc[0])
                    lines.append(f"  {age:<5} T{T}  true {tr:+.4f}  as-of {af:+.4f}"
                                 f"  ARTEFACT {af - tr:+.4f}")
        lines.append("")
    lines += ["READ RULE (pre-committed, as 47b):",
              "  |artefact| < 0.05 at both truncations at 22-25 -> this design can",
              "    carry the paper's register evidence.",
              "  0.05 to half the occupation artefact (0.153 / 0.081) -> usable with",
              "    the artefact stated beside every estimate.",
              "  at or above half -> the register route is closed by lag.",
              "",
              "NOT DONE HERE: the industry-reweighted contrast (no industry column in",
              "the cached frames) and hires as an outcome (needs the 12-month-earlier",
              "presence flag). Both are additions, not corrections.",
              f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "47i_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

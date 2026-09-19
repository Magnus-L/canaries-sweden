#!/usr/bin/env python3
"""
47j_within_employer_triple.py -- their exposure idea, our within-employer
identification: a triple difference in which no young worker is ever
classified.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_47j/. Reads 47h's cached year frames, so no SQL of
  its own once 47h has pulled (about 10 minutes warm).
  Local end-to-end test: revision/local/test_47j_synthetic.py
======================================================================

THE PROBLEM THIS SOLVES (19 Sep 2026). Two designs, each missing what the
other has:

  47b / 47h  within-employer across exposure quartiles -- the paper's
             identification, strong, but it must sort each 22-25 year old
             into a quartile from their own education record, which at that
             age is a record of who they were before the degree. Artefact
             -0.36.
  47i        firm-mix exposure, after Nordstrom Skans and Sokolow Romin --
             robust, because a firm's mix is dominated by settled
             incumbents, but a firm holds ONE quartile, so identification
             is across firms and every firm-level shock is a confounder.

The combination. Put the exposure on the FIRM, where it is stable, and take
the within-employer variation from AGE, which is the one worker attribute
that cannot go stale: birth year is in the register, complete, and correct
for everyone. Compare young to older workers INSIDE the same employer in
the same month, and ask whether that gap moves more in exposed firms.

  outcome    employment in employer x age band x month cells
  treatment  PostGPT x High(firm) x Young(22-25)
  absorbed   employer x month  (every firm-time shock, including the
                                firm-level PostGPT x High that 47i relies on)
             employer x age    (a firm's standing age composition)
             month x age       (the economy-wide path of each age band,
                                including any general young-worker decline)
  left       exactly the triple interaction, identified off young versus
             older workers within one employer in one month.

WHY THE CLASSIFIER CANNOT GO STALE. Firm exposure is the worker-weighted
mean education score over the firm's INCUMBENTS ONLY -- workers aged 31 and
over -- measured in 2019 and held fixed. Three consequences: the young never
enter the classifier, so their records cannot contaminate it; 2019 is deep
inside the education register's coverage, so the measure is not a cascade;
and being fixed pre-shock, it cannot respond to the shock. The backtest is
run anyway, because that is an argument and the artefact is a measurement.

WHAT IT COSTS. The estimand changes: this is the young-old gap within
exposed employers, not the level of young employment in exposed cells. A
shock that hit every age equally inside exposed firms would not show up.
That is stated in the output.

READ RULE, PRE-COMMITTED, the same bar as 47b and 47i: artefact below 0.05
in absolute value at both truncations means the design can carry register
evidence; 0.05 to half the occupation artefact (0.153 / 0.081) means usable
with the artefact beside every estimate; at or above half, closed.
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
OUT = HERE / "output_47j"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YEARS = [2019, 2020, 2021, 2022, 2023]
TRUNCATIONS = (2021, 2022)
BASE_YEAR = 2019                      # exposure is fixed here, pre-shock
INCUMBENT_BANDS = ["31-34", "35-40", "41-49", "50+"]
ALL_BANDS = ["22-25", "26-30"] + INCUMBENT_BANDS
YOUNG_BANDS = ["22-25", "26-30"]      # each is treated in turn
MIN_FIRM_INCUMBENTS = 5
OCC_ARTEFACT = {2021: -0.3068, 2022: -0.1627}
TERMS = ["post_rb_x_high_x_young", "post_gpt_x_high_x_young"]
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")


def _h47():
    import importlib.util
    # locate 47h relative to THIS FILE, not to the module-level HERE: HERE is
    # a mutable global and a caller that reassigns it must not break the import
    _here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("h47", _here / "47h_edu_horserace.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def incumbent_exposure(frame19: pd.DataFrame, book, name: str, spec: dict,
                       arm: str, T: int):
    """
    Firm exposure from INCUMBENTS ONLY (31+) in the base year, worker-weighted,
    with quartile cutoffs fixed on that same worker-weighted distribution.
    Returns (employer_id, fq, mix, n_incumbents) and the cutoffs.
    """
    if arm == "true":
        cols = ["niva_t", "inr_t", "expb_t"]
        enr = None
    else:
        cols = [f"niva_{T % 100}", f"inr_{T % 100}", f"expb_{T % 100}"]
        enr = f"enr_{T % 100}" if spec.get("enrol") else None
    f = frame19[frame19["age_group"].astype(str).isin(INCUMBENT_BANDS)]
    keycols = cols + ([enr] if enr else []) + ["age_group"]
    combos = f[keycols].drop_duplicates().reset_index(drop=True)
    if combos.empty:
        return pd.DataFrame(columns=["employer_id", "fq", "mix", "n"]), None
    s = book.score_frame(name, spec, combos[cols[0]], combos[cols[1]],
                         combos[cols[2]], combos[enr] if enr else None,
                         combos["age_group"])
    combos["_s"] = s
    g = f.merge(combos, on=keycols, how="left")
    g = g[g["_s"].notna()]
    if g.empty:
        return pd.DataFrame(columns=["employer_id", "fq", "mix", "n"]), None
    g["_ws"] = g["_s"] * g["n_emp"]
    fy = (g.groupby("employer_id", observed=True)
          .agg(ws=("_ws", "sum"), n=("n_emp", "sum")).reset_index())
    fy = fy[fy["n"] >= MIN_FIRM_INCUMBENTS]
    if fy.empty:
        return pd.DataFrame(columns=["employer_id", "fq", "mix", "n"]), None
    fy["mix"] = fy["ws"] / fy["n"]
    o = np.argsort(fy["mix"].to_numpy(), kind="stable")
    v, w = fy["mix"].to_numpy()[o], fy["n"].to_numpy()[o]
    cum = np.cumsum(w) / w.sum()
    cuts = [float(v[np.searchsorted(cum, q, side="left")]) for q in (0.25, 0.5, 0.75)]
    fy["fq"] = np.searchsorted(np.asarray(cuts), fy["mix"].to_numpy(), side="right") + 1
    return fy[["employer_id", "fq", "mix", "n"]], cuts


def build_panel(frames: dict, expo: pd.DataFrame, young: str) -> pd.DataFrame:
    """Balanced employer x age band x month panel with the triple interaction."""
    pieces = []
    for y in YEARS:
        f = frames[y]
        sub = f[f["age_group"].astype(str).isin(ALL_BANDS)]
        pieces.append(sub.groupby(["employer_id", "year_month", "age_group"],
                                  observed=True)["n_emp"].sum().reset_index())
    panel = pd.concat(pieces, ignore_index=True)
    panel["year_month"] = panel["year_month"].astype(str)
    panel["age_group"] = panel["age_group"].astype(str)
    panel = panel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    # an employer must hold BOTH the young band and at least one older band,
    # or it contributes nothing to a within-employer age comparison
    have = panel.groupby("employer_id")["age_group"].agg(set)
    keep = have[have.apply(lambda s: young in s and bool(s & set(INCUMBENT_BANDS)))].index
    panel = panel[panel["employer_id"].isin(keep)]
    if panel.empty:
        return panel
    months = sorted(panel["year_month"].unique())
    bands = [young] + INCUMBENT_BANDS
    emp = panel[["employer_id", "fq"]].drop_duplicates("employer_id")
    full = pd.MultiIndex.from_product([emp["employer_id"].to_numpy(), bands, months],
                                      names=["employer_id", "age_group", "year_month"])
    bal = (panel.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index()
           .merge(emp, on="employer_id", how="left"))
    bal["n_emp"] = bal["n_emp"].astype(int)
    bal["post_rb"] = (bal["year_month"] >= mc.RIKSBANK_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= mc.CHATGPT_YM).astype(int)
    bal["high"] = (bal["fq"] == 4).astype(int)
    bal["young"] = (bal["age_group"] == young).astype(int)
    bal["post_rb_x_high_x_young"] = bal["post_rb"] * bal["high"] * bal["young"]
    bal["post_gpt_x_high_x_young"] = bal["post_gpt"] * bal["high"] * bal["young"]
    e = bal["employer_id"].astype(str)
    bal["fe_emp_t"] = e + "_" + bal["year_month"]
    bal["fe_emp_age"] = e + "_" + bal["age_group"]
    bal["fe_t_age"] = bal["year_month"] + "_" + bal["age_group"]
    return bal


def fit(bal: pd.DataFrame, tag: str) -> dict:
    row = {"gamma3": np.nan, "se": np.nan, "n_obs": len(bal), "status": "empty"}
    if bal.empty:
        return row
    r = mc.run_fepois_multi(bal, OUT, tag=tag, terms=TERMS, fes=FES)
    row["status"] = "no_output"
    if not r.empty and (r["term"] == "post_gpt_x_high_x_young").any():
        g = r[r["term"] == "post_gpt_x_high_x_young"].iloc[0]
        row.update(gamma3=float(g["coef"]), se=float(g["se"]),
                   status=str(g.get("status", "ok")))
    return row


def main():
    mc.Tee(OUT / "47j_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("47j: WITHIN-EMPLOYER TRIPLE DIFFERENCE ON FIRM-MIX EXPOSURE")
    print("     no young worker is classified; age carries the within variation")
    print("=" * 70)
    print(mc.mem_line("  "))
    h47 = _h47()
    key, scores = h47.load_key(), h47.load_scores()
    conn = None

    counts = {}
    for y in (2019, 2020, 2021):
        cf = CACHE / f"edu_hr_weights_{y}.parquet"
        w = mc.read_cache(cf)
        if w is None:
            conn = conn or mc.connect()
            w = h47.pull_weights(y, conn)
            w.to_parquet(cf, index=False)
        counts[y] = w
        print(f"  weights {y}: {len(w):,} cells")
    book = h47.ScoreBook(counts, key, scores)
    designs = {k: dict(h47.DESIGNS[k]) for k in ("OL_daioe", "entrant")}
    for nm, sp in designs.items():
        book.build(nm, sp)

    frames = {}
    for y in YEARS:
        cf = CACHE / f"edu_hr_{y}.parquet"
        f = mc.read_cache(cf)
        if f is None:
            conn = conn or mc.connect()
            print(f"  {y}: no 47h cache, pulling")
            f = h47.pull_year(y, conn, True)
            f.to_parquet(cf, index=False)
        else:
            print(f"  {y}: cached ({len(f):,} cells)")
        frames[y] = f

    rows, diag = [], []
    for nm, sp in designs.items():
        for T in TRUNCATIONS:
            for arm in ("true", "asof"):
                expo, cuts = incumbent_exposure(frames[BASE_YEAR], book, nm, sp, arm, T)
                if expo.empty:
                    print(f"  {nm} {arm} T{T}: no firms pass the incumbent floor")
                    continue
                diag.append(expo.assign(design=nm, arm=arm, trunc=T))
                for young in YOUNG_BANDS:
                    bal = build_panel(frames, expo, young)
                    r = fit(bal, f"tj_{nm}_{arm}_{T}_{young.replace('-', '_')}")
                    r.update(design=nm, arm=arm, trunc=T, young_band=young,
                             n_firms=int(expo["employer_id"].nunique()))
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(OUT / "triple_estimates.csv", index=False)
                    print(f"  {nm:<10} {arm:<4} T{T} young={young} gamma3 "
                          f"{r['gamma3']:+.4f} (SE {r['se']:.4f}) n {r['n_obs']:,} "
                          f"{r['status']}")
                    del bal
                    gc.collect()
    est = pd.DataFrame(rows)
    if diag:
        d = pd.concat(diag, ignore_index=True)
        q = (d.groupby(["design", "arm", "trunc", "fq"], observed=True)
             .agg(firms=("employer_id", "nunique"), incumbents=("n", "sum")).reset_index())
        mc.enforce_min_cell(q, count_col="firms").to_csv(
            OUT / "triple_quartile_sizes.csv", index=False)
        # how much does staleness move the 2019 incumbent classifier at all?
        for nm in designs:
            for T in TRUNCATIONS:
                a = d[(d.design == nm) & (d.trunc == T) & (d.arm == "true")]
                b = d[(d.design == nm) & (d.trunc == T) & (d.arm == "asof")]
                j = a.merge(b, on="employer_id", suffixes=("_t", "_a"))
                if len(j):
                    print(f"  classifier stability {nm} T{T}: "
                          f"{(j.fq_t == j.fq_a).mean():.1%} of firms keep their quartile, "
                          f"mean |relative mix shift| "
                          f"{float(np.mean(np.abs(j.mix_a - j.mix_t) / j.mix_t.abs())):.3%}")

    lines = ["WITHIN-EMPLOYER TRIPLE DIFFERENCE, FIRM-MIX EXPOSURE", "=" * 60,
             "Exposure: worker-weighted education mix of the firm's INCUMBENTS",
             f"(aged 31+) in {BASE_YEAR}, quartiles fixed, held for every year.",
             "Identification: young versus older workers WITHIN one employer in",
             "one month. Employer x month, employer x age and month x age are all",
             "absorbed, so no firm-time shock and no economy-wide age trend can",
             "enter. No young worker is ever classified; age comes from birth year.",
             "",
             "Estimand: the young-old gap inside exposed employers. A shock that",
             "hit every age equally within exposed firms would NOT appear here.",
             "",
             "Occupation design (45), 22-25: "
             + "  ".join(f"T{T} {v:+.4f}" for T, v in OCC_ARTEFACT.items()),
             "Worker-level education (47b), 22-25: T2021 -0.3596  T2022 -0.2941",
             ""]
    for nm in designs:
        lines.append(f"{nm}:")
        for young in YOUNG_BANDS:
            for T in TRUNCATIONS:
                s = est[(est.design == nm) & (est.young_band == young) & (est.trunc == T)]
                if len(s) == 2:
                    tr = float(s[s.arm == "true"]["gamma3"].iloc[0])
                    af = float(s[s.arm == "asof"]["gamma3"].iloc[0])
                    lines.append(f"  young={young} T{T}  true {tr:+.4f}  as-of {af:+.4f}"
                                 f"  ARTEFACT {af - tr:+.4f}")
        lines.append("")
    lines += ["READ RULE (pre-committed, as 47b and 47i):",
              "  |artefact| < 0.05 at both truncations -> carries register evidence.",
              "  0.05 to half the occupation artefact (0.153 / 0.081) -> usable with",
              "    the artefact stated beside every estimate.",
              "  at or above half -> closed.",
              "",
              "NOT DONE HERE: the event study by half-year, the industry-reweighted",
              "contrast, and hires rather than employment as the outcome. Additions,",
              "not corrections.",
              f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "47j_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
47k_settled_sample.py -- R1 and R2 of the analysis plan: keep the paper's
question by restricting to young workers whose education record is
CORRECT rather than stale.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_47k/. Reads 47h's cached year frames, so it needs
  no SQL of its own once 47h has pulled (about 2 hours of fits).
  Local end-to-end test: revision/local/test_47k_synthetic.py
======================================================================

THE QUESTION THIS PROTECTS (analysis plan, section 1). Within the same
employer, in the same month, do young workers who are more exposed to AI
fare worse than OTHER YOUNG WORKERS? Scripts 47i and 47j gave that up:
one compares firms, the other compares young to old. This one keeps it.

THE IDEA. The artefact 47b measured is produced by workers whose record is
out of date. Rather than find a cleverer classifier, remove those workers.
A 24-year-old in 2025 who graduated in 2022 is recorded correctly in the
2023 vintage: for them the register is not stale, it is right.

THE TRAP, AND WHY THIS SCRIPT HAS TWO RULES. "The record shows an old
completion" does NOT mean the record is current. Someone finishing a
master's in 2024 still carries their 2018 bachelor's or their upper
secondary record, which looks perfectly settled and is exactly wrong. The
people who are mid-completion are, however, mostly the people who are
ENROLLED, and the higher-education register sees that. So:

  RULE "feasible"  the record shows a completion more than two years old
                   (experience band not 0-2 and not missing) AND the
                   person has no registration in the enrolment register
                   within three academic years of the truncation. Both
                   conditions are observable in 2024-25, so this rule can
                   actually be applied to the paper's window.

  RULE "oracle"    the truncated record is IDENTICAL to the true record.
                   Not computable in 2024-25 -- it uses the truth -- and
                   included precisely for that reason: it is the ceiling,
                   the artefact that remains when settledness is detected
                   perfectly. The gap between "feasible" and "oracle" is
                   the cost of not being able to see who is mid-degree.

  RULE "all"       no restriction: 47b's sample, the baseline both are
                   measured against.

THE ONE THING THAT WOULD INVALIDATE THE TEST. If the restriction selected
different people in the two arms, the backtest would compare different
samples and the "artefact" would mix classification error with sample
change. So the filter is computed ONCE, from the as-of side (the side that
is feasible), and the SAME rows are used in both arms. Only the
classification differs between arms. This is asserted in the local test.

WHAT IT COSTS, and it is reported rather than argued: the newest entrants
are excluded, and they are plausibly the most affected group, so the
estimate is conservative for the phenomenon. Retention is reported by age
and year, and the included and excluded are compared on observables.

READ RULE, PRE-COMMITTED, as 47b: artefact below 0.05 in absolute value at
both truncations -> this can carry the paper's register evidence; 0.05 to
half the occupation artefact (0.153 / 0.081) -> usable with the artefact
beside every estimate; at or above half -> closed.
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
OUT = HERE / "output_47k"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YEARS = [2019, 2020, 2021, 2022, 2023]
TRUNCATIONS = (2021, 2022)
AGES = ["22-25", "50+"]                 # 50+ is the placebo
AGES_EXTRA = ["26-30"]                  # added for whichever rule passes
RULES = ["all", "feasible", "oracle"]
SETTLED_BANDS = ["3-5", "6-10", "11-20", "21+"]   # NOT 0-2, NOT na
OCC_ARTEFACT = {2021: -0.3068, 2022: -0.1627}
STEP1_MIN_CUMULATIVE = 5


def cache_name(stem: str) -> Path:
    """
    47h writes its pulls to cache/edu_hr_*.parquet. If 47h is running in
    another console, reading those half-written files is a corruption risk,
    and mona_common's footer check would DELETE the file it finds truncated
    -- destroying the other console's work. So: use 47h's caches only when
    47h has finished, which its summary file proves, and otherwise keep a
    private copy under a "_k" suffix.
    """
    if (HERE / "output_47h" / "47h_summary.txt").exists():
        return CACHE / f"{stem}.parquet"
    return CACHE / f"{stem}_k.parquet"


def opt(label: str, fn, *a, **kw):
    """
    Stata's `capture noisily`. Runs something INESSENTIAL: a diagnostic, a
    side table, a print. A failure is reported loudly and the run continues.
    Never wrap an estimate or a primary export in this.
    """
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}): {str(ex)[:200]}")
        print(f"  [optional] continuing; this does not affect the estimates")
        return None


def _h47():
    import importlib.util
    # locate 47h relative to THIS FILE, not to the module-level HERE: HERE is
    # a mutable global and a caller that reassigns it must not break the import
    _here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("h47", _here / "47h_edu_horserace.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def settled_mask(frame: pd.DataFrame, T: int, rule: str) -> pd.Series:
    """
    Which rows count as settled. Computed from the AS-OF columns for the
    feasible rule, so the same rows are selected whichever arm is then
    estimated: the backtest must vary the classification, never the sample.
    """
    s = f"{T % 100}"
    if rule == "all":
        return pd.Series(True, index=frame.index)
    if rule == "feasible":
        band_ok = frame[f"expb_{s}"].astype("string").isin(SETTLED_BANDS)
        not_enrolled = frame[f"enr_{s}"].isna()
        return (band_ok & not_enrolled).fillna(False)
    if rule == "oracle":
        same = ((frame[f"niva_{s}"].astype("string")
                 == frame["niva_t"].astype("string"))
                & (frame[f"inr_{s}"].astype("string")
                   == frame["inr_t"].astype("string")))
        return same.fillna(False)
    raise ValueError(rule)


def collapse(frame: pd.DataFrame, mask: pd.Series, book, name: str, spec: dict,
             arm: str, T: int, ages: list) -> pd.DataFrame:
    """Assign the quartile under one arm and collapse to the paper's cell."""
    f = frame[mask & frame["age_group"].astype(str).isin(ages)]
    if f.empty:
        return pd.DataFrame(columns=["employer_id", "year_month",
                                     "exposure_quartile", "age_group", "n_emp"])
    if arm == "true":
        cols, enr = ["niva_t", "inr_t", "expb_t"], None
    else:
        cols = [f"niva_{T % 100}", f"inr_{T % 100}", f"expb_{T % 100}"]
        enr = f"enr_{T % 100}" if spec.get("enrol") else None
    keycols = cols + ([enr] if enr else []) + ["age_group"]
    combos = f[keycols].drop_duplicates().reset_index(drop=True)
    s = book.score_frame(name, spec, combos[cols[0]], combos[cols[1]],
                         combos[cols[2]], combos[enr] if enr else None,
                         combos["age_group"])
    h47 = _H47
    combos["_q"] = h47.to_quartile(s, book.cuts[name])
    g = f.merge(combos, on=keycols, how="left")
    g = g[g["_q"] > 0]
    return (g.groupby(["employer_id", "year_month", "_q", "age_group"],
                      observed=True)["n_emp"].sum().reset_index()
            .rename(columns={"_q": "exposure_quartile"}))


def estimate(coll: pd.DataFrame, age: str, tag: str) -> dict:
    row = {"gamma2": np.nan, "se": np.nan, "n_obs": 0, "status": "empty"}
    if coll.empty:
        return row
    sub = coll[coll["age_group"].astype(str) == age].copy()
    if sub.empty:
        return row
    sub["year_month"] = sub["year_month"].astype(str)
    sub["exposure_quartile"] = sub["exposure_quartile"].astype(int)
    cum = sub.groupby("employer_id")["n_emp"].sum()
    sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
    if sub.empty:
        return row
    months = sorted(coll["year_month"].astype(str).unique())
    bal = _H47.fast_balance(sub, months)
    if bal.empty:
        return row
    row["n_obs"] = len(bal)
    r = mc.run_fepois(mc.add_treatment(bal), OUT, tag=tag)
    row["status"] = "no_output"
    if not r.empty and (r["term"] == "post_gpt_x_high").any():
        g = r[r["term"] == "post_gpt_x_high"].iloc[0]
        row.update(gamma2=float(g["coef"]), se=float(g["se"]),
                   status=str(g.get("status", "ok")))
    del bal
    gc.collect()
    return row


def verdict(a21, a22) -> str:
    a = [abs(x) for x in (a21, a22)]
    if any(np.isnan(a)):
        return "PENDING"
    if all(x < 0.05 for x in a):
        return "CLEAN"
    if a[0] < abs(OCC_ARTEFACT[2021]) / 2 and a[1] < abs(OCC_ARTEFACT[2022]) / 2:
        return "USABLE WITH CAVEAT"
    return "CLOSED"


_H47 = None


def main():
    global _H47
    mc.Tee(OUT / "47k_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("47k: THE SETTLED-EDUCATION SUBSAMPLE (R1 + R2)")
    print("     keeps the paper's question: exposed young vs other young,")
    print("     within employer, monthly")
    print("=" * 70)
    print(mc.mem_line("  "))
    _H47 = h47 = _h47()
    key, scores = h47.load_key(), h47.load_scores()

    conn = None
    counts = {}
    for y in (2019, 2020, 2021):
        cf = cache_name(f"edu_hr_weights_{y}")
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
        cf = cache_name(f"edu_hr_{y}")
        f = mc.read_cache(cf)
        if f is None:
            conn = conn or mc.connect()
            print(f"  {y}: no 47h cache, pulling")
            f = h47.pull_year(y, conn, True)
            f.to_parquet(cf, index=False)
        else:
            print(f"  {y}: cached ({len(f):,} cells)")
        frames[y] = f

    # ---- retention: who does each rule keep, by age and year ----
    ret = []
    for T in TRUNCATIONS:
        for rule in RULES:
            for y in YEARS:
                f = frames[y]
                m = settled_mask(f, T, rule)
                g = (f.assign(_k=m).groupby("age_group", observed=True)
                     .apply(lambda d: pd.Series({
                         "n_total": int(d["n_emp"].sum()),
                         "n_kept": int(d.loc[d["_k"], "n_emp"].sum())}),
                         include_groups=False).reset_index())
                g["trunc"], g["rule"], g["year"] = T, rule, y
                ret.append(g)
    retention = pd.concat(ret, ignore_index=True)
    retention["kept_share"] = retention["n_kept"] / retention["n_total"].clip(lower=1)
    opt("retention export", lambda: mc.enforce_min_cell(
        retention, count_col="n_kept").to_csv(OUT / "retention.csv", index=False))
    r22 = retention[(retention.age_group == "22-25") & (retention.rule != "all")]
    for T in TRUNCATIONS:
        for rule in ("feasible", "oracle"):
            s = r22[(r22.trunc == T) & (r22.rule == rule)]
            if len(s):
                print(f"  retention 22-25, T{T}, {rule}: "
                      + ", ".join(f"{int(r.year)} {r.kept_share:.0%}"
                                  for r in s.itertuples()))

    # ---- estimates ----
    rows = []
    def run(nm, sp, rule, T, ages, tier):
        masks = {y: settled_mask(frames[y], T, rule) for y in YEARS}
        out = {}
        for arm in ("true", "asof"):
            coll = pd.concat([collapse(frames[y], masks[y], book, nm, sp, arm, T, ages)
                              for y in YEARS], ignore_index=True)
            for age in ages:
                r = estimate(coll, age, f"k_{nm}_{rule}_{arm}_{T}_"
                                        f"{age.replace('-', '_').replace('+', 'p')}")
                out[(arm, age)] = r["gamma2"]
                rows.append(dict(design=nm, rule=rule, trunc=T, arm=arm,
                                 age_group=age, tier=tier, **r))
                pd.DataFrame(rows).to_csv(OUT / "settled_estimates.csv", index=False)
                print(f"  [{tier}] {nm:<10} {rule:<9} {arm:<4} T{T} {age:<5} "
                      f"gamma2 {r['gamma2']:+.4f} (SE {r['se']:.4f}) "
                      f"n {r['n_obs']:,} {r['status']}")
            del coll
            gc.collect()
        return {age: out[("asof", age)] - out[("true", age)] for age in ages}

    print("\nTIER A: the three sampling rules, reference scorer")
    art = {}
    for rule in RULES:
        for T in TRUNCATIONS:
            a = run("OL_daioe", designs["OL_daioe"], rule, T, AGES, "A")
            for age, v in a.items():
                art[("OL_daioe", rule, T, age)] = v
            print(f"  ==> OL_daioe {rule:<9} T{T}: 22-25 ARTEFACT "
                  f"{a['22-25']:+.4f}   50+ {a['50+']:+.4f}")

    print("\nTIER B: entrant weights on the feasible rule")
    for T in TRUNCATIONS:
        a = run("entrant", designs["entrant"], "feasible", T, AGES, "B")
        for age, v in a.items():
            art[("entrant", "feasible", T, age)] = v
        print(f"  ==> entrant feasible T{T}: 22-25 ARTEFACT {a['22-25']:+.4f}   "
              f"50+ {a['50+']:+.4f}")

    print("\nTIER C: 26-30 for whichever rule passes at 22-25")
    for nm in ("OL_daioe", "entrant"):
        for rule in RULES:
            v = verdict(art.get((nm, rule, 2021, "22-25"), np.nan),
                        art.get((nm, rule, 2022, "22-25"), np.nan))
            if v in ("CLEAN", "USABLE WITH CAVEAT"):
                for T in TRUNCATIONS:
                    a = run(nm, designs[nm], rule, T, AGES_EXTRA, "C")
                    for age, x in a.items():
                        art[(nm, rule, T, age)] = x

    # ---- summary ----
    est = pd.DataFrame(rows)
    lines = ["THE SETTLED-EDUCATION SUBSAMPLE", "=" * 60,
             "Question kept: exposed young vs OTHER YOUNG workers, within",
             "employer, monthly. The sampling rule is computed from the as-of",
             "side and applied to BOTH arms, so only the classification differs.",
             "",
             "Occupation design (45), 22-25: "
             + "  ".join(f"T{T} {v:+.4f}" for T, v in OCC_ARTEFACT.items()),
             "Unrestricted education (47b), 22-25: T2021 -0.3596  T2022 -0.2941",
             "",
             "rule      = all       47b's sample, the baseline",
             "            feasible  record older than 2 years AND not enrolled;",
             "                      applicable to 2024-25",
             "            oracle    record identical to the truth; NOT applicable,",
             "                      included as the ceiling",
             ""]
    for nm in ("OL_daioe", "entrant"):
        for rule in RULES:
            a21 = art.get((nm, rule, 2021, "22-25"), np.nan)
            a22 = art.get((nm, rule, 2022, "22-25"), np.nan)
            if np.isnan(a21) and np.isnan(a22):
                continue
            p21 = art.get((nm, rule, 2021, "50+"), np.nan)
            tr = est[(est.design == nm) & (est.rule == rule) & (est.arm == "true")
                     & (est.age_group == "22-25") & (est.trunc == 2021)]
            t22 = float(tr["gamma2"].iloc[0]) if len(tr) else np.nan
            lines.append(f"{nm:<10} {rule:<9} artefact T2021 {a21:+.4f}  "
                         f"T2022 {a22:+.4f}  50+ {p21:+.4f}  "
                         f"true 22-25 {t22:+.4f}   {verdict(a21, a22)}")
    lines += ["",
              "The gap between 'feasible' and 'oracle' is the cost of not being",
              "able to see who is mid-degree. If 'oracle' is clean and 'feasible'",
              "is not, the enrolment register is the binding constraint and the",
              "fix is a newer delivery of it, not a better classifier.",
              "",
              "Retention (share of young person-months kept) is in retention.csv.",
              "The excluded are the newest entrants, so a negative estimate here is",
              "CONSERVATIVE for the phenomenon if they are more affected.",
              "",
              "READ RULE (pre-committed, as 47b): clean below 0.05 at both",
              "truncations; usable below half the occupation artefact; else closed.",
              f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "47k_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

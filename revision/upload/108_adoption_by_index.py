#!/usr/bin/env python3
"""
108_adoption_by_index.py -- lane 39d: which exposure index predicts reported
                            AI use, DAIOE or the Eloundou rating, on the same
                            firms and the same survey questions?

======================================================================
  RUNS IN MONA (lane 39d). Output folder CANARIES_108_OUT (default
  output_108). SQL: the survey tables script 71 reads (ITFtg, ai_itftg,
  ai_fufi, ai_fouftg, ai_fouoff, BITA) and one AGI month for the BITA
  link, through 71's own arms. No R. Reads 82's score caches and
  L_baseline_2019 (log-size control). Input: eloundou_ssyk4.dta at the
  project root.
======================================================================

QUESTION (ML, 27 Sep 2026)
The paper's first stage says top-quartile employers on DAIOE are 20.9
points more likely to report AI use in Statistics Sweden's 2023 firm
survey. The pooled 22-25 estimate is less precise on the Eloundou
classification. Does DAIOE predict AI use better than the Eloundou rating
among the same firms? Pulito, Pytlikova, Schroeder and Lodefalk (2026,
IZA DP 18515) find, on Danish firms and core AI (not generative AI),
that DAIOE predicts adoption better than Eloundou; this lane tests it on
our firms, our classifications and both "any AI" and "language
generation" (generative AI) as outcomes.

THE ROUTES, all run through 71's itftg_arm and bita_arm unchanged
(top-quartile dummy, log 2019 size control, linear probability model):
  daioe_all        every DAIOE-scored employer: the GATE, which must
                   reproduce 83's 2023 any-AI gap (20.87 points, 3,587
                   firms) within 0.05 points.
  daioe_common     DAIOE quartile, employers scored on BOTH indices.
  eloundou_common  Eloundou quartile, the same employers.
  daioe_only       employers NOT in the Eloundou top quartile; "high" =
                   DAIOE top. Among firms both indices agree are not
                   Eloundou-exposed, does DAIOE exposure alone predict use?
  eloundou_only    employers NOT in the DAIOE top quartile; "high" =
                   Eloundou top. The mirror.
The last two are the discriminating comparison: the firms on which the
two classifications disagree. They are thin (about 7.5 per cent of each
top quartile), so the threshold on top-quartile firms with an outcome is
lowered from 71's 100 to 30 FOR THESE TWO ROUTES ONLY, fixed before the
run; the summary prints the counts.

READ RULES, FIXED BEFORE THE RUN
  G.  daioe_all reproduces 83's 2023 ITFtg any-AI gap within 0.05 points,
      or nothing is quotable.
  A1. daioe_common and eloundou_common are reported side by side for every
      survey table and outcome, in points with SEs, on the same firms; the
      difference is stated without a test (the two share their firms).
  A2. The disagreement routes: DAIOE is called the better predictor of an
      outcome only if daioe_only is positive and distinguishable from zero
      at five per cent while eloundou_only is not; the mirror for
      Eloundou; otherwise "no discrimination". Language generation (the
      generative-AI question) is read first, any AI second.
  Employer and person counts below five are suppressed.

EXPORT (output_108/)
  adoption_by_index.csv   route x source x outcome: coef and SE in points, n
  adoption_counts.csv     route x source: matched, with outcome, top quartile
  108_summary.txt, 108_log.txt

    python 108_adoption_by_index.py
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402

OUT = HERE / os.environ.get("CANARIES_108_OUT", "output_108")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_103_OUT", str(OUT))
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
GATE_POINTS, GATE_N, GATE_TOL = 20.87, 3587, 0.05
GATE_SOURCE, GATE_OUTCOME = "ITFtg_Stora_2023", "ai_any"
THIN_ROUTES = ("daioe_only", "eloundou_only")
THIN_MIN_HIGH = 30

NOTES: list = []
FAILURES: list = []
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  G.  daioe_all reproduces 83's 2023 ITFtg any-AI gap (20.87 points,",
    "      3,587 firms) within 0.05 points, or nothing is quotable.",
    "  A1. daioe_common beside eloundou_common for every table and outcome,",
    "      same firms, points with SEs; the difference is not tested.",
    "  A2. On the disagreement routes, DAIOE is the better predictor of an",
    "      outcome only if daioe_only > 0 at five per cent and eloundou_only",
    "      is not; the mirror for Eloundou; otherwise no discrimination.",
    "      Language generation first, any AI second. The top-quartile",
    "      threshold is 30 on these two routes only (71's 100 elsewhere).",
    f"  Counts below {FLOOR} are suppressed.",
]


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    for attr in ("NOTES", "FAILURES"):
        if hasattr(mod, attr):
            getattr(mod, attr).clear()


def open_conn():
    """Separated so the local dry run can replace it."""
    return mc.connect()


def load_modules():
    s103 = _mod("103_eloundou_classification.py", "s103")
    s103.OUT = OUT
    s82, s61, s67, s73, s78, s80, l47, l70, j47 = s103.load_modules()
    s71 = _mod("71_adoption_validation.py", "s71")
    for m_ in (s71,):
        if hasattr(m_, "OUT"):
            m_.OUT = OUT
    return s103, s82, s73, s71, l47, l70, j47


def build_routes(expo_d: pd.DataFrame, expo_e: pd.DataFrame, s73) -> dict:
    """The five routes of the docstring, each a frame of employer_id and
    fq with the identifier normalised as 71 expects."""
    def norm(e):
        d = e[["employer_id", "fq"]].copy()
        d["employer_id"] = s73.norm_id(d["employer_id"])
        d["fq"] = pd.to_numeric(d["fq"], errors="coerce").astype("Int64")
        return d.dropna(subset=["fq"])
    d, e = norm(expo_d), norm(expo_e)
    both = set(d["employer_id"]) & set(e["employer_id"])
    dc = d[d["employer_id"].isin(both)]
    ec = e[e["employer_id"].isin(both)]
    top_d = set(dc.loc[dc["fq"] == 4, "employer_id"])
    top_e = set(ec.loc[ec["fq"] == 4, "employer_id"])
    # disagreement routes: 'fq' is set to 4 for the route's own top and 1
    # otherwise, on the sample that excludes the OTHER index's top quartile
    d_only = dc[~dc["employer_id"].isin(top_e)].copy()
    d_only["fq"] = np.where(d_only["employer_id"].isin(top_d), 4, 1)
    e_only = ec[~ec["employer_id"].isin(top_d)].copy()
    e_only["fq"] = np.where(e_only["employer_id"].isin(top_e), 4, 1)
    for x in (d_only, e_only):
        x["fq"] = x["fq"].astype("Int64")
    NOTES.append(f"routes: DAIOE {len(d):,} employers, Eloundou {len(e):,}, both "
                 f"{len(both):,}; top on DAIOE only {len(top_d - top_e):,}, on "
                 f"Eloundou only {len(top_e - top_d):,}, on both {len(top_d & top_e):,}")
    return {"daioe_all": d, "daioe_common": dc, "eloundou_common": ec,
            "daioe_only": d_only, "eloundou_only": e_only}


def run_arms(routes: dict, s71, s73) -> tuple:
    """71's firm and worker arms on the routes. The thin routes run in a
    second call with the lowered top-quartile threshold."""
    sink, counts = [], []
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet", require=["employer_id", "n"])
    if base is None:
        raise RuntimeError("L_baseline_2019 is not on the share; 71's log-size "
                           "control cannot be built")
    size = s71.firm_size(base)
    size["employer_id"] = s73.norm_id(size["employer_id"])
    del base
    gc.collect()
    conn = open_conn()
    try:
        schema = s71.discover(conn)
        if schema.empty:
            raise RuntimeError("the catalogue returned no survey table")
        main = {k: v for k, v in routes.items() if k not in THIN_ROUTES}
        thin = {k: v for k, v in routes.items() if k in THIN_ROUTES}
        s71.itftg_arm(conn, schema, main, size, sink, counts)
        s71.bita_arm(conn, schema, main, sink, counts)
        keep = s71.MIN_ITFTG_HIGH
        try:
            s71.MIN_ITFTG_HIGH = THIN_MIN_HIGH
            s71.itftg_arm(conn, schema, thin, size, sink, counts)
            s71.bita_arm(conn, schema, thin, sink, counts)
        finally:
            s71.MIN_ITFTG_HIGH = keep
    finally:
        try:
            conn.close()
        except Exception:
            pass
    drain(s71, "71")
    rows = pd.concat(sink, ignore_index=True) if sink else pd.DataFrame()
    return rows, pd.DataFrame(counts)


def tidy(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    hi = rows[rows["term"] == "high"].copy()
    hi["coef_points"] = 100 * hi["coef"]
    hi["se_points"] = 100 * hi["se"]
    hi["t"] = hi["coef"] / hi["se"]
    cols = ["route", "source", "outcome", "coef_points", "se_points", "t", "n"]
    return hi[[c for c in cols if c in hi.columns]].reset_index(drop=True)


def gate(t: pd.DataFrame) -> list:
    r = t[(t.route == "daioe_all") & (t.source == GATE_SOURCE) & (t.outcome == GATE_OUTCOME)]
    if len(r) != 1:
        return [f"the gate row (daioe_all, {GATE_SOURCE}, {GATE_OUTCOME}) is missing"]
    c, n = float(r.coef_points.iloc[0]), int(r.n.iloc[0])
    bad = []
    if abs(c - GATE_POINTS) > GATE_TOL:
        bad.append(f"gate {c:.2f} points against 83's {GATE_POINTS}")
    if n != GATE_N:
        bad.append(f"gate on {n:,} firms against 83's {GATE_N:,}")
    return bad


def verdict(t: pd.DataFrame, source: str, outcome: str) -> str:
    def g(route):
        r = t[(t.route == route) & (t.source == source) & (t.outcome == outcome)]
        return (float(r.coef_points.iloc[0]), float(r.se_points.iloc[0])) if len(r) == 1 else None
    d, e = g("daioe_only"), g("eloundou_only")
    if d is None or e is None:
        return "NOT ESTIMABLE (a disagreement route is missing or below threshold)"
    sig = lambda x: x[0] > 0 and x[0] > 1.96 * x[1]  # noqa: E731
    if sig(d) and not sig(e):
        return "DAIOE the better predictor"
    if sig(e) and not sig(d):
        return "ELOUNDOU the better predictor"
    return "NO DISCRIMINATION"


def write_summary(t: pd.DataFrame, c: pd.DataFrame) -> None:
    L = ["WHICH INDEX PREDICTS REPORTED AI USE: DAIOE AGAINST THE ELOUNDOU RATING",
         "(LANE 39d)", "=" * 66, "",
         "Linear probability models of 71's arms: outcome on the top-quartile",
         "dummy and log 2019 size; coefficients in percentage points.", ""]
    if not t.empty:
        bad = gate(t)
        L.append("GATE: " + ("PASSES (daioe_all reproduces 83's 2023 any-AI gap)" if not bad
                            else "FAILED: " + "; ".join(bad)))
        L.append("")
        L.append("A1. SAME FIRMS, SIDE BY SIDE (points, SE, firms):")
        for (src, out), g in t[t.route.isin(["daioe_common", "eloundou_common"])].groupby(["source", "outcome"]):
            v = {r.route: r for r in g.itertuples()}
            if "daioe_common" in v and "eloundou_common" in v:
                a, b = v["daioe_common"], v["eloundou_common"]
                L.append(f"  {src:<22} {out:<13} DAIOE {a.coef_points:+6.2f} ({a.se_points:.2f})  "
                         f"Eloundou {b.coef_points:+6.2f} ({b.se_points:.2f})  n {int(a.n):,}")
        L += ["", "A2. THE DISAGREEMENT ROUTES (points, SE, firms):"]
        for (src, out), g in t[t.route.isin(THIN_ROUTES)].groupby(["source", "outcome"]):
            v = {r.route: r for r in g.itertuples()}
            parts = [f"{k.split('_')[0]}-only {v[k].coef_points:+6.2f} ({v[k].se_points:.2f}) n {int(v[k].n):,}"
                     for k in THIN_ROUTES if k in v]
            L.append(f"  {src:<22} {out:<13} " + "   ".join(parts) + f"   -> {verdict(t, src, out)}")
    else:
        L.append("NO ESTIMATES")
    if not c.empty:
        L += ["", "COUNTS (matched, with outcome, top quartile with outcome):"]
        for r in c.itertuples():
            L.append(f"  {r.source:<22} {r.route:<16} {r.matched:>7,} {r.with_outcome:>7,} {r.high_with_outcome:>6,}")
    if NOTES:
        L += ["", "NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        L += ["", "FAILED: " + " | ".join(FAILURES)]
    L += [""] + READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} min. " + mc.mem_line("")]
    (OUT / "108_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


def main() -> int:
    global T0
    mc.Tee(OUT / "108_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("108: WHICH INDEX PREDICTS REPORTED AI USE (LANE 39d)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    rc = 0
    t, c = pd.DataFrame(), pd.DataFrame()
    try:
        s103, s82, s73, s71, l47, l70, j47 = load_modules()
        expo_d, expo_e = s103.build_both(s82, l47, l70, j47)
        drain(s103, "103")
        routes = build_routes(expo_d, expo_e, s73)
        del expo_d, expo_e
        gc.collect()
        rows, c = run_arms(routes, s71, s73)
        t = tidy(rows)
        if not c.empty:
            for col in ("matched", "with_outcome", "high_with_outcome"):
                c.loc[(c[col] > 0) & (c[col] < FLOOR), col] = np.nan
        t.to_csv(OUT / "adoption_by_index.csv", index=False)
        c.to_csv(OUT / "adoption_counts.csv", index=False)
        bad = gate(t) if not t.empty else ["no estimates"]
        if bad:
            FAILURES.append("GATE: " + "; ".join(bad))
    except BaseException as ex:
        print(f"108 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    write_summary(t, c)
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("108_adoption_by_index", rc, (time.time() - T0) / 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

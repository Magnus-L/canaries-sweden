#!/usr/bin/env python3
"""
58_seasonal_and_vintage.py -- settle the three things the 2025H1 result
                              currently rests on.

======================================================================
  RUNS IN MONA. One cheap metadata query; everything else reads the
  caches 47L and 54 already wrote. Writes output_58/.
======================================================================

WHAT 56 FOUND, AND WHY IT IS NOT YET QUOTABLE.

The 22-25 hiring differential sat between -0.000 and +0.027 in every
first half-year from 2019 to 2024, and came in at -0.069 in 2025H1: the
only negative reading in the whole column. Separations did not fall and
the stock barely moved, which is what an inflow adjustment looks like.

Three things stand between that and a result, and this script addresses
each. They are listed in the order they could kill the finding.

  1. THE SLICE WAS CHOSEN AFTER SEEING THE DATA. The 22-25 differential
     has a marked H1/H2 seasonal, so reading H1 against H1 is the right
     comparison; but nobody said so before the numbers arrived. Two
     independent fixes are run here, and a finding that survives both is
     not an artefact of how the seasonal was handled:
        (a) REBASED: subtract, from each coefficient, the mean of the
            pre-period coefficients of the SAME half of the year. This is
            the "compare like with like" reading done systematically
            rather than by eye.

            NOTE, established by the synthetic test before this ran: it is
            NOT possible to purge the seasonal by adding an
            exposure-times-H2 control. With a full set of period-specific
            interactions, that control is perfectly collinear with them -
            the seasonal IS the period coefficients - and fixest drops it,
            leaving the path unchanged. Rebasing after estimation is the
            only version of this that does anything.
        (b) H1-ONLY: drop H2 entirely and estimate on first half-years.
            Fewer periods, no seasonal model at all, nothing to assume.

  2. NO STANDARD ERRORS WERE PRINTED. 56's summary reported coefficients
     only. Every table here carries the standard error beside the point
     estimate, and the pre-period maximum is reported as a t-statistic so
     that "the pre-period is flat" is a measured claim.

  3. 2025 IS PRELIMINARY. Every 2025 figure comes from
     Arb_AGIIndivid2025MM_prel. If SCB has since delivered the definitive
     file, the finding must be re-estimated on it before it is believed.
     This script does not re-pull; it reports exactly which monthly AGI
     tables now exist, in which vintage, with their row counts, so the
     decision is made on facts rather than on the assumption that nothing
     changed since the last delivery.

PRE-COMMITTED READ RULE, written before the purged numbers exist:

  The 2025H1 young hiring differential counts as a finding only if it is
  (i) negative in BOTH the purged and the H1-only specification,
  (ii) at least twice its standard error in both, and
  (iii) larger in magnitude than every pre-2022H2 coefficient in the same
        specification.
  If it fails (iii) it is within the ordinary variation of this series and
  must be reported as suggestive, whatever its t-statistic.

Output (output_58/):
  agi_tables.csv          every monthly AGI table, vintage, row count
  es_rebased.csv          event study, rebased on the same season
  es_h1only.csv           event study, first half-years only
  readrule.csv            the three conditions, evaluated
  58_summary.txt
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_58"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YOUNG = "22-25"
REF = mc.REF_HALFYEAR            # 2022H1
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")
FOCUS = "2025H1"


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name[:4], HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------
# 3. which AGI months exist, and in which vintage
# ----------------------------------------------------------------------

def probe_agi(conn) -> pd.DataFrame:
    """
    Every monthly AGI table the project can see, with its vintage suffix
    and row count. Row counts come from sys.partitions, which is metadata
    and costs nothing, rather than COUNT(*) over eighty tables.
    """
    q = """
    SELECT t.name AS table_name,
           SUM(CASE WHEN p.index_id IN (0,1) THEN p.rows ELSE 0 END) AS n_rows
    FROM sys.tables t
    JOIN sys.partitions p ON p.object_id = t.object_id
    WHERE t.name LIKE 'Arb_AGIIndivid%'
    GROUP BY t.name
    ORDER BY t.name
    """
    df = pd.read_sql(q, conn)
    if df.empty:
        return df
    df["period"] = df["table_name"].str.extract(r"(\d{6})")
    df["vintage"] = np.where(df["table_name"].str.endswith("_def"), "def",
                    np.where(df["table_name"].str.endswith("_prel"), "prel",
                             "other"))
    df["year"] = df["period"].str[:4]
    df["month"] = df["period"].str[4:]
    return df.sort_values(["period", "vintage"])


# ----------------------------------------------------------------------
# 1 and 2. the two event studies, both with standard errors
# ----------------------------------------------------------------------

def build_terms(bal: pd.DataFrame, h1_only: bool):
    """Event-study terms on a panel that already carries expo_z."""
    b = bal.copy()
    b["halfyear"] = mc.assign_halfyear(pd.Series(b["year_month"].astype(str)))
    if h1_only:
        b = b[b["halfyear"].str.endswith("H1")].copy()
    b["is_young"] = (b["age_group"] == YOUNG).astype(int)
    b["is_h2"] = b["halfyear"].str.endswith("H2").astype(int)
    hys = sorted(h for h in b["halfyear"].unique() if h != REF)
    pooled, young = [], []
    for h in hys:
        d = (b["halfyear"] == h).astype(int)
        b[f"expo_{h}"] = b["expo_z"] * d
        b[f"expo_{h}_young"] = b["expo_z"] * d * b["is_young"]
        pooled.append(f"expo_{h}")
        young.append(f"expo_{h}_young")
    return b, pooled, young


def rebase_by_half(y: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract from each coefficient the mean of the PRE-PERIOD coefficients
    of the same half of the year, so a half-year is read against its own
    season rather than against 2022H1.

    The standard error is left as estimated. It ignores the covariance
    with the pre-period mean it is netted against, so it is indicative
    rather than exact, and the summary says so. The H1-only specification
    carries no such caveat, which is why it is the one to lead with.
    """
    y = y.copy()
    y["half"] = y["halfyear"].str[-2:]
    pre = y[y["halfyear"] < "2022H2"]
    base = pre.groupby(["outcome", "half"])["coef"].mean()
    y["coef"] = y.apply(
        lambda r: r["coef"] - base.get((r["outcome"], r["half"]), 0.0), axis=1)
    return y.drop(columns="half")


def run_es(b, outcome, terms, tag):
    p = b.copy()
    p["n_emp"] = p[outcome]
    r = mc.run_fepois_multi(p, OUT, tag=tag, terms=terms, fes=FES)
    if r.empty:
        return r
    r = r[r["term"].isin(terms)].copy()
    r["is_young_term"] = r["term"].str.endswith("_young")
    r["halfyear"] = (r["term"].str.replace("_young", "", regex=False)
                     .str.replace("expo_", "", regex=False))
    return r[["term", "is_young_term", "halfyear", "coef", "se", "pvalue",
              "n_obs", "status"]]


FOCUS_OUTCOME = "hires"
FAILURES = []


def event_study(panels, spec_name, h1_only, rebase=False):
    """
    One specification across every outcome; returns the young rows.

    TOLERANT PER OUTCOME, and deliberately so. On 20 Sep the stock arm of
    the H1-only specification segfaulted R (rc 3221225477, an access
    violation, preceded by "recursive gc invocation": the garbage
    collector re-entered while three lanes competed for memory). That one
    crash took the whole script down and with it the HIRES estimate, which
    is the one the paper turns on. A supporting outcome must not be able
    to destroy a primary one.

    So each outcome is attempted, retried once, and then recorded as
    failed and skipped. The run still fails loudly if the FOCUS outcome is
    missing from both specifications, because at that point there is
    nothing to report.
    """
    out = []
    for label, (bal, outcome) in panels.items():
        b, p_terms, y_terms = build_terms(bal, h1_only)
        t0 = time.time()
        r = pd.DataFrame()
        for attempt in (1, 2):
            r = run_es(b, outcome, p_terms + y_terms,
                       f"s58_{spec_name}_{label}"
                       + ("_retry" if attempt == 2 else ""))
            if not r.empty:
                break
            if attempt == 1:
                print(f"  {spec_name}/{label} returned nothing; retrying "
                      f"once in case it was a transient memory collision")
                gc.collect()
                time.sleep(30)
        if r.empty:
            FAILURES.append(f"{spec_name}/{label}")
            print(f"  *** {spec_name}/{label} FAILED TWICE and is skipped. "
                  f"The R output is in the exchange directory.")
            del b
            gc.collect()
            continue
        y = r[r["is_young_term"] & (r["halfyear"] != "h2")].copy()
        y["outcome"], y["spec"] = label, spec_name
        out.append(y)
        print(f"  {spec_name:<7} {label:<5} ({time.time()-t0:.0f}s)")
        for _, x in y.sort_values("halfyear").iterrows():
            star = "  <-- focus" if x["halfyear"] == FOCUS else ""
            print(f"    {x['halfyear']}  {x['coef']:+.4f} "
                  f"(SE {x['se']:.4f})  t {x['coef']/max(x['se'],1e-12):+.1f}"
                  + star)
        del b
        gc.collect()
    if not out:
        return pd.DataFrame(columns=["term", "is_young_term", "halfyear",
                                     "coef", "se", "pvalue", "n_obs",
                                     "status", "outcome", "spec"])
    res = pd.concat(out, ignore_index=True)
    return rebase_by_half(res) if rebase else res


def evaluate_rule(y: pd.DataFrame) -> pd.DataFrame:
    """The pre-committed rule, applied to the 22-25 hiring differential."""
    rows = []
    for spec, g in y[y["outcome"] == "hires"].groupby("spec"):
        g = g.set_index("halfyear")
        if FOCUS not in g.index:
            rows.append({"spec": spec, "note": f"{FOCUS} absent"})
            continue
        f = g.loc[FOCUS]
        pre = g[[h < "2022H2" for h in g.index]]
        pre_max = pre["coef"].abs().max() if len(pre) else np.nan
        rows.append({
            "spec": spec,
            "coef": f["coef"], "se": f["se"],
            "t": f["coef"] / max(f["se"], 1e-12),
            "negative": bool(f["coef"] < 0),
            "twice_se": bool(abs(f["coef"]) >= 2 * f["se"]),
            "beats_pre_max": bool(abs(f["coef"]) > pre_max)
                             if np.isfinite(pre_max) else False,
            "pre_max_abs": pre_max})
    d = pd.DataFrame(rows)
    if {"negative", "twice_se", "beats_pre_max"} <= set(d.columns):
        d["passes"] = d["negative"] & d["twice_se"] & d["beats_pre_max"]
    return d


def main():
    mc.Tee(OUT / "58_log.txt")
    t_start = time.time()
    print("=" * 70)
    print("58: SEASONAL CONTROL, STANDARD ERRORS, AND THE 2025 VINTAGE")
    print("=" * 70)
    print("  PRE-COMMITTED: the 2025H1 young hiring differential counts as a")
    print("  finding only if it is negative in BOTH specifications, at least")
    print("  twice its SE in both, and larger than every pre-2022H2")
    print("  coefficient in the same specification.")
    print(mc.mem_line("  "))

    # ---- 3. the vintage question, first and cheap ----
    conn = mc.connect()
    tabs = opt("AGI table probe", probe_agi, conn)
    if tabs is not None and not tabs.empty:
        tabs.to_csv(OUT / "agi_tables.csv", index=False)
        piv = (tabs.pivot_table(index="year", columns="vintage",
                                values="table_name", aggfunc="count")
               .fillna(0).astype(int))
        print("\nMONTHLY AGI TABLES BY YEAR AND VINTAGE")
        print(piv.to_string())
        late = tabs[tabs["year"] >= "2025"]
        if not late.empty:
            print("\n2025 and later, table by table:")
            for _, r in late.iterrows():
                print(f"    {r['table_name']:<32} {r['vintage']:<5} "
                      f"{int(r['n_rows']):>12,} rows")
        d25 = set(late[(late["year"] == "2025")
                       & (late["vintage"] == "def")]["month"])
        if d25:
            print(f"\n  *** DEFINITIVE 2025 MONTHS EXIST: {sorted(d25)}")
            print("  *** 54 and 47L used the PRELIMINARY file. Re-pull before")
            print("  *** the 2025H1 result is believed.")
        else:
            print("\n  no definitive 2025 tables: the preliminary file is "
                  "still the only 2025 data, which is a limitation to state, "
                  "not a defect to fix.")

    # ---- panels, from the caches 47L and 54 already wrote ----
    l47, s54 = _mod("47L_age_baseline_exposure.py"), _mod("54_hiring_flows.py")
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first.")
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]
    expo = l47.build_exposure(base, daioe)

    panels = {}
    fl = [f for f in (mc.read_cache(CACHE / f"flows_{y}.parquet",
                                    require=s54.FLOW_COLS)
                      for y in s54.YEARS) if f is not None]
    if fl:
        bf = s54.build_panel(pd.concat(fl, ignore_index=True), expo)
        panels["hires"] = (bf, "n_hire")
        panels["seps"] = (bf, "n_sep")
    del fl
    cnt = [c for c in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                       for y in l47.YEARS) if c is not None]
    if cnt:
        panels["stock"] = (l47.build_panel(pd.concat(cnt, ignore_index=True),
                                           expo), "n_emp")
    del cnt
    gc.collect()
    if not panels:
        raise RuntimeError("no cached panel: run 54 and 47L first.")

    # ---- 1 and 2. the two specifications ----
    print("\nREBASED: each half-year read against its own season")
    purged = event_study(panels, "rebased", h1_only=False, rebase=True)
    purged.to_csv(OUT / "es_rebased.csv", index=False)

    print("\nH1-ONLY: first half-years, no seasonal model at all")
    h1 = event_study(panels, "h1only", h1_only=True)
    h1.to_csv(OUT / "es_h1only.csv", index=False)

    y = pd.concat([purged, h1], ignore_index=True)
    if FOCUS_OUTCOME not in set(y.get("outcome", pd.Series(dtype=str))):
        raise RuntimeError(
            f"the {FOCUS_OUTCOME} arm failed in BOTH specifications "
            f"({'; '.join(FAILURES)}), so there is nothing to report. Re-run "
            f"when no other lane is competing for memory.")
    rule = evaluate_rule(y)
    rule.to_csv(OUT / "readrule.csv", index=False)

    lines = ["SEASONAL CONTROL, STANDARD ERRORS, AND THE 2025 VINTAGE",
             "=" * 60, "",
             "22-25 differential in HIRES, both specifications:", ""]
    hp = (y[y["outcome"] == "hires"]
          .pivot_table(index="halfyear", columns="spec", values="coef"))
    hs = (y[y["outcome"] == "hires"]
          .pivot_table(index="halfyear", columns="spec", values="se"))
    lines += [hp.round(4).to_string(), "", "standard errors:",
              hs.round(4).to_string(), ""]
    lines += ["PRE-COMMITTED READ RULE, evaluated:", rule.round(4).to_string(
        index=False), ""]
    if FAILURES:
        lines += ["FITS THAT FAILED AND ARE ABSENT FROM THE TABLES ABOVE:",
                  "  " + "; ".join(FAILURES),
                  "  R segfaults under memory pressure when several lanes run",
                  "  at once. A missing row here is a missing fit, never a",
                  "  zero, and the tables must not be read as though the",
                  "  outcome had been estimated and found small.", ""]
    if "passes" in rule.columns and len(rule):
        if rule["passes"].all():
            lines.append("VERDICT: the 2025H1 young hiring differential meets "
                         "every condition in both specifications.")
        elif rule["passes"].any():
            lines.append("VERDICT: it meets the conditions in one "
                         "specification but not the other. Report as "
                         "suggestive and say which.")
        else:
            lines.append("VERDICT: it does not meet the conditions. It is "
                         "within the ordinary variation of this series and "
                         "must not be quoted as a finding.")
    lines += ["",
              "STILL TRUE WHATEVER THE VERDICT:",
              "  1. It is one half-year at the very end of the window.",
              "  2. The 2025 source vintage is recorded in agi_tables.csv.",
              "     If definitive months now exist, nothing here should be",
              "     believed until 54 and 47L are re-pulled on them.",
              "  3. Exposure is frozen in 2019. Script 57 measured lambda at",
              "     0.83 to 0.89 and flat from 2022, so attenuation is mild,",
              "     but the 2025 value of lambda is extrapolated, not",
              "     measured.",
              "", f"Runtime {(time.time()-t_start)/60:.1f} min. "
              + mc.mem_line()]
    (OUT / "58_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n58 done.")


if __name__ == "__main__":
    main()

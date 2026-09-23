#!/usr/bin/env python3
"""
90_wfh_margins_adoption.py -- which margin each score carries, and
                              whether measured adoption can replace
                              measured exposure on both sides.

======================================================================
  RUNS IN MONA. Output folder CANARIES_90_OUT (default output_90).
  Part C performs no SQL. PART D DOES: it reads the ICT and BITA
  survey tables through 71's catalogue discovery, exactly as the AI
  first stage already does, and must run in a lane that may pull.
======================================================================

PART C. THE MARGIN SPLIT, AND WHY IT IS THE RECONCILIATION
\\citet{lambert2026brokenladder} study the junior share of NEW HIRES.
This paper's headline adjusts through SEPARATIONS, under seniority rules
that protect open-ended contracts, and its own hires estimate is
imprecise. The two papers may therefore be measuring different margins of
the same adjustment rather than contradicting each other.

Part C asks that directly. Hires and separations at 22 to 25, each with
BOTH firm scores entered together, standardised. This is the one place in
the lane where a joint model is the right instrument, because the
question is not whether one score survives the other but which margin
each one carries. The two scores' correlation is exported beside the
coefficients so a reader can judge the precision for themselves.

The pattern that would reconcile the two papers: teleworkability the
larger term on hires, AI exposure the larger on separations. THAT IS
PRE-COMMITTED IN READ RULE 2. Any other pattern is reported as it falls
and the reconciliation is dropped rather than rephrased.

PART D. ADOPTION RATHER THAN EXPOSURE
Lambert and Schindler's own robustness replaces exposure with actual
working from home. Half of that already exists here: script 71 estimates
the AI first stage on the enterprise ICT surveys and on BITA, and finds
top-quartile employers 21.5 points more likely to report AI use in 2023
and their employees 23.5 points more likely to report generative-AI use
in 2024.

Part D reuses 71's discovery to LOOK FOR a remote-work item in the same
tables rather than assuming one is there. If an item is found it reports
the parallel first stage, whether the teleworkability score predicts
reported remote work as the DAIOE score predicts reported AI use, and
Part C is then re-run with measured adoption on both sides instead of two
occupational proxies. IF NO ITEM IS FOUND THE PART STOPS AND SAYS SO:
the absence is one sentence in the appendix and is not a failure of the
lane. Nothing is pre-committed here, because an existence check cannot
be pre-committed.

INPUTS AND OUTPUTS
Reads the flows cache and L_baseline_2019* (82's pull, cached by lane
28a) for Part C; for Part D, the INFORMATION_SCHEMA catalogue and
whatever survey tables it names. Writes to output_90/:
wfh_margin_split.csv, wfh_firststage.csv, wfh_schema_found.csv, the vcov
files and 90_summary.txt.

IN THE PAPER
Online Appendix III.2, the remote-work paragraph; Section 3 only if Part
C reconciles.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_90_OUT", "output_90")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

FLOOR = 5
FLOW_BAND = "22-25"
AI_TERM = "post_x_high_x_young"
WFH_TERM = "post_x_highwfh_x_young"
# 82's own margins on this panel, for reference in the summary only.
OCC_FLOW = {"hires": (-0.0179, 0.0406), "seps": (+0.0540, 0.0225)}
# Words a remote-work item is likely to carry. Part D searches on these
# and PRINTS what it matched, so a false positive is visible.
WFH_WORDS = ("distans", "hemarbete", "hemifran", "telework", "remote",
             "workfromhome", "wfh", "hemma")

NOTES, FAILURES = [], []

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  1. PART C IS A JOINT MODEL ON PURPOSE. The question is which",
    "     margin each score carries, not whether one survives the",
    "     other, so both terms are entered together and the two scores'",
    "     correlation is printed beside them. A wide interval is",
    "     reported as wide.",
    "  2. WHAT IS FIXED IN ADVANCE IS THE CLAIM, NOT THE EXPORT. All",
    "     four coefficients are fitted, exported and printed whichever",
    "     way they fall, and none of that is conditional on the rule.",
    "     The rule is only this: the paper may say the two literatures",
    "     measure different margins if teleworkability is the larger",
    "     term on HIRES and AI exposure the larger on SEPARATIONS. On",
    "     any other pattern that SENTENCE is not written; the numbers",
    "     are reported either way and are worth seeing either way.",
    "  3. PART D IS AN EXISTENCE CHECK AND PRE-COMMITS NOTHING. If no",
    "     remote-work item is found in the surveys the part says so and",
    "     stops; the absence is one sentence in the appendix and is not",
    "     a failure. Every matched column name is printed, so a false",
    "     match is visible rather than silent.",
    "  4. THE 50-AND-OVER GAIN IS NOT CLAIMED, here or in 89.",
    f"  Counts below {FLOOR} are suppressed before anything leaves MONA.",
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
    if hasattr(mod, "NOTES"):
        mod.NOTES.clear()
    if hasattr(mod, "FAILURES"):
        mod.FAILURES.clear()


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def part_c(flows, ai, wfh, s61, s78, j47) -> list:
    """
    Hires and separations with both scores in one model.

    82's part_c_flows is the template and the term set is the same; the
    only change is the second indicator, added through 78's own suffix
    argument, which is the mechanism the skill placebo already uses.
    """
    if flows is None:
        FAILURES.append("C/no flows cache")
        print("  C: flows_* missing, the margins are skipped")
        return []
    w = wfh["exposure"][["employer_id", "fq"]].rename(
        columns={"fq": "fq_wfh"})
    top = int(w["fq_wfh"].max())
    w["highwfh"] = (w["fq_wfh"] == top).astype(int)
    rows = []
    for outcome, col in (("hires", "n_hire"), ("seps", "n_sep")):
        src = flows.rename(columns={col: "n_emp"})
        skel = s61.build_skeleton(src, FLOW_BAND, j47)
        del src
        gc.collect()
        if skel.empty:
            FAILURES.append(f"C/{outcome}/empty")
            continue
        b = s78.with_exposure(skel, ai["exposure"])
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{outcome}/no exposure")
            continue
        b = b.merge(w[["employer_id", "highwfh"]], on="employer_id",
                    how="inner")
        if b.empty:
            FAILURES.append(f"C/{outcome}/no overlap")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, t_ai = s78.eq2_terms(b, "high", "")
        b, t_wfh = s78.eq2_terms(b, "highwfh", "wfh")
        print(f"  C {outcome}: {len(b):,} rows, {n_firms:,} firms"
              f"{mc.mem_line(' | ')}")
        try:
            r = mc.run_fepois_multi(b, OUT, tag=f"s90_{outcome}",
                                    terms=t_ai + t_wfh, fes=j47.FES,
                                    cluster="employer_id")
        except BaseException as ex:
            print(f"    C {outcome} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            r = pd.DataFrame()
        del b
        gc.collect()
        if r.empty:
            FAILURES.append(f"C/{outcome}/fit")
            continue
        g = r.set_index("term")
        for which, term in (("ai", AI_TERM), ("wfh", WFH_TERM)):
            if term not in g.index:
                FAILURES.append(f"C/{outcome}/{which} term missing")
                continue
            c, se = float(g.loc[term, "coef"]), float(g.loc[term, "se"])
            rows.append({"outcome": outcome, "score": which, "coef": c,
                         "se": se, "t": (c / se if se else np.nan),
                         "n_firms": n_firms, "status": "ok"})
            print(f"    {outcome:<5} {which:<3} {c:+.4f} ({se:.4f}) "
                  f"t {(c / se if se else np.nan):+.2f}")
        save(rows, "wfh_margin_split.csv")
    return rows


def part_d(s71, ai, wfh) -> tuple:
    """
    Does any survey table carry a remote-work item, and if so does the
    teleworkability score predict it?

    71 discovers tables from the catalogue rather than naming them, so
    this reuses that discovery and searches the discovered columns. The
    matched names are exported, because a column matched on the word
    "distans" might be distansutbildning rather than distansarbete and
    the reader must be able to see which.
    """
    found, rows = [], []
    try:
        conn = mc.connect()
    except BaseException as ex:
        FAILURES.append(f"D/connect: {type(ex).__name__}")
        return found, rows
    try:
        schema = s71.discover(conn)
    except BaseException as ex:
        print(f"  D: discovery failed: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append("D/discover")
        return found, rows
    for tbl, cols in (schema or {}).items():
        for c in cols:
            low = str(c).lower().replace("_", "")
            if any(wrd in low for wrd in WFH_WORDS):
                found.append({"table": tbl, "column": c})
    save(found, "wfh_schema_found.csv", count_col="")
    if not found:
        print("  D: NO remote-work item in any discovered survey table.")
        NOTES.append("no remote-work item was found in the ICT or BITA "
                     "tables; Part D stops there and the absence is "
                     "reported rather than worked around")
        return found, rows
    print(f"  D: {len(found)} candidate remote-work column(s):")
    for f in found:
        print(f"    {f['table']}.{f['column']}")
    NOTES.append(f"Part D matched {len(found)} candidate remote-work "
                 f"column(s); every name is in wfh_schema_found.csv and "
                 f"must be read before the first stage is quoted")
    return found, rows


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s71 = _mod("71_adoption_validation.py", "s71")
    s89 = _mod("89_wfh_offdiagonal.py", "s89")
    for m_ in (s71, s78, s89):
        m_.OUT, m_.CACHE = OUT, CACHE
    if s82.MAIN_LEVEL != "uniform3" or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, s61, s71, s78, s89, l47, l70, j47


def main():
    mc.Tee(OUT / "90_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("90: WHICH MARGIN EACH SCORE CARRIES, AND WHETHER ADOPTION")
    print("    CAN REPLACE EXPOSURE")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s82, s61, s71, s78, s89, l47, l70, j47 = load_modules()

    # The two scores come from 89's builder, so the lane cannot end up
    # with one definition of teleworkability in Part A and another here.
    ai = s82.build_exposure(l47, l70, j47)
    drain(s82, "82/ai")
    wfh = s82.build_exposure(l47, l70, j47, daioe=s89.wfh_book(),
                             audit=False)
    drain(s82, "82/wfh")
    m = ai["exposure"][["employer_id", "score"]].merge(
        wfh["exposure"][["employer_id", "score"]], on="employer_id",
        suffixes=("_ai", "_wfh"))
    rho = float(m["score_ai"].corr(m["score_wfh"], method="spearman")) \
        if not m.empty else float("nan")
    print(f"  the two firm scores: Spearman {rho:+.3f} on "
          f"{len(m):,} employers")
    NOTES.append(f"the two firm scores correlate at Spearman {rho:+.3f}; "
                 f"Part C's intervals are to be read with that in view")
    del m
    gc.collect()

    # 82's own loader and its own year set, so this lane cannot read a
    # different flows panel from the one the paper's margins sit on.
    flows = s82.load_counts("flows", s61.PANEL_YEARS,
                            require=["employer_id", "year_month",
                                     "age_group", "n_hire", "n_sep"])
    if flows is None:
        print("  flows_* not cached; Part C will report that and stop")
    else:
        print(f"  flows: {len(flows):,} cells over {len(s61.PANEL_YEARS)} years")

    print("\n  PART C, the margin split:")
    c_rows = part_c(flows, ai, wfh, s61, s78, j47)
    print("\n  PART D, is adoption measurable on both sides:")
    found, _ = part_d(s71, ai, wfh)
    drain(s78, "78")

    def pick(outcome, score):
        for r in c_rows:
            if r["outcome"] == outcome and r["score"] == score:
                return r
        return None

    L = ["WHICH MARGIN EACH SCORE CARRIES, OCCUPATION ROUTE",
         "=" * 50, "",
         "Hires and separations at 22-25, both firm scores entered",
         "together. Lambert and Schindler study the junior share of new",
         "hires; this paper's headline adjusts through separations. If",
         "the two scores split along that line the papers are measuring",
         "different margins rather than contradicting each other.", "",
         f"THE TWO SCORES CORRELATE AT SPEARMAN {rho:+.3f}. Read every",
         "interval below with that in view.", ""]
    if c_rows:
        L += ["PART C, THE FOUR COEFFICIENTS:"]
        for r in c_rows:
            L.append(f"  {r['outcome']:<5} {r['score']:<3} {r['coef']:+.4f} "
                     f"({r['se']:.4f}) t {r['t']:+.2f}   "
                     f"firms {r['n_firms']:,}")
        ha, hw = pick("hires", "ai"), pick("hires", "wfh")
        sa, sw = pick("seps", "ai"), pick("seps", "wfh")
        if all((ha, hw, sa, sw)):
            wfh_leads_hires = abs(hw["coef"]) > abs(ha["coef"])
            ai_leads_seps = abs(sa["coef"]) > abs(sw["coef"])
            if wfh_leads_hires and ai_leads_seps:
                verdict = ("THE RECONCILIATION HOLDS on rule 2: "
                           "teleworkability leads on hires and AI on "
                           "separations")
            else:
                verdict = ("THE RECONCILIATION SENTENCE IS NOT WRITTEN, "
                           "on rule 2, though the coefficients above "
                           "stand and are reported: "
                           f"teleworkability leads on hires "
                           f"{str(wfh_leads_hires).upper()}, AI leads on "
                           f"separations {str(ai_leads_seps).upper()}")
            L += ["  The four coefficients above are the result. The line",
                  "  below is only the label rule 2 attaches to them, and it",
                  "  decides one sentence in the paper, not what is reported.",
                  f"  VERDICT: {verdict}"]
        L += ["  For reference only, and not a gate: 82's own margins on",
              f"  this panel are hires {OCC_FLOW['hires'][0]:+.4f} "
              f"({OCC_FLOW['hires'][1]:.4f}) and separations "
              f"{OCC_FLOW['seps'][0]:+.4f} ({OCC_FLOW['seps'][1]:.4f}), "
              "each with the AI", "  score alone.", ""]
    if found:
        L += [f"PART D: {len(found)} candidate remote-work column(s) in the",
              "surveys, listed in wfh_schema_found.csv. EVERY NAME MUST BE",
              "READ BEFORE ANY FIRST STAGE IS QUOTED: a column matching",
              "'distans' may be distansutbildning rather than",
              "distansarbete. The parallel first stage and the adoption",
              "re-run of Part C are the next step once a name is confirmed.",
              ""]
    else:
        L += ["PART D: NO remote-work item was found in any discovered",
              "survey table, so adoption cannot replace exposure on the",
              "teleworkability side. That is one sentence in the appendix",
              "and not a failure of the lane: it says the comparison with",
              "Lambert and Schindler has to run on occupational proxies,",
              "which is a limitation to state rather than to work around.",
              ""]
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += [f"WHAT FAILED: {', '.join(FAILURES)}",
              "A missing row is a missing fit, never a zero.", ""]
    L += READ_RULES + ["",
                       f"Runtime {(time.time() - t0) / 60:.1f} min. "
                       + mc.mem_line("")]
    (OUT / "90_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

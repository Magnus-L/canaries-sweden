#!/usr/bin/env python3
"""
65_occupation_arm.py -- the same firms, classified by the other register.

======================================================================
  RUNS IN MONA. No SQL. Reads 47L's cached 2019 occupation baseline and
  monthly counts. Writes output_65/.
======================================================================

THE QUESTION, WHICH A CO-AUTHOR ASKED AND WHICH DESERVED A BETTER ANSWER
THAN THE ONE IT GOT.

Script 61 classifies a firm by the EDUCATION mix of its incumbents aged
31 and over in 2019. Occupation enters only as the bridge that gives
each education group its exposure score. Yet almost every worker has an
occupation code in 2019, the register for that year is final, and a
frozen 2019 occupational exposure would be the more direct measure. So
why the detour?

Two reasons, of unequal strength.

The strong one is SCB's own documentation of the occupation register:
it samples about half the workforce, and only about two per cent of the
smallest firms, with the rest imputed. The imputation is fine for a
cross-section and is not designed for analysing transitions. For a small
firm, therefore, a 2019 occupational exposure is substantially an
imputation of the quantity we want to measure. Education is a census.
Occupation is also missing outright for about fifteen per cent of wage
earners in 2019.

The weak one is history. The within-employer design was built while we
were trying to replace occupation wholesale, before freezing exposure in
2019 made the detour unnecessary.

So this runs the identical design on the identical firms with the
identical outcome, and changes only which register does the
classification. If the two agree in sign, the paper can say the result
does not depend on the register, and can keep the education version as
primary for the coverage reason above. That is a stronger position than
choosing one route and defending it.

WHAT TO EXPECT, WRITTEN BEFORE THE RUN. Script 62 ran both measures at
firm level on the employment stock and found the occupational one
LARGER: -0.0278 at 22-25 against -0.0067 for education. If that carries
over, the education route is the conservative one and our headline
understates. If the occupational arm instead comes back near zero, the
headline depends on the register and we say so.

WHAT THIS DOES NOT FIX. About a third of 2019 occupation codes were
assigned in an earlier year, so the measure is pre-treatment but not
contemporaneous. That attenuates it toward zero and cannot explain a
result; it can only hide one. Filtering on the assignment year needs a
column the cached baseline does not carry, so it is a SQL job for
another round rather than a change here.

Output (output_65/):
  occ_step.csv     the three disjoint windows, occupational classification
  occ_pooled.csv   the single post-2024 coefficient
  65_summary.txt   both registers side by side
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_65"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
MIN_FIRM_INCUMBENTS = 5          # 47j's floor, kept so the samples match
FAILURES = []


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def occupation_exposure(base: pd.DataFrame, daioe: pd.DataFrame,
                        incumbent_bands) -> pd.DataFrame:
    """
    Firm exposure from the 2019 OCCUPATIONS of incumbents aged 31 and
    over, worker-weighted, with quartile cutoffs fixed on that same
    worker-weighted distribution.

    This mirrors 47j's incumbent_exposure line for line, including the
    floor on incumbents and the weighting of the cutoffs, so that the two
    arms differ in the register and in nothing else. Returns
    employer_id, fq, mix, n.
    """
    b = base.copy()
    b["ssyk4"] = b["ssyk4"].astype(str).str.zfill(4)
    b = b[b["age_group"].astype(str).isin(incumbent_bands)]
    b = b[b["ssyk4"] != "____"].merge(daioe, on="ssyk4", how="inner")
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    b = b[b["n"] > 0]
    if b.empty:
        return pd.DataFrame(columns=["employer_id", "fq", "mix", "n"])
    b["ws"] = b["score"] * b["n"]
    fy = (b.groupby("employer_id", observed=True)
          .agg(ws=("ws", "sum"), n=("n", "sum")).reset_index())
    fy = fy[fy["n"] >= MIN_FIRM_INCUMBENTS]
    if fy.empty:
        return pd.DataFrame(columns=["employer_id", "fq", "mix", "n"])
    fy["mix"] = fy["ws"] / fy["n"]
    o = np.argsort(fy["mix"].to_numpy(), kind="stable")
    v, w = fy["mix"].to_numpy()[o], fy["n"].to_numpy()[o]
    cum = np.cumsum(w) / w.sum()
    cuts = [float(v[np.searchsorted(cum, q, side="left")])
            for q in (0.25, 0.5, 0.75)]
    fy["fq"] = np.searchsorted(np.asarray(cuts), fy["mix"].to_numpy(),
                               side="right") + 1
    return fy[["employer_id", "fq", "mix", "n"]]


def main():
    mc.Tee(OUT / "65_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("65: THE SAME DESIGN, CLASSIFIED BY OCCUPATION INSTEAD")
    print("=" * 70)
    print("  Same firms, same outcome, same windows, same fixed effects.")
    print("  Only the register that assigns the exposure quartile changes.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    j47 = s61._j47()

    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first. "
                           "This script performs no SQL.")
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                                "score"]]
    expo = occupation_exposure(base, daioe, j47.INCUMBENT_BANDS)
    if expo.empty:
        raise RuntimeError("no firm could be classified by occupation")
    shares = (expo.groupby("fq")["n"].sum() / expo["n"].sum()).round(3)
    print(f"  occupational exposure: {len(expo):,} firms, quartile shares of "
          f"incumbent employment " + " ".join(f"Q{k} {v:.2f}"
                                              for k, v in shares.items()))
    del base
    gc.collect()

    cnt = []
    for y in s61.PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise RuntimeError(f"L_counts_{y}.parquet missing: run 47L first.")
        cnt.append(c)
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(counts["year_month"].max())
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")
    if last < s61.POOLED_FROM:
        raise RuntimeError(f"the counts end at {last}, before the adoption "
                           f"window opens at {s61.POOLED_FROM}. Refusing.")

    step_rows, pooled_rows = [], []
    for band in j47.YOUNG_BANDS:
        t1 = time.time()
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            print(f"  {band}: empty panel, skipped")
            continue
        b, step_terms, pool_terms = s61.attach_exposure(skel, expo)
        del skel
        gc.collect()
        if b.empty:
            print(f"  {band}: no firms matched, skipped")
            continue
        print(f"\n  {band}: panel {len(b):,} rows ({time.time()-t1:.0f}s)")
        for label, terms, sink in (("step", step_terms, step_rows),
                                   ("pooled", pool_terms, pooled_rows)):
            t2 = time.time()
            r = mc.run_fepois_multi(b, OUT,
                                    tag=f"o65_{label}_{band.replace('-','_')}",
                                    terms=terms, fes=j47.FES)
            if r.empty:
                FAILURES.append(f"{label}/{band}")
                print(f"    {label}: FAILED, recorded and skipped")
                continue
            for _, x in r.iterrows():
                sink.append({"register": "occupation", "young_band": band,
                             "term": x["term"], "coef": float(x["coef"]),
                             "se": float(x["se"]), "n_obs": int(x["n_obs"]),
                             "status": str(x.get("status", "ok"))})
            pd.DataFrame(step_rows).to_csv(OUT / "occ_step.csv", index=False)
            pd.DataFrame(pooled_rows).to_csv(OUT / "occ_pooled.csv",
                                             index=False)
            show = r[r["term"].str.startswith(("s_", "post2024"))]
            for _, x in show.iterrows():
                t = x["coef"] / max(x["se"], 1e-12)
                print(f"    {label:<6} {x['term']:<26} {x['coef']:+.4f} "
                      f"(SE {x['se']:.4f}) t {t:+.2f} "
                      f"[{(time.time()-t2)/60:.1f} min]")
        del b
        gc.collect()

    lines = ["THE SAME DESIGN, CLASSIFIED BY OCCUPATION INSTEAD", "=" * 52, "",
             "Firm exposure from the 2019 OCCUPATIONS of incumbents aged 31+,",
             "against 61's education mix of the same incumbents. Same panel,",
             "same windows, same fixed effects, same floor on incumbents.", ""]
    P = pd.DataFrame(pooled_rows)
    if not P.empty:
        pp = P[P["term"] == "post2024_x_high_x_young"]
        lines += ["POOLED from 2024-01, occupational classification:"]
        for _, r in pp.iterrows():
            lines.append(f"  {r['young_band']:<6} {r['coef']:+.4f} "
                         f"({r['se']:.4f}) t "
                         f"{r['coef']/max(r['se'],1e-12):+.2f}")
        lines.append("")
        # 61's own answer, if it has been run, so the comparison is in one file
        e = HERE / "output_61" / "redated_pooled.csv"
        if e.exists():
            try:
                E = pd.read_csv(e)
                E = E[(E.term == "post2024_x_high_x_young")
                      & (E.arm == "true") & (E.design == "OL_daioe")]
                lines += ["THE TWO REGISTERS SIDE BY SIDE, pooled from 2024-01:",
                          "  band    education      occupation"]
                for _, r in pp.iterrows():
                    m = E[E.young_band == r["young_band"]]
                    if m.empty:
                        continue
                    lines.append(f"  {r['young_band']:<6} "
                                 f"{float(m['coef'].iloc[0]):+.4f} "
                                 f"({float(m['se'].iloc[0]):.4f})   "
                                 f"{r['coef']:+.4f} ({r['se']:.4f})")
                lines.append("")
            except Exception as ex:
                lines.append(f"  (61's pooled file unreadable: "
                             f"{type(ex).__name__})")
        else:
            lines.append("  (61 has not been run here, so no side-by-side)")
            lines.append("")
    S = pd.DataFrame(step_rows)
    if not S.empty:
        st = S[S["term"].str.startswith("s_")]
        for band in sorted(st["young_band"].unique()):
            d = st[st.young_band == band]
            bits = "  ".join(f"{r['term'][2:]} {r['coef']:+.4f}"
                             f"({r['coef']/max(r['se'],1e-12):+.1f})"
                             for _, r in d.iterrows())
            lines.append(f"STEP, young = {band}: {bits}")
        lines.append("")
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES), ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. Agreement in sign is the claim: the result does not depend on",
        "     which register classifies the firm. Agreement in size is not",
        "     expected, since the two measures rank firms differently and",
        "     each is standardised on its own distribution.",
        "  2. Education stays primary. The occupation register samples about",
        "     half the workforce and about two per cent of the smallest",
        "     firms, and SCB says the imputed part is not built for",
        "     analysing transitions. That is a coverage argument and it is",
        "     not settled by this comparison.",
        "  3. About a third of 2019 occupation codes were assigned earlier,",
        "     so this measure is pre-treatment but not contemporaneous. That",
        "     attenuates it and cannot manufacture a result.",
        "  4. No as-of arm is run here and none is needed: the measure uses",
        "     2019 codes only, and 2019 is final. The lag this project is",
        "     about lives in the years after it.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "65_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n65 done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
65_occupation_arm.py: the same design with employers classified by the
2019 occupations of their incumbents instead of their education.

QUESTION
A first occupational classification of the within-employer design: an
employer scored by the 2019 occupations of its incumbents aged 31 to 69
rather than by their education, on script 61's panel, so the two
registers can be compared in sign. The paper's occupation route is
script 82, which differs from this one in its scoring floor (script 82
counts incumbent person-months, where this script counts coded
incumbents) and in its three-digit score book.

DESIGN
Exposure (occupation_exposure): the worker-weighted mean DAIOE
generative-AI percentile of the four-digit occupations held in November
2019 by the employer's incumbents aged 31 to 69, from script 47L's
baseline; employers with fewer than five coded incumbents are not scored;
quartile cut points are weighted by incumbent employment, as in script
47j. Panel, windows, terms and fixed effects are script 61's: employer by
age band by month from January 2021 to June 2025, PostRB x High x Young
plus the three disjoint windows or the single step from January 2024,
employer-by-month, employer-by-age and month-by-age effects, Poisson
pseudo-maximum likelihood, standard errors clustered by employer. No
as-of arm is run, since the 2019 register is final.

INPUTS AND OUTPUTS
Reads the caches L_baseline_2019 and L_counts_2021 to 2025 (script 47L)
and the input file daioe_quartiles.dta; performs no SQL. Writes to
output_65/: occ_step.csv, occ_pooled.csv and 65_summary.txt, the last
with script 61's education-based estimate beside the occupational one
when output_61/ is present.

IN THE PAPER
No coefficient from this script is quoted. Its occupation_exposure
function is imported by scripts 70 and 71, which are in turn imported by
the occupation-route scripts 82 and 83.
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

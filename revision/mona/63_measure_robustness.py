#!/usr/bin/env python3
"""
63_measure_robustness.py -- every design we have rests on one exposure
                            measure. Does the answer?

======================================================================
  RUNS IN MONA. No SQL: reads the caches 47L and 54 already wrote.
  Writes output_63/.
======================================================================

THE PROBLEM THIS ADDRESSES, WHICH IS THE LARGEST ONE LEFT.

Four designs now agree, and they were built to fail differently: the
register lag reaches some and not others, some identify across firms and
some within, some read the stock and some the flow. That is the point of
having four.

They share one thing. Every one of them assigns exposure from DAIOE at
four-digit occupation. If DAIOE mismeasures what generative AI does to an
occupation, all four are wrong together and their agreement is worth
nothing. A common mode defeats triangulation completely, and it is the
one weakness our design portfolio cannot see.

We cannot validate DAIOE inside P1207: there is no survey of who actually
uses a language model at work in this project's data. What we CAN do is
ask whether the conclusions depend on the measure, using two alternatives
already in the input directory.

  ELOUNDOU. A different research team, a different method, the same
  object. If it gives the same answer, the finding does not depend on our
  own measure. If it does not, we have a problem we did not know about.

  DINGEL AND NEIMAN TELEWORKABILITY, AND THIS IS THE REAL TEST. It scores
  how far a job can be done from home. It is not an AI measure at all, and
  it correlates with AI exposure because both load on desk work. So it is
  a PLACEBO MEASURE: if teleworkability produces the same age gradient as
  DAIOE, then what we are measuring is "office work", not "exposure to
  generative AI", and the paper's interpretation is wrong regardless of
  how clean the identification is.

  That second test can fail, and it is the one I would run first if we
  could only run one.

HOW STRONG A PLACEBO IS IT? NOT VERY, AND SAYING SO IS PART OF THE JOB.

Across the 393 four-digit occupations all three measures score, our genAI
percentile correlates 0.74 with teleworkability and 0.87 with Eloundou.
At 0.74 a placebo cannot be expected to return nothing: a good deal of
any DAIOE result will reappear in the telework column for arithmetic
reasons alone. Reading "telework is also negative" as a refutation would
therefore be as wrong as reading "telework is smaller" as a vindication.

That is why this script also runs the HORSE RACE. Both measures enter one
regression, so the DAIOE coefficient is identified off the part of AI
exposure that teleworkability does not explain. That is the coefficient
worth quoting, and it is the one that can actually separate the two
stories.

HOW IT IS FAST. The panels are cached and identical across measures, so
only the exposure column changes. Three measures times two outcomes is
six fits, and the expensive pulls never repeat.

Output (output_63/):
  robustness_gradient.csv   age band x measure x outcome x dating
  horserace.csv             both measures in one regression
  measure_correlation.csv   how much the three measures agree at SSYK4
  63_summary.txt
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
OUT = HERE / "output_63"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
FAILURES = []

MEASURES = [
    ("daioe", "daioe_quartiles.dta", "pctl_rank_genai",
     "our own measure, the one every other design uses"),
    ("eloundou", "eloundou_ssyk4.dta", "eloundou_score",
     "a different team, a different method, the same object"),
    ("telework", "dingel_neiman_ssyk4.dta", "teleworkable",
     "NOT an AI measure: a placebo that should NOT reproduce the result"),
]


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_measure(fname: str, col: str) -> pd.DataFrame:
    d = pd.read_stata(str(Path(mc.SHARE) / fname))
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    d = d[["ssyk4", col]].rename(columns={col: "score"})
    d = d[d["score"].notna()]
    # standardise so the coefficients are comparable across measures: each
    # is then the effect of a one standard deviation more exposed
    # occupation on its own scale, which is the only basis on which a
    # teleworkability score and a genAI percentile can be set side by side
    mu, sd = d["score"].mean(), d["score"].std(ddof=0)
    d["score"] = (d["score"] - mu) / (sd if sd > 0 else 1.0)
    return d


# The launch is what every earlier estimate uses; January 2024 is where
# SCB's adoption data and scripts 60 and 61 put the treatment. Both are
# pre-specified here, and neither was chosen after seeing any output.
POST_DATES = (("launch", mc.CHATGPT_YM), ("adoption", "2024-01"))


def redate(bal: pd.DataFrame, l47, post_from: str) -> pd.DataFrame:
    """
    Move the treatment date on a panel that already exists.

    The panel, the fixed effects and the exposure do not depend on when
    the treatment is deemed to start; four interaction columns do. So a
    second dating costs one fit rather than a second panel build, which
    on these panels is the difference between forty minutes and ninety.
    """
    ym = bal["year_month"].astype(str)
    bal["post_gpt"] = (ym >= post_from).astype(int)
    bal["post_gpt_x_expo"] = bal["post_gpt"] * bal["expo_z"]
    for a in l47.AGES:
        bal[l47.age_term(a)] = (bal["post_gpt"] * bal["expo_z"]
                                * (bal["age_group"] == a).astype(int))
    return bal


def tele_term(age: str) -> str:
    """The telework counterpart of 47L's age-specific interaction."""
    return "t_" + age.replace("-", "_").replace("+", "p")


def horserace(bal: pd.DataFrame, tele: pd.DataFrame, l47, tag: str):
    """
    Both measures in one regression, on the panel 47L already built.

    Two specifications, because they answer different questions and the
    paper needs the second one.

      pooled  one coefficient per measure. Precise, but it averages over
              six age bands and the claim is about one of them, so a
              young-specific effect comes back diluted by roughly the
              share of the young in the panel.
      by age  one coefficient per measure PER BAND, directly comparable
              with the gradient tables above. This is where the claim
              lives. With two measures correlated 0.74 it is a demanding
              specification and the standard errors show it.

    Returns a long frame with a `spec` column, and the number of cells the
    two measures jointly score.
    """
    b = bal.merge(tele.rename(columns={"expo": "tele"})[
        ["employer_id", "age_group", "tele"]],
        on=["employer_id", "age_group"], how="inner")
    if b.empty:
        return pd.DataFrame(), 0
    cells = b[["employer_id", "age_group", "tele"]].drop_duplicates(
        ["employer_id", "age_group"])
    mu, sd = cells["tele"].mean(), cells["tele"].std(ddof=0)
    b["tele_z"] = (b["tele"] - mu) / (sd if sd > 0 else 1.0)
    b["post_rb_x_tele"] = b["post_rb"] * b["tele_z"]
    b["post_gpt_x_tele"] = b["post_gpt"] * b["tele_z"]
    tele_age = []
    for a in l47.AGES:
        c = tele_term(a)
        b[c] = b["post_gpt"] * b["tele_z"] * (b["age_group"] == a).astype(int)
        tele_age.append(c)
    specs = {
        "pooled": ["post_rb_x_expo", "post_rb_x_tele",
                   "post_gpt_x_expo", "post_gpt_x_tele"],
        "by_age": ["post_rb_x_expo", "post_rb_x_tele"]
                  + [l47.age_term(a) for a in l47.AGES] + tele_age,
    }
    out = []
    for name, terms in specs.items():
        r = mc.run_fepois_multi(b, OUT, tag=f"{tag}_{name}", terms=terms,
                                fes=l47.FES)
        if r.empty:
            print(f"    horse race {name}: FAILED")
            continue
        r = r.copy(); r["spec"] = name
        out.append(r)
    n = len(cells)
    del b
    gc.collect()
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), n


def main():
    mc.Tee(OUT / "63_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("63: DOES THE ANSWER DEPEND ON THE EXPOSURE MEASURE?")
    print("=" * 70)
    for k, f, c, why in MEASURES:
        print(f"  {k:<9} {why}")
    print("  Panels and fixed effects are held fixed; only the score changes.")
    print(mc.mem_line("  "))

    l47 = _mod("47L_age_baseline_exposure.py", "l47")
    s54 = _mod("54_hiring_flows.py", "s54")

    scores = {}
    for k, f, c, _ in MEASURES:
        d = opt(f"load {k}", load_measure, f, c)
        if d is not None:
            scores[k] = d
            print(f"  {k}: {len(d)} occupations scored")
    if "daioe" not in scores:
        raise RuntimeError("the reference measure did not load")

    # how far do the measures agree at all? A placebo that correlates 0.9
    # with the real thing is not much of a placebo, and the reader needs
    # to know which case they are in.
    cors = []
    ks = list(scores)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            m = scores[a].merge(scores[b], on="ssyk4", suffixes=("_a", "_b"))
            if len(m) > 10:
                cors.append({"a": a, "b": b, "n_occ": len(m),
                             "corr": float(np.corrcoef(m.score_a,
                                                       m.score_b)[0, 1])})
    if cors:
        pd.DataFrame(cors).to_csv(OUT / "measure_correlation.csv", index=False)
        print("\n  correlation between measures, across occupations:")
        for r in cors:
            print(f"    {r['a']:<9} vs {r['b']:<9} {r['corr']:+.3f} "
                  f"on {r['n_occ']} occupations")

    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first.")

    # The two outcome sources are loaded one at a time, not both at once.
    # Holding the stock counts and the hiring flows together while a panel
    # is being built is three large frames in memory for no reason: the
    # stock work finishes before the flow work starts.
    def load_counts():
        c = [x for x in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                         for y in l47.YEARS) if x is not None]
        return pd.concat(c, ignore_index=True) if c else None

    def load_flows():
        f = [x for x in (mc.read_cache(CACHE / f"flows_{y}.parquet",
                                       require=s54.FLOW_COLS)
                         for y in s54.YEARS) if x is not None]
        return pd.concat(f, ignore_index=True) if f else None

    # Exposures first, and all of them, so the horse race can reuse a panel
    # rather than build a second one. build_exposure is cheap: it touches the
    # 2019 baseline only, never the monthly counts.
    expos = {}
    for key in scores:
        e = opt(f"exposure {key}", l47.build_exposure, base, scores[key])
        if e is not None and not e.empty:
            expos[key] = e
            print(f"  {key}: {len(e):,} firm-age cells scored")

    # SEPARATIONS ARE HERE FOR A REASON. If hiring falls in exposed firms
    # and the stock does not, the difference has to come out of
    # separations, and a fall in both is reduced churn rather than reduced
    # employment. That is the first thing a referee will say, so it is
    # estimated rather than argued about. The flow panel is built once per
    # measure and both flow outcomes are fitted on it.
    SOURCES = (("stock", load_counts, (("stock", "n_emp"),)),
               ("flows", load_flows, (("hires", "n_hire"),
                                      ("seps", "n_sep"))))

    rows, hr_rows = [], []
    for sname, loader, outcomes in SOURCES:
        src = loader()
        if src is None:
            print(f"\n  {sname}: no cache found, skipped")
            continue
        print(f"\n  {sname}: {len(src):,} rows loaded")
        for key, e in expos.items():
            t1 = time.time()
            bal = (l47.build_panel(src, e) if sname == "stock"
                   else s54.build_panel(src, e))
            if bal is None or bal.empty:
                continue
            print(f"    {key} {sname}: panel {len(bal):,} rows "
                  f"({(time.time()-t1)/60:.1f} min to build)")
            for label, outcome in outcomes:
                bal["n_emp"] = bal[outcome]

                for dname, dfrom in POST_DATES:
                    t2 = time.time()
                    redate(bal, l47, dfrom)
                    tag = f"{key}_{label}_{dname}"
                    gr = opt(tag, l47.fit_gradient, bal, f"m63_{tag}")
                    if gr is None or gr.empty:
                        FAILURES.append(tag)
                        print(f"      {dname}: FAILED, recorded and skipped")
                    else:
                        gr = gr.copy()
                        gr["measure"], gr["outcome"] = key, label
                        gr["dating"], gr["post_from"] = dname, dfrom
                        rows.append(gr)
                        pd.concat(rows, ignore_index=True).to_csv(
                            OUT / "robustness_gradient.csv", index=False)
                        print(f"      {dname} (post from {dfrom}) "
                              f"[{(time.time()-t2)/60:.1f} min]")
                        for _, x in gr.iterrows():
                            t = x["coef"] / max(x["se"], 1e-12)
                            print(f"        {x['age_group']:<6} {x['coef']:+.4f} "
                                  f"(SE {x['se']:.4f}) t {t:+.2f}")

                    # the horse race rides on the panel we already have
                    if key != "daioe" or "telework" not in expos:
                        continue
                    got = opt(f"horse race {tag}", horserace, bal,
                              expos["telework"], l47, f"hr63_{tag}")
                    r, ncell = got if got is not None else (None, 0)
                    if r is None or r.empty:
                        FAILURES.append(f"horserace/{tag}")
                        continue
                    for _, x in r.iterrows():
                        hr_rows.append({"outcome": label, "dating": dname,
                                        "post_from": dfrom,
                                        "spec": str(x["spec"]),
                                        "term": x["term"],
                                        "coef": float(x["coef"]),
                                        "se": float(x["se"]),
                                        "n_obs": int(x["n_obs"]),
                                        "n_cells": ncell,
                                        "status": str(x.get("status", "ok"))})
                    pd.DataFrame(hr_rows).to_csv(OUT / "horserace.csv",
                                                 index=False)
                    print(f"      horse race, {label}, {dname}, both measures:")
                    for h in hr_rows:
                        if (h["outcome"] != label or h["dating"] != dname
                                or h["term"].startswith("post_rb")):
                            continue
                        print(f"        {h['spec']:<7} {h['term']:<18} "
                              f"{h['coef']:+.4f} (SE {h['se']:.4f}) t "
                              f"{h['coef']/max(h['se'],1e-12):+.2f}")
            del bal
            gc.collect()
        del src
        gc.collect()

    lines = ["DOES THE ANSWER DEPEND ON THE EXPOSURE MEASURE?", "=" * 52, ""]
    for k, f, c, why in MEASURES:
        lines.append(f"  {k:<9} {why}")
    lines.append("")
    if cors:
        lines += ["correlation across occupations:"]
        for r in cors:
            lines.append(f"  {r['a']:<9} vs {r['b']:<9} {r['corr']:+.3f}")
        lines.append("")
    if rows:
        G = pd.concat(rows, ignore_index=True)
        for dname, dfrom in POST_DATES:
            for outcome in sorted(G["outcome"].unique()):
                d = G[(G.outcome == outcome) & (G.dating == dname)]
                if d.empty:
                    continue
                piv = d.pivot_table(index="age_group", columns="measure",
                                    values="coef")
                order = [a for a in l47.AGES if a in piv.index]
                lines += [f"age gradient, outcome = {outcome}, treatment "
                          f"dated {dname} (post from {dfrom}):",
                          piv.reindex(order).round(4).to_string(), ""]
        lines += ["The launch rows are comparable with every earlier estimate",
                  "in this round. The adoption rows are where scripts 60 and",
                  "61 put the treatment, and they are the ones to read if the",
                  "question is whether the measure carries a real effect",
                  "rather than whether three measures agree about a null.", ""]
        if {"daioe", "telework"} <= set(G["measure"]):
            lines += ["THE PLACEBO TEST. If the telework column reproduces the",
                      "daioe column, we are measuring office work rather than",
                      "exposure to generative AI, and the interpretation fails",
                      "however clean the identification is. Read it against the",
                      "correlation above: at 0.74 the columns SHOULD resemble",
                      "each other, and only the horse race separates them.", ""]
    if hr_rows:
        H = pd.DataFrame(hr_rows)
        lines += ["HORSE RACE: both measures in one regression. The daioe row",
                  "is the effect of AI exposure holding teleworkability fixed,",
                  "which is the coefficient worth quoting.", ""]
        pooled = H[(H["spec"] == "pooled")
                   & H["term"].str.startswith("post_gpt")]
        lines.append("  pooled over age:")
        for _, h in pooled.iterrows():
            nm = "daioe" if h["term"].endswith("expo") else "telework"
            lines.append(f"    {h['outcome']:<6} {h['dating']:<9} {nm:<9} "
                         f"{h['coef']:+.4f} (SE {h['se']:.4f}) t "
                         f"{h['coef']/max(h['se'],1e-12):+.2f}")
        by = H[(H["spec"] == "by_age") & ~H["term"].str.startswith("post_rb")]
        if not by.empty:
            lines += ["", "  by age band, which is where the claim lives:"]
            for dname, _ in POST_DATES:
                for outcome in sorted(by["outcome"].unique()):
                    d = by[(by.outcome == outcome) & (by.dating == dname)]
                    for a in l47.AGES:
                        dd = d[d["term"] == l47.age_term(a)]
                        tt = d[d["term"] == tele_term(a)]
                        if dd.empty or tt.empty:
                            continue
                        lines.append(
                            f"    {outcome:<6} {dname:<9} {a:<6} daioe "
                            f"{float(dd['coef'].iloc[0]):+.4f} "
                            f"({float(dd['se'].iloc[0]):.4f})   telework "
                            f"{float(tt['coef'].iloc[0]):+.4f} "
                            f"({float(tt['se'].iloc[0]):.4f})")
        lines += ["", "The pooled row averages over six age bands, so a decline",
                  "concentrated in one band arrives divided by roughly six.",
                  "Compare the by-age rows with the gradient table, not with",
                  "the pooled row.", ""]
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing column is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. Every score is standardised, so a coefficient is the effect of",
        "     a one standard deviation more exposed occupation ON THAT",
        "     MEASURE'S OWN SCALE. Sizes are comparable only in that sense.",
        "  2. Agreement between daioe and eloundou is reassurance about the",
        "     measure, not about the design. They may share a common error.",
        "  3. Disagreement with telework is what we WANT, but the two",
        "     measures correlate 0.74 across occupations, so the telework",
        "     column being negative is not by itself bad news. The horse",
        "     race is what settles it.",
        "  4. The three measures score slightly different sets of",
        "     occupations, so the marginal columns rest on slightly",
        "     different samples of firm-age cells; the log gives the counts.",
        "     The horse race uses only cells both measures score, which is",
        "     the comparison that holds the sample fixed.",
        "  5. READ HIRES AND SEPARATIONS TOGETHER. Hiring falling in",
        "     exposed firms while the stock holds means separations fell",
        "     too, and that is reduced churn rather than reduced",
        "     employment. Only a fall in hiring WITHOUT a matching fall in",
        "     separations is a fall in jobs.",
        "  6. This still says nothing about whether DAIOE predicts actual",
        "     use of the technology. That needs a survey we do not hold.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "63_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n63 done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
47j_within_employer_triple.py: the within-employer age design, with
exposure placed on the employer so that no young worker is ever
classified.

QUESTION
Does the employment of young workers relative to their older colleagues
inside the same employer change more in exposed employers after
generative AI arrives? Exposure is measured on the employer, from the 2019
education mix of its incumbents aged 31 to 69, and the within-employer
variation comes from age, which is read from the birth year and cannot go
stale. No worker under 31 enters the exposure measure, and no occupation
code recorded after 2019 enters anything. This is the design the paper
reports; scripts 61, 68 and 75 re-estimate it with the treatment dated at
adoption and the calendar cycle removed.

DESIGN
Unit: employer by age band by month counts of employed persons, 2019 to
2023 here, for the young band under study (22-25 or 26-30) beside the
four incumbent bands 31-34, 35-40, 41-49 and 50-69. Exposure
(incumbent_exposure): the worker-weighted mean of the education score
(script 47h's OL_daioe or entrant score book) over the employer's
incumbents aged 31 to 69 in 2019; employers with fewer than five incumbent
person-months are not scored; quartile cut points are weighted by
incumbent employment, so the top quartile holds a quarter of incumbent
employment rather than a quarter of employers. Sample: an employer enters
if it holds the young band and at least one incumbent band; the panel is
balanced and zero-filled over the window; an employer-band cell that is
zero in every month is dropped, since the employer-by-age effect predicts
it exactly, and an employer left with one band is dropped with it.
Specification: Poisson pseudo-maximum likelihood on the counts with
PostRB x High x Young (from April 2022) and PostGPT x High x Young (from
December 2022), under employer-by-month, employer-by-age and
month-by-age effects, standard errors clustered by employer. The
employer-by-month effect absorbs every firm-level shock common to the
ages in the panel, including the firm-level exposure interaction itself.
The estimand is therefore the change in the young-to-older ratio inside
exposed employers relative to less exposed ones.

Both arms are estimated: with the 2019 incumbents scored from the
education register as it stood in 2019 (true) and as it stood in 2021 or
2022 (as-of), for each of the two score books. The read rule is script
47h's: an artefact below 0.05 at both truncations carries register
evidence.

INPUTS AND OUTPUTS
Reads the caches script 47h writes (edu_hr_weights_2019 to 2021 and
edu_hr_2019 to 2023) and, through 47h, the education key and the score
files; pulls nothing itself unless a cache is missing. Writes to
output_47j/: triple_estimates.csv (one row per design, arm, truncation
and young band), triple_quartile_sizes.csv (employers and incumbent
employment per quartile) and 47j_summary.txt.

IN THE PAPER
Section 2: the exposure construction (the 2019 education mix of
incumbents aged 31 to 69, at least five person-months, quartiles weighted
by incumbent employment) and the fixed effects and sample rules of
Equation (2); Online Appendix III.2, the quartile sizes. The
incumbent_exposure function, INCUMBENT_BANDS, YOUNG_BANDS and FES are
imported by every later register script. The coefficients in
triple_estimates.csv, dated at the launch on a panel ending in 2023, are
not quoted; the paper's estimates come from scripts 68 and 75 on the same
exposure and effects.
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


def _drop_dead_cells(bal: pd.DataFrame) -> pd.DataFrame:
    """
    Remove employer x age cells that are zero in EVERY month, and any
    employer left with fewer than two age bands.

    This changes no estimate. Under a Poisson with an employer x age fixed
    effect, a cell whose outcome is zero throughout has that effect at
    minus infinity and contributes nothing to any other parameter; fixest
    discards it internally. Doing it here instead means the rows never
    reach R.

    Zero-filling a balanced panel over five age bands creates a great many
    cells that are empty for the life of the panel, most often in the
    youngest band, and removing them before R sees them keeps the exchange
    file and the fit within the memory the job has.
    """
    alive = bal.groupby(["employer_id", "age_group"], observed=True)["n_emp"].transform("sum") > 0
    out = bal[alive]
    bands = out.groupby("employer_id", observed=True)["age_group"].transform("nunique")
    out = out[bands >= 2]
    dropped = len(bal) - len(out)
    if dropped:
        print(f"    dropped {dropped:,} of {len(bal):,} rows in cells that "
              f"are zero in every month ({dropped/len(bal):.0%}); this is "
              f"what fixest would discard internally")
    return out.copy()


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
    bal = _drop_dead_cells(bal)
    if bal.empty:
        return bal
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
        w = mc.read_cache(cf, require=h47.WEIGHT_COLS)
        if w is None:
            conn = conn or mc.connect()
            w = h47.pull_weights(y, conn)
            mc.write_cache(w, cf)
        counts[y] = w
        print(f"  weights {y}: {len(w):,} cells")
    book = h47.ScoreBook(counts, key, scores)
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

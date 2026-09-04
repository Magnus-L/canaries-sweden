#!/usr/bin/env python3
"""
47_edu_exposure.py -- T12/R2.3: education-based DAIOE exposure.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY (round 2).
  Standalone: submit this file itself to BatchClient; it does not need
  run_all_mona. Requires input/utb_grupp2_sun2020_niva3_inr4_nyckel.dta
  (hash-pinned below) beside daioe_quartiles.dta.
======================================================================

WHY. The occupation register lags (the editor's coverage objection);
education records do not. If the 22-25 decline is real, it must also be
visible when workers are classified by what they STUDIED rather than by
a possibly stale occupation code. Erik Engberg's recipe (4 Sep 2026):
average DAIOE over the occupations each education group's holders work
in, then run the paper's design on employer x education-exposure cells.

DESIGN DECISIONS, each argued in
correspondence/erik-2026-09-04/digest-and-implications.md and the
composition-year memo (4 Sep):

  1. COMPOSITION WEIGHTS FROM 2019, not the latest year. Individ_2019 is
     the first vintage carrying Sun2020Niva/Inr (zero nulls measured),
     the first all-4-digit SSYK year, the LAST year military and police
     carry occupation codes, and pre-COVID -- and it predates ChatGPT,
     so AI-driven shifts in where graduates go cannot leak into the
     measure (Koch's drift caveat; script 34's failure). 2021 weights
     are produced alongside as the robustness check.
  2. Employment-weighted mean of the DAIOE genAI percentile per
     utbildningsgrupp; groups binned into EMPLOYMENT-WEIGHTED quartiles
     (each bin ~25% of 2019 workers, mirroring how occupation quartiles
     partition workers rather than codes).
  3. Workers in the panel carry their MOST RECENT education record
     (own-year Individ through 2023; the 2023->22->21 cascade for
     2024-25). Education is far stickier than occupation, so this
     cascade is mild -- but the match rates are REPORTED per age x year,
     because script 34 silently kept ~47% of 22-25.
  4. The headline read is the EVENT-STUDY SHAPE and the single-break
     post-ChatGPT effect. The Riksbank/ChatGPT split is estimated for
     completeness but is NOT the headline: education cells mix
     destination occupations with different responses to the two
     shocks, which is what destroyed 34's timing decomposition.
  5. Missingness encodings differ by vintage (2019: NULL; 2021+: empty
     string). Every filter handles both. Codes are normalised
     (strip + lower) on BOTH sides of the key join.

Outputs (output_47/), all export-safe aggregates:
  edu_exposure_groups_2019.csv   utbildningsgrupp: n workers, mean DAIOE,
                                 quartile (min-cell >= 5 enforced)
  edu_exposure_groups_2021.csv   the robustness weights
  edu_match_rates.csv            per age x year: edu-record and key-match shares
  edu_poisson_pooled.csv         age x term (RB + GPT spec)
  edu_poisson_singlebreak.csv    age x post_gpt_x_high (headline spec)
  edu_poisson_es.csv             age x halfyear (headline shape)
  47_summary.txt
Caches (cache/): edu_weights_{2019,2021}.parquet, edu_year_{Y}.parquet
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_47"
OUT.mkdir(exist_ok=True)
mc.CACHE_DIR.mkdir(exist_ok=True)

KEY_PATH = mc.SHARE + r"\utb_grupp2_sun2020_niva3_inr4_nyckel.dta"
# Pinned to the file delivered by Erik Engberg 31 Aug 2026 and filed in
# data-notes/codelists/utbildningsgrupp-sun2020 (10,241 rows, one per SUN2020
# niva(3) x inr(4) cell). A different hash means a different key: stop.
KEY_SHA256 = "c760361ba21554951a0744ee00de2f02f22f2e021b87f0863d9ece049e786637"

WEIGHT_YEARS = (2019, 2021)      # primary, robustness
PANEL_YEARS = range(2019, 2026)
STEP1_MIN_CUMULATIVE = 5
AGES = list(mc.AGE_GROUPS)


# ----------------------------------------------------------------------
# Small pure helpers (unit-tested locally in test_dryrun.py)
# ----------------------------------------------------------------------

def norm_code(s: pd.Series) -> pd.Series:
    """One normalisation for every SUN code join: trim + lowercase, and
    both NULL and '' (the 2019 vs 2021+ encodings) become <NA>."""
    out = s.astype("string").str.strip().str.lower()
    return out.where(out.notna() & (out != ""), other=pd.NA)


def load_key() -> pd.DataFrame:
    import hashlib
    got = hashlib.sha256(Path(KEY_PATH).read_bytes()).hexdigest()
    if got != KEY_SHA256:
        raise RuntimeError(f"key hash mismatch: {got[:16]}... is not the "
                           f"delivered key {KEY_SHA256[:16]}...")
    key = pd.read_stata(KEY_PATH)
    key = key.rename(columns={"sun2020niva_3_kod": "niva",
                              "sun2020inr_4_kod": "inr",
                              "utb_grupp2": "grp"})
    key["niva"] = norm_code(key["niva"])
    key["inr"] = norm_code(key["inr"])
    assert not key.duplicated(["niva", "inr"]).any(), "key not unique on niva x inr"
    return key[["niva", "inr", "grp"]]


def build_weights(counts: pd.DataFrame, key: pd.DataFrame,
                  daioe_scores: pd.DataFrame) -> tuple:
    """
    counts: person counts per (niva, inr, ssyk4) from one Individ year,
    occupation-coded people only. Returns (group table, diagnostics dict).

    Economic content: a group's exposure is the exposure of the jobs its
    holders actually do, weighted by how many of them do each job.
    """
    n0 = counts["n"].sum()
    m = counts.merge(key, on=["niva", "inr"], how="left")
    matched = m["grp"].notna()
    m = m[matched]
    m = m.merge(daioe_scores, on="ssyk4", how="inner")   # drops unscored SSYK
    grp = (m.groupby("grp")
             .apply(lambda g: pd.Series({
                 "n_workers": g["n"].sum(),
                 "mean_daioe": np.average(g["pctl_rank_genai"], weights=g["n"]),
             }), include_groups=False)
             .reset_index())
    grp = grp[grp["n_workers"] >= 5].copy()              # export floor
    # Employment-weighted quartiles: rank groups by exposure, cut the
    # CUMULATIVE worker mass at 25/50/75 -- each bin is ~a quarter of
    # workers, not a quarter of the 105 group codes.
    grp = grp.sort_values("mean_daioe").reset_index(drop=True)
    cum = grp["n_workers"].cumsum() / grp["n_workers"].sum()
    grp["edu_quartile"] = np.searchsorted([0.25, 0.5, 0.75], cum, side="left") + 1
    diag = {"n_total": int(n0),
            "key_match_share": float(counts["n"][matched.values].sum() / n0),
            "n_groups": int(len(grp))}
    return grp, diag


def map_and_collapse(raw: pd.DataFrame, grp_q: pd.DataFrame) -> tuple:
    """
    One year of employer x month x (niva, inr) x age cells -> employer x
    month x edu_quartile x age, plus match accounting. grp_q maps
    utbildningsgrupp -> edu_quartile (from the 2019 weights).
    """
    raw = raw.copy()
    raw["niva"] = norm_code(raw["niva"])
    raw["inr"] = norm_code(raw["inr"])
    total = raw.groupby("age_group", observed=True)["n_emp"].sum()
    m = raw.merge(load_key(), on=["niva", "inr"], how="left")
    keyed = m[m["grp"].notna()].merge(grp_q, on="grp", how="inner")
    kept = keyed.groupby("age_group", observed=True)["n_emp"].sum()
    coll = (keyed.groupby(["employer_id", "year_month", "edu_quartile",
                           "age_group"], observed=True)["n_emp"]
            .sum().reset_index())
    rates = pd.DataFrame({"n_total": total, "n_mapped": kept}).reset_index()
    return coll, rates


# ----------------------------------------------------------------------
# SQL pulls
# ----------------------------------------------------------------------

def pull_weight_counts(year: int, conn) -> pd.DataFrame:
    """Person counts per (niva, inr, ssyk4) for one Individ year -- the
    composition the measure is built from. Aggregated in SQL: tiny result."""
    q = f"""
    SELECT Sun2020Niva AS niva, Sun2020Inr AS inr,
           RIGHT('0000' + CAST(Ssyk4_2012_J16 AS VARCHAR(4)), 4) AS ssyk4,
           COUNT(*) AS n
    FROM dbo.Individ_{year}
    WHERE Sun2020Niva IS NOT NULL AND LTRIM(Sun2020Niva) <> ''
      AND Sun2020Inr  IS NOT NULL AND LTRIM(Sun2020Inr)  <> ''
      AND Ssyk4_2012_J16 IS NOT NULL AND LTRIM(Ssyk4_2012_J16) <> ''
    GROUP BY Sun2020Niva, Sun2020Inr, Ssyk4_2012_J16
    """
    return pd.read_sql(q, conn)


def pull_edu_year(year: int, conn) -> pd.DataFrame:
    """One AGI year aggregated to employer x month x (niva, inr) x age.
    Education from the year's own Individ through 2023, else the
    2023->2022->2021 cascade (education is sticky; rates reported)."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    if year <= 2023:
        joins = (f"LEFT JOIN dbo.Individ_{year} e1 "
                 f"ON agi.P1207_LOPNR_PERSONNR = e1.P1207_LopNr_PersonNr")
        niva, inr, born = "e1.Sun2020Niva", "e1.Sun2020Inr", "e1.FodelseAr"
    else:
        joins = "\n".join(
            f"LEFT JOIN dbo.Individ_{y} e{i} "
            f"ON agi.P1207_LOPNR_PERSONNR = e{i}.P1207_LopNr_PersonNr"
            for i, y in enumerate((2023, 2022, 2021), 1))
        niva = "COALESCE(e1.Sun2020Niva, e2.Sun2020Niva, e3.Sun2020Niva)"
        inr = "COALESCE(e1.Sun2020Inr,  e2.Sun2020Inr,  e3.Sun2020Inr)"
        born = "COALESCE(e1.FodelseAr,  e2.FodelseAr,  e3.FodelseAr)"
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, {niva} AS niva, {inr} AS inr,
               {born} AS birth_year,
               agi.P1207_LOPNR_PERSONNR AS person_id
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        {joins}""" for m in range(1, max_month + 1))
    age_case = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""
    q = f"""
    WITH base AS ({monthly}),
    age_calc AS (
        SELECT employer_id, period, niva, inr, person_id,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           niva, inr, {age_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, niva, inr, {age_case}
    """
    return pd.read_sql(q, conn)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    mc.Tee(OUT / "47_log.txt")
    print("=" * 70)
    print("47: EDUCATION-BASED DAIOE EXPOSURE (T12/R2.3)")
    print("=" * 70)

    need_sql = (
        any(not (mc.CACHE_DIR / f"edu_weights_{y}.parquet").exists()
            for y in WEIGHT_YEARS)
        or any(not (mc.CACHE_DIR / f"edu_year_{y}.parquet").exists()
               for y in PANEL_YEARS))
    conn = mc.connect() if need_sql else None

    daioe_full = (pd.read_stata(mc.DAIOE_PATH)
                  if mc.DAIOE_PATH.endswith(".dta") else pd.read_csv(mc.DAIOE_PATH))
    daioe_full["ssyk4"] = daioe_full["ssyk4"].astype(str).str.zfill(4)
    scores = daioe_full[["ssyk4", "pctl_rank_genai"]]
    key = load_key()
    print(f"  key ok: {len(key):,} niva x inr cells, hash verified")

    # ---- Stage A: the measure, 2019 primary + 2021 robustness ----
    weights = {}
    for wy in WEIGHT_YEARS:
        cachef = mc.CACHE_DIR / f"edu_weights_{wy}.parquet"
        if cachef.exists():
            counts = pd.read_parquet(cachef)
        else:
            t0 = time.time()
            counts = pull_weight_counts(wy, conn)
            counts.to_parquet(cachef, index=False)
            print(f"  weights pull {wy}: {len(counts):,} cells ({time.time()-t0:.0f}s)")
        counts["niva"] = norm_code(counts["niva"])
        counts["inr"] = norm_code(counts["inr"])
        counts["ssyk4"] = counts["ssyk4"].astype(str).str.zfill(4)
        grp, diag = build_weights(counts, key, scores)
        grp.to_csv(OUT / f"edu_exposure_groups_{wy}.csv", index=False)
        weights[wy] = grp
        print(f"  {wy}: {diag['n_groups']} groups, key match "
              f"{diag['key_match_share']:.1%} of coded workers")

    grp_q = weights[2019][["grp", "edu_quartile"]].rename(columns={"grp": "grp"})
    # Pre-committed cross-check, printed not asserted: quartile agreement
    both = weights[2019].merge(weights[2021], on="grp", suffixes=("_19", "_21"))
    agree = (both["edu_quartile_19"] == both["edu_quartile_21"]).mean()
    print(f"  2019 vs 2021 quartile agreement: {agree:.1%} of groups")

    # ---- Stage B: the panel, streamed one year at a time (memory rule) ----
    frames, rate_rows = [], []
    for y in PANEL_YEARS:
        cachef = mc.CACHE_DIR / f"edu_year_{y}.parquet"
        if cachef.exists():
            raw = pd.read_parquet(cachef)
            print(f"  {y}: cached ({len(raw):,} cells)")
        else:
            t0 = time.time()
            raw = pull_edu_year(y, conn)
            raw.to_parquet(cachef, index=False)
            print(f"  {y}: {len(raw):,} cells ({time.time()-t0:.0f}s)")
        coll, rates = map_and_collapse(raw, grp_q)
        rates["year"] = y
        frames.append(coll)
        rate_rows.append(rates)
        del raw
    panel = pd.concat(frames, ignore_index=True)
    del frames
    rates = pd.concat(rate_rows, ignore_index=True)
    rates["mapped_share"] = rates["n_mapped"] / rates["n_total"]
    rates = mc.enforce_min_cell(rates, count_col="n_total")
    rates.to_csv(OUT / "edu_match_rates.csv", index=False)
    r2225 = rates[rates["age_group"] == "22-25"]
    print("  22-25 mapped share by year: "
          + ", ".join(f"{int(r.year)}: {r.mapped_share:.1%}"
                      for r in r2225.itertuples()))

    # ---- Stage C: estimation, mirroring 43 on education quartiles ----
    panel = panel.rename(columns={"edu_quartile": "exposure_quartile"})
    all_months = sorted(panel["year_month"].unique())
    pooled_rows, single_rows, es_frames = [], [], []
    for age in AGES:
        print(f"\n--- {age} ---")
        sub = panel[panel["age_group"] == age]
        cum = sub.groupby("employer_id")["n_emp"].sum()
        sub = sub[sub["employer_id"].isin(cum[cum >= STEP1_MIN_CUMULATIVE].index)]
        bal = mc.add_treatment(mc.balance_panel(sub, all_months))
        bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
        pres = mc.run_fepois(bal, OUT, tag=f"edu_pool_{age}")
        if not pres.empty:
            pres["age_group"] = age
            pooled_rows.append(pres)
        sres = mc.run_fepois_multi(bal, OUT, tag=f"edu_single_{age}",
                                   terms=["post_gpt_x_high"])
        if not sres.empty:
            sres["age_group"] = age
            single_rows.append(sres)
        eres = mc.run_fepois_es(bal, OUT, tag=f"edu_es_{age}")
        if not eres.empty:
            eres["age_group"] = age
            es_frames.append(eres)

    for rows, name in ((pooled_rows, "edu_poisson_pooled.csv"),
                       (single_rows, "edu_poisson_singlebreak.csv"),
                       (es_frames, "edu_poisson_es.csv")):
        if rows:
            pd.concat(rows).to_csv(OUT / name, index=False)

    with open(OUT / "47_summary.txt", "w") as f:
        f.write("47: education-based DAIOE exposure -- run complete\n")
        f.write(f"quartile agreement 2019 vs 2021: {agree:.1%}\n")
        f.write("HEADLINE READ: the event-study shape (edu_poisson_es.csv) and\n")
        f.write("the single-break post-ChatGPT effect. The RB/GPT split is for\n")
        f.write("completeness only -- see the design notes in the docstring.\n")
    print("\n47 done.")


if __name__ == "__main__":
    main()

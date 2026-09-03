#!/usr/bin/env python3
"""
45_asof_backtest.py -- T3/E5 centrepiece (MONA run M6): the as-of
backtest and the missingness x misclassification frontier.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY (stages 1-2).
  Stage 3 (the frontier) is pure arithmetic on exported aggregates and
  can be re-run locally on the exported CSVs.
======================================================================

THE IDEA
========
The editor's mechanical story is a hypothesis about what register
latency does to measured employment. On 2019-2023 the truth is
observable: every year has its own Individ table. So impose the 2024-25
staleness structure there and MEASURE the artefact instead of arguing
about it.

Stage 1 -- as-of panels. For truncation year T in {2021, 2022}: years
  <= T keep own-year codes; years > T get the cascade
  COALESCE(Individ_T, Individ_{T-1}, Individ_{T-2}) -- exactly the
  production rule 2024-25 lives under, shifted back. Alongside, the same
  years' TRUE own-year codes. Both assignments in one pull, so the
  confusion matrix (as-of quartile vs true quartile, by age group and
  year) falls out of the same query.

Stage 2 -- re-estimation. For each T: Poisson pooled DiD + half-year ES
  for 22-25 on (a) true codes and (b) as-of codes over 2019-2023, with a
  PSEUDO treatment date placed T+1 relative to the truncation the way
  ChatGPT (Dec 2022) sits relative to the 2023 register end. The
  DIFFERENCE (b) - (a) in the post coefficients is the measured
  artificial effect that latency alone generates. Report it against the
  production-window headline.

Stage 3 -- the frontier. Grid over (dm, mc):
    dm = extra nonmatch growth among true-Q4 young workers relative to
         Q1-Q3, 2024-25 (percentage points)
    mc = share of stale-coded Q4 workers misclassified into Q1-Q3
  Under the editor's mechanism, measured relative Q4 employment falls by
  approximately ln(1 - dm) + ln(1 - mc) log points with no true change.
  The frontier marks the (dm, mc) combinations that would generate the
  full headline coefficient; the backtest's measured confusion matrix
  and script 40's match-rate gaps place the calibrated point on the
  grid. Distance between the calibrated point and the frontier is the
  slack the response letter quotes.

NOTE (R1.14): the Facius-Iacono-style backdated-treatment placebo is
already in the submitted record -- script 25 ran Nov-2021 and Jul-2022
false dates. Cite those beside this backtest; no new run needed.

Output (output_45/):
  asof_confusion_T{T}.csv     true x as-of quartile counts, by age, year
  asof_matchrates_T{T}.csv    match rate by year x age under truncation
  asof_estimates.csv          pooled + endpoint: true vs as-of, per T
  asof_es_T{T}.csv            event studies, both assignments
  frontier_grid.csv           artificial-coefficient surface + calibration
  45_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_45"
OUT.mkdir(exist_ok=True)

TRUNCATIONS = (2021, 2022)
AGE = "22-25"
STEP1_MIN_CUMULATIVE = 5
WINDOW_YEARS = range(2019, 2024)          # backtest window: truth observable


# ----------------------------------------------------------------------
# Stage 1: dual-assignment pull
# ----------------------------------------------------------------------

def pull_year_dual(year, conn, trunc):
    """
    One year, aggregated to employer x ssyk_true x ssyk_asof x age x month.

    ssyk_true: the year's own Individ code (the truth; year <= 2023).
    ssyk_asof: the code the production cascade WOULD assign if the
    register ended at `trunc`: own code for year <= trunc, else
    COALESCE(Individ_trunc, Individ_{trunc-1}, Individ_{trunc-2}).
    Missing codes -> '____'.
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    if year <= trunc:
        asof_expr = "own.Ssyk4_2012_J16"
        asof_joins = ""
    else:
        asof_expr = (f"COALESCE(a1.Ssyk4_2012_J16, a2.Ssyk4_2012_J16, "
                     f"a3.Ssyk4_2012_J16)")
        asof_joins = f"""
            LEFT JOIN dbo.Individ_{trunc} a1
                ON agi.P1207_LOPNR_PERSONNR = a1.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_{trunc-1} a2
                ON agi.P1207_LOPNR_PERSONNR = a2.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_{trunc-2} a3
                ON agi.P1207_LOPNR_PERSONNR = a3.P1207_LopNr_PersonNr"""

    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period,
               own.Ssyk4_2012_J16 AS ssyk_true,
               {asof_expr} AS ssyk_asof,
               own.FodelseAr AS birth_year,
               agi.P1207_LOPNR_PERSONNR AS person_id
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_{year} own
            ON agi.P1207_LOPNR_PERSONNR = own.P1207_LopNr_PersonNr
        {asof_joins}"""
        for m in range(1, max_month + 1))

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
        SELECT employer_id, period,
               COALESCE(RIGHT('0000'+CAST(ssyk_true AS VARCHAR(4)),4),'____')
                   AS ssyk_true,
               COALESCE(RIGHT('0000'+CAST(ssyk_asof AS VARCHAR(4)),4),'____')
                   AS ssyk_asof,
               person_id,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4)+'-'+SUBSTRING(period,5,2) AS year_month,
           ssyk_true, ssyk_asof, {age_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, ssyk_true, ssyk_asof, {age_case}
    """
    return pd.read_sql(q, conn)


def get_dual_panel(conn, trunc):
    cache = OUT / f"panel_dual_T{trunc}.parquet"
    if cache.exists():
        print(f"  cached panel_dual_T{trunc}")
        return pd.read_parquet(cache)
    frames = []
    for y in WINDOW_YEARS:
        t0 = time.time()
        f = pull_year_dual(y, conn, trunc)
        print(f"  {y}: {len(f):,} cells ({time.time()-t0:.0f}s)")
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(cache, index=False)
    return panel


# ----------------------------------------------------------------------
# Stage 1b: diagnostics from the dual panel
# ----------------------------------------------------------------------

def confusion_and_matchrates(panel, daioe, trunc):
    d4 = daioe.rename(columns={"ssyk4": "ssyk_true",
                               "exposure_quartile": "q_true"})
    da = daioe.rename(columns={"ssyk4": "ssyk_asof",
                               "exposure_quartile": "q_asof"})
    p = panel.copy()
    p["year"] = p["year_month"].str[:4].astype(int)
    p = p.merge(d4, on="ssyk_true", how="left")
    p = p.merge(da, on="ssyk_asof", how="left")
    p["q_true"] = p["q_true"].fillna(0).astype(int)    # 0 = uncoded/unmatched
    p["q_asof"] = p["q_asof"].fillna(0).astype(int)

    conf = (p[p["year"] > trunc]
            .groupby(["year", "age_group", "q_true", "q_asof"],
                     observed=True)["n_emp"].sum().reset_index())
    conf = mc.enforce_min_cell(conf)
    conf.to_csv(OUT / f"asof_confusion_T{trunc}.csv", index=False)

    mrate = (p.assign(asof_coded=(p["ssyk_asof"] != "____").astype(int))
             .groupby(["year", "age_group"], observed=True)
             .apply(lambda g: pd.Series({
                 "n": g["n_emp"].sum(),
                 "asof_match": g.loc[g["asof_coded"] == 1, "n_emp"].sum()}))
             .reset_index())
    mrate["asof_match_rate"] = mrate["asof_match"] / mrate["n"]
    mrate.to_csv(OUT / f"asof_matchrates_T{trunc}.csv", index=False)
    print(f"  T{trunc}: confusion + match rates written")
    return conf, mrate


# ----------------------------------------------------------------------
# Stage 2: estimation under both assignments
# ----------------------------------------------------------------------

def estimate_both(panel, daioe, trunc):
    """Poisson pooled + ES for the headline age group, true vs as-of."""
    results = []
    es_frames = []
    # Pseudo-dates: place the pseudo-ChatGPT the same distance after the
    # truncation as Dec 2022 sits after the 2023 register end (i.e. minus
    # one year: production has codes THROUGH 2023 and treatment 2022-12;
    # the pseudo pair for T=2021 is treatment 2020-12? No: the artefact
    # window is AFTER the register ends. Production: codes end 2023,
    # artefact years 2024-25, treatment 2022-12 (pre-dates the coverage
    # break by 13 months). Backtest T=2021: artefact years 2022-23,
    # pseudo-treatment 2020-12. The pseudo event study's post window then
    # crosses into the artefact years exactly as the production one does.
    pseudo_gpt = f"{trunc - 1}-12"
    pseudo_rb = f"{trunc - 1}-04"

    for which in ("true", "asof"):
        col = f"ssyk_{which}"
        agg = (panel[panel[col] != "____"]
               .groupby(["employer_id", "year_month", col, "age_group"],
                        observed=True)["n_emp"].sum().reset_index()
               .rename(columns={col: "ssyk4"}))
        agg = mc.merge_daioe_and_filter(agg, daioe)
        agg = mc.aggregate_to_quartile(agg)
        months = sorted(agg["year_month"].unique())
        sub = agg[agg["age_group"] == AGE]
        cum = sub.groupby("employer_id")["n_emp"].sum()
        sub = sub[sub["employer_id"].isin(
            cum[cum >= STEP1_MIN_CUMULATIVE].index)]
        bal = mc.balance_panel(sub, months)
        bal["post_rb"] = (bal["year_month"] >= pseudo_rb).astype(int)
        bal["post_gpt"] = (bal["year_month"] >= pseudo_gpt).astype(int)
        bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
        bal["post_rb_x_high"] = bal["post_rb"] * bal["high"]
        bal["post_gpt_x_high"] = bal["post_gpt"] * bal["high"]
        bal["fe_emp_bin"] = (bal["employer_id"].astype(str) + "_"
                             + bal["exposure_quartile"].astype(str))
        bal["fe_emp_t"] = (bal["employer_id"].astype(str) + "_"
                           + bal["year_month"])
        bal["halfyear"] = mc.assign_halfyear(bal["year_month"])
        print(f"  [{which} T{trunc}] {len(bal):,} cells")

        pres = mc.run_fepois(bal, OUT, tag=f"bt_{which}_T{trunc}")
        g2 = pres.loc[pres["term"] == "post_gpt_x_high"]
        if len(g2):
            results.append({
                "trunc": trunc, "assignment": which,
                "gamma2": float(g2["coef"].iloc[0]),
                "se2": float(g2["se"].iloc[0]),
                "p2": float(g2["pvalue"].iloc[0]),
                "n_obs": int(g2["n_obs"].iloc[0])})
        # pseudo-ES with reference at the pseudo pre-treatment half-year
        ref = f"{trunc - 1}H1"
        eres = mc.run_fepois_es(bal, OUT, tag=f"bt_es_{which}_T{trunc}",
                                ref=ref)
        if not eres.empty:
            eres["trunc"] = trunc
            eres["assignment"] = which
            es_frames.append(eres)
    return results, es_frames


# ----------------------------------------------------------------------
# Stage 3: the frontier (pure arithmetic; also runnable locally)
# ----------------------------------------------------------------------

def frontier(headline_gamma2=-0.174, dm_grid=None, mc_grid=None,
             calib=None):
    """
    Artificial coefficient from the mechanical channel:
        beta_art(dm, mcl) = ln((1 - dm) * (1 - mcl))
    dm  : differential nonmatch growth in Q4 vs Q1-3 (share of true-Q4
          young workers additionally dropped post-2023)
    mcl : share of retained stale-coded true-Q4 young workers whose
          stale code places them OUTSIDE Q4
    The frontier is the contour beta_art = headline_gamma2. `calib`
    (dm_hat, mcl_hat) marks the empirically measured point from the
    backtest confusion matrix and script 40's match-rate gaps.
    """
    dm_grid = dm_grid if dm_grid is not None else np.arange(0, 0.201, 0.005)
    mc_grid = mc_grid if mc_grid is not None else np.arange(0, 0.201, 0.005)
    rows = []
    for dm in dm_grid:
        for mcl in mc_grid:
            beta = np.log((1 - dm) * (1 - mcl))
            rows.append({"dm": round(dm, 3), "mcl": round(mcl, 3),
                         "beta_artificial": beta,
                         "erases_headline": beta <= headline_gamma2})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "frontier_grid.csv", index=False)
    need = 1 - np.exp(headline_gamma2)
    lines = [
        "FRONTIER", "=" * 40,
        f"headline gamma2 (Poisson): {headline_gamma2:+.3f}",
        f"combined differential loss needed to erase it: {need:.1%}",
    ]
    if calib is not None:
        dm_hat, mcl_hat = calib
        beta_hat = np.log((1 - dm_hat) * (1 - mcl_hat))
        lines += [f"calibrated point: dm = {dm_hat:.3f}, "
                  f"mcl = {mcl_hat:.3f} -> beta_art = {beta_hat:+.4f}",
                  f"share of headline the calibrated mechanism explains: "
                  f"{beta_hat / headline_gamma2:.1%}"]
    (OUT / "45_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return grid


def main():
    mc.Tee(OUT / "45_log.txt")
    print("=" * 70)
    print("45: AS-OF BACKTEST + FRONTIER (E5 centrepiece)")
    print("=" * 70)
    # Connect only if a dual panel still has to be pulled. Both panels are
    # cached in output_45/, so a re-run after a mid-script failure (or after
    # a fix to the estimation stage) costs no SQL and needs no connection.
    need_pull = any(not (OUT / f"panel_dual_T{t}.parquet").exists()
                    for t in TRUNCATIONS)
    conn = mc.connect() if need_pull else None
    daioe = mc.load_daioe()

    all_est, all_es = [], []
    for trunc in TRUNCATIONS:
        print(f"\n=== truncation T = {trunc} ===")
        panel = get_dual_panel(conn, trunc)
        confusion_and_matchrates(panel, daioe, trunc)
        est, es = estimate_both(panel, daioe, trunc)
        all_est += est
        all_es += es

    est_df = pd.DataFrame(all_est)
    est_df.to_csv(OUT / "asof_estimates.csv", index=False)
    for trunc in TRUNCATIONS:
        sub = [e for e in all_es if not e.empty
               and e["trunc"].iloc[0] == trunc]
        if sub:
            pd.concat(sub).to_csv(OUT / f"asof_es_T{trunc}.csv", index=False)

    print("\nBACKTEST HEADLINE:")
    for trunc in TRUNCATIONS:
        t = est_df[(est_df["trunc"] == trunc)]
        try:
            g_true = t[t["assignment"] == "true"]["gamma2"].iloc[0]
            g_asof = t[t["assignment"] == "asof"]["gamma2"].iloc[0]
            print(f"  T={trunc}: true {g_true:+.4f}, as-of {g_asof:+.4f}, "
                  f"ARTEFACT = {g_asof - g_true:+.4f}")
        except IndexError:
            print(f"  T={trunc}: incomplete")

    # Frontier with default calibration = None; re-run frontier() locally
    # with (dm_hat, mcl_hat) read from the exported confusion/match files.
    frontier()


if __name__ == "__main__":
    main()

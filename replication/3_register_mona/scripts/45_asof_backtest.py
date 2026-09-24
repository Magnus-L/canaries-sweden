#!/usr/bin/env python3
"""
45_asof_backtest.py: the as-of backtest of the occupation-coding cascade.

QUESTION
The submitted version classified each worker by their own occupation code
from the annual register. That register is published with a lag, so the
2024 and 2025 records carry codes from 2023 or earlier and workers the
truncated register cannot code drop out. What does that staleness alone
do to the coefficient? On 2019 to 2023 the truth is observable, since
every year has its own register, so the staleness of 2024 and 2025 can be
imposed there and the artefact measured rather than argued about. This
is the test that withdrew the submitted employment design.

DESIGN
Stage 1: for each truncation year T in {2021, 2022}, one pull of the
monthly employer declarations for 2019 to 2023 with two occupation codes
per worker: the year's own Individ code (true) and the code the
production cascade would assign if the register ended at T (the own code
for years up to T; Individ_T, then T minus 1, then T minus 2 for later
years), aggregated to employer by true code by as-of code by age band by
month. From that panel: the confusion matrix of true against as-of
exposure quartile by age band and year, and the as-of match rate by year
and age band.

Stage 2: the submitted design on both assignments. Employer by exposure
quartile by month cells for ages 22-25, employers with a cumulative count
of at least five, balanced and zero-filled, restricted to employers in
both the top quartile and a lower one; Poisson pseudo-maximum likelihood
with PostRB x High and PostGPT x High under employer-by-quartile and
employer-by-month effects, standard errors clustered by employer; and the
half-year event study with the same effects. The pseudo-launch is placed
in December of the year before T and the pseudo-hike in April of that
year, so the launch sits thirteen months before the first month the
truncated register cannot code, as December 2022 sits before January 2024
in the production panel; the event-study reference is the first half of
T minus 1. The artefact is the as-of coefficient minus the true one.

Stage 3: a grid of the artificial coefficient ln((1 - dm)(1 - mc)) over
differential non-match growth dm and misclassification mc. Its default
comparison value is the withdrawn submitted coefficient; the grid and the
summary line it writes are not quoted.

INPUTS AND OUTPUTS
Reads, in MONA, Arb_AGIIndivid for 2019 to 2023 joined to Individ_2019 to
2023, and daioe_quartiles.dta; caches cache/panel_dual_T2021.parquet and
panel_dual_T2022.parquet, so a re-run after the pulls needs no
connection. Writes to output_45/: asof_confusion_T<T>.csv,
asof_matchrates_T<T>.csv, asof_estimates.csv, asof_es_T<T>.csv,
frontier_grid.csv and 45_summary.txt.

IN THE PAPER
Online Appendix IV.3: with true codes the submitted design returns +0.0193
(SE 0.0129) and with as-of codes -0.2875 at the 2021 truncation, an
artefact of -0.3068; +0.0176 to -0.1452 at the 2022 truncation, an
artefact of -0.1627; the as-of arm's cell count against the true arm's.
Table A27 and Figure A8 are built from asof_estimates.csv by the
exhibit builders in 4_exhibits/. Section 2 refers to Part IV for why no occupation code
after 2019 enters the reported design.
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
mc.CACHE_DIR.mkdir(exist_ok=True)

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
    cache = mc.CACHE_DIR / f"panel_dual_T{trunc}.parquet"
    cached = mc.read_cache(cache)
    if cached is not None:
        print(f"  cached panel_dual_T{trunc}")
        return cached
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
    # Pseudo-dates. In production the codes end in 2023, the artefact
    # years are 2024 and 2025, and the treatment is December 2022,
    # thirteen months before the first uncodable month. For truncation T
    # the artefact years are T+1 and T+2, so the pseudo-launch is December
    # of T-1 and the pseudo-hike April of T-1; the pseudo event study's
    # post window then crosses into the artefact years exactly as the
    # production one does.
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
    need_pull = any(not mc.cache_ok(mc.CACHE_DIR / f"panel_dual_T{t}.parquet")
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

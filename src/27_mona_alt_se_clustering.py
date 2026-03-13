#!/usr/bin/env python3
"""
27_mona_alt_se_clustering.py -- DiD with alternative SE clustering levels.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

PURPOSE:
  Re-estimate the main DiD with three different SE clustering levels:

    (a) employer x quartile  -- current baseline (cluster_entity=True)
    (b) employer              -- coarser, fewer clusters
    (c) SSYK 2-digit          -- occupation-level clustering

  The baseline specification clusters by employer x quartile (the entity
  in the panel). With millions of entity clusters, this could understate
  standard errors. Clustering at the employer level (coarser) or the
  occupation level (the treatment source) provides a robustness check.

  Cameron & Miller (2015) recommend clustering at the level of treatment
  assignment. Since DAIOE exposure varies at the occupation level, SSYK2
  clustering is a natural robustness check. Employer-level clustering
  addresses within-employer correlation across quartiles.

SPECIFICATIONS:
  DiD:  ln(n_emp+1) = a_{f,q} + b_{f,t} + g1*PostRB*High + g2*PostGPT*High

  FE: employer x quartile (entity) + employer x month (other_effects)
  Three clustering variants per age group.

PRACTICAL APPROACH:
  (a) Entity clustering: PanelOLS with cluster_entity=True (baseline)
  (b) Employer clustering: PanelOLS with clusters= employer_id variable.
      We extract employer_id from the fe_emp_q index by maintaining a
      mapping from entity to employer.
  (c) SSYK 2-digit clustering: We track the modal (most common) SSYK
      2-digit code within each employer x quartile cell. Since the
      panel is aggregated across occupations within a quartile, we
      assign each cell the SSYK2 that contributes most employment.
      This gives ~40 clusters (SSYK 2-digit codes), which is the
      most conservative level.

ESTIMATED RUNTIME:
  ~60-90 min (3 clustering variants x 6 age groups, but the PanelOLS
  estimation itself is the same -- only the SE computation differs).
  Optimisation: estimate the model ONCE per age group and recompute
  SEs with different cluster variables where possible. In practice,
  linearmodels requires re-fitting, so we run three separate fits.

OUTPUT FILES (in output_27/):
  1. alt_se_did_results.csv      -- DiD coefficients with all three clustering levels
  2. alt_se_comparison.txt       -- side-by-side comparison table
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import time
warnings.filterwarnings("ignore")


# ======================================================================
#   CONFIGURATION
# ======================================================================

# --- MONA SQL connection (same as scripts 14-18) ---
import pyodbc
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# --- DAIOE quartile file ---
DAIOE_PATH = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\daioe_quartiles.csv"

# --- Output directory ---
OUTPUT_DIR = Path("output_27")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Treatment dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Reference period (omitted dummy) ---
REF_HALFYEAR = "2022H1"

# --- Age group definitions (same as scripts 15 and 18) ---
AGE_GROUPS = {
    "22-25": (22, 25),
    "26-30": (26, 30),
    "31-34": (31, 34),
    "35-40": (35, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# --- Minimum employer size ---
MIN_EMPLOYER_SIZE = 5

# --- Colours ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"


# ======================================================================
#   STEP 1: LOAD DATA (same pipeline as scripts 14-18)
# ======================================================================

def pull_year(year, conn):
    """
    Pull one year of AGI data with cascading SSYK lookup (2023 -> 2022 -> 2021).
    Same approach as scripts 14 and 15 for data consistency.
    """
    print(f"  Processing year {year}...")

    individ_year = min(year, 2023)

    if year < 2025:
        suffix = "_def"
        max_month = 12
    else:
        suffix = "_prel"
        max_month = 6

    monthly_queries = []
    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"

        if individ_year >= 2023:
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    COALESCE(ind23.Ssyk4_2012_J16,
                             ind22.Ssyk4_2012_J16,
                             ind21.Ssyk4_2012_J16) AS ssyk4,
                    COALESCE(ind23.FodelseAr,
                             ind22.FodelseAr,
                             ind21.FodelseAr) AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_2023 ind23
                    ON agi.P1207_LOPNR_PERSONNR = ind23.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2022 ind22
                    ON agi.P1207_LOPNR_PERSONNR = ind22.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2021 ind21
                    ON agi.P1207_LOPNR_PERSONNR = ind21.P1207_LopNr_PersonNr
            """)
        else:
            monthly_queries.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    ind.FodelseAr AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
            """)

    union_query = "\nUNION ALL\n".join(monthly_queries)

    query = f"""
    WITH base AS (
        {union_query}
    ),
    age_calc AS (
        SELECT
            employer_id,
            period,
            RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4) AS ssyk4,
            person_id,
            CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
    )
    SELECT
        employer_id,
        LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
        ssyk4,
        CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL
        END AS age_group,
        COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc
    WHERE age BETWEEN 22 AND 69
    GROUP BY
        employer_id,
        period,
        ssyk4,
        CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL
        END
    """
    return pd.read_sql(query, conn)


def load_and_prepare():
    """
    Load AGI data year by year (same pipeline as scripts 14-18),
    merge DAIOE quartiles, assign age groups,
    aggregate to employer x quartile x age_group x month cells.

    ADDITION FOR SCRIPT 27: Before aggregating across ssyk4, we compute
    the modal SSYK 2-digit code for each employer x quartile cell. This
    is stored as a mapping and used for occupation-level clustering.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("=" * 70)

    # Pull year by year (consistent with scripts 14 and 15)
    frames = []
    t0 = time.time()
    for year in range(2019, 2026):
        frames.append(pull_year(year, conn))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

    # Drop rows without SSYK or age group
    df = df[df["ssyk4"].notna() & (df["ssyk4"] != "None")].copy()
    df = df[df["age_group"].notna()].copy()

    # SSYK4 as string
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # Merge DAIOE quartiles
    print("  Merging DAIOE quartiles...")
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    if daioe["exposure_quartile"].dtype == object:
        q_map = {"Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4}
        daioe["exposure_quartile"] = daioe["exposure_quartile"].map(q_map)
    df = df.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After DAIOE merge: {len(df):,} records")

    # Rename for consistency with downstream code
    df = df.rename(columns={"n_emp": "person_count"})

    # Filter small employers
    emp_size = df.groupby("employer_id")["person_count"].sum()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers with >={MIN_EMPLOYER_SIZE} workers: "
          f"{df['employer_id'].nunique():,}")

    # --- SCRIPT 27 ADDITION: compute modal SSYK2 per employer x quartile ---
    # This is needed for occupation-level clustering. We compute it BEFORE
    # aggregating across ssyk4.
    print("  Computing modal SSYK 2-digit per employer x quartile...")
    df["ssyk2"] = df["ssyk4"].str[:2]

    # Total employment by employer x quartile x ssyk2 (across all time)
    modal_ssyk2 = (
        df.groupby(["employer_id", "exposure_quartile", "ssyk2"])
        ["person_count"]
        .sum()
        .reset_index()
    )
    # Keep the ssyk2 with the most employment for each employer x quartile
    modal_ssyk2 = (
        modal_ssyk2.sort_values("person_count", ascending=False)
        .drop_duplicates(subset=["employer_id", "exposure_quartile"], keep="first")
        [["employer_id", "exposure_quartile", "ssyk2"]]
    )
    print(f"  Unique SSYK 2-digit codes in modal mapping: "
          f"{modal_ssyk2['ssyk2'].nunique()}")

    # --- Standard aggregation ---
    print("  Aggregating to employer x quartile x age_group x month...")
    agg = (
        df.groupby(["employer_id", "exposure_quartile", "age_group", "year_month"])
        ["person_count"]
        .sum()
        .reset_index()
        .rename(columns={"person_count": "n_emp"})
    )
    print(f"  Panel cells: {len(agg):,}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")

    return agg, modal_ssyk2


# ======================================================================
#   STEP 1b: BALANCED PANEL (ZERO-FILL)
# ======================================================================

def balance_panel_for_age(agg, age_label):
    """
    Build balanced panel for one age group: every (employer, quartile)
    combination the employer is ever observed in x all months.
    Missing cells filled with n_emp = 0.

    Applies identification restriction: employers with both Q4 and Q1-Q3.

    WHY: Without zero-filling, firms that shed ALL workers of an age group
    in a quartile disappear from the data. This drops the strongest
    treatment cases and biases the DiD toward zero.
    """
    sub = agg[agg["age_group"] == age_label].copy()
    all_months = sorted(agg["year_month"].unique())

    # Unique employer-quartile pairs for this age group
    emp_q = (
        sub.groupby(["employer_id", "exposure_quartile"])
        .size()
        .reset_index()[["employer_id", "exposure_quartile"]]
    )

    # Identification restriction: employers spanning Q4 and Q1-Q3
    q4_emps = set(emp_q.loc[emp_q["exposure_quartile"] == 4, "employer_id"])
    low_emps = set(emp_q.loc[emp_q["exposure_quartile"] < 4, "employer_id"])
    valid_emps = q4_emps & low_emps
    emp_q = emp_q[emp_q["employer_id"].isin(valid_emps)]

    # Cross join: employer-quartile pairs x all months
    months_df = pd.DataFrame({"year_month": all_months})
    emp_q["_k"] = 1
    months_df["_k"] = 1
    balanced = emp_q.merge(months_df, on="_k").drop(columns="_k")

    # Left join actual employment counts
    balanced = balanced.merge(
        sub[["employer_id", "exposure_quartile", "year_month", "n_emp"]],
        on=["employer_id", "exposure_quartile", "year_month"],
        how="left",
    )
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)
    balanced["age_group"] = age_label

    n_zeros = (balanced["n_emp"] == 0).sum()
    pct_zero = 100 * n_zeros / len(balanced) if len(balanced) > 0 else 0
    print(f"  Balanced panel ({age_label}): {len(balanced):,} cells "
          f"({n_zeros:,} zeros = {pct_zero:.1f}%)")
    print(f"  Employers with both Q4 and Q1-Q3: {len(valid_emps):,}")

    return balanced


# ======================================================================
#   STEP 2: DiD WITH ALTERNATIVE CLUSTERING
# ======================================================================

def run_did_alt_clustering(agg, modal_ssyk2):
    """
    For each age group, estimate the main DiD three times with different
    SE clustering:

      (a) cluster_entity=True     -- employer x quartile (baseline)
      (b) clusters=employer_id    -- employer level
      (c) clusters=ssyk2          -- SSYK 2-digit occupation level

    The point estimates are identical across (a)-(c) because the model
    specification is the same -- only the variance-covariance matrix
    differs. The question is whether the SE (and hence significance)
    changes with coarser clustering.

    For (b), we extract employer_id from the panel and pass it as the
    cluster variable. For (c), we merge the modal SSYK2 computed during
    data loading.
    """
    from linearmodels.panel import PanelOLS

    print("\n" + "=" * 70)
    print("STEP 2: DiD with alternative SE clustering")
    print("=" * 70)

    all_results = []

    for age_label in AGE_GROUPS:
        print(f"\n{'=' * 50}")
        print(f"Age group: {age_label}")
        print(f"{'=' * 50}")

        # Build balanced panel
        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        # Create treatment variables
        sub["post_rb"] = (sub["year_month"] >= RIKSBANK_YM).astype(int)
        sub["post_gpt"] = (sub["year_month"] >= CHATGPT_YM).astype(int)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["post_rb_x_high"] = sub["post_rb"] * sub["high"]
        sub["post_gpt_x_high"] = sub["post_gpt"] * sub["high"]

        # Log outcome
        sub["ln_emp"] = np.log(sub["n_emp"] + 1)

        # FE identifiers
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        # Merge modal SSYK2 for occupation-level clustering
        sub = sub.merge(
            modal_ssyk2,
            on=["employer_id", "exposure_quartile"],
            how="left",
        )
        # For any cells without a modal ssyk2 (should be rare after
        # balanced-panel construction), assign a placeholder
        sub["ssyk2"] = sub["ssyk2"].fillna("99")

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Unique SSYK2 clusters: {sub['ssyk2'].nunique()}")
        print(f"  Entity (emp x Q) clusters: {sub['fe_emp_q'].nunique():,}")

        # --- Prepare panel for PanelOLS ---
        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")
        panel = panel.set_index(["fe_emp_q", "date"])

        other_fe = pd.DataFrame(
            {"fe_emp_t": panel["fe_emp_t"]},
            index=panel.index,
        )

        # ----- (a) Entity clustering (baseline) -----
        print(f"\n  (a) Entity clustering (employer x quartile)...")
        t0 = time.time()
        try:
            mod = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
                entity_effects=True,
                time_effects=False,
                other_effects=other_fe,
            )
            res_entity = mod.fit(cov_type="clustered", cluster_entity=True)

            elapsed = time.time() - t0
            g1 = res_entity.params["post_rb_x_high"]
            g2 = res_entity.params["post_gpt_x_high"]
            se1 = res_entity.std_errors["post_rb_x_high"]
            se2 = res_entity.std_errors["post_gpt_x_high"]
            p1 = res_entity.pvalues["post_rb_x_high"]
            p2 = res_entity.pvalues["post_gpt_x_high"]

            print(f"      [{elapsed:.0f}s] g1={g1:+.4f} (SE={se1:.4f}), "
                  f"g2={g2:+.4f} (SE={se2:.4f})")

            all_results.append({
                "age_group": age_label,
                "clustering": "entity (emp x Q)",
                "n_clusters": int(sub["fe_emp_q"].nunique()),
                "n_obs": int(res_entity.nobs),
                "gamma1_rb_high": g1,
                "se1": se1,
                "pval1": p1,
                "gamma2_gpt_high": g2,
                "se2": se2,
                "pval2": p2,
            })

        except Exception as e:
            print(f"      ERROR: {e}")

        # ----- (b) Employer-level clustering -----
        print(f"  (b) Employer-level clustering...")
        t0 = time.time()
        try:
            # We need a cluster variable aligned with the panel index.
            # Extract employer_id from the fe_emp_q index (first element
            # of the MultiIndex) by splitting on "_".
            cluster_emp = panel.index.get_level_values(0).str.rsplit("_", n=1).str[0]
            cluster_emp = pd.Categorical(cluster_emp)

            mod_b = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
                entity_effects=True,
                time_effects=False,
                other_effects=other_fe,
            )
            res_emp = mod_b.fit(
                cov_type="clustered",
                cluster_entity=False,
                clusters=cluster_emp,
            )

            elapsed = time.time() - t0
            g1 = res_emp.params["post_rb_x_high"]
            g2 = res_emp.params["post_gpt_x_high"]
            se1 = res_emp.std_errors["post_rb_x_high"]
            se2 = res_emp.std_errors["post_gpt_x_high"]
            p1 = res_emp.pvalues["post_rb_x_high"]
            p2 = res_emp.pvalues["post_gpt_x_high"]

            n_emp_clusters = len(set(cluster_emp))
            print(f"      [{elapsed:.0f}s] g1={g1:+.4f} (SE={se1:.4f}), "
                  f"g2={g2:+.4f} (SE={se2:.4f})")
            print(f"      Employer clusters: {n_emp_clusters:,}")

            all_results.append({
                "age_group": age_label,
                "clustering": "employer",
                "n_clusters": n_emp_clusters,
                "n_obs": int(res_emp.nobs),
                "gamma1_rb_high": g1,
                "se1": se1,
                "pval1": p1,
                "gamma2_gpt_high": g2,
                "se2": se2,
                "pval2": p2,
            })

        except Exception as e:
            print(f"      ERROR: {e}")

        # ----- (c) SSYK 2-digit clustering -----
        print(f"  (c) SSYK 2-digit clustering...")
        t0 = time.time()
        try:
            # ssyk2 is already in the panel (merged before set_index)
            cluster_ssyk2 = pd.Categorical(panel["ssyk2"])

            mod_c = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
                entity_effects=True,
                time_effects=False,
                other_effects=other_fe,
            )
            res_ssyk2 = mod_c.fit(
                cov_type="clustered",
                cluster_entity=False,
                clusters=cluster_ssyk2,
            )

            elapsed = time.time() - t0
            g1 = res_ssyk2.params["post_rb_x_high"]
            g2 = res_ssyk2.params["post_gpt_x_high"]
            se1 = res_ssyk2.std_errors["post_rb_x_high"]
            se2 = res_ssyk2.std_errors["post_gpt_x_high"]
            p1 = res_ssyk2.pvalues["post_rb_x_high"]
            p2 = res_ssyk2.pvalues["post_gpt_x_high"]

            n_ssyk2_clusters = len(set(cluster_ssyk2))
            print(f"      [{elapsed:.0f}s] g1={g1:+.4f} (SE={se1:.4f}), "
                  f"g2={g2:+.4f} (SE={se2:.4f})")
            print(f"      SSYK2 clusters: {n_ssyk2_clusters}")

            all_results.append({
                "age_group": age_label,
                "clustering": "SSYK 2-digit",
                "n_clusters": n_ssyk2_clusters,
                "n_obs": int(res_ssyk2.nobs),
                "gamma1_rb_high": g1,
                "se1": se1,
                "pval1": p1,
                "gamma2_gpt_high": g2,
                "se2": se2,
                "pval2": p2,
            })

        except Exception as e:
            print(f"      ERROR: {e}")

    return all_results


# ======================================================================
#   STEP 3: COMPARISON OUTPUT
# ======================================================================

def format_comparison(results):
    """
    Format a side-by-side comparison of the three clustering levels,
    highlighting where significance changes.
    """
    lines = [
        "=" * 80,
        "ALTERNATIVE SE CLUSTERING: SIDE-BY-SIDE COMPARISON",
        "=" * 80,
        "",
        "Specification: ln(n_emp+1) = a_{f,q} + b_{f,t} + g1*PostRB*High + g2*PostGPT*High",
        "FE: employer x quartile (entity) + employer x month (other_effects)",
        "Point estimates are identical; only SEs and p-values change.",
        "",
        f"{'Age':>6s}  {'Clustering':<22s}  {'#Clust':>8s}  "
        f"{'g1':>8s}  {'SE(g1)':>8s}  {'p(g1)':>8s}  "
        f"{'g2':>8s}  {'SE(g2)':>8s}  {'p(g2)':>8s}",
        "-" * 100,
    ]

    df = pd.DataFrame(results)

    for age_label in AGE_GROUPS:
        age_rows = df[df["age_group"] == age_label]
        if age_rows.empty:
            continue

        for _, row in age_rows.iterrows():
            # Significance stars for g2
            p2 = row["pval2"]
            sig2 = "***" if p2 < 0.01 else "**" if p2 < 0.05 \
                   else "*" if p2 < 0.10 else ""

            lines.append(
                f"{row['age_group']:>6s}  {row['clustering']:<22s}  "
                f"{row['n_clusters']:>8,d}  "
                f"{row['gamma1_rb_high']:>+8.4f}  ({row['se1']:.4f})  "
                f"{row['pval1']:>8.4f}  "
                f"{row['gamma2_gpt_high']:>+8.4f}{sig2:<3s}  ({row['se2']:.4f})  "
                f"{row['pval2']:>8.4f}"
            )

        lines.append("")  # blank line between age groups

    # Interpretation
    lines.extend([
        "=" * 80,
        "INTERPRETATION GUIDE:",
        "=" * 80,
        "",
        "If SEs increase substantially with coarser clustering (employer or SSYK2),",
        "the baseline entity-level SEs may understate uncertainty. Key question:",
        "does g2 for 22-25 remain significant at conventional levels?",
        "",
        "Entity clusters (emp x Q): ~millions. Most precise SEs but may",
        "  understate uncertainty if errors are correlated within employers.",
        "",
        "Employer clusters: fewer (one per employer). Absorbs within-employer",
        "  correlation across quartiles. A natural robustness check.",
        "",
        "SSYK 2-digit clusters: ~40. Most conservative. Treatment (DAIOE",
        "  exposure) varies at the occupation level, so this is the cluster",
        "  level recommended by Cameron & Miller (2015) for treatment-level",
        "  inference. CAVEAT: with only ~40 clusters, the cluster-robust",
        "  variance estimator may be unreliable (rule of thumb: G >= 50).",
        "  Wild cluster bootstrap would be the gold standard here, but",
        "  is computationally infeasible with this panel size.",
    ])

    return "\n".join(lines)


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("27_mona_alt_se_clustering.py")
    print("DiD with alternative SE clustering levels")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load data
    agg, modal_ssyk2 = load_and_prepare()

    # Step 2: Run DiD with three clustering variants
    results = run_did_alt_clustering(agg, modal_ssyk2)

    # Step 3: Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / "alt_se_did_results.csv", index=False)
        print(f"\n  Saved -> alt_se_did_results.csv")

        # Comparison table
        comparison = format_comparison(results)
        (OUTPUT_DIR / "alt_se_comparison.txt").write_text(comparison)
        print(f"  Saved -> alt_se_comparison.txt")
        print(f"\n{comparison}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
    print(f"  All output in: {OUTPUT_DIR}/")

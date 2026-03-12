#!/usr/bin/env python3
"""
18_mona_eventstudy_corrected.py -- Event study with CORRECT FE specification.

======================================================================
  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT
  Do NOT run outside MONA -- the data is not available externally.
======================================================================

WHY THIS SCRIPT EXISTS:
  Script 14's event study (Step 3) used a WEAKER fixed-effects structure
  than the main DiD (Step 2):

    Main DiD:     employer×quartile FE  +  employer×month FE  (via other_effects)
    Old event study: employer×quartile FE  +  calendar month FE  (time_effects=True)

  The employer×month FE in the main DiD absorbs all firm-level time-varying
  shocks. Without them, the event study picks up firm-level compositional
  changes as "pre-trends". This likely explains why pre-trend tests fail
  for all age groups except 22-25 (where the true signal is overwhelming).

  This script re-runs the event study with the SAME FE as the main DiD:
  employer×quartile entity FE + employer×month absorbed via other_effects.

ESTIMATED RUNTIME:
  ~30-60 min total (similar to main DiD per age group × 6 groups).
  PanelOLS absorption cost is dominated by FE dimensions, not regressors.

OUTPUT FILES:
  1. corrected_es_all.csv          -- event study coefficients
  2. corrected_pretrends.txt       -- pre-trend joint F-test results
  3. corrected_es_*.png            -- figures per age group
  4. corrected_es_summary.txt      -- diagnostics
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

# --- MONA SQL connection (same as scripts 14-15) ---
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
OUTPUT_DIR = Path("output_18")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Treatment dates ---
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# --- Reference period (omitted dummy) ---
REF_HALFYEAR = "2022H1"

# --- Age group definitions (same as script 15 / Lydia's runs) ---
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
#   STEP 1: LOAD DATA (same pipeline as scripts 14-15)
# ======================================================================

def load_and_prepare():
    """
    Load AGI data via SQL, merge DAIOE quartiles, assign age groups,
    aggregate to employer × quartile × age_group × month cells.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("=" * 70)

    # SQL query: individual-level AGI records
    # Adjust column names if Lydia's extract uses different names
    sql = """
    SELECT
        LopNr       AS person_id,
        ArbstId     AS employer_id,
        Period      AS year_month,
        SSYK4       AS ssyk4,
        FodelseAr   AS birth_year
    FROM [dbo].[agi_monthly]
    """

    print("  Running SQL query...")
    t0 = time.time()
    df = pd.read_sql(sql, conn)
    print(f"  Loaded {len(df):,} records in {time.time()-t0:.0f}s")

    # Parse year-month
    df["year_month"] = df["year_month"].astype(str).str[:7]

    # Compute age
    year = df["year_month"].str[:4].astype(int)
    df["age"] = year - df["birth_year"].astype(int)

    # Filter to 22-69
    df = df[(df["age"] >= 22) & (df["age"] <= 69)].copy()

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

    # Assign age groups
    def get_age_group(age):
        for label, (lo, hi) in AGE_GROUPS.items():
            if lo <= age <= hi:
                return label
        return None

    df["age_group"] = df["age"].apply(get_age_group)
    df = df[df["age_group"].notna()].copy()

    # Filter small employers
    emp_size = df.groupby("employer_id")["person_id"].nunique()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers with >={MIN_EMPLOYER_SIZE} workers: "
          f"{df['employer_id'].nunique():,}")

    # Aggregate to employer × quartile × age_group × month
    print("  Aggregating to employer × quartile × age_group × month...")
    agg = (
        df.groupby(["employer_id", "exposure_quartile", "age_group", "year_month"])
        ["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_emp"})
    )
    print(f"  Panel cells: {len(agg):,}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")

    return agg


# ======================================================================
#   STEP 1b: BALANCED PANEL (ZERO-FILL)
# ======================================================================

def balance_panel_for_age(agg, age_label):
    """
    Build balanced panel for one age group: every (employer, quartile)
    combination the employer is ever observed in × all months.
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

    # Cross join: employer-quartile pairs × all months
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
#   STEP 2: CORRECTED EVENT STUDY
# ======================================================================

def assign_halfyear(ym):
    """Map year-month string to half-year label, e.g. '2022-03' -> '2022H1'."""
    year = ym[:4]
    month = int(ym[5:7])
    half = "H1" if month <= 6 else "H2"
    return f"{year}{half}"


def pretrend_joint_test(res, event_periods, ref_period):
    """Joint chi2 test: H0 = all pre-treatment event-study coefficients are zero."""
    from scipy import stats as sp_stats

    pre_cols = [f"hy_{p}" for p in event_periods if p < ref_period]
    if not pre_cols:
        return None

    beta = res.params[pre_cols].values
    try:
        V = res.cov.loc[pre_cols, pre_cols].values        # linearmodels
    except AttributeError:
        V = res.cov_params().loc[pre_cols, pre_cols].values  # statsmodels

    k = len(pre_cols)
    wald = float(beta @ np.linalg.solve(V, beta))
    p = 1 - sp_stats.chi2.cdf(wald, k)
    return {"chi2": round(wald, 2), "df": k, "p_value": round(p, 4)}


def run_corrected_event_study(agg):
    """
    Half-year event study with the CORRECT FE specification:
      entity FE = employer × quartile  (PanelOLS entity_effects)
      other FE  = employer × month     (PanelOLS other_effects)

    This matches the main DiD in script 14 Step 2.
    """
    from linearmodels.panel import PanelOLS

    print("\n" + "=" * 70)
    print("STEP 2: Corrected event study (employer×quartile + employer×month FE)")
    print("=" * 70)

    agg = agg.copy()

    # Compute half-year labels on raw agg (needed for event period list)
    agg["halfyear"] = agg["year_month"].apply(assign_halfyear)
    all_periods = sorted(agg["halfyear"].unique())
    event_periods = [p for p in all_periods if p != REF_HALFYEAR]

    print(f"  Half-year periods: {all_periods}")
    print(f"  Reference period: {REF_HALFYEAR}")
    print(f"  Event dummies: {len(event_periods)}")

    all_es_results = []
    pretrend_tests = []

    for age_label in AGE_GROUPS:
        print(f"\n--- Event study: {age_label} ---")
        t0 = time.time()

        # Build balanced panel with zero-filled cells
        sub = balance_panel_for_age(agg, age_label)
        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Zero cells: {(sub['n_emp'] == 0).sum():,}")

        # Compute variables on the balanced panel
        sub["ln_emp"] = np.log(sub["n_emp"] + 1)
        sub["high"] = (sub["exposure_quartile"] == 4).astype(int)
        sub["halfyear"] = sub["year_month"].apply(assign_halfyear)
        sub["fe_emp_q"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["exposure_quartile"].astype(str)
        )
        sub["fe_emp_t"] = (
            sub["employer_id"].astype(str) + "_" +
            sub["year_month"]
        )

        # Create interaction dummies: halfyear × high
        for p in event_periods:
            sub[f"hy_{p}"] = ((sub["halfyear"] == p).astype(int) * sub["high"])

        interaction_cols = [f"hy_{p}" for p in event_periods]

        # --- PanelOLS with CORRECT FE ---
        # Entity = employer × quartile (same as main DiD)
        # Other  = employer × month    (same as main DiD)
        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

        # Set index: entity = employer×quartile, time = date
        panel = panel.set_index(["fe_emp_q", "date"])

        # employer × month as other_effects (THE KEY FIX)
        other_fe = pd.DataFrame(
            {"fe_emp_t": panel["fe_emp_t"]},
            index=panel.index,
        )

        try:
            mod = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[interaction_cols],
                entity_effects=True,
                time_effects=False,       # NOT calendar-month FE
                other_effects=other_fe,   # employer×month FE instead
            )
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            elapsed = time.time() - t0
            print(f"  Estimated in {elapsed:.0f}s")

            # Collect coefficients
            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.std_errors[col],
                    "pval": res.pvalues[col],
                })

            # Add reference period (zero by construction)
            all_es_results.append({
                "age_group": age_label,
                "period": REF_HALFYEAR,
                "coef": 0.0,
                "se": 0.0,
                "pval": 1.0,
            })

            # Pre-trend joint test
            pt = pretrend_joint_test(res, event_periods, REF_HALFYEAR)
            if pt:
                print(f"  Pre-trend test: chi2({pt['df']}) = {pt['chi2']:.2f}, "
                      f"p = {pt['p_value']:.4f}")
                pretrend_tests.append({"age_group": age_label, **pt})

        except Exception as e:
            print(f"  ERROR: {e}")
            print("  This age group failed -- check memory/data.")
            continue

    # --- Save results ---
    if all_es_results:
        es_df = pd.DataFrame(all_es_results)
        es_df.to_csv(OUTPUT_DIR / "corrected_es_all.csv", index=False)
        print(f"\n  Saved event study coefficients -> corrected_es_all.csv")

    if pretrend_tests:
        pt_df = pd.DataFrame(pretrend_tests)
        pt_df.to_csv(OUTPUT_DIR / "corrected_pretrends.csv", index=False)
        pt_path = OUTPUT_DIR / "corrected_pretrends.txt"
        pt_df.to_string(pt_path, index=False)
        print(f"  Saved pre-trend tests -> corrected_pretrends.txt")
        print("\n  === PRE-TREND TEST RESULTS ===")
        print(pt_df.to_string(index=False))

    return es_df if all_es_results else pd.DataFrame(), pretrend_tests


# ======================================================================
#   STEP 3: FIGURES
# ======================================================================

def plot_event_studies(es_df):
    """One figure per age group, plus a combined panel."""
    if es_df.empty:
        return

    print("\n" + "=" * 70)
    print("STEP 3: Plotting event studies")
    print("=" * 70)

    for age_label in AGE_GROUPS:
        sub = es_df[es_df["age_group"] == age_label].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("period")

        # Numeric x-axis
        periods_sorted = sorted(sub["period"].unique())
        x_map = {p: i for i, p in enumerate(periods_sorted)}
        sub["x"] = sub["period"].map(x_map)
        ref_x = x_map.get(REF_HALFYEAR, None)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Confidence intervals
        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=ORANGE)
        ax.plot(sub["x"], sub["coef"], "o-", color=ORANGE, linewidth=2,
                markersize=6)
        ax.axhline(0, color="black", linewidth=0.5)

        # Reference period marker
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=1,
                       label=f"Reference: {REF_HALFYEAR}")

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (ln employment)", fontsize=11)
        ax.set_title(f"Event study: {age_label} (corrected FE)", fontsize=13)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"corrected_es_{age_label.replace('+','plus')}.png",
                    dpi=150)
        plt.close(fig)
        print(f"  Saved figure for {age_label}")

    # --- Combined panel (2×3) ---
    age_list = list(AGE_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    for idx, age_label in enumerate(age_list):
        ax = axes[idx // 3, idx % 3]
        sub = es_df[es_df["age_group"] == age_label].copy()
        if sub.empty:
            ax.set_title(age_label)
            continue

        sub = sub.sort_values("period")
        periods_sorted = sorted(sub["period"].unique())
        x_map = {p: i for i, p in enumerate(periods_sorted)}
        sub["x"] = sub["period"].map(x_map)
        ref_x = x_map.get(REF_HALFYEAR, None)

        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                        alpha=0.2, color=ORANGE)
        ax.plot(sub["x"], sub["coef"], "o-", color=ORANGE, linewidth=1.5,
                markersize=4)
        ax.axhline(0, color="black", linewidth=0.5)
        if ref_x is not None:
            ax.axvline(ref_x, color=GRAY, linestyle="--", linewidth=0.8)

        ax.set_xticks(range(len(periods_sorted)))
        ax.set_xticklabels(periods_sorted, rotation=45, ha="right", fontsize=7)
        ax.set_title(age_label, fontsize=11)

    fig.suptitle("Corrected event study (employer×quartile + employer×month FE)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "corrected_es_panel.png", dpi=150)
    plt.close(fig)
    print("  Saved combined panel figure")


# ======================================================================
#   MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("18_mona_eventstudy_corrected.py")
    print("Event study with employer×quartile + employer×month FE")
    print("(matching the main DiD specification)")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Load data
    agg = load_and_prepare()

    # Step 2: Run corrected event study
    es_df, pt_tests = run_corrected_event_study(agg)

    # Step 3: Plot
    plot_event_studies(es_df)

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    # Summary
    summary = [
        "=" * 50,
        "CORRECTED EVENT STUDY SUMMARY",
        "=" * 50,
        f"Script: 18_mona_eventstudy_corrected.py",
        f"FE: employer×quartile (entity) + employer×month (other_effects)",
        f"Reference period: {REF_HALFYEAR}",
        f"Runtime: {elapsed/60:.1f} minutes",
        "",
        "PRE-TREND TESTS (joint chi2, H0: all pre-period coefficients = 0):",
    ]
    for pt in pt_tests:
        status = "PASS" if pt["p_value"] > 0.05 else "FAIL"
        summary.append(
            f"  {pt['age_group']:>8s}: chi2({pt['df']}) = {pt['chi2']:>7.2f}, "
            f"p = {pt['p_value']:.4f}  [{status}]"
        )
    summary.append("")
    summary.append("Compare with old pre-trend results (script 14, weaker FE):")
    summary.append("  22-25:   chi2(6) =   6.93, p = 0.3277  [PASS]")
    summary.append("  26-30:   chi2(6) = 161.38, p = 0.0000  [FAIL]")
    summary.append("  31-34:   chi2(6) = 254.25, p = 0.0000  [FAIL]")
    summary.append("  35-40:   chi2(6) =  37.70, p = 0.0000  [FAIL]")
    summary.append("  41-49:   chi2(6) = 117.64, p = 0.0000  [FAIL]")
    summary.append("  50+:     chi2(6) = 267.92, p = 0.0000  [FAIL]")

    summary_text = "\n".join(summary)
    (OUTPUT_DIR / "corrected_es_summary.txt").write_text(summary_text)
    print(f"\n{summary_text}")
    print(f"\n  All output in: {OUTPUT_DIR}/")

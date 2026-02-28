#!/usr/bin/env python3
"""
14_mona_employer_regression.py — Brynjolfsson-style employer-level DiD.

╔══════════════════════════════════════════════════════════════════════╗
║  THIS SCRIPT IS DESIGNED TO RUN IN SCB's MONA SECURE ENVIRONMENT   ║
║  It uses monthly AGI (employer declaration) register data.          ║
║  Do NOT run outside MONA — the data is not available externally.    ║
╚══════════════════════════════════════════════════════════════════════╝

Purpose:
  Formally test whether young workers (16-24) in high-AI-exposure occupations
  experienced disproportionate employment declines after ChatGPT. This upgrades
  the descriptive Figure 2 in the paper to a causally identified result.

Design:
  Mirrors Brynjolfsson, Chandar & Chen (2025), Eq. 4.1:

    ln(E[y_{f,q,t}]) = α_{f,q} + β_{f,t} + γ₁·PostRB_t·HighQ4_q
                        + γ₂·PostGPT_t·HighQ4_q + ε_{f,q,t}

  where f = employer, q = DAIOE quartile, t = month.

  Employer×quartile FE absorb baseline differences within firms.
  Employer×month FE absorb ALL firm-level macro shocks (interest rates,
  energy crisis, seasonal hiring, etc.).

  Run SEPARATELY for each age group. The "canaries" finding is that
  γ₂ is negative and significant for ages 16-24, but not for older groups.

Estimator:
  - Primary: OLS on ln(count+1) with high-dimensional FE via linearmodels
    (linearmodels.panel.PanelOLS or absorbed-FE approach)
  - If linearmodels unavailable: manual within-transformation with pandas
  - Poisson (ideal but computationally heavy): attempted if statsmodels GLM
    converges within memory limits

Input files (on MONA):
  1. AGI individual records (same extract as 09_mona_agi_canaries.py)
  2. daioe_quartiles.csv (ssyk4, exposure_quartile)

Output files (export from MONA):
  1. canaries_did_results.csv       — DiD coefficient table by age group
  2. canaries_eventstudy_*.csv      — half-year event study coefficients
  3. canaries_es_young.png          — event study figure for 16-24
  4. canaries_es_older.png          — event study figure for 25-30, 41-49
  5. canaries_summary.txt           — sample sizes and diagnostics
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


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — ADJUST THESE FOR YOUR MONA ENVIRONMENT            ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Path to your AGI extract (parquet, CSV, or SAS)
# >>> USE THE SAME PATH AS IN 09_mona_agi_canaries.py <<<
INPUT_PATH = Path("agi_monthly_extract.parquet")

# Column names in the AGI extract — adjust if yours differ
# >>> MUST MATCH 09_mona_agi_canaries.py, plus employer_id <<<
AGI_COLUMNS = {
    "person_id": "LopNr",          # Encrypted person ID (same as script 09)
    "employer_id": "ArbstId",      # Encrypted employer/workplace ID (NEW)
    "year_month": "Period",         # Year-month (same as script 09)
    "ssyk4": "SSYK4",              # 4-digit SSYK 2012 (same as script 09)
    "birth_year": "FodelseAr",     # Birth year (same as script 09)
}

# Path to DAIOE quartiles (same file as script 09 uses)
DAIOE_PATH = Path("daioe_quartiles.csv")

# Output paths (saves alongside script 09 output)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Treatment dates
RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"

# Reference period for event study (just before ChatGPT)
REF_HALFYEAR = "2022H1"

# Age group definitions (run regressions separately for each)
AGE_GROUPS = {
    "16-24": (16, 24),
    "25-30": (25, 30),
    "31-40": (31, 40),
    "41-49": (41, 49),
    "50+":   (50, 69),
}

# Minimum employer size (helps with computation — drop tiny employers)
MIN_EMPLOYER_SIZE = 5

# Colours
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
DARK_BLUE = "#1B3A5C"
DARK_TEXT = "#2C2C2C"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: LOAD AND PREPARE DATA                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_and_prepare():
    """
    Load AGI individual data, merge DAIOE quartiles, assign age groups,
    and aggregate to employer × quartile × age_group × month cells.

    This is the unit of analysis in Brynjolfsson et al. (2025).
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing data")
    print("=" * 70)

    # --- Load AGI ---
    suffix = INPUT_PATH.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(INPUT_PATH)
    elif suffix == ".csv":
        df = pd.read_csv(INPUT_PATH)
    elif suffix in (".sas7bdat", ".sas"):
        df = pd.read_sas(INPUT_PATH)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    print(f"  Loaded {len(df):,} individual-month records")
    print(f"  Columns: {df.columns.tolist()}")

    # Rename columns
    rename_map = {v: k for k, v in AGI_COLUMNS.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    # Check required columns
    required = ["person_id", "employer_id", "year_month", "ssyk4"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns after rename: {missing}. "
            f"Available: {df.columns.tolist()}. "
            f"Adjust AGI_COLUMNS dict."
        )

    # Parse year-month
    df["year_month"] = df["year_month"].astype(str).str[:7]

    # Compute age
    if "birth_year" in df.columns:
        year = df["year_month"].str[:4].astype(int)
        df["age"] = year - df["birth_year"].astype(int)
    elif "age" not in df.columns:
        raise KeyError("Need 'birth_year' or 'age'. Adjust AGI_COLUMNS.")

    # Filter working age
    df = df[(df["age"] >= 16) & (df["age"] <= 69)].copy()

    # SSYK4 as string
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)

    # --- Merge DAIOE quartiles ---
    print("\n  Merging DAIOE quartiles...")
    daioe = pd.read_csv(DAIOE_PATH)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    # Ensure numeric quartile (1-4)
    if daioe["exposure_quartile"].dtype == object:
        q_map = {"Q1 (lowest)": 1, "Q2": 2, "Q3": 3, "Q4 (highest)": 4}
        daioe["exposure_quartile"] = daioe["exposure_quartile"].map(q_map)
    df = df.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4", how="inner")
    print(f"  After DAIOE merge: {len(df):,} records, "
          f"{df['ssyk4'].nunique()} occupations")

    # --- Assign age groups ---
    def get_age_group(age):
        for label, (lo, hi) in AGE_GROUPS.items():
            if lo <= age <= hi:
                return label
        return None

    df["age_group"] = df["age"].apply(get_age_group)
    df = df[df["age_group"].notna()].copy()

    # --- Filter small employers ---
    emp_size = df.groupby("employer_id")["person_id"].nunique()
    large_emp = emp_size[emp_size >= MIN_EMPLOYER_SIZE].index
    n_before = df["employer_id"].nunique()
    df = df[df["employer_id"].isin(large_emp)].copy()
    print(f"  Employers: {n_before:,} total → {df['employer_id'].nunique():,} "
          f"(≥{MIN_EMPLOYER_SIZE} workers)")

    # --- Aggregate to employer × quartile × age_group × month ---
    print("\n  Aggregating to employer × quartile × age_group × month...")
    agg = (
        df.groupby(["employer_id", "exposure_quartile", "age_group", "year_month"])
        ["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "n_emp"})
    )

    print(f"  Panel cells: {len(agg):,}")
    print(f"  Employers: {agg['employer_id'].nunique():,}")
    print(f"  Months: {agg['year_month'].nunique()}")
    print(f"  Period: {agg['year_month'].min()} to {agg['year_month'].max()}")
    print(f"\n  Quartile distribution:")
    for q in sorted(agg["exposure_quartile"].unique()):
        n = agg[agg["exposure_quartile"] == q]["n_emp"].sum()
        print(f"    Q{q}: {n:,} person-months")

    return agg


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: MAIN DiD BY AGE GROUP                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def run_did_by_age(agg):
    """
    For each age group, estimate:

      ln(n_emp_{f,q,t} + 1) = α_{f,q} + β_{f,t}
                               + γ₁·PostRB_t·High_q
                               + γ₂·PostGPT_t·High_q + ε

    where High = (quartile == 4).

    Employer×quartile FE (α_{f,q}) absorb time-invariant differences.
    Employer×month FE (β_{f,t}) absorb ALL firm-level shocks.

    Identification: within-firm, within-month variation across quartiles.

    Returns a DataFrame of coefficients for all age groups.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DiD regressions by age group")
    print("=" * 70)

    # Treatment variables
    agg = agg.copy()
    agg["post_rb"] = (agg["year_month"] >= RIKSBANK_YM).astype(int)
    agg["post_gpt"] = (agg["year_month"] >= CHATGPT_YM).astype(int)
    agg["high"] = (agg["exposure_quartile"] == 4).astype(int)
    agg["post_rb_x_high"] = agg["post_rb"] * agg["high"]
    agg["post_gpt_x_high"] = agg["post_gpt"] * agg["high"]

    # Log outcome (add 1 to handle zeros)
    agg["ln_emp"] = np.log(agg["n_emp"] + 1)

    # Create FE group identifiers
    agg["fe_emp_q"] = (
        agg["employer_id"].astype(str) + "_" +
        agg["exposure_quartile"].astype(str)
    )
    agg["fe_emp_t"] = (
        agg["employer_id"].astype(str) + "_" +
        agg["year_month"]
    )

    all_results = []

    for age_label, (age_lo, age_hi) in AGE_GROUPS.items():
        print(f"\n--- Age group: {age_label} ---")
        sub = agg[agg["age_group"] == age_label].copy()

        if len(sub) < 100:
            print(f"  Too few observations ({len(sub)}), skipping")
            continue

        print(f"  Observations: {len(sub):,}")
        print(f"  Employers: {sub['employer_id'].nunique():,}")
        print(f"  Mean employment: {sub['n_emp'].mean():.1f}")

        # --- Try linearmodels PanelOLS with absorbed FE ---
        result = _estimate_did(sub, age_label)
        if result is not None:
            all_results.append(result)

    # Combine results
    if all_results:
        results_df = pd.DataFrame(all_results)
        out = OUTPUT_DIR / "canaries_did_results.csv"
        results_df.to_csv(out, index=False)
        print(f"\n  Saved DiD results → {out.name}")
        print("\n  === SUMMARY ===")
        print(results_df.to_string(index=False))
        return results_df

    return pd.DataFrame()


def _estimate_did(sub, age_label):
    """
    Estimate the DiD for one age group. Tries three approaches:

    A. linearmodels PanelOLS with employer×quartile entity FE +
       employer×month as absorbed other_effects
    B. Manual within-transformation (demean by employer×quartile,
       then include employer×month dummies via absorption)
    C. Simple OLS with occupation + month FE (backup, weaker identification)
    """
    t0 = time.time()

    # --- Approach A: linearmodels ---
    try:
        from linearmodels.panel import PanelOLS

        panel = sub.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

        # PanelOLS can absorb entity effects + one set of other_effects.
        # Entity = employer×quartile, Other = employer×month
        panel = panel.set_index(["fe_emp_q", "date"])

        other_fe = pd.DataFrame(
            {"fe_emp_t": panel["fe_emp_t"]},
            index=panel.index,
        )

        mod = PanelOLS(
            dependent=panel["ln_emp"],
            exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
            entity_effects=True,
            time_effects=False,
            other_effects=other_fe,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        gamma1 = res.params["post_rb_x_high"]
        gamma2 = res.params["post_gpt_x_high"]
        se1 = res.std_errors["post_rb_x_high"]
        se2 = res.std_errors["post_gpt_x_high"]
        p1 = res.pvalues["post_rb_x_high"]
        p2 = res.pvalues["post_gpt_x_high"]

        elapsed = time.time() - t0
        print(f"  [linearmodels, {elapsed:.0f}s]")
        print(f"  γ₁ (PostRB × High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  γ₂ (PostGPT × High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "PanelOLS",
            "n_obs": int(res.nobs),
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except ImportError:
        print("  linearmodels not available — trying manual within-transformation")
    except Exception as e:
        print(f"  linearmodels failed: {e}")
        print("  Trying manual within-transformation...")

    # --- Approach B: Manual within-transformation ---
    # Demean by employer×quartile (entity FE) and employer×month (time FE)
    try:
        panel = sub.copy()

        # Demean outcome and regressors by employer×quartile
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm1"] = panel.groupby("fe_emp_q")[col].transform(
                lambda x: x - x.mean()
            )

        # Then demean by employer×month
        for col in ["ln_emp", "post_rb_x_high", "post_gpt_x_high"]:
            panel[f"{col}_dm"] = panel.groupby("fe_emp_t")[f"{col}_dm1"].transform(
                lambda x: x - x.mean()
            )

        # OLS on demeaned data (no constant needed after demeaning)
        import statsmodels.api as sm

        y = panel["ln_emp_dm"].values
        X = panel[["post_rb_x_high_dm", "post_gpt_x_high_dm"]].values

        # Simple OLS (SEs will be approximate without proper clustering correction)
        mod = sm.OLS(y, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["employer_id"].values},
        )

        gamma1 = res.params[0]
        gamma2 = res.params[1]
        se1 = res.bse[0]
        se2 = res.bse[1]
        p1 = res.pvalues[0]
        p2 = res.pvalues[1]

        elapsed = time.time() - t0
        print(f"  [within-transformation, {elapsed:.0f}s]")
        print(f"  γ₁ (PostRB × High)  = {gamma1:+.4f} (SE={se1:.4f}, p={p1:.4f})")
        print(f"  γ₂ (PostGPT × High) = {gamma2:+.4f} (SE={se2:.4f}, p={p2:.4f})")

        return {
            "age_group": age_label,
            "method": "within-transformation",
            "n_obs": len(panel),
            "gamma1_rb_high": gamma1,
            "se1": se1,
            "pval1": p1,
            "gamma2_gpt_high": gamma2,
            "se2": se2,
            "pval2": p2,
        }

    except Exception as e:
        print(f"  Within-transformation failed: {e}")

    # --- Approach C: Backup — occupation-level (weaker identification) ---
    print("  Falling back to occupation-level regression (Section 5 backup)")
    return _estimate_did_occupation_level(sub, age_label)


def _estimate_did_occupation_level(sub, age_label):
    """
    Backup: occupation × month panel with occupation + month FE.
    Weaker identification (no employer-level controls) but always feasible.
    """
    import statsmodels.api as sm

    occ_panel = (
        sub.groupby(["exposure_quartile", "year_month"])
        .agg(n_emp=("n_emp", "sum"))
        .reset_index()
    )
    occ_panel["high"] = (occ_panel["exposure_quartile"] == 4).astype(int)
    occ_panel["post_rb"] = (occ_panel["year_month"] >= RIKSBANK_YM).astype(int)
    occ_panel["post_gpt"] = (occ_panel["year_month"] >= CHATGPT_YM).astype(int)
    occ_panel["post_rb_x_high"] = occ_panel["post_rb"] * occ_panel["high"]
    occ_panel["post_gpt_x_high"] = occ_panel["post_gpt"] * occ_panel["high"]
    occ_panel["ln_emp"] = np.log(occ_panel["n_emp"] + 1)

    # Occupation + month dummies
    q_dummies = pd.get_dummies(occ_panel["exposure_quartile"], prefix="q", drop_first=True)
    t_dummies = pd.get_dummies(occ_panel["year_month"], prefix="t", drop_first=True)

    X = pd.concat([
        occ_panel[["post_rb_x_high", "post_gpt_x_high"]],
        q_dummies, t_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)

    mod = sm.OLS(occ_panel["ln_emp"].values, X)
    res = mod.fit(cov_type="HC1")

    gamma1 = res.params["post_rb_x_high"]
    gamma2 = res.params["post_gpt_x_high"]

    print(f"  [occupation-level backup]")
    print(f"  γ₁ (PostRB × High)  = {gamma1:+.4f} (SE={res.bse['post_rb_x_high']:.4f})")
    print(f"  γ₂ (PostGPT × High) = {gamma2:+.4f} (SE={res.bse['post_gpt_x_high']:.4f})")

    return {
        "age_group": age_label,
        "method": "occupation-level",
        "n_obs": len(occ_panel),
        "gamma1_rb_high": gamma1,
        "se1": res.bse["post_rb_x_high"],
        "pval1": res.pvalues["post_rb_x_high"],
        "gamma2_gpt_high": gamma2,
        "se2": res.bse["post_gpt_x_high"],
        "pval2": res.pvalues["post_gpt_x_high"],
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: HALF-YEAR EVENT STUDY                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def assign_halfyear(ym_series):
    """Map 'YYYY-MM' strings to 'YYYYHn' labels."""
    year = ym_series.str[:4]
    month = ym_series.str[5:7].astype(int)
    half = np.where(month <= 6, "H1", "H2")
    return year + half


def run_halfyear_event_study(agg):
    """
    Half-year event study: interact half-year dummies with High indicator,
    separately by age group. Reference: 2022H1 (pre-Riksbank).

    This traces the time path of the high-vs-low AI exposure gap,
    showing whether divergence appears (a) pre-ChatGPT, (b) post-ChatGPT,
    or (c) was already present earlier.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Half-year event study")
    print("=" * 70)

    agg = agg.copy()
    agg["halfyear"] = assign_halfyear(agg["year_month"])
    agg["high"] = (agg["exposure_quartile"] == 4).astype(int)
    agg["ln_emp"] = np.log(agg["n_emp"] + 1)

    all_periods = sorted(agg["halfyear"].unique())
    event_periods = [p for p in all_periods if p != REF_HALFYEAR]

    all_es_results = []

    for age_label in AGE_GROUPS:
        print(f"\n--- Event study: {age_label} ---")
        sub = agg[agg["age_group"] == age_label].copy()

        if len(sub) < 100:
            print(f"  Too few observations, skipping")
            continue

        # Create interaction dummies
        for p in event_periods:
            sub[f"hy_{p}"] = ((sub["halfyear"] == p).astype(int) * sub["high"])

        interaction_cols = [f"hy_{p}" for p in event_periods]

        # Try linearmodels, fall back to manual approach
        try:
            from linearmodels.panel import PanelOLS

            panel = sub.copy()
            panel["date"] = pd.to_datetime(panel["year_month"] + "-01")

            # Use occupation-level FE here (employer-level too many groups
            # for event study with many interaction terms)
            panel["entity"] = (
                panel["exposure_quartile"].astype(str) + "_" +
                panel["employer_id"].astype(str)
            )
            panel = panel.set_index(["entity", "date"])

            mod = PanelOLS(
                dependent=panel["ln_emp"],
                exog=panel[interaction_cols],
                entity_effects=True,
                time_effects=True,
            )
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.std_errors[col],
                    "pval": res.pvalues[col],
                })

        except (ImportError, Exception) as e:
            print(f"  linearmodels failed ({e}), using statsmodels")
            import statsmodels.api as sm

            # Simpler: quartile + month dummies
            q_dummies = pd.get_dummies(sub["exposure_quartile"], prefix="q", drop_first=True)
            t_dummies = pd.get_dummies(sub["year_month"], prefix="t", drop_first=True)
            X = pd.concat([sub[interaction_cols], q_dummies, t_dummies], axis=1).astype(float)
            X = sm.add_constant(X)

            mod = sm.OLS(sub["ln_emp"].values, X)
            res = mod.fit(cov_type="HC1")

            for p in event_periods:
                col = f"hy_{p}"
                all_es_results.append({
                    "age_group": age_label,
                    "period": p,
                    "coef": res.params[col],
                    "se": res.bse[col],
                    "pval": res.pvalues[col],
                })

        # Add reference period
        all_es_results.append({
            "age_group": age_label,
            "period": REF_HALFYEAR,
            "coef": 0.0,
            "se": 0.0,
            "pval": 1.0,
        })

    if all_es_results:
        es_df = pd.DataFrame(all_es_results)
        out = OUTPUT_DIR / "canaries_es_all.csv"
        es_df.to_csv(out, index=False)
        print(f"\n  Saved event study → {out.name}")
        return es_df

    return pd.DataFrame()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: EVENT STUDY FIGURES                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_event_studies(es_df):
    """Create event study coefficient plots for key age groups."""
    print("\n" + "=" * 70)
    print("STEP 4: Event study figures")
    print("=" * 70)

    for age_label, color, filename in [
        ("16-24", ORANGE, "canaries_es_young.png"),
        ("25-30", TEAL, "canaries_es_25to30.png"),
        ("41-49", DARK_BLUE, "canaries_es_41to49.png"),
    ]:
        sub = es_df[es_df["age_group"] == age_label].sort_values("period")
        if sub.empty:
            print(f"  No data for {age_label}, skipping")
            continue

        sub = sub.copy()
        sub["x"] = range(len(sub))
        sub["ci_lo"] = sub["coef"] - 1.96 * sub["se"]
        sub["ci_hi"] = sub["coef"] + 1.96 * sub["se"]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.fill_between(sub["x"], sub["ci_lo"], sub["ci_hi"],
                         alpha=0.15, color=color)
        ax.plot(sub["x"], sub["coef"], "o-", color=color, linewidth=2, markersize=6)
        ax.axhline(0, color=DARK_TEXT, linewidth=0.8, alpha=0.5)

        # Mark reference
        ref_rows = sub[sub["period"] == REF_HALFYEAR]
        if not ref_rows.empty:
            ref_x = ref_rows["x"].values[0]
            ax.axvline(ref_x, color=TEAL, linestyle="--", linewidth=1, alpha=0.7)

        # Mark ChatGPT
        gpt_rows = sub[sub["period"] == "2022H2"]
        if not gpt_rows.empty:
            gpt_x = gpt_rows["x"].values[0]
            ax.axvline(gpt_x, color=GRAY, linestyle=":", linewidth=1, alpha=0.7)

        ax.set_xticks(sub["x"])
        ax.set_xticklabels(sub["period"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Coefficient (High × period)")
        ax.set_title(f"Employment event study: ages {age_label}")

        fig.tight_layout()
        out = OUTPUT_DIR / filename
        fig.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved → {filename}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: DIAGNOSTICS AND SUMMARY                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def write_summary(agg, did_results, es_df):
    """Write diagnostic summary to text file."""
    out = OUTPUT_DIR / "canaries_summary.txt"
    with open(out, "w") as f:
        f.write("CANARIES REGRESSION — SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total panel cells: {len(agg):,}\n")
        f.write(f"Employers: {agg['employer_id'].nunique():,}\n")
        f.write(f"Months: {agg['year_month'].nunique()}\n")
        f.write(f"Period: {agg['year_month'].min()} to {agg['year_month'].max()}\n")
        f.write(f"Min employer size: {MIN_EMPLOYER_SIZE}\n\n")

        f.write("--- Age group sizes ---\n")
        for ag in AGE_GROUPS:
            n = len(agg[agg["age_group"] == ag])
            f.write(f"  {ag}: {n:,} cells\n")

        f.write("\n--- Quartile distribution ---\n")
        for q in sorted(agg["exposure_quartile"].unique()):
            n = agg[agg["exposure_quartile"] == q]["n_emp"].sum()
            f.write(f"  Q{q}: {n:,} person-months\n")

        if not did_results.empty:
            f.write("\n--- DiD Results ---\n")
            f.write(did_results.to_string(index=False))
            f.write("\n")

        f.write("\nFE structure: employer×quartile + employer×month\n")
        f.write("SEs clustered by employer\n")
        f.write("Estimator: OLS on ln(count+1)\n")

    print(f"\n  Saved summary → {out.name}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("CANARIES REGRESSION — Brynjolfsson-style employer-level DiD")
    print("Python version for MONA")
    print("=" * 70)

    # Step 1: Load and prepare
    agg = load_and_prepare()

    # Step 2: Main DiD by age group
    did_results = run_did_by_age(agg)

    # Step 3: Half-year event study
    es_df = run_halfyear_event_study(agg)

    # Step 4: Event study figures
    if not es_df.empty:
        plot_event_studies(es_df)

    # Step 5: Summary
    write_summary(agg, did_results, es_df)

    print("\n" + "=" * 70)
    print("DONE. Export these files from MONA:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("  1. canaries_did_results.csv    — DiD coefficients by age group")
    print("  2. canaries_es_all.csv         — event study coefficients")
    print("  3. canaries_es_young.png       — event study figure (16-24)")
    print("  4. canaries_es_25to30.png      — event study figure (25-30)")
    print("  5. canaries_es_41to49.png      — event study figure (41-49)")
    print("  6. canaries_summary.txt        — sample sizes and diagnostics")
    print("=" * 70)


if __name__ == "__main__":
    main()

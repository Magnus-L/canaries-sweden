#!/usr/bin/env python3
"""
07_robustness.py — Robustness checks for online appendix.

Alternative specifications:
  R1. Alternative AI measure: DAIOE all-apps (pctl_rank_allapps) instead of genAI
  R2. Vacancy-weighted results (sum of vacancies instead of ad count)
  R3. Exclude pandemic months (drop Jan–Jun 2020)
  R4. Tercile classification instead of quartiles
  R5. Exclude IT/tech occupations (SSYK 25xx) — cf. Brynjolfsson et al. (2025)
  R6. Balanced panel (only occupations observed every month)
  R7. Language-modelling exposure (DAIOE task-level measure)
  R8. Event study — monthly DiD coefficients (figure, not table row)

All results saved to tables/ and figures/ for inclusion in the appendix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, TABDIR, FIGDIR,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH, DAIOE_REF_YEAR,
    DARK_BLUE, ORANGE, TEAL, GRAY, set_rcparams,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

set_rcparams()


def load_data():
    """Load the main datasets needed for robustness checks."""
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    daioe_raw = pd.read_csv(RAW / "daioe_ssyk2012.csv", sep="\t")
    return merged, daioe_raw


def prepare_panel_generic(df):
    """Prepare panel with treatment dummies (reusable across specs)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)

    df["post_riksbank"] = (df["date"] >= rb_date).astype(int)
    df["post_chatgpt"] = (df["date"] >= gpt_date).astype(int)
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])
    return df


def run_panel_regression(df, high_var="high_exposure"):
    """
    Run the main DiD specification with a given high-exposure variable.
    Returns coefficients and standard errors as a dict.
    """
    df = df.copy()
    df["post_rb_x_high"] = df["post_riksbank"] * df[high_var]
    df["post_gpt_x_high"] = df["post_chatgpt"] * df[high_var]

    try:
        from linearmodels.panel import PanelOLS

        df = df.copy()
        df["date"] = pd.to_datetime(df["year_month"] + "-01")
        panel = df.set_index(["ssyk4", "date"])
        mod = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        return {
            "beta_rb": res.params["post_rb_x_high"],
            "se_rb": res.std_errors["post_rb_x_high"],
            "p_rb": res.pvalues["post_rb_x_high"],
            "beta_gpt": res.params["post_gpt_x_high"],
            "se_gpt": res.std_errors["post_gpt_x_high"],
            "p_gpt": res.pvalues["post_gpt_x_high"],
            "n_obs": res.nobs,
            "n_entities": res.entity_info["total"],
        }
    except ImportError:
        import statsmodels.api as sm

        occ_dummies = pd.get_dummies(df["ssyk4"], prefix="occ", drop_first=True)
        time_dummies = pd.get_dummies(df["year_month"], prefix="t", drop_first=True)
        X = pd.concat([df[["post_rb_x_high", "post_gpt_x_high"]], occ_dummies, time_dummies], axis=1)
        X = sm.add_constant(X).astype(float)

        mod = sm.OLS(df["ln_ads"].values, X)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["ssyk4"].values})

        return {
            "beta_rb": res.params["post_rb_x_high"],
            "se_rb": res.bse["post_rb_x_high"],
            "p_rb": res.pvalues["post_rb_x_high"],
            "beta_gpt": res.params["post_gpt_x_high"],
            "se_gpt": res.bse["post_gpt_x_high"],
            "p_gpt": res.pvalues["post_gpt_x_high"],
            "n_obs": int(res.nobs),
            "n_entities": df["ssyk4"].nunique(),
        }


# ── Robustness 1: Alternative AI measure (all-apps) ──────────────────────────

def robustness_allapps(merged, daioe_raw):
    """Re-estimate using pctl_rank_allapps instead of pctl_rank_genai."""
    print("  R1: Alternative AI measure (all-apps)...")

    daioe_ref = daioe_raw[daioe_raw["year"] == DAIOE_REF_YEAR].copy()
    daioe_ref["ssyk4"] = daioe_ref["ssyk2012_4"].str[:4].str.strip()
    daioe_ref = daioe_ref[["ssyk4", "pctl_rank_allapps"]].dropna().drop_duplicates("ssyk4")

    q75 = daioe_ref["pctl_rank_allapps"].quantile(0.75)
    daioe_ref["high_allapps"] = (daioe_ref["pctl_rank_allapps"] > q75).astype(int)

    # Re-merge
    postings = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv")
    postings["ssyk4"] = postings["ssyk4"].astype(str).str.zfill(4)
    daioe_ref["ssyk4"] = daioe_ref["ssyk4"].astype(str).str.zfill(4)
    m = postings.merge(daioe_ref, on="ssyk4", how="inner")

    panel = prepare_panel_generic(m)
    return run_panel_regression(panel, high_var="high_allapps")


# ── Robustness 2: Vacancy-weighted ───────────────────────────────────────────

def robustness_vacancy_weighted(merged):
    """Use ln(vacancies) instead of ln(ads) as outcome."""
    print("  R2: Vacancy-weighted results...")

    df = merged.copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["post_riksbank"] = (df["date"] >= pd.Timestamp(RIKSBANKEN_HIKE)).astype(int)
    df["post_chatgpt"] = (df["date"] >= pd.Timestamp(CHATGPT_LAUNCH)).astype(int)
    df["post_rb_x_high"] = df["post_riksbank"] * df["high_exposure"]
    df["post_gpt_x_high"] = df["post_chatgpt"] * df["high_exposure"]

    df = df[df["n_vacancies"] > 0].copy()
    df["ln_ads"] = np.log(df["n_vacancies"])  # Override: use vacancies

    return run_panel_regression(df, high_var="high_exposure")


# ── Robustness 3: Exclude pandemic months ────────────────────────────────────

def robustness_no_pandemic(merged):
    """Drop Jan–Jun 2020 to avoid pandemic distortion in the base period."""
    print("  R3: Exclude pandemic months (Jan–Jun 2020)...")

    df = merged[merged["year_month"] >= "2020-07"].copy()
    panel = prepare_panel_generic(df)
    return run_panel_regression(panel, high_var="high_exposure")


# ── Robustness 4: Terciles instead of quartiles ──────────────────────────────

def robustness_terciles(merged):
    """Use top tercile instead of top quartile as high exposure."""
    print("  R4: Tercile classification...")

    df = merged.copy()
    t67 = df["pctl_rank_genai"].quantile(0.667)
    df["high_tercile"] = (df["pctl_rank_genai"] > t67).astype(int)

    panel = prepare_panel_generic(df)
    return run_panel_regression(panel, high_var="high_tercile")


# ── Robustness 5: Exclude IT/tech occupations (SSYK 25xx) ────────────────────

def robustness_excl_tech(merged):
    """
    Drop all SSYK 25xx (IT specialist) occupations, then re-estimate.

    Brynjolfsson et al. exclude tech occupations (SOC 15-1) to show results
    aren't driven by the tech-specific downturn. Our equivalent: SSYK 25xx,
    which are all in Q4 with genAI pctl_rank > 80.
    """
    print("  R5: Exclude IT/tech occupations (SSYK 25xx)...")

    df = merged.copy()
    # SSYK 25xx = IT specialists (2511–2519)
    df = df[~df["ssyk4"].astype(str).str.startswith("25")].copy()
    print(f"    After dropping IT: {df['ssyk4'].nunique()} occupations remain")

    panel = prepare_panel_generic(df)
    return run_panel_regression(panel, high_var="high_exposure")


# ── Robustness 6: Balanced panel ─────────────────────────────────────────────

def robustness_balanced_panel(merged):
    """
    Restrict to occupations observed in EVERY month of the core sample.

    Tests whether entry/exit of rare occupations drives the results.
    Brynjolfsson et al. run an unbalanced panel check; we do the converse
    by enforcing strict balance.

    We first trim the data to the core sample period (2020-01 to 2025-12)
    to exclude sparse edge months and erroneous future dates.
    """
    print("  R6: Balanced panel...")

    df = merged.copy()
    # Restrict to core sample period (avoid sparse edge months)
    df = df[(df["year_month"] >= "2020-01") & (df["year_month"] <= "2025-12")].copy()
    n_months = df["year_month"].nunique()
    occ_counts = df.groupby("ssyk4")["year_month"].nunique()
    balanced_occs = occ_counts[occ_counts == n_months].index
    df = df[df["ssyk4"].isin(balanced_occs)].copy()
    print(f"    Core period: {n_months} months (2020-01 to 2025-12)")
    print(f"    Balanced occupations: {len(balanced_occs)} of {merged['ssyk4'].nunique()}")

    panel = prepare_panel_generic(df)
    return run_panel_regression(panel, high_var="high_exposure")


# ── Robustness 7: Language-modelling exposure ────────────────────────────────

def robustness_language_modelling(merged, daioe_raw):
    """
    Use DAIOE language-modelling task exposure (pctl_rank_lngmod) instead
    of the composite genAI measure.

    This isolates the LLM-specific channel — the most directly relevant
    AI capability for ChatGPT-era displacement.
    """
    print("  R7: Language-modelling exposure...")

    daioe_ref = daioe_raw[daioe_raw["year"] == DAIOE_REF_YEAR].copy()
    daioe_ref["ssyk4"] = daioe_ref["ssyk2012_4"].str[:4].str.strip()
    daioe_ref = daioe_ref[["ssyk4", "pctl_rank_lngmod"]].dropna().drop_duplicates("ssyk4")

    q75 = daioe_ref["pctl_rank_lngmod"].quantile(0.75)
    daioe_ref["high_lngmod"] = (daioe_ref["pctl_rank_lngmod"] > q75).astype(int)

    postings = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv")
    postings["ssyk4"] = postings["ssyk4"].astype(str).str.zfill(4)
    daioe_ref["ssyk4"] = daioe_ref["ssyk4"].astype(str).str.zfill(4)
    m = postings.merge(daioe_ref, on="ssyk4", how="inner")

    panel = prepare_panel_generic(m)
    return run_panel_regression(panel, high_var="high_lngmod")


# ── Robustness 8: Event study (dynamic DiD) ──────────────────────────────────

def robustness_event_study(merged):
    """
    Estimate monthly DiD coefficients relative to base month (Feb 2020).

    This is the standard event-study design: interact HighExposure with
    month dummies (omitting one base period). Plots the coefficients to
    visually assess pre-trends and treatment timing.

    Key for referees: shows whether high- and low-exposure occupations
    were trending similarly before April 2022 (Riksbanken) and Dec 2022
    (ChatGPT), and whether any divergence aligns with those dates.
    """
    print("  R8: Event study (dynamic DiD)...")

    df = merged.copy()
    # Restrict to core sample period (avoid erroneous dates like 2099-01)
    df = df[(df["year_month"] >= "2020-01") & (df["year_month"] <= "2025-12")].copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])

    # Create month dummies interacted with high_exposure
    # Omit base month (2020-02) as reference period
    base_month = "2020-02"
    months = sorted(df["year_month"].unique())
    months_excl_base = [m for m in months if m != base_month]

    # Build interaction terms: D_t × HighExp for each month t ≠ base
    for m in months_excl_base:
        col = f"m_{m}_x_high"
        df[col] = ((df["year_month"] == m) & (df["high_exposure"] == 1)).astype(int)

    interaction_cols = [f"m_{m}_x_high" for m in months_excl_base]

    try:
        from linearmodels.panel import PanelOLS

        panel = df.copy()
        panel["date"] = pd.to_datetime(panel["year_month"] + "-01")
        panel = panel.set_index(["ssyk4", "date"])

        mod = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[interaction_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        coefs = res.params[interaction_cols]
        ses = res.std_errors[interaction_cols]

    except ImportError:
        import statsmodels.api as sm

        occ_dummies = pd.get_dummies(df["ssyk4"], prefix="occ", drop_first=True)
        time_dummies = pd.get_dummies(df["year_month"], prefix="t", drop_first=True)
        X = pd.concat([df[interaction_cols], occ_dummies, time_dummies], axis=1)
        X = sm.add_constant(X).astype(float)

        mod = sm.OLS(df["ln_ads"].values, X)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["ssyk4"].values})

        coefs = res.params[interaction_cols]
        ses = res.bse[interaction_cols]

    # ── Formal pre-trends test (joint Wald test) ──────────────────────────
    # H₀: all pre-Riksbank interaction coefficients are jointly zero
    # This tests the parallel trends assumption formally (requested by R2).
    pre_months = [m for m in months_excl_base if m < "2022-04"]
    pre_cols_test = [f"m_{m}_x_high" for m in pre_months]
    n_pre = len(pre_cols_test)
    pretrend_result = {}

    try:
        from scipy import stats as scipy_stats

        pre_beta = np.array([coefs[c] for c in pre_cols_test])
        # Covariance matrix for pre-period coefficients (uses clustered SEs)
        pre_vcov = res.cov.loc[pre_cols_test, pre_cols_test].values
        # Wald statistic: β' V⁻¹ β ~ χ²(q)
        W = float(pre_beta @ np.linalg.inv(pre_vcov) @ pre_beta)
        F_stat = W / n_pre
        p_wald = 1 - scipy_stats.chi2.cdf(W, n_pre)

        print(f"    Pre-trends Wald test: χ²({n_pre}) = {W:.2f}, p = {p_wald:.4f}")
        print(f"    (F-stat = {F_stat:.3f})")

        pretrend_result = {
            "n_pre_periods": n_pre,
            "wald_stat": round(W, 3),
            "f_stat": round(F_stat, 3),
            "p_value": round(p_wald, 4),
            "reject_at_05": p_wald < 0.05,
        }
        pd.DataFrame([pretrend_result]).to_csv(
            TABDIR / "pretrend_test.csv", index=False
        )
        print(f"    Saved → pretrend_test.csv")
    except Exception as e:
        print(f"    Pre-trends test failed: {e}")

    # Build results DataFrame for plotting
    event_df = pd.DataFrame({
        "year_month": months_excl_base,
        "coef": [coefs[f"m_{m}_x_high"] for m in months_excl_base],
        "se": [ses[f"m_{m}_x_high"] for m in months_excl_base],
    })
    event_df["date"] = pd.to_datetime(event_df["year_month"] + "-01")
    event_df = event_df.sort_values("date")
    event_df["ci_lo"] = event_df["coef"] - 1.96 * event_df["se"]
    event_df["ci_hi"] = event_df["coef"] + 1.96 * event_df["se"]

    # Add the base month as zero
    base_row = pd.DataFrame({
        "year_month": [base_month],
        "coef": [0.0], "se": [0.0],
        "date": [pd.Timestamp(base_month + "-01")],
        "ci_lo": [0.0], "ci_hi": [0.0],
    })
    event_df = pd.concat([event_df, base_row]).sort_values("date").reset_index(drop=True)

    # Save coefficients
    event_df.to_csv(TABDIR / "event_study_coefficients.csv", index=False)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(event_df["date"], event_df["ci_lo"], event_df["ci_hi"],
                    alpha=0.2, color=DARK_BLUE)
    ax.plot(event_df["date"], event_df["coef"], color=DARK_BLUE, linewidth=1.5,
            marker="o", markersize=3)
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")

    # Event markers
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)
    ax.axvline(rb_date, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.axvline(gpt_date, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.9)

    # ── Fix 4: Place labels inside plot area, below top, clearly legible ──
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.08  # 8% below top
    ax.annotate(
        "Riksbanken hike\n(Apr 2022)",
        xy=(rb_date, label_y), fontsize=9,
        color=ORANGE, fontweight="bold", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, alpha=0.85),
    )
    ax.annotate(
        "ChatGPT launch\n(Nov 2022)",
        xy=(gpt_date, label_y), fontsize=9,
        color=TEAL, fontweight="bold", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEAL, alpha=0.85),
    )

    ax.set_xlabel("")
    ax.set_ylabel("Coefficient (relative to Feb 2020)")
    ax.set_title("Event study: High vs low genAI exposure (monthly DiD coefficients)")

    fig.tight_layout()
    fig.savefig(FIGDIR / "figA3_event_study.png")
    plt.close(fig)
    print(f"    Saved → figA3_event_study.png")
    print(f"    Saved → event_study_coefficients.csv")

    return event_df


# ── Robustness 9: Quarterly event study ──────────────────────────────────────

def robustness_quarterly_event_study(merged):
    """
    Quarterly-aggregated event study to reduce monthly noise in pre-trends test.

    Economic rationale: AI adoption and monetary policy transmission operate at
    quarterly (or slower) frequency. Monthly noise from seasonal hiring patterns
    inflates the Wald statistic with 26 pre-period monthly interactions.
    Aggregating to quarters reduces this to ~9 pre-period coefficients,
    giving a cleaner test (Bertrand, Duflo & Mullainathan 2004; Rak 2025).
    """
    print("  R9: Quarterly event study...")

    df = merged.copy()
    df = df[(df["year_month"] >= "2020-01") & (df["year_month"] <= "2025-12")].copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # Aggregate to occupation × quarter
    quarterly = (
        df.groupby(["ssyk4", "quarter", "high_exposure"])
        .agg(n_ads=("n_ads", "sum"))
        .reset_index()
    )
    quarterly = quarterly[quarterly["n_ads"] > 0].copy()
    quarterly["ln_ads"] = np.log(quarterly["n_ads"])

    # Create date for panel index (first day of quarter)
    quarterly["date"] = pd.PeriodIndex(quarterly["quarter"], freq="Q").to_timestamp()

    # Base quarter: 2020Q1 (first full quarter)
    base_quarter = "2020Q1"
    quarters = sorted(quarterly["quarter"].unique())
    quarters_excl_base = [q for q in quarters if q != base_quarter]

    # Build interaction terms
    for q in quarters_excl_base:
        col = f"q_{q}_x_high"
        quarterly[col] = (
            (quarterly["quarter"] == q) & (quarterly["high_exposure"] == 1)
        ).astype(int)

    interaction_cols = [f"q_{q}_x_high" for q in quarters_excl_base]

    try:
        from linearmodels.panel import PanelOLS

        panel = quarterly.copy().set_index(["ssyk4", "date"])

        mod = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[interaction_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        coefs = res.params[interaction_cols]
        ses = res.std_errors[interaction_cols]

        # ── Formal pre-trends test (quarterly) ──
        # Pre-Riksbank quarters: before 2022Q2 (April 2022 is in Q2)
        pre_quarters = [q for q in quarters_excl_base if q < "2022Q2"]
        pre_cols = [f"q_{q}_x_high" for q in pre_quarters]
        n_pre_q = len(pre_cols)

        try:
            from scipy import stats as scipy_stats

            pre_beta = np.array([coefs[c] for c in pre_cols])
            pre_vcov = res.cov.loc[pre_cols, pre_cols].values
            W = float(pre_beta @ np.linalg.inv(pre_vcov) @ pre_beta)
            p_wald = 1 - scipy_stats.chi2.cdf(W, n_pre_q)

            print(f"    Quarterly pre-trends Wald test: χ²({n_pre_q}) = {W:.2f}, "
                  f"p = {p_wald:.4f}")

            pd.DataFrame([{
                "n_pre_periods": n_pre_q,
                "wald_stat": round(W, 3),
                "p_value": round(p_wald, 4),
                "reject_at_05": p_wald < 0.05,
                "frequency": "quarterly",
            }]).to_csv(TABDIR / "pretrend_test_quarterly.csv", index=False)
            print(f"    Saved → pretrend_test_quarterly.csv")
        except Exception as e:
            print(f"    Quarterly pre-trends test failed: {e}")

    except ImportError:
        print("    linearmodels not available — skipping quarterly event study")
        return None

    # Build results for plotting
    event_df = pd.DataFrame({
        "quarter": quarters_excl_base,
        "coef": [coefs[f"q_{q}_x_high"] for q in quarters_excl_base],
        "se": [ses[f"q_{q}_x_high"] for q in quarters_excl_base],
    })
    event_df["date"] = pd.PeriodIndex(event_df["quarter"], freq="Q").to_timestamp()
    event_df = event_df.sort_values("date")
    event_df["ci_lo"] = event_df["coef"] - 1.96 * event_df["se"]
    event_df["ci_hi"] = event_df["coef"] + 1.96 * event_df["se"]

    # Add base quarter as zero
    base_row = pd.DataFrame({
        "quarter": [base_quarter], "coef": [0.0], "se": [0.0],
        "date": [pd.Timestamp("2020-01-01")],
        "ci_lo": [0.0], "ci_hi": [0.0],
    })
    event_df = pd.concat([event_df, base_row]).sort_values("date").reset_index(drop=True)
    event_df.to_csv(TABDIR / "event_study_quarterly.csv", index=False)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(event_df["date"], event_df["ci_lo"], event_df["ci_hi"],
                    alpha=0.2, color=DARK_BLUE)
    ax.plot(event_df["date"], event_df["coef"], color=DARK_BLUE, linewidth=2,
            marker="o", markersize=5)
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")

    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)
    ax.axvline(rb_date, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.axvline(gpt_date, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.9)

    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.08
    ax.annotate("Riksbanken hike\n(Apr 2022)",
                xy=(rb_date, label_y), fontsize=9,
                color=ORANGE, fontweight="bold", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, alpha=0.85))
    ax.annotate("ChatGPT launch\n(Nov 2022)",
                xy=(gpt_date, label_y), fontsize=9,
                color=TEAL, fontweight="bold", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEAL, alpha=0.85))

    ax.set_xlabel("")
    ax.set_ylabel("Coefficient (relative to 2020 Q1)")
    ax.set_title("Quarterly event study: High vs low genAI exposure")

    fig.tight_layout()
    fig.savefig(FIGDIR / "figA5_event_study_quarterly.png")
    plt.close(fig)
    print(f"    Saved → figA5_event_study_quarterly.png")

    return event_df


# ── Collect all results ──────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 7: Robustness checks")
    print("=" * 70)

    merged, daioe_raw = load_data()

    # Baseline
    print("\n  Baseline (genAI Q4)...")
    panel = prepare_panel_generic(merged)
    baseline = run_panel_regression(panel, high_var="high_exposure")

    # All robustness checks
    results = {"Baseline (genAI Q4)": baseline}

    try:
        results["All-apps measure"] = robustness_allapps(merged, daioe_raw)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Vacancy-weighted"] = robustness_vacancy_weighted(merged)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Excl. pandemic"] = robustness_no_pandemic(merged)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Terciles"] = robustness_terciles(merged)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Excl. IT/tech"] = robustness_excl_tech(merged)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Balanced panel"] = robustness_balanced_panel(merged)
    except Exception as e:
        print(f"    FAILED: {e}")

    try:
        results["Language model"] = robustness_language_modelling(merged, daioe_raw)
    except Exception as e:
        print(f"    FAILED: {e}")

    # Event studies (separate outputs — figures, not table rows)
    try:
        robustness_event_study(merged)
    except Exception as e:
        print(f"    Monthly event study FAILED: {e}")

    try:
        robustness_quarterly_event_study(merged)
    except Exception as e:
        print(f"    Quarterly event study FAILED: {e}")

    # Format results table
    print("\n" + "=" * 70)
    print("Robustness summary:")
    print("=" * 70)

    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    rows = []
    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    β₁ (PostRB × High) = {r['beta_rb']:.4f}{stars(r['p_rb'])} "
              f"(SE = {r['se_rb']:.4f})")
        print(f"    β₂ (PostGPT × High) = {r['beta_gpt']:.4f}{stars(r['p_gpt'])} "
              f"(SE = {r['se_gpt']:.4f})")
        print(f"    N = {r['n_obs']:,}, occupations = {r['n_entities']}")

        rows.append({
            "specification": name,
            "beta_rb": r["beta_rb"],
            "se_rb": r["se_rb"],
            "p_rb": r["p_rb"],
            "beta_gpt": r["beta_gpt"],
            "se_gpt": r["se_gpt"],
            "p_gpt": r["p_gpt"],
            "n_obs": r["n_obs"],
            "n_entities": r["n_entities"],
        })

    # Save
    robustness_df = pd.DataFrame(rows)
    out = TABDIR / "robustness_results.csv"
    robustness_df.to_csv(out, index=False)
    print(f"\nSaved → {out.name}")

    # LaTeX table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness checks}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        r"Specification & $\hat\beta_1$ (Riksbank) & $\hat\beta_2$ (ChatGPT) & $N$ & Occ. \\",
        r"\hline",
    ]

    for _, row in robustness_df.iterrows():
        name = row["specification"]
        b1 = f"{row['beta_rb']:.4f}{stars(row['p_rb'])}"
        b2 = f"{row['beta_gpt']:.4f}{stars(row['p_gpt'])}"
        lines.append(
            f"{name} & {b1} & {b2} & {int(row['n_obs']):,} & {int(row['n_entities'])} \\\\"
        )
        lines.append(
            f" & ({row['se_rb']:.4f}) & ({row['se_gpt']:.4f}) & & \\\\"
        )

    lines.extend([
        r"\hline\hline",
        r"\multicolumn{5}{p{0.95\textwidth}}{\footnotesize \textit{Notes:} "
        r"All specifications include occupation and month fixed effects. "
        r"Standard errors (in parentheses) clustered at occupation level. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_out = TABDIR / "robustness_results.tex"
    tex_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {tex_out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

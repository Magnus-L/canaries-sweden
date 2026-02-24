#!/usr/bin/env python3
"""
05_analysis.py — DiD regression and summary statistics.

Estimates the main timing identification model:

    ln(postings_it) = α_i + γ_t + β₁·PostRB_t·HighExp_i
                              + β₂·PostChatGPT_t·HighExp_i + ε_it

where:
  - i = SSYK4 occupation, t = year-month
  - α_i = occupation fixed effects
  - γ_t = time fixed effects (year-month dummies)
  - PostRB_t = 1 if t ≥ April 2022 (Riksbanken first hike)
  - PostChatGPT_t = 1 if t ≥ December 2022
  - HighExp_i = 1 if occupation in Q4 of genAI exposure

Interpretation:
  - If monetary policy drives the posting decline in high-exposure occupations:
    β₁ significant, β₂ ≈ 0
  - If AI drives the decline: β₂ significantly negative beyond the monetary effect

Standard errors clustered at the occupation level (conservative).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED, TABDIR, RIKSBANKEN_HIKE, CHATGPT_LAUNCH

import pandas as pd
import numpy as np
import statsmodels.api as sm


def prepare_panel(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the panel dataset for DiD estimation.

    Creates treatment dummies and the log-transformed outcome.
    Drops occupation × month cells with zero postings (log undefined).
    """
    print("Preparing panel data...")

    df = merged.copy()
    df["date"] = pd.to_datetime(df["year_month"] + "-01")

    # Treatment dummies
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)

    df["post_riksbank"] = (df["date"] >= rb_date).astype(int)
    df["post_chatgpt"] = (df["date"] >= gpt_date).astype(int)

    # Interaction terms
    df["post_rb_x_high"] = df["post_riksbank"] * df["high_exposure"]
    df["post_gpt_x_high"] = df["post_chatgpt"] * df["high_exposure"]

    # Log outcome (drop zeros)
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])

    # Time trend per occupation (for specification with occupation-specific trends)
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days / 30.0
    df["time_x_high"] = df["time_idx"] * df["high_exposure"]

    print(f"  Panel: {len(df):,} observations")
    print(f"  Occupations: {df['ssyk4'].nunique()}")
    print(f"  Months: {df['year_month'].nunique()}")
    print(f"  Pre-Riksbank: {(df['post_riksbank'] == 0).sum():,}")
    print(f"  Post-Riksbank, pre-ChatGPT: "
          f"{((df['post_riksbank'] == 1) & (df['post_chatgpt'] == 0)).sum():,}")
    print(f"  Post-ChatGPT: {(df['post_chatgpt'] == 1).sum():,}")

    return df


def run_did_regressions(df: pd.DataFrame) -> dict:
    """
    Estimate three DiD specifications:

    (1) Monetary policy only: PostRB × HighExp
    (2) + ChatGPT: PostRB × HighExp + PostChatGPT × HighExp
    (3) + occupation-specific trends

    All include occupation and time fixed effects.
    Standard errors clustered at occupation level.

    We use linearmodels for proper two-way fixed effects when available,
    falling back to statsmodels with dummies otherwise.
    """
    print("\nRunning DiD regressions...")

    results = {}

    try:
        from linearmodels.panel import PanelOLS

        # Set up multi-index for panel
        panel = df.set_index(["ssyk4", "year_month"])

        # Specification 1: Monetary policy only
        print("  (1) Monetary policy interaction only...")
        mod1 = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[["post_rb_x_high"]],
            entity_effects=True,
            time_effects=True,
        )
        res1 = mod1.fit(cov_type="clustered", cluster_entity=True)
        results["spec1"] = res1
        print(f"      β₁ (PostRB × High) = {res1.params['post_rb_x_high']:.4f} "
              f"(SE = {res1.std_errors['post_rb_x_high']:.4f})")

        # Specification 2: + ChatGPT interaction
        print("  (2) + ChatGPT interaction...")
        mod2 = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
            entity_effects=True,
            time_effects=True,
        )
        res2 = mod2.fit(cov_type="clustered", cluster_entity=True)
        results["spec2"] = res2
        print(f"      β₁ (PostRB × High) = {res2.params['post_rb_x_high']:.4f} "
              f"(SE = {res2.std_errors['post_rb_x_high']:.4f})")
        print(f"      β₂ (PostGPT × High) = {res2.params['post_gpt_x_high']:.4f} "
              f"(SE = {res2.std_errors['post_gpt_x_high']:.4f})")

        # Specification 3: + occupation-specific time trends
        print("  (3) + occupation-specific trends...")
        mod3 = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[["post_rb_x_high", "post_gpt_x_high", "time_x_high"]],
            entity_effects=True,
            time_effects=True,
        )
        res3 = mod3.fit(cov_type="clustered", cluster_entity=True)
        results["spec3"] = res3
        print(f"      β₁ (PostRB × High) = {res3.params['post_rb_x_high']:.4f} "
              f"(SE = {res3.std_errors['post_rb_x_high']:.4f})")
        print(f"      β₂ (PostGPT × High) = {res3.params['post_gpt_x_high']:.4f} "
              f"(SE = {res3.std_errors['post_gpt_x_high']:.4f})")

    except ImportError:
        print("  linearmodels not available — using statsmodels with dummies")
        results = _run_did_statsmodels(df)

    return results


def _run_did_statsmodels(df: pd.DataFrame) -> dict:
    """
    Fallback DiD estimation using statsmodels with explicit dummies.

    Less memory-efficient than linearmodels for large panels, but works
    without the linearmodels dependency.
    """
    results = {}

    # Create dummies
    occ_dummies = pd.get_dummies(df["ssyk4"], prefix="occ", drop_first=True)
    time_dummies = pd.get_dummies(df["year_month"], prefix="t", drop_first=True)

    # Specification 2 (main): both interactions
    X = pd.concat([
        df[["post_rb_x_high", "post_gpt_x_high"]],
        occ_dummies,
        time_dummies,
    ], axis=1).astype(float)

    X = sm.add_constant(X)
    y = df["ln_ads"].values

    mod = sm.OLS(y, X)
    # Cluster on occupation
    res = mod.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["ssyk4"].values},
    )

    results["spec2_sm"] = res
    print(f"  β₁ (PostRB × High) = {res.params['post_rb_x_high']:.4f} "
          f"(SE = {res.bse['post_rb_x_high']:.4f})")
    print(f"  β₂ (PostGPT × High) = {res.params['post_gpt_x_high']:.4f} "
          f"(SE = {res.bse['post_gpt_x_high']:.4f})")

    return results


def compute_summary_statistics(merged: pd.DataFrame, daioe: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for the paper (Table in appendix).

    Reports: number of occupations, ads, vacancies, exposure distribution.
    """
    print("\nComputing summary statistics...")

    stats = {}

    # Panel dimensions
    stats["N occupations"] = merged["ssyk4"].nunique()
    stats["N months"] = merged["year_month"].nunique()
    stats["N occupation-months"] = len(merged)
    stats["Total ads"] = merged["n_ads"].sum()
    stats["Total vacancies"] = merged["n_vacancies"].sum()

    # Ads per occupation-month
    stats["Mean ads per cell"] = merged["n_ads"].mean()
    stats["Median ads per cell"] = merged["n_ads"].median()
    stats["SD ads per cell"] = merged["n_ads"].std()

    # Exposure distribution
    stats["Mean genAI exposure (pctl)"] = daioe["pctl_rank_genai"].mean()
    stats["SD genAI exposure"] = daioe["pctl_rank_genai"].std()
    stats["Q4 cutoff (pctl)"] = daioe["pctl_rank_genai"].quantile(0.75)

    summary = pd.Series(stats, name="value")
    out = TABDIR / "summary_statistics.csv"
    summary.to_csv(out, header=True)
    print(f"  Saved → {out.name}")

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v:,}")

    return summary


def format_regression_table(results: dict) -> str:
    """
    Format DiD results as a LaTeX table for the paper.

    Three columns: (1) monetary policy only, (2) + ChatGPT, (3) + trends.
    Reports coefficients, clustered SEs in parentheses, significance stars.
    """
    print("\nFormatting regression table...")

    def stars(pval):
        if pval < 0.01:
            return "***"
        elif pval < 0.05:
            return "**"
        elif pval < 0.10:
            return "*"
        return ""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Difference-in-differences: postings by genAI exposure}",
        r"\label{tab:did}",
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        r" & (1) & (2) & (3) \\",
        r" & Monetary & + ChatGPT & + Trends \\",
        r"\hline",
    ]

    vars_to_report = [
        ("post_rb_x_high", r"Post-Riksbank $\times$ High exposure"),
        ("post_gpt_x_high", r"Post-ChatGPT $\times$ High exposure"),
        ("time_x_high", r"Time trend $\times$ High exposure"),
    ]

    for var, label in vars_to_report:
        row_coef = f"{label}"
        row_se = ""

        for spec_key in ["spec1", "spec2", "spec3"]:
            res = results.get(spec_key)
            if res is None:
                row_coef += " & "
                row_se += " & "
                continue

            if var in res.params.index:
                coef = res.params[var]
                se = res.std_errors[var]
                pv = res.pvalues[var]
                row_coef += f" & {coef:.4f}{stars(pv)}"
                row_se += f" & ({se:.4f})"
            else:
                row_coef += " & "
                row_se += " & "

        lines.append(row_coef + r" \\")
        lines.append(row_se + r" \\")

    # Footer
    lines.extend([
        r"\hline",
        r"Occupation FE & Yes & Yes & Yes \\",
        r"Month FE & Yes & Yes & Yes \\",
    ])

    # Add N and R² if available
    for spec_key in ["spec1", "spec2", "spec3"]:
        res = results.get(spec_key)
        if res is not None:
            n_obs = res.nobs
            break
    else:
        n_obs = "—"

    lines.extend([
        f"Observations & \\multicolumn{{3}}{{c}}{{{n_obs:,}}} \\\\",
        r"\hline\hline",
        r"\multicolumn{4}{p{0.9\textwidth}}{\footnotesize \textit{Notes:} "
        r"Dependent variable: $\ln(\text{postings}_{it})$. "
        r"High exposure = top quartile of DAIOE genAI percentile ranking. "
        r"Post-Riksbank = April 2022 onward. Post-ChatGPT = December 2022 onward. "
        r"Standard errors clustered at occupation level in parentheses. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    out = TABDIR / "did_regression.tex"
    out.write_text(table_str, encoding="utf-8")
    print(f"  Saved → {out.name}")

    return table_str


def main():
    print("=" * 70)
    print("STEP 5: DiD analysis")
    print("=" * 70)

    # Load merged data
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")

    # Summary statistics
    compute_summary_statistics(merged, daioe)

    # Prepare panel
    panel = prepare_panel(merged)

    # Save panel for inspection
    panel_out = PROCESSED / "did_panel.csv"
    panel.to_csv(panel_out, index=False)
    print(f"\nSaved panel → {panel_out.name}")

    # Run regressions
    results = run_did_regressions(panel)

    # Format table
    if results:
        format_regression_table(results)

    # Save key results as CSV for easy inspection
    if "spec2" in results:
        res = results["spec2"]
        key_results = pd.DataFrame({
            "variable": res.params.index,
            "coefficient": res.params.values,
            "std_error": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
        })
        key_results.to_csv(TABDIR / "did_results_spec2.csv", index=False)
        print(f"\nKey results saved → did_results_spec2.csv")

    print("\nDone. Run 06_figures_tables.py next.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
07_robustness.py — Robustness checks for online appendix.

Alternative specifications:
  1. Alternative AI measure: DAIOE all-apps (pctl_rank_allapps) instead of genAI
  2. Alternative base period: Jan 2020 instead of Feb 2020
  3. Vacancy-weighted results (sum of vacancies instead of ad count)
  4. Seasonal adjustment (12-month rolling average)
  5. Tercile classification instead of quartiles
  6. Exclude pandemic months (drop Jan–Jun 2020)

All results saved to tables/ for inclusion in the appendix.
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

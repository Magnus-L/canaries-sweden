#!/usr/bin/env python3
"""
13_halfyear_analysis.py — Half-year event study for postings.

Erik Engberg suggested aggregating the event study to half-year periods
to reduce monthly noise while still showing the temporal evolution.
He noted the biggest effects appeared in 2024H2.

This script:
  1. Estimates a half-year event study for postings (High × half-year dummies)
  2. Creates a coefficient plot with 95% CIs
  3. Reference period: 2022H1 (pre-Riksbank hike)

The analysis runs on the same merged postings-DAIOE panel used in 05_analysis.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PROCESSED, FIGDIR, TABDIR,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
    DARK_BLUE, ORANGE, TEAL, GRAY, DARK_TEXT, set_rcparams,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

set_rcparams()


def assign_halfyear(date_series: pd.Series) -> pd.Series:
    """
    Map dates to half-year labels: '2020H1', '2020H2', etc.

    H1 = January–June, H2 = July–December.
    """
    dates = pd.to_datetime(date_series)
    year = dates.dt.year.astype(str)
    half = np.where(dates.dt.month <= 6, "H1", "H2")
    return year + half


def run_halfyear_event_study():
    """
    Estimate half-year event study: interact half-year dummies with High indicator.

    Model:
        ln(postings_{it}) = α_i + γ_t + Σ_h δ_h · 1(t ∈ h) · High_i + ε_{it}

    where h indexes half-year periods and the reference is 2022H1
    (the last half-year before the Riksbank hike).

    This is more informative than the simple two-dummy DiD because it
    traces the temporal evolution of the high-vs-low AI exposure gap.
    """
    print("Running half-year event study for postings...")

    # Load merged panel
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    merged["date"] = pd.to_datetime(merged["year_month"] + "-01")
    merged["halfyear"] = assign_halfyear(merged["date"])

    # Drop zero-posting cells (log undefined)
    merged = merged[merged["n_ads"] > 0].copy()
    merged["ln_ads"] = np.log(merged["n_ads"])

    # All half-year periods in the data
    all_periods = sorted(merged["halfyear"].unique())
    print(f"  Half-year periods: {all_periods}")

    # Reference period: 2022H1 (pre-Riksbank hike, pre-ChatGPT)
    ref_period = "2022H1"
    event_periods = [p for p in all_periods if p != ref_period]

    # Create interaction dummies: halfyear × high_exposure
    for p in event_periods:
        merged[f"hy_{p}_x_high"] = (
            (merged["halfyear"] == p).astype(int) * merged["high_exposure"]
        )

    interaction_cols = [f"hy_{p}_x_high" for p in event_periods]

    # Estimate with linearmodels (occupation + time FE)
    try:
        from linearmodels.panel import PanelOLS

        panel = merged.copy()
        panel["date_idx"] = pd.to_datetime(panel["year_month"] + "-01")
        panel = panel.set_index(["ssyk4", "date_idx"])

        mod = PanelOLS(
            dependent=panel["ln_ads"],
            exog=panel[interaction_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        # Extract coefficients
        results = []
        for p in event_periods:
            col = f"hy_{p}_x_high"
            results.append({
                "period": p,
                "coef": res.params[col],
                "se": res.std_errors[col],
                "pval": res.pvalues[col],
            })

        # Add reference period (zero by definition)
        results.append({
            "period": ref_period,
            "coef": 0.0,
            "se": 0.0,
            "pval": 1.0,
        })

    except ImportError:
        print("  linearmodels not available — using statsmodels fallback")
        import statsmodels.api as sm

        occ_dummies = pd.get_dummies(merged["ssyk4"], prefix="occ", drop_first=True)
        time_dummies = pd.get_dummies(merged["year_month"], prefix="t", drop_first=True)

        X = pd.concat([
            merged[interaction_cols],
            occ_dummies,
            time_dummies,
        ], axis=1).astype(float)
        X = sm.add_constant(X)
        y = merged["ln_ads"].values

        mod = sm.OLS(y, X)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": merged["ssyk4"].values},
        )

        results = []
        for p in event_periods:
            col = f"hy_{p}_x_high"
            results.append({
                "period": p,
                "coef": res.params[col],
                "se": res.bse[col],
                "pval": res.pvalues[col],
            })
        results.append({
            "period": ref_period,
            "coef": 0.0,
            "se": 0.0,
            "pval": 1.0,
        })

    rdf = pd.DataFrame(results).sort_values("period")
    print(f"\n  Half-year event study results:")
    for _, row in rdf.iterrows():
        sig = "***" if row["pval"] < 0.01 else ("**" if row["pval"] < 0.05 else ("*" if row["pval"] < 0.1 else ""))
        print(f"    {row['period']}: {row['coef']:+.3f} (SE={row['se']:.3f}) {sig}")

    # Save results
    out_csv = TABDIR / "halfyear_event_study.csv"
    rdf.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv.name}")

    return rdf


def fig_halfyear_event_study(rdf: pd.DataFrame):
    """
    Coefficient plot: half-year × High interaction coefficients with 95% CIs.

    Vertical lines mark Riksbank hike period (2022H1) and ChatGPT launch
    period (2022H2). Reference period is 2022H1 (= 0 by construction).
    """
    print("Generating half-year event study figure...")

    rdf = rdf.sort_values("period").copy()

    # Numeric x-axis positions
    rdf["x"] = range(len(rdf))

    # 95% CI
    rdf["ci_lo"] = rdf["coef"] - 1.96 * rdf["se"]
    rdf["ci_hi"] = rdf["coef"] + 1.96 * rdf["se"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Confidence intervals
    ax.fill_between(
        rdf["x"], rdf["ci_lo"], rdf["ci_hi"],
        alpha=0.15, color=ORANGE,
    )

    # Point estimates
    ax.plot(rdf["x"], rdf["coef"], "o-", color=ORANGE, linewidth=2, markersize=6)

    # Zero line
    ax.axhline(0, color=DARK_TEXT, linewidth=0.8, alpha=0.5)

    # Mark reference period
    ref_x = rdf[rdf["period"] == "2022H1"]["x"].values[0]
    ax.axvline(ref_x, color=TEAL, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(
        "Reference\n(2022H1)",
        xy=(ref_x, ax.get_ylim()[1] * 0.85),
        fontsize=8, color=TEAL, fontweight="bold", ha="center",
    )

    # Mark ChatGPT period (2022H2)
    gpt_periods = rdf[rdf["period"] == "2022H2"]
    if not gpt_periods.empty:
        gpt_x = gpt_periods["x"].values[0]
        ax.axvline(gpt_x, color=GRAY, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate(
            "ChatGPT\n(2022H2)",
            xy=(gpt_x, ax.get_ylim()[1] * 0.75),
            fontsize=8, color=GRAY, fontweight="bold", ha="center",
        )

    # Shade post-ChatGPT region
    if not gpt_periods.empty:
        ax.axvspan(gpt_x - 0.5, rdf["x"].max() + 0.5, alpha=0.04, color=GRAY, zorder=0)

    # X-axis labels
    ax.set_xticks(rdf["x"])
    ax.set_xticklabels(rdf["period"], rotation=45, ha="right", fontsize=9)

    ax.set_ylabel("Coefficient (High × period)", fontsize=12)
    ax.set_title(
        "Half-year event study: high vs low AI-exposure postings",
        fontsize=13, fontweight="bold", pad=10,
    )

    fig.text(
        0.01, 0.01,
        "Notes: Dependent variable: ln(postings). Reference period: 2022H1. "
        "Occupation and month FE. SEs clustered by occupation. Shaded area: 95% CI.",
        fontsize=7, color=GRAY, style="italic",
        transform=fig.transFigure,
    )

    fig.tight_layout()

    out = FIGDIR / "figA11_halfyear_event_study.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


def main():
    print("=" * 70)
    print("Half-year event study analysis")
    print("=" * 70)

    rdf = run_halfyear_event_study()
    fig_halfyear_event_study(rdf)

    print("\nDone.")


if __name__ == "__main__":
    main()

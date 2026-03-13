#!/usr/bin/env python3
"""
12_create_figure2_age_gradient.py — Publication-quality age gradient figure.

Reads the employer-level DiD coefficients from canaries_did_results.csv
and creates Figure 2 for the main paper: the monotonic age gradient
showing how AI exposure affects employment differently by age group.

No MONA access needed — uses the exported regression results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Paths
    PROJECT = Path(__file__).resolve().parent.parent
    DATA = PROJECT / "data" / "mona" / "Output" / "Regression" / "canaries_did_results.csv"
    OUT = PROJECT / "figures" / "fig2_age_gradient.png"
    OUT_PDF = PROJECT / "figures" / "fig2_age_gradient.pdf"

    # Load DiD results
    df = pd.read_csv(DATA)

    # The ChatGPT coefficient (gamma2) is the one we plot
    # Convert log-point coefficient to exact percentage: 100 * (exp(coef) - 1)
    df["pct"] = 100 * (np.exp(df["gamma2_gpt_high"]) - 1)
    df["ci_lo"] = 100 * (np.exp(df["gamma2_gpt_high"] - 1.96 * df["se2"]) - 1)
    df["ci_hi"] = 100 * (np.exp(df["gamma2_gpt_high"] + 1.96 * df["se2"]) - 1)

    # Significance classification
    df["sig"] = df["pval2"].apply(
        lambda p: "p<0.01" if p < 0.01 else "p<0.05" if p < 0.05 else "n.s."
    )

    # Age group order and x-positions
    age_order = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
    df = df.set_index("age_group").loc[age_order].reset_index()
    x = np.arange(len(age_order))

    # Colours: orange for youngest two (significant negative), teal for rest
    # Using the paper's colour scheme
    ORANGE = "#E8873A"
    TEAL = "#2E7D6F"

    colours = []
    for _, row in df.iterrows():
        if row["pval2"] < 0.01 and row["pct"] < 0:
            colours.append(ORANGE)
        elif row["pval2"] < 0.01 and row["pct"] > 0:
            colours.append(TEAL)
        else:
            colours.append(TEAL)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Error bars (95% CI)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([x[i], x[i]], [row["ci_lo"], row["ci_hi"]],
                color=colours[i], linewidth=2.5, solid_capstyle="round")

    # Points
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(x[i], row["pct"], color=colours[i], s=100, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # Percentage labels
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row['pct']:+.1f}%"
        # Position label to avoid overlap with CI
        offset = -1.2 if row["pct"] < 0 else 0.8
        va = "top" if row["pct"] < 0 else "bottom"
        ax.annotate(label, (x[i], row["pct"] + offset),
                    ha="center", va=va, fontsize=9, fontweight="bold",
                    color=colours[i])

    # Zero line
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="-")

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(age_order, fontsize=11)
    ax.set_xlabel("Age group", fontsize=12)
    ax.set_ylabel("Employment change (%)", fontsize=12)
    ax.set_ylim(-9, 3.5)

    # Tick formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.tick_params(axis="both", labelsize=10)

    # Remove top and right spines for clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Note below figure
    ax.text(0.5, -0.15,
            r"Note: Employer$\times$quartile and employer$\times$month fixed effects. "
            "95% CI shown. Bold = p < 0.01.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, fontstyle="italic", color="#666666")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT}")
    print(f"Saved: {OUT_PDF}")

    # Print verification
    print("\nCoefficients used:")
    for _, row in df.iterrows():
        print(f"  {row['age_group']:6s}: \u03b3\u2082 = {row['gamma2_gpt_high']:+.4f} "
              f"(SE = {row['se2']:.4f}, p = {row['pval2']:.4f}) \u2192 {row['pct']:+.1f}%")


if __name__ == "__main__":
    main()

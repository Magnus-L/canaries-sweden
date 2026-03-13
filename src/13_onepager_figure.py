#!/usr/bin/env python3
"""
13_onepager_figure.py — Event study figure for policy one-pager.

Shows the accelerating divergence between young (22-25) and older (50+)
workers in AI-exposed occupations after ChatGPT's launch. This replaced
the earlier static dot plot (which showed the old 6.5% level effect)
with the corrected event study trajectory.

Input:  data/output/corrected_es_all_ref2022H1.csv
Output: figures/onepager_age_gradient.pdf / .png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# --- Colours (matching one-pager LaTeX theme) ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
DARK_BLUE = "#1B3A5C"
DARK_TEXT = "#2C2C2C"
LIGHT_GRAY = "#F0F0F0"


def main():
    # --- Paths ---
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data" / "output" / "corrected_es_all_ref2022H1.csv"
    OUT_PDF = ROOT / "figures" / "onepager_age_gradient.pdf"
    OUT_PNG = ROOT / "figures" / "onepager_age_gradient.png"
    OUT_PDF.parent.mkdir(exist_ok=True)

    # --- Load corrected event study data ---
    df = pd.read_csv(DATA)

    # Period ordering — 2022H1 is the reference period (coefficient = 0)
    period_order = [
        "2019H1", "2019H2", "2020H1", "2020H2", "2021H1", "2021H2",
        "2022H1", "2022H2", "2023H1", "2023H2", "2024H1", "2024H2", "2025H1"
    ]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Plot settings per age group
    # Orange for young (negative = concerning), teal for 50+ (positive = resilient)
    groups = {
        "22-25": {"color": ORANGE, "label": "22\u201325 year olds"},
        "50+":   {"color": TEAL,   "label": "50+ year olds"},
    }

    for age, cfg in groups.items():
        sub = df[df["age_group"] == age].copy()
        sub["period_idx"] = sub["period"].map({p: i for i, p in enumerate(period_order)})
        sub = sub.dropna(subset=["period_idx"]).sort_values("period_idx")

        # Convert log-point coefficients to percentage points for readability
        coef_pct = sub["coef"] * 100
        se_pct = sub["se"] * 100

        # Line with markers
        ax.plot(sub["period_idx"], coef_pct, "o-",
                color=cfg["color"], linewidth=2.2, markersize=5,
                label=cfg["label"], zorder=3)

        # 95% CI band
        ax.fill_between(sub["period_idx"],
                        coef_pct - 1.96 * se_pct,
                        coef_pct + 1.96 * se_pct,
                        color=cfg["color"], alpha=0.12)

    # --- Annotations: endpoint values for policy audience ---
    # 22-25 at 2025H1: -5.5%
    young_end = df[(df["age_group"] == "22-25") & (df["period"] == "2025H1")]
    if not young_end.empty:
        val = young_end["coef"].iloc[0] * 100
        x_pos = period_order.index("2025H1")
        ax.annotate(f"{val:.1f}%", (x_pos, val),
                    textcoords="offset points", xytext=(-38, -8),
                    fontsize=8.5, fontweight="bold", color=ORANGE)

    # 50+ at 2025H1: +1.3%
    old_end = df[(df["age_group"] == "50+") & (df["period"] == "2025H1")]
    if not old_end.empty:
        val = old_end["coef"].iloc[0] * 100
        x_pos = period_order.index("2025H1")
        ax.annotate(f"+{val:.1f}%", (x_pos, val),
                    textcoords="offset points", xytext=(-38, 8),
                    fontsize=8.5, fontweight="bold", color=TEAL)

    # --- Zero line ---
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    # --- ChatGPT launch marker (Nov 2022, between 2022H1 and 2022H2) ---
    chatgpt_x = period_order.index("2022H1") + 0.5
    ax.axvline(chatgpt_x, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(chatgpt_x + 0.15, ax.get_ylim()[0] * 0.15, "ChatGPT",
            fontsize=8, color="grey", ha="left", va="bottom")

    # --- X-axis ---
    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order, rotation=45, ha="right", fontsize=8.5)

    # --- Y-axis: approximate percentage change ---
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("Employment change (%)", fontsize=10,
                  color=DARK_TEXT)

    # --- Clean spines ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=DARK_TEXT, labelsize=9)

    # --- Subtle grid ---
    ax.yaxis.grid(True, alpha=0.2, linestyle="-")
    ax.set_axisbelow(True)

    # --- Title ---
    ax.set_title(
        "Employment in AI-exposed occupations\nafter ChatGPT launch, by age group",
        fontsize=12, fontweight="bold", color=DARK_BLUE, pad=12
    )

    # --- Legend ---
    ax.legend(fontsize=9.5, loc="lower left", frameon=False)

    # --- Note ---
    fig.text(0.98, -0.02,
             "Note: Employer\u00d7quartile and employer\u00d7month FE. "
             "Ref. period: 2022H1. 95% CI shown.",
             ha="right", fontsize=7.5, color="#888888", style="italic")

    fig.tight_layout()
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()

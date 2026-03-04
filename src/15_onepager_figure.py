#!/usr/bin/env python3
"""
15_onepager_figure.py — Age gradient figure for policy one-pager.

Creates a clean dot-with-CI plot showing the ChatGPT employment effect (γ₂)
by age group. The monotonic gradient from negative (young) to positive (old)
is the key visual message.

Input: data/mona/canaries_did_results.csv
Output: figures/onepager_age_gradient.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mona" / "canaries_did_results.csv"
OUT = ROOT / "figures" / "onepager_age_gradient.pdf"
OUT.parent.mkdir(exist_ok=True)

# --- Colours (from paper config) ---
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
DARK_BLUE = "#1B3A5C"
DARK_TEXT = "#2C2C2C"
LIGHT_GRAY = "#F0F0F0"

# --- Load data ---
df = pd.read_csv(DATA)

# Sort by age group order
age_order = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
df["sort"] = df["age_group"].map({a: i for i, a in enumerate(age_order)})
df = df.sort_values("sort")

# Extract coefficients — convert to percentage for readability
# exp(gamma) - 1 ≈ gamma for small values, but let's be precise
df["pct"] = (np.exp(df["gamma2_gpt_high"]) - 1) * 100
df["ci_lo"] = (np.exp(df["gamma2_gpt_high"] - 1.96 * df["se2"]) - 1) * 100
df["ci_hi"] = (np.exp(df["gamma2_gpt_high"] + 1.96 * df["se2"]) - 1) * 100

# --- Assign colours: negative = orange, positive = teal ---
colours = [ORANGE if v < 0 else TEAL for v in df["pct"]]

# --- Figure ---
fig, ax = plt.subplots(figsize=(7, 4))

x = np.arange(len(df))

# CI whiskers
for i, row in enumerate(df.itertuples()):
    ax.plot([i, i], [row.ci_lo, row.ci_hi], color=colours[i],
            linewidth=2.5, solid_capstyle="round", zorder=2)

# Dots
ax.scatter(x, df["pct"], color=colours, s=100, zorder=3, edgecolors="white",
           linewidth=1.2)

# Zero line
ax.axhline(0, color=DARK_TEXT, linewidth=0.8, linestyle="-", alpha=0.4, zorder=1)

# Labels on each dot
for i, row in enumerate(df.itertuples()):
    offset = -1.2 if row.pct < 0 else 0.8
    weight = "bold" if row.pval2 < 0.01 else "normal"
    ax.annotate(f"{row.pct:+.1f}%", (i, row.pct + offset),
                ha="center", va="center", fontsize=9, fontweight=weight,
                color=colours[i])

# Axes
ax.set_xticks(x)
ax.set_xticklabels(df["age_group"], fontsize=11)
ax.set_xlabel("Age group", fontsize=11, color=DARK_TEXT)
ax.set_ylabel("Employment change (%)", fontsize=11, color=DARK_TEXT)

# y-axis as percentage
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))

# Clean up
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(colors=DARK_TEXT, labelsize=10)

# Subtle grid
ax.yaxis.grid(True, alpha=0.2, linestyle="-")
ax.set_axisbelow(True)

# Title — descriptive for policy audience
ax.set_title(
    "Relative employment change in AI-exposed occupations\nafter ChatGPT launch",
    fontsize=12, fontweight="bold", color=DARK_BLUE, pad=12
)

# Note
fig.text(0.98, -0.02,
         "Note: Employer×quartile and employer×month fixed effects. "
         "95% CI shown. Bold = p < 0.01.",
         ha="right", fontsize=7.5, color="#888888", style="italic")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {OUT}")
print(f"Saved: {OUT.with_suffix('.png')}")

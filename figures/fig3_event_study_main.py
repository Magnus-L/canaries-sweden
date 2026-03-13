"""
Figure 3 for main paper: Event study of employment effects by age group.
Shows 22-25 and 50+ age groups with 95% confidence intervals.
Reference period: 2022H1. Coefficients as approximate percentage changes.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Load corrected event study coefficients
data_dir = Path(__file__).parent.parent / "data" / "output"
df = pd.read_csv(data_dir / "corrected_es_all_ref2022H1.csv")

# Period ordering for x-axis
period_order = [
    "2019H1", "2019H2", "2020H1", "2020H2", "2021H1", "2021H2",
    "2022H1", "2022H2", "2023H1", "2023H2", "2024H1", "2024H2", "2025H1"
]

fig, ax = plt.subplots(figsize=(10, 5.5))

# Colours: orange for young, dark blue for 50+
colors = {"22-25": "#d35400", "50+": "#1a3a5c"}
labels = {"22-25": "22–25 year olds", "50+": "50+ year olds"}

for age in ["22-25", "50+"]:
    sub = df[df["age_group"] == age].copy()
    sub["period_idx"] = sub["period"].map({p: i for i, p in enumerate(period_order)})
    sub = sub.dropna(subset=["period_idx"]).sort_values("period_idx")

    # Convert to approximate percentage change for readability
    coef_pct = sub["coef"] * 100
    se_pct = sub["se"] * 100

    ax.plot(sub["period_idx"], coef_pct, "o-", color=colors[age],
            linewidth=2.2, markersize=5, label=labels[age], zorder=3)
    ax.fill_between(sub["period_idx"],
                    coef_pct - 1.96 * se_pct,
                    coef_pct + 1.96 * se_pct,
                    color=colors[age], alpha=0.12)

# Reference line at zero
ax.axhline(0, color="black", linewidth=0.6)

# ChatGPT launch marker (between 2022H1 and 2022H2)
chatgpt_x = period_order.index("2022H1") + 0.5
ax.axvline(chatgpt_x, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(chatgpt_x + 0.15, ax.get_ylim()[0] + 0.3, "ChatGPT", fontsize=8,
        color="grey", ha="left", va="bottom")

# X-axis
ax.set_xticks(range(len(period_order)))
ax.set_xticklabels(period_order, rotation=45, ha="right", fontsize=8.5)

# Y-axis: approximate percentage change
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylabel("Employment change (%)")

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(fontsize=10, loc="lower left", frameon=False)

plt.tight_layout()

# Save
out = Path(__file__).parent
fig.savefig(out / "fig3_event_study_main.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "fig3_event_study_main.pdf", bbox_inches="tight")
print(f"Saved to {out / 'fig3_event_study_main.png'}")

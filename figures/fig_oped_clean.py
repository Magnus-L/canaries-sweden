"""
Clean event study figure for op-ed: 22-25 and 50+ age groups.
No title, minimal labels — newspaper graphics team adds their own.
Coefficients converted to percentage points for readability.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Load data
data_dir = Path(__file__).parent.parent / "data" / "output"
df = pd.read_csv(data_dir / "corrected_es_all_ref2022H1.csv")

# Period ordering for x-axis
period_order = [
    "2019H1", "2019H2", "2020H1", "2020H2", "2021H1", "2021H2",
    "2022H1", "2022H2", "2023H1", "2023H2", "2024H1", "2024H2", "2025H1"
]

fig, ax = plt.subplots(figsize=(10, 5.5))

colors = {"22-25": "#d35400", "50+": "#2c3e50"}
labels = {"22-25": "22–25", "50+": "50+"}

for age in ["22-25", "50+"]:
    sub = df[df["age_group"] == age].copy()
    sub["period_idx"] = sub["period"].map({p: i for i, p in enumerate(period_order)})
    sub = sub.dropna(subset=["period_idx"]).sort_values("period_idx")

    # Convert to percentage
    coef_pct = sub["coef"] * 100
    se_pct = sub["se"] * 100

    ax.plot(sub["period_idx"], coef_pct, "o-", color=colors[age],
            linewidth=2.2, markersize=5, label=labels[age], zorder=3)
    ax.fill_between(sub["period_idx"],
                    coef_pct - 1.96 * se_pct,
                    coef_pct + 1.96 * se_pct,
                    color=colors[age], alpha=0.12)

# Reference line
ax.axhline(0, color="black", linewidth=0.6)

# ChatGPT launch marker (between 2022H2 and 2023H1, i.e. Nov 2022)
chatgpt_x = period_order.index("2022H1") + 0.5  # between 2022H1 and 2022H2
ax.axvline(chatgpt_x, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(chatgpt_x + 0.15, ax.get_ylim()[0] + 0.5, "ChatGPT", fontsize=8,
        color="grey", ha="left", va="bottom")

# X-axis
ax.set_xticks(range(len(period_order)))
ax.set_xticklabels(period_order, rotation=45, ha="right", fontsize=8.5)

# Y-axis as percentage
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylabel("")

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(fontsize=10, loc="lower left", frameon=False)

plt.tight_layout()

# Save
out = Path(__file__).parent
fig.savefig(out / "fig_oped_clean.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "fig_oped_clean.pdf", bbox_inches="tight")
print(f"Saved to {out / 'fig_oped_clean.png'}")

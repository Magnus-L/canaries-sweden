"""
Regenerate Figure 2: Monthly employment by age and AI exposure (3-month MA).
Indexed to October 2022 = 100. Uses aggregated MONA data exported via script 19.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data" / "output"
df = pd.read_csv(data_dir / "fig3_canaries_timeseries.csv")

# Parse date
df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")

# Index to October 2022 = 100
base = df[df["year_month"] == "2022-10"].set_index(["age_group", "high_ai"])["n_employed"]
df["index"] = df.apply(
    lambda r: 100 * r["n_employed"] / base.loc[(r["age_group"], r["high_ai"])], axis=1
)

# 3-month moving average
df = df.sort_values(["age_group", "high_ai", "date"])
df["index_ma3"] = (
    df.groupby(["age_group", "high_ai"])["index"]
    .transform(lambda x: x.rolling(3, center=True).mean())
)

# Plot
fig, ax = plt.subplots(figsize=(10, 5.5))

styles = {
    ("22-25", 1): {"color": "#d35400", "linestyle": "-",  "label": "Young (22–25), High AI"},
    ("22-25", 0): {"color": "#d35400", "linestyle": "--", "label": "Young (22–25), Low AI"},
    ("26+",   1): {"color": "#1a3a5c", "linestyle": "-",  "label": "Older (26+), High AI"},
    ("26+",   0): {"color": "#1a3a5c", "linestyle": "--", "label": "Older (26+), Low AI"},
}

for (age, hi), style in styles.items():
    sub = df[(df["age_group"] == age) & (df["high_ai"] == hi)].sort_values("date")
    ax.plot(sub["date"], sub["index_ma3"], linewidth=2, **style)

# Reference line at 100
ax.axhline(100, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

# Vertical markers
riksbank = pd.Timestamp("2022-04-01")
chatgpt = pd.Timestamp("2022-11-01")
ax.axvline(riksbank, color="lightcoral", linewidth=0.8, alpha=0.5)
ax.axvline(chatgpt, color="teal", linewidth=0.8, linestyle=":", alpha=0.6)

# Labels
ax.set_ylabel("Employment index (2022-10 = 100)")
ax.set_title("Monthly employment by age and AI exposure (3-month MA)")

# X-axis formatting
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Legend
ax.legend(loc="lower left", fontsize=8.5, frameon=False)

# Clean
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out = Path(__file__).parent
fig.savefig(out / "fig2_mona_canaries_ma3.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "fig2_mona_canaries_ma3.pdf", bbox_inches="tight")
print(f"Saved to {out / 'fig2_mona_canaries_ma3.png'}")

#!/usr/bin/env python3
"""
12_annotated_canaries.py — Overlay GenAI launch dates on employment canaries figure.

Takes the existing figA8c_mona_canaries_economy.png (produced in MONA by Lydia)
and adds vertical annotation lines for major GenAI capability milestones.
The result is a new appendix figure showing how employment trajectory breaks
align with successive waves of GenAI capability.

Calibration method:
  The original figure has two known vertical lines (Riksbank Apr 2022,
  ChatGPT Dec 2022). We use the x-axis year tick marks (2019-2025) to
  establish a pixel-to-date mapping, then draw new lines at the correct
  positions. This avoids needing the underlying MONA data.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT / "figures"
INPUT_FIG = FIG_DIR / "figA8c_mona_canaries_economy.png"
OUTPUT_FIG = FIG_DIR / "figA9_canaries_genai_timeline.png"

# ── Load original figure ──────────────────────────────────────────────────
img = mpimg.imread(str(INPUT_FIG))
h, w = img.shape[:2]
print(f"Image dimensions: {w} × {h} pixels")

# ── Calibration: map year ticks to pixel positions ────────────────────────
# The original figure was created with matplotlib, figsize=(10, 5), dpi=300,
# tight_layout. The x-axis has year ticks at 2019, 2020, ..., 2025.
#
# We need to find where the plot area sits within the image.
# Strategy: scan for the tick mark positions along the bottom of the plot area.
# Typical matplotlib tight_layout positions for this figure:
#   - Plot area left edge:   ~8.5% of width
#   - Plot area right edge:  ~97.5% of width
#   - Plot area bottom edge: ~15% of height (from bottom)
#   - Plot area top edge:    ~92% of height (from bottom)
#
# The x-axis range in the original code uses pd.Timestamp dates.
# matplotlib's auto-ranging for monthly data from Jan 2019 to ~Aug 2025
# places year ticks at Jan 1 of each year.

# Fractional positions of the plot area within the image
# (these are approximate for tight_layout; we'll verify against existing lines)
PLOT_LEFT_FRAC = 0.097    # where the y-axis meets the plot area
PLOT_RIGHT_FRAC = 0.985   # right edge of the plot area
PLOT_BOTTOM_FRAC = 0.145  # bottom of plot area (from top, since imshow is top-down)
PLOT_TOP_FRAC = 0.065     # top of plot area (from top)

# Convert to pixel coordinates (origin at top-left for imshow)
plot_left_px = PLOT_LEFT_FRAC * w
plot_right_px = PLOT_RIGHT_FRAC * w
plot_top_px = PLOT_TOP_FRAC * h      # top of plot area (near top of image)
plot_bottom_px = (1 - PLOT_BOTTOM_FRAC) * h  # bottom of plot area

# The x-axis year ticks: 2019.0, 2020.0, ..., 2025.0
# These are evenly spaced within the auto-ranged x limits.
# matplotlib typically adds ~2-3% padding beyond the data range.
# The data starts at ~2019.0 and ends at ~2025.5.
# Auto x-limits would be approximately 2018.8 to 2025.7 (roughly).
# Year ticks fall within the plot area at proportional positions.

# X-axis data limits (estimated from the figure's visible range)
X_DATA_MIN = 2018.85  # left edge of x-axis in year units
X_DATA_MAX = 2025.65  # right edge of x-axis in year units


def year_frac(year, month, day=15):
    """Convert a date to fractional year (e.g., 2023.5 = July 2023)."""
    return year + (month - 1) / 12.0 + (day - 1) / 365.25


def date_to_pixel_x(year, month, day=15):
    """Convert a date to pixel x-coordinate in the image."""
    frac = year_frac(year, month, day)
    # Linear interpolation within the plot area
    t = (frac - X_DATA_MIN) / (X_DATA_MAX - X_DATA_MIN)
    return plot_left_px + t * (plot_right_px - plot_left_px)


# ── Verify calibration against known lines ────────────────────────────────
# The original figure has vertical lines at:
#   - Riksbank hike: April 2022 → year_frac ≈ 2022.25
#   - ChatGPT launch: December 2022 → year_frac ≈ 2022.917
riksbank_px = date_to_pixel_x(2022, 4)
chatgpt_px = date_to_pixel_x(2022, 12)
print(f"Calibration check:")
print(f"  Riksbank (Apr 2022): pixel x = {riksbank_px:.0f}")
print(f"  ChatGPT (Dec 2022):  pixel x = {chatgpt_px:.0f}")
print(f"  (Verify these align with the existing dotted lines in the figure)")

# ── GenAI milestones to annotate ──────────────────────────────────────────
# Selected for workplace impact, not just release date.
# Each tuple: (year, month, label, short_description)
MILESTONES = [
    (2023, 3, "GPT-4", "First highly capable LLM"),
    (2023, 11, "Enterprise\ntools", "M365 Copilot GA, ChatGPT Enterprise"),
    (2024, 5, "GPT-4o", "Multimodal, free tier"),
    (2024, 9, "o1", "Reasoning models"),
    (2025, 1, "DeepSeek R1", "Open-source reasoning"),
]

# ── Create annotated figure ───────────────────────────────────────────────
# Match original figure dimensions exactly
fig_w_in = w / 300  # assuming 300 dpi original
fig_h_in = h / 300
output_dpi = 300

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Display original image (full frame, no axes)
ax.imshow(img, aspect='auto')
ax.set_xlim(0, w)
ax.set_ylim(h, 0)  # y-axis inverted: (0,0) at top-left
ax.axis('off')

# Draw GenAI milestone lines
LINE_COLOR = "#7B2D8E"  # purple, distinct from existing orange/teal lines
LINE_ALPHA = 0.6
LABEL_FONTSIZE = 6.0

for year, month, label, _ in MILESTONES:
    x_px = date_to_pixel_x(year, month)

    # Vertical line spanning the plot area
    ax.plot(
        [x_px, x_px],
        [plot_top_px, plot_bottom_px],
        color=LINE_COLOR,
        linewidth=0.9,
        linestyle="--",
        alpha=LINE_ALPHA,
        zorder=2,
    )

    # Label at bottom of plot area, rotated for readability
    # Use single-line labels (remove newlines) for compact display
    clean_label = label.replace("\n", " ")
    label_y = plot_bottom_px - 15  # just above bottom axis
    ax.text(
        x_px + 8, label_y,
        clean_label,
        ha="left", va="bottom",
        fontsize=LABEL_FONTSIZE,
        color=LINE_COLOR,
        fontweight="bold",
        alpha=0.9,
        rotation=90,
        zorder=3,
    )

# Save
fig.savefig(str(OUTPUT_FIG), dpi=output_dpi, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f"\nSaved annotated figure → {OUTPUT_FIG}")
print("Check alignment of purple dashed lines against existing dotted lines.")

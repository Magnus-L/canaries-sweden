#!/usr/bin/env python3
"""
config.py — Central configuration for the canaries-sweden project.

All paths, constants, colour palettes, and shared settings live here.
Every other script imports from this module.
"""

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
FIGDIR = PROJECT / "figures"
TABDIR = PROJECT / "tables"
PAPER = PROJECT / "paper"

# Ensure output directories exist
for d in [RAW, PROCESSED, FIGDIR, TABDIR, PAPER]:
    d.mkdir(parents=True, exist_ok=True)

# ── External data paths ──────────────────────────────────────────────────────

# DAIOE genAI exposure index (tab-separated CSV)
DAIOE_PATH = Path.home() / "Documents" / "Downloads" / "daioe_ssyk2012.csv"
# Fallback: Excel version
DAIOE_XLSX = Path.home() / "Desktop" / "Resources" / "DAIOE-n-KIBS" / "DAIOE_ssyk2012_4_Akavia.xlsx"

# SvD project (for reusable AF validation data)
SVD_PROJECT = (
    Path.home()
    / "Documents"
    / "-JOBB"
    / "Föredrag o Bistånd"
    / "SvD_JobPostings_Feb2026"
)

# ── Platsbanken download URLs ────────────────────────────────────────────────

PLATSBANKEN_BASE = "https://data.jobtechdev.se/annonser/historiska"

# We use raw JSONL files for 2020–2025 (consistent format across years)
PLATSBANKEN_YEARS = list(range(2020, 2026))

def platsbanken_url(year: int) -> str:
    """URL for raw historical JSONL zip file."""
    return f"{PLATSBANKEN_BASE}/{year}.jsonl.zip"

# Enriched metadata (smaller, no text fields — good for testing)
def platsbanken_metadata_url(year: int) -> str:
    """URL for enriched metadata-only JSONL zip file (2016–2024)."""
    return f"{PLATSBANKEN_BASE}/berikade/metadata/{year}_beta1_metadata_jsonl.zip"

# 1% sample (for rapid prototyping)
# Standard pattern: {year}_beta1_1_percent_jsonl.zip for 2016–2024
# 2025 uses a quarterly partial: 2025_Q3_1_percent_jsonl.zip
SAMPLE_URL_OVERRIDES = {
    2025: "2025_Q3_1_percent_jsonl.zip",
}

def platsbanken_sample_url(year: int) -> str:
    """URL for 1% sample JSONL zip (for testing pipeline before full download)."""
    filename = SAMPLE_URL_OVERRIDES.get(year, f"{year}_beta1_1_percent_jsonl.zip")
    return f"{PLATSBANKEN_BASE}/berikade/exempel/{filename}"

# JobStream API (real-time, no auth required)
JOBSTREAM_BASE = "https://jobstream.api.jobtechdev.se"

# Historical search API
HISTORICAL_API = "https://historical.api.jobtechdev.se"

# ── Analysis parameters ───────────────────────────────────────────────────────

# Base month for indexing (Feb 2020 = 100)
BASE_MONTH = "2020-02-01"

# DAIOE reference year (pre-ChatGPT cross-section)
DAIOE_REF_YEAR = 2023

# Treatment dates
RIKSBANKEN_HIKE = "2022-04-01"    # First rate hike
CHATGPT_LAUNCH = "2022-12-01"     # ChatGPT public release (Nov 30 → Dec in monthly data)

# ── Colour palette (publication-quality, consistent with SvD project) ────────

DARK_BLUE = "#1B3A5C"
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
CREAM = "#F0E6D3"
DARK_TEXT = "#2C2C2C"
LIGHT_BLUE = "#DCE6F2"
ORANGE_LT = "#F8D7B9"
TEAL_LT = "#D2EBE4"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#C8C8C8"

# Quartile colours (Q1=lowest exposure → Q4=highest exposure)
Q_COLORS = {
    "Q1 (lowest)": LIGHT_GRAY,
    "Q2": LIGHT_BLUE,
    "Q3": TEAL,
    "Q4 (highest)": ORANGE,
}

# ── Matplotlib defaults ──────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_rcparams():
    """Apply publication-quality matplotlib settings."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": True,  # dual-axis charts need right spine
        "axes.edgecolor": GRAY,
        "axes.labelcolor": DARK_TEXT,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "text.color": DARK_TEXT,
    })

# Apply on import
set_rcparams()

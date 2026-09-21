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


# ----------------------------------------------------------------------
# POSTINGS SOURCE AND CUTOFF (added 21 Sep 2026)
#
# Two defects made the manuscript's own figures carry a false 2026
# collapse, and both are fixed here rather than in nine call sites.
#
# SOURCE. 01_download fetches a JobStream /v2/snapshot and 02_process
# splices it onto the bulk series. JobStream returns ads CURRENTLY
# PUBLISHED, not everything ever published, so a recent month is
# undercounted as its ads expire: the 24 February build shows 40,733 ads
# in December 2025 and 10,790 in January 2026, an 85 per cent drop no
# labour market produces. JobTech now publishes closed-quarter bulk
# files, and the revision rebuilt the merged panel from them as
# postings_daioe_merged_extended.csv, which runs to 2026-06 with
# sensible volumes. Prefer it wherever it exists.
#
# CUTOFF. 06_figures_tables applied a 2020-01-01 lower bound in three
# places and NO upper bound, so any stale input reached the right edge
# of a figure. POSTINGS_END is now enforced by the loaders below.
# ----------------------------------------------------------------------

POSTINGS_END = "2026-06"          # last month any posting series may show

_MERGED_PREFERRED = "postings_daioe_merged_extended.csv"
_MERGED_FALLBACK = "postings_daioe_merged.csv"


def load_postings_merged(end: str = None):
    """The merged SSYK4 x month panel, bulk-sourced where available."""
    import pandas as pd
    path = PROCESSED / _MERGED_PREFERRED
    if not path.exists():
        path = PROCESSED / _MERGED_FALLBACK
        print(f"  WARNING: {_MERGED_PREFERRED} missing; falling back to "
              f"{_MERGED_FALLBACK}, which may carry a JobStream tail")
    df = pd.read_csv(path)
    end = end or POSTINGS_END
    return df[df["year_month"] <= end].copy()


def _index_to_base(df, group_cols, base_month="2020-01"):
    """n_ads indexed to 100 at base_month, matching 04's own construction."""
    import pandas as pd
    out = df.copy()
    out["date"] = pd.to_datetime(out["year_month"] + "-01")
    keys = group_cols + ["year_month", "date"]
    agg = (out.groupby(keys, as_index=False)
           .agg(n_ads=("n_ads", "sum"),
                n_vacancies=("n_vacancies", "sum"))
           if "n_vacancies" in out.columns else
           out.groupby(keys, as_index=False).agg(n_ads=("n_ads", "sum")))
    if group_cols:
        base = (agg[agg["year_month"] == base_month]
                .set_index(group_cols)["n_ads"])
        agg["ads_idx"] = agg.apply(
            lambda r: 100.0 * r["n_ads"] / base.loc[tuple(r[c] for c in
                                                          group_cols)
                                                    if len(group_cols) > 1
                                                    else r[group_cols[0]]],
            axis=1)
    else:
        b = float(agg.loc[agg["year_month"] == base_month, "n_ads"].iloc[0])
        agg["ads_idx"] = 100.0 * agg["n_ads"] / b
    return agg.sort_values("date").reset_index(drop=True)


def load_postings_indexed(by_quartile: bool = True, end: str = None):
    """
    The indexed posting series, DERIVED from the merged panel.

    The stored postings_quartile_indexed.csv and postings_total_indexed.csv
    are the 24 February build and were never regenerated from the bulk
    quarters, so they still end at the JobStream tail. Deriving them here
    keeps one source of truth.
    """
    m = load_postings_merged(end=end)
    return _index_to_base(m, ["exposure_quartile"] if by_quartile else [])


def load_postings_ssyk4(end: str = None):
    """
    The pre-merge SSYK4 x month counts, with the same cutoff.

    This file is also a 24 February build and has no extended twin, so
    the cutoff is the only protection available until 02_process stops
    splicing the JobStream snapshot.
    """
    import pandas as pd
    df = pd.read_csv(PROCESSED / "postings_ssyk4_monthly.csv")
    end = end or POSTINGS_END
    return df[df["year_month"] <= end].copy()

#!/usr/bin/env python3
"""
config.py -- v2 revision configuration. Self-contained: every input lives under
the project tree (fixes defects D1 and D2 from the code-read digest; the v1
config pointed at ~/Downloads, ~/Desktop, /tmp and the deleted -JOBB tree).

The v1 processed files are REUSED read-only where the revision does not change
them (postings_ssyk4_monthly.csv etc.); v2 outputs go to revision/output,
revision/tables, revision/figures so the submitted version's outputs survive.
"""

from pathlib import Path

# -- Project tree ------------------------------------------------------------
PROJECT   = Path(__file__).resolve().parent.parent   # projects/canaries-sweden
REVISION  = PROJECT / "revision"
RAW       = PROJECT / "data" / "raw"                 # shared with v1 (read-only here)
PROCESSED = PROJECT / "data" / "processed"           # v1 outputs, read-only here
V2_OUT    = REVISION / "output"
V2_TAB    = REVISION / "tables"
V2_FIG    = REVISION / "figures"

for d in (V2_OUT, V2_TAB, V2_FIG):
    d.mkdir(parents=True, exist_ok=True)

# -- Inputs, all inside the tree --------------------------------------------
DAIOE_RAW          = RAW / "daioe_ssyk2012.csv"        # tab-separated, vendored
DAIOE_QUARTILES    = PROCESSED / "daioe_quartiles.csv" # built by src/04
BLS_SOC_ISCO_XLS   = RAW / "isco_soc_crosswalk2.xls"   # single home (was also /tmp)
SCB_SSYK_ISCO_XLSX = RAW / "ssyk2012_isco08.xlsx"
DINGEL_NEIMAN_CSV  = RAW / "dingel_neiman_telework.csv"

PLATSBANKEN_YEARS = list(range(2020, 2026))

def platsbanken_zip(year: int) -> Path:
    """Raw JSONL zip for one year; falls back to the 1% sample for testing."""
    full = RAW / f"{year}.jsonl.zip"
    return full if full.exists() else RAW / f"{year}_sample.jsonl.zip"

# -- Analysis constants (unchanged from v1 where the paper fixed them) -------
BASE_MONTH      = "2020-02-01"
DAIOE_REF_YEAR  = 2023
RIKSBANKEN_HIKE = "2022-04-01"
CHATGPT_LAUNCH  = "2022-12-01"

# Descriptive posting series are cut here: the final two collection months are
# harvesting artefacts (see reference_canaries_postings_tail_artefact).
POSTINGS_DESCRIPTIVE_END = "2025-12"

# -- Colours (as v1) ---------------------------------------------------------
DARK_BLUE = "#1B3A5C"; ORANGE = "#E8873A"; TEAL = "#2E7D6F"
GRAY = "#8C8C8C"; LIGHT_GRAY = "#C8C8C8"; DARK_TEXT = "#2C2C2C"

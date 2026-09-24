"""
config.py: paths and analysis constants shared by every script outside MONA.

Every script in packs 1, 2, 4 and 5 imports this module, so a replicator edits
paths in one place only. Three inputs are large or live outside the package and
can be pointed at an existing copy through environment variables:

    CANARIES_JOBADS_DIR   folder holding the Platsbanken archives
                          (2020.jsonl.zip ... 2025.jsonl.zip, 2026-Q1.jsonl.zip,
                          2026-Q2.jsonl.zip); default data/raw/platsbanken/
    CANARIES_FIRM_CUBE    employer-by-month-by-occupation advertisement counts
                          (firm_month_v2.csv.gz, see README, "Data availability")
    CANARIES_SCB_BULK     Statistics Sweden's business-register bulk file
                          (scb_bulkfil.zip, distributed by Bolagsverket)
    CANARIES_PAPER_DIR    the manuscript folder, used only by the verification
                          scripts in 0_verification/ to compare output with print

Nothing in the register tier (3_register_mona/) reads this file: the scripts
that run inside Statistics Sweden's MONA environment take their paths from
3_register_mona/scripts/mona_common.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# -- The package tree ----------------------------------------------------------
PACKAGE = Path(__file__).resolve().parent

DATA = PACKAGE / "data"
RAW = DATA / "raw"                 # public inputs, small ones shipped (DATA-MANIFEST.csv)
PROCESSED = DATA / "processed"     # built by 1_data_public/

OUTPUT = PACKAGE / "output"
TABLES = OUTPUT / "tables"         # the LaTeX tables the manuscript inputs
FIGURES = OUTPUT / "figures"       # the figures the manuscript includes
RESULTS = OUTPUT / "results"       # estimates and diagnostics behind them (CSV, text)

# Aggregated exports brought out of MONA, one folder per export run
# (3_register_mona/exports/EXPORT_RUNS.csv lists every file with its SHA-256).
EXPORTS = PACKAGE / "3_register_mona" / "exports"

for _d in (RAW, PROCESSED, TABLES, FIGURES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# -- Inputs that may live outside the package -----------------------------------
JOBADS_DIR = Path(os.environ.get("CANARIES_JOBADS_DIR", RAW / "platsbanken"))
FIRM_CUBE = Path(os.environ.get("CANARIES_FIRM_CUBE", RAW / "firm_month_v2.csv.gz"))
SCB_BULK = Path(os.environ.get("CANARIES_SCB_BULK", RAW / "scb_bulkfil.zip"))
PAPER_DIR = Path(os.environ.get(
    "CANARIES_PAPER_DIR", PACKAGE.parent.parent / "canaries-sweden-paper"))

# -- Public inputs shipped with the package --------------------------------------
DAIOE_RAW = RAW / "daioe_ssyk2012.csv"            # DAIOE, SSYK 2012, tab-separated
BLS_SOC_ISCO_XLS = RAW / "isco_soc_crosswalk2.xls"  # BLS SOC 2010 to ISCO-08
SCB_SSYK_ISCO_XLSX = RAW / "ssyk2012_isco08.xlsx"   # SCB SSYK 2012 to ISCO-08
DINGEL_NEIMAN_CSV = RAW / "dingel_neiman_telework.csv"
DAIOE_QUARTILES = PROCESSED / "daioe_quartiles.csv"  # built by 1_data_public/04

PLATSBANKEN_YEARS = list(range(2020, 2026))       # annual archives, 2020 to 2025
PLATSBANKEN_QUARTERS = ["2026-Q1", "2026-Q2"]     # closed-quarter archives


def platsbanken_zip(stem: str | int) -> Path:
    """Path of one Platsbanken archive, e.g. platsbanken_zip(2021) or ("2026-Q1")."""
    return JOBADS_DIR / f"{stem}.jsonl.zip"


# -- Analysis constants fixed by the paper ---------------------------------------
BASE_MONTH = "2020-02-01"          # indices are 100 in February 2020
DAIOE_REF_YEAR = 2023              # the DAIOE cross-section used throughout
RIKSBANKEN_HIKE = "2022-04-01"     # first Riksbank rate rise
CHATGPT_LAUNCH = "2022-12-01"      # first full month after the launch

# Descriptive series built from the submitted version's files stop in December
# 2025. Those files appended months from JobTech's live feed (JobStream), which
# returns only the advertisements still published on the day of extraction, so
# any recent month is under-counted. No month from the live feed is used.
POSTINGS_DESCRIPTIVE_END = "2025-12"
# The posting regressions run to the end of the newest closed-quarter archive.
POSTINGS_REGRESSION_END = "2026-06"

# -- Colours used by every figure -------------------------------------------------
DARK_BLUE = "#1B3A5C"
ORANGE = "#E8873A"
TEAL = "#2E7D6F"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#C8C8C8"
DARK_TEXT = "#2C2C2C"

# -- Posting margin: the frozen occupation-by-month counts -----------------------
# The occupation-by-month advertisement counts for October 2019 to February 2026
# that every posting estimate starts from, as built on 24 February 2026 from the
# annual archives. That build also appended the live feed (JobStream), whose
# extraction was not kept, so the file cannot be rebuilt byte for byte from the
# archives; 1_data_public/02_process_platsbanken.py rebuilds it from the archives
# and reports the difference. Point CANARIES_POSTINGS_SSYK4 at
# data/processed/postings_ssyk4_monthly.csv to run everything on the rebuild.
POSTINGS_SSYK4 = Path(os.environ.get(
    "CANARIES_POSTINGS_SSYK4", RAW / "postings_ssyk4_monthly_2026-02-24.csv"))

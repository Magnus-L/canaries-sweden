"""Assemble MANIFEST.csv from the main-text rows, the online-appendix rows and one
identity row per generated table (exhibit numbers as the manuscript prints them)."""
import csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
TABLES = [("Table 1", "table1_headline_v3"), ("Table A1", "top_bottom_occupations"),
          ("Table A2", "tableI2_sumstats_employment"), ("Table A3", "tableI2b_sumstats_postings_v3"),
          ("Table A4", "postings_accounting"), ("Table A5", "coverage_by_source"),
          ("Table A6", "postings_extended"), ("Table A7", "postings_seasonality"),
          ("Table A8", "postings_deciles"), ("Table A9", "tableA_descriptive_bands"),
          ("Table A11", "tableA_window"), ("Table A12", "tableA_fixed_contrasts"),
          ("Table A13", "tableA_profile_split65"), ("Table A14", "tableA_age_profile"),
          ("Table A15", "tableA_gender_split"), ("Table A16", "tableA_education_mix"),
          ("Table A17", "tableA_contrast_by_track"), ("Table A18", "tableA_occ_mix_by_sex"),
          ("Table A19", "tableA_prepath"), ("Table A20", "tableA_industry_credit"),
          ("Table A21", "tableA_cluster_industry"), ("Table A22", "tableA_size_reliability"),
          ("Table A24", "public_yreg"), ("Table A25", "tableIV1_coverage"),
          ("Table A26", "tableIV2_vintage"), ("Table A27", "tableIV3_backtest"),
          ("Table A28", "tableA_occ_coverage"), ("Table A29", "firm_within_variants"),
          ("Table A30", "firm_within_did"), ("Table A31", "firm_heterogeneity"),
          ("Table A33", "tableA_uncounted")]
HDR = ["id", "kind", "document", "where", "statistic", "printed", "source", "locator",
       "transform", "status", "note"]
rows = []
for f in ("manifest_rows_main.csv", "manifest_rows_oa.csv"):
    rows += list(csv.DictReader(open(HERE / f, newline="", encoding="utf-8")))
for i, (ex, n) in enumerate(TABLES, 1):
    rows.append(dict(id=f"T{i:03d}", kind="table", document=f"tables/{n}.tex", where=ex,
                     statistic="every number in the generated table", printed="",
                     source=f"output/tables/{n}.tex", locator="", transform="", status="", note=""))
with open(HERE.parent / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HDR); w.writeheader(); w.writerows(rows)
print(f"MANIFEST.csv: {len(rows)} rows")

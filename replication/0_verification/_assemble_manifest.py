"""Assemble MANIFEST.csv from the main-text rows, the online-appendix rows and one
identity row per generated table (exhibit numbers as the manuscript prints them)."""
import csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
# Exhibit numbers as the online appendix compiles on 26 September 2026
# (Tables A13, A28 and A38 are typed in the manuscript and have no builder).
TABLES = [("Table 1", "table1_headline_v3"), ("Table A1", "top_bottom_occupations"),
          ("Table A2", "tableI2_sumstats_employment"), ("Table A3", "tableI2b_sumstats_postings_v3"),
          ("Table A4", "tableA_remote_measures"), ("Table A5", "postings_accounting"),
          ("Table A6", "coverage_by_source"), ("Table A7", "tableA_posting_robustness"),
          ("Table A8", "postings_extended"), ("Table A9", "postings_seasonality"),
          ("Table A10", "tableA_eloundou_postings"), ("Table A11", "postings_deciles"),
          ("Table A12", "tableA_descriptive_bands"), ("Table A14", "tableA_window"),
          ("Table A15", "tableA_headline_components"), ("Table A16", "tableA_fixed_contrasts"),
          ("Table A17", "tableA_profile_split65"), ("Table A18", "tableA_age_profile"),
          ("Table A19", "tableA_gender_split"), ("Table A20", "tableA_education_mix"),
          ("Table A21", "tableA_contrast_by_track"), ("Table A22", "tableA_occ_mix_by_sex"),
          ("Table A23", "tableA_prepath"), ("Table A24", "tableA_industry_credit"),
          ("Table A25", "tableA_final_checks"), ("Table A26", "tableA_cluster_industry"),
          ("Table A27", "tableA_size_reliability"), ("Table A29", "public_yreg"),
          ("Table A30", "tableIV1_coverage"), ("Table A31", "tableIV2_vintage"),
          ("Table A32", "tableIV3_backtest"), ("Table A33", "tableIV4_backtest_arms"),
          ("Table A34", "tableA_occ_coverage"), ("Table A35", "firm_within_variants"),
          ("Table A36", "firm_within_did"), ("Table A37", "firm_heterogeneity"),
          ("Table A39", "tableA_uncounted"), ("Table A40", "tableA_unlinked"),
          ("Table A41", "tableA_nonmatch")]
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

#!/usr/bin/env python3
"""
probe_vintage.py -- which register vintages exist now? Run in Spyder (F5).

Two questions, both cheap metadata lookups, no job slot needed:
  1. Which monthly AGI tables exist, and are the 2025 ones still only
     preliminary? Every 2025 figure we have rests on the _prel file.
  2. Has an Individ_2024 appeared? That is the occupation register. If it
     has, the lag we have spent two days working around just shrank by a
     year, and several designs become far less constrained.
"""
import pandas as pd
import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;DATABASE=P1207;Trusted_Connection=yes;")

t = pd.read_sql("""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME LIKE 'Arb_AGIIndivid%' OR TABLE_NAME LIKE 'Individ[_]%'
    ORDER BY TABLE_NAME""", conn)

agi = t[t.TABLE_NAME.str.startswith("Arb_AGIIndivid")].copy()
agi["period"] = agi.TABLE_NAME.str.extract(r"(\d{6})")
agi["year"] = agi.period.str[:4]
agi["vintage"] = agi.TABLE_NAME.str.rsplit("_", n=1).str[-1]

print("\nMONTHLY AGI: months present, by year and vintage")
print(agi.pivot_table(index="year", columns="vintage", values="TABLE_NAME",
                      aggfunc="count").fillna(0).astype(int).to_string())

print("\n2025 and later, one line each:")
late = agi[agi.year >= "2025"].sort_values("TABLE_NAME")
print("  (none)" if late.empty else
      "\n".join(f"  {r.TABLE_NAME}" for r in late.itertuples()))

if "def" in set(late.vintage):
    print("\n  *** DEFINITIVE 2025 MONTHS EXIST. Scripts 54 and 47L used the")
    print("  *** preliminary file. Re-pull before believing any 2025 number.")
else:
    print("\n  no definitive 2025 months: preliminary is still all we have.")

occ = sorted(t[t.TABLE_NAME.str.match(r"Individ_\d{4}$")].TABLE_NAME)
print(f"\nOCCUPATION REGISTER: {', '.join(occ) if occ else '(none found)'}")
if any(x >= "Individ_2024" for x in occ):
    print("  *** Individ_2024 OR LATER EXISTS. The occupation lag is shorter")
    print("  *** than we assumed and the backtest should be re-based on it.")
else:
    print("  latest is 2023, as assumed: 2024 and 2025 inherit stale codes.")

conn.close()

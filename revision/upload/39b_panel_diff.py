#!/usr/bin/env python3
"""
39b_panel_diff.py -- localise the 2.40% N gap the canary gate refused to
wave through (4 Sep 2026: coefficients reproduce to the 4th decimal, but
the 22-25 balanced panel has 11,682,918 cells against v1's 11,970,426).

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. No SQL: it reads the v2
  cache and v1's own panel cache from the archive, read-only.
======================================================================

HYPOTHESIS UNDER TEST (the only one available without a code defect,
since the coefficients match): the DATABASE moved between v1's pull
(Feb-Apr 2026) and v2's (Sep 2026) -- the 2024 AGI tables went
_prel -> _def with revisions, and the repaired 2023 Individ delivery
changed the cascade. If the hypothesis is right, the missing mass
concentrates in 2023-2024 and the diff is a database revision, not a
construction error. If it is spread evenly over 2019-2022 -- years no
revision should touch -- the v2 pull itself is under suspicion and the
round stays halted.

Diagnostic only: always exits 0, reports whatever it finds.
Output (output_39b/): panel_diff_by_year.csv, panel_diff_by_quartile.csv,
pair_diff_22_25.csv, 39b_summary.txt
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_39b"
OUT.mkdir(exist_ok=True)

# v1's aggregated, DAIOE-merged, size-filtered panel -- the archive is
# read, never written. CANARIES_V1_PANEL overrides for the local test.
V1_PANEL = os.environ.get(
    "CANARIES_V1_PANEL",
    mc.V1_ARCHIVE + r"\All_output\agg_panel_filtered.parquet")

COLMAP = {  # tolerate v1 naming drift; first match wins per canonical name
    "employer_id": ["employer_id", "peorgnr", "employer"],
    "year_month": ["year_month", "ym", "yearmonth", "month"],
    "exposure_quartile": ["exposure_quartile", "quartile", "daioe_quartile"],
    "age_group": ["age_group", "agegrp", "age_band"],
    "n_emp": ["n_emp", "n", "employment", "n_workers"],
}


def canonise(df: pd.DataFrame, label: str) -> pd.DataFrame:
    print(f"  {label} columns: {list(df.columns)}")
    ren = {}
    for canon, cands in COLMAP.items():
        hit = next((c for c in df.columns if c.lower() in cands), None)
        if hit is None:
            print(f"  {label}: no column for {canon} -- comparisons "
                  f"needing it are skipped")
        else:
            ren[hit] = canon
    df = df.rename(columns=ren)
    if "exposure_quartile" in df.columns and df["exposure_quartile"].dtype == object:
        df["exposure_quartile"] = (df["exposure_quartile"].astype(str)
                                   .str.extract(r"(\d)").astype(float))
    return df


def main():
    mc.Tee(OUT / "39b_log.txt")
    print("=" * 70)
    print("39b: WHERE DOES THE 2.4% LIVE? (v2 pull vs v1 archive cache)")
    print("=" * 70)

    v2 = pd.read_parquet(mc.PANEL_CACHE)
    v2 = mc.aggregate_to_quartile(
        mc.merge_daioe_and_filter(mc.collapse_vintage(v2), mc.load_daioe()))
    v2 = canonise(v2, "v2")

    if not Path(V1_PANEL).exists():
        print(f"  v1 cache not found at {V1_PANEL} -- nothing to compare")
        (OUT / "39b_summary.txt").write_text("v1 cache not found\n")
        return
    v1 = canonise(pd.read_parquet(V1_PANEL), "v1")

    have = [c for c in ("year_month", "exposure_quartile", "age_group")
            if c in v1.columns and c in v2.columns]
    if "year_month" not in have or "n_emp" not in v1.columns:
        print("  v1 schema too different for the standard comparison; "
              "schemas printed above are the deliverable")
        (OUT / "39b_summary.txt").write_text("schema mismatch; see log\n")
        return

    v1["year"] = v1["year_month"].astype(str).str[:4]
    v2["year"] = v2["year_month"].astype(str).str[:4]

    lines = []

    def diff_by(keys, fname):
        a = v1.groupby(keys, observed=True)["n_emp"].agg(["sum", "size"])
        b = v2.groupby(keys, observed=True)["n_emp"].agg(["sum", "size"])
        d = a.join(b, lsuffix="_v1", rsuffix="_v2", how="outer").fillna(0)
        d["demp"] = d["sum_v2"] - d["sum_v1"]
        d["dcells"] = d["size_v2"] - d["size_v1"]
        d["demp_pct"] = 100 * d["demp"] / d["sum_v1"].replace(0, pd.NA)
        d = d.reset_index()
        d.to_csv(OUT / fname, index=False)
        return d

    dy = diff_by(["year"], "panel_diff_by_year.csv")
    print("\n  employment mass, v2 minus v1, by year:")
    for r in dy.itertuples():
        line = (f"    {r.year}: cells {int(r.dcells):+,}  "
                f"emp {int(r.demp):+,} ({0 if pd.isna(r.demp_pct) else r.demp_pct:+.2f}%)")
        print(line)
        lines.append(line)
    if "exposure_quartile" in have:
        diff_by(["year", "exposure_quartile"], "panel_diff_by_quartile.csv")

    # The gate's own margin: which employer x quartile pairs balance in one
    # build and not the other, for 22-25.
    if "age_group" in have and "employer_id" in v1.columns:
        def pairs(df):
            s = df[df["age_group"] == "22-25"]
            cum = s.groupby("employer_id")["n_emp"].sum()
            s = s[s["employer_id"].isin(cum[cum >= 5].index)]
            eq = s.groupby(["employer_id", "exposure_quartile"],
                           observed=True).size().reset_index()
            q4 = set(eq.loc[eq.exposure_quartile == 4, "employer_id"])
            lo = set(eq.loc[eq.exposure_quartile < 4, "employer_id"])
            keep = q4 & lo
            eq = eq[eq.employer_id.isin(keep)]
            return set(map(tuple, eq[["employer_id", "exposure_quartile"]]
                           .itertuples(index=False)))
        p1, p2 = pairs(v1), pairs(v2)
        only1, only2 = p1 - p2, p2 - p1
        msg = (f"\n  22-25 balanced pairs: v1 {len(p1):,}, v2 {len(p2):,}; "
               f"v1-only {len(only1):,}, v2-only {len(only2):,}")
        print(msg)
        lines.append(msg)
        # export COUNTS by quartile, never employer ids (cells >= 5 rule)
        pd.DataFrame([
            {"side": "v1_only", "quartile": q,
             "n_pairs": sum(1 for _, qq in only1 if qq == q)}
            for q in (1, 2, 3, 4)] + [
            {"side": "v2_only", "quartile": q,
             "n_pairs": sum(1 for _, qq in only2 if qq == q)}
            for q in (1, 2, 3, 4)]).to_csv(OUT / "pair_diff_22_25.csv",
                                           index=False)

    (OUT / "39b_summary.txt").write_text(
        "39b: v2 minus v1 panel composition\n" + "\n".join(lines) + "\n"
        "\nREAD RULE (pre-committed): revision-driven drift concentrates in\n"
        "2023-2024 (prel->def, repaired Individ_2023). Mass missing from\n"
        "2019-2022 is NOT explained by revisions and keeps the round halted.\n")
    print("\n39b done.")


if __name__ == "__main__":
    main()

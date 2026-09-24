#!/usr/bin/env python3
"""
17_accounting_to_june_2026.py: the advertisement accounting and the coverage
of the occupation field, carried to June 2026.

METHOD
2020 to 2025: the classification of 01 over the annual archives, in its order
(parse error, no occupation field, invalid code, no date, year outside 2006
to 2026, repeated identifier across years, kept), and 01's month-by-channel
coverage counts (repeats included). January to June 2026: the two quarter
archives, classified the same way and de-duplicated as in 03, within the half
year and before the window is applied; an advertisement that passes and is
dated outside January to June 2026 counts as out of range. The kept count is
therefore the 2026 sample the estimates use. Advertisements kept in 2026
whose identifier already appeared in 2020 to 2025 are counted separately
(the seam overlap); the estimates do not remove them, so neither does the
accounting.

CHECKS (the script stops if one fails)
  1  the 2020 to 2025 accounting equals postings_accounting.csv (01) in every cell;
  2  the 2020 to 2025 month-by-channel coverage equals postings_coverage_monthly.csv (01);
  2b the 2020 to 2025 rows of the coverage table print as 16 printed them;
  3  the 2026 kept advertisements, by occupation and month, equal 03's
     postings_ssyk4_monthly_2026H1.csv in every cell (289,601 in total);
  4  02's active-occupation and zero-cell series, recomputed on 03's panel and
     cut at December 2025, equal 02's files.

INPUTS   config.JOBADS_DIR (all eight archives); output/results/ files of 01,
         02 and 03; output/tables/coverage_by_source.tex (16)
OUTPUTS  output/results/postings_accounting_extended.csv,
         postings_coverage_monthly_extended.csv,
         coverage_active_occupations_extended.csv, coverage_zero_cells_extended.csv,
         l51_seam_overlap.csv; output/tables/postings_accounting.tex and
         coverage_by_source.tex
SERVES   Online Appendix II.4 (Table A4) and II.5 (Table A5 and the two
         coverage files the text names)
RUNTIME  about 25 minutes (four archives read in parallel)
"""
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import config, sibling  # noqa: E402

clf = sibling("01_postings_accounting")
TAB = config.TABLES
RES = config.RESULTS
H1 = ("2026-01", "2026-06")


def scan(zpath: str) -> list:
    """One pass over a zip: per line (reason, ad_id, ym, source, ssyk4).
    ym and source are the raw month and source used for coverage."""
    recs = []
    with zipfile.ZipFile(zpath) as zf:
        for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
            with zf.open(name) as f:
                for line in f:
                    try:
                        ad = json.loads(line)
                    except json.JSONDecodeError:
                        recs.append(("parse_error", "", "", "", ""))
                        continue
                    reason, rec = clf.classify_ad(ad)
                    ym = str(ad.get("publication_date") or "")[:7]
                    src = str(ad.get("source_type") or "")
                    if reason == "ok":
                        recs.append(("ok", rec["ad_id"], rec["year_month"],
                                     src, rec["ssyk4"]))
                    else:
                        recs.append((reason, "", ym, src, ""))
    return recs


def blank(label):
    return dict(year=label, n_raw=0, n_parse_error=0, n_no_occfield=0,
                n_bad_code=0, n_no_date=0, n_out_of_range=0,
                n_duplicate=0, n_kept=0)


def bump(monthly, ym, src, field):
    if len(ym) == 7 and ym[4] == "-":
        d = monthly.setdefault((ym, src or "(none)"),
                               {"n_ads": 0, "n_valid_code": 0})
        d[field] += 1


def main():
    print("Advertisement accounting and coverage to June 2026")
    years = list(config.PLATSBANKEN_YEARS)
    paths = [str(config.platsbanken_zip(y)) for y in years] + \
            [str(config.platsbanken_zip(q)) for q in config.PLATSBANKEN_QUARTERS]
    with ProcessPoolExecutor(max_workers=4) as ex:
        scans = list(ex.map(scan, paths))
    print("  scans done")

    # ---- 2020-2025, the logic of 01 ----------------------------------------
    seen, monthly, rows = set(), {}, []
    for y, recs in zip(years, scans[:len(years)]):
        c = blank(y)
        for reason, ad_id, ym, src, _ in recs:
            c["n_raw"] += 1
            if reason == "parse_error":
                c["n_parse_error"] += 1
                continue
            bump(monthly, ym, src, "n_ads")
            if reason != "ok":
                c[f"n_{reason}"] += 1
                continue
            bump(monthly, ym, src, "n_valid_code")
            if ad_id and ad_id in seen:
                c["n_duplicate"] += 1
                continue
            if ad_id:
                seen.add(ad_id)
            c["n_kept"] += 1
        rows.append(c)
    old = pd.DataFrame(rows)

    # Check 1
    ref = pd.read_csv(RES / "postings_accounting.csv")
    ref = ref[ref["year"] != "TOTAL"].astype({"year": int}).reset_index(drop=True)
    new = old[ref.columns].astype(int).reset_index(drop=True)
    assert new.equals(ref.astype(int)), f"check 1 failed\n{new}\n{ref}"
    print("  check 1 passed: the 2020-2025 accounting of 01 reproduced in every cell")

    # Check 2
    mref = pd.read_csv(RES / "postings_coverage_monthly.csv")
    m_old = pd.DataFrame([{"year_month": k[0], "source_type": k[1], **v}
                          for k, v in sorted(monthly.items())])
    cmp = m_old.merge(mref, on=["year_month", "source_type"], how="outer",
                      suffixes=("", "_ref"), indicator=True)
    assert (cmp["_merge"] == "both").all() \
        and (cmp["n_ads"] == cmp["n_ads_ref"]).all() \
        and (cmp["n_valid_code"] == cmp["n_valid_code_ref"]).all(), "check 2 failed"
    print(f"  check 2 passed: {len(mref)} month-by-source coverage rows of 01 reproduced")

    # ---- January to June 2026, the logic of 03 ------------------------------
    c = blank("2026")
    seen26, counts, seam = set(), {}, 0
    m26 = {}
    for recs in scans[len(years):]:
        for reason, ad_id, ym, src, ssyk4 in recs:
            c["n_raw"] += 1
            if reason == "parse_error":
                c["n_parse_error"] += 1
                continue
            bump(m26, ym, src, "n_ads")
            if reason != "ok":
                c[f"n_{reason}"] += 1
                continue
            bump(m26, ym, src, "n_valid_code")
            if ad_id and ad_id in seen26:
                c["n_duplicate"] += 1
                continue
            if ad_id:
                seen26.add(ad_id)
            if not (H1[0] <= ym <= H1[1]):
                c["n_out_of_range"] += 1
                continue
            c["n_kept"] += 1
            counts[(ssyk4, ym)] = counts.get((ssyk4, ym), 0) + 1
            if ad_id and ad_id in seen:
                seam += 1
    parts = ["n_parse_error", "n_no_occfield", "n_bad_code", "n_no_date",
             "n_out_of_range", "n_duplicate", "n_kept"]
    assert sum(c[p] for p in parts) == c["n_raw"], "2026 does not sum"

    # Check 3
    k26 = pd.DataFrame([{"ssyk4": k[0], "year_month": k[1], "n_ads": v}
                        for k, v in counts.items()])
    r26 = pd.read_csv(RES / "postings_ssyk4_monthly_2026H1.csv",
                      dtype={"ssyk4": str})
    r26["ssyk4"] = r26["ssyk4"].str.zfill(4)
    g = k26.merge(r26, on=["ssyk4", "year_month"], how="outer",
                  suffixes=("", "_ref"), indicator=True)
    assert (g["_merge"] == "both").all() and \
        (g["n_ads"] == g["n_ads_ref"]).all(), "check 3 failed"
    print(f"  check 3 passed: the 2026 sample equals that of 03 cell for cell ({c['n_kept']:,})")

    # 2026 out-of-window months in the coverage series: keep only Jan-Jun
    m26 = {k: v for k, v in m26.items() if H1[0] <= k[0] <= H1[1]}
    pd.DataFrame([{"year_2026_kept": c["n_kept"],
                   "kept_ids_seen_2020_2025": seam,
                   "share": seam / c["n_kept"]}]).to_csv(
        RES / "l51_seam_overlap.csv", index=False)
    print(f"  seam: {seam:,} of the 2026 kept ads carry an identifier seen "
          f"in 2020-2025 ({100 * seam / c['n_kept']:.2f} per cent)")

    # ---- write the extended accounting ----------------------------------
    acc = pd.concat([old, pd.DataFrame([c])], ignore_index=True)
    acc["year"] = acc["year"].astype(str)
    sub = old.drop(columns="year").sum().to_dict()
    sub["year"] = "TOTAL 2020-2025"
    tot = acc.drop(columns="year").sum().to_dict()
    tot["year"] = "TOTAL 2020-2026H1"
    acc = pd.concat([acc, pd.DataFrame([sub, tot])], ignore_index=True)
    acc = acc[ref.columns]
    acc.to_csv(RES / "postings_accounting_extended.csv", index=False)
    print(acc.to_string(index=False))

    cols = ["year", "n_raw", "n_no_occfield", "n_bad_code", "n_no_date",
            "n_out_of_range", "n_duplicate", "n_parse_error", "n_kept"]
    hdr = ["Year", "Raw ads", "No occ.\\ field", "Invalid code", "No date",
           "Out of range", "Duplicate", "Parse error", "Kept"]
    lab = {"2026": "2026 (Jan--Jun)", "TOTAL 2020-2025": "Total 2020--2025",
           "TOTAL 2020-2026H1": "Total to June 2026"}
    lines = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
             r"\hline\hline", " & ".join(hdr) + r" \\", r"\hline"]
    for _, r in acc.iterrows():
        if r["year"] == "2026" or r["year"] == "TOTAL 2020-2025":
            lines.append(r"\hline")
        vals = [lab.get(r["year"], r["year"])] + \
               [f"{int(r[k]):,}" for k in cols[1:]]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}"]
    body = "\n".join(lines)
    (TAB / "postings_accounting.tex").write_text(body)

    # ---- coverage by year and source (the table of 16, one more year) --------
    m_all = {**monthly, **m26}
    mdf = pd.DataFrame([{"year_month": k[0], "source_type": k[1], **v}
                        for k, v in sorted(m_all.items())])
    mdf["valid_share"] = mdf["n_valid_code"] / mdf["n_ads"]
    mdf.to_csv(RES / "postings_coverage_monthly_extended.csv", index=False)
    d = mdf.copy()
    d["year"] = d["year_month"].str[:4]
    d = d[d["year"].between("2020", "2026")]
    LABELS = {"(none)": "No source", "VIA_AIS": "AIS",
              "VIA_ANNONSERA": "Annonsera", "VIA_JOBPOSTING": "JobPosting",
              "VIA_PLATSBANKEN_DXA": "Platsbanken DXA"}
    d["source_type"] = d["source_type"].map(LABELS).fillna(d["source_type"])
    gg = (d.groupby(["year", "source_type"])
          .agg(n_ads=("n_ads", "sum"), n_valid=("n_valid_code", "sum"))
          .reset_index())
    gg = gg[gg["n_ads"] >= 100]
    gg["share"] = 100 * gg["n_valid"] / gg["n_ads"]
    piv = gg.pivot(index="year", columns="source_type", values="share")
    t = d.groupby("year").agg(n=("n_ads", "sum"), v=("n_valid_code", "sum"))
    piv["All sources"] = 100 * t["v"] / t["n"]
    ccols = list(piv.columns)
    # Check 2b: the 2020 to 2025 rows must print exactly as 16 printed them
    ylab = {"2026": "2026 (Jan--Jun)"}
    crow = [f"{ylab.get(y, y)} & " + " & ".join(
        "--" if pd.isna(piv.loc[y, k]) else f"{piv.loc[y, k]:.1f}"
        for k in ccols) + r" \\" for y in piv.index]
    oldtex = (TAB / "coverage_by_source.tex").read_text().splitlines()
    head = "Year & " + " & ".join(str(k) for k in ccols) + r" \\"
    assert head in oldtex, "coverage columns changed"
    for rr in crow[:-1]:
        assert rr in oldtex, f"check 2b failed on {rr}"
    print("  check 2b passed: coverage table rows 2020-2025 print unchanged")
    cbody = "\n".join([r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
                       r"\begin{tabular}{l" + "r" * len(ccols) + "}",
                       r"\hline\hline", head, r"\hline", *crow,
                       r"\hline\hline", r"\end{tabular}"])
    (TAB / "coverage_by_source.tex").write_text(cbody)
    print(cbody)

    # ---- 02's series (a) and (b) on the panel to June 2026 ----------------
    post = pd.read_csv(RES / "postings_ssyk4_monthly_extended.csv",
                       dtype={"ssyk4": str})
    post["ssyk4"] = post["ssyk4"].str.zfill(4)
    daioe = pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    daioe["ssyk4"] = daioe["ssyk4"].str.zfill(4)

    def coverage_series(p):
        months = sorted(p["year_month"].unique())
        active = (p[p["n_ads"] > 0].groupby("year_month")["ssyk4"].nunique()
                  .rename("n_active_occupations").reset_index())
        mg = p.merge(daioe[["ssyk4", "exposure_quartile"]], on="ssyk4")
        occs = mg[["ssyk4", "exposure_quartile"]].drop_duplicates()
        full = (occs.assign(_k=1)
                .merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
                .drop(columns="_k")
                .merge(mg[["ssyk4", "year_month", "n_ads"]],
                       on=["ssyk4", "year_month"], how="left"))
        full["n_ads"] = full["n_ads"].fillna(0)
        z = (full.assign(zero=lambda x: (x["n_ads"] == 0).astype(int))
             .groupby(["year_month", "exposure_quartile"])
             .agg(n_occ=("ssyk4", "nunique"), n_zero=("zero", "sum"))
             .reset_index())
        z["zero_share"] = z["n_zero"] / z["n_occ"]
        return active, z

    a_old, z_old = coverage_series(post[post["year_month"] <= "2025-12"])
    ra = pd.read_csv(RES / "coverage_active_occupations.csv")
    rz = pd.read_csv(RES / "coverage_zero_cells.csv")
    assert a_old.reset_index(drop=True).equals(ra), "check 4 failed (active)"
    assert (z_old[["n_occ", "n_zero"]].values == rz[["n_occ", "n_zero"]].values).all(), \
        "check 4 failed (zero cells)"
    print("  check 4 passed: the series of 02 reproduced to December 2025")
    a_new, z_new = coverage_series(post[post["year_month"] <= "2026-06"])
    a_new.to_csv(RES / "coverage_active_occupations_extended.csv", index=False)
    z_new.to_csv(RES / "coverage_zero_cells_extended.csv", index=False)
    a26 = a_new[a_new["year_month"] >= "2026-01"]["n_active_occupations"]
    zq4 = z_new[z_new["exposure_quartile"] == "Q4 (highest)"]
    print(f"  active occupations, full window: {a_new['n_active_occupations'].min()}"
          f"-{a_new['n_active_occupations'].max()}; 2026: {a26.min()}-{a26.max()}")
    print(f"  Q4 zero-cell share: mean {zq4['zero_share'].mean():.4f}, "
          f"final month ({zq4['year_month'].iloc[-1]}) "
          f"{zq4['zero_share'].iloc[-1]:.4f}; old window mean "
          f"{z_old[z_old['exposure_quartile'] == 'Q4 (highest)']['zero_share'].mean():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()

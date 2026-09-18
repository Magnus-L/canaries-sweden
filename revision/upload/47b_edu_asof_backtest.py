#!/usr/bin/env python3
"""
47b_edu_asof_backtest.py -- does the register lag manufacture a decline in
the EDUCATION-based design too?

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit this
  file itself. Writes output_47b/.
======================================================================

WHY THIS IS NOW THE DECISIVE RUN (18 Sep 2026).

Script 45 established that the occupation-based design fails its own
backtest. On 2019-2023, where every worker can be coded contemporaneously,
the true coefficient for ages 22-25 is +0.019; truncate the register so
later years inherit stale codes and the same specification gives -0.288.
Our headline is -0.174. Occupation codes go stale fast for the young: with
a two-year-old code only 37 per cent of genuinely top-quartile 22-25 cells
are still classified there, and the measured top quartile shrinks by 46 per
cent.

Education should not behave that way. A person's highest completed
education rarely changes after their early twenties, the exposure is
assigned from the occupational composition of an education group rather
than from the worker's own current job, and script 47 fixes those
composition weights at 2019, before the shock. The share of workers mapped
to an education group moves by 0.36 percentage points between 2019 and 2025,
against the 46 per cent collapse in the measured top occupation quartile.

But "should not" is not evidence, and after today it would be negligent to
build the paper on an untested assumption of stability. This script runs
the same backtest on the education design that 45 ran on the occupation
design, so the two artefacts can be set side by side in the same table.

DESIGN. For truncation T in {2021, 2022}: pull each year 2019-2023 twice,
once with the education record from the year's own Individ table (true) and
once with the record from the truncated cascade (as-of), exactly as 45 does
for occupations. Map both to utbildningsgrupp through Erik's key, bin both
with the SAME fixed 2019 employment-weighted quartiles, build employer x
edu-quartile x month cells, and estimate the paper's Poisson specification
on each. The difference is what the education register's lag manufactures.

READ RULE, PRE-COMMITTED BEFORE THE RUN. Let A_occ be 45's artefact, -0.163
at T=2022 and -0.307 at T=2021, and A_edu this script's.
  - |A_edu| < 0.05 at both truncations: the education design is robust to
    the lag, and it, not the occupation design, is the paper's register
    evidence.
  - 0.05 <= |A_edu| < half of |A_occ|: the design is better but not clean;
    it can only be reported with the artefact stated beside every estimate.
  - |A_edu| >= half of |A_occ|: the register route is closed by lag, whatever
    the classifier, and the paper stands on the advertisement evidence.

EXPORT: coefficients and match rates only.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_47b"
OUT.mkdir(exist_ok=True)

YEARS = range(2019, 2024)
TRUNCATIONS = (2021, 2022)
AGES = ["22-25", "26-30", "50+"]

# Copied from 47 rather than imported. 47b's first attempt imported 47 with
# importlib inside main(), before the log was open, and BatchClient discards
# stderr: the job died instantly and left nothing to read. Standalone scripts
# in this round stay standalone (48, 49 do the same).
KEY_PATH = mc.SHARE + r"\utb_grupp2_sun2020_niva3_inr4_nyckel.dta"
KEY_SHA256 = "c760361ba21554951a0744ee00de2f02f22f2e021b87f0863d9ece049e786637"
AGE_CASE = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def norm_code(s: pd.Series) -> pd.Series:
    """One normalisation for every SUN code join: trim + lowercase, and
    both NULL and '' (the 2019 vs 2021+ encodings) become <NA>."""
    out = s.astype("string").str.strip().str.lower()
    return out.where(out.notna() & (out != ""), other=pd.NA)

def load_key() -> pd.DataFrame:
    import hashlib
    got = hashlib.sha256(Path(KEY_PATH).read_bytes()).hexdigest()
    if got != KEY_SHA256:
        raise RuntimeError(f"key hash mismatch: {got[:16]}... is not the "
                           f"delivered key {KEY_SHA256[:16]}...")
    key = pd.read_stata(KEY_PATH)
    key = key.rename(columns={"sun2020niva_3_kod": "niva",
                              "sun2020inr_4_kod": "inr",
                              "utb_grupp2": "grp"})
    key["niva"] = norm_code(key["niva"])
    key["inr"] = norm_code(key["inr"])
    assert not key.duplicated(["niva", "inr"]).any(), "key not unique on niva x inr"
    return key[["niva", "inr", "grp"]]

def build_weights(counts: pd.DataFrame, key: pd.DataFrame,
                  daioe_scores: pd.DataFrame, weight_col: str = "n_all") -> tuple:
    """
    counts: person counts per (niva, inr, ssyk4) from one Individ year,
    occupation-coded people only. Returns (group table, diagnostics dict).

    Economic content: a group's exposure is the exposure of the jobs its
    holders actually do, weighted by how many of them do each job.
    """
    counts = counts.rename(columns={weight_col: "n"}) \
        if weight_col != "n" else counts
    counts = counts[counts["n"] > 0]
    n0 = counts["n"].sum()
    m = counts.merge(key, on=["niva", "inr"], how="left")
    matched = m["grp"].notna()
    m = m[matched]
    m = m.merge(daioe_scores, on="ssyk4", how="inner")   # drops unscored SSYK
    grp = (m.groupby("grp")
             .apply(lambda g: pd.Series({
                 "n_workers": g["n"].sum(),
                 "mean_daioe": np.average(g["pctl_rank_genai"], weights=g["n"]),
             }), include_groups=False)
             .reset_index())
    grp = grp[grp["n_workers"] >= 5].copy()              # export floor
    # Employment-weighted quartiles: rank groups by exposure, cut the
    # CUMULATIVE worker mass at 25/50/75 -- each bin is ~a quarter of
    # workers, not a quarter of the 105 group codes.
    grp = grp.sort_values("mean_daioe").reset_index(drop=True)
    cum = grp["n_workers"].cumsum() / grp["n_workers"].sum()
    grp["edu_quartile"] = np.searchsorted([0.25, 0.5, 0.75], cum, side="left") + 1
    diag = {"n_total": int(n0),
            "key_match_share": float(counts["n"][matched.values].sum() / n0),
            "n_groups": int(len(grp))}
    return grp, diag

def map_and_collapse(raw: pd.DataFrame, grp_q: pd.DataFrame) -> tuple:
    """
    One year of employer x month x (niva, inr) x age cells -> employer x
    month x edu_quartile x age, plus match accounting. grp_q maps
    utbildningsgrupp -> edu_quartile (from the 2019 weights).
    """
    raw = raw.copy()
    raw["niva"] = norm_code(raw["niva"])
    raw["inr"] = norm_code(raw["inr"])
    total = raw.groupby("age_group", observed=True)["n_emp"].sum()
    m = raw.merge(load_key(), on=["niva", "inr"], how="left")
    keyed = m[m["grp"].notna()].merge(grp_q, on="grp", how="inner")
    kept = keyed.groupby("age_group", observed=True)["n_emp"].sum()
    coll = (keyed.groupby(["employer_id", "year_month", "edu_quartile",
                           "age_group"], observed=True)["n_emp"]
            .sum().reset_index())
    rates = pd.DataFrame({"n_total": total, "n_mapped": kept}).reset_index()
    return coll, rates


# ----------------------------------------------------------------------
# SQL pulls
# ----------------------------------------------------------------------

def pull_weight_counts(year: int, conn) -> pd.DataFrame:
    """
    Person counts per (niva, inr, ssyk4) for one Individ year -- the
    composition the measure is built from. Aggregated in SQL: tiny result.

    Two counts per cell: everyone (n_all) and the young (n_young, 22-35 in
    the weight year). The all-ages stock says where anyone with education g
    works, which is dominated by older cohorts; the young count says where
    its RECENT holders go, which is the mapping the 22-25 margin actually
    turns on (ML's point, 4 Sep; Uppsala's graduate-destination logic).
    """
    q = f"""
    SELECT Sun2020Niva AS niva, Sun2020Inr AS inr,
           RIGHT('0000' + CAST(Ssyk4_2012_J16 AS VARCHAR(4)), 4) AS ssyk4,
           COUNT(*) AS n_all,
           SUM(CASE WHEN {year} - FodelseAr BETWEEN 22 AND 35
                    THEN 1 ELSE 0 END) AS n_young
    FROM dbo.Individ_{year}
    WHERE Sun2020Niva IS NOT NULL AND LTRIM(Sun2020Niva) <> ''
      AND Sun2020Inr  IS NOT NULL AND LTRIM(Sun2020Inr)  <> ''
      AND Ssyk4_2012_J16 IS NOT NULL AND LTRIM(Ssyk4_2012_J16) <> ''
    GROUP BY Sun2020Niva, Sun2020Inr, Ssyk4_2012_J16
    """
    return pd.read_sql(q, conn)

def pull_dual(year: int, trunc: int, conn) -> pd.DataFrame:
    """One year with BOTH education assignments on the same rows: the
    year's own record, and the record a register truncated at `trunc`
    would have supplied."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    own = f"dbo.Individ_{min(year, 2023)}"
    casc = [y for y in (trunc, trunc - 1, trunc - 2) if y >= 2019]
    joins = [f"LEFT JOIN {own} t ON agi.P1207_LOPNR_PERSONNR = t.P1207_LopNr_PersonNr"]
    joins += [f"LEFT JOIN dbo.Individ_{y} a{i} "
              f"ON agi.P1207_LOPNR_PERSONNR = a{i}.P1207_LopNr_PersonNr"
              for i, y in enumerate(casc, 1)]
    a_niva = "COALESCE(" + ", ".join(f"a{i}.Sun2020Niva" for i in
                                     range(1, len(casc) + 1)) + ")"
    a_inr = "COALESCE(" + ", ".join(f"a{i}.Sun2020Inr" for i in
                                    range(1, len(casc) + 1)) + ")"
    born = "COALESCE(t.FodelseAr, " + ", ".join(
        f"a{i}.FodelseAr" for i in range(1, len(casc) + 1)) + ")"
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period,
               t.Sun2020Niva AS niva_true, t.Sun2020Inr AS inr_true,
               {a_niva} AS niva_asof, {a_inr} AS inr_asof,
               {born} AS birth_year,
               agi.P1207_LOPNR_PERSONNR AS person_id
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        {' '.join(joins)}""" for m in range(1, max_month + 1))
    q = f"""
    WITH base AS ({monthly}),
    age_calc AS (
        SELECT employer_id, period, niva_true, inr_true, niva_asof,
               inr_asof, person_id,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           niva_true, inr_true, niva_asof, inr_asof,
           {AGE_CASE} AS age_group, COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, niva_true, inr_true, niva_asof, inr_asof,
             {AGE_CASE}
    """
    return pd.read_sql(q, conn)


def main():
    # The log opens before anything can fail, and uncaught exceptions are
    # written into it: BatchClient keeps no stderr, so a crash before this
    # point is invisible (runtime conventions, section 3).
    mc.Tee(OUT / "47b_log.txt")
    import sys as _sys
    import traceback as _tb
    _sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(_tb.format_exception(et, ev, tb)))
    print("=" * 70)
    print("47b: AS-OF BACKTEST ON THE EDUCATION-BASED DESIGN")
    print("=" * 70)
    print(mc.mem_line("  "))

    key = load_key()
    wcache = mc.CACHE_DIR / "edu_weights_2019.parquet"
    counts = mc.read_cache(wcache)
    if counts is None:
        counts = pull_weight_counts(2019, mc.connect())
        counts.to_parquet(wcache, index=False)
    daioe_full = (pd.read_stata(mc.DAIOE_PATH)
                  if mc.DAIOE_PATH.endswith(".dta")
                  else pd.read_csv(mc.DAIOE_PATH))
    daioe_full["ssyk4"] = daioe_full["ssyk4"].astype(str).str.zfill(4)
    scores = daioe_full[["ssyk4", "pctl_rank_genai"]]
    grp_q, diag = build_weights(counts, key, scores)
    grp_q = grp_q[["grp", "edu_quartile"]]
    print(f"  weights: {diag['n_groups']} groups, key match "
          f"{diag['key_match_share']:.3f}, quartiles fixed at 2019")

    conn = mc.connect()
    rows, mrows = [], []
    for trunc in TRUNCATIONS:
        print(f"\n=== truncation T = {trunc} ===")
        frames = []
        for y in YEARS:
            cache = mc.CACHE_DIR / f"edu_dual_{y}_T{trunc}.parquet"
            f = mc.read_cache(cache)
            if f is None:
                t0 = time.time()
                f = pull_dual(y, trunc, conn)
                f.to_parquet(cache, index=False)
                print(f"  {y}: {len(f):,} cells ({time.time()-t0:.0f}s)")
            else:
                print(f"  {y}: cached")
            frames.append(f)
        panel = pd.concat(frames, ignore_index=True)

        for which in ("true", "asof"):
            raw = panel.rename(columns={f"niva_{which}": "niva",
                                        f"inr_{which}": "inr"})
            agg, mrate = map_and_collapse(raw, grp_q)
            mrate["assignment"], mrate["trunc"] = which, trunc
            mrows.append(mrate)
            for age in AGES:
                sub = agg[agg["age_group"] == age]
                months = sorted(agg["year_month"].unique())
                bal = mc.add_treatment(mc.balance_panel(sub, months))
                print(f"  [{which} T{trunc}] {age}: {len(bal):,} cells")
                r = mc.run_fepois(bal, OUT, tag=f"edu_{which}_T{trunc}_{age}")
                if not r.empty:
                    g = r[r["term"] == "post_gpt_x_high"].iloc[0]
                    rows.append({"trunc": trunc, "assignment": which,
                                 "age_group": age, "gamma2": g["coef"],
                                 "se": g["se"], "p": g["pvalue"],
                                 "n_obs": len(bal)})
                del bal

    est = pd.DataFrame(rows)
    est.to_csv(OUT / "edu_asof_estimates.csv", index=False)
    pd.concat(mrows).to_csv(OUT / "edu_asof_matchrates.csv", index=False)

    lines = ["EDUCATION DESIGN: WHAT THE REGISTER LAG MANUFACTURES", "=" * 56,
             "Occupation design, script 45, ages 22-25:",
             "  T=2021  true +0.0193  as-of -0.2875  ARTEFACT -0.3068",
             "  T=2022  true +0.0176  as-of -0.1452  ARTEFACT -0.1627", ""]
    for age in AGES:
        lines.append(f"Education design, ages {age}:")
        for trunc in TRUNCATIONS:
            s = est[(est["age_group"] == age) & (est["trunc"] == trunc)]
            if len(s) == 2:
                tr = float(s[s["assignment"] == "true"]["gamma2"].iloc[0])
                af = float(s[s["assignment"] == "asof"]["gamma2"].iloc[0])
                lines.append(f"  T={trunc}  true {tr:+.4f}  as-of {af:+.4f}"
                             f"  ARTEFACT {af - tr:+.4f}")
        lines.append("")
    lines += ["READ RULE (pre-committed):",
              "  |artefact| < 0.05 at both truncations -> the education design",
              "    is the paper's register evidence.",
              "  0.05 to half of the occupation artefact -> usable only with",
              "    the artefact reported beside every estimate.",
              "  at or above half -> the register route is closed by lag."]
    (OUT / "47b_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n47b done. " + mc.mem_line())


if __name__ == "__main__":
    main()

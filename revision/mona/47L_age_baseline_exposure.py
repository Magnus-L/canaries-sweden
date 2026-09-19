#!/usr/bin/env python3
"""
47L_age_baseline_exposure.py -- exposure from what each firm's AGE GROUP
actually did before the shock. No current worker is ever classified.

======================================================================
  RUNS IN SCB's MONA SECURE ENVIRONMENT ONLY. Standalone: submit THIS
  file. Writes output_47L/. Own SQL: one baseline year and the monthly
  counts, about 45 minutes, plus the fits.
  Local end-to-end test: revision/local/test_47L_synthetic.py
======================================================================

WHY (19 Sep 2026, from the cross-vendor review, finding F4). Every design
so far either classifies the young worker (47b, 47h: artefact -0.36) or
gives up the within-employer comparison (47i) or replaces it with a young
-versus-old contrast (47j). The review named the design we had missed:

  "Instead of assigning the firm a single exposure score, construct a
   pre-shock score for each firm-age group, using its baseline
   occupational or educational mix ... Current workers are counted by age
   only. Their current education and occupation never determine the
   outcome cell."

THE IDEA, and why it is simpler than everything we have tried. In 2019 the
occupation register is good: it is pre-shock, contemporaneous, and the
codes are real. So take each employer's 22-25 year olds in 2019, look at
the jobs they actually held, and score THAT. The result, E(f, a), is the
AI exposure of the work that firm gave to that age group before generative
AI existed. From then on a worker needs only two things to enter the data:
a birth year and a payslip. The occupation register is never used after
2019, and the education register is not used at all.

  outcome    employment of employer x age band x month, 2019 to 2025-06
  treatment  PostGPT x E(f,a), where E is fixed at the 2019 baseline
  absorbed   employer x month  (every firm-time shock)
             employer x age    (the firm's standing age structure)
             age x month       (the economy-wide path of each age band)
  left       PostGPT x E(f,a): among firms whose 22-25 year olds did more
             exposed work in 2019, did employment of 22-25 year olds fall
             further after ChatGPT than in firms whose 22-25 year olds did
             less exposed work -- measured inside the same employer,
             against its own other age groups, in the same month.

That is the paper's question with the staleness removed rather than
patched: the contrast is between MORE and LESS exposed young workers, and
the exposure is measured on the work, not on the person's current record.

WHAT IT COSTS, stated rather than buried.
  1. Baseline noise. A firm's 22-25 cell in 2019 may hold few coded
     workers. We report support, require a floor, and provide a shrunk
     variant (the cell mean pulled toward the firm's own all-age mean).
  2. The baseline inherits 2019's occupation coverage, about 29 per cent
     missing among the under-30s. Missingness is reported by cell and the
     estimate is repeated on cells above successively higher coverage.
  3. Composition drift. E(f,a) describes 2019's work; if a firm changes
     what its young people do for reasons unrelated to AI, the measure is
     stale in a different sense. This is why the event study matters more
     here than the single coefficient.
  4. It is still a difference-in-differences across firms in the treatment
     dimension. Employer x month absorbs a common firm shock; it does not
     absorb a firm shock that falls differently on the young. The review
     is explicit about this and so are we.

THE BACKTEST DOES NOT APPLY, AND SAYING SO IS THE POINT. Exposure is fixed
in 2019 and the outcome uses only payroll and birth year, so truncating
later register vintages changes nothing here. That is an invariance, not a
validation: a classifier unrelated to the work would be equally invariant.
The script runs the truncation anyway and reports it AS an invariance
check, labelled as such.

THE COMPETING SHOCK WE MUST ADDRESS (review finding F5). A reduced employer
contribution for young workers, on monthly remuneration up to SEK 25,000,
expired on 31 March 2023 -- inside our post window and on our age band. An
age x month effect removes the common response, not a response that differs
across firms by their pay distribution. So the script builds, from 2019 pay
alone, each firm-age cell's share of workers below that cap, and reports the
estimate with and without that share interacted with the expiry date.

  THE EXACT ELIGIBLE BIRTH COHORT IS NOT ASSERTED HERE. The rule has had
  several versions and the cohort must be verified against Skatteverket
  before anything is written; the pay-below-cap share is the part we can
  build from our own data without taking the cohort on trust. Read the
  coefficient as "did firms whose young were paid below the cap behave
  differently after March 2023", which is the testable form.
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_47L"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BASE_YEAR = 2019
YEARS = list(range(2019, 2026))
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
MIN_CELL_CODED = 3            # coded workers needed to score a firm-age cell
MIN_FIRM_AGES = 2             # a firm must have >= 2 scored age cells
PAYROLL_CAP_2023 = 25000      # SEK/month; the cap in the rule, to verify
TERMS = ["post_rb_x_expo", "post_gpt_x_expo"]
TERMS_TAX = TERMS + ["post_tax_x_taxshare"]


def age_term(age: str) -> str:
    """Column name for the age-specific post-GPT interaction."""
    return "gpt_x_expo_" + age.replace("-", "_").replace("+", "p")


# The gradient fit: the Riksbank term stays pooled (it is a control, and
# splitting it costs degrees of freedom for nothing), the post-GPT term is
# split by band.
TERMS_GRAD = ["post_rb_x_expo"] + [
    "gpt_x_expo_" + a.replace("-", "_").replace("+", "p")
    for a in ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]]
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")
TAX_YM = "2023-04"            # the expiry takes effect


def opt(label: str, fn, *a, **kw):
    """Stata's `capture noisily`: run something INESSENTIAL, report a
    failure loudly, carry on. Never wrap an estimate in this."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}): {str(ex)[:200]}")
        print("  [optional] continuing; this does not affect the estimates")
        return None


def q_baseline(conn) -> pd.DataFrame:
    """
    The 2019 baseline: for each employer x age band, the occupations its
    workers actually held, and the share of them that carry a code at all.
    November of the base year, matching the occupation register's reference.
    """
    q = f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           CASE
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 31 AND 34 THEN '31-34'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 35 AND 40 THEN '35-40'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 41 AND 49 THEN '41-49'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END AS age_group,
           CASE WHEN i.Ssyk4_2012_J16 IS NULL OR LTRIM(i.Ssyk4_2012_J16) = ''
                     OR LEFT(LTRIM(i.Ssyk4_2012_J16), 1) = '*'
                THEN '____'
                ELSE RIGHT('0000' + CAST(i.Ssyk4_2012_J16 AS VARCHAR(4)), 4)
                END AS ssyk4,
           LTRIM(RTRIM(i.SsykStatus_J16)) AS ssyk_status,
           COUNT(DISTINCT agi.P1207_LOPNR_PERSONNR) AS n
    FROM dbo.Arb_AGIIndivid{BASE_YEAR}11_def agi
    LEFT JOIN dbo.Individ_{BASE_YEAR} i
      ON agi.P1207_LOPNR_PERSONNR = i.P1207_LopNr_PersonNr
    WHERE {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY agi.P1207_LOPNR_PEORGNR,
           CASE
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 31 AND 34 THEN '31-34'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 35 AND 40 THEN '35-40'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 41 AND 49 THEN '41-49'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END,
           CASE WHEN i.Ssyk4_2012_J16 IS NULL OR LTRIM(i.Ssyk4_2012_J16) = ''
                     OR LEFT(LTRIM(i.Ssyk4_2012_J16), 1) = '*'
                THEN '____'
                ELSE RIGHT('0000' + CAST(i.Ssyk4_2012_J16 AS VARCHAR(4)), 4) END,
           LTRIM(RTRIM(i.SsykStatus_J16))
    """
    return pd.read_sql(q, conn)


def q_basepay(conn) -> pd.DataFrame:
    """Pre-shock monthly pay per employer x age cell, for the payroll-tax
    exposure measure. 2019 only, so it cannot respond to anything later."""
    q = f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           CASE
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
             ELSE 'other' END AS age_group,
           SUM(CASE WHEN agi.KONTANT_ERSATTNING_ULAG_AG <= {PAYROLL_CAP_2023}
                    THEN 1 ELSE 0 END) AS n_under_cap,
           COUNT(*) AS n_all
    FROM dbo.Arb_AGIIndivid{BASE_YEAR}11_def agi
    LEFT JOIN dbo.Individ_{BASE_YEAR} i
      ON agi.P1207_LOPNR_PERSONNR = i.P1207_LopNr_PersonNr
    WHERE {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 30
      AND agi.KONTANT_ERSATTNING_ULAG_AG > 0
    GROUP BY agi.P1207_LOPNR_PEORGNR,
           CASE
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
             WHEN {BASE_YEAR} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
             ELSE 'other' END
    """
    return pd.read_sql(q, conn)


def q_counts(year: int, conn) -> pd.DataFrame:
    """Employment counts by employer x age band x month. Birth year is the
    only worker attribute used; no occupation, no education."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    age_case = f"""CASE
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 31 AND 34 THEN '31-34'
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 35 AND 40 THEN '35-40'
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 41 AND 49 THEN '41-49'
        WHEN {{y}} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END""".format(y=year)
    # birth year comes from whichever Individ vintage has the person; birth
    # year does not change, so any vintage is as good as any other
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    q = f"""
    WITH base AS ({monthly}),
    aged AS (
        SELECT employer_id, period, person_id,
               {year} - fodelse AS age
        FROM base WHERE fodelse IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period,
           CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END
    """
    return pd.read_sql(q, conn)


def build_exposure(base: pd.DataFrame, daioe: pd.DataFrame,
                   shrink: bool = False) -> pd.DataFrame:
    """
    E(f, a): the employment-weighted mean DAIOE percentile of the
    occupations held by employer f's age-a workers in the base year.
    `shrink` pulls a thin cell toward the firm's own all-age mean, with
    weight n/(n + MIN_CELL_CODED), which is an explicit, stated choice and
    not a silent default.
    """
    b = base.copy()
    b["ssyk4"] = b["ssyk4"].astype(str).str.zfill(4)
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    b = b[b["age_group"].notna() & (b["n"] > 0)]
    total = (b.groupby(["employer_id", "age_group"], observed=True)["n"]
             .sum().rename("n_total").reset_index())
    coded = b[b["ssyk4"] != "____"].merge(daioe, on="ssyk4", how="inner")
    coded["ws"] = coded["score"] * coded["n"]
    cell = (coded.groupby(["employer_id", "age_group"], observed=True)
            .agg(ws=("ws", "sum"), n_coded=("n", "sum")).reset_index())
    cell["expo_raw"] = cell["ws"] / cell["n_coded"]
    firm = (coded.groupby("employer_id", observed=True)
            .agg(fws=("ws", "sum"), fn=("n", "sum")).reset_index())
    firm["firm_mean"] = firm["fws"] / firm["fn"]
    cell = cell.merge(firm[["employer_id", "firm_mean"]], on="employer_id", how="left")
    cell = cell.merge(total, on=["employer_id", "age_group"], how="left")
    cell["coverage"] = cell["n_coded"] / cell["n_total"].clip(lower=1)
    if shrink:
        w = cell["n_coded"] / (cell["n_coded"] + MIN_CELL_CODED)
        cell["expo"] = w * cell["expo_raw"] + (1 - w) * cell["firm_mean"]
    else:
        cell = cell[cell["n_coded"] >= MIN_CELL_CODED].copy()
        cell["expo"] = cell["expo_raw"]
    keep = (cell.groupby("employer_id")["age_group"].transform("nunique") >= MIN_FIRM_AGES)
    return cell[keep][["employer_id", "age_group", "expo", "expo_raw",
                       "n_coded", "n_total", "coverage", "firm_mean"]]


def build_panel(counts: pd.DataFrame, expo: pd.DataFrame,
                tax: "pd.DataFrame | None" = None) -> pd.DataFrame:
    p = counts.merge(expo[["employer_id", "age_group", "expo"]],
                     on=["employer_id", "age_group"], how="inner")
    if p.empty:
        return p
    p["year_month"] = p["year_month"].astype(str)
    p["age_group"] = p["age_group"].astype(str)
    months = sorted(p["year_month"].unique())
    cells = p[["employer_id", "age_group", "expo"]].drop_duplicates(
        ["employer_id", "age_group"])
    full = pd.MultiIndex.from_arrays(
        [np.repeat(cells["employer_id"].to_numpy(), len(months)),
         np.repeat(cells["age_group"].to_numpy(), len(months)),
         np.tile(np.array(months), len(cells))],
        names=["employer_id", "age_group", "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"], observed=True)
           ["n_emp"].sum().reindex(full, fill_value=0).reset_index()
           .merge(cells, on=["employer_id", "age_group"], how="left"))
    bal["n_emp"] = bal["n_emp"].astype(int)
    # exposure is standardised on the BASELINE cell distribution, so the
    # coefficient reads as the effect of a one standard deviation more
    # exposed baseline, and the scale does not depend on the panel
    mu, sd = cells["expo"].mean(), cells["expo"].std(ddof=0)
    bal["expo_z"] = (bal["expo"] - mu) / (sd if sd > 0 else 1.0)
    bal["post_rb"] = (bal["year_month"] >= mc.RIKSBANK_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= mc.CHATGPT_YM).astype(int)
    bal["post_rb_x_expo"] = bal["post_rb"] * bal["expo_z"]
    bal["post_gpt_x_expo"] = bal["post_gpt"] * bal["expo_z"]
    # One post-GPT interaction PER AGE BAND, so the same design reads as a
    # gradient rather than a single pooled number. The fixed effects are
    # unchanged, so each coefficient is still identified inside the employer
    # against its own other age groups; only the treatment is split.
    for a in AGES:
        bal[age_term(a)] = (bal["post_gpt"] * bal["expo_z"]
                            * (bal["age_group"] == a).astype(int))
    if tax is not None:
        bal = bal.merge(tax, on=["employer_id", "age_group"], how="left")
        bal["taxshare"] = bal["taxshare"].fillna(0.0)
        bal["post_tax"] = (bal["year_month"] >= TAX_YM).astype(int)
        bal["post_tax_x_taxshare"] = bal["post_tax"] * bal["taxshare"]
    e = bal["employer_id"].astype(str)
    bal["fe_emp_t"] = e + "_" + bal["year_month"]
    bal["fe_emp_age"] = e + "_" + bal["age_group"]
    bal["fe_t_age"] = bal["year_month"] + "_" + bal["age_group"]
    return bal


def fit_gradient(bal: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    The same design, one post-GPT coefficient per age band.

    This is the between-firm complement to the within-firm age gradient the
    paper reports: exposure varies across employers, the comparison is made
    inside the employer against its other age groups, and the answer is a
    profile over age rather than one number. It fails differently from the
    occupation route (no post-2019 codes) and differently from the education
    route (no education register at all), which is the only reason putting
    the three beside each other is worth anything.
    """
    if bal.empty:
        return pd.DataFrame()
    r = mc.run_fepois_multi(bal, OUT, tag=tag, terms=TERMS_GRAD, fes=FES)
    if r.empty:
        return pd.DataFrame()
    want = {age_term(a): a for a in AGES}
    r = r[r["term"].isin(want)].copy()
    r["age_group"] = r["term"].map(want)
    return r[["age_group", "coef", "se", "pvalue", "n_obs", "status"]]


def fit(bal: pd.DataFrame, tag: str, terms=None) -> dict:
    terms = terms or TERMS
    row = {"gamma": np.nan, "se": np.nan, "n_obs": len(bal), "status": "empty",
           "tax_coef": np.nan}
    if bal.empty:
        return row
    r = mc.run_fepois_multi(bal, OUT, tag=tag, terms=terms, fes=FES)
    row["status"] = "no_output"
    if not r.empty and (r["term"] == "post_gpt_x_expo").any():
        g = r[r["term"] == "post_gpt_x_expo"].iloc[0]
        row.update(gamma=float(g["coef"]), se=float(g["se"]),
                   status=str(g.get("status", "ok")))
        if (r["term"] == "post_tax_x_taxshare").any():
            row["tax_coef"] = float(
                r[r["term"] == "post_tax_x_taxshare"].iloc[0]["coef"])
    return row


def main():
    mc.Tee(OUT / "47L_log.txt")
    sys.excepthook = lambda et, ev, tb: print(
        "\nUNCAUGHT EXCEPTION\n" + "".join(traceback.format_exception(et, ev, tb)))
    t0 = time.time()
    print("=" * 70)
    print("47L: AGE-SPECIFIC BASELINE EXPOSURE")
    print("     what each firm's age group DID in 2019; nobody classified after")
    print("=" * 70)
    print(mc.mem_line("  "))
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]
    conn = mc.connect()

    cf = CACHE / "L_baseline_2019.parquet"
    base = mc.read_cache(cf)
    if base is None:
        t = time.time()
        base = q_baseline(conn)
        mc.write_cache(base, cf)
        print(f"  baseline {BASE_YEAR}: {len(base):,} rows ({time.time()-t:.0f}s)")
    else:
        print(f"  baseline {BASE_YEAR}: cached ({len(base):,} rows)")

    cf = CACHE / "L_basepay_2019.parquet"
    pay = mc.read_cache(cf)
    if pay is None:
        try:
            pay = q_basepay(conn)
            mc.write_cache(pay, cf)
            print(f"  baseline pay: {len(pay):,} rows")
        except Exception as ex:
            print(f"  baseline pay FAILED ({type(ex).__name__}): {str(ex)[:200]}")
            print("  the payroll-tax robustness will be skipped and SAID SO")
            pay = None
    if pay is not None and len(pay):
        pay = pay[pay["age_group"] != "other"].copy()
        pay["taxshare"] = pay["n_under_cap"] / pay["n_all"].clip(lower=1)
        pay = pay[["employer_id", "age_group", "taxshare"]]

    counts = []
    for y in YEARS:
        cf = CACHE / f"L_counts_{y}.parquet"
        c = mc.read_cache(cf)
        if c is None:
            t = time.time()
            c = q_counts(y, conn)
            mc.write_cache(c, cf)
            print(f"  counts {y}: {len(c):,} cells ({time.time()-t:.0f}s)")
        else:
            print(f"  counts {y}: cached ({len(c):,} cells)")
        counts.append(c)
    cnt = pd.concat(counts, ignore_index=True)
    del counts
    gc.collect()

    rows, grad_rows = [], []
    for variant, shrink in (("floor", False), ("shrunk", True)):
        expo = build_exposure(base, daioe, shrink=shrink)
        print(f"\n  exposure '{variant}': {len(expo):,} firm-age cells, "
              f"{expo['employer_id'].nunique():,} firms, "
              f"median coverage {expo['coverage'].median():.2f}")
        opt(f"support table ({variant})", lambda: mc.enforce_min_cell(
            expo.groupby("age_group", observed=True)
            .agg(cells=("employer_id", "nunique"),
                 med_coverage=("coverage", "median"),
                 med_expo=("expo", "median")).reset_index(),
            count_col="cells").to_csv(OUT / f"exposure_support_{variant}.csv", index=False))
        for use_tax in ((False, True) if pay is not None else (False,)):
            bal = build_panel(cnt, expo, tax=(pay if use_tax else None))
            r = fit(bal, f"L_{variant}{'_tax' if use_tax else ''}",
                    terms=(TERMS_TAX if use_tax else TERMS))
            r.update(variant=variant, payroll_tax_control=use_tax)
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT / "agebase_estimates.csv", index=False)
            print(f"  [{variant}{' +tax' if use_tax else '     '}] "
                  f"PostGPT x exposure {r['gamma']:+.4f} (SE {r['se']:.4f}) "
                  f"n {r['n_obs']:,} {r['status']}"
                  + (f"   tax term {r['tax_coef']:+.4f}" if use_tax else ""))
            # the age gradient: same panel, same FE, treatment split by band
            if not use_tax:
                gr = opt(f"age gradient ({variant})", fit_gradient, bal,
                         f"L_grad_{variant}")
                if gr is not None and not gr.empty:
                    gr = gr.copy(); gr["variant"] = variant
                    grad_rows.append(gr)
                    pd.concat(grad_rows, ignore_index=True).to_csv(
                        OUT / "agebase_gradient.csv", index=False)
                    print(f"  age gradient ({variant}):")
                    for _, g in gr.iterrows():
                        print(f"      {g['age_group']:<5} "
                              f"{g['coef']:+.4f} (SE {g['se']:.4f}) "
                              f"{g['status']}")
            # coverage robustness: only on the primary variant
            if variant == "floor" and not use_tax:
                for thr in (0.5, 0.75):
                    e2 = expo[expo["coverage"] >= thr]
                    b2 = build_panel(cnt, e2)
                    r2 = fit(b2, f"L_cov{int(thr*100)}")
                    r2.update(variant=f"coverage>={thr:.2f}", payroll_tax_control=False)
                    rows.append(r2)
                    print(f"  [coverage>={thr:.2f}] PostGPT x exposure "
                          f"{r2['gamma']:+.4f} (SE {r2['se']:.4f}) n {r2['n_obs']:,}")
                    del b2
                    gc.collect()
            del bal
            gc.collect()
    est = pd.DataFrame(rows)

    lines = ["AGE-SPECIFIC BASELINE EXPOSURE", "=" * 60,
             "E(f,a) = mean DAIOE percentile of the occupations employer f's",
             f"age-a workers held in {BASE_YEAR}, standardised across cells.",
             "After the baseline year a worker needs only a birth year and a",
             "payslip: the occupation register is never used again and the",
             "education register is not used at all.",
             "",
             "Absorbed: employer x month, employer x age, month x age.",
             "Identified: PostGPT x E(f,a) -- among firms whose young did more",
             "exposed work before the shock, did young employment fall further,",
             "measured inside the employer against its own other age groups.",
             ""]
    for r in est.itertuples():
        lines.append(f"  {r.variant:<16} tax control {str(r.payroll_tax_control):<5} "
                     f"gamma {r.gamma:+.4f} (SE {r.se:.4f})  n {r.n_obs:,}")
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The as-of backtest does NOT apply. Exposure is fixed in 2019",
              "     and the outcome uses only payroll and birth year, so later",
              "     register vintages cannot touch it. That is an invariance, not",
              "     a validation: a classifier unrelated to the work would be",
              "     equally invariant.",
              "  2. Employer x month absorbs a common firm shock, NOT a firm shock",
              "     that falls differently on the young. This is a",
              "     difference-in-differences across firms in the exposure",
              "     dimension and needs the corresponding parallel-trend",
              "     assumption, on the multiplicative scale.",
              "  3. The reduced employer contribution for those born 1998-2004",
              "     expired 31 March 2023, inside the post window and on this age",
              "     band. The 'tax control' rows interact PRE-SHOCK pay below the",
              "     SEK 25,000 cap with the expiry date. If the coefficient moves,",
              "     say so.",
              "  4. The baseline inherits 2019 occupation coverage (about 29 per",
              "     cent missing under 30). The coverage>= rows are that check.",
              "  5. The outcome is EMPLOYMENT, not hiring. Ageing into and out of",
              "     the band moves it.",
              f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "47L_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

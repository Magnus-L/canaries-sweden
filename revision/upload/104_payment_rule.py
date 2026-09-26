#!/usr/bin/env python3
"""
104_payment_rule.py -- lane 38e: what turns an employer-declaration record
                       into a counted person-month, and whether the answer
                       differs by exposure, age and period.

======================================================================
  RUNS IN MONA (lane 38e). Output folder CANARIES_104_OUT (default
  output_104). SQL: one pass over the monthly declaration tables
  Arb_AGIIndivid202101_def to Arb_AGIIndivid202506_prel, joined to the
  2023, 2021 and 2019 individual registers for birth year, exactly as
  47L's q_counts; no R. Reads 82's score caches for the exposure quartile
  and 47L's L_counts_2021-2025 for the gate.
======================================================================

QUESTION (the final external read, 26 September 2026, point 1)
The paper's person-month is any distinct person for whom the employer
files an individual declaration record in the month, with a birth year in
the 2023, 2021 or 2019 individual register and an age of 22 to 69; cash
pay is not required (47L's q_counts and 101's reconstruction apply the
same rule). Declarations also carry payments that are not employment: an
occupational pension paid by the employer itself, taxable benefits without
cash pay. Two things are therefore reported, and the OA and letter state
the rule with them.

  P1. The share of counted person-months that carry cash pay subject to
      employer contributions (KONTANT_ERSATTNING_ULAG_AG > 0), by age band
      (22-25, 26-30, 31-69 pooled, all), exposure group (top quartile of
      the paper's 2019 score, the other three quartiles, unscored, all)
      and period (reference Jan 2021-Mar 2022, tightening Apr-Nov 2022,
      interim Dec 2022-Dec 2023, later Jan 2024-Jun 2025).
  P2. The top-minus-rest difference in that share in each period, and its
      change from the interim to the later period, in percentage points,
      for 22-25 and for the older reference; this is the object that could
      move tau if non-employment records were counted differentially. No
      verdict beyond the numbers.
  P3. Among counted person-months WITHOUT cash pay, the share carrying an
      occupational-pension amount and the share carrying a taxable benefit,
      by band, nationally. The column names TJANSTEPENSION,
      SP_BILFORMAN_ULAG_AG and SP_OVRIGA_FORMANER_ULAG_AG come from the
      delivery dictionary and are NOT verified names, so the script probes
      INFORMATION_SCHEMA first and uses each only if it exists; the summary
      says which were found. KONTANT_ERSATTNING_ULAG_AG is verified (47L).

THE GATE (a miss is a hard stop; nothing from the run is quotable)
  The counted person-months by period and age band, summed over every
  employer, must equal 47L's L_counts (the cache the paper's panel is built
  from) to within 0.01 per cent. That is what proves the rule reproduced
  here is the paper's.

EXPORT (output_104/)
  payment_rule.csv          period x band x group: n_emp, n_pay, share_pay,
                            n_nopay_pension, n_nopay_benefit
  payment_rule_by_year.csv  year x band, all employers: the same
  104_summary.txt, 104_log.txt
  Every cell is an aggregate over at least five employers or suppressed.

    python 104_payment_rule.py
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402

OUT = HERE / os.environ.get("CANARIES_104_OUT", "output_104")
OUT.mkdir(exist_ok=True)
os.environ.setdefault("CANARIES_82_OUT", str(OUT))
os.environ.setdefault("CANARIES_80_OUT", str(OUT))
os.environ.setdefault("CANARIES_73_OUT", str(OUT))
CACHE = mc.CACHE_DIR

FLOOR = 5
YEARS = list(range(2021, 2026))
PROBE_TABLE = "Arb_AGIIndivid202101_def"
PAY_COL = "KONTANT_ERSATTNING_ULAG_AG"            # verified: 47L ran on it
PENSION_COL = "TJANSTEPENSION"                    # dictionary wording, probed
BENEFIT_COLS = ("SP_BILFORMAN_ULAG_AG", "SP_OVRIGA_FORMANER_ULAG_AG")  # probed
GATE_TOL = 1e-4                                   # relative, per period x band
BANDS6 = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
OLDER = ["31-34", "35-40", "41-49", "50+"]
PERIODS = ["reference", "tightening", "interim", "later"]

NOTES: list = []
FAILURES: list = []
GATE_LINES: list = []
T0 = time.time()

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  GATE. Counted person-months by period x band over all employers equal",
    "  47L's L_counts within 0.01 per cent; a miss is a hard stop.",
    "  P1. Share of counted person-months with cash pay subject to employer",
    "  contributions, by band x exposure group x period.",
    "  P2. Top minus rest in that share, each period, and its change from the",
    "  interim to the later period, in percentage points, at 22-25 and 31-69.",
    "  No verdict beyond the numbers.",
    "  P3. Among counted person-months without cash pay, the share with an",
    "  occupational-pension amount and with a taxable benefit, by band,",
    "  nationally; only for columns INFORMATION_SCHEMA confirms.",
    f"  Cells resting on fewer than {FLOOR} employers are suppressed.",
]


# ----------------------------------------------------------------------
# plumbing
# ----------------------------------------------------------------------

def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drain(mod, tag: str) -> None:
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    for attr in ("NOTES", "FAILURES"):
        if hasattr(mod, attr):
            getattr(mod, attr).clear()


def load_modules():
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    if s82.MAIN_LEVEL != "uniform3" or s82.MAIN_ARM != "backward" \
            or s82.FLOOR_MAIN != FLOOR:
        raise RuntimeError("82's primary arm is not the one the paper "
                           "reports; refusing to run.")
    return s82, l47, l70, j47


def open_conn():
    """Separated so the local dry run can replace it."""
    return mc.connect()


def period_of(ym: str) -> str:
    if ym <= "2022-03":
        return "reference"
    if ym <= "2022-11":
        return "tightening"
    if ym <= "2023-12":
        return "interim"
    return "later"


# ----------------------------------------------------------------------
# SQL
# ----------------------------------------------------------------------

def probe_columns(conn) -> set:
    """The columns one declaration table actually has. Names from the
    delivery dictionary are used only if they appear here."""
    q = ("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
         f"WHERE TABLE_NAME = '{PROBE_TABLE}'")
    cols = set(pd.read_sql(q, conn)["COLUMN_NAME"].astype(str).str.upper())
    if PAY_COL not in cols:
        raise RuntimeError(f"{PROBE_TABLE} has no column {PAY_COL}; the "
                           f"columns found: {sorted(cols)[:40]}")
    return cols


def flag_sql(col: str, present: bool) -> str:
    return (f"CASE WHEN TRY_CAST(agi.{col} AS FLOAT) > 0 THEN 1 ELSE 0 END"
            if present else "0")


def year_sql(year: int, cols: set) -> str:
    """One year of declarations, collapsed to one row per employer,
    person and month (47L's unit), then counted by employer, period label
    and age band with the payment flags summed."""
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    pen = flag_sql(PENSION_COL, PENSION_COL in cols)
    ben = " + ".join(flag_sql(c, True) for c in BENEFIT_COLS if c in cols) or "0"
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               CASE WHEN agi.{PAY_COL} > 0 THEN 1 ELSE 0 END AS has_pay,
               {pen} AS has_pension,
               CASE WHEN ({ben}) > 0 THEN 1 ELSE 0 END AS has_benefit
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    return f"""
    WITH base AS ({monthly}),
    pm AS (
        SELECT employer_id, period, person_id, fodelse,
               MAX(has_pay) AS has_pay, MAX(has_pension) AS has_pension,
               MAX(has_benefit) AS has_benefit
        FROM base WHERE fodelse IS NOT NULL
        GROUP BY employer_id, period, person_id, fodelse
    ),
    aged AS (
        SELECT employer_id, period, has_pay, has_pension, has_benefit,
               {year} - fodelse AS age
        FROM pm
    )
    SELECT employer_id,
           CASE WHEN period <= '202203' THEN 'reference'
                WHEN period <= '202211' THEN 'tightening'
                WHEN period <= '202312' THEN 'interim'
                ELSE 'later' END AS period_label,
           CASE WHEN age BETWEEN 22 AND 25 THEN '22-25'
                WHEN age BETWEEN 26 AND 30 THEN '26-30'
                WHEN age BETWEEN 31 AND 34 THEN '31-34'
                WHEN age BETWEEN 35 AND 40 THEN '35-40'
                WHEN age BETWEEN 41 AND 49 THEN '41-49'
                WHEN age BETWEEN 50 AND 69 THEN '50+' END AS age_group,
           COUNT(*) AS n_emp,
           SUM(has_pay) AS n_pay,
           SUM(CASE WHEN has_pay = 0 AND has_pension = 1 THEN 1 ELSE 0 END) AS n_nopay_pension,
           SUM(CASE WHEN has_pay = 0 AND has_benefit = 1 THEN 1 ELSE 0 END) AS n_nopay_benefit
    FROM aged
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id,
           CASE WHEN period <= '202203' THEN 'reference'
                WHEN period <= '202211' THEN 'tightening'
                WHEN period <= '202312' THEN 'interim'
                ELSE 'later' END,
           CASE WHEN age BETWEEN 22 AND 25 THEN '22-25'
                WHEN age BETWEEN 26 AND 30 THEN '26-30'
                WHEN age BETWEEN 31 AND 34 THEN '31-34'
                WHEN age BETWEEN 35 AND 40 THEN '35-40'
                WHEN age BETWEEN 41 AND 49 THEN '41-49'
                WHEN age BETWEEN 50 AND 69 THEN '50+' END
    """


def pull_year(year: int, conn, cols: set) -> pd.DataFrame:
    """Separated so the local dry run can replace it."""
    t = time.time()
    d = pd.read_sql(year_sql(year, cols), conn)
    d["year"] = year
    print(f"    {year}: {len(d):,} employer x period x band rows, "
          f"{int(d['n_emp'].sum()):,} person-months, {(time.time() - t) / 60:.1f} min"
          f"{mc.mem_line(' | ')}")
    return d


# ----------------------------------------------------------------------
# the gate: the rule reproduced here is the paper's
# ----------------------------------------------------------------------

def gate(pulled: pd.DataFrame) -> None:
    frames = []
    for y in YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet",
                          require=["employer_id", "year_month", "age_group", "n_emp"])
        if c is None:
            raise RuntimeError(f"L_counts_{y} is not on the share; the gate "
                               "cannot run (run 47L)")
        c = c.copy()
        c["period_label"] = c["year_month"].astype(str).map(period_of)
        frames.append(c.groupby(["period_label", "age_group"], observed=True)["n_emp"].sum())
    ref = pd.concat(frames).groupby(level=[0, 1]).sum()
    got = pulled.groupby(["period_label", "age_group"], observed=True)["n_emp"].sum()
    both = pd.concat([ref.rename("l_counts"), got.rename("this_run")], axis=1).fillna(0)
    both["rel"] = (both["this_run"] - both["l_counts"]).abs() / both["l_counts"].clip(lower=1)
    bad = both[both["rel"] > GATE_TOL]
    print("  GATE, counted person-months by period x band, L_counts against this run:")
    for (p, b), r in both.iterrows():
        print(f"    {p:10s} {b:6s} {int(r['l_counts']):>12,} {int(r['this_run']):>12,} "
              f"rel {r['rel']:.6f}")
    if len(bad):
        msg = ("THE GATE FAILED: this run's counts differ from L_counts by more "
               f"than {GATE_TOL:.0e} in {len(bad)} period x band cells; the rule "
               "reproduced here is not the paper's. Nothing is quotable.")
        print("  " + msg)
        FAILURES.append(msg)
        GATE_LINES.append(msg)
        write_summary(pd.DataFrame(), pd.DataFrame(), set())
        raise SystemExit("104: gate failed")
    GATE_LINES.append(f"THE GATE PASSES: counted person-months by period x band reproduce 47L's "
                      f"L_counts in all {len(both)} cells (max relative deviation "
                      f"{both['rel'].max():.2e}; {int(both['this_run'].sum()):,} person-months)")
    print("  " + GATE_LINES[-1])


# ----------------------------------------------------------------------
# the shares
# ----------------------------------------------------------------------

def attach_group(d: pd.DataFrame, expo: pd.DataFrame) -> pd.DataFrame:
    e = expo[["employer_id", "fq"]]
    d = d.merge(e, on="employer_id", how="left")
    d["group"] = np.where(d["fq"].isna(), "unscored",
                          np.where(d["fq"] == 4, "top", "rest"))
    return d


def shares(d: pd.DataFrame) -> pd.DataFrame:
    """period x band x group aggregates, with 31-69 pooled and 'all' rows
    for band and group, and the employer count behind each cell."""
    d = d.copy()
    d["band"] = np.where(d["age_group"].isin(OLDER), "31-69", d["age_group"])
    rows = []
    for band_set, band_lab in [(["22-25"], "22-25"), (["26-30"], "26-30"),
                               (["31-69"], "31-69"), (["22-25", "26-30", "31-69"], "all")]:
        for grp_set, grp_lab in [(["top"], "top"), (["rest"], "rest"),
                                 (["unscored"], "unscored"), (["top", "rest", "unscored"], "all")]:
            sub = d[d["band"].isin(band_set) & d["group"].isin(grp_set)]
            for p, s in sub.groupby("period_label", observed=True):
                rows.append({"period": p, "band": band_lab, "group": grp_lab,
                             "n_employers": int(s["employer_id"].nunique()),
                             "n_emp": int(s["n_emp"].sum()), "n_pay": int(s["n_pay"].sum()),
                             "n_nopay_pension": int(s["n_nopay_pension"].sum()),
                             "n_nopay_benefit": int(s["n_nopay_benefit"].sum())})
    out = pd.DataFrame(rows)
    out["share_pay"] = out["n_pay"] / out["n_emp"].clip(lower=1)
    out["share_nopay_pension"] = out["n_nopay_pension"] / (out["n_emp"] - out["n_pay"]).clip(lower=1)
    out["share_nopay_benefit"] = out["n_nopay_benefit"] / (out["n_emp"] - out["n_pay"]).clip(lower=1)
    out["period"] = pd.Categorical(out["period"], PERIODS, ordered=True)
    return out.sort_values(["band", "group", "period"]).reset_index(drop=True)


def by_year(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["band"] = np.where(d["age_group"].isin(OLDER), "31-69", d["age_group"])
    g = d.groupby(["year", "band"], observed=True).agg(
        n_employers=("employer_id", "nunique"), n_emp=("n_emp", "sum"), n_pay=("n_pay", "sum"),
        n_nopay_pension=("n_nopay_pension", "sum"),
        n_nopay_benefit=("n_nopay_benefit", "sum")).reset_index()
    a = d.groupby("year", observed=True).agg(
        n_employers=("employer_id", "nunique"), n_emp=("n_emp", "sum"), n_pay=("n_pay", "sum"),
        n_nopay_pension=("n_nopay_pension", "sum"),
        n_nopay_benefit=("n_nopay_benefit", "sum")).reset_index()
    a["band"] = "all"
    g = pd.concat([g, a[g.columns]], ignore_index=True)
    g["share_pay"] = g["n_pay"] / g["n_emp"].clip(lower=1)
    return g.sort_values(["year", "band"]).reset_index(drop=True)


def suppress(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    small = out["n_employers"] < FLOOR
    if small.any():
        print(f"  export floor: suppressing {int(small.sum())} cells under {FLOOR} employers")
        out.loc[small, [c for c in out.columns if c not in ("period", "band", "group", "year")]] = np.nan
    return out


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def pick(sh: pd.DataFrame, period: str, band: str, group: str) -> float:
    r = sh[(sh["period"] == period) & (sh["band"] == band) & (sh["group"] == group)]
    return float(r["share_pay"].iloc[0]) if len(r) == 1 and r["share_pay"].notna().all() else float("nan")


def write_summary(sh: pd.DataFrame, yr: pd.DataFrame, cols: set) -> None:
    L = ["WHAT COUNTS AS A PERSON-MONTH, AND WHETHER IT DIFFERS BY EXPOSURE,",
         "AGE AND PERIOD (LANE 38e)", "=" * 66, "",
         "A counted person-month is a distinct person with an individual",
         "declaration record from the employer in the month, birth year known,",
         "age 22 to 69 (47L's q_counts; 101's reconstruction). Cash pay is not",
         f"required. Cash pay = {PAY_COL} > 0 (verified column).", ""]
    L += GATE_LINES + [""] if GATE_LINES else ["GATE: not reached", ""]
    if cols:
        L.append(f"Probed {PROBE_TABLE}: pension column {PENSION_COL} "
                 f"{'FOUND' if PENSION_COL in cols else 'ABSENT'}; benefit columns "
                 + ", ".join(f"{c} {'found' if c in cols else 'absent'}" for c in BENEFIT_COLS))
        L.append("")
    if not sh.empty:
        L.append("P1. SHARE OF COUNTED PERSON-MONTHS WITH CASH PAY, by period (top quartile | rest | all):")
        for band in ("22-25", "26-30", "31-69", "all"):
            L.append(f"  {band}:")
            for p in PERIODS:
                L.append(f"    {p:10s} top {pick(sh, p, band, 'top'):.4f}  rest "
                         f"{pick(sh, p, band, 'rest'):.4f}  all {pick(sh, p, band, 'all'):.4f}")
        L.append("")
        L.append("P2. TOP MINUS REST, percentage points, and the change from the interim to the later period:")
        for band in ("22-25", "31-69"):
            diffs = {p: 100 * (pick(sh, p, band, "top") - pick(sh, p, band, "rest")) for p in PERIODS}
            L.append(f"  {band}: " + "  ".join(f"{p} {diffs[p]:+.3f}" for p in PERIODS)
                     + f"  | later minus interim {diffs['later'] - diffs['interim']:+.3f} pp")
        L.append("  (a positive change means exposed employers' counted person-months gained cash-pay")
        L.append("   share relative to the rest; the sign says nothing about tau by itself)")
        L.append("")
        L.append("P3. AMONG COUNTED PERSON-MONTHS WITHOUT CASH PAY (all employers, all periods):")
        for band in ("22-25", "26-30", "31-69", "all"):
            sub = sh[(sh["band"] == band) & (sh["group"] == "all")]
            nopay = int((sub["n_emp"] - sub["n_pay"]).sum())
            pen = int(sub["n_nopay_pension"].sum()); ben = int(sub["n_nopay_benefit"].sum())
            L.append(f"  {band}: {nopay:,} person-months without cash pay "
                     f"({100 * nopay / max(int(sub['n_emp'].sum()), 1):.2f} per cent of counted); "
                     f"with a pension amount {100 * pen / max(nopay, 1):.1f} per cent"
                     + (" (column absent)" if PENSION_COL not in cols else "")
                     + f"; with a taxable benefit {100 * ben / max(nopay, 1):.1f} per cent")
        L.append("")
    if not yr.empty:
        L.append("BY YEAR, all employers, share with cash pay:")
        for _, r in yr.iterrows():
            L.append(f"  {int(r['year'])} {r['band']:6s} {r['share_pay']:.4f}  ({int(r['n_emp']):,} person-months)")
        L.append("")
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += ["FAILED: " + " | ".join(FAILURES), "A missing row is a missing pull, never a zero.", ""]
    L += READ_RULES + ["", f"Runtime {(time.time() - T0) / 60:.1f} min. A pass over 54 monthly "
                       "tables should take about an hour; a return under 10 minutes is a failure. "
                       + mc.mem_line("")]
    (OUT / "104_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    global T0
    mc.Tee(OUT / "104_log.txt")
    T0 = time.time()
    print("=" * 70)
    print("104: THE PAYMENT RULE BEHIND THE PERSON-MONTH COUNTS (LANE 38e)")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))
    rc = 0
    sh, yr, cols = pd.DataFrame(), pd.DataFrame(), set()
    try:
        s82, l47, l70, j47 = load_modules()
        if not all((CACHE / f"L_counts_{y}.parquet").exists() for y in YEARS):
            raise RuntimeError("L_counts_2021-2025 are not all on the share; the gate needs them")
        conn = open_conn()
        cols = probe_columns(conn)
        print(f"  {PROBE_TABLE}: {len(cols)} columns; pay column present; pension column "
              f"{'present' if PENSION_COL in cols else 'ABSENT'}; benefit columns "
              f"{[c for c in BENEFIT_COLS if c in cols]}")
        pulled = pd.concat([pull_year(y, conn, cols) for y in YEARS], ignore_index=True)
        try:
            conn.close()
        except Exception:
            pass
        gate(pulled)
        built = s82.build_exposure(l47, l70, j47, audit=False)
        drain(s82, "82")
        expo = built["exposure"]
        del built
        gc.collect()
        print(f"  score: {len(expo):,} employers, {int((expo['fq'] == 4).sum()):,} in the top quartile")
        pulled = attach_group(pulled, expo)
        sh = suppress(shares(pulled))
        yr = suppress(by_year(pulled))
        sh.to_csv(OUT / "payment_rule.csv", index=False)
        yr.to_csv(OUT / "payment_rule_by_year.csv", index=False)
        print(f"  wrote payment_rule.csv ({len(sh)} rows) and payment_rule_by_year.csv ({len(yr)} rows)")
    except SystemExit:
        mc.runlog("104_payment_rule", 2, (time.time() - T0) / 60)
        raise
    except BaseException as ex:
        print(f"104 FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        FAILURES.append(f"main/{type(ex).__name__}: {ex}")
        rc = 1
    write_summary(sh, yr, cols)
    rc = rc or (1 if FAILURES else 0)
    mc.runlog("104_payment_rule", rc, (time.time() - T0) / 60)
    print("\n104 done.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
mona_common.py -- shared infrastructure for the v2 MONA scripts (39-47).

======================================================================
  RUNS ONLY IN SCB's MONA SECURE ENVIRONMENT (except LOCAL_DRYRUN).
======================================================================

Fixes defect D8 from the code-read: v1 copy-pasted the same SQL pull into
seven scripts, with drift. In v2 every MONA script imports from here, so a
change to the cascade or the panel builder cannot miss a script.

Extracted from the proven script-32 architecture (panel cache, staged
outputs, _Tee logging, R + fixest subprocess), with ONE addition the whole
coverage battery needs: `pull_year_vintage()` tags every worker-month with
WHICH Individ vintage supplied the SSYK code (2023 / 2022 / 2021 / none)
instead of collapsing the cascade inside COALESCE. The baseline panel is
the vintage panel with the tags summed out, so both views come from one
pull and cannot disagree.

LOCAL_DRYRUN: set env CANARIES_DRYRUN=1 to import this module outside MONA
(pyodbc mocked, SQL functions raise, everything else testable). The local
test suite runs the panel builder and the R wrapper against synthetic data.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_DRYRUN = os.environ.get("CANARIES_DRYRUN") == "1"

if not LOCAL_DRYRUN:
    import pyodbc  # noqa: F401

# ----------------------------------------------------------------------
# Configuration (single source for every v2 MONA script)
# ----------------------------------------------------------------------

SQL_CONN_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# Project root on the MONA share, under the GROUP CONVENTION (ML, 4 Sep 2026):
# every researcher has one folder at P1207_Gem root; every project has ONE main
# owner and lives in that owner's folder. Magnus owns canaries (he runs all the
# revision empirics), so the round lives beside proworker-gov in Magnus_P1207.
# The v1 work stays untouched in "Lydia P1207\CANARIES\" as the archive.
PROJECT = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden"

# Inputs live in input\ and are named for what they are. CANARIES_SHARE lets the
# local dry-run test point this elsewhere; in MONA the variable is unset.
SHARE = os.environ.get("CANARIES_SHARE", PROJECT + r"\input")

# The v1 tree, for reference only. Nothing in v2 reads from it; it is recorded so
# the provenance of daioe_quartiles.csv is traceable and so a future session does
# not rediscover the layout the hard way.
V1_ARCHIVE = (r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207"
              r"\CANARIES")

# daioe_quartiles.csv is copied into input\ rather than read across from the v1
# tree, so this round does not depend on a folder nobody designed. The copy is
# verified by hash at pre-flight, which makes "the same file" provable rather
# than assumed -- the objection to copying, answered.
DAIOE_SHA256 = "0cae790c086a93a63d681c8307eecace456123d62e322d75ec121176c535d2d8"
DAIOE_PATH = SHARE + r"\daioe_quartiles.csv"

RIKSBANK_YM = "2022-04"
CHATGPT_YM = "2022-12"
REF_HALFYEAR = "2022H1"

AGE_GROUPS = {
    "22-25": (22, 25), "26-30": (26, 30), "31-34": (31, 34),
    "35-40": (35, 40), "41-49": (41, 49), "50+": (50, 69),
}

MIN_EMPLOYER_SIZE = 5

_THIS_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Storage discipline (handbook standard, mona-register-rounds point 7,
# implemented here for the first time 4 Sep 2026)
# ----------------------------------------------------------------------
# Every expensive pull is cached under ONE disposable directory, never in a
# results folder. Cache during the round so a downstream failure does not
# cost the SQL time; retire the whole directory at close of round with
# `run_all_mona.py --retire-caches` once the exports are out and verified.
CACHE_DIR = _THIS_DIR / "cache"
PANEL_CACHE = CACHE_DIR / "panel_vintage.parquet"


def runlog(script: str, rc: int, minutes: float):
    """
    Append one provenance line to the project root RUNLOG.txt: when, WHO
    (MONA accounts are personal), what, exit code, runtime. The group
    convention: a shared project folder must show who has touched it.
    No-op outside MONA (the UNC root does not resolve locally).
    """
    import getpass
    try:
        root = Path(PROJECT)
        line = "%s | %-12s | %-34s | exit %d | %5.1f min\n" % (
            time.strftime("%Y-%m-%d %H:%M"), getpass.getuser(), script,
            rc, minutes)
        with open(root / "RUNLOG.txt", "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def storage_report():
    """Bytes by top-level entry under the round folder, cache/ separated,
    so every run ends with the footprint known (the 17 Aug lesson)."""
    rows = []
    for d in sorted(_THIS_DIR.iterdir()):
        if d.name.startswith((".", "__")):
            continue
        n = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) \
            if d.is_dir() else d.stat().st_size
        rows.append((n, d.name + ("/" if d.is_dir() else "")))
    print("\nSTORAGE (round folder)")
    for n, name in sorted(rows, reverse=True):
        tag = "  <- disposable, --retire-caches" if name == "cache/" else ""
        print("  %8.1f MB  %s%s" % (n / 1e6, name, tag))
    print("  %8.1f MB  TOTAL" % (sum(n for n, _ in rows) / 1e6))
R_FEPOIS = _THIS_DIR / "r_fepois.R"
R_FEPOIS_ES = _THIS_DIR / "r_fepois_es.R"


# ----------------------------------------------------------------------
# Logging (from script 32)
# ----------------------------------------------------------------------

class Tee:
    """
    Mirror stdout to a log file. ASCII-safe for MONA terminals.

    Every log now opens with a provenance header: MONA account, timestamp,
    script. This is the per-file half of the group convention that any output
    in a shared project folder must say who produced it -- accounts in MONA
    are personal, so getpass.getuser() is the runner's identity.
    """

    def __init__(self, path: Path):
        import getpass
        self._header = ("run by %s at %s | %s\n" % (
            getpass.getuser(), time.strftime("%Y-%m-%d %H:%M"),
            Path(sys.argv[0]).name))
        self._f = open(path, "a", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
        sys.stdout = self
        print("=" * 70 + "\n" + self._header + "=" * 70)

    def write(self, s):
        self._stdout.write(s)
        self._f.write(s)
        self._f.flush()

    def flush(self):
        self._stdout.flush()
        self._f.flush()


def connect():
    if LOCAL_DRYRUN:
        raise RuntimeError("SQL access is not available in LOCAL_DRYRUN")
    import pyodbc
    return pyodbc.connect(SQL_CONN_STRING)


# ----------------------------------------------------------------------
# SQL pulls
# ----------------------------------------------------------------------

def _year_suffix(year: int):
    return ("_def", 12) if year < 2025 else ("_prel", 6)


def pull_year_vintage(year: int, conn, force_cascade: bool = False) -> pd.DataFrame:
    """
    One year of AGI, aggregated to employer x ssyk4 x age_group x month x
    SSYK VINTAGE. The vintage column records which Individ table supplied
    the code:

        'own'   -- the year's own Individ table       (years <= 2022)
        '2023' / '2022' / '2021'                      (years >= 2023 cascade)
        'none'  -- no code in any cascade vintage     (the excluded workers
                   the editor asks to be counted, E3)

    Rows with vintage 'none' carry ssyk4 = '____' and are NOT usable for
    exposure assignment; they exist so the coverage denominators are right.
    Summing n_emp over vintage reproduces the v1 COALESCE pull exactly.

    force_cascade=True applies the 2023/2022/2021 cascade to EVERY year,
    including years that have their own Individ table. This is the frozen-
    cohort assignment (script 42): membership and exposure are fixed by the
    2021-2023 registers for the whole window, so post-2023 coverage
    deterioration cannot enter either the numerator or the composition.
    """
    suffix, max_month = _year_suffix(year)
    individ_year = 2023 if force_cascade else min(year, 2023)
    monthly = []
    for month in range(1, max_month + 1):
        ym = f"{year}{month:02d}"
        if individ_year >= 2023:
            monthly.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    COALESCE(i23.Ssyk4_2012_J16, i22.Ssyk4_2012_J16,
                             i21.Ssyk4_2012_J16) AS ssyk4,
                    CASE WHEN i23.Ssyk4_2012_J16 IS NOT NULL THEN '2023'
                         WHEN i22.Ssyk4_2012_J16 IS NOT NULL THEN '2022'
                         WHEN i21.Ssyk4_2012_J16 IS NOT NULL THEN '2021'
                         ELSE 'none' END AS vintage,
                    COALESCE(i23.FodelseAr, i22.FodelseAr, i21.FodelseAr)
                        AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_2023 i23
                    ON agi.P1207_LOPNR_PERSONNR = i23.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2022 i22
                    ON agi.P1207_LOPNR_PERSONNR = i22.P1207_LopNr_PersonNr
                LEFT JOIN dbo.Individ_2021 i21
                    ON agi.P1207_LOPNR_PERSONNR = i21.P1207_LopNr_PersonNr
            """)
        else:
            monthly.append(f"""
                SELECT
                    agi.P1207_LOPNR_PEORGNR AS employer_id,
                    agi.PERIOD AS period,
                    ind.Ssyk4_2012_J16 AS ssyk4,
                    CASE WHEN ind.Ssyk4_2012_J16 IS NOT NULL
                         THEN 'own' ELSE 'none' END AS vintage,
                    ind.FodelseAr AS birth_year,
                    agi.P1207_LOPNR_PERSONNR AS person_id
                FROM dbo.Arb_AGIIndivid{ym}{suffix} agi
                LEFT JOIN dbo.Individ_{individ_year} ind
                    ON agi.P1207_LOPNR_PERSONNR = ind.P1207_LopNr_PersonNr
            """)

    union = "\nUNION ALL\n".join(monthly)
    age_case = """CASE
            WHEN age BETWEEN 22 AND 25 THEN '22-25'
            WHEN age BETWEEN 26 AND 30 THEN '26-30'
            WHEN age BETWEEN 31 AND 34 THEN '31-34'
            WHEN age BETWEEN 35 AND 40 THEN '35-40'
            WHEN age BETWEEN 41 AND 49 THEN '41-49'
            WHEN age BETWEEN 50 AND 69 THEN '50+'
            ELSE NULL END"""
    query = f"""
    WITH base AS ({union}),
    age_calc AS (
        SELECT employer_id, period,
               COALESCE(RIGHT('0000'+CAST(ssyk4 AS VARCHAR(4)),4), '____')
                   AS ssyk4,
               vintage, person_id,
               CAST(LEFT(period,4) AS INT) - birth_year AS age
        FROM base
        WHERE birth_year IS NOT NULL
    )
    SELECT employer_id,
           LEFT(period,4) + '-' + SUBSTRING(period,5,2) AS year_month,
           ssyk4, vintage,
           {age_case} AS age_group,
           COUNT(DISTINCT person_id) AS n_emp
    FROM age_calc
    WHERE age BETWEEN 22 AND 69
    GROUP BY employer_id, period, ssyk4, vintage, {age_case}
    """
    return pd.read_sql(query, conn)


def pull_panel(years, conn, cache_path: Path, vintage: bool = True,
               force_cascade: bool = False):
    """
    Pull all years to one panel, cached as parquet. With vintage=True the
    panel carries the vintage column; collapse_vintage() reproduces the v1
    view. Cache-first: if cache_path exists it is loaded, not re-pulled.
    force_cascade is passed through to pull_year_vintage (frozen cohort).
    """
    if cache_path.exists():
        print(f"  Loading cached panel {cache_path.name}")
        return pd.read_parquet(cache_path)
    frames = []
    for y in years:
        t0 = time.time()
        f = pull_year_vintage(y, conn, force_cascade=force_cascade)
        print(f"  {y}: {len(f):,} cells in {time.time()-t0:.0f}s")
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(cache_path, index=False)
    print(f"  Cached -> {cache_path.name}")
    return panel


def collapse_vintage(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum out the vintage tag; drop uncoded rows. Reproduces the v1 pull."""
    coded = panel[panel["ssyk4"] != "____"]
    return (coded.groupby(
        ["employer_id", "year_month", "ssyk4", "age_group"], observed=True)
        ["n_emp"].sum().reset_index())


# ----------------------------------------------------------------------
# DAIOE merge, size filter, balanced panel (from scripts 15/18/32)
# ----------------------------------------------------------------------

def load_daioe(path: str = DAIOE_PATH) -> pd.DataFrame:
    daioe = pd.read_csv(path)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    # The delivered file stores the quartile as "Q3". Test on numeric-ness,
    # not on `dtype == object`: pandas 3 gives string columns a `str` dtype,
    # so the old object test silently skipped the conversion and left every
    # `exposure_quartile == 4` comparison False, i.e. an empty panel.
    if not pd.api.types.is_numeric_dtype(daioe["exposure_quartile"]):
        daioe["exposure_quartile"] = (daioe["exposure_quartile"].astype(str)
                                      .str.strip().str.extract(r"(\d)")
                                      .astype(int))
    q = daioe["exposure_quartile"]
    if not q.between(1, 4).all():
        raise ValueError(f"exposure_quartile outside 1-4: {sorted(q.unique())}")
    return daioe[["ssyk4", "exposure_quartile"]]


def merge_daioe_and_filter(agg: pd.DataFrame, daioe: pd.DataFrame,
                           min_size: int = MIN_EMPLOYER_SIZE) -> pd.DataFrame:
    agg = agg.copy()
    agg["ssyk4"] = agg["ssyk4"].astype(str).str.zfill(4)
    agg = agg.merge(daioe, on="ssyk4", how="inner")
    size = agg.groupby("employer_id")["n_emp"].sum()
    keep = size[size >= min_size].index
    return agg[agg["employer_id"].isin(keep)].copy()


def aggregate_to_quartile(agg: pd.DataFrame) -> pd.DataFrame:
    return (agg.groupby(["employer_id", "year_month",
                         "exposure_quartile", "age_group"], observed=True)
            ["n_emp"].sum().reset_index())


def balance_panel(sub: pd.DataFrame, all_months) -> pd.DataFrame:
    """
    Balanced zero-filled employer x quartile x month panel for ONE age
    group's rows, with the Q4-and-below identification restriction.
    Identical logic to v1 (15/18) -- kept bit-for-bit so the canary gate
    reproduces.
    """
    emp_q = (sub.groupby(["employer_id", "exposure_quartile"]).size()
             .reset_index()[["employer_id", "exposure_quartile"]])
    q4 = set(emp_q.loc[emp_q["exposure_quartile"] == 4, "employer_id"])
    lo = set(emp_q.loc[emp_q["exposure_quartile"] < 4, "employer_id"])
    emp_q = emp_q[emp_q["employer_id"].isin(q4 & lo)]
    months_df = pd.DataFrame({"year_month": sorted(all_months)})
    # One row per cell guaranteed before the merge (defensive; upstream
    # aggregation already ensures it, but a duplicate would silently
    # double-count under a plain left-merge).
    cell = (sub.groupby(["employer_id", "exposure_quartile", "year_month"],
                        observed=True)["n_emp"].sum().reset_index())
    balanced = (emp_q.assign(_k=1)
                .merge(months_df.assign(_k=1), on="_k").drop(columns="_k")
                .merge(cell,
                       on=["employer_id", "exposure_quartile", "year_month"],
                       how="left"))
    balanced["n_emp"] = balanced["n_emp"].fillna(0).astype(int)
    return balanced


def add_treatment(balanced: pd.DataFrame) -> pd.DataFrame:
    b = balanced
    b["post_rb"] = (b["year_month"] >= RIKSBANK_YM).astype(int)
    b["post_gpt"] = (b["year_month"] >= CHATGPT_YM).astype(int)
    b["high"] = (b["exposure_quartile"] == 4).astype(int)
    b["post_rb_x_high"] = b["post_rb"] * b["high"]
    b["post_gpt_x_high"] = b["post_gpt"] * b["high"]
    b["fe_emp_bin"] = (b["employer_id"].astype(str) + "_"
                       + b["exposure_quartile"].astype(str))
    b["fe_emp_t"] = b["employer_id"].astype(str) + "_" + b["year_month"]
    return b


def assign_halfyear(ym: pd.Series) -> pd.Series:
    return ym.str[:4] + np.where(ym.str[5:7].astype(int) <= 6, "H1", "H2")


# ----------------------------------------------------------------------
# Estimation via R + fixest (from script 32; pyfixest unavailable in MONA)
# ----------------------------------------------------------------------

def _rscript() -> str:
    from shutil import which
    for cand in ("Rscript", r"C:\Program Files\R\R-4.3.1\bin\Rscript.exe"):
        if which(cand) or Path(cand).exists():
            return cand
    raise RuntimeError("Rscript not found")


def run_fepois(panel: pd.DataFrame, workdir: Path, tag: str,
               cluster: str = "employer_id") -> pd.DataFrame:
    """Pooled Poisson DiD via r_fepois.R. Returns the coefficient table."""
    inp = workdir / f"_rin_{tag}.csv"
    outp = workdir / f"_rout_{tag}.csv"
    cols = ["n_emp", "post_rb_x_high", "post_gpt_x_high",
            "fe_emp_bin", "fe_emp_t", cluster]
    panel[cols].to_csv(inp, index=False)
    cmd = [_rscript(), str(R_FEPOIS), "--input", str(inp),
           "--output", str(outp), "--cluster", cluster]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  fepois FAILED ({tag}): {r.stderr[-500:]}")
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    inp.unlink(missing_ok=True)
    return res


def run_fepois_es(panel: pd.DataFrame, workdir: Path, tag: str,
                  ref: str = REF_HALFYEAR,
                  cluster: str = "employer_id") -> pd.DataFrame:
    """Half-year Poisson event study via r_fepois_es.R."""
    inp = workdir / f"_rin_es_{tag}.csv"
    outp = workdir / f"_rout_es_{tag}.csv"
    cols = ["n_emp", "high", "halfyear", "fe_emp_bin", "fe_emp_t", cluster]
    panel[cols].to_csv(inp, index=False)
    cmd = [_rscript(), str(R_FEPOIS_ES), "--input", str(inp),
           "--output", str(outp), "--cluster", cluster, "--ref", ref]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  fepois_es FAILED ({tag}): {r.stderr[-500:]}")
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    inp.unlink(missing_ok=True)
    return res


def run_fepois_multi(panel: pd.DataFrame, workdir: Path, tag: str,
                     terms: list, cluster: str = "employer_id",
                     fes: tuple = ("fe_emp_bin", "fe_emp_t")) -> pd.DataFrame:
    """Poisson with an arbitrary term list via r_fepois_multi.R."""
    inp = workdir / f"_rin_multi_{tag}.csv"
    outp = workdir / f"_rout_multi_{tag}.csv"
    cols = ["n_emp"] + list(terms) + list(fes) + [cluster]
    panel[cols].to_csv(inp, index=False)
    cmd = [_rscript(), str(_THIS_DIR / "r_fepois_multi.R"),
           "--input", str(inp), "--output", str(outp),
           "--terms", ",".join(terms), "--cluster", cluster,
           "--fe", ",".join(fes)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  fepois_multi FAILED ({tag}): {r.stderr[-500:]}")
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    inp.unlink(missing_ok=True)
    return res


# ----------------------------------------------------------------------
# Export safety
# ----------------------------------------------------------------------

def enforce_min_cell(df: pd.DataFrame, count_col: str = "n_emp",
                     floor: int = 5) -> pd.DataFrame:
    """Suppress cells below the export floor (counts 1-4 -> NaN, 0 stays)."""
    out = df.copy()
    small = (out[count_col] > 0) & (out[count_col] < floor)
    if small.any():
        print(f"  export floor: suppressing {small.sum():,} cells < {floor}")
        out.loc[small, count_col] = np.nan
    return out

#!/usr/bin/env python3
"""
mona_common.py: the infrastructure every register script imports.

QUESTION
The register scripts in this folder estimate variants of one design on one
panel, so the parts they share live here once: the database connection, the
pull of the monthly employer declarations, the balanced panel, the Poisson
fits through R, the caches and the export floor. A change to any of these
reaches every script at the same time, which is what keeps the estimates
comparable across scripts. No number in the paper is produced in this
module, and every register estimate passes through it.

WHAT IT PROVIDES
Configuration: the P1207 database connection; the project folder on the
MONA share and its input folder, which holds the DAIOE exposure file
(hash-checked on every run) and the education and teleworkability files;
the treatment dates (RIKSBANK_YM, April 2022, the Riksbank's first rate
rise; CHATGPT_YM, December 2022, the first full month after the ChatGPT
launch); the six age bands (22-25, 26-30, 31-34, 35-40, 41-49 and 50-69);
and the five person-month floor on an employer.

The register pull (pull_year_vintage, pull_panel, collapse_vintage): one
year of the employer declarations (Arb_AGIIndivid, one table per month),
aggregated to employer by four-digit occupation by age band by month, with
a column recording which vintage of the annual Individ register supplied
the occupation code: the year's own register up to 2022; the 2023, 2022 or
2021 register, in that order, from 2023 onward; 'none' when no vintage
holds a code. Summing over that column reproduces the plain cascade. Age
is the calendar year minus the birth year. This pull serves the coverage
diagnostics of Part IV of the online appendix and the withdrawn
occupation design. The design the paper reports reads no occupation code
recorded after 2019; its counts come from the scripts 47L and 54.

The panel builder (load_daioe, merge_daioe_and_filter,
aggregate_to_quartile, balance_panel, add_treatment): the employer by
exposure quartile by month panel of the withdrawn design, balanced and
zero-filled over the months of the window, restricted to employers
observed in both the top quartile and a lower one, with the two treatment
interactions (PostRB x High, PostGPT x High) and the two fixed-effect keys
(employer by quartile, employer by month). Its logic is the one the
submitted version used, kept so that the submitted numbers reproduce.

Estimation (run_fepois, run_fepois_es, run_fepois_multi): every Poisson
pseudo-maximum likelihood fit runs in R through fixest, since pyfixest is
not installed in MONA. The panel is written to a compressed exchange file
on the batch node's local disk, with the fixed-effect and cluster columns
as integer codes; R is located through a pin file, the PATH, the Windows
registry and a version search, in that order; the fit runs with a bounded
thread count and is retried at two threads and then at one if R's
allocator fails; the coefficient table is read back with the standard
errors clustered as requested. run_fepois_multi also copies the clustered
covariance of the treatment terms into the calling script's output folder
as vcov_<tag>.csv, so that a linear combination of terms (a level, a
difference between bands, a sum of quarters) gets a standard error from
the same fit.

Caches and logs (cache_ok, read_cache, write_cache, Tee, runlog,
storage_report, mem_available_gb, mem_line): every expensive pull is
cached as parquet under one disposable folder, written atomically and
validated against the column list the caller needs, so a cache written
before a change to a pull is rebuilt rather than reused. Every script
mirrors its output to its own log file, which opens with a provenance
header (account, time, script), and appends one line per run to the
project's RUNLOG.txt. The terminal echo is capped because the batch
client's standard output is an unread pipe.

Export safety (enforce_min_cell): counts of one to four are set to
missing before any table leaves MONA; zero counts stay.

INPUTS AND OUTPUTS
Reads the tables Arb_AGIIndividYYYYMM and Individ_YYYY in P1207, and the
input file daioe_quartiles.dta. Writes cache/panel_vintage.parquet and the
R exchange files. Nothing written here is an export.

LOCAL TESTING
With the environment variable CANARIES_DRYRUN=1 the module imports
outside MONA: the database driver is not loaded, the SQL functions raise,
and the panel builder and the R wrappers run on synthetic data in the test
files under revision/local/.

IN THE PAPER
Section 2 (the treatment dates and the age bands), and every register
estimate in Section 3, Table 1 and Parts III and IV of the online
appendix, all of which are fitted through run_fepois_multi.
"""

import os
import shutil
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
# Configuration (single source for every register script)
# ----------------------------------------------------------------------

SQL_CONN_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=monasql.micro.intra;"
    "DATABASE=P1207;"
    "Trusted_Connection=yes;"
)

# Project root on the MONA share, under the group convention: every researcher
# has one folder at the P1207_Gem root and every project lives in its main
# owner's folder. The submitted version's work stays untouched in the archive
# folder recorded as V1_ARCHIVE below.
PROJECT = r"\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden"

# Inputs live in input\ and are named for what they are. CANARIES_SHARE lets the
# local dry-run test point this elsewhere; in MONA the variable is unset.
SHARE = os.environ.get("CANARIES_SHARE", PROJECT + r"\input")

# The submitted version's tree, for reference only. Nothing here reads from it;
# it is recorded so the provenance of the DAIOE quartile file is traceable.
V1_ARCHIVE = (r"\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207"
              r"\CANARIES")

# The DAIOE quartile file is copied into the input folder rather than read from
# the archive, and the copy is verified against this hash before any SQL runs,
# so that "the same file" is proved rather than assumed.
DAIOE_SHA256 = "e217df0d3cf03f3e4020fec565c280b9990a90736377e3c9e95091b874341bbb"
DAIOE_PATH = SHARE + r"\daioe_quartiles.dta"

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
# Storage discipline
# ----------------------------------------------------------------------
# Every expensive pull is cached under one disposable directory, never in a
# results folder, so that a failure downstream does not cost the SQL time.
# The directory is retired at the close of a round with
# `run_all_mona.py --retire-caches` once the exports are out and verified.
CACHE_DIR = _THIS_DIR / "cache"
PANEL_CACHE = CACHE_DIR / "panel_vintage.parquet"


def cache_ok(path) -> bool:
    """Cheap integrity check (parquet footer via pyarrow metadata, no data
    read): used by need-SQL decisions, so a truncated cache counts as absent
    and the connection is opened for the rebuild."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        pq.ParquetFile(path)
        return True
    except Exception:
        print(f"  cache {path.name} failed the footer check; deleting")
        try:
            path.unlink()
        except OSError:
            pass
        return False


def write_cache(df: "pd.DataFrame", path) -> Path:
    """
    Write a cache parquet atomically: to a unique temporary name in the same
    directory, then rename onto the target. os.replace is atomic on Windows
    as well as POSIX, so a concurrent reader sees either the whole previous
    file or the whole new one, never a truncated write. This is what lets
    several scripts share one cache directory without keeping private copies
    of the same frames.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    df.to_parquet(tmp, index=False)
    try:
        os.replace(tmp, path)
    except OSError as ex:
        # Windows refuses a rename onto a file another process holds open.
        # The caller already has the frame in memory, so this is survivable:
        # drop the temp copy rather than leave a duplicate behind.
        print(f"  cache {path.name} not replaced ({type(ex).__name__}); "
              f"another console is reading it")
        Path(tmp).unlink(missing_ok=True)
    return path


def read_cache(path, require=None) -> "pd.DataFrame | None":
    """
    Read a cache parquet, or return None if it is missing, unreadable, or
    written under an older schema than the caller now needs.

    A batch job killed mid-write leaves a truncated parquet; cache-first
    logic treats that as absent and rebuilds rather than failing on a
    corrupt read.

    `require` is the column list the caller will use. A cache that predates
    a change to the pull is wrong rather than broken: it reads without error
    and then fails deep in the analysis, far from the cause. The schema is
    therefore validated where the cache is read, once, for every caller,
    rather than by a version string that has to be remembered.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as ex:
        print(f"  cache {path.name} unreadable ({type(ex).__name__}); "
              f"deleting and rebuilding")
        path.unlink(missing_ok=True)
        return None
    if require:
        missing = [c for c in require if c not in df.columns]
        if missing:
            print(f"  cache {path.name} predates a schema change "
                  f"(missing {missing}); deleting and rebuilding")
            path.unlink(missing_ok=True)
            return None
    return df


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
        # Three consoles append to this one file. A single small append is
        # atomic enough on SMB, but the handle can be briefly locked by
        # another console; retry rather than silently lose the row.
        for attempt in range(4):
            try:
                with open(root / "RUNLOG.txt", "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2)
    except OSError:
        pass


def mem_available_gb():
    """
    Physical memory still available, in GB, or None off Windows. Standard
    library only: psutil is not installed in MONA and shell escapes are not
    permitted. Every stage prints this before and after it runs, since the
    batch job's memory is bounded and several jobs share the node.
    """
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPhys / 1e9
    except Exception:
        return None


# The batch client reports a per-job memory ceiling of 100 GB. The cap is
# soft, and the fits that fail do so because R's own allocator gives up
# ("*** recursive gc invocation" is the collector re-entered during a
# collection) rather than because a supervisor ends the process. The number
# below is therefore a planning figure, and the response to a failure is to
# cut R's peak rather than to rely on where the ceiling sits.
JOB_MEM_CAP_GB = 100


def mem_line(prefix: str = "") -> str:
    g = mem_available_gb()
    return f"{prefix}memory available: {g:.1f} GB" if g is not None else ""


def storage_report():
    """Bytes by top-level entry under the round folder, cache/ separated,
    so every run ends with the footprint known."""
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
# Logging
# ----------------------------------------------------------------------

class Tee:
    """
    Mirror stdout to a log file. ASCII-safe for MONA terminals.

    Every log opens with a provenance header: MONA account, timestamp,
    script. This is the per-file half of the group convention that any output
    in a shared project folder must say who produced it; accounts in MONA
    are personal, so getpass.getuser() is the runner's identity.
    """

    # The batch client's stdout is an OS pipe with no reader. It fills at
    # about 4 KB and the next write to it blocks indefinitely. Echoing before
    # writing the file would mean the blocked line never reaches the log
    # either, so the log is written and flushed first and the echo is capped
    # at 2 KB. A local test rig can raise the cap (CANARIES_ECHO_LIMIT) so
    # its own output stays readable; the variable is never set in MONA.
    TERMINAL_ECHO_LIMIT = int(os.environ.get("CANARIES_ECHO_LIMIT", "2048"))

    def __init__(self, path: Path):
        import getpass
        self._header = ("run by %s at %s | %s\n" % (
            getpass.getuser(), time.strftime("%Y-%m-%d %H:%M"),
            Path(sys.argv[0]).name))
        self._f = open(path, "a", encoding="utf-8", errors="replace")
        self._stdout = sys.stdout
        self._echoed = 0
        self._echo_stopped = False
        sys.stdout = self
        print("=" * 70 + "\n" + self._header + "=" * 70)

    def write(self, s):
        # the log is the record and must survive a kill: file first, always
        try:
            self._f.write(s)
            self._f.flush()
        except BaseException:
            pass
        if self._echo_stopped:
            return
        try:
            self._echoed += len(s)
            if self._echoed > self.TERMINAL_ECHO_LIMIT:
                self._echo_stopped = True
                note = ("\n[echo capped; the full log is in the script's own "
                        "log file]\n")
                # the LOG must say so too, or a reader cannot tell that the
                # terminal view is partial
                try:
                    self._f.write(note)
                    self._f.flush()
                except BaseException:
                    pass
                self._stdout.write(note)
            else:
                self._stdout.write(s)
        except BaseException:
            self._echo_stopped = True     # a cp1252 failure lands here too

    def flush(self):
        try:
            self._f.flush()
        except BaseException:
            pass
        try:
            if not self._echo_stopped:
                self._stdout.flush()
        except BaseException:
            self._echo_stopped = True


def connect():
    if LOCAL_DRYRUN:
        raise RuntimeError("SQL access is not available in LOCAL_DRYRUN")
    import pyodbc
    conn = pyodbc.connect(SQL_CONN_STRING)
    # pyodbc waits indefinitely by default; one stalled query would hold a
    # multi-hour batch job with it.
    conn.timeout = 3600
    return conn


# ----------------------------------------------------------------------
# SQL pulls
# ----------------------------------------------------------------------

def _year_suffix(year: int):
    return ("_def", 12) if year < 2025 else ("_prel", 6)


def pull_year_vintage(year: int, conn, force_cascade: bool = False) -> pd.DataFrame:
    """
    One year of the employer declarations, aggregated to employer x ssyk4 x
    age_group x month x vintage. The vintage column records which Individ
    table supplied the code:

        'own'                       the year's own Individ table (years to 2022)
        '2023' / '2022' / '2021'    the cascade (years from 2023)
        'none'                      no code in any cascade vintage (the
                                    excluded workers the coverage tables count)

    Rows with vintage 'none' carry ssyk4 = '____' and are not usable for
    exposure assignment; they exist so the coverage denominators are right.
    Summing n_emp over vintage reproduces the plain COALESCE pull of the
    submitted version exactly.

    force_cascade=True applies the 2023/2022/2021 cascade to every year,
    including years that have their own Individ table. This is the
    frozen-cohort assignment (script 42): membership and exposure are fixed
    by the 2021 to 2023 registers for the whole window.
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
    panel carries the vintage column; collapse_vintage() reproduces the
    submitted version's view. Cache-first: if cache_path exists it is loaded, not re-pulled.
    force_cascade is passed through to pull_year_vintage (frozen cohort).
    """
    cached = read_cache(cache_path)
    if cached is not None:
        print(f"  Loading cached panel {cache_path.name}")
        return cached
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
    """Sum out the vintage tag; drop uncoded rows. Reproduces the submitted
    version's pull."""
    coded = panel[panel["ssyk4"] != "____"]
    return (coded.groupby(
        ["employer_id", "year_month", "ssyk4", "age_group"], observed=True)
        ["n_emp"].sum().reset_index())


# ----------------------------------------------------------------------
# DAIOE merge, size filter, balanced panel
# ----------------------------------------------------------------------

def load_daioe(path: str = DAIOE_PATH) -> pd.DataFrame:
    # .dta, not .csv: dta is an allowed upload format, so the file arrives
    # under its own name with no rename step.
    daioe = pd.read_stata(path) if str(path).endswith(".dta") else pd.read_csv(path)
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    # The delivered file stores the quartile as "Q3". Test on numeric-ness,
    # not on `dtype == object`: pandas 3 gives string columns a `str` dtype,
    # and an object test would skip the conversion and leave every
    # `exposure_quartile == 4` comparison False.
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
    The logic of the submitted version, kept unchanged so that its estimates
    reproduce.
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
# Estimation via R + fixest (pyfixest is unavailable in MONA)
# ----------------------------------------------------------------------

def _rscript() -> str:
    """
    Find Rscript on the MONA node: a pin file beside the scripts, then the
    PATH, then the Windows registry (R-core InstallPath, both hives and
    WOW6432), then a version search over the program folders. The batch
    nodes differ in where R is installed, which is why every route is
    tried. Cached after the first hit; raises with the full search record so
    a failure names every place that was tried.
    """
    global _RSCRIPT_CACHED
    if _RSCRIPT_CACHED:
        return _RSCRIPT_CACHED
    from shutil import which
    tried = []
    # A pin file beats every search: when the search fails on a node, one
    # line of text naming the executable is uploaded rather than a change to
    # this code.
    pin = _THIS_DIR / "rscript_path.txt"
    if pin.exists():
        cand = pin.read_text().strip().strip('"')
        if cand and Path(cand).exists():
            _RSCRIPT_CACHED = cand
            return _RSCRIPT_CACHED
        tried.append(f"pin file {pin.name} -> {cand!r} (does not exist)")
    cand = which("Rscript")
    if cand:
        _RSCRIPT_CACHED = cand
        return cand
    tried.append("PATH")
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for subkey in (r"SOFTWARE\R-core\R", r"SOFTWARE\WOW6432Node\R-core\R",
                           r"SOFTWARE\R-core\R64", r"SOFTWARE\WOW6432Node\R-core\R64"):
                try:
                    with winreg.OpenKey(hive, subkey) as key_:
                        base = winreg.QueryValueEx(key_, "InstallPath")[0]
                    for sub in (r"bin\x64\Rscript.exe", r"bin\Rscript.exe"):
                        exe = Path(base) / sub
                        if exe.exists():
                            _RSCRIPT_CACHED = str(exe)
                            return _RSCRIPT_CACHED
                    tried.append(f"registry {subkey} -> {base}")
                except OSError:
                    pass
    except ImportError:
        tried.append("winreg unavailable (not Windows)")
    import glob as _glob
    # The node carries several R installations side by side, and the version
    # matters: the estimates were produced on R 4.5.0 with fixest 0.13.2, and
    # an install without fixest, or with a different fixest, either fails or
    # answers a slightly different question. Take 4.5.0 when it is there,
    # then the newest, and prefer the x64 launcher.
    for pref in (r"E:\Programs\R-4.5.0\bin\x64\Rscript.exe",
                 r"E:\Programs\R-4.5.0\bin\Rscript.exe"):
        if Path(pref).exists():
            _RSCRIPT_CACHED = pref
            return _RSCRIPT_CACHED
    roots = [r"{d}:\Program Files\R\R-*", r"{d}:\Program Files (x86)\R\R-*",
             r"{d}:\Programs\R-*", r"{d}:\Program\R\R-*", r"{d}:\R\R-*",
             r"{d}:\Apps\R\R-*", r"{d}:\Tools\R\R-*"]
    for drive in "CDEFGH":
        for root in roots:
            for sub in (r"\bin\x64\Rscript.exe", r"\bin\Rscript.exe"):
                pattern = root.format(d=drive) + sub
                hits = _glob.glob(pattern)
                if hits:
                    def _ver(path):
                        import re as _re
                        m = _re.search(r"R-(\d+)\.(\d+)\.(\d+)", path)
                        return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)
                    _RSCRIPT_CACHED = sorted(hits, key=_ver)[-1]
                    return _RSCRIPT_CACHED
    tried.append("globs over drives C-H under Program Files, Programs, R, "
                 "Apps and Tools")
    raise RuntimeError(
        "Rscript not found; searched " + "; ".join(tried)
        + ". FIX: create rscript_path.txt beside these scripts holding the "
          "full path to Rscript.exe, or run probe_rscript.py to locate it.")


_RSCRIPT_CACHED = None


_R_WORKDIR_SWEPT = False


def _r_workdir(workdir: Path) -> Path:
    """
    The R exchange files (multi-million-row CSVs) go to the batch node's
    local disk, not the share: writes over SMB dominate the runtime and a
    stalled SMB handle blocks indefinitely. Falls back to the share if the
    temporary directory is unavailable.

    The exchange directory is per script (and per job tag), not shared, so
    that two jobs running at once can never delete each other's input
    between the write and R reading it.
    """
    import tempfile
    try:
        # CANARIES_RWORK_TAG lets one script run as several batch jobs at
        # once: each job gets its own exchange directory, so their fits can
        # never share a file and the sweep below only ever sees this job's
        # own leftovers.
        d = (Path(tempfile.gettempdir()) / "canaries_rwork"
             / (Path(sys.argv[0]).stem + os.environ.get("CANARIES_RWORK_TAG", "")))
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return workdir
    # First call in this process only: sweep what a previous crashed run of
    # this script left behind, and say how much room is left. A run killed
    # mid-fit leaves its exchange file on disk, and several of those fill
    # the node's temporary volume. The sweep is confined to this script's
    # own subdirectory, so one job can never delete another's live input.
    global _R_WORKDIR_SWEPT
    if not _R_WORKDIR_SWEPT:
        _R_WORKDIR_SWEPT = True
        # The per-script subdirectory stops two different scripts colliding;
        # it does nothing when one script is submitted twice, in which case
        # the second job's sweep would delete the first job's live input.
        # Anything modified in the last hour is therefore left alone: a
        # crashed run's leftovers are older than that, and a live run's are
        # not.
        import time as _t
        cutoff = _t.time() - 3600
        freed = skipped = 0
        for f in (list(d.glob("_rin_*")) + list(d.glob("_rout_*"))
                  + list(d.glob("_rerr_*"))):
            try:
                if f.stat().st_mtime > cutoff:
                    skipped += 1
                    continue
                freed += f.stat().st_size
                f.unlink()
            except OSError:
                pass
        if skipped:
            print(f"  WARNING: {skipped} exchange file(s) in {d} were "
                  f"written in the last hour and were NOT swept. Another "
                  f"copy of this script is probably running. Two jobs "
                  f"sharing this directory will corrupt each other's fits.")
        try:
            free_gb = shutil.disk_usage(d).free / 1e9
            print(f"  R exchange dir {d}: swept {freed/1e6:,.0f} MB, "
                  f"{free_gb:,.1f} GB free")
        except OSError:
            pass
    return d


_READER_REPORTED = False


def _report_reader(stdout: str) -> None:
    """
    Echo R's reader choice into the Python log, once per process.

    The R scripts print which reader they used into their own stdout, which
    is captured and written out only when a fit fails; a successful fit
    would otherwise leave no record of whether data.table or the base
    reader ran. A line nobody sees is not a diagnostic.
    """
    global _READER_REPORTED
    if _READER_REPORTED or not stdout:
        return
    for line in stdout.splitlines():
        if line.startswith("reader:"):
            print(f"  R {line}")
            if "pre-allocated" in line:
                print(f"  R   (base reader, ~{JOB_MEM_CAP_GB} GB soft job cap; "
                      f"data.table is absent from MONA and cannot be "
                      f"installed, so the row count is passed instead)")
            elif "read.csv" in line:
                print("  R WARNING: reading WITHOUT a row count, so the "
                      "frame grows by reallocation. That is what killed "
                      "fits on 21 September: R's allocator failed, no "
                      "supervisor killed them.")
            _READER_REPORTED = True
            return


def _r_failed(tag: str, kind: str, r, workdir: Path) -> None:
    """
    Report an R failure so it can be diagnosed, not only noticed.

    The tail of R's standard error is usually a package warning box, and the
    cause has scrolled past it. The whole stream is kept on disk, and both
    ends of it are printed: the first lines carry the cause, the last the
    collapse.
    """
    err = (r.stderr or "").strip()
    out = (r.stdout or "").strip()
    # Save beside the script's output, on the share, not in the batch
    # node's temporary folder, which is local to the node and cannot be read
    # from the interactive session or exported.
    path = workdir / f"_rerr_{tag}.txt"
    try:
        import inspect
        for frame in inspect.stack():
            cand = frame.frame.f_globals.get("OUT")
            if isinstance(cand, Path) and cand.is_dir():
                path = cand / f"_rerr_{tag}.txt"
                break
    except BaseException:
        pass
    try:
        path.write_text(f"returncode {r.returncode}\n\n=== stderr ===\n{err}"
                        f"\n\n=== stdout ===\n{out}", encoding="utf-8",
                        errors="replace")
        where = f"  full R output: {path}"
    except BaseException as ex:
        where = f"  (could not save R output: {type(ex).__name__})"
    # MONA prints a boxed banner about its CRAN mirror at the start of every
    # R session; it is dropped before choosing what to show.
    def _is_banner(ln: str) -> bool:
        t = ln.strip()
        return (not t or set(t) <= set("+-|") or t.startswith("|")
                or "CRAN-mirror" in t or "install.packages" in t
                or "MONA has a local" in t or "R sessions (batch" in t
                or "Reinstalling a package" in t or "remain installed" in t
                or "reinstall the package" in t or "lines ..." in t)

    lines = [ln for ln in err.splitlines() if not _is_banner(ln)]
    head = "\n    ".join(lines[:12]) if lines else "(stderr empty once the "
    if not lines:
        head = "(no error text: stderr held only MONA's startup banner)"
    tail = "\n    ".join(lines[-6:]) if len(lines) > 18 else ""
    print(f"  {kind} FAILED ({tag}) rc={r.returncode}")
    print(f"    {head}")
    if tail:
        print(f"    ... {len(lines) - 18} lines ...\n    {tail}")
    print(where)


def _write_r_input(panel: pd.DataFrame, cols: list, inp: Path,
                   recode: tuple = (), cluster: str = "") -> Path:
    """
    Write the R exchange file compactly, and return the path actually used.

    A panel of ten million rows whose fixed-effect columns are concatenated
    strings runs to about a gigabyte as plain CSV. Two things keep the file
    small, neither of which touches an estimate: the fixed-effect columns
    are written as integer factor codes, since every R script coerces them
    with as.factor() and a factor's labels are never used; and the file is
    gzipped at level 1, which read.csv() reads transparently. Level 1 rather
    than 9 because the constraint is disk, not bandwidth.
    """
    # De-duplicate, preserving order: a caller may use employer_id as both a
    # fixed effect and the cluster, so the column list can name it twice, and
    # a column written twice is renamed and ignored by read.csv.
    cols = list(dict.fromkeys(cols))
    out = panel[cols]
    recode = [c for c in recode if c in out.columns]
    # The cluster column is recoded too, when it is not already numeric: a
    # cluster label is used for grouping and nothing else, exactly like a
    # fixed effect, and twenty million character values cost R far more
    # than the same column as integers. Only the cluster, not every
    # non-numeric column: run_fepois_es carries `halfyear` as text and uses
    # its labels, since --ref names one of its levels.
    if (cluster and cluster in out.columns and cluster not in recode
            and not pd.api.types.is_numeric_dtype(out[cluster])):
        recode = list(recode) + [cluster]
        print(f"  exchange: factorising the cluster column {cluster!r} "
              f"(its labels are never used)")
    if recode:
        out = out.copy()
        for c in recode:
            out[c] = pd.factorize(out[c], sort=False)[0].astype("int32")
    inp = inp.with_suffix(inp.suffix + ".gz")
    free_mb = shutil.disk_usage(inp.parent).free / 1e6
    if free_mb < 500:
        print(f"  WARNING: only {free_mb:,.0f} MB free on {inp.parent}; "
              f"the R exchange may not fit")
    out.to_csv(inp, index=False,
               compression={"method": "gzip", "compresslevel": 1})
    return inp


# Windows STATUS_ACCESS_VIOLATION: R's allocator failing inside the job,
# not the node running out of memory. fixest gives every thread its own
# demeaning workspace, so peak memory scales with the thread count while
# the per-job cap does not.
R_MEMORY_DEATH = 3221225477
R_RETRY_THREADS = 2
R_RETRY_FLOOR = 1      # one last attempt single-threaded


def _looks_like_memory_death(r) -> bool:
    txt = (r.stderr or "") + (r.stdout or "")
    return (r.returncode == R_MEMORY_DEATH
            or "recursive gc invocation" in txt
            or "cannot allocate" in txt)


def _threads_in(cmd: list) -> int:
    if "--nthreads" in cmd:
        try:
            return int(cmd[cmd.index("--nthreads") + 1])
        except (IndexError, ValueError):
            return 0
    return 0


def _run_r(cmd: list, workdir: Path, tag: str, kind: str):
    """
    Run one R fit, and retry at a lower thread count if it failed the way
    an over-threaded fixest fails (an access violation with "*** recursive
    gc invocation"). The thread count travels on the command line because
    the batch submitter cannot set environment variables. A retry costs
    nothing when a fit succeeds and recovers the few that do not.
    """
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(workdir))
    _report_reader(r.stdout)
    if r.returncode == 0 or not _looks_like_memory_death(r):
        return r
    # Two threads is not always enough, so the retry walks the ladder: two
    # threads, then one, then gives up.
    ladder = [R_RETRY_THREADS, R_RETRY_FLOOR]
    had = _threads_in(cmd)
    for nxt in ladder:
        if had and had <= nxt:
            continue
        print(f"  {kind} ({tag}) died with rc={r.returncode} "
              f"(fixest memory, not the node); retrying at {nxt} thread(s)")
        retry = [c for c in cmd]
        if "--nthreads" in retry:
            retry[retry.index("--nthreads") + 1] = str(nxt)
        else:
            retry += ["--nthreads", str(nxt)]
        r = subprocess.run(retry, capture_output=True, text=True,
                           cwd=str(workdir))
        _report_reader(r.stdout)
        had = nxt
        if r.returncode == 0:
            print(f"  {kind} ({tag}) SUCCEEDED on the {nxt}-thread retry")
            return r
        if not _looks_like_memory_death(r):
            return r
    print(f"  {kind} ({tag}) died at {had} thread(s); nothing lower to try")
    return r


def run_fepois(panel: pd.DataFrame, workdir: Path, tag: str,
               cluster: str = "employer_id") -> pd.DataFrame:
    """Pooled Poisson DiD via r_fepois.R. Returns the coefficient table."""
    workdir = _r_workdir(workdir)
    inp = workdir / f"_rin_{tag}.csv"
    outp = workdir / f"_rout_{tag}.csv"
    cols = ["n_emp", "post_rb_x_high", "post_gpt_x_high",
            "fe_emp_bin", "fe_emp_t", cluster]
    inp = _write_r_input(panel, cols, inp,
                         recode=("fe_emp_bin", "fe_emp_t"),
                         cluster=cluster)
    cmd = [_rscript(), str(R_FEPOIS), "--input", str(inp),
           "--output", str(outp), "--cluster", cluster,
           "--nrows", str(len(panel))]
    r = _run_r(cmd, workdir, tag, "fepois")
    if r.returncode != 0:
        _r_failed(tag, "fepois", r, workdir)
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    inp.unlink(missing_ok=True)
    return res


def run_fepois_es(panel: pd.DataFrame, workdir: Path, tag: str,
                  ref: str = REF_HALFYEAR,
                  cluster: str = "employer_id") -> pd.DataFrame:
    """Half-year Poisson event study via r_fepois_es.R."""
    workdir = _r_workdir(workdir)
    inp = workdir / f"_rin_es_{tag}.csv"
    outp = workdir / f"_rout_es_{tag}.csv"
    cols = ["n_emp", "high", "halfyear", "fe_emp_bin", "fe_emp_t", cluster]
    # `halfyear` is NOT recoded: r_fepois_es.R names its coefficients after
    # the level ("halfyear2021H1") and matches --ref against the label.
    inp = _write_r_input(panel, cols, inp,
                         recode=("fe_emp_bin", "fe_emp_t"),
                         cluster=cluster)
    cmd = [_rscript(), str(R_FEPOIS_ES), "--input", str(inp),
           "--output", str(outp), "--cluster", cluster, "--ref", ref,
           "--nrows", str(len(panel))]
    r = _run_r(cmd, workdir, tag, "fepois_es")
    if r.returncode != 0:
        _r_failed(tag, "fepois_es", r, workdir)
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    inp.unlink(missing_ok=True)
    return res


def run_fepois_multi(panel: pd.DataFrame, workdir: Path, tag: str,
                     terms: list, cluster: str = "employer_id",
                     fes: tuple = ("fe_emp_bin", "fe_emp_t"),
                     nthreads: int = 0) -> pd.DataFrame:
    """Poisson with an arbitrary term list via r_fepois_multi.R.

    The R side also writes the clustered covariance of the terms; it is
    copied into the caller's output directory as vcov_<tag>.csv so that it
    leaves MONA with the rest of the script's exports, and its path is
    recorded in res.attrs["vcov"]."""
    out_dir = Path(workdir)
    workdir = _r_workdir(workdir)
    inp = workdir / f"_rin_multi_{tag}.csv"
    outp = workdir / f"_rout_multi_{tag}.csv"
    cols = ["n_emp"] + list(terms) + list(fes) + [cluster]
    inp = _write_r_input(panel, cols, inp, recode=tuple(fes),
                         cluster=cluster)
    cmd = [_rscript(), str(_THIS_DIR / "r_fepois_multi.R"),
           "--input", str(inp), "--output", str(outp),
           "--nrows", str(len(panel)),
           *(["--nthreads", str(nthreads)] if nthreads else []),
           "--terms", ",".join(terms), "--cluster", cluster,
           "--fe", ",".join(fes)]
    r = _run_r(cmd, workdir, tag, "fepois_multi")
    if r.returncode != 0:
        _r_failed(tag, "fepois_multi", r, workdir)
    res = pd.read_csv(outp) if outp.exists() else pd.DataFrame()
    vcp = outp.with_name(outp.stem + "_vcov.csv")
    if vcp.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"vcov_{tag}.csv"
            shutil.copy(vcp, dest)
            res.attrs["vcov"] = str(dest)
        except OSError as ex:
            print(f"  vcov for {tag} not copied: {ex}")
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

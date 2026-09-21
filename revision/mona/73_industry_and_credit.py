#!/usr/bin/env python3
"""
73_industry_and_credit.py -- the two things referees asked for that we
                             still answer with argument instead of data.

======================================================================
  RUNS IN MONA. SQL against FDB and Serrano, which this project has not
  used, so it DISCOVERS the schema first. Writes output_73/.
======================================================================

PART A. INDUSTRY x AGE x MONTH.

ChatGPT and Fable named this independently as the most valuable missing
specification, for the same reason. Employer-by-month already absorbs
industry-by-month, because industry is fixed within an employer. It does
NOT absorb industry-by-AGE-by-month. Our month-by-age effects absorb
national age shocks; they do not absorb an age shock that hits one
industry harder than another.

So the question this answers is exactly the one the referees keep
asking in different words: did exposed employers change their age
composition differently from less exposed employers facing the SAME
industry-specific age shock?

Industry is taken from FDB_JE_2019, frozen pre-shock. Using a later
vintage would let a firm's post-shock reclassification into the
treatment, which is the kind of thing this whole revision exists to
avoid.

PART B. THE MONETARY CHANNEL, TESTED RATHER THAN ARGUED.

R1.2, R1.3 and R2.1 all raise monetary transmission, and ChatGPT's
verdict on our current answer is blunt: the objection "remains
unanswered by the timing comparisons". We reply with a date and a
teleworkability horse race, and the horse race is not the placebo we
have been treating it as.

There is a direct test. If the decline is credit-driven it should be
concentrated in LEVERAGED firms, because those are the firms a rate
cycle actually binds on. Leverage comes from Serrano's financial
statements for 2019, again frozen pre-shock.

Read this as a discriminating test, not a horse race:

  MONETARY  post x young x leverage is significantly negative AND the
            exposure term loses at least half its size and its
            significance once leverage is in. Then the paper has a
            serious problem and we would rather know now.
  AI SURVIVES  the exposure term keeps at least half its size and stays
            significant with leverage in, whatever leverage itself does.
            That is a far stronger answer to the referees than the
            timing argument we currently make.
  INCONCLUSIVE otherwise, and it is reported as inconclusive.

PART C. ARE THE SEPARATIONS REAL?

Serrano's corporate-event table carries bankruptcy and liquidation
status. A worker leaving a firm that went bankrupt is not an AI effect.
This drops those firms and re-estimates. It matters only if the paper
ends up claiming anything about separations, so it runs last and its
failure costs nothing else.

GATES. Every arm counts its matched firms before it estimates, against
thresholds fixed below before the run, and refuses rather than lowering
them. The schema is discovered rather than assumed: our own dictionary
lists no variables at all for Serrano_bokslut, so the debt and asset
columns are found by pattern and REPORTED, and if nothing matches the
script says which columns it did see instead of guessing again.

Output (output_73/):
  schema_found.csv    what the discovery step saw
  industry_fe.csv     Part A
  credit_test.csv     Part B
  bankruptcy.csv      Part C
  73_summary.txt
"""

import gc
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_73"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BASE_YEAR = 2019            # industry and leverage are frozen pre-shock
YOUNG_BANDS = ["22-25", "26-30"]
PANEL_FROM = "2021-01"
PANEL_YEARS = list(range(2021, 2026))
POOLED_FROM = "2024-01"
TRUNC = 2021
DESIGN, ARM = "OL_daioe", "true"

MIN_MATCH_RATE = 0.30       # of panel firms that must carry the covariate
MIN_FIRMS = 500
ATTEN_MAX = 0.50            # exposure term must keep at least half its size
NOTES = []
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(label)
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rss_gb() -> float:
    """
    This process's resident memory in GB, stdlib only.

    Python and R share the job's allocation, so what Python is holding
    when it spawns R is the budget R does not get. mem_available_gb()
    reports the NODE and is useless for this.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        m = _PMC(); m.cb = ctypes.sizeof(_PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(m),
            m.cb)
        return m.WorkingSetSize / 1e9
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        except Exception:
            return float("nan")


# THREADS. This is why 73 kept dying where 68 lived. fixest defaults to
# every core and each thread carries its own demeaning workspace, so peak
# memory scales with the core count while the job cap does not. 68 ran
# when fewer lanes were up; 73 ran with three lanes on the same node, and
# died at 26.4M rows on a fit 68 had done at 36.5M. The signature is
# rc=3221225477 with "*** recursive gc invocation", R's collector giving
# up, and the node reporting 468 GB free at that moment: the machine had
# the memory, this job did not. Two threads costs wall-clock and buys the
# fit. The env var CANARIES_R_THREADS cannot be set from the MONA batch
# submitter, which is why this now travels on the command line.
R_THREADS = 2


def norm_id(x) -> pd.Series:
    """One canonical spelling on both sides of every join (see 71)."""
    v = pd.Series(x).astype(str).str.strip()
    return v.str.replace(r"\.0$", "", regex=True)


def pick(cols, *patterns):
    for p in patterns:
        rx = re.compile(p, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def discover(conn) -> pd.DataFrame:
    q = """
    SELECT t.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE
    FROM INFORMATION_SCHEMA.TABLES t
    JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
    WHERE t.TABLE_NAME LIKE 'FDB[_]JE[_]%' OR t.TABLE_NAME LIKE 'Serrano%'
       OR t.TABLE_NAME LIKE 'FE[_]%'
       OR t.TABLE_NAME LIKE 'Ftg[_]%' OR t.TABLE_NAME LIKE 'Arbst[_]%'
    ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
    """
    return pd.read_sql(q, conn)


def edu_exposure(j47):
    """47j's incumbent education exposure, as 61 builds it."""
    h47 = j47._h47()
    counts = {}
    for y in h47.WEIGHT_YEARS:
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise SystemExit(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[DESIGN])
    book.build(DESIGN, spec)
    frame19 = mc.read_cache(CACHE / f"edu_hr_{j47.BASE_YEAR}.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise SystemExit(f"edu_hr_{j47.BASE_YEAR}.parquet missing: run 47h.")
    expo, _ = j47.incumbent_exposure(frame19, book, DESIGN, spec, ARM, TRUNC)
    del frame19
    gc.collect()
    return expo


def _sni3(raw: pd.Series) -> pd.Series:
    """SNI2007 at three digits, from a code of any length."""
    v = raw.astype(str).str.strip().str.replace(r"\D", "", regex=True)
    return v.where(v.str.len().between(2, 5)).str[:3]


def firm_industry(conn, schema) -> pd.DataFrame:
    """
    employer_id -> frozen 2019 three-digit industry.

    SOURCE ORDER, AND WHY IT IS THIS ORDER.

    Our employer_id is AGI's P1207_LOPNR_PEORGNR, the legal entity.
    LISA's firm table Ftg_<year> is keyed on exactly that identifier and
    carries Org_Sni2007, the statistical industry SCB itself attaches to
    the employer, for every firm in the employment register. That is the
    right source.

    FDB_JE is the business register's legal-entity file. It carries the
    same key, which is why it looked right, but the delivered extract is
    not the employer population: at ar=2019 it returned 139,203 entities
    and matched 12.3 per cent of the 22-25 panel. Every Swedish employer
    has an industry code, so a 12.3 per cent match measures the register
    we chose, not the firms. FDB_JE is now the last resort.

    Arbst_<year> is the workplace file, keyed on the workplace but
    carrying the parent LopNr_PeOrgNr, so a firm with workplaces in
    several industries takes the industry of its largest workplace.

    Whichever source answers, the coverage it achieves is reported, and
    the gate below still applies: a source that cannot reach the match
    threshold produces no estimate rather than a thin one.
    """
    tabs = sorted(schema["TABLE_NAME"].unique())

    def cols_of(t):
        return schema[schema.TABLE_NAME == t]["COLUMN_NAME"].tolist()

    # ---- 1. LISA firm table: one row per firm, the industry SCB uses --
    ftg = next((t for t in tabs
                if re.fullmatch(rf"Ftg_{BASE_YEAR}", t, re.I)), None)
    if ftg:
        c = cols_of(ftg)
        key = pick(c, r"^LopNr_PeOrgNr$", r"PeOrgNr")
        ind = pick(c, r"^Org_Sni2007$", r"Sni2007", r"^Sni")
        if key and ind:
            d = pd.read_sql(f"SELECT [{key}] AS employer_id, "
                            f"[{ind}] AS ind FROM dbo.[{ftg}]", conn)
            d["employer_id"] = norm_id(d["employer_id"])
            d["ind3"] = _sni3(d["ind"])
            d = d.dropna(subset=["ind3"]).drop_duplicates("employer_id")
            if len(d):
                msg = (f"industry: {ftg} via {ind} (LISA firm table, the "
                       f"employer population), {len(d):,} firms, "
                       f"{d['ind3'].nunique()} three-digit groups")
                print(f"  {msg}"); NOTES.append(msg)
                return d[["employer_id", "ind3"]]
        NOTES.append(f"{ftg}: key {key}, industry {ind}; unusable")

    # ---- 2. LISA workplace table, rolled up to the firm ---------------
    arb = next((t for t in tabs
                if re.fullmatch(rf"Arbst_{BASE_YEAR}", t, re.I)), None)
    if arb:
        c = cols_of(arb)
        key = pick(c, r"^LopNr_PeOrgNr$", r"PeOrgNr")
        ind = pick(c, r"^AstSNI2007$", r"Sni2007", r"^Sni")
        size = pick(c, r"^Anst$", r"AntAnst", r"^Syss")
        if key and ind:
            sel = f"SELECT [{key}] AS employer_id, [{ind}] AS ind"
            sel += f", [{size}] AS n" if size else ""
            d = pd.read_sql(sel + f" FROM dbo.[{arb}]", conn)
            d["employer_id"] = norm_id(d["employer_id"])
            d["ind3"] = _sni3(d["ind"])
            d = d.dropna(subset=["ind3"])
            if size:
                d["n"] = pd.to_numeric(d["n"], errors="coerce").fillna(0)
                d = d.sort_values("n", ascending=False)
            d = d.drop_duplicates("employer_id")
            if len(d):
                how = "largest workplace" if size else "first workplace"
                msg = (f"industry: {arb} via {ind} ({how}; Ftg_{BASE_YEAR} "
                       f"was not available), {len(d):,} firms, "
                       f"{d['ind3'].nunique()} three-digit groups")
                print(f"  {msg}"); NOTES.append(msg)
                return d[["employer_id", "ind3"]]

    # ---- 3. FDB_JE, last resort ---------------------------------------
    # Year RANGES with an `ar` column; there is no FDB_JE_2019, and the
    # first FDB_JE% alphabetically is FDB_JE_1990_1993 with sni69ng1,
    # the 1969 classification.
    def _covers(t):
        yrs = [int(x) for x in re.findall(r"((?:19|20)\d{2})", t)]
        if len(yrs) == 2:
            return yrs[0] <= BASE_YEAR <= yrs[1]
        return len(yrs) == 1 and yrs[0] == BASE_YEAR

    cands = [t for t in tabs if t.lower().startswith("fdb_je")]
    tab = next((t for t in cands if _covers(t)), None) \
        or next((t for t in cands if "all_years" in t.lower()), None)
    if tab is None:
        NOTES.append(f"no industry source for {BASE_YEAR}; Part A cannot run")
        return pd.DataFrame()
    c = cols_of(tab)
    key = pick(c, r"^P1207_Lopnr_peorgnr$", r"PeOrgNr", r"PEORGNR")
    ind = pick(c, r"^ng3$", r"^ngs1$", r"^ng2$")
    yrc = pick(c, r"^ar$", r"^year$")
    if key is None or ind is None:
        NOTES.append(f"{tab}: key {key}, industry {ind}; Part A cannot run. "
                     f"Columns seen: {c[:20]}")
        return pd.DataFrame()
    sel = f"SELECT [{key}] AS employer_id, [{ind}] AS ind"
    d = pd.read_sql(sel + f" FROM dbo.[{tab}]"
                    + (f" WHERE [{yrc}] = {BASE_YEAR}" if yrc else ""), conn)
    d["employer_id"] = norm_id(d["employer_id"])
    d["ind3"] = _sni3(d["ind"])
    d = d.dropna(subset=["ind3"]).drop_duplicates("employer_id")
    msg = (f"industry: {tab} via {ind} (FALLBACK, the business register "
           f"rather than the employer population), {len(d):,} firms, "
           f"{d['ind3'].nunique()} three-digit groups")
    print(f"  {msg}"); NOTES.append(msg)
    return d[["employer_id", "ind3"]]


def leverage_from_fek(conn, schema) -> pd.DataFrame:
    """
    Frozen 2019 leverage from FEK, which is the better source.

    SCB's own structural business statistics, population rather than
    sample for private non-financial firms, and two documented variables
    give the standard ratio: 1 - SummaEgetKapital / SummaTillgangar.

    TWO DOCUMENTED TRAPS, both handled here rather than discovered later.

    UNIT. FE tables before 2022 are keyed on LopNr_PeOrgNrHE, the group
    HEAD unit, not the legal entity our AGI panel is built on, and the
    figures are CONSOLIDATED, so an FE is not the sum of its JEs. For a
    single-entity firm the two coincide; for a group member they do not,
    and what attaches is the group's balance sheet rather than the
    employer's. We use 2019, so the head-unit key is the right one, and
    the match rate is reported so the reader can see how much of the
    panel this covers.

    POPULATION. FEK excludes the public sector, financial firms and
    non-profits. The credit test therefore runs on private non-financial
    employers, which must be said in the paper rather than left implicit.
    """
    cands = [t for t in sorted(schema["TABLE_NAME"].unique())
             if re.match(r"^FE_\d{4}$", t, re.I)]
    tab = next((t for t in cands if t.endswith(str(BASE_YEAR))), None)
    if tab is None:
        earlier = [t for t in cands if int(t.split("_")[1]) <= 2021]
        tab = max(earlier, key=lambda t: int(t.split("_")[1])) \
            if earlier else None
    if tab is None:
        return pd.DataFrame()
    cols = schema[schema.TABLE_NAME == tab]["COLUMN_NAME"].tolist()
    key = pick(cols, r"PeOrgNrHE", r"PeOrgNr", r"FENr")
    assets = pick(cols, r"SummaTillgangar")
    equity = pick(cols, r"SummaEgetKapital")
    if not (key and assets and equity):
        NOTES.append(f"{tab}: key {key}, assets {assets}, equity {equity}; "
                     f"FEK leverage unavailable. Columns: {cols[:25]}")
        return pd.DataFrame()
    d = pd.read_sql(f"SELECT [{key}] AS employer_id, [{assets}] AS assets, "
                    f"[{equity}] AS equity FROM dbo.[{tab}]", conn)
    d["employer_id"] = norm_id(d["employer_id"])
    for c in ("assets", "equity"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ".",
                                                          regex=False),
                             errors="coerce")
    d = d[(d["assets"] > 0) & d["equity"].notna()]
    d["lev"] = (1.0 - d["equity"] / d["assets"]).clip(0, 3)
    d = d.drop_duplicates("employer_id")
    print(f"  leverage: {tab} via 1 - {equity}/{assets}, {len(d):,} firms, "
          f"median {d['lev'].median():.2f}")
    NOTES.append(
        f"leverage built from {tab}: 1 - {equity}/{assets}, keyed on {key}. "
        f"Pre-2022 FE is the CONSOLIDATED group head unit, so a group "
        f"member carries its group's balance sheet; FEK also excludes "
        f"public, financial and non-profit employers.")
    return d[["employer_id", "lev"]]


def firm_leverage(conn, schema) -> pd.DataFrame:
    """
    Frozen 2019 leverage.

    FEK would be the better source, being SCB's own and population
    level, but no FE_YYYY table answered a LIKE 'FE[_]%' probe on
    21 September, so it may simply not be in this delivery. Serrano
    carries the full balance sheet and is what actually runs; FEK stays
    first in case it appears.
    """
    fek = leverage_from_fek(conn, schema)
    if len(fek):
        return fek
    NOTES.append("no FEK table answered the probe; using Serrano")
    tab = next((t for t in sorted(schema["TABLE_NAME"].unique())
                if "bokslut" in t.lower()), None)
    if tab is None:
        NOTES.append("no Serrano bokslut table either; Part B cannot run")
        return pd.DataFrame()
    cols = schema[schema.TABLE_NAME == tab]["COLUMN_NAME"].tolist()
    # Serrano uses abbreviated Swedish codes: no column contains the
    # words skuld or tillgang, which is why the 21 September patterns
    # found nothing. Verified names, from the live catalogue:
    #   TILLGSU total assets      EKSU   total equity
    #   LSKSU   long-term debt    KSKSU  short-term debt
    #   EKSKSU  equity+liabilities   NTOMS turnover
    key = pick(cols, r"^P1207_Lopnr_ORGNR$", r"ORGNR", r"PeOrgNr")
    asset = pick(cols, r"^TILLGSU$", r"^EKSKSU$")
    equity = pick(cols, r"^EKSU$")
    dlong, dshort = pick(cols, r"^LSKSU$"), pick(cols, r"^KSKSU$")
    yr = pick(cols, r"^BSLSLUT$", r"^ar$", r"year", r"bokslutsar")
    if asset and equity:
        debt = None            # leverage as 1 - equity/assets
    else:
        debt = dlong or dshort
    if not (key and asset and (equity or debt or (dlong and dshort))):
        NOTES.append(
            f"{tab}: could not identify the balance sheet (key {key}, "
            f"assets {asset}, equity {equity}, debt {dlong}/{dshort}). "
            f"Columns seen: {cols[:40]}")
        return pd.DataFrame()
    parts = [f"[{key}] AS employer_id", f"[{asset}] AS assets"]
    if equity:
        parts.append(f"[{equity}] AS equity")
    if dlong:
        parts.append(f"[{dlong}] AS dlong")
    if dshort:
        parts.append(f"[{dshort}] AS dshort")
    sel = ", ".join(parts)
    if yr:
        sel += f", [{yr}] AS yr"
    d = pd.read_sql(f"SELECT {sel} FROM dbo.[{tab}]", conn)
    if yr and "yr" in d:
        y = pd.to_numeric(d["yr"], errors="coerce")
        if (y == BASE_YEAR).any():
            d = d[y == BASE_YEAR]
    d["employer_id"] = norm_id(d["employer_id"])
    for c in [c for c in ("assets", "equity", "dlong", "dshort")
              if c in d.columns]:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ".",
                                                          regex=False),
                             errors="coerce")
    d = d[d["assets"] > 0]
    if "equity" in d.columns and d["equity"].notna().any():
        d["lev"] = (1.0 - d["equity"] / d["assets"]).clip(0, 3)
        how = f"1 - {equity}/{asset}"
    else:
        dd = d.get("dlong", 0).fillna(0) + d.get("dshort", 0).fillna(0)
        d["lev"] = (dd / d["assets"]).clip(0, 3)
        how = f"({dlong}+{dshort})/{asset}"
    d = d[d["lev"].notna()].drop_duplicates("employer_id")
    print(f"  leverage: {tab} via {how}, {len(d):,} firms, "
          f"median {d['lev'].median():.2f}")
    NOTES.append(f"leverage built from {tab}: {how}, keyed on {key}")
    return d[["employer_id", "lev"]]


def firm_failed(conn, schema) -> set:
    """Firms recorded as bankrupt or liquidated in Serrano's events."""
    tab = next((t for t in sorted(schema["TABLE_NAME"].unique())
                if re.search(r"serrano.*bol", t, re.I)), None)
    if tab is None:
        NOTES.append("no Serrano corporate-event table; Part C cannot run")
        return set()
    cols = schema[schema.TABLE_NAME == tab]["COLUMN_NAME"].tolist()
    key = pick(cols, r"^ORGNR$", r"ORGNR", r"PeOrgNr")
    st = pick(cols, r"status")
    if not (key and st):
        NOTES.append(f"{tab}: key {key}, status {st}; Part C cannot run")
        return set()
    d = pd.read_sql(f"SELECT [{key}] AS employer_id, [{st}] AS status "
                    f"FROM dbo.[{tab}]", conn)
    bad = d["status"].astype(str).str.contains(
        r"konkurs|likvid|bankrupt|liquidat", case=False, na=False)
    out = set(norm_id(d.loc[bad, "employer_id"]))
    print(f"  corporate events: {tab}, {len(out):,} failed firms")
    return out


def add_ind_fe(b: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """industry x age x month, as one integer key beside the existing FEs."""
    b = b.merge(ind, on="employer_id", how="inner")
    if b.empty:
        return b
    ic = pd.factorize(b["ind3"], sort=False)[0].astype("int64")
    ac = pd.factorize(b["age_group"], sort=False)[0].astype("int64")
    tc = pd.factorize(b["year_month"], sort=False)[0].astype("int64")
    n_a, n_t = int(ac.max()) + 1, int(tc.max()) + 1
    b["fe_ind_age_t"] = (ic * n_a + ac) * n_t + tc
    return b


def base_terms(b: pd.DataFrame) -> tuple:
    ym = b["year_month"].astype(str)
    hy = b["high"] * b["young"]
    b["post_rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    b["post_x_high_x_young"] = (ym >= POOLED_FROM).astype(int) * hy
    return b, ["post_rb_x_high_x_young", "post_x_high_x_young"]


def fit(b, terms, fes, tag):
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag=f"r73_{tag}", terms=terms, fes=fes,
                            nthreads=R_THREADS)
    if r.empty:
        FAILURES.append(tag)
        return None
    return {row["term"]: (float(row["coef"]), float(row["se"]))
            for _, row in r.iterrows()}


def gate(b, cov, label) -> bool:
    n = b["employer_id"].nunique()
    matched = b[b["employer_id"].isin(set(cov["employer_id"]))][
        "employer_id"].nunique() if len(cov) else 0
    rate = matched / max(n, 1)
    msg = (f"{label}: {matched:,} of {n:,} panel firms carry the covariate "
           f"({rate:.1%})")
    print(f"    {msg}")
    NOTES.append(msg)
    if matched < MIN_FIRMS or rate < MIN_MATCH_RATE:
        NOTES.append(f"{label}: BELOW THRESHOLD ({matched} < {MIN_FIRMS} or "
                     f"{rate:.1%} < {MIN_MATCH_RATE:.0%}); no estimate "
                     f"reported, by the rule fixed before the run")
        return False
    return True


def run_band(counts, expo, ind, lev, failed, band, j47, sinks):
    l61 = _mod("61_redated_triple.py", "l61")
    skel = l61.build_skeleton(counts, band, j47)
    # counts is 43.8M rows and is not needed again in this band. main()
    # still holds it for the next band, so this only helps if main drops
    # it too -- see the loop there.
    if skel.empty:
        print(f"  {band}: skeleton empty"); return
    # Normalise BEFORE the merge, not after. On 21 September both bands
    # died here with "merge on float64 and object columns": main() had
    # normalised expo to string while the skeleton still carried the
    # float employer_id that L_counts supplies.
    skel["employer_id"] = norm_id(skel["employer_id"])
    b0 = skel.merge(expo[["employer_id", "fq"]], on="employer_id",
                    how="inner")
    del skel
    gc.collect()
    if b0.empty:
        print(f"  {band}: no firms matched exposure"); return
    b0["high"] = (b0["fq"] == 4).astype(int)
    # norm_id leaves employer_id as ~26M Python strings, which every fit
    # then re-hashes (nunique, the cluster factorisation, the merges).
    # The joins are done by here and the labels are never reported, so
    # swap in one compact code and keep the lookup for the arms that
    # still need to match on the original id.
    id_codes, id_labels = pd.factorize(b0["employer_id"], sort=False)
    id_map = pd.Series(np.arange(len(id_labels), dtype="int32"),
                       index=id_labels)
    b0["employer_id"] = id_codes.astype("int32")
    ind = ind.assign(employer_id=ind["employer_id"].map(id_map)).dropna(
        subset=["employer_id"]) if len(ind) else ind
    lev = lev.assign(employer_id=lev["employer_id"].map(id_map)).dropna(
        subset=["employer_id"]) if len(lev) else lev
    if len(ind):
        ind["employer_id"] = ind["employer_id"].astype("int32")
    if len(lev):
        lev["employer_id"] = lev["employer_id"].astype("int32")
    failed = {int(v) for v in pd.Series(list(failed)).map(id_map).dropna()} \
        if failed else failed
    del id_codes, id_labels, id_map
    gc.collect()

    # THE BASELINE FIT, and the reason 73 kept dying where 68 did not.
    # Python and R share ONE job allocation. 68 holds counts and a single
    # panel when it spawns R; 73 was holding counts (43.8M rows), skel,
    # b0 AND a .copy() of b0 -- four large frames -- so R got what was
    # left. Build the terms in place, drop everything not needed, and
    # say how much the process is using so the next run settles it
    # instead of another guess.
    b0, terms = base_terms(b0)
    gc.collect()
    print(f"    python holding {_rss_gb():.1f} GB before the baseline fit")
    base = fit(b0, terms, j47.FES, f"base_{band}")
    for c in terms:
        if c in b0.columns:
            del b0[c]
    gc.collect()
    b = None
    if base:
        sinks["ind"].append({"band": band, "spec": "baseline",
                             **dict(zip(("coef", "se"),
                                        base["post_x_high_x_young"]))})

    # PART A
    if len(ind) and gate(b0, ind, f"industry/{band}"):
        b = add_ind_fe(b0.copy(), ind)
        if not b.empty:
            b, terms = base_terms(b)
            r = fit(b, terms, tuple(j47.FES) + ("fe_ind_age_t",),
                    f"ind_{band}")
            if r:
                sinks["ind"].append({"band": band, "spec": "industry_age_t",
                                     **dict(zip(("coef", "se"),
                                                r["post_x_high_x_young"]))})
        del b
        gc.collect()

    # PART B
    if len(lev) and gate(b0, lev, f"leverage/{band}"):
        b = b0.merge(lev, on="employer_id", how="inner")
        if not b.empty:
            # A median split is not guaranteed to split. If leverage is
            # lumpy, `lev >= median` can select EVERY firm, and then
            # post x young x levhi is just post x young, which the
            # month-by-age effects absorb: the term comes back NaN and
            # the arm silently reports nothing. Check the split, try the
            # strict inequality, and refuse rather than return a hole.
            u = b.drop_duplicates("employer_id")["lev"]
            cut = u.median()
            share = float((u >= cut).mean())
            op = "ge"
            if not (0.05 < share < 0.95):
                share = float((u > cut).mean())
                op = "gt"
            if not (0.05 < share < 0.95):
                NOTES.append(
                    f"leverage/{band}: a median split puts {share:.1%} of "
                    f"firms on one side, so the credit term would not be "
                    f"identified. Part B SKIPPED and said so.")
                b = b.iloc[0:0]
            else:
                b["levhi"] = ((b["lev"] >= cut) if op == "ge"
                              else (b["lev"] > cut)).astype(int)
                NOTES.append(f"leverage/{band}: split at {cut:.3f} "
                             f"({op}), {share:.1%} high")
        if len(b):
            ym = b["year_month"].astype(str)
            post = (ym >= POOLED_FROM).astype(int)
            b, terms = base_terms(b)
            b["post_x_young_x_lev"] = post * b["young"] * b["levhi"]
            b["post_x_high_x_young_x_lev"] = (post * b["high"] * b["young"]
                                              * b["levhi"])
            terms = terms + ["post_x_young_x_lev",
                             "post_x_high_x_young_x_lev"]
            r = fit(b, terms, j47.FES, f"lev_{band}")
            if r:
                for t in ("post_x_high_x_young", "post_x_young_x_lev",
                          "post_x_high_x_young_x_lev"):
                    if t in r:
                        sinks["lev"].append({"band": band, "term": t,
                                             "coef": r[t][0], "se": r[t][1]})
        del b
        gc.collect()

    # PART C
    if failed:
        b = b0[~b0["employer_id"].isin(failed)].copy()
        dropped = b0["employer_id"].nunique() - b["employer_id"].nunique()
        if dropped and not b.empty:
            b, terms = base_terms(b)
            r = fit(b, terms, j47.FES, f"nofail_{band}")
            if r:
                sinks["bank"].append(
                    {"band": band, "firms_dropped": int(dropped),
                     **dict(zip(("coef", "se"), r["post_x_high_x_young"]))})
        del b
        gc.collect()
    del b0
    gc.collect()


def verdict(ind, lev, bank) -> list:
    out = []
    for band in YOUNG_BANDS:
        bl = ind[(ind.band == band) & (ind.spec == "baseline")] \
            if len(ind) and "band" in ind else pd.DataFrame()
        if bl.empty:
            out.append(f"{band}: no baseline fit, nothing to compare")
            continue
        b0 = float(bl.iloc[0]["coef"])
        out.append(f"{band}: baseline {b0:+.4f} "
                   f"({float(bl.iloc[0]['se']):.4f})")

        iv = ind[(ind.band == band) & (ind.spec == "industry_age_t")]
        if len(iv):
            c, se = float(iv.iloc[0]["coef"]), float(iv.iloc[0]["se"])
            keeps = abs(c) >= ATTEN_MAX * abs(b0)
            sig = abs(c / se) >= 1.96 if se else False
            out.append(
                f"{band}: with industry x age x month {c:+.4f} ({se:.4f}). "
                + ("SURVIVES: an age shock common to the industry does not "
                   "explain it." if keeps and sig else
                   "DOES NOT SURVIVE: most of the effect is an "
                   "industry-specific age shock, and the paper must say so."))

        ls = lev[lev.band == band] if len(lev) and "band" in lev else \
            pd.DataFrame()
        if len(ls):
            g = {r["term"]: (r["coef"], r["se"]) for _, r in ls.iterrows()}
            e = g.get("post_x_high_x_young")
            m = g.get("post_x_young_x_lev")
            if e and m:
                keeps = abs(e[0]) >= ATTEN_MAX * abs(b0)
                esig = abs(e[0] / e[1]) >= 1.96 if e[1] else False
                msig = (m[0] / m[1] <= -1.96) if m[1] else False
                out.append(f"{band}: with leverage in, exposure "
                           f"{e[0]:+.4f} ({e[1]:.4f}), leverage x young "
                           f"{m[0]:+.4f} ({m[1]:.4f})")
                if msig and not (keeps and esig):
                    out.append(f"{band}: MONETARY. The credit channel "
                               f"carries it and the exposure term does not "
                               f"survive. We would rather know now.")
                elif keeps and esig:
                    out.append(f"{band}: AI SURVIVES the credit test, which "
                               f"answers R1.2, R1.3 and R2.1 with evidence "
                               f"rather than with a date.")
                else:
                    out.append(f"{band}: INCONCLUSIVE on the credit channel.")

        bk = bank[bank.band == band] if len(bank) and "band" in bank else \
            pd.DataFrame()
        if len(bk):
            c = float(bk.iloc[0]["coef"])
            out.append(f"{band}: dropping {int(bk.iloc[0]['firms_dropped']):,}"
                       f" failed firms gives {c:+.4f}. "
                       + ("Separations are not bankruptcies."
                          if abs(c) >= ATTEN_MAX * abs(b0) else
                          "Much of the effect sits in firms that failed, "
                          "which is not an AI story."))
    return out


def main():
    mc.Tee(OUT / "73_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("73 industry and credit: the two referee points we still argue")
    print(f"gates fixed before the run: {MIN_FIRMS} firms and "
          f"{MIN_MATCH_RATE:.0%} match; attenuation limit {ATTEN_MAX:.0%}")
    print("=" * 70)

    j47 = _mod("47j_within_employer_triple.py", "j47")
    conn = mc.connect()
    schema = discover(conn)
    if schema.empty:
        (OUT / "73_summary.txt").write_text(
            "No FDB_JE or Serrano table is visible to this project. "
            "Neither part can run.\n", encoding="utf-8")
        print("NO TABLES VISIBLE; stopping.")
        return
    schema.to_csv(OUT / "schema_found.csv", index=False)
    print(f"  catalogue: {schema['TABLE_NAME'].nunique()} tables")

    expo = edu_exposure(j47)
    expo["employer_id"] = norm_id(expo["employer_id"])
    print(f"  exposure: {len(expo):,} firms")

    ind = opt("industry", firm_industry, conn, schema)
    lev = opt("leverage", firm_leverage, conn, schema)
    failed = opt("corporate events", firm_failed, conn, schema) or set()
    ind = ind if ind is not None else pd.DataFrame()
    lev = lev if lev is not None else pd.DataFrame()

    cnt = []
    for y in PANEL_YEARS:
        c = mc.read_cache(CACHE / f"L_counts_{y}.parquet")
        if c is None:
            raise SystemExit(f"L_counts_{y}.parquet missing: run 47L first.")
        cnt.append(c)
    counts = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()
    last = str(counts["year_month"].max())
    if last < POOLED_FROM:
        raise SystemExit(f"counts end at {last}, before {POOLED_FROM}.")

    sinks = {"ind": [], "lev": [], "bank": []}
    print(f"  python holding {_rss_gb():.1f} GB before the first band")
    for band in YOUNG_BANDS:
        opt(f"band {band}", run_band, counts, expo, ind, lev, failed, band,
            j47, sinks)
        gc.collect()
        print(f"  python holding {_rss_gb():.1f} GB after band {band}")

    dfi = pd.DataFrame(sinks["ind"])
    dfl = pd.DataFrame(sinks["lev"])
    dfb = pd.DataFrame(sinks["bank"])
    for df, nm in ((dfi, "industry_fe.csv"), (dfl, "credit_test.csv"),
                   (dfb, "bankruptcy.csv")):
        if len(df):
            df.to_csv(OUT / nm, index=False)

    lines = ["73 industry and credit", "=" * 70, "",
             "Industry and leverage are both taken from 2019, frozen "
             "pre-shock: a later vintage would let a firm's own response "
             "into the treatment.", ""]
    lines += verdict(dfi, dfl, dfb)
    lines += ["", "What this cannot do: leverage is a proxy for exposure to "
              "the rate cycle, not a measure of it, and surviving a credit "
              "test is not the same as identifying an AI effect.", ""]
    if NOTES:
        lines += ["NOTES:"] + [f"  {n}" for n in NOTES]
    if FAILURES:
        lines += ["", "FAILED:"] + [f"  {f}" for f in FAILURES]
    (OUT / "73_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("73_industry_and_credit", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

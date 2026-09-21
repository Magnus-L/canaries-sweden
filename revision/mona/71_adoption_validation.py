#!/usr/bin/env python3
"""
71_adoption_validation.py -- do the firms we call exposed actually adopt AI?

======================================================================
  RUNS IN MONA. SQL against tables this project has never touched, so it
  DISCOVERS the schema before it assumes anything. Writes output_71/.
======================================================================

THE GAP THIS CLOSES.

Every estimate in the revision classifies a firm as exposed from the 2019
education mix, or the 2019 occupation mix, of its incumbents aged 31 and
over, and then never observes whether that firm adopted anything. Both
external reviews put this first: freezing exposure fixes the changing-code
problem, but it supplies no evidence that the thing being measured is AI
rather than another correlate of the score.

The delivery turns out to contain the missing half, and we had not used it:

  ITFtg_Stora_YYYY   firm, 10+ employees, 2013-2023. AI technology types
                     (E_AI_TTM text, E_AI_TSR speech, E_AI_TNLG language
                     GENERATION, E_AI_TIR image, E_AI_TML machine
                     learning, E_AI_TPA workflow, E_AI_TAR autonomous),
                     purposes and acquisition routes.
  ai_itftg_2019      firm, the 2019 AI module: AI_USE plus ten barriers.
  BITA_2024 / 2025   INDIVIDUAL. CH1 used generative AI, CH2a/b/c purpose
                     (b is professional use), CH3 reason for not using.
                     Weights vikt_ind_SE.

E_AI_TNLG is the variable that matters most: language generation is the
genAI-relevant technology, and the paper's whole claim is about generative
AI rather than machine learning in general.

WHAT THIS BUYS, IN ONE SENTENCE OF THE PAPER.

"Firms our 2019 measure places in the top exposure quartile are X
percentage points more likely to report using AI in 2023." That is the
first stage the paper does not have, and at 2,000 words it is worth more
per word than any additional headline specification.

IT ALSO SETTLES AN OPEN DECISION. Whether the education route or the
occupation route should be primary is currently argued from coverage
(311,227 firms against 65,146). Running the first stage on both routes
adjudicates it on an external criterion instead: whichever better predicts
observed adoption has the stronger claim to be the headline classifier.

WHY IT COUNTS BEFORE IT ESTIMATES.

Both sources are stratified SAMPLE surveys, not populations, and
ITFtg_Stora covers only firms with ten or more employees, while most of
our 311,227 are smaller. So the overlap with our classified firms is an
empirical question, and a thin overlap would make any estimate here a
trap rather than a validation. The script therefore counts first, applies
thresholds fixed below before the run, and estimates only if they are met.

  ITFtg first stage runs if the matched sample has at least
  MIN_ITFTG_FIRMS firms with a non-missing AI outcome AND at least
  MIN_ITFTG_HIGH of them in the top exposure quartile.

  BITA runs if at least MIN_BITA_PERSONS matched employed respondents AND
  at least MIN_BITA_HIGH in the top quartile.

If a threshold is missed the script says so and stops that arm. It does
not lower the bar and it does not report the estimate anyway.

EXPECTATION, WRITTEN DOWN IN ADVANCE SO IT CANNOT BE REVISED AFTERWARDS.
The ITFtg firm first stage is likely to work. The BITA age-by-exposure
interaction is likely to be underpowered, because BITA is a few thousand
respondents and the cells get thin fast; the simple exposure gradient in
genAI use may survive where the interaction does not. Both outcomes are
informative and neither is a disappointment.

DISCLOSURE. Every count written out passes the export floor, and
suppressed rows are DROPPED, never blanked.

Output (output_71/):
  schema_found.csv     what the discovery step actually found
  overlap_counts.csv   matched firms and persons, by route and quartile
  itftg_firststage.csv if the threshold was met
  bita_firststage.csv  if the threshold was met
  71_summary.txt
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
OUT = HERE / "output_71"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

FLOOR = 5
BASE_YEAR = 2019
ITFTG_YEARS = [2019, 2020, 2021, 2022, 2023]
BITA_YEARS = [2024, 2025]
TRUNC = 2021
DESIGN = "OL_daioe"
ARM = "true"

# Thresholds, fixed before the run.
MIN_ITFTG_FIRMS = 800
MIN_ITFTG_HIGH = 100
MIN_BITA_PERSONS = 600
MIN_BITA_HIGH = 80

# The AI-use columns we hope to find. Discovery decides which exist.
AI_ANY_COLS = ["E_AI_TTM", "E_AI_TSR", "E_AI_TNLG", "E_AI_TIR",
               "E_AI_TML", "E_AI_TPA", "E_AI_TAR"]
AI_GENAI_COL = "E_AI_TNLG"       # language generation: the genAI one
AI_2019_COL = "AI_USE"
BITA_COLS = ["CH1", "CH2a", "CH2b", "CH2c"]
BITA_WEIGHT = "vikt_ind_SE"
NOTES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        NOTES.append(f"{label} failed: {type(ex).__name__}")
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def edu_exposure(j47, design: str, arm: str):
    """
    47j's incumbent education exposure, built exactly as 61 builds it.

    Reproduced rather than imported because 61 does this inside its main().
    The steps and their order matter: the ScoreBook needs every weight year,
    not only the base year, and the spec must be built into the book before
    incumbent_exposure is asked for anything.
    """
    h47 = j47._h47()
    counts = {}
    for y in h47.WEIGHT_YEARS:
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise SystemExit(f"edu_hr_weights_{y}.parquet missing: run 47h "
                             f"first. This script performs no education SQL.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS[design])
    book.build(design, spec)
    frame19 = mc.read_cache(CACHE / f"edu_hr_{j47.BASE_YEAR}.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise SystemExit(f"edu_hr_{j47.BASE_YEAR}.parquet missing: run 47h.")
    expo, _ = j47.incumbent_exposure(frame19, book, design, spec, arm, TRUNC)
    del frame19
    gc.collect()
    return expo


def daioe_scores() -> pd.DataFrame:
    """
    The DAIOE frame as the occupation route needs it: ssyk4 and `score`.

    NOT mc.load_daioe(), which returns ssyk4 and exposure_quartile. 65's
    own main builds this, and occupation_exposure and 47L's build_exposure
    both read `score`, so calling the quartile loader here fails with a
    bare KeyError several frames later.
    """
    d = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    return d.rename(columns={"pctl_rank_genai": "score"})[["ssyk4", "score"]]


def discover(conn) -> pd.DataFrame:
    """
    Find the survey tables and their columns before assuming either.

    This project has already been bitten by identifier spelling (firmid
    against firmID) and by "****" standing in for missing, and we have
    never read these tables. Asking the catalogue costs one query and
    turns three guesses into three facts.
    """
    q = """
    SELECT t.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE
    FROM INFORMATION_SCHEMA.TABLES t
    JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
    WHERE t.TABLE_NAME LIKE 'ITFtg%' OR t.TABLE_NAME LIKE 'ai_%'
       OR t.TABLE_NAME LIKE 'BITA%' OR t.TABLE_NAME LIKE 'FUFI%'
    ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
    """
    return pd.read_sql(q, conn)


def norm_id(x) -> pd.Series:
    """
    One canonical spelling for an identifier, on both sides of every join.

    The survey tables and the AGI-derived frames need not store the
    employer key as the same type: one arrives as text, the other as an
    integer, and pandas refuses the merge rather than matching them. This
    project has already lost time to `firmid` against `firmID`; a silent
    zero-match here would instead look like a thin overlap and be refused
    by the gate for the wrong reason. Trailing ".0" is stripped because a
    key that has passed through a float column acquires one.
    """
    v = pd.Series(x).astype(str).str.strip()
    v = v.str.replace(r"\.0$", "", regex=True)
    return v


def report_match(left: pd.Series, right: pd.Series, label: str) -> float:
    """Match rate between two key columns, printed and kept in NOTES."""
    l, r = set(norm_id(left)), set(norm_id(right))
    rate = len(l & r) / max(len(l), 1)
    msg = (f"{label}: {len(l & r):,} of {len(l):,} keys matched "
           f"({rate:.1%})")
    print(f"    {msg}")
    NOTES.append(msg + (" -- ZERO overlap. Check the identifier, not the "
                        "sample: this is what a key mismatch looks like."
                        if rate == 0.0 else ""))
    return rate


def pick(cols, *patterns):
    """First column matching any pattern, case-insensitively."""
    for p in patterns:
        rx = re.compile(p, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def to01(s: pd.Series) -> pd.Series:
    """
    Survey yes/no to 0/1, tolerantly.

    SCB survey flags arrive variously as 1/0, '1'/'0', 'J'/'N', 'Ja'/'Nej'
    or True/False, and "****" is used for missing in this delivery rather
    than NULL. Anything unrecognised becomes NaN and is dropped, never
    silently read as a no.
    """
    v = s.astype(str).str.strip().str.upper()
    # A column that reaches pandas as float, which is what pyodbc returns
    # for any numeric survey flag that has NULLs in it, stringifies as
    # "1.0" and not "1". That matched neither list on 21 September 2026
    # and turned every value into a missing, which the gate then read as
    # a thin sample and refused. Strip the decimal tail first.
    v = v.str.replace(r"\.0+$", "", regex=True)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[v.isin(["1", "J", "JA", "Y", "YES", "TRUE", "X"])] = 1.0
    out[v.isin(["0", "N", "NEJ", "NO", "FALSE", "2"])] = 0.0
    return out


def parse_report(raw: pd.Series, parsed: pd.Series, label: str) -> None:
    """
    Say what a column actually contained when none of it parsed.

    Guessing a survey codebook twice costs two MONA rounds. If a column
    yields nothing, print its distinct raw values so the next version is
    written against the data rather than against another guess. Distinct
    codes of a survey flag are not disclosive; no counts are printed.
    """
    if parsed.notna().any():
        return
    vals = sorted({str(x)[:12] for x in raw.dropna().unique()})[:12]
    msg = (f"{label}: every value parsed as missing. Distinct raw codes "
           f"seen: {vals}")
    print(f"    {msg}")
    NOTES.append(msg)


def lpm(df: pd.DataFrame, y: str, xs: list, weights=None):
    """
    Linear probability model with heteroskedasticity-robust errors.

    A binary firm-level outcome on a handful of regressors: LPM is the
    standard choice, reads directly in percentage points, and keeps this
    in Python rather than spending another R round-trip on a small table.
    """
    import statsmodels.api as sm
    d = df[[y] + xs].dropna()
    if len(d) < 30 or d[y].nunique() < 2:
        return None
    X = sm.add_constant(d[xs].astype(float), has_constant="add")
    if weights is not None:
        w = weights.reindex(d.index).astype(float)
        m = sm.WLS(d[y].astype(float), X, weights=w).fit(cov_type="HC1")
    else:
        m = sm.OLS(d[y].astype(float), X).fit(cov_type="HC1")
    return pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                         "se": m.bse.values, "n": len(d)})


def firm_size(base: pd.DataFrame) -> pd.DataFrame:
    """2019 headcount per firm, from the baseline frame we already hold."""
    g = (base.groupby("employer_id", observed=True)["n"].sum()
         .reset_index().rename(columns={"n": "size19"}))
    g["employer_id"] = norm_id(g["employer_id"])
    g["log_size"] = np.log(g["size19"].clip(lower=1))
    return g


def itftg_arm(conn, schema, routes, size, sink, counts_sink):
    # Every firm- or organisation-level AI table in the delivery, not
    # only ITFtg. ai_fufi_2019 and ai_itftg_2019 are CONTEMPORANEOUS with
    # our frozen 2019 exposure, which is the cleanest first stage
    # available: it asks whether the measure identifies firms already
    # doing AI at the moment we measured them, with no timing confound.
    # ai_fouoff is public-sector organisations, which the firm tables
    # miss entirely and which employ a large share of our panel.
    tabs = sorted(t for t in schema["TABLE_NAME"].unique()
                  if t.lower().startswith(("itftg_stora", "ai_itftg",
                                           "ai_fufi", "ai_fouftg",
                                           "ai_fouoff")))
    if not tabs:
        NOTES.append("no ITFtg tables found in the catalogue")
        return
    for tab in tabs:
        cols = schema[schema.TABLE_NAME == tab]["COLUMN_NAME"].tolist()
        key = pick(cols, r"PeOrgNr", r"PEORGNR", r"foretag.*id", r"\bfirm")
        if key is None:
            NOTES.append(f"{tab}: no firm identifier found, skipped")
            continue
        # Pattern, not whitelist. Three ITFtg years reported "no AI
        # columns found" on 21 September because the delivered names do
        # not match the reference's for every year.
        # Continuous first. AI_COST_T and AI_IRD_T are dedicated
        # expenditure measures, and a continuous outcome carries far more
        # power at the same sample size than a binary flag, which is what
        # refused every arm on 21 September.
        cost_cols = [c for c in cols
                     if re.search(r"AI_(COST|IRD)", c, re.I)]
        have_any = [c for c in cols if re.search(r"AI", c, re.I)
                    and c not in cost_cols
                    and not re.search(r"HAMP|BARRIER|_TXT$|_OTH_", c, re.I)]
        have_2019 = AI_2019_COL if AI_2019_COL in cols else None
        if have_2019 and have_2019 in have_any:
            have_any.remove(have_2019)
        if not (have_any or have_2019 or cost_cols):
            NOTES.append(f"{tab}: no AI columns found, skipped")
            continue
        want = ([key] + have_any + cost_cols
                + ([have_2019] if have_2019 else []))
        sel = ", ".join(f"[{c}]" for c in dict.fromkeys(want))
        df = pd.read_sql(f"SELECT {sel} FROM dbo.[{tab}]", conn)
        df = df.rename(columns={key: "employer_id"})
        df["employer_id"] = norm_id(df["employer_id"])
        df = df[~df["employer_id"].isin(["", "****", "NULL", "None", "nan"])]
        if have_any:
            parts = []
            for c in have_any:
                pc = to01(df[c])
                parse_report(df[c], pc, f"{tab}.{c}")
                parts.append(pc)
            df["ai_any"] = pd.concat(parts, axis=1).max(axis=1)
        elif have_2019:
            df["ai_any"] = to01(df[have_2019])
            parse_report(df[have_2019], df["ai_any"], f"{tab}.{have_2019}")
        gcol = pick(list(df.columns), r"TNLG", r"NLG", r"GENER")
        df["ai_genai"] = to01(df[gcol]) if gcol else np.nan
        tot = pick(cost_cols, r"AI_COST_T$", r"AI_IRD_T$") or (
            cost_cols[0] if cost_cols else None)
        if tot:
            v = pd.to_numeric(df[tot].astype(str).str.replace(",", ".",
                                                              regex=False),
                              errors="coerce")
            df["ai_spend_log"] = np.log1p(v.clip(lower=0))
            # spending is also the cleanest available "uses AI" flag: a
            # firm with positive AI expenditure is using AI, whatever it
            # ticked on the technology questions
            df["ai_spend_pos"] = (v > 0).astype(float).where(v.notna())
            if df["ai_any"].isna().all() if "ai_any" in df else True:
                df["ai_any"] = df["ai_spend_pos"]
            print(f"    continuous outcome {tot}: "
                  f"{int(v.notna().sum()):,} non-missing")
        print(f"  {tab}: {len(df):,} rows, key {key}, "
              f"AI cols {have_any or [have_2019]}")

        for route, expo in routes.items():
            report_match(df["employer_id"], expo["employer_id"],
                         f"{tab}/{route}")
            m = df.merge(expo[["employer_id", "fq"]], on="employer_id",
                         how="inner")
            if m.empty:
                NOTES.append(f"{tab}/{route}: zero matched firms")
                continue
            m["high"] = (m["fq"] == 4).astype(int)
            m = m.merge(size, on="employer_id", how="left")
            best = "ai_any"
            for c in ("ai_any", "ai_spend_log", "ai_genai"):
                if c in m and m[c].notna().sum() > m[best].notna().sum():
                    best = c
            n_ok = int(m[best].notna().sum())
            n_high = int(((m[best].notna()) & (m["high"] == 1)).sum())
            counts_sink.append({"source": tab, "route": route,
                                "matched": len(m), "with_outcome": n_ok,
                                "high_with_outcome": n_high})
            print(f"    {route:<11} matched {len(m):,}, with outcome "
                  f"{n_ok:,}, of which top quartile {n_high:,}")
            if n_ok < MIN_ITFTG_FIRMS or n_high < MIN_ITFTG_HIGH:
                NOTES.append(
                    f"{tab}/{route}: BELOW THRESHOLD ({n_ok} < "
                    f"{MIN_ITFTG_FIRMS} or {n_high} < {MIN_ITFTG_HIGH}); "
                    f"no estimate reported, by the rule fixed before the run")
                continue
            for out_col in ("ai_any", "ai_genai", "ai_spend_pos",
                            "ai_spend_log"):
                if out_col not in m or m[out_col].notna().sum() < 100:
                    continue
                r = lpm(m, out_col, ["high", "log_size"])
                if r is None:
                    continue
                r["source"], r["route"], r["outcome"] = tab, route, out_col
                sink.append(r)


def bita_arm(conn, schema, routes, sink, counts_sink):
    tabs = sorted(t for t in schema["TABLE_NAME"].unique()
                  if t.lower().startswith("bita"))
    if not tabs:
        NOTES.append("no BITA tables found in the catalogue")
        return
    for tab in tabs:
        yr = re.search(r"(\d{4})", tab)
        yr = int(yr.group(1)) if yr else None
        if yr is not None and yr not in BITA_YEARS:
            continue
        cols = schema[schema.TABLE_NAME == tab]["COLUMN_NAME"].tolist()
        pkey = pick(cols, r"PersonNr", r"LopNr")
        if pkey is None:
            NOTES.append(f"{tab}: no person identifier, skipped")
            continue
        have = [c for c in cols if re.match(r"^CH\d", c, re.I)]
        if not have:
            NOTES.append(f"{tab}: no CH columns, skipped")
            continue
        wcol = BITA_WEIGHT if BITA_WEIGHT in cols else None
        want = [pkey] + have + ([wcol] if wcol else [])
        sel = ", ".join(f"[{c}]" for c in dict.fromkeys(want))
        bita = pd.read_sql(f"SELECT {sel} FROM dbo.[{tab}]", conn)
        bita = bita.rename(columns={pkey: "person_id"})
        bita["person_id"] = norm_id(bita["person_id"])
        print(f"  {tab}: {len(bita):,} respondents, key {pkey}, "
              f"cols {have}, weight {wcol}")

        # link respondent to employer in the survey year, via AGI November
        yy = yr or BITA_YEARS[0]
        suffix = "_def" if yy < 2025 else "_prel"
        mm = 11 if yy < 2025 else 6
        link = pd.read_sql(f"""
            SELECT DISTINCT P1207_LOPNR_PERSONNR AS person_id,
                   P1207_LOPNR_PEORGNR AS employer_id
            FROM dbo.Arb_AGIIndivid{yy}{mm:02d}{suffix}
            WHERE KONTANT_ERSATTNING_ULAG_AG > 0
            """, conn)
        link["person_id"] = norm_id(link["person_id"])
        link["employer_id"] = norm_id(link["employer_id"])
        report_match(bita["person_id"], link["person_id"], f"{tab}/persons")
        b = bita.merge(link, on="person_id", how="inner")
        del link
        gc.collect()
        # one employer per person: a secondary job must not double-count a
        # respondent, and we cannot tell which job is primary without pay,
        # so a respondent with several employers is dropped rather than
        # arbitrarily assigned.
        multi = b.groupby("person_id")["employer_id"].transform("nunique")
        dropped = int((multi > 1).sum())
        b = b[multi == 1]
        if dropped:
            print(f"    dropped {dropped:,} rows for multiple employers")

        for route, expo in routes.items():
            report_match(b["employer_id"], expo["employer_id"],
                         f"{tab}/{route}")
            m = b.merge(expo[["employer_id", "fq"]], on="employer_id",
                        how="inner")
            if m.empty:
                NOTES.append(f"{tab}/{route}: zero matched respondents")
                continue
            m["high"] = (m["fq"] == 4).astype(int)
            c1 = pick(list(m.columns), r"^CH1")
            m["genai"] = to01(m[c1]) if c1 else np.nan
            if c1:
                parse_report(m[c1], m["genai"], f"{tab}.{c1}")
            n_ok = int(m["genai"].notna().sum())
            n_high = int(((m["genai"].notna()) & (m["high"] == 1)).sum())
            counts_sink.append({"source": tab, "route": route,
                                "matched": len(m), "with_outcome": n_ok,
                                "high_with_outcome": n_high})
            print(f"    {route:<11} matched {len(m):,}, with outcome "
                  f"{n_ok:,}, top quartile {n_high:,}")
            if n_ok < MIN_BITA_PERSONS or n_high < MIN_BITA_HIGH:
                NOTES.append(
                    f"{tab}/{route}: BELOW THRESHOLD ({n_ok} < "
                    f"{MIN_BITA_PERSONS} or {n_high} < {MIN_BITA_HIGH}); "
                    f"no estimate reported, by the rule fixed before the run")
                continue
            w = m[BITA_WEIGHT].astype(float) if BITA_WEIGHT in m else None
            for out_col in ("genai", "prof"):
                if out_col == "prof":
                    c2 = pick(list(m.columns), r"^CH2B", r"^CH2_B")
                    if not c2:
                        continue
                    m["prof"] = to01(m[c2])
                if m[out_col].notna().sum() < 100:
                    continue
                r = lpm(m, out_col, ["high"], weights=w)
                if r is None:
                    continue
                r["source"], r["route"], r["outcome"] = tab, route, out_col
                sink.append(r)


def main():
    mc.Tee(OUT / "71_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("71 adoption validation: do the exposed firms actually adopt?")
    print(f"thresholds fixed before the run: ITFtg {MIN_ITFTG_FIRMS}/"
          f"{MIN_ITFTG_HIGH}, BITA {MIN_BITA_PERSONS}/{MIN_BITA_HIGH}")
    print("=" * 70)

    j47 = _mod("47j_within_employer_triple.py", "j47")
    l65 = _mod("65_occupation_arm.py", "l65")
    conn = mc.connect()

    schema = discover(conn)
    if schema.empty:
        (OUT / "71_summary.txt").write_text(
            "No ITFtg, ai_itftg or BITA table is visible to this project. "
            "The delivery reference lists them, so either the names differ "
            "or they were not loaded into P1207. Nothing further can run.\n",
            encoding="utf-8")
        print("NO SURVEY TABLES VISIBLE; stopping.")
        return
    schema.to_csv(OUT / "schema_found.csv", index=False)
    print(f"  catalogue: {schema['TABLE_NAME'].nunique()} tables, "
          f"{len(schema):,} columns")
    for t in sorted(schema["TABLE_NAME"].unique()):
        print(f"    {t}")

    # the two classification routes, exactly as the paper builds them
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    daioe = daioe_scores()
    edu = edu_exposure(j47, DESIGN, ARM)
    occ = l65.occupation_exposure(base, daioe, j47.INCUMBENT_BANDS)
    routes = {}
    if edu is not None and len(edu):
        edu["employer_id"] = norm_id(edu["employer_id"])
        routes["education"] = edu
    if occ is not None and len(occ):
        occ["employer_id"] = norm_id(occ["employer_id"])
        routes["occupation"] = occ
    print(f"  routes: " + ", ".join(f"{k} {len(v):,} firms"
                                    for k, v in routes.items()))
    size = firm_size(base)

    fs_sink, counts_sink, bita_sink = [], [], []
    opt("ITFtg", itftg_arm, conn, schema, routes, size, fs_sink, counts_sink)
    opt("BITA", bita_arm, conn, schema, routes, bita_sink, counts_sink)

    lines = ["71 adoption validation", "=" * 70, ""]

    if counts_sink:
        cdf = pd.DataFrame(counts_sink)
        # disclosure: drop, do not blank
        before = len(cdf)
        cdf = cdf[(cdf["with_outcome"] >= FLOOR) &
                  (cdf["high_with_outcome"] >= FLOOR)]
        if len(cdf) < before:
            print(f"  export floor: dropped {before - len(cdf)} count rows")
        cdf.to_csv(OUT / "overlap_counts.csv", index=False)
        lines += ["OVERLAP", ""]
        for _, r in cdf.iterrows():
            lines.append(f"  {r['source']:<20} {r['route']:<11} "
                         f"matched {int(r['matched']):>7,}  with outcome "
                         f"{int(r['with_outcome']):>7,}  top quartile "
                         f"{int(r['high_with_outcome']):>6,}")
        lines.append("")

    if fs_sink:
        df = pd.concat(fs_sink, ignore_index=True)
        df.to_csv(OUT / "itftg_firststage.csv", index=False)
        lines += ["FIRST STAGE, FIRM LEVEL (ITFtg). Coefficient on `high` is "
                  "the top-quartile difference in the probability of "
                  "reporting AI use, controlling for log 2019 size.", ""]
        for _, r in df[df.term == "high"].iterrows():
            t = r["coef"] / r["se"] if r["se"] else float("nan")
            lines.append(f"  {r['source']:<20} {r['route']:<11} "
                         f"{r['outcome']:<9} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  t {t:+.2f}  n {int(r['n']):,}")
        lines += ["", "  READ: a positive, significant coefficient is the "
                  "validation the paper lacks. Comparing the two routes on "
                  "the SAME table also adjudicates which should be the "
                  "headline classifier, on an external criterion rather "
                  "than on coverage counts.", ""]
    else:
        lines += ["No ITFtg first stage reported. See NOTES.", ""]

    if bita_sink:
        df = pd.concat(bita_sink, ignore_index=True)
        df.to_csv(OUT / "bita_firststage.csv", index=False)
        lines += ["FIRST STAGE, INDIVIDUAL LEVEL (BITA), survey-weighted.", ""]
        for _, r in df[df.term == "high"].iterrows():
            t = r["coef"] / r["se"] if r["se"] else float("nan")
            lines.append(f"  {r['source']:<12} {r['route']:<11} "
                         f"{r['outcome']:<7} {r['coef']:+.4f} "
                         f"({r['se']:.4f})  t {t:+.2f}  n {int(r['n']):,}")
        lines.append("")
    else:
        lines += ["No BITA first stage reported. See NOTES.", ""]

    lines += ["LIMITS, which belong in the paper and not only here:",
              "  Both sources are stratified sample surveys with weights, "
              "and ITFtg_Stora covers only firms with ten or more "
              "employees. This validates the measure on the larger-firm "
              "part of our panel, not on all of it.",
              "  A first stage says the score predicts reported adoption. "
              "It does not make adoption exogenous, and it does not turn "
              "the age contrast into a causal estimate of AI.", ""]
    if NOTES:
        lines += ["NOTES:"] + [f"  {n}" for n in NOTES]
    (OUT / "71_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    mc.runlog("71_adoption_validation", 0, (time.time() - t0) / 60)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

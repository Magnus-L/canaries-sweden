#!/usr/bin/env python3
"""
76_gender_decomposition.py: how much of the female differential is where
young women work, and how much is being affected more in the same work.

QUESTION
The female differential of Table 1 compares young women with young men in
the same employers. Young women and young men hold different educations,
so part of the differential may be composition. Occupation cannot answer
this on the reported design, which classifies no young worker by
occupation after 2019, so the split is made on the bridge the exposure
itself uses: education, recorded for every worker in every year. The
education register ends in 2023, so the 2024 and 2025 records are the
2023 ones carried forward, which is why the split runs on five broad
tracks rather than on the 105 fine groups.

DESIGN
Tracks by two-digit SUN 2020 field: ict (48); engineering (52, 54, 58);
business, law and social science (31, 32, 34, 38); health, education and
care (14, 72, 76); other; a missing record is 'na' and is reported, never
fitted. Levels: below upper secondary, upper secondary, post-secondary.

1. Composition, descriptive: employed 22-25 year olds in 2023 by sex,
   inside and outside the top exposure quartile, as person-months by track
   and by level, their shares, and the mean DAIOE score of the education
   groups they hold (script 47h's OL_daioe group score); cells with fewer
   than five persons on average across the year's months are dropped.
2. Within, estimated: script 68's gender specification (the sex panel of
   script 67 at 22-25 on the stock; PostRB and Post x High x Young, Post x
   High x Female, Post x High x Young x Female, the three quarter terms for
   High x Young and for High x Young x Female; employer-by-month,
   employer-by-age-and-sex and month-by-age-and-sex effects; Poisson;
   clustered by employer), first on all workers as a reproduction gate and
   then on the workers of one track at a time, so that the differential is
   identified among women and men with the same broad education against
   their same-track older colleagues.
3. The split: within equals the sum over fitted tracks of young women's
   track share in exposed employers in 2023 times that track's
   differential, weights renormalised over the fitted tracks, with a
   standard error that treats the track fits as independent; composition
   is the pooled differential minus within.

Read rule fixed before the run: the all-track differential must lie within
one standard error of script 68's or nothing is quoted; the verdict is
composition if within is at or below half the pooled differential in
absolute value, being affected more in the same work if at or above three
quarters, ambiguous between. No track is promoted above the pooled
profile.

INPUTS AND OUTPUTS
Pulls, in MONA, the monthly employer declarations for 2021 to 2025 joined
to Individ_2023, 2021 and 2019 for birth year and sex and to the year's
own Individ table (Individ_2023 from 2023 onward) for the education level
and field, aggregated to employer by month by age band by sex by level by
field and cached as L_counts_sex_edu_YYYY.parquet. Reads script 47h's
caches for the exposure and the score book. Writes to output_76/:
education_mix_by_sex.csv, gender_by_track.csv, gender_split.csv,
vcov_s76_<track>.csv and 76_summary.txt.

IN THE PAPER
Table 1, the within-track row (-0.0508, SE 0.0118); Section 3 (three
quarters of the differential survives within broad education tracks);
Online Appendix III.2, "The female differential, split", and the tables
tableA_gender_split and tableA_education_mix.
"""

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_76"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YEARS = list(range(2021, 2026))          # 61's panel window
POST_FROM = "2024-01"
MIX_YEAR = 2023                          # last complete Individ year
LAST_EDU_YEAR = 2023                     # Individ ends here; later years carry it forward
EXPORT_FLOOR = 5
SEX = {"1": "men", "2": "women"}         # SCB coding, verified in script 48
YOUNG = "22-25"

# SUN 2020 inriktning, two-digit field. The delivered key
# (utb_grupp2_sun2020_niva3_inr4_nyckel) carries the SUN 2000 field
# numbering, which SUN 2020 retained: 48 data, 52 teknik och teknisk
# industri, 54 material och tillverkning, 58 samhallsbyggnad och
# byggnadsteknik, 31 samhalls- och beteendevetenskap, 32 journalistik,
# 34 foretagsekonomi, handel och administration, 38 juridik, 14 pedagogik,
# 72 halso- och sjukvard, 76 socialt arbete. Everything else is "other";
# a missing or unmatched record is "na" and is reported, never fitted.
TRACKS = {
    "ict":                   ("48",),
    "engineering":           ("52", "54", "58"),
    "business_law_social":   ("31", "32", "34", "38"),
    "health_education_care": ("14", "72", "76"),
}
TRACK_ORDER = ["ict", "engineering", "business_law_social",
               "health_education_care", "other"]
LEVELS = {"below_upper_secondary": ("0", "1", "2"),
          "upper_secondary": ("3",), "post_secondary": ("4", "5", "6")}

S68_DIFF = (-0.0659, 0.0131)             # 68's pooled differential (gate)
RULE_COMPOSITION, RULE_HIT = 0.50, 0.75  # the split's read rule
FAILURES = []
GATE = {"ok": None, "detail": ""}


def _mod(fname, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def quarter_of_year(ym: pd.Series) -> pd.Series:
    return ((ym.str.slice(5, 7).astype(int) - 1) // 3) + 1


# ----------------------------------------------------------------------
# SQL: 67's counts-by-sex query with the worker's education added
# ----------------------------------------------------------------------

AGE_CASE = """CASE
        WHEN {y} - fodelse BETWEEN 22 AND 25 THEN '22-25'
        WHEN {y} - fodelse BETWEEN 26 AND 30 THEN '26-30'
        WHEN {y} - fodelse BETWEEN 31 AND 34 THEN '31-34'
        WHEN {y} - fodelse BETWEEN 35 AND 40 THEN '35-40'
        WHEN {y} - fodelse BETWEEN 41 AND 49 THEN '41-49'
        WHEN {y} - fodelse BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def q_counts_sex_edu(year: int, conn) -> pd.DataFrame:
    """
    Employer x month x age band x sex x (SUN 2020 level, field) -> distinct
    persons. Birth year and sex come from whichever Individ vintage holds
    the person, exactly as in 67. Education comes from the Individ table
    of the year itself up to 2023 and from Individ_2023 thereafter, the
    same rule 47h applies (own = min(year, 2023)); the carry-forward is
    measured and reported, not hidden. Column names are the ones 47h
    reads (Sun2020Niva, Sun2020Inr), not guessed.
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    own = min(year, LAST_EDU_YEAR)
    if own == 2023:
        edu, extra = "a", ""
    elif own == 2021:
        edu, extra = "b", ""
    else:
        edu = "t"
        extra = (f"LEFT JOIN dbo.Individ_{own} t "
                 f"ON agi.P1207_LOPNR_PERSONNR = t.P1207_LopNr_PersonNr")
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               COALESCE(a.Kon, b.Kon, c.Kon) AS gender,
               NULLIF(LTRIM(RTRIM({edu}.Sun2020Niva)),'') AS niva,
               NULLIF(LTRIM(RTRIM({edu}.Sun2020Inr)),'')  AS inr
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        {extra}
        """ for m in range(1, max_month + 1))
    age_case = AGE_CASE.format(y=year)
    q = f"""
    WITH base AS ({monthly})
    SELECT employer_id,
           CONCAT(LEFT(period, 4), '-', RIGHT(period, 2)) AS year_month,
           {age_case} AS age_group,
           LTRIM(RTRIM(gender)) AS gender,
           niva, inr,
           COUNT(DISTINCT person_id) AS n_emp
    FROM base
    WHERE fodelse IS NOT NULL AND gender IN ('1', '2')
      AND {year} - fodelse BETWEEN 22 AND 69
    GROUP BY employer_id, period, {age_case}, LTRIM(RTRIM(gender)), niva, inr
    """
    chunks = [compact(ch) for ch in pd.read_sql(q, conn, chunksize=2_000_000)]
    if not chunks:
        return compact(pd.DataFrame(columns=EDU_COLS + ["n_emp"]))
    out = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return out


EDU_COLS = ["employer_id", "year_month", "age_group", "gender", "niva", "inr"]


def compact(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("year_month", "age_group", "gender", "niva", "inr"):
        if c in df.columns:
            df[c] = df[c].astype("string").astype("category")
    if "n_emp" in df.columns:
        df["n_emp"] = df["n_emp"].astype("int32")
    return df


# ----------------------------------------------------------------------
# Tracks, levels, and the collapse to a sex panel
# ----------------------------------------------------------------------

def track_of(inr: pd.Series) -> pd.Series:
    """Two-digit SUN field -> track; missing or unmatched -> 'na'."""
    s = inr.astype("string").str.strip().str.slice(0, 2)
    out = pd.Series("other", index=inr.index, dtype="object")
    for name, codes in TRACKS.items():
        out[s.isin(list(codes)).fillna(False).to_numpy()] = name
    out[(s.isna() | (s == "")).to_numpy()] = "na"
    return out


def level_of(niva: pd.Series) -> pd.Series:
    s = niva.astype("string").str.strip().str.slice(0, 1)
    out = pd.Series("na", index=niva.index, dtype="object")
    for name, codes in LEVELS.items():
        out[s.isin(list(codes)).fillna(False).to_numpy()] = name
    return out


def tag_frame(df: pd.DataFrame, h47) -> pd.DataFrame:
    """Add normalised niva/inr (47h's rule), track and level columns."""
    d = df.copy()
    d["niva_n"] = h47.norm_code(d["niva"].astype("string"))
    d["inr_n"] = h47.norm_code(d["inr"].astype("string"))
    d["track"] = track_of(d["inr_n"])
    d["level"] = level_of(d["niva_n"])
    return d


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Sum the education dimension away: 67's cell, employer x age x sex x month."""
    c = (df.groupby(["employer_id", "year_month", "age_group", "gender"],
                    observed=True)["n_emp"].sum().reset_index())
    c["year_month"] = c["year_month"].astype(str)
    c["age_group"] = c["age_group"].astype(str)
    c["gender"] = c["gender"].astype(str)
    c["n_emp"] = c["n_emp"].astype(int)
    return c


# ----------------------------------------------------------------------
# 68's gender specification, verbatim in its term set
# ----------------------------------------------------------------------

def gender_terms_seasonal(b: pd.DataFrame) -> tuple:
    ym = b["year_month"].astype(str)
    q = quarter_of_year(ym)
    hy = b["high"] * b["young"]
    hyf = hy * b["female"]
    post = (ym >= POST_FROM).astype(int)
    b["rb_x_high_x_young"] = (ym >= mc.RIKSBANK_YM).astype(int) * hy
    b["post_x_high_x_young"] = post * hy
    b["post_x_high_x_female"] = post * b["high"] * b["female"]
    b["post_x_high_x_young_x_female"] = post * hyf
    terms = ["rb_x_high_x_young", "post_x_high_x_young",
             "post_x_high_x_female", "post_x_high_x_young_x_female"]
    for qq in (1, 2, 3):
        b[f"q{qq}_x_high_x_young"] = (q == qq).astype(int) * hy
        b[f"q{qq}_x_high_x_young_x_female"] = (q == qq).astype(int) * hyf
        terms += [f"q{qq}_x_high_x_young", f"q{qq}_x_high_x_young_x_female"]
    return b, terms


def fit_gender(counts_sex: pd.DataFrame, expo: pd.DataFrame, s67, j47,
               tag: str) -> dict | None:
    skel = s67.build_skeleton_sex(counts_sex, YOUNG, j47, "n_emp")
    if skel.empty:
        print(f"  {tag}: empty skeleton")
        return None
    b = skel.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    del skel
    gc.collect()
    if b.empty:
        print(f"  {tag}: no firms matched exposure")
        return None
    b["high"] = (b["fq"] == 4).astype(int)
    b, terms = gender_terms_seasonal(b)
    n_firms = int(b["employer_id"].nunique())
    print(f"  {tag}: {len(b):,} rows, {n_firms:,} firms{mc.mem_line(' | ')}")
    r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=j47.FES)
    del b
    gc.collect()
    if r.empty:
        return None
    g = r.set_index("term")
    need = ("post_x_high_x_young", "post_x_high_x_young_x_female")
    if any(t not in g.index for t in need):
        return None
    return {"male": float(g.loc[need[0], "coef"]),
            "male_se": float(g.loc[need[0], "se"]),
            "diff": float(g.loc[need[1], "coef"]),
            "diff_se": float(g.loc[need[1], "se"]),
            "n_obs": int(g.loc[need[1], "n_obs"]), "n_firms": n_firms,
            "status": str(g.loc[need[1]].get("status", "ok"))}


# ----------------------------------------------------------------------
# Composition, descriptive, 2023
# ----------------------------------------------------------------------

def education_mix(frame: pd.DataFrame, expo: pd.DataFrame, book, spec,
                  h47) -> pd.DataFrame:
    """
    Employed 22-25 year olds in MIX_YEAR by sex, inside and outside the
    top exposure quartile: person-months by track and by level, the share
    of each sex in each, and the mean DAIOE score of the education groups
    they hold (47h's group score, on its own scale). Floored at
    EXPORT_FLOOR persons on average across the year's months.
    """
    f = frame[frame["age_group"].astype(str) == YOUNG]
    f = f.merge(expo[["employer_id", "fq"]], on="employer_id", how="inner")
    f["exposed"] = (f["fq"] == 4).astype(int)
    n = len(f)
    score = book.score_frame("OL_daioe", spec, f["niva_n"].to_numpy(),
                             f["inr_n"].to_numpy(), np.full(n, "na"), None)
    f["score"] = score
    months = f["year_month"].astype(str).nunique()
    rows = []

    def _cell(d: pd.DataFrame) -> pd.Series:
        # The mean score is taken over the workers whose education group
        # HAS a score; a pair the score book cannot place (no group in the
        # key, or a group below 47h's MIN_CELL) is left out of the mean
        # and counted in scored_share, so the reader sees how much of the
        # cell the mean describes.
        ok = d["score"].notna()
        pm = int(d["n_emp"].sum())
        pm_ok = int(d.loc[ok, "n_emp"].sum())
        mean = (float(np.average(d.loc[ok, "score"], weights=d.loc[ok, "n_emp"]))
                if pm_ok > 0 else np.nan)
        return pd.Series({"person_months": pm, "mean_score": mean,
                          "scored_share": (pm_ok / pm) if pm else np.nan})

    for dim in ("track", "level"):
        g = (f.groupby(["exposed", "gender", dim], observed=True)
             .apply(_cell).reset_index())
        tot = g.groupby(["exposed", "gender"])["person_months"].transform("sum")
        g["share"] = g["person_months"] / tot
        g["persons_avg"] = g["person_months"] / max(months, 1)
        g = g[g["persons_avg"] >= EXPORT_FLOOR]
        g["dimension"] = dim
        g = g.rename(columns={dim: "cell"})
        rows.append(g[["dimension", "exposed", "gender", "cell",
                       "person_months", "persons_avg", "share", "mean_score",
                       "scored_share"]])
    out = pd.concat(rows, ignore_index=True)
    out["gender"] = out["gender"].astype(str).map(SEX).fillna(out["gender"].astype(str))
    out["year"] = MIX_YEAR
    return out


def carry_forward_share(frames: dict) -> dict:
    """Share of young person-months in each post-2023 year whose education
    record is the carried-forward 2023 one (all of them, by construction)
    and the share with NO record at all."""
    out = {}
    for y, fr in frames.items():
        f = fr[fr["age_group"].astype(str) == YOUNG]
        tot = int(f["n_emp"].sum())
        na = int(f.loc[f["track"] == "na", "n_emp"].sum())
        out[y] = {"person_months": tot, "no_record_share": (na / tot) if tot else np.nan,
                  "carried_forward": y > LAST_EDU_YEAR}
    return out


# ----------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------

def split(pooled: dict, by_track: dict, weights: dict) -> dict:
    """within = sum_g w_g * diff_g over fitted tracks (weights renormalised
    to the fitted tracks), composition = pooled - within."""
    fitted = [g for g in TRACK_ORDER if g in by_track and by_track[g]]
    wsum = sum(weights.get(g, 0.0) for g in fitted)
    if not fitted or wsum <= 0 or not pooled:
        return {}
    w = {g: weights.get(g, 0.0) / wsum for g in fitted}
    within = sum(w[g] * by_track[g]["diff"] for g in fitted)
    within_se = float(np.sqrt(sum((w[g] * by_track[g]["diff_se"]) ** 2
                                  for g in fitted)))
    comp = pooled["diff"] - within
    ratio = within / pooled["diff"] if pooled["diff"] else np.nan
    if np.isnan(ratio):
        verdict = "NO VERDICT"
    elif ratio <= RULE_COMPOSITION:
        verdict = "COMPOSITION"
    elif ratio >= RULE_HIT:
        verdict = "HIT HARDER"
    else:
        verdict = "AMBIGUOUS"
    return {"pooled": pooled["diff"], "pooled_se": pooled["diff_se"],
            "within": within, "within_se": within_se,
            "composition": comp, "ratio_within": ratio,
            "weights_renormalised_over": ",".join(fitted),
            "weight_mass_fitted": wsum, "verdict": verdict}


def main():
    mc.Tee(OUT / "76_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("76: THE FEMALE DIFFERENTIAL, SPLIT INTO COMPOSITION AND WITHIN")
    print("=" * 70)
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    j47 = s61._j47()
    h47 = j47._h47()

    wt = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        wt[y] = w
    book = h47.ScoreBook(wt, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    expo, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, "true",
                                     s61.TRUNC)
    print(f"  exposure: {len(expo):,} firms")
    del frame19
    gc.collect()

    # ---- the counts, by education, cached ----------------------------
    conn = None

    def _conn():
        nonlocal conn
        if conn is None:
            conn = mc.connect()
        return conn

    frames = {}
    for y in YEARS:
        cf = CACHE / f"L_counts_sex_edu_{y}.parquet"
        c = mc.read_cache(cf, require=EDU_COLS + ["n_emp"])
        if c is None:
            t = time.time()
            c = q_counts_sex_edu(y, _conn())
            mc.write_cache(c, cf)
            print(f"  counts by sex and education {y}: {len(c):,} cells "
                  f"({(time.time()-t)/60:.1f} min)")
        else:
            print(f"  counts by sex and education {y}: cached ({len(c):,} cells)")
        frames[y] = tag_frame(c, h47)
        del c
        gc.collect()
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    # ---- 1. composition, 2023 ----------------------------------------
    mix = opt("education mix", education_mix, frames[MIX_YEAR], expo, book,
              spec, h47)
    weights = {}
    if mix is not None:
        mix.to_csv(OUT / "education_mix_by_sex.csv", index=False)
        w = mix[(mix["dimension"] == "track") & (mix["exposed"] == 1)
                & (mix["gender"] == "women")]
        weights = dict(zip(w["cell"], w["share"]))
        print("  young women's track shares in exposed firms, 2023: "
              + ", ".join(f"{k} {v:.3f}" for k, v in weights.items()))
    cf_share = carry_forward_share(frames)

    # ---- 2. within, per track, with the reproduction gate first -------
    allf = pd.concat(frames.values(), ignore_index=True)
    by_track, rows = {}, []
    pooled = fit_gender(collapse(allf), expo, s67, j47, "s76_all")
    if pooled is None:
        FAILURES.append("gender/all")
    else:
        rows.append(dict(track="all", **pooled))
        d, se = pooled["diff"], pooled["diff_se"]
        GATE["ok"] = abs(d - S68_DIFF[0]) <= max(se, S68_DIFF[1])
        GATE["detail"] = (f"all-track differential {d:+.4f} ({se:.4f}) against "
                          f"68's {S68_DIFF[0]:+.4f} ({S68_DIFF[1]:.4f})")
        print(f"  GATE {'PASS' if GATE['ok'] else 'FAIL'}: {GATE['detail']}")
    for g in TRACK_ORDER:
        sub = allf[allf["track"] == g]
        if sub.empty:
            print(f"  track {g}: no workers")
            continue
        res = opt(f"gender fit, track {g}", fit_gender, collapse(sub), expo,
                  s67, j47, f"s76_{g}")
        if res is None:
            FAILURES.append(f"gender/{g}")
            continue
        by_track[g] = res
        rows.append(dict(track=g, **res))
        pd.DataFrame(rows).to_csv(OUT / "gender_by_track.csv", index=False)
        print(f"    {g:<24} men {res['male']:+.4f} ({res['male_se']:.4f})  "
              f"female differential {res['diff']:+.4f} ({res['diff_se']:.4f}) "
              f"t {res['diff']/max(res['diff_se'],1e-12):+.2f}")
    del allf
    gc.collect()
    pd.DataFrame(rows).to_csv(OUT / "gender_by_track.csv", index=False)

    # ---- 3. the split ------------------------------------------------
    sp = split(pooled, by_track, weights) if pooled else {}
    if sp:
        pd.DataFrame([sp]).to_csv(OUT / "gender_split.csv", index=False)

    # ---- summary -----------------------------------------------------
    lines = ["THE FEMALE DIFFERENTIAL AT 22-25, SPLIT", "=" * 52, "",
             "68's gender specification (cycle removed, employer-by-month",
             "effects), run on all workers and then within each broad",
             "education track. Education is the worker's own SUN 2020 record;",
             "Individ ends at 2023 and later years carry it forward.", ""]
    if GATE["ok"] is not None:
        lines += [f"REPRODUCTION GATE: {'PASS' if GATE['ok'] else 'FAIL'}. "
                  f"{GATE['detail']}",
                  "  A FAIL means this panel is not the paper's; quote nothing.", ""]
    if rows:
        lines += ["MALE EFFECT AND FEMALE DIFFERENTIAL BY TRACK (post from "
                  f"{POST_FROM}, cycle removed):"]
        for r in rows:
            lines.append(f"  {r['track']:<24} men {r['male']:+.4f} ({r['male_se']:.4f})"
                         f"   women minus men {r['diff']:+.4f} ({r['diff_se']:.4f})"
                         f" t {r['diff']/max(r['diff_se'],1e-12):+.2f}"
                         f"   firms {r['n_firms']:,}")
        lines.append("")
    if weights:
        lines += ["YOUNG WOMEN'S TRACK SHARES IN EXPOSED FIRMS, 2023 (the weights):",
                  "  " + ", ".join(f"{k} {v:.3f}" for k, v in weights.items()), ""]
    if sp:
        lines += ["THE SPLIT:",
                  f"  pooled differential      {sp['pooled']:+.4f} ({sp['pooled_se']:.4f})",
                  f"  within tracks            {sp['within']:+.4f} ({sp['within_se']:.4f})"
                  f"   [weights renormalised over {sp['weights_renormalised_over']},"
                  f" mass {sp['weight_mass_fitted']:.3f}; track fits treated as independent]",
                  f"  composition (residual)   {sp['composition']:+.4f}",
                  f"  ratio within / pooled    {sp['ratio_within']:.3f}",
                  f"  VERDICT on the rule fixed before the run "
                  f"(<= {RULE_COMPOSITION:.2f} composition, >= {RULE_HIT:.2f} hit harder): "
                  f"{sp['verdict']}", ""]
    lines += ["EDUCATION RECORD COVERAGE AMONG 22-25 PERSON-MONTHS:"]
    for y, d in cf_share.items():
        lines.append(f"  {y}: {d['person_months']:,} person-months, no record "
                     f"{100*d['no_record_share']:.1f} per cent"
                     + ("; education carried forward from 2023" if d["carried_forward"] else ""))
    lines += ["",
              "READ THIS BEFORE QUOTING ANY OF IT:",
              "  1. The gate must PASS. Otherwise nothing here is the paper's.",
              "  2. Report BOTH within and composition, and the ratio, whichever",
              "     way the verdict falls. One sentence in the paper, one OA table.",
              "  3. No track becomes a headline. The cut is heterogeneity.",
              "  4. The 2024-25 education record is the 2023 one for every worker;",
              "     the tracks are broad for that reason. Say so once.",
              "  5. The mix table is descriptive and floored at "
              f"{EXPORT_FLOOR} persons per cell.", ""]
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "76_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    mc.runlog("76_gender_decomposition", 0, (time.time() - t0) / 60)
    print("\n76 done.")


if __name__ == "__main__":
    main()

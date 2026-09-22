#!/usr/bin/env python3
"""
67_gender_on_the_new_design.py: the design split by sex, and the sex panel
the gender estimates are fitted on.

QUESTION
Does the decline of the young inside exposed employers fall on young
women, on young men, or on both? Two coefficients estimated separately by
sex cannot establish a difference between them; the difference needs its
own interaction and standard error. This script adds sex as a fourth
dimension of the cell and estimates the female differential directly,
within the same employers and against the same older colleagues as the
headline.

DESIGN
Panel (build_skeleton_sex): employer by age band by sex by month counts
from January 2021, the young band beside the four incumbent bands, an
employer entering if it holds the young band and an incumbent band, cells
zero-filled, employer-band-sex cells that are zero in every month dropped.
The fixed effects move with the cell: employer by month, employer by
age-and-sex, and month by age-and-sex, so the national path of young
women is absorbed as the national path of the young is in the headline.
Exposure is the headline classification (script 47j's incumbent_exposure
on the OL_daioe score book), true and as-of arms.

Terms (add_gender_terms): for each of the tightening date (April 2022)
and the post date, the interactions with High x Young, High x Female and
High x Young x Female. The last is the female differential and its t
statistic is the test; the interaction with High alone is constant within
an employer-month and is absorbed. The post date is the launch (December
2022) and, separately, adoption (January 2024). Outcomes: the employment
stock, hires and separations; both young bands; the as-of arm at 22-25 on
the stock. Separate regressions by sex, with script 61's step terms, are
fitted at 22-25 for the stock and hires. Poisson pseudo-maximum
likelihood, standard errors clustered by employer.

INPUTS AND OUTPUTS
Pulls, in MONA, the monthly employer declarations for 2021 to 2025 joined
to Individ_2023, 2021 and 2019 for birth year and sex (Kon, a character
column holding '1' for men and '2' for women), as counts and as consecutive
month flows, cached as L_counts_sex_YYYY.parquet and flows_sex_YYYY.parquet.
Reads script 47h's caches for the exposure. Writes to output_67/:
gender_interaction.csv, gender_by_sex.csv and 67_summary.txt.

IN THE PAPER
The sex panel (q_counts_sex, build_skeleton_sex) is the one script 68 fits
with the calendar cycle removed, which gives the female differential of
Table 1 and Section 3 (young women minus young men, and the estimate for
young men beside it), and the one script 76 restricts to each education
track. The coefficients in gender_interaction.csv and gender_by_sex.csv,
estimated without the calendar terms, are not quoted.
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
OUT = HERE / "output_67"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
YEARS = list(range(2021, 2026))          # 61's panel window
SEX = {"1": "men", "2": "women"}         # SCB coding, verified in script 48
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AGE_CASE = """CASE
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 22 AND 25 THEN '22-25'
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 26 AND 30 THEN '26-30'
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 31 AND 34 THEN '31-34'
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 35 AND 40 THEN '35-40'
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 41 AND 49 THEN '41-49'
        WHEN {y} - TRY_CAST(i.FodelseAr AS INT) BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


def q_counts_sex(year: int, conn) -> pd.DataFrame:
    """
    47L's counts query with sex added to the SELECT and the GROUP BY.

    Kon is a character column, so it arrives as "1" and "2" rather than as
    integers, and is normalised on the way in.
    """
    suffix, max_month = ("_def", 12) if year < 2025 else ("_prel", 6)
    monthly = "\nUNION ALL\n".join(f"""
        SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
               agi.PERIOD AS period, agi.P1207_LOPNR_PERSONNR AS person_id,
               COALESCE(TRY_CAST(a.FodelseAr AS INT), TRY_CAST(b.FodelseAr AS INT),
                        TRY_CAST(c.FodelseAr AS INT)) AS fodelse,
               COALESCE(a.Kon, b.Kon, c.Kon) AS gender
        FROM dbo.Arb_AGIIndivid{year}{m:02d}{suffix} agi
        LEFT JOIN dbo.Individ_2023 a ON agi.P1207_LOPNR_PERSONNR = a.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2021 b ON agi.P1207_LOPNR_PERSONNR = b.P1207_LopNr_PersonNr
        LEFT JOIN dbo.Individ_2019 c ON agi.P1207_LOPNR_PERSONNR = c.P1207_LopNr_PersonNr
        """ for m in range(1, max_month + 1))
    age_case = AGE_CASE.format(y=year).replace("i.FodelseAr", "CAST(fodelse AS VARCHAR)")
    q = f"""
    WITH base AS ({monthly})
    SELECT employer_id,
           CONCAT(LEFT(period, 4), '-', RIGHT(period, 2)) AS year_month,
           {age_case} AS age_group,
           LTRIM(RTRIM(gender)) AS gender,
           COUNT(DISTINCT person_id) AS n_emp
    FROM base
    WHERE fodelse IS NOT NULL AND gender IN ('1', '2')
      AND {year} - fodelse BETWEEN 22 AND 69
    GROUP BY employer_id, period, {age_case}, LTRIM(RTRIM(gender))
    """
    return pd.read_sql(q, conn)


def q_flows_sex(year: int, conn) -> pd.DataFrame:
    """54's flow query, one month at a time, with sex carried through."""
    s54 = _mod("54_hiring_flows.py", "s54")
    max_month = 12 if year < 2025 else 6
    out = []
    for m in range(1, max_month + 1):
        py, pm = s54._prev(year, m)
        age_case = """CASE
             WHEN age BETWEEN 22 AND 25 THEN '22-25'
             WHEN age BETWEEN 26 AND 30 THEN '26-30'
             WHEN age BETWEEN 31 AND 34 THEN '31-34'
             WHEN age BETWEEN 35 AND 40 THEN '35-40'
             WHEN age BETWEEN 41 AND 49 THEN '41-49'
             WHEN age BETWEEN 50 AND 69 THEN '50+'
             ELSE NULL END"""
        q = f"""
        WITH cur AS (
            SELECT DISTINCT P1207_LOPNR_PEORGNR AS emp,
                            P1207_LOPNR_PERSONNR AS per
            FROM {s54._tbl(year, m)}),
        prv AS (
            SELECT DISTINCT P1207_LOPNR_PEORGNR AS emp,
                            P1207_LOPNR_PERSONNR AS per
            FROM {s54._tbl(py, pm)}),
        flow AS (
            SELECT COALESCE(cur.emp, prv.emp) AS employer_id,
                   COALESCE(cur.per, prv.per) AS person_id,
                   CASE WHEN prv.per IS NULL THEN 1 ELSE 0 END AS is_hire,
                   CASE WHEN cur.per IS NULL THEN 1 ELSE 0 END AS is_sep
            FROM cur
            FULL OUTER JOIN prv
              ON cur.per = prv.per AND cur.emp = prv.emp
            WHERE cur.per IS NULL OR prv.per IS NULL),
        aged AS (
            SELECT f.employer_id, f.is_hire, f.is_sep,
                   LTRIM(RTRIM(COALESCE(a.Kon, b.Kon, c.Kon))) AS gender,
                   {year} - COALESCE(TRY_CAST(a.FodelseAr AS INT),
                                     TRY_CAST(b.FodelseAr AS INT),
                                     TRY_CAST(c.FodelseAr AS INT)) AS age
            FROM flow f
            LEFT JOIN dbo.Individ_2023 a ON f.person_id = a.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_2021 b ON f.person_id = b.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_2019 c ON f.person_id = c.P1207_LopNr_PersonNr)
        SELECT employer_id,
               {age_case} AS age_group, gender,
               SUM(CAST(is_hire AS INT)) AS n_hire,
               SUM(CAST(is_sep  AS INT)) AS n_sep
        FROM aged
        WHERE age BETWEEN 22 AND 69 AND gender IN ('1', '2')
        GROUP BY employer_id, {age_case}, gender
        """
        d = pd.read_sql(q, conn)
        d["year_month"] = f"{year}-{m:02d}"
        out.append(d)
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=["employer_id", "age_group", "gender",
                                       "n_hire", "n_sep", "year_month"]))


def build_skeleton_sex(counts: pd.DataFrame, young: str, j47,
                       value: str) -> pd.DataFrame:
    """
    61's skeleton with sex as a fourth dimension of the cell.

    The fixed effects move with it: employer by month as before, but
    employer by age-and-sex and month by age-and-sex, so that the
    national path of young women is absorbed exactly as the national path
    of the young is in the headline. Anything less and the gender
    interaction would be picking up aggregate female employment.
    """
    bands = [young] + j47.INCUMBENT_BANDS
    p = counts[counts["age_group"].astype(str).isin(bands)].copy()
    p["gender"] = p["gender"].astype(str).str.strip()
    p = p[p["gender"].isin(SEX)]
    p["year_month"] = p["year_month"].astype(str)
    p["age_group"] = p["age_group"].astype(str)
    p = (p.groupby(["employer_id", "age_group", "gender", "year_month"],
                   observed=True)[value].sum().reset_index())
    have = p.groupby("employer_id")["age_group"].agg(set)
    keep = have[have.apply(lambda v: young in v
                           and bool(v & set(j47.INCUMBENT_BANDS)))].index
    p = p[p["employer_id"].isin(keep)]
    if p.empty:
        return p
    months = sorted(p["year_month"].unique())
    emp = p["employer_id"].drop_duplicates().to_numpy()
    full = pd.MultiIndex.from_product(
        [emp, bands, sorted(SEX), months],
        names=["employer_id", "age_group", "gender", "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "gender", "year_month"],
                     observed=True)[value].sum()
           .reindex(full, fill_value=0).reset_index())
    bal = bal.rename(columns={value: "n_emp"})
    bal["n_emp"] = bal["n_emp"].astype(int)
    # dead cells, now defined on employer x age x sex
    alive = bal.groupby(["employer_id", "age_group", "gender"],
                        observed=True)["n_emp"].transform("sum") > 0
    bal = bal[alive]
    nb = bal.groupby("employer_id", observed=True)["age_group"].transform("nunique")
    bal = bal[nb >= 2]
    if bal.empty:
        return bal
    bal["young"] = (bal["age_group"] == young).astype(int)
    bal["female"] = (bal["gender"] == "2").astype(int)
    ec = pd.factorize(bal["employer_id"], sort=False)[0].astype("int64")
    tc = pd.factorize(bal["year_month"], sort=False)[0].astype("int64")
    ac = pd.factorize(bal["age_group"].astype(str) + "_" + bal["gender"],
                      sort=False)[0].astype("int64")
    n_t, n_a = int(tc.max()) + 1, int(ac.max()) + 1
    bal["fe_emp_t"] = ec * n_t + tc
    bal["fe_emp_age"] = ec * n_a + ac
    bal["fe_t_age"] = tc * n_a + ac
    return bal


def add_gender_terms(bal: pd.DataFrame, post_from: str) -> tuple:
    """
    The headline interaction, the female differential, and the two lower
    order terms that are not absorbed.

    post x high alone is constant within an employer and month, so the
    employer-by-month effect takes it and it must NOT be listed. The
    other three vary across cells within an employer and month and are
    identified.
    """
    ym = bal["year_month"].astype(str)
    post = (ym >= post_from).astype(int)
    rb = (ym >= mc.RIKSBANK_YM).astype(int)
    h, y, f = bal["high"], bal["young"], bal["female"]
    bal["post_x_high_x_young"] = post * h * y
    bal["post_x_high_x_female"] = post * h * f
    bal["post_x_high_x_young_x_female"] = post * h * y * f
    bal["rb_x_high_x_young"] = rb * h * y
    bal["rb_x_high_x_female"] = rb * h * f
    bal["rb_x_high_x_young_x_female"] = rb * h * y * f
    return bal, ["rb_x_high_x_young", "rb_x_high_x_female",
                 "rb_x_high_x_young_x_female",
                 "post_x_high_x_young", "post_x_high_x_female",
                 "post_x_high_x_young_x_female"]


def main():
    mc.Tee(OUT / "67_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("67: THE GENDER SPLIT, ON THE DESIGN THAT SURVIVES")
    print("=" * 70)
    print("  The claim in the paper rests on script 48, which splits the")
    print("  occupation design the backtest closed. This re-estimates it on")
    print("  61's design and TESTS the difference rather than asserting it.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
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
    expos = {}
    for arm in ("true", "asof"):
        e, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, arm,
                                      s61.TRUNC)
        expos[arm] = e
        print(f"  exposure {arm}: {len(e):,} firms")
    del frame19
    gc.collect()

    conn = None
    def _conn():
        nonlocal conn
        if conn is None:
            conn = mc.connect()
        return conn

    stock, flows = [], []
    for y in YEARS:
        cf = CACHE / f"L_counts_sex_{y}.parquet"
        c = mc.read_cache(cf, require=["employer_id", "year_month",
                                       "age_group", "gender", "n_emp"])
        if c is None:
            t = time.time()
            c = q_counts_sex(y, _conn())
            mc.write_cache(c, cf)
            print(f"  counts by sex {y}: {len(c):,} cells "
                  f"({(time.time()-t)/60:.1f} min)")
        else:
            print(f"  counts by sex {y}: cached ({len(c):,} cells)")
        stock.append(c)
        ff = CACHE / f"flows_sex_{y}.parquet"
        f = mc.read_cache(ff, require=["employer_id", "year_month",
                                       "age_group", "gender", "n_hire",
                                       "n_sep"])
        if f is None:
            t = time.time()
            f = opt(f"flows by sex {y}", q_flows_sex, y, _conn())
            if f is not None:
                mc.write_cache(f, ff)
                print(f"  flows by sex {y}: {len(f):,} cells "
                      f"({(time.time()-t)/60:.1f} min)")
        else:
            print(f"  flows by sex {y}: cached ({len(f):,} cells)")
        if f is not None:
            flows.append(f)
    stock = pd.concat(stock, ignore_index=True)
    flows = pd.concat(flows, ignore_index=True) if flows else None
    gc.collect()

    SOURCES = [("stock", stock, "n_emp")]
    if flows is not None:
        SOURCES += [("hires", flows, "n_hire"), ("seps", flows, "n_sep")]
    inter_rows, sex_rows = [], []

    for band in j47.YOUNG_BANDS:
        for label, src, value in SOURCES:
            t1 = time.time()
            skel = build_skeleton_sex(src, band, j47, value)
            if skel.empty:
                print(f"  {band} {label}: empty panel, skipped")
                continue
            print(f"\n  {band} {label}: skeleton {len(skel):,} rows "
                  f"({(time.time()-t1)/60:.1f} min)")
            for arm in ("true", "asof"):
                # the as-of arm only where the claim lives, to keep this
                # to one run rather than a grid
                if arm == "asof" and not (band == "22-25"
                                          and label == "stock"):
                    continue
                b = skel.merge(expos[arm][["employer_id", "fq"]],
                               on="employer_id", how="inner")
                if b.empty:
                    continue
                b["high"] = (b["fq"] == 4).astype(int)
                for dname, dfrom in (("launch", mc.CHATGPT_YM),
                                     ("adoption", "2024-01")):
                    b, terms = add_gender_terms(b, dfrom)
                    tag = f"g67_{band.replace('-','_')}_{label}_{arm}_{dname}"
                    r = mc.run_fepois_multi(b, OUT, tag=tag, terms=terms,
                                            fes=j47.FES)
                    if r.empty:
                        FAILURES.append(tag)
                        print(f"    {arm} {dname}: FAILED, recorded")
                        continue
                    g = r.set_index("term")
                    for t_ in ("post_x_high_x_young",
                               "post_x_high_x_young_x_female"):
                        if t_ not in g.index:
                            continue
                        inter_rows.append(
                            {"young_band": band, "outcome": label, "arm": arm,
                             "dating": dname, "term": t_,
                             "coef": float(g.loc[t_, "coef"]),
                             "se": float(g.loc[t_, "se"]),
                             "n_obs": int(g.loc[t_, "n_obs"]),
                             "status": str(g.loc[t_].get("status", "ok"))})
                    pd.DataFrame(inter_rows).to_csv(
                        OUT / "gender_interaction.csv", index=False)
                    m = g.loc["post_x_high_x_young"]
                    d = (g.loc["post_x_high_x_young_x_female"]
                         if "post_x_high_x_young_x_female" in g.index else None)
                    print(f"    {arm:<5} {dname:<9} men "
                          f"{float(m['coef']):+.4f} ({float(m['se']):.4f})"
                          + (f"   female differential "
                             f"{float(d['coef']):+.4f} ({float(d['se']):.4f}) "
                             f"t {float(d['coef'])/max(float(d['se']),1e-12):+.2f}"
                             if d is not None else ""))

                # the separate per-sex fits, where the claim lives, because
                # a reader wants the two numbers and not only their gap
                if arm == "true" and band == "22-25" and label in ("stock",
                                                                   "hires"):
                    for code, name in SEX.items():
                        sub = b[b["gender"] == code].copy()
                        if sub.empty:
                            continue
                        sub, st, pt = s61.add_terms(
                            sub.assign(high=sub["high"], young=sub["young"]))
                        rr = mc.run_fepois_multi(
                            sub, OUT,
                            tag=f"g67_sex_{name}_{label}", terms=pt,
                            fes=j47.FES)
                        del sub
                        gc.collect()
                        if rr.empty:
                            FAILURES.append(f"per-sex/{name}/{label}")
                            continue
                        x = rr.set_index("term").loc[
                            "post2024_x_high_x_young"]
                        sex_rows.append({"young_band": band,
                                         "outcome": label, "sex": name,
                                         "coef": float(x["coef"]),
                                         "se": float(x["se"]),
                                         "n_obs": int(x["n_obs"]),
                                         "status": str(x.get("status", "ok"))})
                        pd.DataFrame(sex_rows).to_csv(
                            OUT / "gender_by_sex.csv", index=False)
                        print(f"    per-sex {name:<6} {label:<6} "
                              f"{float(x['coef']):+.4f} "
                              f"({float(x['se']):.4f})")
                del b
                gc.collect()
            del skel
            gc.collect()

    lines = ["THE GENDER SPLIT, ON THE DESIGN THAT SURVIVES", "=" * 52, "",
             "Within-employer, exposure from the 2019 education mix of",
             "incumbents aged 31 and over, young against their own older",
             "colleagues, with sex as a fourth dimension of the cell and the",
             "fixed effects moved to employer-by-age-and-sex and",
             "month-by-age-and-sex.", "",
             "The FEMALE DIFFERENTIAL is the coefficient on",
             "post x high x young x female. It is the gender difference and",
             "its t is the test. The per-sex rows are reported because a",
             "reader wants them, but the claim rests on the differential.", ""]
    I = pd.DataFrame(inter_rows)
    if not I.empty:
        for dname in ("launch", "adoption"):
            d = I[(I.dating == dname) & (I.arm == "true")]
            if d.empty:
                continue
            lines.append(f"Dated {dname}:")
            for (band, oc), grp in d.groupby(["young_band", "outcome"]):
                m = grp[grp.term == "post_x_high_x_young"]
                f = grp[grp.term == "post_x_high_x_young_x_female"]
                if m.empty:
                    continue
                s = (f"  {band:<6} {oc:<6} men {float(m['coef'].iloc[0]):+.4f} "
                     f"({float(m['se'].iloc[0]):.4f})")
                if not f.empty:
                    fc, fs = float(f['coef'].iloc[0]), float(f['se'].iloc[0])
                    s += (f"   female extra {fc:+.4f} ({fs:.4f}) t "
                          f"{fc/max(fs,1e-12):+.2f}   women "
                          f"{float(m['coef'].iloc[0]) + fc:+.4f}")
                lines.append(s)
            lines.append("")
        a = I[(I.arm == "asof") & (I.dating == "adoption")]
        t = I[(I.arm == "true") & (I.dating == "adoption")
              & (I.young_band == "22-25") & (I.outcome == "stock")]
        if not a.empty and not t.empty:
            for term in ("post_x_high_x_young", "post_x_high_x_young_x_female"):
                aa = a[(a.term == term) & (a.young_band == "22-25")
                       & (a.outcome == "stock")]
                tt = t[t.term == term]
                if not aa.empty and not tt.empty:
                    lines.append(f"ARTEFACT on {term}: "
                                 f"{float(aa['coef'].iloc[0]) - float(tt['coef'].iloc[0]):+.4f}")
            lines.append("")
    S = pd.DataFrame(sex_rows)
    if not S.empty:
        lines += ["Separate regressions by sex, 22-25, pooled from 2024-01:"]
        for _, r in S.iterrows():
            lines.append(f"  {r['sex']:<6} {r['outcome']:<6} {r['coef']:+.4f} "
                         f"({r['se']:.4f}) t "
                         f"{r['coef']/max(r['se'],1e-12):+.2f}")
        lines.append("")
    if FAILURES:
        lines += ["FITS THAT FAILED: " + "; ".join(FAILURES), ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. The sentence 'concentrated among young women' requires the",
        "     female differential to be negative AND beyond two standard",
        "     errors. Two separate per-sex coefficients of different size do",
        "     not establish it, which is what the paper currently does.",
        "  2. If the differential is not significant, the honest sentence is",
        "     that the decline is present for both sexes and we cannot",
        "     distinguish their magnitudes. That is a perfectly publishable",
        "     result and it is what script 48 should have been asked.",
        "  3. Read the three margins together. A female result concentrated",
        "     in hiring means something different from one in separations.",
        "  4. The artefact is measured at the cell the claim is about. The",
        "     threshold is the project's standing 0.05.",
        "  5. Kon is a CHAR column and arrives as '1' and '2'. Script 48",
        "     lost a day to that. It is normalised here on the way in.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "67_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n67 done.")


if __name__ == "__main__":
    main()

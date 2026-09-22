#!/usr/bin/env python3
"""
54_hiring_flows.py: hires and separations by employer, age band and month.

QUESTION
Through which margin does the employment of young workers adjust: fewer
hires, or more separations? The stock is the slowest object in the
Swedish labour market, since notice periods and seniority rules stand
between a change in demand and a change in headcount, and the claim that
entry-level work is affected is a claim about flows. This script builds
the monthly flows from the employer declarations, with no occupation or
education record, and estimates the same design as script 47L on them.

DESIGN
A hire in month t is an employer-person pair present in t and absent from
that employer in t minus 1; a separation is a pair present in t minus 1
and absent in t, counted in t. Both come from one full outer join of
consecutive months, one query per month, so the two flows are computed
from the same comparison. A move to another employer or a month off the
payroll counts as a separation; the declarations record no reason for the
exit. January 2019 has no predecessor and is dropped. Age is read from the
birth year in the Individ register (2023, 2021 or 2019 vintage, whichever
holds the person).

Panel (build_panel): employer by age band by month, January 2019 to June
2025, balanced and zero-filled, so that a cell that hires in some months
and not others carries its zeros; a variant drops every January, the
month of year-end contract turnover. Exposure is script 47L's E(f, a),
imported rather than rebuilt, standardised across cells. Terms: PostRB x E
and PostGPT x E, and in the gradient fit one PostGPT x E per age band.
Fixed effects: employer by month, employer by age, month by age. Poisson
pseudo-maximum likelihood, standard errors clustered by employer, one fit
per outcome.

INPUTS AND OUTPUTS
Reads, in MONA, consecutive pairs of Arb_AGIIndivid monthly tables joined
to Individ_2023, 2021 and 2019; the L_baseline_2019 cache and
daioe_quartiles.dta through script 47L. Caches flows_YYYY.parquet
(employer, month, age band, n_hire, n_sep). Writes to output_54/:
flow_estimates.csv, flow_gradient.csv, flow_support.csv and
54_summary.txt.

IN THE PAPER
Section 2 defines a hire and a separation as above. The flows_YYYY caches
are the outcome behind the hires and separations rows of Table 1 and
Online Appendix Table III.2, Panel A (fitted by scripts 68 and 75 on the
headline classification), and behind the hiring and separation rows of
Panel B (fitted by script 63 on the continuous route). The estimates in
flow_estimates.csv, dated at the launch, are not quoted.
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
OUT = HERE / "output_54"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
CACHE.mkdir(exist_ok=True)

YEARS = list(range(2019, 2026))
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
OUTCOMES = ("n_hire", "n_sep")
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")
FLOW_COLS = ["employer_id", "year_month", "age_group", "n_hire", "n_sep"]


def opt(label: str, fn, *a, **kw):
    """Stata's `capture noisily`: for the inessential only. Never an
    estimate, never a primary export."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _l47():
    """
    Import 47L for build_exposure and q_baseline, so the exposure measure is
    the SAME OBJECT in both scripts rather than a second implementation that
    can drift. Behind a function so an import failure lands in this script's
    log rather than in a stderr BatchClient discards.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "l47", HERE / "47L_age_baseline_exposure.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tbl(year: int, month: int) -> str:
    """AGI monthly table name. 2025 is preliminary and stops at June."""
    suffix = "_def" if year < 2025 else "_prel"
    return f"dbo.Arb_AGIIndivid{year}{month:02d}{suffix}"


def _prev(year: int, month: int):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def q_flows(year: int, conn) -> pd.DataFrame:
    """
    Hires and separations per employer x age band x month.

    A hire in month t is an (employer, person) pair present in t and absent
    in t-1; a separation is present in t-1 and absent in t, counted at t.
    Both come from one FULL OUTER JOIN of consecutive months, so the two
    flows are computed from the same comparison and cannot disagree about
    who was where.

    ONE QUERY PER MONTH, not twelve unioned. Nothing of this shape has run
    on this data before: the join is between two five-million-row monthly
    tables rather than between a monthly table and a register. Twelve of
    those in a single statement give the optimiser one enormous plan, and a
    failure anywhere in it loses the year and names no month. Per month,
    each query is small and predictable, the result is already aggregated
    to a few hundred thousand rows, and a failure says which month.

    The comparison crosses the year boundary (January looks at the previous
    December, in the previous year's table) and the preliminary/definitive
    suffix, so the table name is resolved per month, never per year.
    """
    max_month = 12 if year < 2025 else 6
    out = []
    for m in range(1, max_month + 1):
        if year == 2019 and m == 1:
            continue                      # no predecessor exists at all
        py, pm = _prev(year, m)
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
            FROM {_tbl(year, m)}),
        prv AS (
            SELECT DISTINCT P1207_LOPNR_PEORGNR AS emp,
                            P1207_LOPNR_PERSONNR AS per
            FROM {_tbl(py, pm)}),
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
                   {year} - COALESCE(TRY_CAST(a.FodelseAr AS INT),
                                     TRY_CAST(b.FodelseAr AS INT),
                                     TRY_CAST(c.FodelseAr AS INT)) AS age
            FROM flow f
            LEFT JOIN dbo.Individ_2023 a
                   ON f.person_id = a.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_2021 b
                   ON f.person_id = b.P1207_LopNr_PersonNr
            LEFT JOIN dbo.Individ_2019 c
                   ON f.person_id = c.P1207_LopNr_PersonNr)
        SELECT employer_id,
               {age_case} AS age_group,
               SUM(CAST(is_hire AS INT)) AS n_hire,
               SUM(CAST(is_sep  AS INT)) AS n_sep
        FROM aged
        WHERE age BETWEEN 22 AND 69 AND employer_id IS NOT NULL
        GROUP BY employer_id, {age_case}
        """
        d = pd.read_sql(q, conn)
        d["year_month"] = f"{year}-{m:02d}"
        out.append(d)
    if not out:
        return pd.DataFrame(columns=FLOW_COLS)
    df = pd.concat(out, ignore_index=True)
    df["year_month"] = df["year_month"].astype(str)
    df["age_group"] = df["age_group"].astype(str)
    for c in ("n_hire", "n_sep"):
        df[c] = df[c].fillna(0).astype("int32")
    return df[FLOW_COLS]


def age_term(age: str) -> str:
    return "gpt_x_expo_" + age.replace("-", "_").replace("+", "p")


TERMS = ["post_rb_x_expo", "post_gpt_x_expo"]
TERMS_GRAD = ["post_rb_x_expo"] + [age_term(a) for a in AGES]


def build_panel(flows: pd.DataFrame, expo: pd.DataFrame,
                drop_january: bool = False) -> pd.DataFrame:
    """
    Balanced employer x age x month panel of the flow, zero-filled.

    Zero-filling is not cosmetic here: a firm-age cell that hired in some
    months and not others must carry the zeros, or the estimate is
    conditional on hiring and answers a different question.
    """
    p = flows.merge(expo[["employer_id", "age_group", "expo"]],
                    on=["employer_id", "age_group"], how="inner")
    if p.empty:
        return p
    if drop_january:
        p = p[~p["year_month"].str.endswith("-01")]
    months = sorted(p["year_month"].unique())
    cells = p[["employer_id", "age_group", "expo"]].drop_duplicates(
        ["employer_id", "age_group"])
    full = pd.MultiIndex.from_arrays(
        [np.repeat(cells["employer_id"].to_numpy(), len(months)),
         np.repeat(cells["age_group"].to_numpy(), len(months)),
         np.tile(np.array(months), len(cells))],
        names=["employer_id", "age_group", "year_month"])
    bal = (p.groupby(["employer_id", "age_group", "year_month"],
                     observed=True)[["n_hire", "n_sep"]].sum()
           .reindex(full, fill_value=0).reset_index()
           .merge(cells, on=["employer_id", "age_group"], how="left"))
    for c in ("n_hire", "n_sep"):
        bal[c] = bal[c].astype(int)
    mu, sd = cells["expo"].mean(), cells["expo"].std(ddof=0)
    bal["expo_z"] = (bal["expo"] - mu) / (sd if sd > 0 else 1.0)
    bal["post_rb"] = (bal["year_month"] >= mc.RIKSBANK_YM).astype(int)
    bal["post_gpt"] = (bal["year_month"] >= mc.CHATGPT_YM).astype(int)
    bal["post_rb_x_expo"] = bal["post_rb"] * bal["expo_z"]
    bal["post_gpt_x_expo"] = bal["post_gpt"] * bal["expo_z"]
    for a in AGES:
        bal[age_term(a)] = (bal["post_gpt"] * bal["expo_z"]
                            * (bal["age_group"] == a).astype(int))
    e = bal["employer_id"].astype(str)
    bal["fe_emp_t"] = e + "_" + bal["year_month"]
    bal["fe_emp_age"] = e + "_" + bal["age_group"]
    bal["fe_t_age"] = bal["year_month"] + "_" + bal["age_group"]
    return bal


def fit(bal: pd.DataFrame, outcome: str, tag: str, terms) -> pd.DataFrame:
    """
    Poisson on one flow. mona_common's runner expects the count in `n_emp`,
    so the chosen outcome is aliased into it rather than the runner being
    given a second name to know about.
    """
    if bal.empty:
        return pd.DataFrame()
    b = bal.copy()
    b["n_emp"] = b[outcome]
    return mc.run_fepois_multi(b, OUT, tag=tag, terms=terms, fes=FES)


def main():
    mc.Tee(OUT / "54_log.txt")
    t_start = time.time()
    print("=" * 70)
    print("54: HIRING AND SEPARATION FLOWS ON 2019-FROZEN EXPOSURE")
    print("=" * 70)
    print("  the fast margin. No occupation code after 2019, no education")
    print("  register, no register-lag exposure in the outcome at all.")
    print(mc.mem_line("  "))

    l47 = _l47()
    conn = None

    # --- exposure: 47L's, not a second implementation -------------------
    cf = CACHE / "L_baseline_2019.parquet"
    base = mc.read_cache(cf)
    if base is None:
        conn = conn or mc.connect()
        t0 = time.time()
        base = l47.q_baseline(conn)
        mc.write_cache(base, cf)
        print(f"  baseline 2019: {len(base):,} rows ({time.time()-t0:.0f}s)")
    else:
        print(f"  baseline 2019: cached ({len(base):,} rows)")
    # the same score frame 47L builds: the genAI percentile, not the
    # quartile, because build_exposure averages a continuous score
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]
    expo = l47.build_exposure(base, daioe)
    print(f"  exposure: {len(expo):,} firm-age cells, "
          f"{expo['employer_id'].nunique():,} firms")

    # --- flows ----------------------------------------------------------
    frames = []
    for y in YEARS:
        cf = CACHE / f"flows_{y}.parquet"
        f = mc.read_cache(cf, require=FLOW_COLS)
        if f is None:
            conn = conn or mc.connect()
            t0 = time.time()
            f = q_flows(y, conn)
            mc.write_cache(f, cf)
            print(f"  flows {y}: {len(f):,} cells ({time.time()-t0:.0f}s)")
        else:
            print(f"  flows {y}: cached ({len(f):,} cells)")
        frames.append(f)
    flows = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    opt("flow_support.csv", lambda: mc.enforce_min_cell(
        flows.groupby("age_group", observed=True).agg(
            cells=("employer_id", "size"),
            firms=("employer_id", "nunique"),
            mean_hire=("n_hire", "mean"),
            mean_sep=("n_sep", "mean")).reset_index(),
        count_col="firms").to_csv(OUT / "flow_support.csv", index=False))

    # --- estimation -----------------------------------------------------
    rows, grad = [], []
    for drop_jan in (False, True):
        variant = "no_january" if drop_jan else "all_months"
        bal = build_panel(flows, expo, drop_january=drop_jan)
        if bal.empty:
            print(f"  {variant}: EMPTY panel, skipped")
            continue
        print(f"\n  {variant}: {len(bal):,} cells, "
              f"{bal['employer_id'].nunique():,} firms, "
              f"hire zeros {(bal['n_hire'] == 0).mean():.1%}, "
              f"sep zeros {(bal['n_sep'] == 0).mean():.1%}")
        for outcome in OUTCOMES:
            r = fit(bal, outcome, f"h54_{outcome}_{variant}", TERMS)
            if r.empty or not (r["term"] == "post_gpt_x_expo").any():
                raise RuntimeError(
                    f"the pooled Poisson returned nothing for "
                    f"{outcome}/{variant}; this is a primary estimate")
            g = r[r["term"] == "post_gpt_x_expo"].iloc[0]
            rows.append({"outcome": outcome, "variant": variant,
                         "gamma": float(g["coef"]), "se": float(g["se"]),
                         "n_obs": int(g["n_obs"]),
                         "status": str(g.get("status", "ok"))})
            pd.DataFrame(rows).to_csv(OUT / "flow_estimates.csv", index=False)
            print(f"  [{variant:<10} {outcome}] PostGPT x exposure "
                  f"{g['coef']:+.4f} (SE {g['se']:.4f}) n {int(g['n_obs']):,}")
            if not drop_jan:
                gr = opt(f"gradient {outcome}", fit, bal, outcome,
                         f"g54_{outcome}", TERMS_GRAD)
                if gr is not None and not gr.empty:
                    want = {age_term(a): a for a in AGES}
                    gr = gr[gr["term"].isin(want)].copy()
                    gr["age_group"] = gr["term"].map(want)
                    gr["outcome"] = outcome
                    grad.append(gr)
                    pd.concat(grad, ignore_index=True).to_csv(
                        OUT / "flow_gradient.csv", index=False)
                    print(f"    age gradient, {outcome}:")
                    for _, x in gr.iterrows():
                        print(f"      {x['age_group']:<5} {x['coef']:+.4f} "
                              f"(SE {x['se']:.4f}) {x.get('status','ok')}")
        del bal
        gc.collect()

    # --- summary --------------------------------------------------------
    est = pd.DataFrame(rows)
    lines = ["HIRING AND SEPARATION FLOWS, 2019-FROZEN EXPOSURE",
             "=" * 58,
             "PostGPT x E(f,a). Absorbed: employer x month, employer x age,",
             "month x age. The outcome uses no occupation code after 2019.",
             "", est.to_string(index=False), ""]
    if grad:
        g = pd.concat(grad, ignore_index=True)
        piv = g.pivot_table(index="age_group", columns="outcome",
                            values="coef").reindex(
                                [a for a in AGES if a in set(g["age_group"])])
        lines += ["age gradient (all months):", piv.round(4).to_string(), ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. A hire is an employer-person spell absent last month. That",
        "     includes returns from parental and sick leave and seasonal",
        "     re-hires. It is a START, not a labour-market entry.",
        "  2. The first month of the window has no predecessor and is",
        "     dropped. January carries year-end contract churn, which is",
        "     what the no_january variant is for; if the two variants",
        "     disagree, quote both.",
        "  3. Exposure is frozen in 2019 and is therefore a STALE proxy for",
        "     who is exposed in 2025. Stale proxies attenuate toward zero,",
        "     so a null here bounds the effect of BASELINE exposure, not of",
        "     current exposure.",
        "  4. Hires and separations move together in a growing firm. Read",
        "     the two outcomes side by side: a fall in hiring with no rise",
        "     in separations is an adjustment through the inflow, which is",
        "     the entry-level claim; both moving is a scale effect.",
        "  5. This shares the DAIOE measure with every other design in this",
        "     round. It is independent of them in its outcome and its",
        "     register dependence, NOT in its exposure measure.",
        "", f"Total runtime {(time.time()-t_start)/60:.1f} min. "
        + mc.mem_line()]
    (OUT / "54_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n54 done.")


if __name__ == "__main__":
    main()

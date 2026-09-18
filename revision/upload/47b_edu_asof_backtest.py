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
AGE_CASE = """CASE
        WHEN age BETWEEN 22 AND 25 THEN '22-25'
        WHEN age BETWEEN 26 AND 30 THEN '26-30'
        WHEN age BETWEEN 31 AND 34 THEN '31-34'
        WHEN age BETWEEN 35 AND 40 THEN '35-40'
        WHEN age BETWEEN 41 AND 49 THEN '41-49'
        WHEN age BETWEEN 50 AND 69 THEN '50+'
        ELSE NULL END"""


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
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s47", HERE / "47_edu_exposure.py")
    s47 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s47)          # reuse 47's key, weights, binning

    mc.Tee(OUT / "47b_log.txt")
    print("=" * 70)
    print("47b: AS-OF BACKTEST ON THE EDUCATION-BASED DESIGN")
    print("=" * 70)
    print(mc.mem_line("  "))

    key = s47.load_key()
    wcache = mc.CACHE_DIR / "edu_weights_2019.parquet"
    counts = mc.read_cache(wcache)
    if counts is None:
        counts = s47.pull_weight_counts(2019, mc.connect())
        counts.to_parquet(wcache, index=False)
    grp_q = s47.build_weights(counts, key, 2019)
    print(f"  weights: {len(grp_q)} groups, quartiles fixed at 2019")

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
            agg, mrate = s47.map_and_collapse(raw, grp_q)
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

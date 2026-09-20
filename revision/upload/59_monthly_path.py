#!/usr/bin/env python3
"""
59_monthly_path.py -- the path at monthly resolution, where it matters.

======================================================================
  RUNS IN MONA. No SQL. Reads the caches 54 and 47L already wrote.
  Writes output_59/.
======================================================================

WHY.

Everything so far is half-yearly, and that is why 2025 is a single point
with a standard error three to four times its neighbours. A half-year is
also the wrong unit for the question actually being asked. If something
turned at the end of the window we want to know whether it built over
months or appeared in one, and a single coefficient covering January to
June cannot tell us.

THE OBSTACLE, and it is why this is not simply "the same thing, monthly".
A month-by-month event study over 2019 to 2025 needs 77 interaction terms,
doubled to 154 once the age contrast is included, on a panel of eleven
million rows. That design matrix is about thirteen gigabytes. R segfaulted
on this data at 24 terms when two lanes competed for memory, so 154 is not
a reasonable request.

TWO SPECIFICATIONS THAT FIT, each answering a different half of it.

  QUARTER   every quarter from 2019 to 2025H1, 25 terms doubled to 50.
            Four times the resolution of the half-yearly path, and 2025
            contributes Q1 and Q2 separately rather than one blurred
            point. Each quarter is rebased on the mean of the SAME
            quarter in the pre-period, so Q1 2025 is read against Q1 of
            2019, 2020 and 2021 rather than against a mid-2022 base.
            This is the specification to quote.

  ZOOM      one term per MONTH from January 2024, and one per half-year
            before that: 23 terms doubled to 46. Full monthly resolution
            across the eighteen months where something may have happened,
            at a fraction of the cost.
            Read this one for SHAPE, not for level. Its monthly points
            have no same-month pre-period counterpart to be rebased
            against, so a level comparison across the 2024 boundary is
            contaminated by calendar seasonality. Whether the series
            drifts down through 2025 or jumps once is visible regardless,
            and that is what it is for.

A DESCRIPTIVE SERIES IS EXPORTED BESIDE THEM. Mean hires per firm-age cell
by month, split by whether the cell's frozen 2019 exposure is in the top
quartile, for the young and for everyone else. No fixed effects, no
inference, just the four series. If the event study says something the
plotted series cannot show, one of the two is wrong, and it is usually not
the series.

WHAT THIS DOES NOT FIX. The precision problem is arithmetic. Splitting a
half-year into quarters or months buys resolution by spending observations
per coefficient, so individual points will be noisier, not less noisy. The
gain is in seeing a shape across several consecutive points, which is
evidence a single imprecise coefficient cannot provide. Do not read one
significant month as a finding.

Output (output_59/):
  path_quarter.csv     quarterly event study, raw and rebased, with SEs
  path_zoom.csv        monthly from 2024, half-yearly before
  descriptive.csv      mean flow per cell by month, exposure and age split
  59_summary.txt
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
OUT = HERE / "output_59"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

YOUNG = "22-25"
FES = ("fe_emp_t", "fe_emp_age", "fe_t_age")
ZOOM_FROM = "2024-01"
REF_QUARTER = "2022Q2"        # last whole quarter before ChatGPT
REF_ZOOM = "2022H1"
MIN_CELL = 10
FAILURES = []


def opt(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        return None


def _mod(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name[:4], HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def period_of(ym: pd.Series, scheme: str) -> pd.Series:
    """Map year-month to the period label used by a specification."""
    y, m = ym.str[:4], ym.str[5:7].astype(int)
    if scheme == "quarter":
        return y + "Q" + ((m - 1) // 3 + 1).astype(str)
    if scheme == "zoom":
        # months from ZOOM_FROM, half-years before it
        half = y + np.where(m <= 6, "H1", "H2")
        return pd.Series(np.where(ym >= ZOOM_FROM, ym, half), index=ym.index)
    raise ValueError(scheme)


def _tok(period: str) -> str:
    """
    A period label turned into a legal R name.

    Monthly labels carry a hyphen, and `x_2025-01_y` is neither a legal R
    variable nor harmless in a formula, where the hyphen reads as
    subtraction. read.csv silently renames the column to `x_2025.01_y`,
    the formula then names a variable that does not exist, and the fit
    dies with a return code and no useful message. The quarterly scheme
    worked from the start only because `2025Q1` happens to contain no
    punctuation.
    """
    return period.replace("-", "_")


def build_terms(bal: pd.DataFrame, scheme: str, ref: str):
    b = bal.copy()
    b["period"] = period_of(pd.Series(b["year_month"].astype(str)), scheme)
    b["is_young"] = (b["age_group"] == YOUNG).astype(int)
    periods = sorted(p for p in b["period"].unique() if p != ref)
    pooled, young = [], []
    for p in periods:
        d = (b["period"] == p).astype(int)
        b[f"x_{_tok(p)}"] = b["expo_z"] * d
        b[f"x_{_tok(p)}_y"] = b["expo_z"] * d * b["is_young"]
        pooled.append(f"x_{_tok(p)}")
        young.append(f"x_{_tok(p)}_y")
    return b, pooled, young


def rebase_same_season(y: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """
    Subtract the mean of the pre-period coefficients sharing the same
    season. Quarters have a same-quarter counterpart in the pre-period;
    the zoom scheme's monthly points do not, so they are returned
    unchanged and the summary says so rather than implying a comparison
    that was never made.
    """
    y = y.copy()
    if scheme != "quarter":
        y["rebased"] = np.nan
        return y
    y["season"] = y["period"].str[-2:]
    pre = y[y["period"] < "2022Q4"]
    base = pre.groupby(["outcome", "season"])["coef"].mean()
    y["rebased"] = y.apply(
        lambda r: r["coef"] - base.get((r["outcome"], r["season"]), np.nan),
        axis=1)
    return y.drop(columns="season")


def run_spec(panels, scheme, ref):
    out = []
    for label, (bal, outcome) in panels.items():
        b, p_terms, y_terms = build_terms(bal, scheme, ref)
        n_terms = len(p_terms) + len(y_terms)
        print(f"  {scheme}/{label}: {n_terms} terms on {len(b):,} rows")
        t0 = time.time()
        p = b.copy()
        p["n_emp"] = p[outcome]
        r = mc.run_fepois_multi(p, OUT, tag=f"p59_{scheme}_{label}",
                                terms=p_terms + y_terms, fes=FES)
        del p
        gc.collect()
        if r.empty:
            FAILURES.append(f"{scheme}/{label}")
            print(f"  *** {scheme}/{label} FAILED and is skipped. A missing "
                  f"row is a missing fit, never a zero.")
            del b
            gc.collect()
            continue
        r = r[r["term"].isin(y_terms)].copy()
        # map the sanitised token back to the period label it came from
        back = {f"x_{_tok(p)}_y": p for p in
                sorted(set(period_of(pd.Series(bal["year_month"].astype(str)),
                                     scheme)))}
        r["period"] = r["term"].map(back)
        if r["period"].isna().any():
            raise RuntimeError(
                f"{scheme}/{label}: could not map term names back to "
                f"periods: {sorted(r.loc[r['period'].isna(), 'term'])[:3]}")
        r["outcome"], r["scheme"] = label, scheme
        out.append(r[["period", "coef", "se", "pvalue", "n_obs", "status",
                      "outcome", "scheme"]])
        print(f"    done in {(time.time()-t0)/60:.1f} min")
        del b
        gc.collect()
    if not out:
        return pd.DataFrame()
    return rebase_same_season(pd.concat(out, ignore_index=True), scheme)


def describe(bal: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    The plottable series: mean flow per firm-age cell by month, split by
    whether the cell's frozen 2019 exposure is in its top quartile, and by
    young against the rest. Floored, suppressed rows dropped.
    """
    b = bal[["year_month", "age_group", "expo_z", outcome]].copy()
    cut = b["expo_z"].quantile(0.75)
    b["high_exposure"] = (b["expo_z"] >= cut).astype(int)
    b["young"] = (b["age_group"] == YOUNG).astype(int)
    g = (b.groupby(["year_month", "high_exposure", "young"], observed=True)
         .agg(n_cells=(outcome, "size"), mean_flow=(outcome, "mean"))
         .reset_index())
    g["outcome"] = outcome
    return mc.enforce_min_cell(g, count_col="n_cells")


def main():
    mc.Tee(OUT / "59_log.txt")
    t_start = time.time()
    print("=" * 70)
    print("59: THE PATH AT MONTHLY RESOLUTION, WHERE IT MATTERS")
    print("=" * 70)
    print("  quarterly across the window, monthly from 2024.")
    print("  Exposure frozen 2019; the outcome uses payroll and birth year")
    print("  only, so no occupation code after 2019 enters any of this.")
    print(mc.mem_line("  "))

    l47, s54 = _mod("47L_age_baseline_exposure.py"), _mod("54_hiring_flows.py")
    base = mc.read_cache(CACHE / "L_baseline_2019.parquet")
    if base is None:
        raise RuntimeError("L_baseline_2019.parquet missing: run 47L first.")
    daioe = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)
    daioe = daioe.rename(columns={"pctl_rank_genai": "score"})[["ssyk4",
                                                               "score"]]
    expo = l47.build_exposure(base, daioe)

    panels = {}
    fl = [f for f in (mc.read_cache(CACHE / f"flows_{y}.parquet",
                                    require=s54.FLOW_COLS)
                      for y in s54.YEARS) if f is not None]
    if fl:
        bf = s54.build_panel(pd.concat(fl, ignore_index=True), expo)
        panels["hires"] = (bf, "n_hire")
        panels["seps"] = (bf, "n_sep")
    del fl
    gc.collect()
    if not panels:
        raise RuntimeError("flows_* cache missing: run 54 first.")

    # descriptive first: it is cheap and it survives everything
    desc = []
    for label, (bal, outcome) in panels.items():
        d = opt(f"descriptive {label}", describe, bal, outcome)
        if d is not None:
            desc.append(d)
    if desc:
        pd.concat(desc, ignore_index=True).to_csv(OUT / "descriptive.csv",
                                                  index=False)
        print(f"  descriptive series written for {len(desc)} outcomes")

    q = run_spec(panels, "quarter", REF_QUARTER)
    if not q.empty:
        q.to_csv(OUT / "path_quarter.csv", index=False)
    z = run_spec(panels, "zoom", REF_ZOOM)
    if not z.empty:
        z.to_csv(OUT / "path_zoom.csv", index=False)

    lines = ["THE PATH AT MONTHLY RESOLUTION", "=" * 52, "",
             "Exposure frozen 2019. Outcome from payroll and birth year",
             "only. Absorbed: employer x month, employer x age, month x age.",
             ""]
    if not q.empty:
        h = q[q.outcome == "hires"].set_index("period").sort_index()
        lines += ["QUARTERLY, 22-25 differential in hires.",
                  "rebased on the same quarter of the pre-period.", ""]
        for p, r in h.iterrows():
            t = r["coef"] / max(r["se"], 1e-12)
            reb = ("" if not np.isfinite(r["rebased"])
                   else f"   rebased {r['rebased']:+.4f}")
            lines.append(f"  {p}  {r['coef']:+.4f} (SE {r['se']:.4f}) "
                         f"t {t:+.1f}{reb}")
        lines.append("")
    if not z.empty:
        h = z[z.outcome == "hires"].set_index("period").sort_index()
        lines += ["MONTHLY FROM 2024, 22-25 differential in hires.",
                  "READ FOR SHAPE, NOT LEVEL: these monthly points have no",
                  "same-month pre-period counterpart, so the level is",
                  "contaminated by calendar seasonality. Whether the series",
                  "drifts or jumps is what it can tell you.", ""]
        for p, r in h.iterrows():
            t = r["coef"] / max(r["se"], 1e-12)
            lines.append(f"  {p}  {r['coef']:+.4f} (SE {r['se']:.4f}) "
                         f"t {t:+.1f}")
        lines.append("")
    if FAILURES:
        lines += ["FITS THAT FAILED AND ARE ABSENT ABOVE: "
                  + "; ".join(FAILURES),
                  "A missing row is a missing fit, never a zero.", ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. Resolution is bought with precision. Individual quarters and",
        "     months are noisier than half-years by construction, and one",
        "     significant month is not a finding. A consistent shape across",
        "     several consecutive points is the evidence here.",
        "  2. The half-yearly result this refines was NOT significant:",
        "     2025H1 hires came in at -0.069 with a standard error of",
        "     0.079. Nothing below overturns that; it can only show whether",
        "     the point estimate came from a drift or from one month.",
        "  3. 2025 is the preliminary AGI file and stops in June.",
        "  4. descriptive.csv carries the plottable series. If the event",
        "     study says something those four lines cannot show, distrust",
        "     the event study first.",
        "", f"Runtime {(time.time()-t_start)/60:.1f} min. " + mc.mem_line()]
    (OUT / "59_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n59 done.")


if __name__ == "__main__":
    main()

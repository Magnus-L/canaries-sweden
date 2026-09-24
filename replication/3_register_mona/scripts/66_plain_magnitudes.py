#!/usr/bin/env python3
"""
66_plain_magnitudes.py: the descriptive counterpart of the headline.

QUESTION
Every coefficient in the paper is a contrast, the exposed quartile against
the rest net of the fixed effects, and a reader also needs to know what
happened in levels. This script reports the raw employment, hiring and
separation series by exposure quartile and age band, on the same employer
classification the regressions use, so that the descriptive movement and
the estimate can be read side by side without reconciling two definitions
of exposed.

WHAT IT BUILDS
Exposure is the headline classification: script 47j's incumbent_exposure
on script 47h's OL_daioe score book, the 2019 incumbents scored from the
register as it stood in 2019. For each exposure quartile, age band (the
two young bands and the four incumbent bands) and period, the script
gives the mean and total of the outcome over employer-months, the number
of employers behind the cell and the number of cells. Periods: pre is
January 2022 to December 2023, post is January 2024 onward. From the
totals it computes the young-to-older ratio (the young band over the four
incumbent bands) per quartile and period and its log change, and the
difference between the top quartile and the mean of the other three. No
regression, no controls: composition, firm size and the business cycle
are inside every number. Cells resting on fewer than five employers are
dropped.

INPUTS AND OUTPUTS
Reads the caches edu_hr_weights_2019 to 2021 and edu_hr_2019 (script
47h), L_counts_2021 to 2025 (script 47L) and flows_2021 to 2025 (script
54); performs no SQL. Writes to output_66/: plain_stock.csv,
plain_flows.csv and 66_summary.txt.

IN THE PAPER
The descriptive table of Online Appendix III.1 (Table A9) is built on
the occupation-route classification by script 85, which imports this
script's describe(), PRE and POST_FROM so that the windows and the
five-employer floor are the same. The education-route tables written
here (plain_stock.csv) are read by the exhibit builder for comparison.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_66"
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
PRE = ("2022-01", "2023-12")      # the comparison window
POST_FROM = "2024-01"
MIN_FIRMS = 5                     # disclosure floor on any printed cell
FAILURES = []


def _mod(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def period(ym: pd.Series) -> pd.Series:
    out = pd.Series("drop", index=ym.index)
    out[(ym >= PRE[0]) & (ym <= PRE[1])] = "pre"
    out[ym >= POST_FROM] = "post"
    return out


def describe(panel: pd.DataFrame, expo: pd.DataFrame, value: str,
             young_bands, incumbent_bands) -> pd.DataFrame:
    """
    Mean of `value` per exposure quartile, age band and period, with the
    number of firms behind each cell so it can be floored.
    """
    p = panel.merge(expo[["employer_id", "fq"]], on="employer_id",
                    how="inner")
    p["year_month"] = p["year_month"].astype(str)
    p["period"] = period(p["year_month"])
    p = p[p["period"] != "drop"]
    p = p[p["age_group"].astype(str).isin(list(young_bands) +
                                          list(incumbent_bands))]
    if p.empty:
        return p
    g = (p.groupby(["fq", "age_group", "period"], observed=True)
         .agg(mean_value=(value, "mean"),
              total=(value, "sum"),
              n_firms=("employer_id", "nunique"),
              n_cells=("employer_id", "size")).reset_index())
    return g[g["n_firms"] >= MIN_FIRMS]


def age_ratio(g: pd.DataFrame, young: str, incumbent_bands) -> pd.DataFrame:
    """Young-to-older ratio per quartile and period, and its change."""
    y = g[g["age_group"] == young].set_index(["fq", "period"])["total"]
    o = (g[g["age_group"].astype(str).isin(incumbent_bands)]
         .groupby(["fq", "period"])["total"].sum())
    r = (y / o).rename("ratio").reset_index()
    w = r.pivot(index="fq", columns="period", values="ratio")
    if not {"pre", "post"} <= set(w.columns):
        return pd.DataFrame()
    w["log_change"] = np.log(w["post"] / w["pre"])
    return w.reset_index()


def main():
    mc.Tee(OUT / "66_log.txt")
    t0 = time.time()
    print("=" * 70)
    print("66: WHAT AN EXPOSED FIRM ACTUALLY EXPERIENCED")
    print("=" * 70)
    print("  Descriptive. Controls for nothing. Same firm classification as")
    print("  the headline, so the two can be read side by side.")
    print(mc.mem_line("  "))

    s61 = _mod("61_redated_triple.py", "s61")
    j47 = s61._j47()
    h47 = j47._h47()

    counts = {}
    for y in (2019, 2020, 2021):
        w = mc.read_cache(CACHE / f"edu_hr_weights_{y}.parquet",
                          require=h47.WEIGHT_COLS)
        if w is None:
            raise RuntimeError(f"edu_hr_weights_{y}.parquet missing: run 47h.")
        counts[y] = w
    book = h47.ScoreBook(counts, h47.load_key(), h47.load_scores())
    spec = dict(h47.DESIGNS["OL_daioe"])
    book.build("OL_daioe", spec)
    frame19 = mc.read_cache(CACHE / "edu_hr_2019.parquet",
                            require=h47.YEAR_COLS + ["n_emp"])
    if frame19 is None:
        raise RuntimeError("edu_hr_2019.parquet missing: run 47h first.")
    expo, _ = j47.incumbent_exposure(frame19, book, "OL_daioe", spec, "true",
                                     s61.TRUNC)
    del frame19
    gc.collect()
    share = float(expo.loc[expo["fq"] == 4, "n"].sum() / expo["n"].sum())
    print(f"  exposure: {len(expo):,} firms; the exposed quartile is "
          f"{share:.1%} of incumbent employment")

    cnt = [c for c in (mc.read_cache(CACHE / f"L_counts_{y}.parquet")
                       for y in s61.PANEL_YEARS) if c is not None]
    if not cnt:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    stock = pd.concat(cnt, ignore_index=True)
    del cnt
    gc.collect()

    lines = ["WHAT AN EXPOSED FIRM ACTUALLY EXPERIENCED", "=" * 52, "",
             "Descriptive, on the same firm classification as the headline.",
             f"Pre is {PRE[0]} to {PRE[1]}; post is {POST_FROM} onward.",
             "Nothing is controlled for. Read these beside the estimates, "
             "never instead of them.", "",
             f"The exposed quartile is {share:.1%} of incumbent employment, "
             f"so an", f"exposed firm sits {1-share:.2f} of any coefficient "
             f"below the national", f"average and the rest sit {share:.2f} "
             f"of it above.", ""]

    g = describe(stock, expo, "n_emp", j47.YOUNG_BANDS, j47.INCUMBENT_BANDS)
    if g.empty:
        FAILURES.append("stock")
    else:
        g.to_csv(OUT / "plain_stock.csv", index=False)
        for band in j47.YOUNG_BANDS:
            w = age_ratio(g, band, j47.INCUMBENT_BANDS)
            if w.empty:
                continue
            lines.append(f"EMPLOYMENT, {band} relative to workers 31 and over,")
            lines.append("change from the pre period to the post period:")
            for _, r in w.iterrows():
                lines.append(f"  quartile {int(r['fq'])}: "
                             f"{r['log_change']:+.4f}")
            hi = w[w.fq == 4]["log_change"]
            lo = w[w.fq < 4]["log_change"].mean()
            if len(hi):
                lines.append(f"  exposed minus the rest: "
                             f"{float(hi.iloc[0]) - lo:+.4f}")
            lines.append("")
    del stock
    gc.collect()

    fl = [f for f in (mc.read_cache(CACHE / f"flows_{y}.parquet")
                      for y in s61.PANEL_YEARS) if f is not None]
    if not fl:
        lines.append("FLOWS: no cache found, skipped.")
        FAILURES.append("flows")
    else:
        flows = pd.concat(fl, ignore_index=True)
        del fl
        gc.collect()
        out = []
        for col, name in (("n_hire", "HIRING"), ("n_sep", "SEPARATIONS")):
            if col not in flows.columns:
                continue
            gf = describe(flows, expo, col, j47.YOUNG_BANDS,
                          j47.INCUMBENT_BANDS)
            if gf.empty:
                continue
            gf = gf.assign(outcome=col)
            out.append(gf)
            for band in j47.YOUNG_BANDS:
                w = age_ratio(gf, band, j47.INCUMBENT_BANDS)
                if w.empty:
                    continue
                lines.append(f"{name}, {band} relative to workers 31 and "
                             f"over, change pre to post:")
                for _, r in w.iterrows():
                    lines.append(f"  quartile {int(r['fq'])}: "
                                 f"{r['log_change']:+.4f}")
                hi = w[w.fq == 4]["log_change"]
                lo = w[w.fq < 4]["log_change"].mean()
                if len(hi):
                    lines.append(f"  exposed minus the rest: "
                                 f"{float(hi.iloc[0]) - lo:+.4f}")
                lines.append("")
        if out:
            pd.concat(out, ignore_index=True).to_csv(OUT / "plain_flows.csv",
                                                     index=False)
        del flows
        gc.collect()

    if FAILURES:
        lines += ["NOT PRODUCED: " + "; ".join(FAILURES), ""]
    lines += [
        "READ THIS BEFORE QUOTING ANY OF IT:",
        "  1. These are raw means. Composition, firm size and the business",
        "     cycle are all inside them, which is exactly why the regression",
        "     exists. Never present one as though it were the other.",
        "  2. Do NOT expect the raw exposed-minus-rest figure to exceed the",
        "     estimate. An earlier version of this note asserted that and it",
        "     is wrong in both directions: these are ratios of sums across",
        "     firms, dominated by large employers, while the estimate is a",
        "     within-firm Poisson contrast. They can differ in size and in",
        "     SIGN without either being wrong, and where they do the",
        "     difference is the composition the fixed effects remove.",
        "  2b. Check the exposure measure matches before comparing at all.",
        "     These rows use the education-based firm quartile, which is the",
        "     headline classification. Script 63's flow estimates use the",
        "     age-specific OCCUPATIONAL measure. Those two are different",
        "     treatments and their numbers are not each other's benchmark.",
        "  3. Every printed cell rests on at least "
        f"{MIN_FIRMS} firms; thinner cells are dropped rather than blanked.",
        "  4. The pre window starts in 2022 rather than 2019, so the",
        "     comparison is not contaminated by the pandemic recovery.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "66_summary.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("\n66 done.")


if __name__ == "__main__":
    main()

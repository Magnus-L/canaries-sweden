#!/usr/bin/env python3
"""
04_decile_gradient.py: Equation (1) by exposure decile.

WHAT IT ESTIMATES
Occupation-by-month cells from January 2020 to June 2026 with a positive
count; the outcome is ln(postings). Occupations are cut into ten deciles of
the DAIOE generative-AI percentile, each occupation counting once. PostRB
(from April 2022) and PostGPT (from December 2022) are interacted with each
decile except the fifth, the reference, so that every coefficient is a
contrast with the median occupation; the least exposed decile is not the
reference because it carries its own response to the rate cycle. Occupation
and month fixed effects (linearmodels PanelOLS), standard errors clustered by
occupation.

INPUTS   data/processed/postings_daioe_merged_extended.csv (03)
OUTPUTS  output/results/postings_decile_gradient.csv;
         output/figures/postings_decile_gradient.pdf and .png
SERVES   Online Appendix II.8: Figure A4 and Table A7 (through 16)
RUNTIME  under a minute
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (PROCESSED, RESULTS, FIGURES, RIKSBANKEN_HIKE,  # noqa: E402
                    CHATGPT_LAUNCH, ORANGE, TEAL, GRAY, DARK_TEXT, POSTINGS_REGRESSION_END)


def main():
    print("Equation (1) by exposure decile")
    df = pd.read_csv(PROCESSED / ("postings_daioe_merged_extended.csv"
                     if (PROCESSED / "postings_daioe_merged_extended.csv").exists()
                     else "postings_daioe_merged.csv"),
                     dtype={"ssyk4": str})
    df["ssyk4"] = df["ssyk4"].str.zfill(4)
    df = df[(df["year_month"] >= "2020-01") & (df["year_month"] <= POSTINGS_REGRESSION_END)]

    # Unweighted deciles over occupations (same convention as the quartiles:
    # each SSYK4 counts once).
    occ = df[["ssyk4", "pctl_rank_genai"]].drop_duplicates("ssyk4")
    occ["decile"] = pd.qcut(occ["pctl_rank_genai"], 10, labels=False) + 1
    df = df.merge(occ[["ssyk4", "decile"]], on="ssyk4")

    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["post_rb"] = (df["date"] >= pd.Timestamp(RIKSBANKEN_HIKE)).astype(int)
    df["post_gpt"] = (df["date"] >= pd.Timestamp(CHATGPT_LAUNCH)).astype(int)
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])

    # Interactions: decile d x post period, the median decile (5) the
    # reference. Decile 1 is weighted towards construction and manual work
    # and carries its own response to the rate cycle, so contrasts against it
    # would read that response; the median occupation is the benchmark, and
    # the profile is plotted in full.
    DECS = [d for d in range(1, 11) if d != 5]
    for d in DECS:
        df[f"rb_d{d}"] = df["post_rb"] * (df["decile"] == d).astype(int)
        df[f"gpt_d{d}"] = df["post_gpt"] * (df["decile"] == d).astype(int)
    rb_cols = [f"rb_d{d}" for d in DECS]
    gpt_cols = [f"gpt_d{d}" for d in DECS]

    from linearmodels.panel import PanelOLS
    panel = df.set_index(["ssyk4", "date"])
    res = PanelOLS(panel["ln_ads"], panel[rb_cols + gpt_cols],
                   entity_effects=True, time_effects=True
                   ).fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for d in DECS:
        for period, col in (("post_riksbank", f"rb_d{d}"),
                            ("post_chatgpt", f"gpt_d{d}")):
            rows.append({"decile": d, "period": period,
                         "coef": res.params[col], "se": res.std_errors[col],
                         "pval": res.pvalues[col]})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "postings_decile_gradient.csv", index=False)
    print(f"  N = {res.nobs:,}; saved postings_decile_gradient.csv")

    # Gradient figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period, color, label, off in (
            ("post_riksbank", TEAL, "Post-Riksbank x decile", -0.12),
            ("post_chatgpt", ORANGE, "Post-ChatGPT x decile", +0.12)):
        sub = out[out["period"] == period]
        x = sub["decile"] + off
        ax.errorbar(x, sub["coef"], yerr=1.96 * sub["se"], fmt="o",
                    color=color, capsize=3, markersize=5, label=label)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.axvline(5, color=GRAY, lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("DAIOE genAI exposure decile (median decile 5 = reference)")
    ax.set_ylabel("Coefficient, ln(postings)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "postings_decile_gradient.pdf")
    fig.savefig(FIGURES / "postings_decile_gradient.png", dpi=300)
    print("  saved postings_decile_gradient.pdf/.png")

    # Is the profile monotone: is the top decile the most negative after the
    # launch?
    gpt = out[out["period"] == "post_chatgpt"].set_index("decile")["coef"]
    print("  post-ChatGPT profile vs median decile: "
          + ", ".join(f"d{d}={gpt.loc[d]:+.3f}" for d in DECS))


if __name__ == "__main__":
    main()

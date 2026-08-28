#!/usr/bin/env python3
"""
l03_decile_gradient_postings.py -- R1.4 / plan T11: the posting DiD by
exposure DECILE instead of the binary top-quartile split.

Why: R1 asks "why use such a discrete measure of AI exposure?"; Källberg's
thesis uses the top decile where we use the top quartile. Estimating the
gradient decile by decile answers both at once and shows what the quartile
choice buries. Specification is v1's spec 2 (occupation + month FE, both
post-period interactions), with the top-quartile dummy replaced by a full
set of decile × post interactions (decile 1 = least exposed = reference).

Output:
  tables/postings_decile_gradient.csv   (coef, se, p per decile x period)
  figures/postings_decile_gradient.pdf  (gradient plot, both coefficients)
Runtime: ~1 min.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (PROCESSED, V2_TAB, V2_FIG, RIKSBANKEN_HIKE,
                    CHATGPT_LAUNCH, ORANGE, TEAL, GRAY, DARK_TEXT)


def main():
    print("L3: posting decile gradient (R1.4)")
    df = pd.read_csv(PROCESSED / "postings_daioe_merged.csv",
                     dtype={"ssyk4": str})
    df["ssyk4"] = df["ssyk4"].str.zfill(4)
    df = df[(df["year_month"] >= "2020-01") & (df["year_month"] <= "2025-12")]

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

    # Interactions: decile d x post period. REFERENCE = MEDIAN DECILE (5):
    # decile 1 is construction/manual-heavy and carries its own rate-cycle
    # crash, so contrasts against it read as the reference's collapse, in
    # opposite directions under occupation- vs employment-weighting. The
    # median occupation is the benchmark the referee's question needs
    # (top-vs-typical), and the profile is plotted in full.
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
    out.to_csv(V2_TAB / "postings_decile_gradient.csv", index=False)
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
    fig.savefig(V2_FIG / "postings_decile_gradient.pdf")
    fig.savefig(V2_FIG / "postings_decile_gradient.png", dpi=300)
    print("  saved postings_decile_gradient.pdf/.png")

    # Monotonicity read (pre-committed in the plan): is the top decile the
    # most negative post-ChatGPT?
    gpt = out[out["period"] == "post_chatgpt"].set_index("decile")["coef"]
    print("  post-ChatGPT profile vs median decile: "
          + ", ".join(f"d{d}={gpt.loc[d]:+.3f}" for d in DECS))


if __name__ == "__main__":
    main()

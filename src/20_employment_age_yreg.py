#!/usr/bin/env python3
"""
08_employment_age.py — Employment by age group and AI exposure.

Downloads publicly available SCB register data (YREG54BAS) at the
SSYK 4-digit × age group × year level (2020–2024) and tests whether
young workers in high-AI-exposure occupations experienced differential
employment declines after ChatGPT — the "canaries in the coal mine"
hypothesis from Brynjolfsson et al. (2025).

This is the Swedish analogue of the Finnish test in Kauhanen & Rouvinen
(2025), using published aggregate statistics rather than individual
register microdata.

Data source:
  SCB Yrkesregistret, table YREG54BAS
  https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__AM__AM0208__AM0208E/YREG54BAS/

Important caveat:
  From reference year 2022, SCB switched from RAMS to BAS as the
  underlying register source. This methodological break coincides with
  our treatment timing. We flag this in all output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, TABDIR, FIGDIR,
    CHATGPT_LAUNCH, DAIOE_REF_YEAR,
    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY, LIGHT_BLUE,
    set_rcparams,
)

import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

set_rcparams()


# ── Step 1: Download YREG54BAS from SCB API ─────────────────────────────────

def download_scb_employment():
    """
    Download employment by SSYK4 × age group × year from SCB's PxWeb API.

    We omit the industry (SNI2007) and sex (Kon) dimensions so they are
    automatically totalled. Result: ~21,500 cells (430 occ × 10 age × 5 yr).
    """
    url = "https://api.scb.se/OV0104/v1/doris/en/ssd/AM/AM0208/AM0208E/YREG54BAS"

    # POST query: all occupations × all age groups × all years
    # Omitting SNI2007 and Kon → totalled across industry and sex
    query = {
        "query": [
            {
                "code": "Yrke2012",
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": "Alder",
                "selection": {"filter": "all", "values": ["*"]},
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["000006Y1"]},
            },
            {
                "code": "Tid",
                "selection": {"filter": "all", "values": ["*"]},
            },
        ],
        "response": {"format": "json-stat2"},
    }

    out_path = RAW / "scb_yreg54bas.json"

    # Check if already downloaded
    if out_path.exists():
        print(f"  Already downloaded → {out_path.name}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("  Downloading YREG54BAS from SCB API...")
    resp = requests.post(url, json=query, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved → {out_path.name}")

    return data


def parse_jsonstat2(data):
    """
    Parse JSON-stat2 format into a tidy pandas DataFrame.

    JSON-stat2 stores data as a flat array with dimensions defined by
    the 'dimension' and 'size' fields. We reconstruct the multi-index
    from the Cartesian product of dimension categories.

    We keep both the raw code and the human-readable label for each
    dimension, so we can extract SSYK codes from the code field.
    """
    dims = list(data["id"])
    sizes = data["size"]
    values = data["value"]

    # Build category codes and labels for each dimension
    dim_codes = {}
    dim_labels = {}
    for dim_name in dims:
        dim_info = data["dimension"][dim_name]
        cat = dim_info["category"]
        index_map = cat["index"]
        if isinstance(index_map, dict):
            sorted_codes = sorted(index_map.items(), key=lambda x: x[1])
            codes = [c[0] for c in sorted_codes]
        else:
            codes = list(index_map)
        labels = cat.get("label", {})
        dim_codes[dim_name] = codes
        dim_labels[dim_name] = [labels.get(c, c) for c in codes]

    # Build Cartesian product (rightmost dimension varies fastest)
    from itertools import product as cart_product

    code_rows = list(cart_product(*[dim_codes[d] for d in dims]))
    label_rows = list(cart_product(*[dim_labels[d] for d in dims]))

    df = pd.DataFrame(code_rows, columns=[f"{d}_code" for d in dims])
    for i, d in enumerate(dims):
        df[f"{d}_label"] = [r[i] for r in label_rows]
    df["value"] = values

    return df


# ── Step 2: Process and merge with DAIOE ─────────────────────────────────────

def process_employment(data):
    """Parse SCB data and create clean employment DataFrame."""
    print("  Parsing JSON-stat2...")
    df = parse_jsonstat2(data)

    # The parser creates code and label columns for each dimension
    # Yrke2012_code = "1120", Yrke2012_label = "Senior officials..."
    # Alder_label = "16-24 years", Tid_code = "2020"

    # Extract SSYK4 code directly from the code column
    df["ssyk4"] = df["Yrke2012_code"].astype(str)
    df["ssyk_label"] = df["Yrke2012_label"]
    df["age_group"] = df["Alder_label"]
    df["year"] = df["Tid_code"].astype(int)
    df["n_employed"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only 4-digit SSYK codes (filter out aggregates like "0002")
    df = df[df["ssyk4"].str.match(r"^\d{4}$")]

    # Drop missing/suppressed cells
    df = df.dropna(subset=["n_employed"])
    df = df[df["n_employed"] > 0].copy()
    df["n_employed"] = df["n_employed"].astype(int)

    # Keep only needed columns
    df = df[["ssyk4", "ssyk_label", "age_group", "year", "n_employed"]].copy()

    print(f"  Parsed: {len(df):,} cells, "
          f"{df['ssyk4'].nunique()} occupations, "
          f"{df['age_group'].nunique()} age groups, "
          f"{df['year'].nunique()} years")

    return df


def merge_with_daioe(emp_df):
    """Merge employment data with DAIOE AI exposure quartiles."""
    # Load DAIOE quartiles from our processed data
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")
    daioe["ssyk4"] = daioe["ssyk4"].astype(str).str.zfill(4)

    emp_df["ssyk4"] = emp_df["ssyk4"].astype(str).str.zfill(4)

    merged = emp_df.merge(
        daioe[["ssyk4", "pctl_rank_genai", "exposure_quartile", "high_exposure"]],
        on="ssyk4",
        how="inner",
    )

    n_matched = merged["ssyk4"].nunique()
    n_total = emp_df["ssyk4"].nunique()
    print(f"  Matched {n_matched} of {n_total} occupations with DAIOE "
          f"({100 * n_matched / n_total:.0f}%)")

    return merged


# ── Step 3: Create young/old classification ──────────────────────────────────

def classify_age(merged):
    """
    Create binary young/old indicator.

    Young = 16-24 years (entry-level, most comparable to Brynjolfsson's
    "canaries" which focuses on workers with ≤5 years experience).
    Old = 25+ years.

    We also create a broader young group (16-29) for robustness.
    """
    merged = merged.copy()

    # Young = 16-24 (strict, matches Brynjolfsson's entry-level focus)
    merged["young"] = (merged["age_group"] == "16-24 years").astype(int)

    # Broader: young = 16-29
    merged["young_broad"] = merged["age_group"].isin(
        ["16-24 years", "25-29 years"]
    ).astype(int)

    return merged


# ── Step 4: Aggregate and analyse ────────────────────────────────────────────

def compute_canaries_figure(merged):
    """
    Create the main canaries figure: employment indexed to 2020,
    four groups: Young×HighAI, Young×LowAI, Old×HighAI, Old×LowAI.

    This directly tests Brynjolfsson et al.'s hypothesis that young
    workers in AI-exposed occupations are the "canaries."
    """
    print("  Computing canaries trajectories...")

    # Aggregate: sum employment by year × young × high_exposure
    agg = (
        merged.groupby(["year", "young", "high_exposure"])["n_employed"]
        .sum()
        .reset_index()
    )

    # Index to 2020 = 100
    base = agg[agg["year"] == 2020].set_index(["young", "high_exposure"])["n_employed"]
    agg["base"] = agg.set_index(["young", "high_exposure"]).index.map(
        lambda x: base.get(x, np.nan)
    )
    # Re-merge base values properly
    agg = agg.merge(
        agg[agg["year"] == 2020][["young", "high_exposure", "n_employed"]].rename(
            columns={"n_employed": "base_emp"}
        ),
        on=["young", "high_exposure"],
    )
    agg["index"] = 100 * agg["n_employed"] / agg["base_emp"]

    # Labels for the four groups
    def group_label(row):
        age = "Young (16–24)" if row["young"] == 1 else "Older (25+)"
        ai = "High AI" if row["high_exposure"] == 1 else "Low AI"
        return f"{age}, {ai}"

    agg["group"] = agg.apply(group_label, axis=1)

    # Save data
    agg.to_csv(TABDIR / "employment_age_ai.csv", index=False)

    return agg


def plot_canaries(agg):
    """
    Plot: employment indexed to 100 at 2020, four lines.

    If canaries hypothesis holds: "Young, High AI" should diverge
    downward after 2022 relative to the other three groups.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define styles for the four groups
    styles = {
        "Young (16–24), High AI": {"color": ORANGE, "linewidth": 2.5,
                                     "marker": "o", "markersize": 8,
                                     "linestyle": "-"},
        "Young (16–24), Low AI": {"color": ORANGE, "linewidth": 1.5,
                                    "marker": "s", "markersize": 6,
                                    "linestyle": "--"},
        "Older (25+), High AI": {"color": DARK_BLUE, "linewidth": 2.5,
                                   "marker": "o", "markersize": 8,
                                   "linestyle": "-"},
        "Older (25+), Low AI": {"color": DARK_BLUE, "linewidth": 1.5,
                                  "marker": "s", "markersize": 6,
                                  "linestyle": "--"},
    }

    for group, style in styles.items():
        subset = agg[agg["group"] == group].sort_values("year")
        if len(subset) > 0:
            ax.plot(subset["year"], subset["index"], label=group, **style)

    # ChatGPT launch marker (between 2022 and 2023)
    ax.axvline(2022.9, color=TEAL, linewidth=1.5, linestyle=":",
               alpha=0.8, label="ChatGPT (Nov 2022)")

    # RAMS-to-BAS break annotation
    ax.annotate(
        "RAMS-to-BAS\nbreak",
        xy=(2022, ax.get_ylim()[0] + 2), fontsize=8,
        color=GRAY, ha="center", style="italic",
    )

    ax.axhline(100, color=GRAY, linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Year")
    ax.set_ylabel("Employment index (2020 = 100)")
    ax.set_title("Employment by age and AI exposure, Sweden 2020–2024")
    ax.set_xticks([2020, 2021, 2022, 2023, 2024])
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "figA7_canaries_employment.png")
    plt.close(fig)
    print(f"  Saved → figA7_canaries_employment.png")


def triple_diff(merged):
    """
    Simple triple-difference: employment change by young × high AI.

    Computes the DiD in changes:
      ΔΔΔ = (ΔYoung,HighAI - ΔYoung,LowAI) - (ΔOld,HighAI - ΔOld,LowAI)

    where Δ = percentage change from 2020 to 2024.

    Also runs a simple regression on the panel:
      ln(emp_it) = α_i + γ_t + β₁·Post·High + β₂·Post·Young
                   + β₃·Post·Young·High + ε_it

    where Post = 1 if year ≥ 2023 (first full year after ChatGPT).
    """
    print("  Computing triple-difference...")

    # Aggregate to year × young × high_exposure
    agg = (
        merged.groupby(["year", "young", "high_exposure"])["n_employed"]
        .sum()
        .reset_index()
    )

    # Pivot for DiD calculation
    base = agg[agg["year"] == 2020].set_index(["young", "high_exposure"])["n_employed"]
    end = agg[agg["year"] == 2024].set_index(["young", "high_exposure"])["n_employed"]

    pct_change = 100 * (end - base) / base

    # Extract the four cells
    try:
        dy_hi = pct_change.loc[(1, 1)]  # Young, High AI
        dy_lo = pct_change.loc[(1, 0)]  # Young, Low AI
        do_hi = pct_change.loc[(0, 1)]  # Old, High AI
        do_lo = pct_change.loc[(0, 0)]  # Old, Low AI

        # Triple-difference
        did_young = dy_hi - dy_lo  # AI effect on young
        did_old = do_hi - do_lo    # AI effect on old
        triple_did = did_young - did_old

        print(f"\n    Employment change 2020→2024 (%):")
        print(f"      Young, High AI:  {dy_hi:+.1f}%")
        print(f"      Young, Low AI:   {dy_lo:+.1f}%")
        print(f"      Older, High AI:  {do_hi:+.1f}%")
        print(f"      Older, Low AI:   {do_lo:+.1f}%")
        print(f"    DiD (AI effect on young):  {did_young:+.1f} pp")
        print(f"    DiD (AI effect on older):  {did_old:+.1f} pp")
        print(f"    Triple-diff (canaries):    {triple_did:+.1f} pp")
    except KeyError as e:
        print(f"    Could not compute simple DiD: missing key {e}")
        triple_did = None

    # ── Panel regression ──
    # Collapse to occupation × year × young (binary) cells
    panel = (
        merged.groupby(["ssyk4", "year", "young", "high_exposure"])["n_employed"]
        .sum()
        .reset_index()
    )
    panel = panel[panel["n_employed"] > 0].copy()
    panel["ln_emp"] = np.log(panel["n_employed"])
    panel["post_chatgpt"] = (panel["year"] >= 2023).astype(int)

    # Triple interaction
    panel["post_high"] = panel["post_chatgpt"] * panel["high_exposure"]
    panel["post_young"] = panel["post_chatgpt"] * panel["young"]
    panel["post_young_high"] = (
        panel["post_chatgpt"] * panel["young"] * panel["high_exposure"]
    )

    try:
        from linearmodels.panel import PanelOLS

        # Create entity = ssyk4_young (occupation × age group)
        panel["entity"] = panel["ssyk4"] + "_" + panel["young"].astype(str)
        panel_idx = panel.set_index(["entity", "year"])

        exog_cols = ["post_high", "post_young", "post_young_high"]
        mod = PanelOLS(
            dependent=panel_idx["ln_emp"],
            exog=panel_idx[exog_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n    Panel regression (ln employment):")
        for v in exog_cols:
            b = res.params[v]
            se = res.std_errors[v]
            p = res.pvalues[v]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"      {v:25s} = {b:+.4f}{stars} (SE = {se:.4f})")
        print(f"      N = {res.nobs:,}")

        # Save results
        reg_df = pd.DataFrame({
            "variable": res.params.index,
            "coefficient": res.params.values,
            "std_error": res.std_errors.values,
            "p_value": res.pvalues.values,
        })
        reg_df.to_csv(TABDIR / "canaries_regression.csv", index=False)
        print(f"    Saved → canaries_regression.csv")

    except ImportError:
        print("    linearmodels not available — skipping panel regression")
    except Exception as e:
        print(f"    Regression failed: {e}")

    return triple_did


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 8: Employment by age group × AI exposure (canaries test)")
    print("=" * 70)

    # Download
    data = download_scb_employment()

    # Parse
    emp = process_employment(data)

    # Merge with DAIOE
    merged = merge_with_daioe(emp)

    # Classify age groups
    merged = classify_age(merged)

    # Save processed data
    merged.to_csv(PROCESSED / "employment_age_ai.csv", index=False)
    print(f"  Saved → employment_age_ai.csv")

    # Compute and plot
    agg = compute_canaries_figure(merged)
    plot_canaries(agg)

    # Triple-diff
    triple_diff(merged)

    print("\nDone.")


if __name__ == "__main__":
    main()

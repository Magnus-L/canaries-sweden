#!/usr/bin/env python3
"""
11_remote_work_robustness.py — Dingel-Neiman teleworkability robustness check.

Addresses Brynjolfsson et al. (2025, Nov) Figures A25–A27, which show their
canaries effect holds after controlling for remote work feasibility.

Economic logic: AI-exposed occupations overlap heavily with teleworkable
occupations. If the posting decline reflects remote-work-enabled labour
market restructuring rather than AI per se, controlling for teleworkability
should absorb the effect. This script crosswalks Dingel & Neiman (2020)
to SSYK 2012 and re-runs the baseline DiD for teleworkable vs
non-teleworkable occupations separately.

Crosswalk chain:
    O*NET-SOC (XX-XXXX.XX) → SOC 2010 (XX-XXXX) → ISCO-08 (4-digit) → SSYK 2012 (4-digit)

Data sources:
    - Dingel & Neiman (2020): github.com/jdingel/DingelNeiman-workathome
    - BLS SOC 2010 → ISCO-08 crosswalk (local copy)
    - SCB SSYK 2012 → ISCO-08 key (local copy)

Outputs:
    - figA_telework_robustness.png  (DiD event study split by teleworkability)
    - telework_did_results.csv      (regression coefficients)
    - telework_ssyk_mapping.csv     (SSYK-level teleworkability scores)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW, PROCESSED, FIGDIR, TABDIR,
    RIKSBANKEN_HIKE, CHATGPT_LAUNCH,
    DARK_BLUE, ORANGE, TEAL, GRAY, LIGHT_GRAY,
    Q_COLORS, set_rcparams,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

set_rcparams()

# ── Local crosswalk paths (downloaded separately) ────────────────────────────
# These are Excel files from BLS and SCB, stored in /tmp by the research agent.
# If missing, the script will try to download the Dingel-Neiman CSV from GitHub.

BLS_CROSSWALK = Path("/tmp/isco_soc_crosswalk2.xls")
SCB_CROSSWALK = Path("/tmp/ssyk2012_isco08.xlsx")
DN_URL = "https://raw.githubusercontent.com/jdingel/DingelNeiman-workathome/master/occ_onet_scores/output/occupations_workathome.csv"


def load_dingel_neiman() -> pd.DataFrame:
    """
    Load Dingel-Neiman (2020) teleworkability scores.

    The raw data uses O*NET-SOC codes (XX-XXXX.XX). We truncate to
    6-digit SOC 2010 codes (XX-XXXX) and average the binary teleworkable
    indicator within each SOC code (some O*NET detailed codes share
    a SOC parent).

    Returns: DataFrame with columns [soc2010, teleworkable]
    """
    local = RAW / "dingel_neiman_telework.csv"

    if local.exists():
        dn = pd.read_csv(local)
    else:
        print(f"  Downloading Dingel-Neiman from GitHub...")
        dn = pd.read_csv(DN_URL)
        dn.to_csv(local, index=False)
        print(f"  Saved to {local.name}")

    print(f"  Raw O*NET-SOC rows: {len(dn)}")

    # Truncate O*NET-SOC (e.g. "11-1011.00") to SOC 2010 ("11-1011")
    dn["soc2010"] = dn["onetsoccode"].str[:7]

    # Average teleworkable within SOC (most are 1-to-1; a few have 2-3 variants)
    soc = dn.groupby("soc2010")["teleworkable"].mean().reset_index()
    print(f"  Unique SOC 2010 codes: {len(soc)}")
    print(f"  Mean teleworkable: {soc['teleworkable'].mean():.3f}")

    return soc


def load_soc_to_isco() -> pd.DataFrame:
    """
    Load BLS SOC 2010 → ISCO-08 crosswalk.

    Many-to-many: one SOC can map to multiple ISCO codes. We keep all
    pairs for equal-weight averaging downstream.

    Returns: DataFrame with columns [soc2010, isco08]
    """
    if not BLS_CROSSWALK.exists():
        raise FileNotFoundError(
            f"BLS crosswalk not found at {BLS_CROSSWALK}. "
            "Download from https://www.bls.gov/soc/ISCO_SOC_Crosswalk.xls"
        )

    df = pd.read_excel(
        BLS_CROSSWALK,
        sheet_name="2010 SOC to ISCO-08",
        header=None,
        skiprows=8,
    )
    df.columns = ["soc2010", "soc_title", "part", "isco08", "isco_title", "comment"]
    df = df[["soc2010", "isco08"]].dropna()

    # Ensure string types
    df["soc2010"] = df["soc2010"].astype(str).str.strip()
    df["isco08"] = df["isco08"].astype(str).str.strip().str.zfill(4)

    print(f"  BLS crosswalk: {len(df)} SOC-ISCO pairs "
          f"({df['soc2010'].nunique()} SOC → {df['isco08'].nunique()} ISCO)")

    return df


def load_isco_to_ssyk() -> pd.DataFrame:
    """
    Load SCB SSYK 2012 → ISCO-08 key and INVERT it to ISCO → SSYK.

    The SCB key maps SSYK → ISCO (comma-separated). We explode the
    comma-separated ISCO codes and invert the direction.

    Returns: DataFrame with columns [isco08, ssyk2012]
    """
    if not SCB_CROSSWALK.exists():
        raise FileNotFoundError(
            f"SCB crosswalk not found at {SCB_CROSSWALK}. "
            "Download from scb.se"
        )

    df = pd.read_excel(
        SCB_CROSSWALK,
        sheet_name="Nyckel",
        header=None,
        skiprows=5,
    )
    df.columns = ["ssyk2012", "ssyk_title", "isco08", "isco_title", "notes"]
    df = df[["ssyk2012", "isco08"]].dropna()

    # ssyk2012 to 4-digit string
    df["ssyk2012"] = df["ssyk2012"].astype(str).str.strip().str.zfill(4)

    # Explode comma-separated ISCO codes into individual rows
    df["isco08"] = df["isco08"].astype(str)
    df = df.assign(
        isco08=df["isco08"].str.split(r",\s*")
    ).explode("isco08")
    df["isco08"] = df["isco08"].str.strip().str.zfill(4)

    print(f"  SCB crosswalk (inverted): {len(df)} ISCO-SSYK pairs "
          f"({df['isco08'].nunique()} ISCO → {df['ssyk2012'].nunique()} SSYK)")

    return df


def build_ssyk_telework(dn: pd.DataFrame, soc_isco: pd.DataFrame,
                        isco_ssyk: pd.DataFrame) -> pd.DataFrame:
    """
    Chain the crosswalks: SOC → ISCO → SSYK, averaging teleworkability
    at each step (equal-weight, following Dingel-Neiman's international
    crosswalk approach).

    Returns: DataFrame with columns [ssyk4, teleworkable]
    """
    # Step 1: SOC → teleworkable (already done)
    # Step 2: Merge SOC → ISCO, carrying teleworkable scores
    soc_isco_tw = soc_isco.merge(dn, on="soc2010", how="inner")
    print(f"  After SOC-ISCO merge: {len(soc_isco_tw)} rows")

    # Average teleworkable within each ISCO code
    # (multiple SOC codes map to the same ISCO)
    isco_tw = (
        soc_isco_tw
        .groupby("isco08")["teleworkable"]
        .mean()
        .reset_index()
    )
    print(f"  ISCO codes with telework score: {len(isco_tw)}")

    # Step 3: Merge ISCO → SSYK, carrying teleworkable scores
    isco_ssyk_tw = isco_ssyk.merge(isco_tw, on="isco08", how="inner")
    print(f"  After ISCO-SSYK merge: {len(isco_ssyk_tw)} rows")

    # Average teleworkable within each SSYK code
    ssyk_tw = (
        isco_ssyk_tw
        .groupby("ssyk2012")["teleworkable"]
        .mean()
        .reset_index()
        .rename(columns={"ssyk2012": "ssyk4"})
    )
    print(f"  SSYK codes with telework score: {len(ssyk_tw)}")
    print(f"  Telework range: {ssyk_tw['teleworkable'].min():.3f} "
          f"to {ssyk_tw['teleworkable'].max():.3f}")
    print(f"  Median: {ssyk_tw['teleworkable'].median():.3f}")

    return ssyk_tw


def run_did_by_telework(merged: pd.DataFrame, daioe: pd.DataFrame,
                        ssyk_tw: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run the baseline DiD (specification 2) separately for
    teleworkable and non-teleworkable occupations.

    Split at the median teleworkability score. This tests whether
    the (in)significance of beta_2 depends on remote work feasibility.
    """
    # Prepare merged data (already has DAIOE columns from pipeline step 04)
    df = merged.copy()
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["ln_ads"] = np.log(df["n_ads"] + 1)
    df["high"] = (df["exposure_quartile"] == "Q4 (highest)").astype(int)

    # Merge teleworkability
    ssyk_tw["ssyk4"] = ssyk_tw["ssyk4"].astype(str).str.zfill(4)
    df = df.merge(ssyk_tw, on="ssyk4", how="inner")

    # Split at median
    median_tw = df.groupby("ssyk4")["teleworkable"].first().median()
    df["telework_group"] = np.where(
        df["teleworkable"] >= median_tw, "Teleworkable", "Non-teleworkable"
    )

    # Treatment dummies
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)
    df["post_rb"] = (df["date"] >= rb_date).astype(int)
    df["post_gpt"] = (df["date"] >= gpt_date).astype(int)
    df["rb_high"] = df["post_rb"] * df["high"]
    df["gpt_high"] = df["post_gpt"] * df["high"]

    # Year-month string for FE
    df["ym"] = df["year_month"]

    results = []
    for group in ["Teleworkable", "Non-teleworkable", "All"]:
        sub = df if group == "All" else df[df["telework_group"] == group]
        n_occ = sub["ssyk4"].nunique()

        try:
            model = smf.ols(
                "ln_ads ~ C(ssyk4) + C(ym) + rb_high + gpt_high",
                data=sub,
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["ssyk4"]})

            results.append({
                "group": group,
                "n_occ": n_occ,
                "n_obs": len(sub),
                "beta1_rb": model.params.get("rb_high", np.nan),
                "se_rb": model.bse.get("rb_high", np.nan),
                "p_rb": model.pvalues.get("rb_high", np.nan),
                "beta2_gpt": model.params.get("gpt_high", np.nan),
                "se_gpt": model.bse.get("gpt_high", np.nan),
                "p_gpt": model.pvalues.get("gpt_high", np.nan),
            })

            print(f"\n  {group} (n={n_occ} occupations, {len(sub):,} obs):")
            print(f"    β₁ (Riksbank × High) = {results[-1]['beta1_rb']:.3f} "
                  f"(SE={results[-1]['se_rb']:.3f}, p={results[-1]['p_rb']:.3f})")
            print(f"    β₂ (ChatGPT × High)  = {results[-1]['beta2_gpt']:.3f} "
                  f"(SE={results[-1]['se_gpt']:.3f}, p={results[-1]['p_gpt']:.3f})")
        except Exception as e:
            print(f"  {group}: regression failed — {e}")

    return pd.DataFrame(results)


def plot_telework_comparison(results: pd.DataFrame):
    """
    Bar chart comparing β₂ (ChatGPT × High) across teleworkable vs
    non-teleworkable occupations.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    groups = ["Non-teleworkable", "Teleworkable", "All"]
    colors = [LIGHT_GRAY, ORANGE, DARK_BLUE]

    x = np.arange(len(groups))
    betas = []
    ses = []
    for g in groups:
        row = results[results["group"] == g].iloc[0]
        betas.append(row["beta2_gpt"])
        ses.append(row["se_gpt"] * 1.96)  # 95% CI

    bars = ax.bar(x, betas, yerr=ses, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.5, capsize=5, zorder=3)

    ax.axhline(0, color=GRAY, linewidth=0.5, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel(r"$\hat{\beta}_2$ (Post-ChatGPT $\times$ High AI exposure)")
    ax.set_title("ChatGPT effect by teleworkability\n(Dingel-Neiman 2020 classification)")

    # Add p-values
    for i, (b, row_g) in enumerate(zip(betas, groups)):
        row = results[results["group"] == row_g].iloc[0]
        p = row["p_gpt"]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        label = f"{b:.3f}{sig}\n(p={p:.3f})"
        y_pos = b + ses[i] + 0.01 if b >= 0 else b - ses[i] - 0.03
        ax.text(i, y_pos, label, ha="center", va="bottom" if b >= 0 else "top",
                fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "figA_telework_robustness.png"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"\n  Saved: {out.name}")


def main():
    print("=" * 70)
    print("STEP 11: Dingel-Neiman teleworkability robustness")
    print("=" * 70)

    # Load crosswalk components
    print("\n1. Loading Dingel-Neiman teleworkability scores...")
    dn = load_dingel_neiman()

    print("\n2. Loading BLS SOC 2010 → ISCO-08 crosswalk...")
    soc_isco = load_soc_to_isco()

    print("\n3. Loading SCB SSYK 2012 → ISCO-08 key (inverted)...")
    isco_ssyk = load_isco_to_ssyk()

    print("\n4. Building SSYK teleworkability scores...")
    ssyk_tw = build_ssyk_telework(dn, soc_isco, isco_ssyk)

    # Save mapping for reference
    out_map = TABDIR / "telework_ssyk_mapping.csv"
    ssyk_tw.to_csv(out_map, index=False)
    print(f"  Saved: {out_map.name}")

    # Load project data
    print("\n5. Loading Platsbanken + DAIOE data...")
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    daioe = pd.read_csv(PROCESSED / "daioe_quartiles.csv")
    print(f"  {len(merged):,} occupation×month rows")

    # Check match rate
    merged_ssyk = set(merged["ssyk4"].astype(str).str.zfill(4))
    tw_ssyk = set(ssyk_tw["ssyk4"].astype(str).str.zfill(4))
    overlap = merged_ssyk & tw_ssyk
    print(f"  Platsbanken SSYK codes: {len(merged_ssyk)}")
    print(f"  Telework SSYK codes: {len(tw_ssyk)}")
    print(f"  Overlap: {len(overlap)} ({len(overlap)/len(merged_ssyk)*100:.0f}%)")

    # Run DiD
    print("\n6. Running DiD by teleworkability group...")
    results = run_did_by_telework(merged, daioe, ssyk_tw)

    # Save results
    out_res = TABDIR / "telework_did_results.csv"
    results.to_csv(out_res, index=False)
    print(f"  Saved: {out_res.name}")

    # Plot
    print("\n7. Plotting comparison...")
    plot_telework_comparison(results)

    # Interpret
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    tw_row = results[results["group"] == "Teleworkable"].iloc[0]
    ntw_row = results[results["group"] == "Non-teleworkable"].iloc[0]

    if tw_row["p_gpt"] > 0.05 and ntw_row["p_gpt"] > 0.05:
        print("  β₂ is insignificant in BOTH groups → The posting null")
        print("  is not an artefact of teleworkability confounding.")
        print("  Remote work feasibility does not explain the result.")
    elif tw_row["p_gpt"] < 0.05 and ntw_row["p_gpt"] > 0.05:
        print("  β₂ significant only in teleworkable occupations →")
        print("  The AI effect (if any) concentrates in remote-capable jobs.")
    else:
        print("  Mixed results — see table for details.")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
16_eloundou_robustness.py — Robustness check using Eloundou et al. (2023) GPT exposure.

Responds to referee comment (Sune Karlsson): re-run the posting DiD with
an alternative AI exposure measure. Uses the standard β-measure from
Eloundou et al. (2023), "GPTs are GPTs", which combines human expert and
GPT-4 ratings of task-level exposure (E1 + 0.5×E2).

Crosswalk chain: O*NET-SOC 2019 → SOC 2010 (strip suffix) → ISCO-08 (BLS)
                 → SSYK 2012 (SCB). Many-to-many links handled by averaging.

Produces:
  - data/processed/eloundou_ssyk_matched.csv  (crosswalked scores)
  - tables/eloundou_robustness.tex            (LaTeX comparison table)
  - tables/eloundou_robustness.csv            (machine-readable results)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RAW, PROCESSED, TABDIR, RIKSBANKEN_HIKE, CHATGPT_LAUNCH

import pandas as pd
import numpy as np
import urllib.request


# ── Step 1: Download Eloundou data ──────────────────────────────────────────

def download_eloundou() -> pd.DataFrame:
    """
    Fetch occ_level.csv from the GPTs-are-GPTs GitHub repo (MIT license).

    Key column: dv_rating_beta — the standard exposure measure
    (E1 + 0.5×E2, where E1 = direct exposure, E2 = exposure via tools).
    O*NET-SOC codes in format XX-XXXX.XX.
    """
    url = (
        "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/"
        "main/data/occ_level.csv"
    )
    local = RAW / "eloundou_occ_level.csv"

    if not local.exists():
        print(f"  Downloading from GitHub...")
        urllib.request.urlretrieve(url, local)
        print(f"  Saved → {local.name}")
    else:
        print(f"  Using cached {local.name}")

    df = pd.read_csv(local)
    print(f"  {len(df)} occupations, columns: {list(df.columns)}")
    print(f"  dv_rating_beta range: {df['dv_rating_beta'].min():.3f} – "
          f"{df['dv_rating_beta'].max():.3f}")

    return df


# ── Step 2: Build SOC → ISCO → SSYK crosswalk ──────────────────────────────

def load_soc_isco_crosswalk() -> pd.DataFrame:
    """
    Load BLS SOC 2010 → ISCO-08 crosswalk.

    File has header at row 6. Many-to-many: one ISCO can map to multiple SOC
    codes and vice versa.
    """
    path = RAW / "isco_soc_crosswalk2.xls"
    df = pd.read_excel(path, header=6)

    # Standardise column names
    df.columns = ["isco08", "isco08_title", "part", "soc2010", "soc2010_title", "comment"]

    # Clean codes — strip whitespace
    df["isco08"] = df["isco08"].astype(str).str.strip()
    df["soc2010"] = df["soc2010"].astype(str).str.strip()

    # Keep only the code columns we need
    df = df[["soc2010", "isco08"]].dropna().drop_duplicates()

    print(f"  BLS crosswalk: {len(df)} SOC→ISCO pairs")
    return df


def load_ssyk_isco_crosswalk() -> pd.DataFrame:
    """
    Load SCB SSYK 2012 → ISCO-08 crosswalk.

    'Nyckel' sheet, header at row 3. ISCO column can contain comma-separated
    codes (one SSYK maps to multiple ISCO). We explode these into separate rows.
    """
    path = RAW / "ssyk2012_isco08.xlsx"
    df = pd.read_excel(path, sheet_name="Nyckel", header=3)

    # Columns: SSYK 2012 kod, (unnamed), ISCO-08 kod, (unnamed), (unnamed)
    # Keep columns 0 and 2
    df = df.iloc[:, [0, 2]].copy()
    df.columns = ["ssyk4", "isco08"]
    df = df.dropna(subset=["ssyk4", "isco08"])

    # Clean SSYK codes — ensure 4-digit zero-padded strings
    df["ssyk4"] = df["ssyk4"].astype(str).str.strip().str.zfill(4)

    # ISCO column may contain comma-separated codes — explode them
    # First convert everything to string
    df["isco08"] = df["isco08"].astype(str).str.strip()
    df = df.assign(isco08=df["isco08"].str.split(r",\s*")).explode("isco08")
    df["isco08"] = df["isco08"].str.strip().str.zfill(4)

    # Remove any non-numeric artefacts
    df = df[df["isco08"].str.match(r"^\d{4}$")]
    df = df.drop_duplicates()

    print(f"  SCB crosswalk: {len(df)} SSYK→ISCO pairs "
          f"({df['ssyk4'].nunique()} SSYK codes)")
    return df


def build_eloundou_ssyk(eloundou: pd.DataFrame) -> pd.DataFrame:
    """
    Map Eloundou β-scores from O*NET-SOC → SSYK 2012 via ISCO-08.

    Strategy for many-to-many:
      1. Strip O*NET suffix (.XX) to get SOC 6-digit codes
      2. Average β across O*NET detailed occupations within each SOC
      3. Join SOC → ISCO (BLS crosswalk)
      4. Average β across SOC codes within each ISCO
      5. Join ISCO → SSYK (SCB crosswalk, reversed: ISCO → SSYK)
      6. Average β across ISCO codes within each SSYK

    This produces one score per SSYK4 occupation.
    """
    # Step 2a: Strip O*NET suffix → SOC 6-digit
    df = eloundou[["O*NET-SOC Code", "dv_rating_beta"]].copy()
    df.rename(columns={"O*NET-SOC Code": "onet_soc"}, inplace=True)

    # O*NET format: XX-XXXX.XX → take XX-XXXX
    df["soc2010"] = df["onet_soc"].str.replace(r"\.\d+$", "", regex=True)

    # Average within SOC (multiple O*NET detail codes per SOC)
    soc_scores = (
        df.groupby("soc2010")["dv_rating_beta"]
        .mean()
        .reset_index()
    )
    print(f"  {len(soc_scores)} unique SOC 2010 codes with Eloundou scores")

    # Step 2b: SOC → ISCO
    soc_isco = load_soc_isco_crosswalk()
    soc_isco_scored = soc_isco.merge(soc_scores, on="soc2010", how="inner")
    print(f"  {len(soc_isco_scored)} SOC→ISCO links with scores")

    # Average within ISCO (multiple SOC codes may map to one ISCO)
    isco_scores = (
        soc_isco_scored.groupby("isco08")["dv_rating_beta"]
        .mean()
        .reset_index()
    )
    print(f"  {len(isco_scores)} ISCO-08 codes with Eloundou scores")

    # Step 2c: ISCO → SSYK
    ssyk_isco = load_ssyk_isco_crosswalk()

    # Reverse the crosswalk: SSYK→ISCO becomes rows with (ssyk4, isco08),
    # merge on ISCO to bring in scores
    ssyk_scored = ssyk_isco.merge(isco_scores, on="isco08", how="inner")
    print(f"  {len(ssyk_scored)} SSYK→ISCO links with scores")

    # Average within SSYK (multiple ISCO codes may map to one SSYK)
    ssyk_final = (
        ssyk_scored.groupby("ssyk4")["dv_rating_beta"]
        .mean()
        .reset_index()
    )
    print(f"  {len(ssyk_final)} SSYK4 codes with final Eloundou score")

    return ssyk_final


# ── Step 3: Assign quartiles ────────────────────────────────────────────────

def assign_eloundou_quartiles(ssyk_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unweighted quartiles of Eloundou β-score, matching the approach
    in 04_merge_and_classify.py for DAIOE (each SSYK4 counts once).

    Top quartile (Q4) → high_exposure_eloundou = 1.
    """
    q25 = ssyk_scores["dv_rating_beta"].quantile(0.25)
    q50 = ssyk_scores["dv_rating_beta"].quantile(0.50)
    q75 = ssyk_scores["dv_rating_beta"].quantile(0.75)

    print(f"  Eloundou quartile boundaries: "
          f"Q25={q25:.3f}, Q50={q50:.3f}, Q75={q75:.3f}")

    ssyk_scores = ssyk_scores.copy()
    ssyk_scores["high_exposure_eloundou"] = (
        ssyk_scores["dv_rating_beta"] > q75
    ).astype(int)

    n_high = ssyk_scores["high_exposure_eloundou"].sum()
    n_total = len(ssyk_scores)
    print(f"  High-exposure (Q4): {n_high}/{n_total} occupations")

    return ssyk_scores


# ── Step 4: Re-run DiD with Eloundou measure ────────────────────────────────

def prepare_eloundou_panel(
    merged: pd.DataFrame, eloundou_q: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Eloundou quartile classification into the existing posting panel
    and build the same treatment interactions as in 05_analysis.py.

    Drops occupation-months with zero postings or missing Eloundou scores.
    """
    df = merged.copy()

    # Ensure ssyk4 is string and zero-padded in both
    df["ssyk4"] = df["ssyk4"].astype(str).str.zfill(4)
    eloundou_q["ssyk4"] = eloundou_q["ssyk4"].astype(str).str.zfill(4)

    # Merge Eloundou classification
    df = df.merge(
        eloundou_q[["ssyk4", "dv_rating_beta", "high_exposure_eloundou"]],
        on="ssyk4",
        how="inner",
    )

    # Report match rate vs. full panel
    n_occ_full = merged["ssyk4"].nunique()
    n_occ_matched = df["ssyk4"].nunique()
    print(f"  Panel match: {n_occ_matched}/{n_occ_full} occupations "
          f"({100 * n_occ_matched / n_occ_full:.1f}%)")

    # Treatment dummies (same logic as 05_analysis.prepare_panel)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    rb_date = pd.Timestamp(RIKSBANKEN_HIKE)
    gpt_date = pd.Timestamp(CHATGPT_LAUNCH)

    df["post_riksbank"] = (df["date"] >= rb_date).astype(int)
    df["post_chatgpt"] = (df["date"] >= gpt_date).astype(int)

    # Interaction terms — using Eloundou high-exposure dummy
    df["post_rb_x_high"] = df["post_riksbank"] * df["high_exposure_eloundou"]
    df["post_gpt_x_high"] = df["post_chatgpt"] * df["high_exposure_eloundou"]

    # Log outcome
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])

    # Occupation-specific trend
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days / 30.0
    df["time_x_high"] = df["time_idx"] * df["high_exposure_eloundou"]

    # SSYK 1-digit group × month (for Spec 4)
    df["ssyk1"] = df["ssyk4"].astype(str).str[0]
    df["group_time"] = df["ssyk1"] + "_" + df["year_month"]

    print(f"  Panel: {len(df):,} obs, {df['ssyk4'].nunique()} occupations, "
          f"{df['year_month'].nunique()} months")

    return df


def run_eloundou_regressions(df: pd.DataFrame) -> dict:
    """
    Re-run the same four DiD specifications as 05_analysis.py, but with
    Eloundou β-exposure instead of DAIOE.

    Returns dict of PanelOLS results keyed by spec name.
    """
    from linearmodels.panel import PanelOLS

    panel = df.copy()
    panel["date"] = pd.to_datetime(panel["year_month"] + "-01")
    panel = panel.set_index(["ssyk4", "date"])

    results = {}

    # Spec 1: Monetary policy only
    print("  (1) Monetary policy interaction only...")
    mod1 = PanelOLS(
        dependent=panel["ln_ads"],
        exog=panel[["post_rb_x_high"]],
        entity_effects=True,
        time_effects=True,
    )
    res1 = mod1.fit(cov_type="clustered", cluster_entity=True)
    results["spec1"] = res1
    print(f"      β₁ = {res1.params['post_rb_x_high']:.4f} "
          f"(SE = {res1.std_errors['post_rb_x_high']:.4f})")

    # Spec 2: + ChatGPT
    print("  (2) + ChatGPT interaction...")
    mod2 = PanelOLS(
        dependent=panel["ln_ads"],
        exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
        entity_effects=True,
        time_effects=True,
    )
    res2 = mod2.fit(cov_type="clustered", cluster_entity=True)
    results["spec2"] = res2
    print(f"      β₁ = {res2.params['post_rb_x_high']:.4f} "
          f"(SE = {res2.std_errors['post_rb_x_high']:.4f})")
    print(f"      β₂ = {res2.params['post_gpt_x_high']:.4f} "
          f"(SE = {res2.std_errors['post_gpt_x_high']:.4f})")

    # Spec 3: + occupation-specific trends
    print("  (3) + occupation-specific trends...")
    mod3 = PanelOLS(
        dependent=panel["ln_ads"],
        exog=panel[["post_rb_x_high", "post_gpt_x_high", "time_x_high"]],
        entity_effects=True,
        time_effects=True,
    )
    res3 = mod3.fit(cov_type="clustered", cluster_entity=True)
    results["spec3"] = res3
    print(f"      β₁ = {res3.params['post_rb_x_high']:.4f} "
          f"(SE = {res3.std_errors['post_rb_x_high']:.4f})")
    print(f"      β₂ = {res3.params['post_gpt_x_high']:.4f} "
          f"(SE = {res3.std_errors['post_gpt_x_high']:.4f})")

    # Spec 4: SSYK 1-digit × month FE
    print("  (4) + SSYK 1-digit × month FE...")
    group_time_df = pd.DataFrame(
        {"group_time": panel["group_time"]},
        index=panel.index,
    )
    mod4 = PanelOLS(
        dependent=panel["ln_ads"],
        exog=panel[["post_rb_x_high", "post_gpt_x_high"]],
        entity_effects=True,
        time_effects=False,
        other_effects=group_time_df,
    )
    res4 = mod4.fit(cov_type="clustered", cluster_entity=True)
    results["spec4"] = res4
    print(f"      β₁ = {res4.params['post_rb_x_high']:.4f} "
          f"(SE = {res4.std_errors['post_rb_x_high']:.4f})")
    print(f"      β₂ = {res4.params['post_gpt_x_high']:.4f} "
          f"(SE = {res4.std_errors['post_gpt_x_high']:.4f})")

    return results


# ── Step 5: Format comparison table ─────────────────────────────────────────

def load_daioe_results() -> dict:
    """
    Re-run DAIOE regressions to get exact coefficients for comparison.

    Imports prepare_panel and run_did_regressions from 05_analysis.py.
    """
    # Import from the existing analysis script
    from importlib import import_module
    analysis = import_module("05_analysis")

    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    panel = analysis.prepare_panel(merged)
    results = analysis.run_did_regressions(panel)

    return results, panel


def format_comparison_table(
    daioe_res: dict, eloundou_res: dict,
    daioe_panel: pd.DataFrame, eloundou_panel: pd.DataFrame,
) -> str:
    """
    Side-by-side LaTeX table: DAIOE vs Eloundou for all four specifications.

    Reports β₁ (Post-Riksbank × High) and β₂ (Post-ChatGPT × High) with
    clustered SEs for Specs 2–4 using each measure.
    """

    def stars(pval):
        if pval < 0.01:
            return "***"
        elif pval < 0.05:
            return "**"
        elif pval < 0.10:
            return "*"
        return ""

    def coef_cell(res, var):
        """Format coefficient + SE as two lines for a LaTeX cell."""
        if res is None or var not in res.params.index:
            return " & ", " & "
        c = res.params[var]
        se = res.std_errors[var]
        pv = res.pvalues[var]
        return f" & {c:.3f}{stars(pv)}", f" & ({se:.3f})"

    # We show Specs 2–4 for both measures (6 columns)
    # Spec 1 is redundant with Spec 2 for the comparison purpose
    specs = ["spec2", "spec3", "spec4"]
    spec_labels = ["Baseline", "+ Trends", r"Group$\times$Time"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Robustness: DAIOE vs.\ Eloundou et al.\ (2023) GPT exposure}",
        r"\label{tab:eloundou_robust}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\hline\hline",
        r" & \multicolumn{3}{c}{DAIOE} & \multicolumn{3}{c}{Eloundou $\beta$} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
    ]

    # Column headers
    header = " "
    for label in spec_labels:
        header += f" & {label}"
    for label in spec_labels:
        header += f" & {label}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\hline")

    # Row: Post-Riksbank × High
    row_c = r"Post-Riksbank $\times$ High"
    row_s = ""
    for res_dict in [daioe_res, eloundou_res]:
        for spec in specs:
            c, s = coef_cell(res_dict.get(spec), "post_rb_x_high")
            row_c += c
            row_s += s
    lines.append(row_c + r" \\")
    lines.append(row_s + r" \\[3pt]")

    # Row: Post-ChatGPT × High
    row_c = r"Post-ChatGPT $\times$ High"
    row_s = ""
    for res_dict in [daioe_res, eloundou_res]:
        for spec in specs:
            c, s = coef_cell(res_dict.get(spec), "post_gpt_x_high")
            row_c += c
            row_s += s
    lines.append(row_c + r" \\")
    lines.append(row_s + r" \\")

    # Footer
    # N observations
    d_nobs = daioe_res["spec2"].nobs if "spec2" in daioe_res else "—"
    e_nobs = eloundou_res["spec2"].nobs if "spec2" in eloundou_res else "—"

    d_nclusters = int(getattr(daioe_res.get("spec2", None), "entity_info", pd.Series({"total": 0})).total) if "spec2" in daioe_res else "—"
    e_nclusters = int(getattr(eloundou_res.get("spec2", None), "entity_info", pd.Series({"total": 0})).total) if "spec2" in eloundou_res else "—"

    lines.extend([
        r"\hline",
        r"Occupation FE & Yes & Yes & Yes & Yes & Yes & Yes \\",
        r"Month FE & Yes & Yes & & Yes & Yes & \\",
        r"Occ.\ group $\times$ month FE & & & Yes & & & Yes \\",
        f"Occupations & \\multicolumn{{3}}{{c}}{{{d_nclusters}}} "
        f"& \\multicolumn{{3}}{{c}}{{{e_nclusters}}} \\\\",
        f"Observations & \\multicolumn{{3}}{{c}}{{{d_nobs:,}}} "
        f"& \\multicolumn{{3}}{{c}}{{{e_nobs:,}}} \\\\",
        r"\hline\hline",
        r"\multicolumn{7}{p{0.95\textwidth}}{\footnotesize \textit{Notes:} "
        r"Dependent variable: $\ln(\text{postings}_{it})$. "
        r"``High'' = top quartile of the respective exposure measure. "
        r"DAIOE: own genAI exposure index (percentile ranking). "
        r"Eloundou $\beta$: GPT exposure score from Eloundou et al.\ (2023), "
        r"crosswalked via SOC$\to$ISCO$\to$SSYK. "
        r"Standard errors clustered at occupation level in parentheses. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def save_results_csv(
    daioe_res: dict, eloundou_res: dict
) -> pd.DataFrame:
    """
    Machine-readable comparison: one row per coefficient × specification × measure.
    """
    rows = []
    for measure, res_dict in [("DAIOE", daioe_res), ("Eloundou", eloundou_res)]:
        for spec_key in ["spec1", "spec2", "spec3", "spec4"]:
            res = res_dict.get(spec_key)
            if res is None:
                continue
            for var in res.params.index:
                rows.append({
                    "measure": measure,
                    "specification": spec_key,
                    "variable": var,
                    "coefficient": res.params[var],
                    "std_error": res.std_errors[var],
                    "t_stat": res.tstats[var],
                    "p_value": res.pvalues[var],
                    "n_obs": res.nobs,
                })

    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ROBUSTNESS: Eloundou et al. (2023) GPT exposure")
    print("=" * 70)

    # Step 1: Download / load Eloundou data
    print("\n--- Step 1: Eloundou data ---")
    eloundou = download_eloundou()

    # Step 2: Build crosswalk SOC → ISCO → SSYK
    print("\n--- Step 2: Crosswalk SOC → ISCO → SSYK ---")
    ssyk_scores = build_eloundou_ssyk(eloundou)

    # Step 3: Assign quartiles
    print("\n--- Step 3: Quartile assignment ---")
    ssyk_scores = assign_eloundou_quartiles(ssyk_scores)

    # Save crosswalk result
    out_xw = PROCESSED / "eloundou_ssyk_matched.csv"
    ssyk_scores.to_csv(out_xw, index=False)
    print(f"  Saved → {out_xw.name}")

    # Compare with DAIOE high-exposure assignments
    daioe_q = pd.read_csv(PROCESSED / "daioe_quartiles.csv")
    daioe_q["ssyk4"] = daioe_q["ssyk4"].astype(str).str.zfill(4)
    both = daioe_q.merge(ssyk_scores, on="ssyk4", how="inner")
    agreement = (both["high_exposure"] == both["high_exposure_eloundou"]).mean()
    print(f"\n  DAIOE vs Eloundou quartile agreement: {agreement:.1%}")
    print(f"  Both high: {((both['high_exposure'] == 1) & (both['high_exposure_eloundou'] == 1)).sum()}")
    print(f"  DAIOE high only: {((both['high_exposure'] == 1) & (both['high_exposure_eloundou'] == 0)).sum()}")
    print(f"  Eloundou high only: {((both['high_exposure'] == 0) & (both['high_exposure_eloundou'] == 1)).sum()}")

    # Step 4: Re-run DiD
    print("\n--- Step 4a: Eloundou DiD regressions ---")
    merged = pd.read_csv(PROCESSED / "postings_daioe_merged.csv")
    el_panel = prepare_eloundou_panel(merged, ssyk_scores)
    el_results = run_eloundou_regressions(el_panel)

    print("\n--- Step 4b: DAIOE DiD regressions (for comparison) ---")
    daioe_results, daioe_panel = load_daioe_results()

    # Step 5: Output
    print("\n--- Step 5: Comparison table ---")
    table_tex = format_comparison_table(
        daioe_results, el_results, daioe_panel, el_panel
    )
    tex_path = TABDIR / "eloundou_robustness.tex"
    tex_path.write_text(table_tex, encoding="utf-8")
    print(f"  Saved → {tex_path.name}")

    csv_df = save_results_csv(daioe_results, el_results)
    csv_path = TABDIR / "eloundou_robustness.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path.name}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Eloundou SSYK match: {len(ssyk_scores)} occupations")
    print(f"Panel match rate: {el_panel['ssyk4'].nunique()}/{merged['ssyk4'].nunique()} occupations")
    print(f"Quartile agreement with DAIOE: {agreement:.1%}")

    print("\nKey coefficients (Spec 2 — baseline DiD):")
    for label, res in [("DAIOE", daioe_results.get("spec2")),
                       ("Eloundou", el_results.get("spec2"))]:
        if res is None:
            continue
        b1 = res.params["post_rb_x_high"]
        b2 = res.params["post_gpt_x_high"]
        p1 = res.pvalues["post_rb_x_high"]
        p2 = res.pvalues["post_gpt_x_high"]
        print(f"  {label:10s}  β₁={b1:+.3f} (p={p1:.3f})  "
              f"β₂={b2:+.3f} (p={p2:.3f})")

    print("\nDone. Include tables/eloundou_robustness.tex in the Online Appendix.")


if __name__ == "__main__":
    main()

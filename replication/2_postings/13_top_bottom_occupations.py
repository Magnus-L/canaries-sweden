#!/usr/bin/env python3
"""
13_top_bottom_occupations.py: the ten most and the ten least exposed
occupations under the DAIOE generative-AI index.

WHAT IT BUILDS
The 2023 cross-section of DAIOE, ranked by the generative-AI percentile; the
top ten and bottom ten four-digit SSYK 2012 occupations with their
percentiles. Names are the English ISCO-08 titles of the twenty codes, and
the Swedish title from the DAIOE file for any other code.

INPUTS   data/raw/daioe_ssyk2012.csv
OUTPUTS  output/tables/top_bottom_occupations.tex;
         output/results/top_bottom_occupations.csv
SERVES   Online Appendix I.1, Table A1
RUNTIME  seconds
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def table_top_bottom_occupations():
    """List the 10 most and 10 least genAI-exposed occupations."""
    print("The ten most and ten least exposed occupations")

    # DAIOE source file has Swedish names; map to English via ISCO-08 for publication
    SSYK_ENGLISH = {
        "2641": "Authors and related writers",
        "2122": "Statisticians",
        "2121": "Mathematicians and actuaries",
        "2415": "Economists",
        "2512": "Software and systems developers",
        "2145": "Chemical engineers",
        "2111": "Physicists and astronomers",
        "2414": "Securities traders and fund managers",
        "2513": "Game and digital media developers",
        "2112": "Meteorologists",
        "9120": "Vehicle, window and related cleaners",
        "2653": "Dancers and choreographers",
        "8350": "Ships' deck crew and related workers",
        "7113": "Concrete workers",
        "8341": "Agricultural and forestry machinery operators",
        "9310": "Construction labourers",
        "8342": "Earth-moving machinery operators",
        "8111": "Miners and quarry workers",
        "7121": "Roofers",
        "3421": "Professional athletes",
    }

    daioe_raw = pd.read_csv(config.DAIOE_RAW, sep="\t")
    daioe_ref = daioe_raw[daioe_raw["year"] == config.DAIOE_REF_YEAR].copy()
    daioe_ref["ssyk4"] = daioe_ref["ssyk2012_4"].str[:4]
    # Use English name if available, fall back to Swedish from source
    daioe_ref["occupation_name"] = daioe_ref["ssyk4"].map(SSYK_ENGLISH).fillna(
        daioe_ref["ssyk2012_4"].str[5:]
    )

    daioe_ref = daioe_ref.dropna(subset=["pctl_rank_genai"])
    daioe_ref = daioe_ref.sort_values("pctl_rank_genai", ascending=False)

    top10 = daioe_ref.head(10)[["ssyk4", "occupation_name", "pctl_rank_genai"]]
    bottom10 = daioe_ref.tail(10)[["ssyk4", "occupation_name", "pctl_rank_genai"]]

    combined = pd.concat([
        top10.assign(group="Most exposed"),
        bottom10.assign(group="Least exposed"),
    ])

    out = config.RESULTS / "top_bottom_occupations.csv"
    combined.to_csv(out, index=False)
    print(f"  Saved → {out.name}")

    # LaTeX version
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Most and least genAI-exposed occupations (DAIOE)}",
        r"\label{tab:topbottom}",
        r"\begin{tabular}{clc}",
        r"\hline\hline",
        r"SSYK & Occupation & GenAI pctl \\",
        r"\hline",
        r"\multicolumn{3}{l}{\textit{Most exposed (top 10)}} \\",
    ]

    for _, row in top10.iterrows():
        name = row["occupation_name"]
        lines.append(f"{row['ssyk4']} & {name} & {row['pctl_rank_genai']:.1f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{3}{l}{\textit{Least exposed (bottom 10)}} \\")

    for _, row in bottom10.iterrows():
        name = row["occupation_name"]
        lines.append(f"{row['ssyk4']} & {name} & {row['pctl_rank_genai']:.1f} \\\\")

    lines.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_out = config.TABLES / "top_bottom_occupations.tex"
    tex_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {tex_out.name}")


if __name__ == "__main__":
    table_top_bottom_occupations()

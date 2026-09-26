#!/usr/bin/env python3
"""
03_exposure_measure_agreement.py: how far the DAIOE generative-AI index and
the GPT-exposure score of Eloundou et al. (2024) agree on which occupations
are exposed, in the employment of 2024.

WHAT IT BUILDS
The employment design ranks employers by the DAIOE exposure of their 2019
occupation mix and treats the top quartile as exposed. Online Appendix III.2
re-estimates the age profile with the Eloundou score in place of DAIOE, and
the data section states how alike the two rankings are. This script gives
those numbers from public inputs: across the four-digit SSYK 2012
occupations both measures score, the correlation of the two scores, and,
weighting each occupation by its employees in Statistics Sweden's
occupational register (table YREG54BAS, 2024), the share of employment whose
occupation falls on the same side of the top-quartile cut on both measures
and the share of the employment DAIOE places in the top quartile that the
Eloundou score places there too. The correlation is checked against the one
the register run computed inside MONA on the same two files (script 63,
measure_correlation.csv), and nothing is written if they differ.

The agreement is between occupation rankings. The employer classification
itself, which averages the score over an employer's incumbents, exists only
inside MONA and was not rebuilt on the Eloundou score.

INPUTS   3_register_mona/inputs/daioe_quartiles.dta and eloundou_ssyk4.dta,
         the two score files the register scripts read;
         data/raw/scb_yreg54bas_ssyk4_age_sex_2024.json, the saved response
         of the Statistics Sweden API (see 02_occupation_mix_by_sex.py);
         3_register_mona/exports/2026-09-20_2148_s61-s63/
         output_63__measure_correlation.csv
OUTPUTS  output/results/exposure_measure_agreement.csv,
         exposure_measure_agreement.txt
SERVES   Online Appendix I, the data section's sentence on the two measures
         (0.87; 393 occupations; 83 and 73 per cent)
RUNTIME  seconds

    python 5_occupation_register_public/03_exposure_measure_agreement.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

INPUTS = config.PACKAGE / "3_register_mona" / "inputs"
RAW = config.RAW / "scb_yreg54bas_ssyk4_age_sex_2024.json"
S63 = (config.EXPORTS / "2026-09-20_2148_s61-s63"
       / "output_63__measure_correlation.csv")
YOUNGEST = "16-24"   # the register's first band, below the design's incumbents


def load_register() -> pd.DataFrame:
    """Employees by occupation, age band and sex, 2024, from the saved
    API response (the same reader as 02_occupation_mix_by_sex.py)."""
    d = json.loads(RAW.read_text(encoding="utf-8"))
    occ = list(d["dimension"]["Yrke2012"]["category"]["index"].keys())
    age = list(d["dimension"]["Alder"]["category"]["index"].keys())
    sex = list(d["dimension"]["Kon"]["category"]["index"].keys())
    v, rows, k = d["value"], [], 0
    for o in occ:
        for a in age:
            for s in sex:
                rows.append((o, a, s, v[k]))
                k += 1
    e = pd.DataFrame(rows, columns=["ssyk4", "age", "sex", "n"])
    e["n"] = pd.to_numeric(e["n"], errors="coerce").fillna(0.0)
    e["ssyk4"] = e["ssyk4"].astype(str).str.zfill(4)
    return e


def scores() -> pd.DataFrame:
    d = pd.read_stata(INPUTS / "daioe_quartiles.dta")
    e = pd.read_stata(INPUTS / "eloundou_ssyk4.dta")
    for t in (d, e):
        t["ssyk4"] = t["ssyk4"].astype(int).astype(str).str.zfill(4)
    m = d[["ssyk4", "pctl_rank_genai", "high_exposure"]].merge(
        e[["ssyk4", "eloundou_score", "high_exposure_eloundou"]], on="ssyk4")
    m["high_exposure"] = m["high_exposure"].astype(int)
    m["high_exposure_eloundou"] = m["high_exposure_eloundou"].astype(int)
    return m, len(d), len(e)


def weighted(m: pd.DataFrame, w: pd.Series, label: str) -> dict:
    """Employment-weighted agreement on the top-quartile classification."""
    emp = m["ssyk4"].map(w).fillna(0.0)
    total = float(emp.sum())
    same = float(emp[m.high_exposure == m.high_exposure_eloundou].sum())
    top_d = float(emp[m.high_exposure == 1].sum())
    top_e = float(emp[m.high_exposure_eloundou == 1].sum())
    both = float(emp[(m.high_exposure == 1) & (m.high_exposure_eloundou == 1)].sum())
    return {"sample": label, "employment": total, "same_class": same / total,
            "daioe_top": top_d / total, "eloundou_top": top_e / total,
            "both_top": both / total, "daioe_top_also_eloundou_top": both / top_d}


def main() -> int:
    m, n_d, n_e = scores()
    r = float(m.pctl_rank_genai.corr(m.eloundou_score))
    rho = float(m.pctl_rank_genai.corr(m.eloundou_score, method="spearman"))
    print(f"  occupations scored by both measures: {len(m)} "
          f"(DAIOE {n_d}, Eloundou {n_e}); Pearson {r:.3f}, Spearman {rho:.3f}")

    # The register run computed the same correlation on the same two files.
    s63 = pd.read_csv(S63)
    row = s63[(s63.a == "daioe") & (s63.b == "eloundou")]
    if len(row) != 1 or int(row.n_occ.iloc[0]) != len(m) \
            or abs(float(row["corr"].iloc[0]) - r) > 5e-4:
        raise SystemExit(f"  the correlation {r:.4f} on {len(m)} occupations does not "
                         f"reproduce the register run's {row.to_dict('records')}; "
                         "nothing is written")
    print("  reproduces the correlation the register run exported (script 63)")

    hd, he = m.high_exposure, m.high_exposure_eloundou
    unweighted = {"sample": "occupations, unweighted", "employment": float(len(m)),
                  "same_class": float((hd == he).mean()),
                  "daioe_top": float(hd.mean()), "eloundou_top": float(he.mean()),
                  "both_top": float((hd & he).mean()),
                  "daioe_top_also_eloundou_top": float((hd & he).sum() / hd.sum())}

    e = load_register()
    w_all = e.groupby("ssyk4")["n"].sum()
    w_inc = e[e.age != YOUNGEST].groupby("ssyk4")["n"].sum()
    rows = [unweighted,
            weighted(m, w_all, "employees 16-64, 2024"),
            weighted(m, w_inc, "employees 25-64, 2024")]
    out = pd.DataFrame(rows)
    out.insert(0, "n_occupations", len(m))
    out.insert(1, "pearson_r", r)
    out.insert(2, "spearman_rho", rho)
    for k in rows:
        print(f"  {k['sample']:28s} same class {k['same_class']:.3f}; "
              f"DAIOE top {k['daioe_top']:.3f}, Eloundou top {k['eloundou_top']:.3f}, "
              f"both {k['both_top']:.3f}; of DAIOE-top also Eloundou-top "
              f"{k['daioe_top_also_eloundou_top']:.3f}")

    emp = m["ssyk4"].map(w_all).fillna(0.0)
    dis = m.assign(employees_2024=emp)[hd != he].sort_values(
        "employees_2024", ascending=False).head(8)

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    csv_out = config.RESULTS / "exposure_measure_agreement.csv"
    out.to_csv(csv_out, index=False)
    txt = [
        "DAIOE AND THE ELOUNDOU SCORE: AGREEMENT ON WHICH OCCUPATIONS ARE EXPOSED",
        "=" * 72, "",
        f"occupations scored by both: {len(m)} (DAIOE {n_d}, Eloundou {n_e})",
        f"Pearson r {r:.3f} (equals the register run's, script 63); Spearman {rho:.3f}",
        "", "agreement on the top-quartile cut (each measure's own quartile):"]
    for k in rows:
        txt.append(f"  {k['sample']:28s} same class {k['same_class']:.3f}; of DAIOE-top "
                   f"employment also Eloundou-top {k['daioe_top_also_eloundou_top']:.3f}")
    txt += ["", "largest occupations on which the two cuts disagree (employees 2024):",
            dis[["ssyk4", "pctl_rank_genai", "eloundou_score", "high_exposure",
                 "high_exposure_eloundou", "employees_2024"]].to_string(index=False),
            "", "The employer classification averages the score over an employer's 2019",
            "incumbents inside MONA and was not rebuilt on the Eloundou score; the",
            "age profile on the Eloundou score is OA Table A18 (script 63)."]
    (config.RESULTS / "exposure_measure_agreement.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8")
    print(f"\n  wrote {csv_out.name} and exposure_measure_agreement.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

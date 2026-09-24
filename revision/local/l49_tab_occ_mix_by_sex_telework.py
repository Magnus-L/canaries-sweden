#!/usr/bin/env python3
"""
l49_tab_occ_mix_by_sex_telework.py: l40's table priced a second time,
with teleworkability beside generative-AI exposure.

WHY THIS EXISTS
The triple difference nets out older colleagues and anything common to
an employer, including its remote-work policy. It cannot net out young
women and young men holding different occupations inside the same
employer. A return-to-office or remote-work shock that hits teleworkable
jobs would therefore produce a sex gap if young women hold the more
teleworkable jobs. l40 shows, from public statistics, that women's
occupations are the more AI-exposed at every age. This script asks the
same national distribution the rival question: are women's occupations
also the more teleworkable, and by how much compared with the AI gap?

WHAT IS REUSED, UNCHANGED
- Employment by four-digit SSYK 2012, age band and sex, 2024: the saved
  YREG54BAS response l40 reads, loaded with l40's own loader.
- Exposure: `pctl_rank_genai` from revision/upload/daioe_quartiles.dta,
  as in l40.
- Teleworkability: revision/upload/dingel_neiman_ssyk4.dta, the file the
  MONA scripts 46 and 89 (OA II.3) read. It is Dingel and Neiman (2020)
  crosswalked SOC -> ISCO-08 -> SSYK 2012 with unweighted means at each
  step; 423 codes, all of them in the DAIOE file. No new crosswalk is
  built here. Dingel and Neiman classify each O*NET occupation 0 or 1;
  the SSYK score is therefore the share of the US occupations mapped to
  it that can be done entirely at home, shown in per cent.

MAKING THE TWO GAPS COMPARABLE
The two scores are in different units (a percentile rank and a share),
so each women-minus-men gap is also divided by the standard deviation of
its score across occupations, weighted by 2024 employment of all
employees aged 16 to 64 (the weighting l40 uses for its means). A gap of
0.2 then means the same thing on both scores: women's occupational mix
sits a fifth of a cross-occupation standard deviation above men's. The
CSV also carries the gap scaled by the standard deviation within each
age band's own employment, as a check that the scaling is not driving the
comparison.

The employment-weighted correlation of the two scores across occupations
is reported for each band (weights: that band's employees of both sexes).

INPUTS AND OUTPUTS
Reads the files above. Writes revision/tables/occ_mix_by_sex_telework.csv
and revision/tables/tableA_occ_mix_by_sex_telework.tex, and copies the
.tex to canaries-sweden-paper/tables/. No register microdata; public
aggregates only.

    python3 revision/local/l49_tab_occ_mix_by_sex_telework.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

# l40's loader, so the employment counts are read exactly as there.
_spec = importlib.util.spec_from_file_location(
    "l40", Path(__file__).with_name("l40_tab_occ_mix_by_sex.py"))
l40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l40)

DAIOE = REV / "upload" / "daioe_quartiles.dta"
DN = REV / "upload" / "dingel_neiman_ssyk4.dta"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"
SHOW = l40.SHOW  # 16-24, 25-29, 30-34, 40-44, 50-54


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def wsd(x, w):
    mu = wmean(x, w)
    return float(np.sqrt(((x - mu) ** 2 * w).sum() / w.sum()))


def wcorr(x, y, w):
    mx, my = wmean(x, w), wmean(y, w)
    cov = ((x - mx) * (y - my) * w).sum() / w.sum()
    return float(cov / (wsd(x, w) * wsd(y, w)))


def main() -> int:
    e = l40.load()
    d = pd.read_stata(DAIOE)[["ssyk4", "pctl_rank_genai"]]
    t = pd.read_stata(DN)[["ssyk4", "teleworkable"]]
    for f in (d, t):
        f["ssyk4"] = f["ssyk4"].astype(str).str.zfill(4)
    t["tw"] = t["teleworkable"] * 100  # per cent of the occupation teleworkable
    m = e.merge(d, on="ssyk4", how="left").merge(t[["ssyk4", "tw"]],
                                                  on="ssyk4", how="left")
    total = m["n"].sum()
    m = m[m["pctl_rank_genai"].notna() & m["tw"].notna()]
    priced = m["n"].sum()
    print(f"  employees 2024: {total:,.0f}; priced by BOTH scores: "
          f"{priced:,.0f} ({priced / total:.1%})")

    # Occupation-level scores and 16-64 employment weights (both sexes).
    occ = (m.groupby("ssyk4")
            .agg(n=("n", "sum"), ai=("pctl_rank_genai", "first"),
                 tw=("tw", "first")))
    occ = occ[occ["n"] > 0]
    sd_ai, sd_tw = wsd(occ["ai"], occ["n"]), wsd(occ["tw"], occ["n"])
    r_all = wcorr(occ["ai"], occ["tw"], occ["n"])
    print(f"  {len(occ)} occupations; employment-weighted SD: AI {sd_ai:.2f} "
          f"percentile points, telework {sd_tw:.2f} pp; corr {r_all:.3f}")

    rows = []
    for a in l40.AGES + ["All 16-64"]:
        sub = m if a == "All 16-64" else m[m.age == a]
        W, M = sub[sub.sex == "women"], sub[sub.sex == "men"]
        band = sub.groupby("ssyk4").agg(n=("n", "sum"),
                                        ai=("pctl_rank_genai", "first"),
                                        tw=("tw", "first"))
        band = band[band["n"] > 0]
        r = dict(age=a, n_women=W["n"].sum(), n_men=M["n"].sum(),
                 ai_women=wmean(W["pctl_rank_genai"], W["n"]),
                 ai_men=wmean(M["pctl_rank_genai"], M["n"]),
                 tw_women=wmean(W["tw"], W["n"]),
                 tw_men=wmean(M["tw"], M["n"]),
                 corr_ai_tw=wcorr(band["ai"], band["tw"], band["n"]),
                 sd_ai_band=wsd(band["ai"], band["n"]),
                 sd_tw_band=wsd(band["tw"], band["n"]))
        r["ai_gap"] = r["ai_women"] - r["ai_men"]
        r["tw_gap"] = r["tw_women"] - r["tw_men"]
        r["ai_gap_sd"] = r["ai_gap"] / sd_ai
        r["tw_gap_sd"] = r["tw_gap"] / sd_tw
        r["ai_gap_sd_band"] = r["ai_gap"] / r["sd_ai_band"]
        r["tw_gap_sd_band"] = r["tw_gap"] / r["sd_tw_band"]
        r["sd_ai_1664"], r["sd_tw_1664"] = sd_ai, sd_tw
        rows.append(r)
    out = pd.DataFrame(rows)
    csv = V2_TAB / "occ_mix_by_sex_telework.csv"
    out.round(4).to_csv(csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out[["age", "ai_women", "ai_men", "ai_gap", "ai_gap_sd",
                   "tw_women", "tw_men", "tw_gap", "tw_gap_sd",
                   "ai_gap_sd_band", "tw_gap_sd_band",
                   "corr_ai_tw"]].round(3).to_string(index=False))

    def line(r, label):
        return (f"{label} & {r.ai_women:.1f} & {r.ai_men:.1f} & "
                f"{r.ai_gap:.1f} & {r.ai_gap_sd:.2f} & "
                f"{r.tw_women:.1f} & {r.tw_men:.1f} & {r.tw_gap:.1f} & "
                f"{r.tw_gap_sd:.2f} & {r.corr_ai_tw:.2f} \\\\")

    idx = out.set_index("age")
    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{Generative-AI exposure and teleworkability of the "
           r"occupations of Swedish women and men, by age, 2024}",
           r"\label{tab:occ_mix_by_sex_telework}", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lccccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Exposure percentile} & "
           r"\multicolumn{4}{c}{Teleworkable (\%)} & \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"Age & Women & Men & Gap & Gap (SD) & Women & Men & Gap & "
           r"Gap (SD) & Corr. \\", r"\midrule"]
    for a in SHOW:
        tex.append(line(idx.loc[a], a.replace("-", "--")))
    tex += [r"\midrule", line(idx.loc["All 16-64"], "All 16--64")]
    note = (
        r"Employees aged 16 to 64 in Sweden in 2024, by four-digit SSYK 2012 "
        r"occupation, age and sex, from Statistics Sweden's occupational "
        rf"register (table YREG54BAS); {priced:,.0f} of {total:,.0f} "
        r"employees hold an occupation both scores price, the same employees "
        r"as in Table~\ref{tab:occ_mix_by_sex}. The exposure percentile is the "
        r"DAIOE generative-AI index the employment design uses. Teleworkable "
        r"is the \citet{dingel2020many} classification on the route of "
        r"Section~\ref{sec:posting_rivals}: the share of the US occupations "
        r"mapped to a Swedish occupation that can be done entirely at home. Each level is the employment-weighted mean over the "
        r"group's occupational distribution, and each gap is women minus men. "
        r"Gap (SD) divides the gap by the standard deviation of the score "
        r"across occupations, weighted by the employment of all employees "
        rf"aged 16 to 64 ({sd_ai:.1f} percentile points and {sd_tw:.1f} "
        r"percentage points), so the two gaps are in the same units. Corr. is "
        r"the correlation of the two scores across occupations, weighted by "
        r"the age band's employment. The paper's young band, 22 to 25, "
        r"straddles the register's first two bands. Like "
        r"Table~\ref{tab:occ_mix_by_sex}, this is the national distribution "
        r"and states a difference in the occupations the two sexes hold; it "
        r"does not decompose the estimated differential."
    )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.95\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    tf = V2_TAB / "tableA_occ_mix_by_sex_telework.tex"
    tf.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {csv.relative_to(REV)} and {tf.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(tf, PAPER_TAB / tf.name)
        print(f"  copied to {PAPER_TAB / tf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
l40_tab_occ_mix_by_sex.py: how exposed to generative AI the occupations
of Swedish women and men are, by age, from published statistics.

WHY THIS EXISTS
The paper finds the decline concentrated on young women and shows that
three quarters of the differential survives within broad education
tracks. What it cannot show is the other composition channel: whether
young women and young men at the same employer hold DIFFERENT
OCCUPATIONS, and whether women's are the more exposed. That would need a
current occupation code on every young worker, which is the construction
Part~IV closes.

What can be done without it is to ask the same question of the national
workforce, from statistics Statistics Sweden publishes. The occupational
register gives employment by four-digit SSYK, age and sex; the DAIOE
generative-AI percentile prices each occupation; the employment-weighted
mean of the second over the first is how exposed each group's
occupational mix is. It is a national distribution and not a within-firm
one, so it supports a hypothesis and settles nothing, which is how the
paper reports it.

THE EXPOSURE INDEX IS THE PAPER'S. `pctl_rank_genai` from
daioe_quartiles.dta, the same file the register scripts score from, and
the top quartile is that file's own `high_exposure`. The all-applications
index in the same file is NOT used: every exposure statement in the paper
is on the generative-AI measure and mixing the two would put a different
treatment under one name.

INPUTS AND OUTPUTS
Reads data/raw/scb_yreg54bas_ssyk4_age_sex_2024.json, the saved response
to the query in scb_yreg54bas_query.json (Statistics Sweden's
Statistikdatabasen, table YREG54BAS, employees by four-digit SSYK 2012,
industry, age and sex, 2024), and revision/upload/daioe_quartiles.dta.
With --refresh it re-queries the API and overwrites the saved response.
Writes revision/tables/tableA_occ_mix_by_sex.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l40_tab_occ_mix_by_sex.py [--refresh]

IN THE PAPER
Section 3, one sentence on occupational composition; Online Appendix
III.2, the table and what it can and cannot say.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

RAW = REV.parent / "data" / "raw" / "scb_yreg54bas_ssyk4_age_sex_2024.json"
QUERY = REV.parent / "data" / "raw" / "scb_yreg54bas_query.json"
DAIOE = REV / "upload" / "daioe_quartiles.dta"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"
API = ("https://api.scb.se/OV0104/v1/doris/sv/ssd/AM/AM0208/AM0208E/"
       "YREG54BAS")
# The bands the register publishes. The paper's young band, 22 to 25,
# straddles the first two, which the text says rather than hides.
AGES = ["16-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64"]
SHOW = ["16-24", "25-29", "30-34", "40-44", "50-54"]


def refresh() -> None:
    """Re-query the API. Kept behind a flag: the saved response is what
    the table is built from, so a rebuild cannot silently move with a
    revision of the source."""
    import urllib.request
    req = urllib.request.Request(
        API, data=QUERY.read_bytes(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "AI-Econ Lab research (mlodefalk@gmail.com)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        RAW.write_bytes(r.read())
    print(f"  refreshed {RAW.name}")


def load() -> pd.DataFrame:
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
    e["sex"] = e["sex"].map({"1": "men", "2": "women"})
    e["ssyk4"] = e["ssyk4"].astype(str).str.zfill(4)
    return e


def main() -> int:
    if "--refresh" in sys.argv:
        refresh()
    e = load()
    d = pd.read_stata(DAIOE)
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    m = e.merge(d[["ssyk4", "pctl_rank_genai", "high_exposure"]],
                on="ssyk4", how="left")
    total, matched = m["n"].sum(), m.loc[m["pctl_rank_genai"].notna(), "n"].sum()
    m = m[m["pctl_rank_genai"].notna()]
    print(f"  employees 2024: {total:,.0f}; priced by DAIOE: {matched:,.0f} "
          f"({matched / total:.1%})")

    def stat(sub):
        w = sub["n"]
        if w.sum() <= 0:
            return None
        return (float((sub["pctl_rank_genai"] * w).sum() / w.sum()),
                float((sub["high_exposure"] * w).sum() / w.sum()),
                float(w.sum()))

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{How exposed to generative AI the occupations of "
           r"Swedish women and men are, by age, 2024}",
           r"\label{tab:occ_mix_by_sex}", r"\footnotesize",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Mean exposure percentile} & "
           r"\multicolumn{2}{c}{In the top quartile (\%)} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Age & Women & Men & Women & Men \\", r"\midrule"]
    for a in SHOW:
        w_, m_ = stat(m[(m.age == a) & (m.sex == "women")]), \
            stat(m[(m.age == a) & (m.sex == "men")])
        if not w_ or not m_:
            continue
        tex.append(f"{a.replace('-', '--')} & {w_[0]:.1f} & {m_[0]:.1f} & "
                   f"{w_[1] * 100:.1f} & {m_[1] * 100:.1f} \\\\")
        print(f"  {a:6s} women {w_[0]:5.1f} ({w_[1]*100:4.1f}%)  "
              f"men {m_[0]:5.1f} ({m_[1]*100:4.1f}%)  "
              f"gap {w_[0]-m_[0]:+.1f} pts")
    w_, m_ = stat(m[m.sex == "women"]), stat(m[m.sex == "men"])
    tex.append(r"\midrule")
    tex.append(f"All 16--64 & {w_[0]:.1f} & {m_[0]:.1f} & "
               f"{w_[1] * 100:.1f} & {m_[1] * 100:.1f} \\\\")
    print(f"  all    women {w_[0]:5.1f} ({w_[1]*100:4.1f}%)  "
          f"men {m_[0]:5.1f} ({m_[1]*100:4.1f}%)  gap {w_[0]-m_[0]:+.1f} pts")
    note = (
        r"Employees aged 16 to 64 in Sweden in 2024, by four-digit SSYK 2012 "
        r"occupation, age and sex, from Statistics Sweden's occupational "
        rf"register (table YREG54BAS); {matched:,.0f} of {total:,.0f} "
        r"employees hold an occupation the DAIOE generative-AI index prices. "
        r"The exposure percentile is that index, the same one the employment "
        r"design uses, and the top quartile is its own. Each column is the "
        r"employment-weighted mean over the group's occupational "
        r"distribution. This is the NATIONAL distribution: part of the gap it "
        r"shows is women and men working at different employers, which the "
        r"within-employer design absorbs, and the table cannot say how much "
        r"of the rest operates inside a firm. It states a difference in the "
        r"occupations the two sexes hold; it does not decompose the estimated "
        r"differential."
    ).replace(",", "{,}", 0)
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    V2_TAB.mkdir(parents=True, exist_ok=True)
    out = V2_TAB / "tableA_occ_mix_by_sex.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

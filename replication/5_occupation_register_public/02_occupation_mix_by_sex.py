#!/usr/bin/env python3
"""
02_occupation_mix_by_sex.py: how exposed to generative AI the occupations of
Swedish women and men are, by age, from published statistics.

WHAT IT BUILDS
Statistics Sweden's occupational register gives employees by four-digit SSYK
2012 occupation, age band and sex (table YREG54BAS, 2024). Each occupation is
priced by the DAIOE generative-AI percentile, the same index and the same
top quartile the employment design uses (daioe_quartiles.dta, the file the
register scripts read), and each group's exposure is the employment-weighted
mean over its occupations. It is a national distribution, not a
within-employer one: part of any gap is women and men working at different
employers, which the within-employer design absorbs.

INPUTS   data/raw/scb_yreg54bas_ssyk4_age_sex_2024.json, the saved response
         (fetched on 23 September 2026) to the query in
         scb_yreg54bas_query.json; --refresh queries the API again.
         3_register_mona/inputs/daioe_quartiles.dta
OUTPUTS  output/tables/tableA_occ_mix_by_sex.tex
SERVES   Online Appendix III.2, Table A17; the sentence of Section 3 on the
         occupations young women and men hold
RUNTIME  seconds

    python 5_occupation_register_public/02_occupation_mix_by_sex.py [--refresh]
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RAW = config.RAW / "scb_yreg54bas_ssyk4_age_sex_2024.json"
QUERY = config.RAW / "scb_yreg54bas_query.json"
DAIOE = config.PACKAGE / "3_register_mona" / "inputs" / "daioe_quartiles.dta"
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
                 "User-Agent": "canaries-sweden replication package"})
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
    n_m = f"{matched:,.0f}".replace(",", "{,}")
    n_t = f"{total:,.0f}".replace(",", "{,}")
    note = (
        r"Employees aged 16 to 64 in Sweden in 2024, by four-digit occupation, "
        r"age and sex, from Statistics Sweden's occupational register (table "
        rf"YREG54BAS); {n_m} of {n_t} hold an occupation the DAIOE "
        r"generative-AI index prices. Each column is the employment-weighted "
        r"mean over the group's occupations, and the top quartile is the "
        r"index's own. The distribution is national, so part of the gap is "
        r"women and men working at different employers."
    )
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.92\textwidth}\footnotesize\vspace{4pt}",
            note, r"\end{minipage}", r"\end{table}"]
    out = config.TABLES / "tableA_occ_mix_by_sex.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

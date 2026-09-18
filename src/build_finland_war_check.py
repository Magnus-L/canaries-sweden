#!/usr/bin/env python3
"""
build_finland_war_check.py -- does the Russian-invasion story have the sign
it needs to explain the Finnish null?

Runs LOCALLY on public Eurostat data. Writes empirical_data/finland_war_check.csv.

THE QUESTION. Kauhanen and Rouvinen find no youth AI effect in Finland. One
candidate explanation, raised at Katrinelund and sitting in our notes untested
since 12 June 2026, is that Finland absorbed a shock Sweden did not: the
collapse of eastern trade after February 2022.

THE SIGN THE STORY NEEDS. A war shock does not mask an AI decline merely by
being large. It has to fall on the LESS AI-exposed part of the economy. If
Finland lost employment mainly in low-exposure industries, the relative
position of exposed occupations improves for reasons unrelated to AI and a
true AI decline is offset. If the losses fell on high-exposure industries,
the war would deepen a measured AI effect rather than hide it, and the
hypothesis is dead.

WHAT THIS COMPUTES. Employment by NACE section for Finland and Sweden,
2019-2024 (lfsa_egan2), against an AI-exposure score per section built from
our own DAIOE and the occupational composition of each section (lfsa_eisn2,
ISCO-1 by section). Then, per country, the correlation between the 2022-2024
employment change and section exposure, and the implied shift in aggregate
exposure.

WHAT IT IS NOT. Not a causal estimate of the war, and not a counterfactual
for our coefficient. It is the cheapest test of whether the composition story
points the right way, before anyone spends a MONA slot on it.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
UA = {"User-Agent": "AI-Econ Lab research (mlodefalk@gmail.com)"}
ROOT = Path(__file__).resolve().parent.parent
YEARS = ["2019", "2021", "2022", "2023", "2024"]


def fetch(ds: str, **params) -> dict:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{ds}?{q}&format=JSON&lang=EN"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=90) as r:
        return json.loads(r.read().decode())


def tidy(payload: dict, value_name: str) -> pd.DataFrame:
    dims, order, sizes = payload["dimension"], payload["id"], payload["size"]
    cats = {d: sorted(dims[d]["category"]["index"],
                      key=lambda k: dims[d]["category"]["index"][k])
            for d in order}
    rows = []
    for flat, val in payload["value"].items():
        i, coord = int(flat), {}
        for d, s in zip(reversed(order), reversed(sizes)):
            coord[d] = cats[d][i % s]
            i //= s
        coord[value_name] = val
        rows.append(coord)
    return pd.DataFrame(rows)


def employment_by_section(geo: str) -> pd.DataFrame:
    p = fetch("lfsa_egan2", geo=geo, sex="T", age="Y15-64", unit="THS_PER",
              **{"time": "&time=".join(YEARS)})
    d = tidy(p, "emp")
    d["geo"] = geo
    return d[["geo", "nace_r2", "time", "emp"]]


def exposure_by_section(geo: str, daioe_isco: pd.DataFrame) -> pd.DataFrame:
    # This table's own age codes are Y_GE15 and Y20-64, not the LFS Y15-64
    # used elsewhere, and asking for the wrong one returns an empty
    # selection rather than an error.
    p = fetch("lfsa_eisn2", geo=geo, sex="T", age="Y20-64", unit="THS_PER",
              time="2022")
    d = tidy(p, "emp")
    d = d[d["isco08"].str.match(r"^OC[1-9]$")].copy()
    d["isco08"] = d["isco08"].str[-1]
    d = d.merge(daioe_isco, on="isco08", how="inner")
    w = (d.groupby("nace_r2")
           .apply(lambda g: pd.Series({
               "exposure": (g["emp"] * g["daioe"]).sum() / g["emp"].sum()}),
                  include_groups=False)
           .reset_index())
    w["geo"] = geo
    return w


def main():
    daioe = pd.read_csv(ROOT / "data" / "processed" / "daioe_quartiles.csv",
                        dtype={"ssyk4": str})
    col = ("pctl_rank_genai" if "pctl_rank_genai" in daioe.columns
           else [c for c in daioe.columns if "pctl" in c][0])
    daioe["isco08"] = daioe["ssyk4"].str[0]
    daioe_isco = daioe.groupby("isco08")[col].mean().rename("daioe").reset_index()
    print("DAIOE by ISCO-1 major group:")
    print(daioe_isco.to_string(index=False), "\n")

    emp = pd.concat([employment_by_section(g) for g in ("FI", "SE")])
    expo = pd.concat([exposure_by_section(g, daioe_isco) for g in ("FI", "SE")])
    w = emp.pivot_table(index=["geo", "nace_r2"], columns="time",
                        values="emp").reset_index()
    w = w.merge(expo[["geo", "nace_r2", "exposure"]], on=["geo", "nace_r2"])
    w = w[w["nace_r2"].str.len() == 1]
    w["chg_22_24"] = 100 * (w["2024"] / w["2022"] - 1)
    w["share_2022"] = w.groupby("geo")["2022"].transform(lambda s: s / s.sum())

    out = []
    for geo, g in w.groupby("geo"):
        g = g.dropna(subset=["chg_22_24", "exposure"])
        ex22 = (g["exposure"] * g["share_2022"]).sum()
        sh24 = g["2024"] / g["2024"].sum()
        out.append({"geo": geo,
                    "corr_change_exposure": g["chg_22_24"].corr(g["exposure"]),
                    "mean_exposure_2022": ex22,
                    "mean_exposure_2024": (g["exposure"] * sh24).sum(),
                    "composition_shift": (g["exposure"] * sh24).sum() - ex22,
                    "total_emp_change_pct":
                        100 * (g["2024"].sum() / g["2022"].sum() - 1)})
    res = pd.DataFrame(out)
    w.to_csv(ROOT / "empirical_data" / "finland_war_check.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "\n")
    for geo in ("FI", "SE"):
        g = w[w["geo"] == geo].nsmallest(5, "chg_22_24")
        print(f"{geo}: largest employment falls 2022-2024, with section exposure")
        print(g[["nace_r2", "chg_22_24", "exposure"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"), "\n")
    print("READ: masking needs Finland's losses in LOW-exposure sections, a")
    print("POSITIVE correlation between change and exposure in FI, absent in")
    print("SE. A negative correlation kills the hypothesis.")


if __name__ == "__main__":
    main()

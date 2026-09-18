#!/usr/bin/env python3
"""
build_finland_war_withinsector.py -- the within-sector version of the
Russian-invasion explanation for the Finnish null.

Runs LOCALLY on public Eurostat data. NOT for the online appendix: this is
working material for the offline appendix and the private repo.

THE MECHANISM (ML, 18 September 2026). The between-sector version of this
story, which the companion script tests and rejects, asks whether Finland's
job losses fell on less AI-exposed SECTORS. The better version does not need
that. A Finnish manufacturer losing Russian demand cuts production before it
cuts head office. Production work is young, blue-collar and largely
unexposed to generative AI; head office is white-collar and exposed. Inside
such a firm the war pushes the COMPARISON group down while AI pushes the
TREATMENT group down, so the within-firm contrast that identifies an AI
effect is compressed from both sides, and a null is what you would see.

WHAT PUBLIC DATA CAN AND CANNOT DO. We cannot see Finnish firms. We can see
the occupational composition WITHIN a sector, annually, by country
(Eurostat lfsa_eisn2, ISCO-1 by NACE section). If the mechanism operates,
then in the sectors most exposed to Russian demand the blue-collar share
should fall after 2022 relative to the white-collar share, and by more in
Finland than in Sweden. That is a necessary implication, not a sufficient
one: a within-sector tilt is consistent with the story but does not prove it
happens inside firms rather than through firm exit.

EXPOSURE TO RUSSIA. Taken as the goods-producing and transport sectors whose
Finnish exports to Russia collapsed after February 2022: C manufacturing,
B mining, D energy, H transport, and A agriculture and forestry, the last
because Finnish forestry lost Russian roundwood imports. G, I and the
service sectors are the comparison. This is a coarse, pre-committed
assignment, and the script prints every sector so the reader can re-cut it.

AI EXPOSURE OF AN OCCUPATION GROUP. Our own DAIOE, averaged to ISCO-1 major
groups. Groups 1-4 (managers, professionals, technicians, clerical) are the
exposed side, 5-9 (services, agriculture, craft, operators, elementary) the
unexposed side; the mean DAIOE percentiles are 66, 73, 59 and 70 against 35,
20, 21, 16 and 17, so the split is not a close call.
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
RUSSIA_EXPOSED = ["A", "B", "C", "D", "H"]
EXPOSED_ISCO = ["1", "2", "3", "4"]


def fetch(ds: str, **params) -> dict:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}/{ds}?{q}&format=JSON&lang=EN"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=120) as r:
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


def main():
    frames = []
    for geo in ("FI", "SE"):
        p = fetch("lfsa_eisn2", geo=geo, sex="T", age="Y20-64",
                  unit="THS_PER", **{"time": "&time=".join(YEARS)})
        d = tidy(p, "emp")
        d["geo"] = geo
        frames.append(d)
    d = pd.concat(frames)
    d = d[d["isco08"].str.match(r"^OC[1-9]$") & (d["nace_r2"].str.len() == 1)]
    d["isco1"] = d["isco08"].str[-1]
    d["exposed_occ"] = d["isco1"].isin(EXPOSED_ISCO)
    d["russia"] = d["nace_r2"].isin(RUSSIA_EXPOSED)

    # Within sector group, the exposed-occupation share of employment.
    g = (d.groupby(["geo", "russia", "exposed_occ", "time"])["emp"].sum()
         .reset_index())
    tot = g.groupby(["geo", "russia", "time"])["emp"].transform("sum")
    g["share"] = g["emp"] / tot
    sh = (g[g["exposed_occ"]]
          .pivot_table(index=["geo", "russia"], columns="time", values="share"))
    print("Exposed-occupation (ISCO 1-4) share of employment, within sector group")
    print((100 * sh).round(2).to_string(), "\n")
    print("Change in that share, percentage points")
    chg = 100 * (sh["2024"] - sh["2022"])
    print(chg.round(2).to_string(), "\n")

    # The object the mechanism predicts: in Russia-exposed sectors the
    # unexposed side falls faster, so the exposed share RISES, and by more
    # in Finland than in Sweden.
    print("Employment change 2022-2024 by sector group and occupation side, %")
    e = (d.groupby(["geo", "russia", "exposed_occ", "time"])["emp"].sum()
         .reset_index()
         .pivot_table(index=["geo", "russia", "exposed_occ"], columns="time",
                      values="emp"))
    e["chg_pct"] = 100 * (e["2024"] / e["2022"] - 1)
    print(e[["2022", "2024", "chg_pct"]].round(2).to_string(), "\n")

    out = e.reset_index()
    out.to_csv(ROOT / "empirical_data" / "finland_war_withinsector.csv",
               index=False)
    print("READ. The mechanism predicts, in Russia-exposed sectors, a LARGER")
    print("fall for ISCO 5-9 than for ISCO 1-4, and a bigger gap in FI than")
    print("in SE. If the gap is absent or reversed, the within-firm story")
    print("loses its one publicly checkable implication.")


if __name__ == "__main__":
    main()

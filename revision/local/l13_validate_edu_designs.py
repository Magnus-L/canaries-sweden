#!/usr/bin/env python3
"""
l13_validate_edu_designs.py -- predictive validation of every education-
exposure design, LOCALLY, from script 50's export.

    python3 revision/local/l13_validate_edu_designs.py <dir with m6_*, m6b_*, m7_* csv>

For each design: score every education group (and, for the enrolment
designs, every tertiary field) from the m6/m6b matrices with the design's
own weight rule; cut employment-weighted quartiles on the 2019 stock;
predict the quartile of every m7 row (a person freshly coded at t, seen
through the education record k years earlier); compare with the DAIOE
quartile of the occupation actually held at t. Reports, by design, age
band, t and lag k: quartile agreement, top-quartile precision and recall,
and the mean absolute error of the exposure percentile. The drop from
k = 0 to k = 2 is the staleness cost the 2024-25 application pays.

Writes validation_by_design.csv and validation_summary.txt beside the input.
Numbers from a synthetic export mean nothing; the harness is what is tested.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UPLOAD = HERE.parent / "upload"
MIN_CELL = 200
DESIGNS = {
    "OL_daioe":      dict(score="mean",  weights="stock",   fresh=False, expband=False, enrol=False),
    "fresh_stock":   dict(score="mean",  weights="stock",   fresh=True,  expband=False, enrol=False),
    "entrant":       dict(score="mean",  weights="entrant", fresh=True,  expband=False, enrol=False),
    "entrant_share": dict(score="share", weights="entrant", fresh=True,  expband=False, enrol=False),
    "expband":       dict(score="mean",  weights="stock",   fresh=True,  expband=True,  enrol=False),
    "enrol":         dict(score="mean",  weights="entrant", fresh=True,  expband=False, enrol=True),
    "full_nontier":  dict(score="share", weights="entrant", fresh=True,  expband=True,  enrol=True),
}
ENTRANT_BANDS = ["0-2", "3-5"]


def load_daioe():
    d = pd.read_stata(UPLOAD / "daioe_quartiles.dta")
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    q = d["exposure_quartile"]
    if not pd.api.types.is_numeric_dtype(q):
        q = q.astype(str).str.extract(r"(\d)")[0].astype(int)
    return pd.DataFrame({"ssyk4": d["ssyk4"], "score": d["pctl_rank_genai"].astype(float),
                         "high": d["high_exposure"].astype(float), "q": q.astype(int)})


def read_matrices(d: Path, prefix: str, years):
    frames = []
    for y in years:
        f = d / f"{prefix}_{y}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, dtype={"ssyk4": str, "inr": str, "grp": str}).assign(year=y))
    out = pd.concat(frames, ignore_index=True)
    out["n"] = out["n"].fillna(0)          # floored cells count as zero
    out["ssyk4"] = out["ssyk4"].str.zfill(4)
    return out


def wagg(df, by, col):
    d = df.dropna(subset=[col]).copy()
    d["_ws"] = d[col] * d["n"]
    g = d.groupby(by, observed=True).agg(n=("n", "sum"), _ws=("_ws", "sum")).reset_index()
    g["s"] = g["_ws"] / g["n"]
    return g.drop(columns="_ws")


def build_scores(m6, m6b, daioe, spec):
    col = "high" if spec["score"] == "share" else "score"
    pop = m6.merge(daioe[["ssyk4", col]], on="ssyk4", how="inner")
    years = [2019] if spec["weights"] == "stock" and not spec["expband"] else [2019, 2020, 2021]
    pop = pop[pop["year"].isin(years)]
    if spec["fresh"]:
        pop = pop[pop["fresh"] == 1]
    ent = pop[pop["expband"].isin(ENTRANT_BANDS)] if spec["weights"] == "entrant" else pop
    grp = wagg(ent, ["grp"], col)
    stock = wagg(pop, ["grp"], col)
    if spec["weights"] == "entrant":
        thin = grp[grp["n"] < MIN_CELL]["grp"]
        grp = pd.concat([grp[~grp["grp"].isin(thin)],
                         stock[stock["grp"].isin(set(thin) | (set(stock["grp"]) - set(grp["grp"])))]])
    scores = {"grp": dict(zip(grp["grp"], grp["s"]))}
    if spec["expband"]:
        b = wagg(pop, ["grp", "expband"], col)
        b = b[b["n"] >= MIN_CELL]
        scores["band"] = {(g, e): s for g, e, s in zip(b["grp"], b["expband"], b["s"])}
    if spec["enrol"]:
        ter = m6b.merge(daioe[["ssyk4", col]], on="ssyk4", how="inner")
        ter = ter[ter["year"].isin(years)]
        if spec["fresh"]:
            ter = ter[ter["fresh"] == 1]
        if spec["weights"] == "entrant":
            ter = ter[ter["expband"].isin(ENTRANT_BANDS)]
        i = wagg(ter, ["inr"], col)
        i = i[i["n"] >= MIN_CELL]
        scores["inr"] = dict(zip(i["inr"].astype(str), i["s"]))
    # cut points: 2019 stock, every worker at the design's group score
    base = m6[m6["year"] == 2019].groupby("grp")["n"].sum().reset_index()
    base["s"] = base["grp"].map(scores["grp"])
    base = base.dropna(subset=["s"]).sort_values("s")
    cum = base["n"].cumsum() / base["n"].sum()
    cuts = [float(base["s"].to_numpy()[np.searchsorted(cum.to_numpy(), q, side="left")])
            for q in (0.25, 0.5, 0.75)]
    return scores, cuts


def predict(m7, scores, cuts, spec):
    s = m7["grp_lag"].map(scores["grp"]).astype(float).to_numpy()
    if spec["expband"]:
        b = np.array([scores["band"].get((g, e), np.nan)
                      for g, e in zip(m7["grp_lag"], m7["expband_lag"])], dtype=float)
        s = np.where(np.isnan(b), s, b)
    if spec["enrol"]:
        a = m7["enr_inr"].astype(str).map(scores["inr"]).astype(float).to_numpy()
        young = m7["age_group"].isin(["22-25", "26-30"]).to_numpy()
        elig = young & (m7["tertiary_lag"].to_numpy() == 0) & ~np.isnan(a)
        s = np.where(elig, a, s)
    q = np.searchsorted(np.asarray(cuts), s, side="right") + 1
    return np.where(np.isnan(s), 0, q), s


def main(d: Path):
    daioe = load_daioe()
    m6 = read_matrices(d, "m6_matrix", range(2019, 2024))
    m6b = read_matrices(d, "m6b_inr_tertiary", range(2019, 2024))
    m6b["inr"] = m6b["inr"].astype(str)
    rows = []
    for name, spec in DESIGNS.items():
        scores, cuts = build_scores(m6, m6b, daioe, spec)
        for t in (2021, 2022, 2023):
            for k in (0, 2):
                f = d / f"m7_validation_t{t}_k{k}.csv"
                if not f.exists():
                    continue
                m7 = pd.read_csv(f, dtype={"ssyk4_t": str, "enr_inr": str, "grp_lag": str})
                m7["n"] = m7["n"].fillna(0)
                m7["ssyk4_t"] = m7["ssyk4_t"].str.zfill(4)
                m7 = m7.merge(daioe.rename(columns={"ssyk4": "ssyk4_t", "q": "q_true",
                                                    "score": "s_true"})[["ssyk4_t", "q_true", "s_true"]],
                              on="ssyk4_t", how="inner")
                q_hat, s_hat = predict(m7, scores, cuts, spec)
                m7["q_hat"], m7["s_hat"] = q_hat, s_hat
                for age, g in m7.groupby("age_group"):
                    w = g["n"].to_numpy()
                    cl = g["q_hat"].to_numpy() > 0
                    n = w.sum()
                    if n == 0:
                        continue
                    agree = (w * ((g["q_hat"] == g["q_true"]).to_numpy() & cl)).sum() / max(w[cl].sum(), 1)
                    tp = (w * ((g["q_hat"] == 4) & (g["q_true"] == 4)).to_numpy()).sum()
                    prec = tp / max((w * (g["q_hat"] == 4).to_numpy()).sum(), 1)
                    rec = tp / max((w * (g["q_true"] == 4).to_numpy()).sum(), 1)
                    mae = np.nansum(w * np.abs(g["s_hat"] - g["s_true"]).to_numpy()) / max(w[cl].sum(), 1) \
                        if spec["score"] == "mean" else np.nan
                    rows.append(dict(design=name, t=t, k=k, age_group=age, n=int(n),
                                     classified=float(w[cl].sum() / n), agreement=float(agree),
                                     q4_precision=float(prec), q4_recall=float(rec), mae_pctl=float(mae)))
    res = pd.DataFrame(rows)
    res.to_csv(d / "validation_by_design.csv", index=False)
    lines = ["PREDICTIVE VALIDATION OF EDUCATION-EXPOSURE DESIGNS (ages 22-25)",
             "quartile agreement / Q4 recall at lag 0 -> lag 2, pooled over t", ""]
    y = res[res["age_group"] == "22-25"]
    for name in DESIGNS:
        r = y[y["design"] == name]
        if r.empty:
            continue
        a0 = np.average(r[r.k == 0]["agreement"], weights=r[r.k == 0]["n"])
        a2 = np.average(r[r.k == 2]["agreement"], weights=r[r.k == 2]["n"])
        r0 = np.average(r[r.k == 0]["q4_recall"], weights=r[r.k == 0]["n"])
        r2 = np.average(r[r.k == 2]["q4_recall"], weights=r[r.k == 2]["n"])
        lines.append(f"  {name:<14} agreement {a0:.3f} -> {a2:.3f}   Q4 recall {r0:.3f} -> {r2:.3f}")
    (d / "validation_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]))

#!/usr/bin/env python3
"""
l58_tau_recovered.py: the headline contrast tau = gamma_2 - gamma_0 (the
step from the 2023 level) recovered from saved MONA exports for the OA
robustness tables that print only gamma_2 (the step from the tightening
months). No MONA run; public exports only.

WHY THIS EXISTS
A second external review noted that OA tab:industry_credit and
tab:size_reliability report the adoption step gamma_2 while Table 1's
headline is tau = gamma_2 - gamma_0. Where the export carries the interim
coefficient gamma_0 and the full covariance of the fit, tau and its
standard error follow locally: se(tau)^2 = V22 + V00 - 2 V20.

WHAT IS RECOVERABLE (lane round3_20260923-0655-lanes28b-29bcd)
  tab:industry_credit Panel A  occ_rest_industry.csv with
      vcov_s80_indseas2_{base,ind}_{22_25,26_30}.csv. These fits carry the
      full Equation (2) term set: PostRB, the three calendar-quarter terms,
      Interim and Post, each x High x Young (script 80 part_c via
      78.eq2_terms). RECOVERED.
  tab:size_reliability Panel B occ_rest_size.csv with
      vcov_s83_size_{floor_5,floor_60,tercile_2,tercile_3}_{band}.csv, the
      same term set. RECOVERED.
  tab:industry_credit Panel B  occ_rest_credit.csv with vcov_r73_*. Script
      73's base_terms fits only PostRB (from April 2022) and Post (from
      January 2024): no Interim and no calendar-quarter terms. gamma_0 was
      never estimated, so tau is NOT recoverable; the Post coefficient in
      that spec is already a step from the April 2022 to December 2023
      average, without the seasonal terms.

GATES
The standard errors implied by each covariance diagonal must equal the
exported SEs (1e-6), and the floor_5 row of the size table, which is
Table 1's sample, must reproduce Table 1's tau: -0.0399 (0.0102) at 22-25
and -0.0403 (0.0067) at 26-30. The young-men step from the 2023 level
(quoted in the response letter as -0.0058 (0.0120)) is recomputed from
occ_route_gender.csv and vcov_s82_gender_22_25.csv as a further check.

Run:  python3 revision/local/l58_tau_recovered.py
Out:  revision/tables/l58_tau_recovered.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "revision" / "output" / "round3_20260923-0655-lanes28b-29bcd"
OUT = ROOT / "revision" / "tables" / "l58_tau_recovered.csv"
POST, INTER = "post_x_high_x_young", "interim_x_high_x_young"
TABLE1 = {"22-25": (-0.0399, 0.0102), "26-30": (-0.0403, 0.0067)}


def tau(coefs: pd.DataFrame, vfile: str, post=POST, inter=INTER) -> dict:
    v = pd.read_csv(LANE / vfile, index_col=0)
    c = coefs.set_index("term")
    for t in (post, inter):
        assert abs(np.sqrt(v.loc[t, t]) - c.loc[t, "se"]) < 1e-6, (vfile, t)
    g2, g0 = c.loc[post, "coef"], c.loc[inter, "coef"]
    se = np.sqrt(v.loc[post, post] + v.loc[inter, inter] - 2 * v.loc[post, inter])
    return dict(gamma2=g2, se_gamma2=c.loc[post, "se"], gamma0=g0, se_gamma0=c.loc[inter, "se"],
                tau=g2 - g0, se_tau=se, vcov_file=vfile)


def main():
    rows = []
    ind = pd.read_csv(LANE / "occ_rest_industry.csv")
    for band in ("22-25", "26-30"):
        tag = band.replace("-", "_")
        for spec, vk in (("baseline_same_sample", "base"), ("industry_age_month", "ind")):
            g = ind[(ind.young_band == band) & (ind.spec == spec)]
            r = tau(g, f"vcov_s80_indseas2_{vk}_{tag}.csv")
            rows.append(dict(table="tab:industry_credit", panel="A", spec=spec, band=band,
                             n_firms=int(g.n_firms.iloc[0]), seasonal_terms="yes", **r))
    size = pd.read_csv(LANE / "occ_rest_size.csv")
    for band in ("22-25", "26-30"):
        tag = band.replace("-", "_")
        for spec in ("floor_5", "floor_60", "tercile_2", "tercile_3"):
            g = size[(size.young_band == band) & (size.spec == spec)]
            r = tau(g, f"vcov_s83_size_{spec}_{tag}.csv")
            rows.append(dict(table="tab:size_reliability", panel="B", spec=spec, band=band,
                             n_firms=int(g.n_firms.iloc[0]), seasonal_terms="yes", **r))
    res = pd.DataFrame(rows)
    # Retained share of tau under industry x age x month (Panel A).
    for band in ("22-25", "26-30"):
        a = res[(res.panel == "A") & (res.band == band)].set_index("spec")
        res.loc[(res.panel == "A") & (res.band == band) & (res.spec == "industry_age_month"),
                "retained_tau"] = a.loc["industry_age_month", "tau"] / a.loc["baseline_same_sample", "tau"]

    # Gate: floor_5 is Table 1's sample and must give Table 1's tau.
    for band, (t, s) in TABLE1.items():
        r = res[(res.spec == "floor_5") & (res.band == band)].iloc[0]
        assert round(r.tau, 4) == t and round(r.se_tau, 4) == s, (band, r.tau, r.se_tau)
    print("GATE PASSED: size floor_5 reproduces Table 1's tau at both bands")

    # Young men, step from the 2023 level (response letter: -0.0058 (0.0120)).
    gen = pd.read_csv(LANE / "occ_route_gender.csv")
    gm = gen[(gen.block == "term") & (gen.young_band == "22-25")]
    men = tau(gm, "vcov_s82_gender_22_25.csv")
    print(f"young men 22-25: gamma2 {men['gamma2']:+.4f} ({men['se_gamma2']:.4f}), "
          f"gamma0 {men['gamma0']:+.4f}, tau {men['tau']:+.4f} ({men['se_tau']:.4f})")
    res = pd.concat([res, pd.DataFrame([dict(table="Table 1 / response letter", panel="",
                                             spec="young_men_male_terms", band="22-25",
                                             n_firms=np.nan, seasonal_terms="yes", **men)])],
                    ignore_index=True)
    # Not recoverable: the credit panel.
    res = pd.concat([res, pd.DataFrame([dict(
        table="tab:industry_credit", panel="B", spec="credit (all rows)", band="both",
        seasonal_terms="no", vcov_file="vcov_r73_{base,levbase,lev}_<band>.csv: terms "
        "post_rb, post, post_x_young_x_lev, post_x_high_x_young_x_lev only; no interim term, "
        "tau not recoverable")])], ignore_index=True)
    res.to_csv(OUT, index=False)
    print(res.drop(columns="vcov_file").round(4).to_string())


if __name__ == "__main__":
    main()

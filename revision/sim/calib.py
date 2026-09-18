#!/usr/bin/env python3
"""
calib.py -- turn script 50's export into the generator's parameters.

Every quantity the data-generating process needs is measured from
`output_50/` when it is present and falls back to a NAMED, documented
placeholder when it is not, so the simulator runs before the export lands
and switches to measured values the moment it does. `Calibration.source`
records, per field, which of the two was used; `run_sim.py` prints it and
the acceptance tests refuse to certify a run with placeholders in the
fields they bind to.

Nothing here is export-sensitive: every input is an aggregate with cells
floored at five, and floored (blank) cells are read as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = [2019, 2020, 2021, 2022, 2023]
EXP_BANDS = ["0-2", "3-5", "6-10", "11-20", "21+", "na"]
AGE_BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]

# ---------------------------------------------------------------- placeholders
# Each is a measured number from an earlier round, or a stated guess. The
# comment says which, because a guess that is never re-read becomes a fact.
PLACEHOLDER = dict(
    # measured, data-notes/occupation-missingness.md (wage earners, 2019-2023)
    occ_missing_by_age={"22-25": 0.29, "26-30": 0.24, "31-34": 0.07, "35-40": 0.06,
                        "41-49": 0.05, "50+": 0.11},
    # guess: annual P(occupation code changes) among coded persons
    occ_change_by_age={"22-25": 0.30, "26-30": 0.20, "31-34": 0.13, "35-40": 0.10,
                       "41-49": 0.07, "50+": 0.04},
    # guess: P(new code is in the same DAIOE quartile | changed)
    stay_quartile=0.55,
    # guess: annual P(enters a managerial 1xxx code), by experience band
    to_manager={"0-2": 0.002, "3-5": 0.006, "6-10": 0.012, "11-20": 0.015,
                "21+": 0.010, "na": 0.004},
    # guess: mean age at completion by SUN level digit
    completion_age={"3": 19.0, "4": 22.0, "5": 25.0, "6": 30.0},
    # guess: annual P(highest education level rises), 22-25 and 26-30
    level_up={"22-25": 0.14, "26-30": 0.07, "31-34": 0.03, "35-40": 0.02,
              "41-49": 0.01, "50+": 0.005},
    # guess: share of employed under-30s with a recent registration
    enrol_share={"22-25": 0.35, "26-30": 0.15},
    # guess: P(last registration field != completed field)
    field_switch=0.18,
    # measured, 47h weights 2019: employer size distribution is lognormal-ish
    emp_size_lognorm=(2.2, 1.1),
    # guess: mean years between the observation year and SsykAr_J16
    stale_by_age={"22-25": 0.9, "26-30": 0.9, "31-34": 1.0, "35-40": 1.0,
                  "41-49": 1.1, "50+": 1.2},
)


@dataclass
class Calibration:
    """Everything the DGP reads. `occ_by_group_band` is the heart of it."""
    occ_by_group_band: pd.DataFrame          # grp, expband, year, ssyk4, p
    groups: list                             # education groups in play
    occ_missing_by_age: dict = field(default_factory=dict)
    occ_change_by_age: dict = field(default_factory=dict)
    stay_quartile: float = 0.55
    to_manager: dict = field(default_factory=dict)
    completion_age: dict = field(default_factory=dict)
    level_up: dict = field(default_factory=dict)
    enrol_share: dict = field(default_factory=dict)
    field_switch: float = 0.18
    emp_size_lognorm: tuple = (2.2, 1.1)
    stale_by_age: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def measured_fields(self) -> list:
        return sorted(k for k, v in self.source.items() if v == "measured")

    @property
    def placeholder_fields(self) -> list:
        return sorted(k for k, v in self.source.items() if v == "placeholder")


def _read(d: Path, name: str) -> "pd.DataFrame | None":
    f = d / name
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if "n" in df:
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0.0)
    return df


def _norm(df: pd.DataFrame, by: list, val: str = "n") -> pd.DataFrame:
    tot = df.groupby(by, observed=True)[val].transform("sum")
    out = df[tot > 0].copy()
    out["p"] = out[val] / tot[tot > 0]
    return out


def build(export_dir: "Path | None", key: pd.DataFrame, daioe: pd.DataFrame,
          rng: np.random.Generator, n_tracks: int = 80) -> Calibration:
    """
    export_dir: script 50's output_50, or None for a fully synthetic
    calibration. `key` is the real utbildningsgrupp key; `daioe` the real
    score table (ssyk4, score, high, q).
    """
    src, d = {}, Path(export_dir) if export_dir else None

    # ---- the occupation distribution per (group, experience band, year) ----
    m6 = None
    if d is not None:
        frames = [x.assign(year=y) for y in YEARS
                  if (x := _read(d, f"m6_matrix_{y}.csv")) is not None]
        m6 = pd.concat(frames, ignore_index=True) if frames else None
    if m6 is not None and len(m6):
        m6 = m6[m6["grp"].astype(str) != "unmatched"].copy()
        m6["ssyk4"] = m6["ssyk4"].astype(str).str.zfill(4)
        m6 = m6[m6["ssyk4"].isin(set(daioe["ssyk4"]))]
        occ = _norm(m6.groupby(["grp", "expband", "year", "ssyk4"], observed=True)
                    ["n"].sum().reset_index(), ["grp", "expband", "year"])
        occ = occ[["grp", "expband", "year", "ssyk4", "p"]]
        groups = sorted(occ["grp"].unique())
        src["occ_by_group_band"] = "measured"
    else:
        groups = sorted(rng.choice(sorted(key["grp"].unique()),
                                   size=min(n_tracks, key["grp"].nunique()),
                                   replace=False).tolist())
        hi = daioe.loc[daioe["high"] == 1, "ssyk4"].to_numpy()
        lo = daioe.loc[daioe["high"] == 0, "ssyk4"].to_numpy()
        rows = []
        for g in groups:
            tilt = rng.beta(2, 2)                      # group exposure tilt
            codes = np.concatenate([rng.choice(hi, 6, replace=False),
                                    rng.choice(lo, 6, replace=False)])
            for b in EXP_BANDS:
                # experience drift: exposure tilt moves with the band, which
                # is the placeholder stand-in for the measured M6 profile
                shift = {"0-2": 0.0, "3-5": 0.03, "6-10": 0.06,
                         "11-20": 0.08, "21+": 0.05, "na": 0.0}[b]
                w = np.concatenate([np.full(6, tilt + shift), np.full(6, 1 - tilt)])
                w = np.clip(w, 0.01, None)
                for y in YEARS:
                    wy = w * rng.uniform(0.9, 1.1, len(w))   # cohort jitter
                    rows.append(pd.DataFrame({"grp": g, "expband": b, "year": y,
                                              "ssyk4": codes, "p": wy / wy.sum()}))
        occ = pd.concat(rows, ignore_index=True)
        src["occ_by_group_band"] = "placeholder"

    c = Calibration(occ_by_group_band=occ, groups=groups, source=src)

    # ---- scalars and small dicts ----
    def take(nameset, fn):
        got = fn() if d is not None else None
        if got:
            for k, v in got.items():
                setattr(c, k, v)
                src[k] = "measured"
        else:
            for k in nameset:
                setattr(c, k, PLACEHOLDER[k])
                src[k] = "placeholder"

    def f_occ_change():
        m2 = _read(d, "m2_occ_change.csv")
        if m2 is None or not len(m2):
            return None
        chg = (m2.groupby("age_group", observed=True)
               .apply(lambda g: 1 - (g.loc[g["same_code"] == 1, "n"].sum()
                                     / max(g["n"].sum(), 1)), include_groups=False))
        same_q = m2[(m2["same_code"] == 0) & (m2["q_t"] > 0) & (m2["q_t1"] > 0)]
        stay = (same_q.loc[same_q["q_t"] == same_q["q_t1"], "n"].sum()
                / max(same_q["n"].sum(), 1))
        mgr = (m2.groupby("expband", observed=True)
               .apply(lambda g: g.loc[g["enters_mgr"] == 1, "n"].sum()
                      / max(g["n"].sum(), 1), include_groups=False))
        return dict(occ_change_by_age={k: float(v) for k, v in chg.items()},
                    stay_quartile=float(stay),
                    to_manager={k: float(v) for k, v in mgr.items()})

    def f_completion():
        m1 = _read(d, "m1a_completion_age.csv")
        if m1 is None or not len(m1):
            return None
        m1 = m1.dropna(subset=["n"])
        m1 = m1[(m1["exam_age"] >= 15) & (m1["exam_age"] <= 70)]
        w = (m1.assign(x=m1["exam_age"] * m1["n"]).groupby("level", observed=True)
             .agg(x=("x", "sum"), n=("n", "sum")))
        return dict(completion_age={str(k): float(r.x / r.n)
                                    for k, r in w.iterrows() if r.n > 0})

    def f_level_up():
        m1b = _read(d, "m1b_level_change.csv")
        if m1b is None or not len(m1b):
            return None
        x = m1b.dropna(subset=["n"]).copy()
        for cnm in ("level_t", "level_t1"):
            x[cnm] = pd.to_numeric(x[cnm], errors="coerce")
        x = x.dropna(subset=["level_t", "level_t1"])
        up = (x.groupby("age_group", observed=True)
              .apply(lambda g: g.loc[g["level_t1"] > g["level_t"], "n"].sum()
                     / max(g["n"].sum(), 1), include_groups=False))
        return dict(level_up={k: float(v) for k, v in up.items()})

    def f_enrol():
        m4a, m4b = _read(d, "m4a_enrolment_prevalence.csv"), _read(d, "m4b_field_switch.csv")
        if m4a is None or not len(m4a):
            return None
        sh = (m4a.groupby("age_group", observed=True)
              .apply(lambda g: g.loc[g["registered"] == 1, "n"].sum()
                     / max(g["n"].sum(), 1), include_groups=False))
        out = dict(enrol_share={k: float(v) for k, v in sh.items()})
        if m4b is not None and len(m4b):
            tot = m4b["n"].sum()
            diff = m4b.loc[m4b["match"].isin(["different"]), "n"].sum()
            out["field_switch"] = float(diff / max(tot, 1))
        return out

    def f_stale():
        m3 = _read(d, "m3_staleness.csv")
        if m3 is None or not len(m3):
            return None
        x = m3.dropna(subset=["n"])
        mean = (x.assign(z=x["stale_years"] * x["n"]).groupby("age_group", observed=True)
                .agg(z=("z", "sum"), n=("n", "sum")))
        return dict(stale_by_age={k: float(r.z / r.n) for k, r in mean.iterrows() if r.n > 0})

    def f_missing():
        return None      # not in 50's export; the placeholder is a measured number

    def f_size():
        m5 = _read(d, "m5a_employer_size.csv")
        if m5 is None or not len(m5):
            return None
        mid = {"1-4": 2.5, "5-9": 7, "10-19": 14, "20-49": 32, "50-99": 70,
               "100-249": 160, "250-999": 500, "1000+": 2000}
        x = m5.dropna(subset=["n"]).copy()
        x["mid"] = x["size_band"].map(mid)
        x = x.dropna(subset=["mid"])
        lg = np.log(x["mid"].to_numpy())
        w = x["n"].to_numpy()
        mu = float(np.average(lg, weights=w))
        sd = float(np.sqrt(np.average((lg - mu) ** 2, weights=w)))
        return dict(emp_size_lognorm=(mu, sd))

    take(["occ_change_by_age", "stay_quartile", "to_manager"], f_occ_change)
    take(["completion_age"], f_completion)
    take(["level_up"], f_level_up)
    take(["enrol_share", "field_switch"], f_enrol)
    take(["stale_by_age"], f_stale)
    take(["occ_missing_by_age"], f_missing)
    take(["emp_size_lognorm"], f_size)
    c.source = src
    return c


def summary(c: Calibration) -> str:
    lines = ["CALIBRATION", "-" * 60,
             f"  groups {len(c.groups)}, occupation cells "
             f"{len(c.occ_by_group_band):,}"]
    for k in sorted(c.source):
        lines.append(f"  {k:<22} {c.source[k]}")
    return "\n".join(lines)

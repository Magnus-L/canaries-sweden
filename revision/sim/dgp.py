#!/usr/bin/env python3
"""
dgp.py -- the person-level generator, the register layer, and the emitters.

THE TRUTH. Every person holds an occupation in every year; its DAIOE genAI
percentile is that person's exposure, and the top-quartile flag is `High`.
From 2022-12 a person whose occupation is High and whose age band carries a
treatment effect is re-hired more slowly by exp(gamma_a). The ORACLE panel
is built from those true occupations, and the oracle's gamma2 -- not the
structural gamma -- is what designs are scored against: it is the estimate
a perfect assignment would produce on this very sample, which is the
ceiling any education-based design can reach.

CIRCULARITY, STATED. Occupations are drawn from the measured
P(ssyk4 | group, experience band, year) matrix, so a design that scores by
(group, band) is estimating the DGP's own conditional mean and will look
good at the level margin BY CONSTRUCTION. That is why the study ranks
designs on ROBUSTNESS TO LAG (artefact, false-pass, null false-decline),
not on fit, and why persistent person effects below keep the mapping
stochastic rather than deterministic.

THE REGISTER LAYER distorts the truth the way SCB's registers do: an
annual occupation snapshot that surveys large employers and carries codes
forward for small ones, an education register that records a completion
only 1-2 years later, vintages that stop at 2023 so 2024-25 inherit the
cascade, and an enrolment register that ends in 2021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from calib import AGE_BANDS, EXP_BANDS

MONTHS_ALL = [f"{y}-{m:02d}" for y in range(2019, 2026)
              for m in range(1, 13) if not (y == 2025 and m > 6)]
CHATGPT_YM = "2022-12"
BACKTEST_YEARS = [2019, 2020, 2021, 2022, 2023]
PROD_YEARS = list(range(2019, 2026))


@dataclass
class Params:
    n_persons: int = 60_000
    n_emp: int = 1_500
    lag_edu: int = 2                  # years from completion to the register
    occ_change_scale: float = 1.0     # multiplies the measured change rates
    field_switch_scale: float = 1.0
    gamma: dict = field(default_factory=lambda: {"22-25": -0.15})
    gamma_rb: float = 0.0
    # register layer
    size_always_coded: int = 100
    p_small_coded: tuple = (0.02, 0.25)   # (<10 employees, 10-99)
    p_impute: float = 0.5
    hreg_end: int = 2021
    distortions: bool = True          # False = registers report the truth
    # careers
    h_sep: dict = field(default_factory=lambda: {
        "22-25": 0.035, "26-30": 0.028, "31-34": 0.020, "35-40": 0.016,
        "41-49": 0.013, "50+": 0.012})
    rehire_mean_months: float = 3.0
    person_effect_sd: float = 0.6     # persistence in occupation choice


def _hire_mult(h: float, s: float, g: float) -> float:
    """Multiplier on the monthly re-hire hazard that moves the steady-state
    employed share by exp(g). Returns 1 for g == 0, and is clipped so an
    extreme gamma cannot drive the hazard negative."""
    if g == 0.0:
        return 1.0
    A = np.exp(g) * h / (h + s)
    if A >= 1.0:
        return 10.0
    return float(np.clip(A * s / (h * (1.0 - A)), 0.01, 10.0))


def age_band(age: np.ndarray) -> np.ndarray:
    out = np.full(len(age), "", dtype=object)
    for lo, hi, lab in ((22, 25, "22-25"), (26, 30, "26-30"), (31, 34, "31-34"),
                        (35, 40, "35-40"), (41, 49, "41-49"), (50, 69, "50+")):
        out[(age >= lo) & (age <= hi)] = lab
    return out


def exp_band(years) -> np.ndarray:
    y = np.asarray(years, dtype=float)
    out = np.full(len(y), "na", dtype=object)
    ok = ~np.isnan(y)
    out[ok & (y <= 2)] = "0-2"
    out[ok & (y > 2) & (y <= 5)] = "3-5"
    out[ok & (y > 5) & (y <= 10)] = "6-10"
    out[ok & (y > 10) & (y <= 20)] = "11-20"
    out[ok & (y > 20)] = "21+"
    return out


class Sim:
    """Holds the generated population and serves the emitters."""

    def __init__(self, p: Params, cal, key: pd.DataFrame, daioe: pd.DataFrame,
                 rng: np.random.Generator):
        self.p, self.cal, self.rng = p, cal, rng
        self.key = key
        self.daioe = daioe.set_index("ssyk4")
        self._build_lookups()
        self._persons()
        self._occupations()
        self._employment()
        self._registers()

    # ------------------------------------------------------------------ setup
    def _build_lookups(self):
        occ = self.cal.occ_by_group_band
        self.groups = [g for g in self.cal.groups if g in set(occ["grp"])]
        # one (group, band, year) -> (codes, probs) table, as arrays
        self.occ_tab = {}
        for (g, b, y), sub in occ.groupby(["grp", "expband", "year"], observed=True):
            pr = sub["p"].to_numpy(dtype=float)
            if pr.sum() <= 0:
                continue
            self.occ_tab[(g, b, int(y))] = (sub["ssyk4"].to_numpy(), pr / pr.sum())
        # a (niva, inr) cell for each group, so emitted records look real
        k = self.key[self.key["grp"].isin(self.groups)].drop_duplicates("grp")
        self.grp_cell = {r.grp: (r.niva, r.inr) for r in k.itertuples()}
        self.groups = [g for g in self.groups if g in self.grp_cell]
        # The pre-completion record: what the register shows before a degree
        # is visible. It must be a group the occupation matrix KNOWS, or the
        # person is scored NaN and simply drops out, which is attrition rather
        # than the misclassification 47b measured. Upper-secondary groups are
        # preferred; if none is in the matrix, any group serves and the
        # distortion is weaker, which the acceptance tests check for.
        in_matrix = set(g for (g, _, _) in self.occ_tab)
        gym_grp = [g for g in self.groups
                   if g in in_matrix and str(self.grp_cell[g][0])[:1] == "3"]
        self.gym_groups = gym_grp or [g for g in self.groups if g in in_matrix]
        self.gym_cells = [self.grp_cell[g] for g in self.gym_groups] or [("300", "0000")]
        self.score = self.daioe["score"].to_dict()
        self.high = self.daioe["high"].to_dict()
        self.tquart = self.daioe["q"].to_dict()

    def _persons(self):
        p, rng = self.p, self.rng
        n = p.n_persons
        # Birth years must span 1950-2003, not one cross-section: a population
        # drawn as "aged 22-69 in 2019" has NOBODY aged 22-25 by 2023, which is
        # precisely the margin the paper is about. Ages 16-21 in 2022 are the
        # cohorts that ENTER during the window; ages 22-69 carry the 2019
        # cross-sectional shares, spread evenly within each band.
        shares = {"22-25": 0.10, "26-30": 0.13, "31-34": 0.10, "35-40": 0.14,
                  "41-49": 0.23, "50+": 0.30}
        span = {"22-25": (22, 25), "26-30": (26, 30), "31-34": (31, 34),
                "35-40": (35, 40), "41-49": (41, 49), "50+": (50, 69)}
        ages, wts = [], []
        for b, sh in shares.items():
            a0, a1 = span[b]
            for a in range(a0, a1 + 1):
                ages.append(a)
                wts.append(sh / (a1 - a0 + 1))
        young_density = shares["22-25"] / 4.0          # per single year of age
        for a in range(16, 22):
            ages.append(a)
            wts.append(young_density)
        wts = np.array(wts, dtype=float)
        age22 = rng.choice(np.array(ages), n, p=wts / wts.sum())
        birth = 2022 - age22
        grp = rng.choice(self.groups, n)
        # completion year from the group's level and the measured mean age
        niva0 = np.array([self.grp_cell[g][0][:1] for g in grp])
        mean_age = np.array([self.cal.completion_age.get(lv, 22.0) for lv in niva0])
        exam = birth + np.round(mean_age + rng.normal(0, 2.0, n)).astype(int)
        never = rng.uniform(size=n) < 0.06
        exam = np.where(never, -9999, exam)
        # enrolment: the field enrolled in, which is the completed field unless
        # the person switched (measured rate), and the last registration year
        switch = rng.uniform(size=n) < min(self.cal.field_switch * self.p.field_switch_scale, 0.9)
        enrol_grp = np.where(switch, rng.choice(self.groups, n), grp)
        self.per = pd.DataFrame(dict(
            pid=np.arange(n), birth=birth, grp=grp, exam=exam,
            enrol_grp=enrol_grp, last_reg=np.where(never, -9999, exam - 1),
            eff=rng.normal(0, self.p.person_effect_sd, n)))

    def _occupations(self):
        """One occupation per person-year, drawn from the measured matrix for
        the person's (group, experience band, year), tilted by a persistent
        person effect so the mapping is stochastic, and made sticky with the
        measured occupation-change rate."""
        rng, per = self.rng, self.per
        rows = []
        prev = {}
        for y in PROD_YEARS:
            age = y - per["birth"].to_numpy()
            ab = age_band(age)
            yrs = np.where(per["exam"].to_numpy() > 0, y - per["exam"].to_numpy(), np.nan)
            yrs = np.where(yrs < 0, np.nan, yrs)
            eb = exp_band(yrs)
            ysrc = min(max(y, 2019), 2023)
            codes = np.empty(len(per), dtype=object)
            chg = np.array([self.cal.occ_change_by_age.get(b, 0.15) for b in ab])
            chg = np.clip(chg * self.p.occ_change_scale, 0, 1)
            draw_new = (rng.uniform(size=len(per)) < chg) | (y == PROD_YEARS[0])
            for i, (g, b) in enumerate(zip(per["grp"], eb)):
                if not draw_new[i] and i in prev:
                    codes[i] = prev[i]
                    continue
                tab = self.occ_tab.get((g, b, ysrc)) or self.occ_tab.get((g, "na", ysrc))
                if tab is None:
                    cand = list(self.score)
                    codes[i] = cand[rng.integers(len(cand))]
                    continue
                cds, pr = tab
                if self.per["eff"].iat[i]:
                    w = pr * np.exp(self.per["eff"].iat[i]
                                    * np.array([self.high.get(c, 0.0) for c in cds]))
                    pr = w / w.sum()
                codes[i] = cds[rng.choice(len(cds), p=pr)]
                prev[i] = codes[i]
            rows.append(pd.DataFrame(dict(pid=per["pid"], year=y, ssyk4=codes,
                                          age=age, age_group=ab, expband=eb)))
        self.py = pd.concat(rows, ignore_index=True)
        self.py["exposure"] = self.py["ssyk4"].map(self.score).astype(float)
        self.py["high"] = self.py["ssyk4"].map(self.high).fillna(0).astype(int)
        self.py["tq"] = self.py["ssyk4"].map(self.tquart).fillna(0).astype(int)

    def _employment(self):
        """Monthly employer spells with separations, a gap, and re-hiring that
        the shock slows for treated High young workers."""
        p, rng = self.p, self.rng
        mu, sd = self.cal.emp_size_lognorm
        w = rng.lognormal(mu, sd, p.n_emp)
        w = w / w.sum()
        emp_of = rng.choice(p.n_emp, len(self.per), p=w)
        high_by_year = {y: g.set_index("pid")["high"].to_dict()
                        for y, g in self.py.groupby("year")}
        band_by_year = {y: g.set_index("pid")["age_group"].to_dict()
                        for y, g in self.py.groupby("year")}
        n = len(self.per)
        employed = np.ones(n, dtype=bool)
        rows = []
        for ym in MONTHS_ALL:
            y = int(ym[:4])
            hb, bb = high_by_year[y], band_by_year[y]
            bands = np.array([bb.get(i, "") for i in range(n)], dtype=object)
            hs = np.array([hb.get(i, 0) for i in range(n)])
            sep = np.array([p.h_sep.get(b, 0.02) for b in bands])
            leaves = employed & (rng.uniform(size=n) < sep)
            employed = employed & ~leaves
            base = 1.0 / max(p.rehire_mean_months, 1e-6)
            mult = np.ones(n)
            if ym >= CHATGPT_YM:
                # The hire-rate multiplier is SOLVED so that the steady-state
                # employed share falls by exactly exp(gamma): with hazard h and
                # separation s the share is h/(h+s), so multiplying h by
                # m = A s / (h (1 - A)), A = e^g h/(h+s), delivers the target.
                # Scaling the hazard by e^g directly moves the stock by about a
                # tenth of that and is invisible against sampling noise.
                mult = np.array([_hire_mult(base, p.h_sep.get(b, 0.02),
                                            p.gamma.get(b, 0.0)) for b in bands])
                mult = np.where(hs == 1, mult, 1.0)
            hired = (~employed) & (rng.uniform(size=n) < np.clip(base * mult, 0, 1))
            employed = employed | hired
            # a re-hire may move employer
            moved = hired & (rng.uniform(size=n) < 0.5)
            emp_of = np.where(moved, rng.choice(p.n_emp, n, p=w), emp_of)
            idx = np.flatnonzero(employed & np.isin(bands, AGE_BANDS))
            rows.append(pd.DataFrame(dict(pid=idx, ym=ym, employer_id=emp_of[idx])))
        self.pm = pd.concat(rows, ignore_index=True)
        self.pm["year"] = self.pm["ym"].str[:4].astype(int)
        self.emp_size = (self.pm[self.pm["year"] == 2019]
                         .groupby("employer_id")["pid"].nunique())

    # --------------------------------------------------------- register layer
    def _registers(self):
        """Occupation snapshots with survey, carry-forward and imputation; the
        education record with its completion lag; the enrolment register."""
        p, rng = self.p, self.rng
        if not p.distortions:
            self.occ_reg = {y: dict(zip(g["pid"], g["ssyk4"]))
                            for y, g in self.py.groupby("year")}
            self.occ_year = {y: {i: y for i in self.py["pid"].unique()} for y in PROD_YEARS}
            self.edu_reg = {y: dict(zip(self.per["pid"], self.per["grp"]))
                            for y in PROD_YEARS}
            self.exam_reg = {y: dict(zip(self.per["pid"], self.per["exam"]))
                             for y in PROD_YEARS}
            self.gym_of = {int(pid): self.grp_cell[g]
                           for pid, g in zip(self.per["pid"], self.per["grp"])}
            self.gym_grp_of = dict(zip(self.per["pid"].astype(int), self.per["grp"]))
            return
        size = self.emp_size.reindex(range(p.n_emp)).fillna(0)
        emp_by_year = {y: g.drop_duplicates("pid").set_index("pid")["employer_id"].to_dict()
                       for y, g in self.pm.groupby("year")}
        self.occ_reg, self.occ_year = {}, {}
        last_code, last_year = {}, {}
        for y in PROD_YEARS:
            truth = self.py[self.py["year"] == y].set_index("pid")["ssyk4"].to_dict()
            ey = emp_by_year.get(y, {})
            reg, regyear = {}, {}
            for pid, code in truth.items():
                e = ey.get(pid)
                if e is None:
                    continue
                s = size.get(e, 0)
                pc = 1.0 if s >= p.size_always_coded else (
                    p.p_small_coded[1] if s >= 10 else p.p_small_coded[0])
                if rng.uniform() < pc:
                    reg[pid], regyear[pid] = code, y
                    last_code[pid], last_year[pid] = code, y
                elif pid in last_code:
                    reg[pid], regyear[pid] = last_code[pid], last_year[pid]
                elif rng.uniform() < p.p_impute:
                    g = self.per["grp"].iat[pid]
                    tab = self.occ_tab.get((g, "0-2", min(max(y, 2019), 2023)))
                    if tab is not None:
                        reg[pid] = tab[0][rng.choice(len(tab[0]), p=tab[1])]
                        regyear[pid] = y
            self.occ_reg[y], self.occ_year[y] = reg, regyear
        # education: the completion is visible only lag_edu years later
        self.edu_reg, self.exam_reg = {}, {}
        gym = self.gym_cells
        for y in PROD_YEARS:
            done = self.per["exam"].to_numpy() <= (y - p.lag_edu)
            done &= self.per["exam"].to_numpy() > 0
            self.edu_reg[y] = {int(pid): (g if d else None)
                               for pid, g, d in zip(self.per["pid"], self.per["grp"], done)}
            self.exam_reg[y] = {int(pid): (int(e) if d else -9999)
                                for pid, e, d in zip(self.per["pid"], self.per["exam"], done)}
        # a person keeps ONE pre-completion record, so the as-of arm is stable
        self.gym_of = {int(pid): gym[int(pid) % len(gym)] for pid in self.per["pid"]}
        self.gym_grp_of = {int(pid): self.gym_groups[int(pid) % len(self.gym_groups)]
                           for pid in self.per["pid"]} if self.gym_groups else {}

    # -------------------------------------------------------------- emitters
    def _record(self, pid_arr, y: int, vintages):
        """(niva, inr, exam) as a register truncated after `vintages[0]` would
        give it: the first vintage in the list that holds a completion."""
        niva = np.empty(len(pid_arr), dtype=object)
        inr = np.empty(len(pid_arr), dtype=object)
        exam = np.full(len(pid_arr), -9999)
        for j, pid in enumerate(pid_arr):
            g = None
            for v in vintages:
                g = self.edu_reg.get(v, {}).get(int(pid))
                if g is not None:
                    exam[j] = self.exam_reg[v].get(int(pid), -9999)
                    break
            if g is None:
                niva[j], inr[j] = self.gym_of[int(pid)]
            else:
                niva[j], inr[j] = self.grp_cell[g]
        return niva, inr, exam

    def emit_weights(self, year: int) -> pd.DataFrame:
        """47h.pull_weights shape: niva, inr, ssyk4, fresh, expband, young, n."""
        reg, ry = self.occ_reg[year], self.occ_year[year]
        pids = np.fromiter(reg.keys(), dtype=int)
        if len(pids) == 0:
            return pd.DataFrame(columns=["niva", "inr", "ssyk4", "fresh", "expband", "young", "n"])
        niva, inr, exam = self._record(pids, year, [year] if year <= 2023 else [2023, 2022, 2021])
        age = year - self.per["birth"].to_numpy()[pids]
        yrs = np.where(exam > 0, year - exam, np.nan)
        df = pd.DataFrame(dict(
            niva=niva, inr=inr, ssyk4=[reg[p] for p in pids],
            fresh=[1 if ry.get(p) == year else 0 for p in pids],
            expband=exp_band(yrs), young=(age <= 35).astype(int), n=1))
        return (df.groupby(["niva", "inr", "ssyk4", "fresh", "expband", "young"],
                           observed=True)["n"].sum().reset_index())

    def _assignment(self, pids, year, vintages, anchor_T=None):
        niva, inr, exam = self._record(pids, year, vintages)
        yrs = np.where(exam > 0, year - exam, np.nan)
        eb = exp_band(yrs)
        enr = np.array([None] * len(pids), dtype=object)
        if anchor_T is not None:
            lim = min(anchor_T, self.p.hreg_end)
            for j, pid in enumerate(pids):
                lr = self.per["last_reg"].iat[int(pid)]
                if lr > 0 and (lim - 3) <= lr <= lim:
                    g = self.per["enrol_grp"].iat[int(pid)]
                    enr[j] = self.grp_cell[g][1]
        return niva, inr, eb, enr

    def emit_year(self, year: int) -> pd.DataFrame:
        """47h.YEAR_COLS + n_emp: the true record and both truncated records."""
        pm = self.pm[self.pm["year"] == year]
        if pm.empty:
            return pd.DataFrame(columns=list(YEAR_COLS) + ["n_emp"])
        pids = pm["pid"].to_numpy()
        uniq = np.unique(pids)
        cols = {}
        nt, it, et, _ = self._assignment(uniq, year, [min(year, 2023)])
        cols["niva_t"], cols["inr_t"], cols["expb_t"] = nt, it, et
        for T in (2021, 2022):
            v = [y for y in (T, T - 1, T - 2) if y >= 2019]
            n2, i2, e2, a2 = self._assignment(uniq, year, v, anchor_T=T)
            cols[f"niva_{T % 100}"] = n2
            cols[f"inr_{T % 100}"] = i2
            cols[f"expb_{T % 100}"] = e2
            cols[f"enr_{T % 100}"] = a2
        base = pd.DataFrame({"pid": uniq, **cols})
        ages = self.py[self.py["year"] == year].set_index("pid")["age_group"]
        m = pm.merge(base, on="pid", how="left")
        m["age_group"] = m["pid"].map(ages)
        m = m[m["age_group"].isin(AGE_BANDS)]
        m["n_emp"] = 1
        out = (m.rename(columns={"ym": "year_month"})
               .groupby(["employer_id", "year_month"] + [c for c in YEAR_COLS
                        if c not in ("employer_id", "year_month")],
                        observed=True, dropna=False)["n_emp"].sum().reset_index())
        return out

    def emit_production(self, year: int) -> pd.DataFrame:
        """What script 47's estimation sees: own register to 2023, cascade after."""
        pm = self.pm[self.pm["year"] == year]
        if pm.empty:
            return pd.DataFrame(columns=["employer_id", "year_month", "niva", "inr",
                                         "expband", "age_group", "n_emp"])
        uniq = np.unique(pm["pid"].to_numpy())
        v = [min(year, 2023)] if year <= 2023 else [2023, 2022, 2021]
        n2, i2, e2, _ = self._assignment(uniq, year, v)
        base = pd.DataFrame({"pid": uniq, "niva": n2, "inr": i2, "expband": e2})
        ages = self.py[self.py["year"] == year].set_index("pid")["age_group"]
        m = pm.merge(base, on="pid", how="left")
        m["age_group"] = m["pid"].map(ages)
        m = m[m["age_group"].isin(AGE_BANDS)]
        m["n_emp"] = 1
        return (m.rename(columns={"ym": "year_month"})
                .groupby(["employer_id", "year_month", "niva", "inr", "expband",
                          "age_group"], observed=True, dropna=False)["n_emp"]
                .sum().reset_index())

    def emit_oracle(self, year: int) -> pd.DataFrame:
        """employer x month x TRUE quartile x age: the ceiling."""
        pm = self.pm[self.pm["year"] == year]
        tq = self.py[self.py["year"] == year].set_index("pid")[["tq", "age_group"]]
        m = pm.join(tq, on="pid")
        m = m[m["age_group"].isin(AGE_BANDS) & (m["tq"] > 0)]
        m["n_emp"] = 1
        return (m.rename(columns={"ym": "year_month", "tq": "exposure_quartile"})
                .groupby(["employer_id", "year_month", "exposure_quartile",
                          "age_group"], observed=True)["n_emp"].sum().reset_index())


YEAR_COLS = ["employer_id", "year_month", "niva_t", "inr_t", "expb_t",
             "niva_21", "inr_21", "expb_21", "enr_21",
             "niva_22", "inr_22", "expb_22", "enr_22", "age_group"]

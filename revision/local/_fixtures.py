#!/usr/bin/env python3
"""
_fixtures.py -- the synthetic register frames the MONA tests run on.

Extracted from test_47j_synthetic on 20 September 2026, when the fourth
script in a row needed the same two fixtures. One copy means one place to
change when a schema moves, which 47h's has done twice, and it is the
reason a cached frame silently disagreed with the code that read it.

Two families, matching the two SQL pulls the pipeline actually makes:

  EDUCATION (47h shaped)   weights_frame, year_frame -> edu_hr_weights_YYYY
                           and edu_hr_YYYY, the frames 47h caches and 47j,
                           61 and 62 then read.

  OCCUPATION (47L shaped)  baseline_frame -> L_baseline_2019 and
                           counts_frame -> L_counts_YYYY, the 2019 occupation
                           mix and the monthly employment counts.

The two families share their employer numbering and their exposed set, so
a test can plant one shock and look for it through either route. That is
not decoration: script 62 exists to ask whether the occupation route and
the education route disagree, and a fixture whose two routes were
unrelated could not answer it.

Nothing here touches SQL, MONA or the register. Every frame is drawn from
a seeded generator, so two calls with the same arguments return the same
numbers and a test that varies one thing varies exactly one thing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

AGE_W = {"22-25": 0.10, "26-30": 0.13, "31-34": 0.10, "35-40": 0.14,
         "41-49": 0.23, "50+": 0.30}
AGES = list(AGE_W)
INCUMBENT_BANDS = ["31-34", "35-40", "41-49", "50+"]


class Fixture:
    """
    One employer population, drawn once, readable through either route.

    `n_firms` employers numbered from 1; the first `n_exposed` are the
    exposed ones, meaning their YOUNG workers did high-DAIOE work in 2019
    and their incumbents did not. That asymmetry is deliberate: a
    firm-level exposure measure and an age-specific one then disagree
    about the same firms, which is exactly the disagreement script 62 is
    built to decompose.
    """

    def __init__(self, mc, h47, n_firms: int = 140, n_exposed: int = 60,
                 shock: float = 0.65):
        self.mc, self.h47 = mc, h47
        # ScoreBook.build writes score_<design>.csv to 47h's module-level
        # OUT, which is the real output_47h directory in the repo. A local
        # test must not overwrite a recorded MONA output, so point it at
        # the sandbox. Found on 20 Sep 2026, in git status.
        h47.OUT = Path(mc.CACHE_DIR).parent / "h47_out"
        h47.OUT.mkdir(parents=True, exist_ok=True)
        self.n_firms, self.shock = n_firms, shock
        self.exposed = set(range(1, n_exposed + 1))
        self.key = h47.load_key()
        d = pd.read_stata(str(Path(mc.SHARE) / "daioe_quartiles.dta"))
        d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
        self.daioe = d.rename(columns={"pctl_rank_genai": "score"})[
            ["ssyk4", "score"]]
        self.HI = d.loc[d.high_exposure == 1, "ssyk4"].to_numpy()
        self.LO = d.loc[d.high_exposure == 0, "ssyk4"].to_numpy()
        ter = self.key[self.key["niva"].str[:1].isin(["4", "5", "6"])].sample(
            30, random_state=1)
        gym = self.key[self.key["niva"].str[:1] == "3"].sample(12,
                                                               random_state=2)
        self.cells = pd.concat([ter, gym]).reset_index(drop=True)
        self._zmap = None

    # ---- education route (47h shaped) ---------------------------------

    def weights_frame(self, year: int) -> pd.DataFrame:
        rng = np.random.default_rng(1000 + year)
        rows = []
        for _, c in self.cells.iterrows():
            hi = c["niva"][:1] in "456"
            for code in np.concatenate([rng.choice(self.HI, 4),
                                        rng.choice(self.LO, 4)]):
                base = 500 if ((code in set(self.HI)) == hi) else 120
                for band in self.h47.EXP_BANDS:
                    rows.append((c["niva"], c["inr"], code, 1, band, 1,
                                 int(rng.poisson(base)) + 1))
        return pd.DataFrame(rows, columns=["niva", "inr", "ssyk4", "fresh",
                                           "expband", "young", "n"])

    def year_frame(self, year: int, corrupt_young=False, drop_young=False,
                   shock=True, shock_firms=None) -> pd.DataFrame:
        # One seed per YEAR only: two variants of the same year therefore
        # differ in exactly the thing being varied and in nothing else,
        # which is what makes the incumbent-invariance test meaningful.
        rng = np.random.default_rng(2000 + year)
        ters = self.cells[self.cells["niva"].str[:1].isin(list("456"))]
        gyms = self.cells[self.cells["niva"].str[:1] == "3"]
        rows = []
        for emp in range(1, self.n_firms + 1):
            pool = ters if emp in self.exposed else gyms
            mix = pool.sample(min(4, len(pool)), random_state=emp % 97)
            for m in range(1, 13):
                ym = f"{year}-{m:02d}"
                post = ym >= "2022-12"
                for _, c in mix.iterrows():
                    for age, w in AGE_W.items():
                        # Draw for EVERY age, then skip: dropping the young
                        # with a `continue` before the draw shifts the RNG
                        # sequence, so the incumbents differ too and the
                        # invariance test measures the fixture, not the design.
                        lam = 40 * w
                        tgt = self.exposed if shock_firms is None else shock_firms
                        if shock and post and age == "22-25" and emp in tgt:
                            lam *= self.shock
                        n_emp = int(rng.poisson(lam)) + 1
                        if drop_young and age in ("22-25", "26-30"):
                            continue
                        rec = dict(employer_id=emp, year_month=ym,
                                   age_group=age, niva_t=c["niva"],
                                   inr_t=c["inr"], expb_t="3-5", n_emp=n_emp)
                        for T in (2021, 2022):
                            bad = corrupt_young and age in ("22-25", "26-30")
                            g = gyms.iloc[(emp + m) % len(gyms)] if bad else c
                            rec[f"niva_{T%100}"] = g["niva"]
                            rec[f"inr_{T%100}"] = g["inr"]
                            rec[f"expb_{T%100}"] = "3-5"
                            rec[f"enr_{T%100}"] = None
                        # 47h's pull also returns the legacy (47b) cascade
                        # columns. These tests never use that arm, so they
                        # alias the corrected ones; 47h's own gate is where
                        # the two differ.
                        rec["niva_21g"], rec["inr_21g"] = rec["niva_21"], rec["inr_21"]
                        rows.append(rec)
        return self.h47.compact(
            pd.DataFrame(rows)[self.h47.YEAR_COLS + ["n_emp"]])

    def install_edu(self, years, cache=None, min_cell=10, **kw):
        """Write the education caches and return (book, spec, frames)."""
        cache = Path(cache or self.mc.CACHE_DIR)
        for y in (2019, 2020, 2021):
            self.weights_frame(y).to_parquet(
                cache / f"edu_hr_weights_{y}.parquet", index=False)
        for y in years:
            self.year_frame(y, **kw).to_parquet(
                cache / f"edu_hr_{y}.parquet", index=False)
        counts = {y: pd.read_parquet(cache / f"edu_hr_weights_{y}.parquet")
                  for y in (2019, 2020, 2021)}
        self.h47.MIN_CELL = min_cell
        book = self.h47.ScoreBook(counts, self.key, self.h47.load_scores())
        spec = dict(self.h47.DESIGNS["OL_daioe"])
        book.build("OL_daioe", spec)
        frames = {y: pd.read_parquet(cache / f"edu_hr_{y}.parquet")
                  for y in years}
        return book, spec, frames

    # ---- occupation route (47L shaped) --------------------------------

    def baseline_frame(self) -> pd.DataFrame:
        """
        2019 occupation mix per employer x age band. In an exposed firm the
        YOUNG hold high-DAIOE occupations and the incumbents do not, so the
        age-specific and firm-level measures rank the same firm differently.
        """
        rng = np.random.default_rng(11)
        rows = []
        for emp in range(1, self.n_firms + 1):
            for age in AGES:
                young = age in ("22-25", "26-30")
                pool = self.HI if (emp in self.exposed and young) else self.LO
                for code in rng.choice(pool, 3, replace=False):
                    rows.append((emp, age, str(code).zfill(4), "1",
                                 int(rng.integers(4, 15))))
                rows.append((emp, age, "____", "", int(rng.integers(1, 6))))
        return pd.DataFrame(rows, columns=["employer_id", "age_group",
                                           "ssyk4", "ssyk_status", "n"])

    def zmap(self, l47, scores=None) -> dict:
        """(employer, band) -> standardised exposure, on l47's own builder."""
        if self._zmap is None:
            e = l47.build_exposure(self.baseline_frame(),
                                   self.daioe if scores is None else scores)
            mu, sd = e["expo"].mean(), e["expo"].std(ddof=0)
            self._zmap = {(r.employer_id, r.age_group): (r.expo - mu) / (sd or 1)
                          for r in e.itertuples()}
        return self._zmap

    def counts_frame(self, year: int, zmap=None, beta: float = -0.25,
                     bands=("22-25",), from_ym: str = "2022-12",
                     last_month: int = 6) -> pd.DataFrame:
        """
        Monthly employment counts. After `from_ym` a cell in one of `bands`
        has its mean multiplied by exp(beta * z), which is the treatment the
        estimator actually fits, so the coefficient it returns is checkable
        against a number rather than against a sign.
        """
        rng = np.random.default_rng(3000 + year)
        z = zmap or {}
        months = range(1, 13) if year < 2025 else range(1, last_month + 1)
        lam0 = {"22-25": 6, "26-30": 8, "31-34": 7, "35-40": 9,
                "41-49": 12, "50+": 15}
        rows = []
        for emp in range(1, self.n_firms + 1):
            for m in months:
                ym = f"{year}-{m:02d}"
                for age in AGES:
                    lam = lam0[age]
                    if ym >= from_ym and age in bands:
                        lam *= float(np.exp(beta * z.get((emp, age), 0.0)))
                    rows.append((emp, ym, age, int(rng.poisson(lam)) + 1))
        return pd.DataFrame(rows, columns=["employer_id", "year_month",
                                           "age_group", "n_emp"])

    def install_occ(self, years, cache=None, **kw):
        """Write L_baseline_2019 and L_counts_YYYY; return the counts."""
        cache = Path(cache or self.mc.CACHE_DIR)
        self.baseline_frame().to_parquet(cache / "L_baseline_2019.parquet",
                                         index=False)
        out = []
        for y in years:
            c = self.counts_frame(y, **kw)
            c.to_parquet(cache / f"L_counts_{y}.parquet", index=False)
            out.append(c)
        return pd.concat(out, ignore_index=True)

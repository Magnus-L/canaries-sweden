#!/usr/bin/env python3
"""
test_104_payment_rule.py -- the dry run of script 104 (lane 38e).

THE MECHANISM PLANTED. On 97's employer x age band x month grid (240
employers, four DAIOE tiers, tier 3 the top quartile), every counted
person-month carries a cash-pay flag by a deterministic rule: the share
with cash pay is PAY_BASE everywhere except at TOP employers in the LATER
period, where young workers' share falls to PAY_BASE - DROP_YOUNG and older
workers' to PAY_BASE - DROP_OLD. Among person-months without cash pay,
PEN_SHARE carry a pension amount and BEN_SHARE a taxable benefit. The
L_counts caches are written from the same person-months, so the gate must
pass exactly; the SQL pull is replaced by the aggregation of that world.

The 2019 scoring year (P4) is planted on 82's own L_counts_2019 cache
(the floor quantity): incumbents aged 31-69 at TOP employers carry cash pay
at PAY_BASE - DROP19, the rest at PAY_BASE.

Checked: the gate passes on matching counts and STOPS when one year is
missing from the pull; the 2019 gate passes on the cache; P1 reproduces the
planted shares; P2 reproduces the planted top-minus-rest change (-DROP_YOUNG
at 22-25, -DROP_OLD at 31-69); P3 reproduces PEN_SHARE and BEN_SHARE; P4
reproduces DROP19 on the scoring population; the by-year table carries the
exposure group with its 'all' levels; the SQL text uses a probed column
only when present; main() runs end to end, writes all three files, no
identifier leaves, no cell under the floor.

    python3 revision/local/test_104_payment_rule.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("104", ("CANARIES_104_OUT",))
s104 = fx.load("104_payment_rule.py", "s104")
s104.OUT = TMP / "out"
s104.CACHE = mc.CACHE_DIR
check = fx.Check()

EMPS = list(range(1, 241))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS, tier, size, Path(mc.SHARE))
HIGH = {e for e in EMPS if tier(e) == 3}
MONTHS = fx.months()
LAM = {"22-25": 40, "26-30": 45, "31-34": 35, "35-40": 40, "41-49": 50, "50+": 60}
PAY_BASE, DROP_YOUNG, DROP_OLD = 0.97, 0.04, 0.01
PEN_SHARE, BEN_SHARE = 0.6, 0.3
DROP19 = 0.05

E, Bd, M = np.meshgrid(EMPS, list(LAM), MONTHS, indexing="ij")
G = pd.DataFrame({"employer_id": E.ravel(), "age_group": Bd.ravel(), "year_month": M.ravel()})
G["n_emp"] = fx.poisson_same_noise((G["age_group"].map(LAM) * G["employer_id"].map(size)).to_numpy(float), 104)
G["period_label"] = G["year_month"].map(s104.period_of)
G["year"] = G["year_month"].str.slice(0, 4).astype(int)
top = G["employer_id"].isin(HIGH); later = G["period_label"] == "later"
young = G["age_group"].isin(["22-25", "26-30"])
share = np.where(top & later & young, PAY_BASE - DROP_YOUNG,
                 np.where(top & later & ~young, PAY_BASE - DROP_OLD, PAY_BASE))
G["n_pay"] = np.floor(G["n_emp"] * share + 1e-9).astype(int)
G["n_nopay_pension"] = np.floor((G["n_emp"] - G["n_pay"]) * PEN_SHARE + 1e-9).astype(int)
G["n_nopay_benefit"] = np.floor((G["n_emp"] - G["n_pay"]) * BEN_SHARE + 1e-9).astype(int)
fx.write_by_year(G[["employer_id", "year_month", "age_group", "n_emp"]], mc.CACHE_DIR, "L_counts")

# the scoring year: 82's L_counts_2019 cache is the floor quantity, so the
# 2019 world is that cache with the payment flags planted on it
G19 = pd.read_parquet(mc.CACHE_DIR / "L_counts_2019.parquet")
G19["period_label"] = "reference"
G19["year"] = 2019
top19 = G19["employer_id"].isin(HIGH); old19 = G19["age_group"].isin(["31-34", "35-40", "41-49", "50+"])
share19 = np.where(top19 & old19, PAY_BASE - DROP19, PAY_BASE)
G19["n_pay"] = np.floor(G19["n_emp"] * share19 + 1e-9).astype(int)
G19["n_nopay_pension"] = np.floor((G19["n_emp"] - G19["n_pay"]) * PEN_SHARE + 1e-9).astype(int)
G19["n_nopay_benefit"] = np.floor((G19["n_emp"] - G19["n_pay"]) * BEN_SHARE + 1e-9).astype(int)


def fake_pull(year, conn, cols, drop_year=None):
    if year == drop_year:
        return G.iloc[0:0][["employer_id", "period_label", "age_group", "n_emp", "n_pay",
                             "n_nopay_pension", "n_nopay_benefit"]].assign(year=year)
    src = G19 if year == 2019 else G
    d = (src[src["year"] == year]
         .groupby(["employer_id", "period_label", "age_group"], observed=True)
         [["n_emp", "n_pay", "n_nopay_pension", "n_nopay_benefit"]].sum().reset_index())
    d["year"] = year
    return d


ALL_COLS = {s104.PAY_COL, s104.PENSION_COL, *s104.BENEFIT_COLS, "PERIOD"}
s104.open_conn = lambda: object()
s104.probe_columns = lambda conn: ALL_COLS
s104.pull_year = fake_pull

print("\n--- the SQL text follows the probe ---")
q_all, q_pay = s104.year_sql(2023, ALL_COLS), s104.year_sql(2023, {s104.PAY_COL})
check("pension and benefit flags are built only from probed columns",
      f"agi.{s104.PENSION_COL}" in q_all and f"agi.{s104.PENSION_COL}" not in q_pay
      and "0 AS has_pension" in q_pay and "SP_BILFORMAN" in q_all and "SP_BILFORMAN" not in q_pay)
check("the pull collapses to one row per employer, person and month before counting",
      "GROUP BY employer_id, period, person_id, fodelse" in q_all and "COUNT(*) AS n_emp" in q_all)
check("twelve monthly tables in 2023, six in 2025",
      q_all.count("Arb_AGIIndivid2023") == 12 and s104.year_sql(2025, ALL_COLS).count("_prel") == 6)

print("\n--- the gate ---")
pulled = pd.concat([fake_pull(y, None, ALL_COLS) for y in s104.YEARS], ignore_index=True)
s104.FAILURES.clear()
s104.gate(pulled)
check("the gate passes on the paper's own counts", not s104.FAILURES)
short = pd.concat([fake_pull(y, None, ALL_COLS, drop_year=2024) for y in s104.YEARS], ignore_index=True)
s104.FAILURES.clear()
try:
    s104.gate(short); stopped = False
except SystemExit:
    stopped = True
check("the gate STOPS when a year is missing from the pull", stopped)
s104.FAILURES.clear()
s104.GATE_LINES.clear()
pulled19 = fake_pull(2019, None, ALL_COLS)
s104.gate(pulled19, years=[2019], label=" (2019)")
check("the 2019 gate passes on 82's L_counts_2019 cache", not s104.FAILURES
      and any("(2019) PASSES" in ln for ln in s104.GATE_LINES))

print("\n--- the shares ---")
s82, l47, l70, j47 = s104.load_modules()
for m_ in (s82, l47, l70, j47):
    m_.OUT, m_.CACHE = s104.OUT, mc.CACHE_DIR
expo = s82.build_exposure(l47, l70, j47, audit=False)["exposure"]
check("the planted tier is the top quartile", set(expo.loc[expo["fq"] == 4, "employer_id"]) == HIGH)
sh = s104.shares(s104.attach_group(pulled, expo))
p = lambda per, band, grp: s104.pick(sh, per, band, grp)  # noqa: E731
def planted(per, bands, grp_top):
    sub = G[(G["period_label"] == per) & (G["age_group"].isin(bands)) & (G["employer_id"].isin(HIGH) == grp_top)]
    return sub["n_pay"].sum() / sub["n_emp"].sum()
BANDS = {"22-25": ["22-25"], "26-30": ["26-30"], "31-69": ["31-34", "35-40", "41-49", "50+"]}
check("P1: every period x band x group share equals the planted world's own ratio",
      all(abs(p(per, band, grp) - planted(per, BANDS[band], grp == "top")) < 1e-9
          for per in s104.PERIODS for band in BANDS for grp in ("top", "rest")))
check("P1: rest employers sit near the base share (floor rounding lowers it slightly)",
      all(PAY_BASE - 0.01 < p(per, band, "rest") <= PAY_BASE for per in s104.PERIODS for band in BANDS))
check("P1: the planted later-period drops at top employers are read",
      abs(p("later", "22-25", "top") - (PAY_BASE - DROP_YOUNG)) < 0.003
      and abs(p("later", "31-69", "top") - (PAY_BASE - DROP_OLD)) < 0.003,
      f"{p('later', '22-25', 'top'):.4f}, {p('later', '31-69', 'top'):.4f}")
d_young = 100 * ((p("later", "22-25", "top") - p("later", "22-25", "rest")) - (p("interim", "22-25", "top") - p("interim", "22-25", "rest")))
d_old = 100 * ((p("later", "31-69", "top") - p("later", "31-69", "rest")) - (p("interim", "31-69", "top") - p("interim", "31-69", "rest")))
check("P2: the later-minus-interim change in top-minus-rest reproduces the planted drops",
      abs(d_young + 100 * DROP_YOUNG) < 0.35 and abs(d_old + 100 * DROP_OLD) < 0.35,
      f"{d_young:+.3f} pp vs {-100 * DROP_YOUNG:+.1f}; {d_old:+.3f} vs {-100 * DROP_OLD:+.1f}")
sub = sh[(sh["band"] == "all") & (sh["group"] == "all")]
nopay = (sub["n_emp"] - sub["n_pay"]).sum()
gn = (G["n_emp"] - G["n_pay"]).sum()
check("P3: pension and benefit shares among no-pay records equal the planted world's ratios",
      abs(sub["n_nopay_pension"].sum() / nopay - G["n_nopay_pension"].sum() / gn) < 1e-9
      and abs(sub["n_nopay_benefit"].sum() / nopay - G["n_nopay_benefit"].sum() / gn) < 1e-9
      and 0 < sub["n_nopay_pension"].sum() / nopay <= PEN_SHARE,
      f"pension {sub['n_nopay_pension'].sum() / nopay:.3f} (planted {PEN_SHARE} before floor rounding on small cells)")
check("unscored group is empty in this world and 'all' equals top plus rest",
      (sh[sh["group"] == "unscored"]["n_emp"].sum() == 0)
      and int(sh[(sh["band"] == "all") & (sh["group"] == "all")]["n_emp"].sum())
      == int(sh[(sh["band"] == "all") & (sh["group"].isin(["top", "rest"]))]["n_emp"].sum()))
yr = s104.by_year(s104.attach_group(pulled, expo))
def yr_share(y, b, g):
    r = yr[(yr["year"] == y) & (yr["band"] == b) & (yr["group"] == g)]
    return float(r["share_pay"].iloc[0]) if len(r) == 1 else float("nan")
def world_share(y, bands, grp):
    sub = G[(G["year"] == y) & (G["age_group"].isin(bands))]
    if grp == "top":
        sub = sub[sub["employer_id"].isin(HIGH)]
    elif grp == "rest":
        sub = sub[~sub["employer_id"].isin(HIGH)]
    return sub["n_pay"].sum() / sub["n_emp"].sum()
ALLB = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
check("by-year table: year x band x group with 'all' on both, every cell the planted world's own ratio",
      set(yr["group"]) == {"top", "rest", "all"} and set(yr["band"]) == {"22-25", "26-30", "31-69", "all"}
      and len(yr) == 5 * 4 * 3
      and all(abs(yr_share(y, b, g) - world_share(y, BANDS.get(b, ALLB), g)) < 1e-9
              for y in s104.YEARS for b in ("22-25", "26-30", "31-69", "all") for g in ("top", "rest", "all")),
      f"{len(yr)} rows; groups {sorted(set(yr['group']))}")
sc = s104.scoring_population(s104.attach_group(pulled19, expo))
def sc_share(g):
    r = sc[sc["group"] == g]; return float(r["share_pay"].iloc[0])
def w19(grp):
    sub = G19[G19["age_group"].isin(BANDS["31-69"])]
    sub = sub[sub["employer_id"].isin(HIGH)] if grp == "top" else (sub[~sub["employer_id"].isin(HIGH)] if grp == "rest" else sub)
    return sub["n_pay"].sum() / sub["n_emp"].sum()
check("P4: the scoring population's share equals the planted 2019 world's ratio in every group",
      all(abs(sc_share(g) - w19(g)) < 1e-9 for g in ("top", "rest", "all")) and len(sc) == 4)
check("P4: top incumbents sit DROP19 below the rest (floor rounding aside)",
      abs((sc_share("rest") - sc_share("top")) - DROP19) < 0.004,
      f"rest {sc_share('rest'):.4f} top {sc_share('top'):.4f}")

print("\n--- main(), end to end ---")
s104.NOTES.clear(); s104.FAILURES.clear()
_stdout = sys.stdout
rc = s104.main()
sys.stdout = _stdout
out = pd.read_csv(s104.OUT / "payment_rule.csv"); yr2 = pd.read_csv(s104.OUT / "payment_rule_by_year.csv")
sc2 = pd.read_csv(s104.OUT / "payment_rule_2019_incumbents.csv")
summ = (s104.OUT / "104_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s104.FAILURES}")
check("all three export files written with the expected shape (no unscored employer in this world)",
      len(out) == 4 * 3 * 4 and len(yr2) == 5 * 4 * 3 and len(sc2) == 4,
      f"{len(out)} rows, {len(yr2)} rows, {len(sc2)} rows; groups {sorted(set(out['group']))}, bands {sorted(set(out['band']))}")
check("the summary prints both gates, P1, P2, P3, P4 and the by-year block",
      "THE GATE PASSES" in summ and "THE GATE (2019) PASSES" in summ and "P1. SHARE" in summ
      and "later minus interim" in summ and "P3. AMONG" in summ and "P4. THE SCORING POPULATION" in summ
      and "BY YEAR x BAND x GROUP" in summ)
check("the summary says which probed columns were found", "FOUND" in summ and "found" in summ)
check("no identifier column is exported", not any("employer_id" in f.columns for f in (out, yr2, sc2)))
check("no cell under the floor", all(bool((f["n_employers"].isna() | (f["n_employers"] >= 5)).all()) for f in (out, yr2, sc2)))

print("\n--- the columns-absent path ---")
s104.probe_columns = lambda conn: {s104.PAY_COL, "PERIOD"}
s104.NOTES.clear(); s104.FAILURES.clear()
_stdout = sys.stdout
rc = s104.main()
sys.stdout = _stdout
summ = (s104.OUT / "104_summary.txt").read_text()
check("without the pension column main() still returns 0 and the summary says ABSENT",
      rc == 0 and "ABSENT" in summ)
check.done()

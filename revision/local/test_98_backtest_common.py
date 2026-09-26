#!/usr/bin/env python3
"""
test_98_backtest_common.py -- the dry run of script 98 (lane 37c, second
                              stage).

THE MECHANISM PLANTED. Script 45's dual panel: every worker carries the
own-year code (true) and the code a register truncated at T would give
(as-of). Each employer holds young workers in one top-quartile occupation
and one lower-quartile occupation, so it sits in both cells the submitted
design compares. Up to T the as-of code is the true one; after T, among
the young workers whose true code is top-quartile:

  DROP world    a share DROP get no as-of code at all (the truncated
                register cannot code them): pure SAMPLE INCLUSION.
  STALE world   a share STALE get a stale lower-quartile code (their
                pre-degree record, as script 45 measured): pure CODING.

The worlds share every uniform. The decomposition must put the artefact
where it was planted: in the DROP world the coding piece is about zero
and the sample-inclusion piece carries it; in the STALE world the
reverse. The three pieces sum to the artefact exactly in both. Also: a
handful of workers have no own-year code but an as-of one, which is the
third piece; the gate stops on 45's published numbers; main() runs end to
end on the caches with no SQL.

    python3 revision/local/test_98_backtest_common.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fixtures_trip37 as fx  # noqa: E402

mc, TMP = fx.sandbox("98", ("CANARIES_98_OUT",))
s98 = fx.load("98_backtest_common.py", "s98")
s98.OUT = TMP / "out"
check = fx.Check()

D = mc.load_daioe()
Q4 = D.loc[D["exposure_quartile"] == 4, "ssyk4"].iloc[0]
LO = D.loc[D["exposure_quartile"] == 2, "ssyk4"].iloc[0]
STALE_CODE = D.loc[D["exposure_quartile"] == 1, "ssyk4"].iloc[0]
EMPS = list(range(1, 121))
MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2024) for m in range(1, 13)]
DROP, STALE = 0.25, 0.25
# Employers whose young cell in the LOWER exposure group is zero in every
# month, while their top-quartile young cell is populated. 98 keeps them
# (they employ young workers), the zero-filled panel carries their lower
# cell in every month, and fixest drops that all-zero employer-by-group
# block: these are the cells the post-fit count must report as dropped.
# Zeroing BOTH groups does not test the mechanism, because 98 excludes an
# employer with no young workers before the fit. Set below; empty for the
# two decomposition worlds. None divisible by 10, so the 'extra' as-of
# rows never repopulate the cell.
ZERO_EMPS: set = set()


def world(kind: str, trunc: int) -> pd.DataFrame:
    rows = []
    for e in EMPS:
        for ym in MONTHS:
            late = int(ym[:4]) > trunc
            for age, lam in (("22-25", 20.0), ("41-49", 20.0)):
                for code in (Q4, LO):
                    rows.append((e, ym, code, age, lam, late))
    d = pd.DataFrame(rows, columns=["employer_id", "year_month", "ssyk_true",
                                    "age_group", "lam", "late"])
    n = fx.poisson_same_noise(d["lam"].to_numpy(), 98)
    n = np.where(d["employer_id"].isin(ZERO_EMPS) & (d["age_group"] == "22-25")
                 & (d["ssyk_true"] == LO), 0, n)
    u = np.random.default_rng(980).random(len(d))
    hit = (d["late"] & (d["ssyk_true"] == Q4) & (d["age_group"] == "22-25")
           ).to_numpy()
    moved = np.where(hit, np.floor(n * (DROP if kind == "drop" else STALE)
                                   + u), 0).astype(int)
    moved = np.minimum(moved, n)
    keep = d.assign(ssyk_asof=d["ssyk_true"], n_emp=n - moved)
    gone = d.assign(ssyk_asof=("____" if kind == "drop" else STALE_CODE),
                    n_emp=moved)
    # a few workers with no own-year code whom the as-of vintage codes
    extra = d[(d["employer_id"] % 10 == 0) & (d["ssyk_true"] == LO)].assign(
        ssyk_true="____", ssyk_asof=LO, n_emp=2)
    out = pd.concat([keep, gone[gone["n_emp"] > 0], extra], ignore_index=True)
    return out[s98.DUAL_COLS]


def install(kind: str) -> None:
    for t in s98.TRUNCATIONS:
        world(kind, t).to_parquet(mc.CACHE_DIR / f"panel_dual_T{t}.parquet",
                                  index=False)


s98.SKIP_HEADLINE_GATE = True     # the headline gate is tested at the end
RES = {}
for kind in ("drop", "stale"):
    print(f"\n=== the {kind.upper()} world ===")
    install(kind)
    s98.ROWS.clear(); s98.FAILURES.clear(); s98.NOTES.clear()
    keep = s98.check_gate
    s98.check_gate = lambda trunc: None
    daioe = mc.load_daioe()
    scorable = set(daioe["ssyk4"].astype(str))
    for trunc in s98.TRUNCATIONS:
        s98.run_cutoff(trunc, daioe, scorable)
    s98.check_gate = keep
    check(f"{kind}: every fit came back", not s98.FAILURES,
          "; ".join(s98.FAILURES))
    d = {n: s98.val(2021, n, "harmonised") for n, _, _ in s98.DIFFS}
    RES[kind] = d
    print(f"  differences T2021: {d}")
    ca, _ = d["C_minus_A"]
    check(f"{kind}: B-A plus C-B is C-A",
          abs(d["B_minus_A"][0] + d["C_minus_B"][0] - ca) < 1e-9)
    check(f"{kind}: every difference has a standard error",
          all(x[1] == x[1] and x[1] > 0 for x in d.values()),
          str({k: round(v[1], 4) for k, v in d.items()}))
    check(f"{kind}: there is an artefact to decompose", ca < -0.05,
          f"{ca:+.4f}")
    stacked_ok = not any("differs from its separate fit" in n
                         for n in s98.NOTES)
    check(f"{kind}: the stacked fit reproduces the separate harmonised fits",
          stacked_ok, "; ".join(s98.NOTES))
    if kind == "drop":
        check("drop: coding on fixed workers (C-B) is about zero",
              abs(d["C_minus_B"][0]) < 0.01, f"{d['C_minus_B'][0]:+.4f}")
        check("drop: sample inclusion (B-A) carries the artefact",
              d["B_minus_A"][0] / ca > 0.9, f"{d['B_minus_A'][0]:+.4f}")
    else:
        check("stale: sample inclusion (B-A) is about zero",
              abs(d["B_minus_A"][0]) < 0.01, f"{d['B_minus_A'][0]:+.4f}")
        check("stale: coding (C-B) carries the artefact",
              d["C_minus_B"][0] / ca > 0.9, f"{d['C_minus_B'][0]:+.4f}")
    WORLD_GATE = {(t, a): s98.val(t, a) for t in s98.TRUNCATIONS
                  for a in ("A", "asof_all")}

print("\n--- POST-FIT SUPPORT: nine employers with no young workers in the lower group ---")
ZERO_EMPS = set(range(111, 120))
install("stale")
s98.ROWS.clear(); s98.FAILURES.clear(); s98.NOTES.clear()
keep = s98.check_gate
s98.check_gate = lambda trunc: None
daioe = mc.load_daioe()
s98.run_cutoff(2021, daioe, set(daioe["ssyk4"].astype(str)))
s98.check_gate = keep
check("zero world: every fit came back", not s98.FAILURES, "; ".join(s98.FAILURES))
a_own = next(r for r in s98.ROWS if r["trunc"] == 2021 and r["arm"] == "A"
             and r["support"] == "own")
planted = len(ZERO_EMPS) * len(MONTHS)              # employers x months, one group
check("zero world: the post-fit count is populated",
      a_own["cells_used"] == a_own["cells_used"], str(a_own))
# fixest removes the all-zero employer-by-group block (planted cells) and
# then the top-quartile cells it leaves alone in their employer-by-month
# groups (singletons): twice the planted count in the true-code arms.
check("zero world: the fit dropped the planted block plus the singletons it leaves",
      a_own["cells"] - a_own["cells_used"] == 2 * planted,
      f"cells {a_own['cells']:,}, used {a_own['cells_used']}, planted {planted:,}")
harm = [r for r in s98.ROWS if r["trunc"] == 2021 and r["support"] == "harmonised"
        and r["arm"] in s98.STACK]
drops = {r["arm"]: r["cells"] - r["cells_used"] for r in harm}
# In the STALE world the as-of arm C carries a third cell per employer-
# month (the stale code's quartile), so removing the all-zero block leaves
# no singleton and C drops the block alone: the support differs between
# arms, which is what the SUPPORT line is for.
check("zero world: A and B drop block plus singletons, C drops the block alone",
      drops.get("A") == 2 * planted and drops.get("B") == 2 * planted
      and drops.get("C") == planted, str(drops))
s98.save()
s98.write_summary()
summ0 = (s98.OUT / "98_summary.txt").read_text()
check("zero world: the summary prints the non-zero drop",
      f"dropped by PPML {2 * planted:,}" in summ0 and f"dropped by PPML {planted:,}" in summ0, summ0[summ0.find("CUTOFF 2021"):][:400])
check("zero world: the summary states the stacked and separate support",
      "SUPPORT T2021" in summ0, "\n".join(l for l in summ0.splitlines() if "SUPPORT" in l))
ZERO_EMPS = set()
install("stale")                                 # restore the world main() is gated on

print("\n--- the backtest gate stops on 45's published numbers ---")
try:
    s98.check_gate(2021); stopped = False
except SystemExit:
    stopped = True
check("the gate STOPS when the arms do not reproduce 45", stopped)

print("\n--- main(), end to end, STALE world, gate pointed at it ---")
s98.GATE = WORLD_GATE
s98.ROWS.clear(); s98.FAILURES.clear(); s98.NOTES.clear()
_stdout = sys.stdout
rc = s98.main()
sys.stdout = _stdout
out = pd.read_csv(s98.OUT / "backtest_common.csv")
summ = (s98.OUT / "98_summary.txt").read_text()
check("main() returns 0", rc == 0, f"rc {rc}; {s98.FAILURES}")
check("the export carries 4 own + 3 harmonised arms + 3 differences a cutoff",
      len(out) == 2 * (4 + 3 + 3), f"{len(out)} rows")
check("the summary documents estimator, dates, cascade and PPML drops",
      "Pseudo-dates" in summ and "cascade" in summ and "dropped by PPML" in summ)
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ssyk4",
                                         "ssyk_true", "ssyk_asof")))

print("\n--- the headline gate runs first and stops on Table 1's numbers ---")
EMPS2 = list(range(1, 121))
tier = lambda e: e % 4                                          # noqa: E731
size = lambda e: 4.2 if tier(e) == 3 else 1 + (e // 4) % 5     # noqa: E731
fx.install_score(mc, EMPS2, tier, size, Path(mc.SHARE))
LAM = {"22-25": 8, "26-30": 9, "31-34": 7, "35-40": 8, "41-49": 10, "50+": 12}
g = pd.DataFrame([(e, ym, a, lam * size(e)) for e in EMPS2
                  for ym in fx.months() for a, lam in LAM.items()],
                 columns=["employer_id", "year_month", "age_group", "lam"])
g["n_emp"] = fx.poisson_same_noise(g["lam"].to_numpy(), 981)
fx.write_by_year(g.drop(columns="lam"), mc.CACHE_DIR, "L_counts")
s98.ROWS.clear(); s98.FAILURES.clear()
try:
    s98.headline_gate(); stopped = False
except SystemExit:
    stopped = True
check("the headline gate fits and STOPS against Table 1 in a null world",
      stopped and any(r["arm"] == "headline_tau_22_25" for r in s98.ROWS))
check.done()

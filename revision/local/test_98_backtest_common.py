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


RES = {}
for kind in ("drop", "stale"):
    print(f"\n=== the {kind.upper()} world ===")
    install(kind)
    s98.ROWS.clear(); s98.FAILURES.clear()
    keep = s98.check_gate
    s98.check_gate = lambda trunc: None
    daioe = mc.load_daioe()
    scorable = set(daioe["ssyk4"].astype(str))
    for trunc in s98.TRUNCATIONS:
        panel = s98.dual_panel(trunc)
        for arm in s98.ARMS:
            k, col = s98.arm_rows(panel, arm, scorable)
            s98.ROWS.append(s98.estimate(panel, k, col, daioe, trunc, arm))
    s98.check_gate = keep
    check(f"{kind}: every fit came back", not s98.FAILURES,
          "; ".join(s98.FAILURES))
    pc = s98.pieces(2021)
    RES[kind] = pc
    print(f"  pieces T2021: {pc}")
    parts = pc["sample_inclusion"] + pc["coding_same_workers"] \
        + pc["asof_only_workers"]
    check(f"{kind}: the three pieces sum to the artefact",
          abs(parts - pc["artefact"]) < 1e-12)
    check(f"{kind}: there is an artefact to decompose", pc["artefact"] < -0.05,
          f"{pc['artefact']:+.4f}")
    if kind == "drop":
        check("drop: the coding piece is about zero",
              abs(pc["coding_same_workers"]) < 0.01,
              f"{pc['coding_same_workers']:+.4f}")
        check("drop: sample inclusion carries the artefact",
              pc["sample_inclusion"] / pc["artefact"] > 0.9,
              f"{pc['sample_inclusion']:+.4f}")
    else:
        check("stale: the sample-inclusion piece is about zero",
              abs(pc["sample_inclusion"]) < 0.01,
              f"{pc['sample_inclusion']:+.4f}")
        check("stale: coding carries the artefact",
              pc["coding_same_workers"] / pc["artefact"] > 0.9,
              f"{pc['coding_same_workers']:+.4f}")
    WORLD_GATE = {(t, a): s98.val(t, a) for t in s98.TRUNCATIONS
                  for a in ("true_all", "asof_all")}

print("\n--- the gate stops on 45's published numbers ---")
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
check("the export carries four arms and three pieces per truncation",
      len(out) == 2 * (4 + 4), f"{len(out)} rows")
check("the pieces keep their values (not suppressed by the floor)",
      bool(out.loc[out["arm"].str.startswith("piece_"), "gamma2"]
           .notna().all()))
check("the summary reports the shares", "of the artefact" in summ)
check("no identifier column is exported",
      not any(c in out.columns for c in ("employer_id", "ssyk4",
                                         "ssyk_true", "ssyk_asof")))
check.done()

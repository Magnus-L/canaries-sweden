#!/usr/bin/env python3
"""
test_52_slim.py -- 52 must shrink the files without changing any number the
analysis computes from them.

  1 the collapsed m6 reproduces the SAME weighted mean DAIOE per
    (grp, expband, fresh) as the full occupation-level file
  2 the collapsed m7 reproduces the same per-cell quartile counts
  3 files already under the cap are copied through byte-identical
  4 the floor is raised to 10 and suppressed rows are DROPPED, not blanked
  5 one unreadable file does not stop the rest
"""
import importlib.util, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries52_"))
SHARE = TMP / "input"; SHARE.mkdir()
shutil.copy(UPLOAD / "daioe_quartiles.dta", SHARE / "daioe_quartiles.dta")
os.environ["CANARIES_SHARE"] = str(SHARE)
sys.path.insert(0, str(MONA))
spec = importlib.util.spec_from_file_location("s52", MONA / "52_slim_exports.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.SRC = TMP / "output_50"; mod.SRC.mkdir()
mod.OUT = TMP / "output_50_slim"; mod.OUT.mkdir()
mod.DAIOE_PATH = str(SHARE / "daioe_quartiles.dta")
R = np.random.default_rng(52)
D = pd.read_stata(SHARE / "daioe_quartiles.dta")
CODES = D["ssyk4"].astype(str).str.zfill(4).to_numpy()
FAILS = []


def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else ""))
    if not c:
        FAILS.append(n)


def make():
    rows = []
    for g in [f"g{i:02d}" for i in range(20)]:
        for b in ("0-2", "3-5", "6-10"):
            for fr in (0, 1):
                for code in R.choice(CODES, 12, replace=False):
                    rows.append((g, b, fr, code, int(R.integers(1, 400))))
    m6 = pd.DataFrame(rows, columns=["grp", "expband", "fresh", "ssyk4", "n"])
    rows = []
    for age in ("22-25", "50+"):
        for g in [f"g{i:02d}" for i in range(12)]:
            for ter in (0, 1):
                for code in R.choice(CODES, 10, replace=False):
                    rows.append((age, g, "3-5", ter, None if ter else "1234",
                                 code, int(R.integers(1, 300))))
    m7 = pd.DataFrame(rows, columns=["age_group", "grp_lag", "expband_lag",
                                     "tertiary_lag", "enr_inr", "ssyk4_t", "n"])
    m6.to_csv(mod.SRC / "m6_matrix_2019.csv", index=False)
    m7.to_csv(mod.SRC / "m7_validation_t2021_k0.csv", index=False)
    pd.DataFrame({"size_band": ["1-4", "5-9"], "n": [10, 20],
                  "persons": [30, 180]}).to_csv(mod.SRC / "m5a_employer_size.csv",
                                                index=False)
    return m6, m7


if __name__ == "__main__":
    m6, m7 = make()
    mod.main()
    daioe = mod.load_daioe()

    # 1. the weighted mean survives the collapse exactly.
    #    The floor applies to the CELL that is exported, not to the occupation
    #    rows that go into it: dropping small occupation rows before
    #    aggregating would change the statistic, which is the opposite of what
    #    this script is for. So the reference aggregates first, then floors.
    full = m6.merge(daioe, on="ssyk4", how="inner")
    full["ws"] = full.n * full.score
    ref = (full.groupby(["grp", "expband", "fresh"])
           .apply(lambda d: pd.Series({"mean_ref": d.ws.sum() / d.n.sum(),
                                       "n_ref": d.n.sum()}),
                  include_groups=False).reset_index())
    ref = ref[ref.n_ref >= mod.FLOOR_SLIM]
    slim = pd.read_csv(mod.OUT / "m6_matrix_2019.csv")
    slim["mean_slim"] = slim.ws / slim.n
    j = ref.merge(slim, on=["grp", "expband", "fresh"])
    check("the collapsed m6 gives the identical weighted mean DAIOE",
          len(j) > 30 and np.allclose(j.mean_ref, j.mean_slim),
          f"{len(j)} cells, max diff "
          f"{float((j.mean_ref - j.mean_slim).abs().max()):.2e}")

    # 2. m7's quartile counts survive
    f7 = m7.merge(daioe, left_on="ssyk4_t", right_on="ssyk4", how="inner")
    cell7 = (f7.groupby(["age_group", "grp_lag", "expband_lag", "tertiary_lag",
                         "enr_inr", "q"], dropna=False)["n"].sum().reset_index())
    ref7 = (cell7[cell7.n >= mod.FLOOR_SLIM]
            .groupby(["age_group", "q"])["n"].sum().rename("n_ref").reset_index())
    s7 = pd.read_csv(mod.OUT / "m7_validation_t2021_k0.csv")
    s77 = s7.groupby(["age_group", "q_true"])["n"].sum().rename("n_slim").reset_index()
    j7 = ref7.merge(s77, left_on=["age_group", "q"], right_on=["age_group", "q_true"])
    check("the collapsed m7 preserves the quartile counts",
          len(j7) > 0 and (j7.n_ref == j7.n_slim).all(), str(len(j7)))

    # 3, 4, 5
    a = (mod.SRC / "m5a_employer_size.csv").read_bytes()
    b = (mod.OUT / "m5a_employer_size.csv").read_bytes()
    check("small files are copied through byte-identical", a == b)
    for f in ("m6_matrix_2019.csv", "m7_validation_t2021_k0.csv"):
        d = pd.read_csv(mod.OUT / f)
        check(f"{f[:14]}: floor of 10 applied", d["n"].min() >= mod.FLOOR_SLIM,
              str(d["n"].min()))
        check(f"{f[:14]}: no blanked rows left", d["n"].notna().all())
        check(f"{f[:14]}: it is smaller",
              (mod.OUT / f).stat().st_size < (mod.SRC / f).stat().st_size,
              f"{(mod.SRC/f).stat().st_size} -> {(mod.OUT/f).stat().st_size}")
    (mod.SRC / "m6_matrix_2020.csv").write_text("this,is,not\na,valid\n")
    mod.main()
    check("one unreadable file does not stop the rest",
          (mod.OUT / "m6_matrix_2019.csv").exists()
          and "FAILED" in (mod.OUT / "52_log.txt").read_text())
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""test_51_synthetic.py -- 51 end to end locally, no MONA, no R."""
import importlib.util, os, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA = HERE.parent / "mona"
TMP = Path(tempfile.mkdtemp(prefix="canaries51_"))
os.environ["CANARIES_SHARE"] = str(TMP)
sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402
spec = importlib.util.spec_from_file_location("s51", MONA / "51_vintage_ai_unboxed.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.OUT = TMP / "output_51"; mod.OUT.mkdir()
R = np.random.default_rng(51)
FAILS = []


def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else ""))
    if not c:
        FAILS.append(n)


def f_code_age(y, conn):
    rows = []
    for s1 in "12345678":
        for age in ("22-29", "30-49", "50+"):
            for ca in range(0, 4):
                # clerical (4) is deliberately staler, so the test can see it
                lam = 900 if ca == 0 else 300
                if s1 == "4" and ca > 0:
                    lam = 700
                rows.append((s1, age, ca, "1", int(R.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=["ssyk1", "age_group", "code_age",
                                       "ssyk_status", "n"])


def f_flows(t, conn):
    rows = []
    for ct in (0, 1):
        for c1 in (0, 1):
            for ft in (0, 1):
                for f1 in (0, 1):
                    for age in ("22-29", "30-49", "50+"):
                        rows.append((ct, c1, ft, f1, age, int(R.poisson(200)) + 1))
    return pd.DataFrame(rows, columns=["clerical_t", "clerical_t1", "fresh_t",
                                       "fresh_t1", "age_group", "n"])


def test_ok():
    mc.connect = lambda: object()
    mod.q_code_age, mod.q_clerical_flows = f_code_age, f_flows
    mod.main()
    ca = pd.read_csv(mod.OUT / "code_age_by_occupation.csv")
    fl = pd.read_csv(mod.OUT / "clerical_flows_by_freshness.csv")
    for df, nm in ((ca, "code_age"), (fl, "flows")):
        v = df["n"].dropna()
        check(f"{nm} export is floored", ((v == 0) | (v >= 5)).all())
    s = (mod.OUT / "51_summary.txt").read_text()
    check("the summary reports the fresh share per year",
          all(str(y) in s for y in mod.YEARS))
    check("it reports clerical separately", "clerical" in s)
    check("it states the attenuation direction", "at least as large" in s)
    check("it disclaims re-estimating AI Unboxed", "does not re-estimate" in s)


def test_one_year_failing_does_not_kill_it():
    mc.connect = lambda: object()
    def some_fail(y, conn):
        if y == 2021:
            raise RuntimeError("synthetic failure")
        return f_code_age(y, conn)
    mod.q_code_age = some_fail
    mod.q_clerical_flows = lambda t, conn: (_ for _ in ()).throw(RuntimeError("no flows"))
    mod.OUT = TMP / "out2"; mod.OUT.mkdir()
    mod.main()
    ca = pd.read_csv(mod.OUT / "code_age_by_occupation.csv")
    check("a failed year costs that year only", set(ca["year"]) == {2019, 2020, 2022, 2023},
          str(sorted(set(ca["year"]))))
    check("a failed flow query does not stop the summary",
          (mod.OUT / "51_summary.txt").exists()
          and not (mod.OUT / "clerical_flows_by_freshness.csv").exists())


if __name__ == "__main__":
    test_ok(); test_one_year_failing_does_not_kill_it()
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

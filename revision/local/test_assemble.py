#!/usr/bin/env python3
"""
test_assemble.py -- vet the assembler on fabricated MONA outputs.

The assembler turns numbers into sentences, so the thing to test is that
the sentence follows the number: a clean design must be reported as
supported, a lag-contaminated one as closed, a missing file as pending,
and the verdict must come from the pre-committed rule rather than from
whichever way the coefficient happens to point.

    python3 revision/local/test_assemble.py
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
spec = importlib.util.spec_from_file_location("asm", REV / "assemble.py")
asm = importlib.util.module_from_spec(spec); spec.loader.exec_module(asm)
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_rule():
    """The verdict must be the pre-committed rule, at its exact boundaries."""
    check("clean at both truncations", asm.verdict(0.02, -0.03) == "CLEAN")
    check("0.05 is not clean", asm.verdict(0.05, 0.01) == "USABLE WITH CAVEAT")
    check("caveat band", asm.verdict(-0.12, 0.07) == "USABLE WITH CAVEAT")
    check("half the occupation artefact closes it",
          asm.verdict(-0.16, -0.02) == "CLOSED")
    check("47b's own numbers come back CLOSED",
          asm.verdict(-0.3596, -0.2941) == "CLOSED")
    check("a missing arm is pending, not passed", asm.verdict(None, -0.01) == "PENDING")
    check("T2022 alone can close it", asm.verdict(0.01, -0.09) == "CLOSED")


def fake_47h(d: Path, winner_clean=True):
    rows = []
    designs = {"OL_daioe": (-0.36, -0.29), "entrant": (-0.20, -0.15),
               "enrol": ((0.01, 0.02) if winner_clean else (-0.30, -0.25))}
    for design, (a21, a22) in designs.items():
        for age, true in (("22-25", -0.12), ("50+", 0.002)):
            for T, a in ((2021, a21), (2022, a22)):
                aa = a if age == "22-25" else 0.001
                rows.append(dict(design=design, arm="true", trunc=T, age_group=age,
                                 tier="A", gamma2=true, se=0.01, p=0.0, n_obs=10,
                                 status="ok", elapsed_s=1))
                rows.append(dict(design=design, arm="asof", trunc=T, age_group=age,
                                 tier="A", gamma2=true + aa, se=0.01, p=0.0, n_obs=10,
                                 status="ok", elapsed_s=1))
    pd.DataFrame(rows).to_csv(d / "horserace_estimates.csv", index=False)


def fake_47j(d: Path, negative=True):
    rows = []
    for design in ("OL_daioe", "entrant"):
        for yb in ("22-25", "26-30"):
            for T in (2021, 2022):
                true = -0.14 if (negative and yb == "22-25") else 0.01
                for arm, g in (("true", true), ("asof", true - 0.01)):
                    rows.append(dict(gamma3=g, se=0.02, n_obs=100, status="ok",
                                     design=design, arm=arm, trunc=T,
                                     young_band=yb, n_firms=500))
    pd.DataFrame(rows).to_csv(d / "triple_estimates.csv", index=False)


def test_report_follows_the_numbers():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); fake_47h(d, winner_clean=True); fake_47j(d, negative=True)
        found = {"47h": d / "horserace_estimates.csv", "47j": d / "triple_estimates.csv"}
        md = asm.build_report(found, False, "")
        check("a clean design is reported as supported",
              "**Supported, by 1 of 3 designs.**" in md, md[md.find("## 1."):][:200])
        check("the winner is named", "`enrol`" in md)
        check("the contaminated ones are shown as CLOSED", md.count("| CLOSED |") >= 2)
        check("47j supported when the gap is negative and clean",
              "**Supported.**" in md.split("## 2.")[1].split("## 3.")[0])
        check("a missing input is PENDING, not silence",
              "**Pending** -- `firmmix_estimates.csv` not found." in md)
        check("the rule is quoted in the report", "Pre-committed before the runs" in md)
        check("the advertisement claim is stated as unaffected",
              "unaffected by any of the above" in md)

        d2 = Path(t) / "b"; d2.mkdir()
        fake_47h(d2, winner_clean=False); fake_47j(d2, negative=False)
        md2 = asm.build_report({"47h": d2 / "horserace_estimates.csv",
                                "47j": d2 / "triple_estimates.csv"}, False, "")
        check("no clean design -> not supported, and it says go to 47j",
              "**Not supported.**" in md2 and "go to 47j" in md2)
        check("47j with a positive gap is not reported as supported",
              "**Not supported as stated.**" in md2.split("## 2.")[1].split("## 3.")[0])


def test_newest_wins_and_nothing_is_deleted():
    with tempfile.TemporaryDirectory() as t:
        old, new = Path(t) / "old", Path(t) / "new"
        old.mkdir(); new.mkdir()
        fake_47j(old); fake_47j(new)
        import os, time
        os.utime(old / "triple_estimates.csv", (1, 1))
        found = asm.find([str(old), str(new)])
        check("the newest export wins", found["47j"].parent.name == "new",
              found["47j"].parent.name)
        check("nothing was deleted", (old / "triple_estimates.csv").exists()
              and (new / "triple_estimates.csv").exists())


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t); fake_47h(d); fake_47j(d)
        out = Path(t) / "EV"
        r = subprocess.run([sys.executable, str(REV / "assemble.py"), str(d),
                            "--out", str(out)], capture_output=True, text=True)
        check("the command runs", r.returncode == 0, r.stderr[-300:])
        check("it writes the markdown", (Path(str(out) + ".md")).exists())
        check("it lists pending inputs on stdout", "PENDING" in r.stdout)


if __name__ == "__main__":
    test_rule()
    test_report_follows_the_numbers()
    test_newest_wins_and_nothing_is_deleted()
    test_cli_end_to_end()
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

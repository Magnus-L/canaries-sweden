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


def fake_61(d: Path, adoption=-0.09, launch=-0.004, artefact=0.002):
    rows, pooled = [], []
    for design, arm in (("OL_daioe", "true"), ("OL_daioe", "asof"),
                        ("entrant", "true")):
        bump = artefact if arm == "asof" else 0.0
        for term, coef in (("s_launch", launch), ("s_copilot", launch / 2),
                           ("s_adoption", adoption),
                           ("post_rb_x_high_x_young", 0.001)):
            rows.append(dict(design=design, arm=arm, young_band="22-25",
                             term=term, coef=coef + bump, se=0.01, n_obs=10,
                             status="ok"))
        pooled.append(dict(design=design, arm=arm, young_band="22-25",
                           term="post2024_x_high_x_young",
                           coef=adoption + bump, se=0.01, n_obs=10,
                           status="ok"))
    pd.DataFrame(rows).to_csv(d / "redated_step.csv", index=False)
    pd.DataFrame(pooled).to_csv(d / "redated_pooled.csv", index=False)


def fake_62(d: Path, unit_matters=True):
    rows = []
    vals = {"A_occ_age_cont": -0.02,
            "B_occ_firm_cont": 0.05 if unit_matters else -0.021,
            "C_edu_age_cont": -0.019, "D_edu_firm_quart": -0.013}
    for v, c in vals.items():
        for age in ("22-25", "41-49"):
            rows.append(dict(age_group=age, variant=v, label=v,
                             coef=c if age == "22-25" else -0.019,
                             se=0.01, pvalue=0.1, n_obs=10, status="ok"))
    pd.DataFrame(rows).to_csv(d / "gradient_by_variant.csv", index=False)


def fake_63(d: Path, placebo_fires=False):
    rows = []
    for m, c in (("daioe", -0.018), ("eloundou", -0.016),
                 ("telework", -0.014 if not placebo_fires else -0.030)):
        for out in ("stock", "hires"):
            rows.append(dict(age_group="22-25", measure=m, outcome=out,
                             coef=c, se=0.01, pvalue=0.1, n_obs=10,
                             status="ok"))
    pd.DataFrame(rows).to_csv(d / "robustness_gradient.csv", index=False)
    hr = []
    for spec, terms in (("pooled", {"post_gpt_x_expo": -0.004,
                                    "post_gpt_x_tele": -0.001}),
                        ("by_age", {"gpt_x_expo_22_25": -0.017 if not placebo_fires else -0.002,
                                    "t_22_25": -0.002 if not placebo_fires else -0.026})):
        for term, c in terms.items():
            hr.append(dict(outcome="stock", spec=spec, term=term, coef=c,
                           se=0.01, n_obs=10, n_cells=900, status="ok"))
    pd.DataFrame(hr).to_csv(d / "horserace.csv", index=False)


def test_the_three_jobs_are_read_correctly():
    """
    The assembler turns these three into the sentences the paper will use,
    so a number that lands in the wrong sentence is worse than no sentence
    at all. Each is checked against a fabricated result whose answer is
    known, and then against the opposite result.
    """
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        fake_61(d); fake_62(d, unit_matters=True); fake_63(d, placebo_fires=False)
        found = asm.find([str(d)])
        for k in ("61", "61p", "62", "63", "63h"):
            check(f"the assembler finds {k}", k in found)
        md = asm.build_report(found, False, "")
        check("61's adoption coefficient is reported",
              "-0.0900" in md, md[md.find("## 8i."):][:260])
        check("and the launch window beside it, which is what was averaged in",
              "-0.0040" in md)
        check("61's artefact at the new dating is computed, not assumed",
              "artefact at the new dating is +0.0020" in md)
        check("62 attributes the disagreement to the unit",
              "UNIT alone moves 22-25 by +0.0700" in md)
        check("63 reports the horse race, not only the marginal columns",
              "AI exposure **-0.0170**, teleworkability -0.0020" in md)
        check("63 warns that the placebo is a weak one",
              "correlate 0.66 to 0.87" in md)

        d2 = Path(t) / "b"; d2.mkdir()
        fake_62(d2, unit_matters=False); fake_63(d2, placebo_fires=True)
        md2 = asm.build_report(asm.find([str(d2)]), False, "")
        check("a disagreement that is NOT about the unit reads that way",
              "UNIT alone moves 22-25 by -0.0010" in md2)
        check("a placebo that fires is reported as the placebo firing",
              "AI exposure **-0.0020**, teleworkability -0.0260" in md2)
        check("and 61 missing is pending, not silence",
              "## 8i." in md2 and "**Pending.**" in md2.split("## 8i.")[1][:80])


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
    test_the_three_jobs_are_read_correctly()
    test_newest_wins_and_nothing_is_deleted()
    test_cli_end_to_end()
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

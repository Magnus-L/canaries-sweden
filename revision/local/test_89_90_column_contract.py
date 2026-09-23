#!/usr/bin/env python3
"""
test_89_90_column_contract.py -- the frames lane 34 reads must carry the
                                 columns lane 34 asks for.

WHY THIS EXISTS, AND WHY IT IS SEPARATE FROM THE OTHER TESTS
Lane 34 was submitted without a local test and both scripts died in under
a minute on the same mistake: the FIRM-level exposure frame carries the
mean as `mix`, and both scripts asked for `score`. `score` is real, but it
is the OCCUPATION-level column of the score book, one frame upstream. The
run cost an hour of the slot and produced nothing.

The other tests in this directory plant a mechanism in a synthetic world
and check that the design recovers it. That is the right test for a
design and the wrong test for this failure, which needed no data at all:
it is a contract between two modules that could have been checked in a
second. So this test checks the contract, and checks it by reading the
real source rather than by mocking it, because a mock would have agreed
with whatever the scripts assumed.

    python3 revision/local/test_89_90_column_contract.py
"""
import re
import sys
from pathlib import Path

MONA = Path(__file__).resolve().parents[1] / "mona"

# What occ_route_exposure's docstring promises, and what 78's
# with_exposure merges onto a panel. Both are read from the source below
# rather than repeated from memory.
EXPOSURE_DOC = "Returns employer_id, fq, mix, n (the weight), n_coded"

failures = []


def source(name: str) -> str:
    return (MONA / name).read_text(encoding="utf-8")


def firm_frame_columns() -> set:
    """The columns occ_route_exposure actually returns, from its own
    `cols = [...]` literal in 82."""
    t = source("82_occupation_route.py")
    m = re.search(r'cols = \[\s*"employer_id", "fq", "mix"(.*?)\]', t, re.S)
    if not m:
        failures.append("82: could not find occ_route_exposure's cols list")
        return set()
    lit = '["employer_id", "fq", "mix"' + m.group(1) + "]"
    return set(re.findall(r'"([a-z_0-9]+)"', lit))


def with_exposure_merges() -> set:
    """The columns 78's with_exposure puts on the panel."""
    t = source("78_final_checks.py")
    m = re.search(r"def with_exposure.*?expo\[\[(.*?)\]\]", t, re.S)
    if not m:
        failures.append("78: could not find with_exposure's merge list")
        return set()
    cols = set(re.findall(r'"([a-z_0-9]+)"', m.group(1)))
    if 'b["high"]' in t:
        cols.add("high")
    return cols


def check_script(name: str, firm_cols: set, panel_cols: set) -> None:
    """Every column the script asks of an exposure frame must exist."""
    t = source(name)
    # An exposure-frame access is any column list carrying employer_id
    # together with one of the frame's own columns. Matching only
    # X["exposure"][[...]] was not enough: overlap() takes the frame as
    # a parameter, so the access reads a[["employer_id", ...]] and the
    # first version of this test walked straight past the bug it was
    # written for.
    marks = {"mix", "fq", "coverage", "share_not_2019", "n_coded"}
    for m in re.finditer(r"\[\[([^\]]*)\]\]", t):
        cols = re.findall(r'"([a-z_0-9]+)"', m.group(1))
        if "employer_id" not in cols:
            continue
        if not (set(cols) & (marks | {"score"})):
            continue                      # a panel frame, not an exposure
        for c in cols:
            if c not in firm_cols:
                failures.append(
                    f"{name}: asks the firm exposure frame for '{c}', which "
                    f"it does not carry (it has {sorted(firm_cols)})")
    # and the panel after with_exposure must not be asked for a mean
    if re.search(r'with_exposure\(', t):
        for c in ("score", "mix"):
            if re.search(rf'b\["{c}"\]', t):
                failures.append(
                    f"{name}: reads b[\"{c}\"] from a panel built by "
                    f"with_exposure, which merges only {sorted(panel_cols)}")


def check_book_shape() -> None:
    """89's wfh_book must return the shape build_exposure's `daioe`
    argument expects: ssyk4 and score, as l70.daioe_scores does."""
    t = source("89_wfh_offdiagonal.py")
    if 'return d.rename(columns={col: "score"})[["ssyk4", "score"]]' not in t:
        failures.append("89: wfh_book no longer returns ssyk4 and score, "
                        "which is the shape build_exposure's daioe "
                        "argument expects")


def main() -> int:
    t82 = source("82_occupation_route.py")
    if EXPOSURE_DOC not in t82:
        failures.append("82: occ_route_exposure's documented return has "
                        "changed; this test's premise needs rechecking")
    firm_cols = firm_frame_columns()
    panel_cols = with_exposure_merges()
    print(f"  firm exposure frame carries: {sorted(firm_cols)}")
    print(f"  with_exposure puts on the panel: {sorted(panel_cols)}")
    if "mix" not in firm_cols or "score" in firm_cols:
        failures.append("82: the firm frame no longer has 'mix' and not "
                        "'score'; lane 34 is written against that")
    for s in ("89_wfh_offdiagonal.py", "90_wfh_margins_adoption.py"):
        check_script(s, firm_cols, panel_cols)
    check_book_shape()
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n  lane 34 asks for no column that does not exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

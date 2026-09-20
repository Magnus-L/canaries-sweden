#!/usr/bin/env python3
"""
test_lanes.py -- vet the three lane runners.

What a lane must do, and each is tested against a sandbox of fake stages:
  1 run its stages in order
  2 SKIP a stage whose completion file already exists (so a resubmit is cheap)
  3 CONTINUE past a stage that exits non-zero, and run every later stage
  4 CONTINUE past a stage that is missing from the upload, and say so
  5 write its own log, and cap the echo (a blocked pipe loses the log too)
  6 cover every script exactly once across the three lanes, with no stage
    appearing in two lanes and no dependency running before what it needs
    python3 revision/local/test_lanes.py
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MONA = HERE.parent / "mona"
FAILS = []


def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else ""))
    if not c:
        FAILS.append(n)


def sandbox():
    t = Path(tempfile.mkdtemp(prefix="lanes_"))
    (t / "_lane.py").write_text((MONA / "_lane.py").read_text())
    return t


def stage_script(t: Path, name: str, rc: int, done: str, chatty=False):
    body = f'''
import sys, pathlib
p = pathlib.Path(__file__).resolve().parent / "{done}"
p.parent.mkdir(parents=True, exist_ok=True)
{"print('x' * 9000)" if chatty else ""}
p.write_text("done")
sys.exit({rc})
'''
    (t / name).write_text(body)


def run_lane(t: Path, stages):
    runner = f'''
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _lane
_lane.run("T", {stages!r})
'''
    (t / "runner.py").write_text(runner)
    r = subprocess.run([sys.executable, str(t / "runner.py")], cwd=str(t),
                       capture_output=True, text=True, timeout=120)
    log = (t / "run_laneT_log.txt").read_text() if (t / "run_laneT_log.txt").exists() else ""
    return r, log


def test_order_skip_continue():
    t = sandbox()
    stage_script(t, "a.py", 0, "out_a/done.txt")
    stage_script(t, "b.py", 1, "out_b/done.txt")      # fails
    stage_script(t, "c.py", 0, "out_c/done.txt")
    (t / "out_d").mkdir(); (t / "out_d" / "done.txt").write_text("already")
    stage_script(t, "d.py", 0, "out_d/done.txt")      # already finished
    # "finished" means finished with THIS script, so the marker has to be
    # the newer of the two for the stage to count as done
    os.utime(t / "out_d" / "done.txt", (time.time() + 60, time.time() + 60))
    stages = [("a.py", "out_a/done.txt", 1), ("b.py", "out_b/done.txt", 1),
              ("c.py", "out_c/done.txt", 1), ("d.py", "out_d/done.txt", 1),
              ("missing.py", "out_e/done.txt", 1)]
    r, log = run_lane(t, stages)
    check("the lane exits cleanly even when a stage fails", r.returncode == 0,
          r.stderr[-200:])
    check("a finished stage is skipped", "SKIP  d.py" in log)
    check("a stage that was never uploaded is named, not fatal",
          "ABSENT missing.py" in log)
    check("the failing stage is reported with its exit code", "b.py: exit 1" in log)
    check("the lane continues past the failure", "c.py: exit 0" in log)
    check("the stage after the failure actually ran", (t / "out_c" / "done.txt").exists())
    order = [m for m in re.findall(r"^  (\w+\.py): exit", log, re.M)]
    check("stages run in the declared order", order == ["a.py", "b.py", "c.py"], str(order))
    check("the summary names what failed", "failed: b.py" in log)


def test_a_new_script_beats_an_old_result():
    """
    On 20 September 2026 a re-uploaded script was skipped because the
    previous version's summary file was still on the share, and the lane
    reported success having run nothing at all. A result is only a result
    for the code that produced it, so a script newer than its own marker
    must re-run.
    """
    t = sandbox()
    (t / "out_e").mkdir()
    (t / "out_e" / "done.txt").write_text("the previous version's result")
    os.utime(t / "out_e" / "done.txt", (time.time() - 600, time.time() - 600))
    stage_script(t, "e.py", 0, "out_e/done.txt")      # uploaded afterwards
    r, log = run_lane(t, [("e.py", "out_e/done.txt", 1)])
    check("a script newer than its result is re-run, not skipped",
          "RUN   e.py" in log and "SKIP  e.py" not in log)
    check("and the lane says why, so the rerun is not a surprise",
          "NEWER" in log)
    check("the stage actually ran", "e.py: exit 0" in log)


def test_echo_is_capped_but_the_log_is_not():
    t = sandbox()
    stage_script(t, "loud.py", 0, "out_l/done.txt", chatty=True)
    r, log = run_lane(t, [("loud.py", "out_l/done.txt", 1)])
    # The real hazard: a chatty STAGE filling BatchClient's pipe and hanging
    # the lane. The child must never reach the parent's stdout at all.
    check("a chatty stage never reaches the lane's stdout",
          "x" * 100 not in r.stdout, f"parent stdout {len(r.stdout)} bytes")
    check("the parent's own stdout stays small", len(r.stdout) < 4096,
          f"{len(r.stdout)} bytes")
    stagelog = (t / "run_laneT_stages.txt").read_text()
    check("the stage's output is kept, in the stage file", "x" * 1000 in stagelog,
          f"stage file {len(stagelog)} bytes")
    check("the lane log records the stage result", "loud.py: exit 0" in log)


def test_lane_plan_is_coherent():
    """Across the three lanes: no duplicates, and nothing runs before what it
    depends on. 47i/47j read 47h's caches, so they must be in 47h's lane."""
    lanes = {}
    for n in (1, 2, 3):
        spec = importlib.util.spec_from_file_location(f"l{n}", MONA / f"run_lane{n}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"l{n}"] = m
        spec.loader.exec_module(m)
        lanes[n] = [s[0] for s in m.STAGES]
    allstages = [s for v in lanes.values() for s in v]
    check("no script appears in two lanes", len(allstages) == len(set(allstages)),
          str([s for s in set(allstages) if allstages.count(s) > 1]))
    for s in allstages:
        check(f"{s} exists in revision/mona", (MONA / s).exists())
    l3 = lanes[3]
    for dependant in ("47i_firmmix.py", "47j_within_employer_triple.py"):
        check(f"{dependant} is in 47h's lane", dependant in l3)
        check(f"{dependant} runs after 47h",
              l3.index(dependant) > l3.index("47h_edu_horserace.py"))
    check("47k is NOT in 47h's lane (it must not read a cache being written)",
          "47k_settled_sample.py" not in l3)
    check("all three lanes are non-empty", all(len(v) for v in lanes.values()))
    tot = {n: sum(s[2] for s in importlib.import_module(f"l{n}").STAGES) for n in (1, 2, 3)}
    print(f"      lane minutes: {tot}")
    check("the longest lane is under nine hours", max(tot.values()) < 540, str(tot))


def test_parallel_lanes():
    """
    Lanes 8, 9 and 10 are meant to occupy MONA's three slots at once, so
    they have to be independent in a stronger sense than lanes 1 to 3:
    not merely unordered, but incapable of interfering. Two things make
    that true and both are checked here. They share no stage, and none of
    them WRITES to the shared cache directory. A script that wrote a cache
    another was reading would corrupt it silently, and the failure would
    surface as an estimate rather than as an error.
    """
    stages = {}
    for n in (8, 9, 10, 11, 12):
        spec = importlib.util.spec_from_file_location(f"p{n}",
                                                      MONA / f"run_lane{n}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"p{n}"] = m
        spec.loader.exec_module(m)
        stages[n] = [x[0] for x in m.STAGES]
        check(f"lane {n} is a single stage, so a slot holds one job",
              len(m.STAGES) == 1, str(stages[n]))
    flat = [x for v in stages.values() for x in v]
    check("the three parallel lanes share no stage", len(flat) == len(set(flat)))
    for f in flat:
        src = (MONA / f)
        check(f"{f} exists in revision/mona", src.exists())
        if not src.exists():
            continue
        body = src.read_text(errors="replace")
        check(f"{f} never writes to the shared cache",
              "write_cache" not in body,
              "so three jobs can read the same caches at once")
        check(f"{f} performs no SQL", "mc.connect(" not in body)
    tot = {n: sum(x[2] for x in sys.modules[f"p{n}"].STAGES)
           for n in (8, 9, 10, 11, 12)}
    print(f"      parallel lane minutes: {tot}")
    check("each parallel lane is under six hours", max(tot.values()) < 360,
          str(tot))


if __name__ == "__main__":
    test_order_skip_continue()
    test_a_new_script_beats_an_old_result()
    test_echo_is_capped_but_the_log_is_not()
    test_lane_plan_is_coherent()
    test_parallel_lanes()
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)

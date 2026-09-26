#!/usr/bin/env python3
"""
master.py: run the register analysis of "Same Storm, Different Boats" inside
Statistics Sweden's MONA environment, chapter by chapter in the order of the
paper.

Every register estimate in the paper was produced by one of the scripts in
scripts/, run as a separate batch job with a handful of environment
variables that select the parts of a script to run and the folder it writes
to. This file records those jobs, with exactly the settings each one ran
with, so that a replicator with access to project P1207 can repeat them.

THREE MODES
  python master.py --list              print every job, its settings and the
                                       exhibits it produces; runs nothing
  python master.py --chapter 2         print the jobs of chapter 2 (the flag
                                       can be repeated); runs nothing
  python master.py --chapter 2 --run   run the jobs of chapter 2

Nothing runs without --run. With --run the file refuses to start unless it
is inside MONA: the operating system must be Windows and the project folder
named in scripts/mona_common.py (PROJECT) must be reachable. With the
environment variable CANARIES_DRYRUN=1, the convention mona_common.py uses
for work outside MONA, --run prints each command with its environment and
executes nothing.

BATCH JOBS
MONA's batch submitter starts a script with no arguments, cannot set
environment variables and does not read standard output (a job that writes
more than a few kilobytes to it blocks). master.py therefore takes its
arguments from a one-line file master_run.txt beside it when it is started
with none, e.g. "--chapter 2 --run", sets each job's variables itself, and
sends each job's output to master_logs/<chapter>_<script>_<n>.txt rather
than to the console; its own progress goes to master_logs/master_log.txt.

WHERE THE FILES GO
master.py finds the scripts in scripts/ beside it, or, if there is no such
folder, beside itself; the second is the layout on the MONA share, where
the scripts sit in the project folder and write their output_* and cache
folders there.

ORDER
Chapter 1 builds the caches every later chapter reads (monthly counts by
employer and age band, flows, counts by sex and education, the completed
industry key, the uncounted payslips). It must run first and takes about
twenty-three hours run in sequence. Chapters 2 to 8 and 10 can then run in
any order and, within a chapter, jobs can be submitted in parallel: each
writes to its own folder and uses its own exchange folder for R
(CANARIES_RWORK_TAG). Chapter 7 needs the vintage panel that its first job
(script 39) builds. Chapter 9 repeats the education-route comparison that
the paper does not report; it is listed for completeness. Chapter 10 holds
the final checks of 25 and 26 September 2026 (scripts 95 to 102), which
ran through the lane runners shipped beside the scripts (run_lane37a.py to
run_lane38c.py); the runners are kept for the record and master.py sets the
same variables.

The runtimes are the wall-clock minutes observed in MONA in September 2026
or, where no summary recorded one, the time budget the job ran under. The
five scripts of chapter 7 ran as one sequence whose timing was not
recorded; they are shown with a question mark.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts" if (HERE / "scripts").is_dir() else HERE
LOGS = SCRIPTS / "master_logs"
ARGS_FILE = HERE / "master_run.txt"

CHAPTERS = {
    1: "Data build: the caches every later chapter reads",
    2: "Table 1 and the estimation sample (OA Table A2)",
    3: "First stage, reference window, industry, credit and precision",
    4: "Figure 2: the age profile and its plain counterpart",
    5: "Figure 3: timing, the pre-period and the dating of adoption",
    6: "Heterogeneity: sex within track and fields of education",
    7: "Coverage: the vintage panel and the as-of backtest (Part IV)",
    8: "Rival explanations: teleworkability, the youth payroll reduction, "
       "uncounted payslips",
    9: "Comparison only: the education route (not reported in the paper)",
    10: "Final checks on tau: pension ages, birth cohorts, credit, the female "
        "differential, the exposure specification, the raw rebuild of the "
        "counts, the three-arm backtest, unlinked payslips, count provenance",
}

# One entry per batch job. `env` holds the variables the job ran with in
# MONA; CANARIES_RWORK_TAG gives each job its own exchange folder for R so
# that jobs of one chapter can run side by side.
JOBS = [
    # -- 1. data build ------------------------------------------------------
    dict(ch=1, script="47h_edu_horserace.py", minutes=330,
         env={"CANARIES_RWORK_TAG": "_c1_47h"},
         makes="education records and score files (cache/edu_hr_*)"),
    dict(ch=1, script="47L_age_baseline_exposure.py", minutes=75,
         env={"CANARIES_RWORK_TAG": "_c1_47L"},
         makes="counts by employer, month and age band (cache/L_counts_*, "
               "L_baseline_2019)"),
    dict(ch=1, script="54_hiring_flows.py", minutes=90,
         env={"CANARIES_RWORK_TAG": "_c1_54"},
         makes="hires and separations (cache/flows_*)"),
    dict(ch=1, script="67_gender_on_the_new_design.py", minutes=270,
         env={"CANARIES_RWORK_TAG": "_c1_67"},
         makes="counts and flows by sex (cache/L_counts_sex_*, flows_sex_*)"),
    dict(ch=1, script="76_gender_decomposition.py", minutes=76,
         env={"CANARIES_RWORK_TAG": "_c1_76"},
         makes="counts by sex and education (cache/L_counts_sex_edu_*)"),
    dict(ch=1, script="78_final_checks.py", minutes=360,
         env={"CANARIES_78_PARTS": "EF", "CANARIES_78_OUT": "output_78c",
              "CANARIES_RWORK_TAG": "_25c"},
         makes="counts split at 65 (cache/L_counts_split_*)"),
    dict(ch=1, script="79_last_gaps.py", minutes=90,
         env={"CANARIES_79_PARTS": "C", "CANARIES_79_OUT": "output_79c",
              "CANARIES_RWORK_TAG": "_26c"},
         makes="payslips without an occupation code (cache/U_uncounted_*)"),
    dict(ch=1, script="80_industry_key.py", minutes=60,
         env={"CANARIES_80_PARTS": "A", "CANARIES_80_OUT": "output_80a",
              "CANARIES_RWORK_TAG": "_27a"},
         makes="the completed industry key (cache/I_industry_key)"),

    # -- 2. Table 1 ---------------------------------------------------------
    dict(ch=2, script="82_occupation_route.py", minutes=8,
         env={"CANARIES_82_PARTS": "A", "CANARIES_82_OUT": "output_82a",
              "CANARIES_RWORK_TAG": "_28a"},
         makes="the occupation-route score; coverage (OA Tables A2 and A28)"),
    dict(ch=2, script="82_occupation_route.py", minutes=110,
         env={"CANARIES_82_PARTS": "B", "CANARIES_82_OUT": "output_82b",
              "CANARIES_RWORK_TAG": "_28b"},
         makes="Table 1 headline rows; the six-band profile (OA Table A13)"),
    dict(ch=2, script="82_occupation_route.py", minutes=47,
         env={"CANARIES_82_PARTS": "C", "CANARIES_82_OUT": "output_82c",
              "CANARIES_RWORK_TAG": "_28c"},
         makes="Table 1 rows by sex; hires and separations; vintage rows (OA IV.4)"),

    dict(ch=2, script="68_seasonal_control.py", minutes=155,
         env={"CANARIES_RWORK_TAG": "_c2_68"},
         makes="panel counts before the exposure merge (OA Table A2, read "
               "from its log); the quarters of Figure 3"),

    # -- 3. first stage, window, industry, precision ------------------------
    dict(ch=3, script="83_occupation_route_rest.py", minutes=5,
         env={"CANARIES_83_PARTS": "A", "CANARIES_83_OUT": "output_83a",
              "CANARIES_82_OUT": "output_83a", "CANARIES_RWORK_TAG": "_29a"},
         makes="first stage (OA Figure A5); descriptives"),
    dict(ch=3, script="83_occupation_route_rest.py", minutes=210,
         env={"CANARIES_83_PARTS": "B", "CANARIES_83_OUT": "output_83b",
              "CANARIES_82_OUT": "output_83b", "CANARIES_RWORK_TAG": "_29b"},
         makes="reference window, drift, clustering (OA Tables A11, A19 and A21)"),
    dict(ch=3, script="83_occupation_route_rest.py", minutes=69,
         env={"CANARIES_83_PARTS": "C", "CANARIES_83_OUT": "output_83c",
              "CANARIES_82_OUT": "output_83c", "CANARIES_RWORK_TAG": "_29c"},
         makes="industry and credit (OA Table A20)"),
    dict(ch=3, script="83_occupation_route_rest.py", minutes=180,
         env={"CANARIES_83_PARTS": "D", "CANARIES_83_OUT": "output_83d",
              "CANARIES_82_OUT": "output_83d", "CANARIES_RWORK_TAG": "_29d"},
         makes="employer size and reliability (OA Table A22)"),

    # -- 4. Figure 2 ----------------------------------------------------------
    dict(ch=4, script="85_occupation_route_plain_profile.py", minutes=38,
         env={"CANARIES_85_PARTS": "PDS", "CANARIES_85_OUT": "output_85",
              "CANARIES_82_OUT": "output_85", "CANARIES_RWORK_TAG": "_31"},
         makes="Figure 2; OA Tables A9 and A13"),
    dict(ch=4, script="63_measure_robustness.py", minutes=111,
         env={"CANARIES_RWORK_TAG": "_c4_63"},
         makes="the continuous route (OA Table A14)"),

    # -- 5. Figure 3 ------------------------------------------------------------
    dict(ch=5, script="84_occupation_route_path.py", minutes=68,
         env={"CANARIES_84_SHAPES": "QM", "CANARIES_84_OUT": "output_84",
              "CANARIES_82_OUT": "output_84", "CANARIES_RWORK_TAG": "_30"},
         makes="Figure 3; OA Table A12 and Figure A6"),
    dict(ch=5, script="86_occupation_route_prepath.py", minutes=64,
         env={"CANARIES_86_OUT": "output_86", "CANARIES_82_OUT": "output_86",
              "CANARIES_RWORK_TAG": "_32"},
         makes="the path from 2019 (OA Figure A7, Table A19)"),
    dict(ch=5, script="91_dating_sensitivity.py", minutes=240,
         env={"CANARIES_91_OUT": "output_91", "CANARIES_82_OUT": "output_91",
              "CANARIES_RWORK_TAG": "_35"},
         makes="moving the adoption boundary (Section 3, OA Table A10)"),

    # -- 6. heterogeneity -------------------------------------------------------
    dict(ch=6, script="87_occupation_route_gender_split.py", minutes=64,
         env={"CANARIES_87_OUT": "output_87", "CANARIES_82_OUT": "output_87",
              "CANARIES_RWORK_TAG": "_33"},
         makes="Table 1 within-track row; OA Tables A15 and A16"),
    dict(ch=6, script="88_occupation_route_contrast_by_track.py", minutes=31,
         env={"CANARIES_88_OUT": "output_88", "CANARIES_82_OUT": "output_87",
              "CANARIES_RWORK_TAG": "_33"},
         makes="fields of education (Section 3, OA Table A17)"),

    # -- 7. coverage --------------------------------------------------------------
    dict(ch=7, script="39_canary_gate.py", minutes=None,
         env={"CANARIES_RWORK_TAG": "_c7_39"},
         makes="the vintage panel (cache/panel_vintage); the submitted estimate"),
    dict(ch=7, script="40_coverage_diagnostics.py", minutes=None,
         env={"CANARIES_RWORK_TAG": "_c7_40"},
         makes="OA IV.1, Table A25"),
    dict(ch=7, script="41_vintage_event_studies.py", minutes=None,
         env={"CANARIES_RWORK_TAG": "_c7_41"},
         makes="OA IV.2, Table A26"),
    dict(ch=7, script="45_asof_backtest.py", minutes=None,
         env={"CANARIES_RWORK_TAG": "_c7_45"},
         makes="OA IV.3, Table A27, Figure A8"),
    dict(ch=7, script="49_coverage_reconcile.py", minutes=None,
         env={"CANARIES_RWORK_TAG": "_c7_49"},
         makes="OA VI.1, the match-rate definitions"),

    # -- 8. rival explanations ------------------------------------------------
    dict(ch=8, script="89_wfh_offdiagonal.py", minutes=92,
         env={"CANARIES_89_OUT": "output_89", "CANARIES_82_OUT": "output_89",
              "CANARIES_RWORK_TAG": "_34"},
         makes="the two scores at employer level (Section 3, OA II.3)"),
    dict(ch=8, script="90_wfh_margins_adoption.py", minutes=90,
         env={"CANARIES_90_OUT": "output_90", "CANARIES_82_OUT": "output_89",
              "CANARIES_RWORK_TAG": "_34"},
         makes="margins of the teleworkability contrast (not quoted)"),
    dict(ch=8, script="92_youth_payroll_rival.py", minutes=240,
         env={"CANARIES_92_OUT": "output_92", "CANARIES_82_OUT": "output_91",
              "CANARIES_RWORK_TAG": "_35"},
         makes="the youth payroll reduction (Section 3, OA III.2)"),
    dict(ch=8, script="93_uncounted_occupation.py", minutes=30,
         env={"CANARIES_93_OUT": "output_93", "CANARIES_79_OUT": "output_93",
              "CANARIES_82_OUT": "output_93", "CANARIES_RWORK_TAG": "_36"},
         makes="payslips without an occupation code (OA VI.1, Table A33)"),

    # -- 9. comparison only -----------------------------------------------------
    dict(ch=9, script="47j_within_employer_triple.py", minutes=12,
         env={"CANARIES_RWORK_TAG": "_c9_47j"}, makes="education route"),
    dict(ch=9, script="61_redated_triple.py", minutes=65,
         env={"CANARIES_RWORK_TAG": "_c9_61"}, makes="education route"),
    dict(ch=9, script="66_plain_magnitudes.py", minutes=3,
         env={"CANARIES_RWORK_TAG": "_c9_66"},
         makes="education route; optional comparison column for OA Table A9"),
    dict(ch=9, script="70_respecifications.py", minutes=200,
         env={"CANARIES_RWORK_TAG": "_c9_70"}, makes="education route"),
    dict(ch=9, script="71_adoption_validation.py", minutes=70,
         env={"CANARIES_RWORK_TAG": "_c9_71"}, makes="education route"),
    dict(ch=9, script="73_industry_and_credit.py", minutes=200,
         env={"CANARIES_RWORK_TAG": "_c9_73"}, makes="education route"),
    dict(ch=9, script="74_contrast_seasonal.py", minutes=150,
         env={"CANARIES_RWORK_TAG": "_c9_74"}, makes="education route"),
    dict(ch=9, script="75_reference_window.py", minutes=31,
         env={"CANARIES_RWORK_TAG": "_c9_75"}, makes="education route"),
    dict(ch=9, script="77_contrast_by_track.py", minutes=30,
         env={"CANARIES_RWORK_TAG": "_c9_77"}, makes="education route"),
    dict(ch=9, script="78_final_checks.py", minutes=155,
         env={"CANARIES_78_PARTS": "ADG", "CANARIES_78_OUT": "output_78a",
              "CANARIES_RWORK_TAG": "_25a"},
         makes="education route; optional comparison for OA Figure A7 and "
               "Table A19"),
    dict(ch=9, script="78_final_checks.py", minutes=300,
         env={"CANARIES_78_PARTS": "BC", "CANARIES_78_OUT": "output_78b",
              "CANARIES_RWORK_TAG": "_25b"}, makes="education route"),

    # -- 10. final checks on tau (25 and 26 September 2026) ------------------
    # These nine jobs ran through the lane runners shipped in scripts/
    # (run_lane37a.py to run_lane38d.py, each a list of stages for _lane.py),
    # which set the same variables. Every script reproduces Table 1's tau on
    # its own panel before any check runs; the runtimes are those recorded in
    # the summaries.
    dict(ch=10, script="99_measurement.py", minutes=57,
         env={"CANARIES_99_OUT": "output_99", "CANARIES_82_OUT": "output_99",
              "CANARIES_RWORK_TAG": "_37a"},
         makes="the raw rebuild of the counts; non-match over all declared "
               "person-months (OA Table A41); tau by presence in November 2022 "
               "(OA IV.1)"),
    dict(ch=10, script="98_backtest_common.py", minutes=107,
         env={"CANARIES_98_OUT": "output_98", "CANARIES_RWORK_TAG": "_37b"},
         makes="the three-arm as-of backtest on common support (OA Table A33; "
               "exported again by the job below)"),
    dict(ch=10, script="97_headline_checks.py", minutes=244,
         env={"CANARIES_97_OUT": "output_97", "CANARIES_82_OUT": "output_97",
              "CANARIES_80_OUT": "output_97", "CANARIES_73_OUT": "output_97",
              "CANARIES_RWORK_TAG": "_37b"},
         makes="the credit test on tau; the female differential's pre-path, "
               "drift and industry test; the exposure specification (OA Table "
               "A25, Panels D to F; OA Figure A6; the second record of Table 1)"),
    dict(ch=10, script="95_pension_reference.py", minutes=218,
         env={"CANARIES_95_OUT": "output_95", "CANARIES_82_OUT": "output_95",
              "CANARIES_RWORK_TAG": "_37c"},
         makes="tau with the older reference restricted; the eight-band "
               "profile on tau (Figure 2 of the paper; OA Table A25, Panels A "
               "and B)"),
    dict(ch=10, script="96_payroll_cohorts.py", minutes=92,
         env={"CANARIES_96_OUT": "output_96", "CANARIES_82_OUT": "output_95",
              "CANARIES_RWORK_TAG": "_37c"},
         makes="fixed birth cohorts against the youth payroll reduction (OA "
               "Table A25, Panel C)"),
    dict(ch=10, script="101_count_provenance.py", minutes=81,
         env={"CANARIES_101_OUT": "output_101", "CANARIES_82_OUT": "output_100",
              "CANARIES_RWORK_TAG": "_38a"},
         makes="the provenance of the pooled and sex-specific counts (OA VI.1)"),
    dict(ch=10, script="100_tipping_point.py", minutes=134,
         env={"CANARIES_100_OUT": "output_100", "CANARIES_82_OUT": "output_100",
              "CANARIES_RWORK_TAG": "_38a"},
         makes="unlinked payslips, the tipping point and the extremal "
               "allocations (OA Table A40)"),
    dict(ch=10, script="98_backtest_common.py", minutes=93,
         env={"CANARIES_98_OUT": "output_98b", "CANARIES_RWORK_TAG": "_38b"},
         makes="the same backtest, exported with the observations each fit "
               "retained; the export OA Table A33 is built from"),
    dict(ch=10, script="102_month_of_year.py", minutes=180,
         env={"CANARIES_102_OUT": "output_102", "CANARIES_82_OUT": "output_102",
              "CANARIES_80_OUT": "output_102", "CANARIES_73_OUT": "output_102",
              "CANARIES_RWORK_TAG": "_38c"},
         makes="tau and the female differential with month-of-year terms in "
               "place of the calendar-quarter terms (lane 38c; OA Table A25, "
               "Panel G)"),
    dict(ch=10, script="103_eloundou_classification.py", minutes=180,
         env={"CANARIES_103_OUT": "output_103", "CANARIES_82_OUT": "output_103",
              "CANARIES_80_OUT": "output_103", "CANARIES_73_OUT": "output_103",
              "CANARIES_RWORK_TAG": "_38d"},
         makes="the headline design with employers classified by the Eloundou "
               "rating through script 82's chain, DAIOE beside Eloundou on the "
               "employers both indices score, and the agreement of the two "
               "classifications (lane 38d; OA Table A25, Panel H)"),
]

# Inputs the scripts read from the project's input folder (mona_common.SHARE).
# The first three ship in inputs/; the education key is obtainable from the
# authors (see README.md).
INPUTS = ["daioe_quartiles.dta", "dingel_neiman_ssyk4.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"]


def project_folder() -> str:
    """Read PROJECT from mona_common.py without importing it: importing loads
    the database driver, which exists only inside MONA."""
    tree = ast.parse((SCRIPTS / "mona_common.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "PROJECT"):
            return ast.literal_eval(node.value)
    raise RuntimeError("PROJECT not found in mona_common.py")


def show(jobs: list[dict]) -> None:
    total = 0
    for ch in sorted({j["ch"] for j in jobs}):
        print(f"\nChapter {ch}. {CHAPTERS[ch]}")
        for j in (j for j in jobs if j["ch"] == ch):
            total += j["minutes"] or 0
            env = " ".join(f"{k}={v}" for k, v in j["env"].items())
            mins = f"~{j['minutes']:>4} min" if j["minutes"] else "     ? min"
            print(f"  {j['script']:<40} {mins}  {j['makes']}")
            print(f"      {env}")
    print(f"\n{len(jobs)} jobs, at least {total / 60:.0f} hours if run one after another")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print every job and exit")
    ap.add_argument("--chapter", type=int, action="append", choices=sorted(CHAPTERS),
                    help="select a chapter (repeatable); default: chapters 1 to 8")
    ap.add_argument("--run", action="store_true", help="run the selected jobs")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue with the next job when one fails")
    argv = sys.argv[1:]
    if not argv and ARGS_FILE.is_file():
        argv = ARGS_FILE.read_text(encoding="utf-8").split()
    args = ap.parse_args(argv)

    if args.list:
        show(JOBS)
        return 0
    chapters = args.chapter or [c for c in CHAPTERS if c != 9]
    jobs = [j for j in JOBS if j["ch"] in chapters]
    missing = [j["script"] for j in jobs if not (SCRIPTS / j["script"]).is_file()]
    if missing:
        print("scripts not found in scripts/: " + ", ".join(missing))
        return 1
    if not args.run:
        show(jobs)
        print("\nNothing was run. Add --run to run these jobs inside MONA.")
        return 0

    dryrun = os.environ.get("CANARIES_DRYRUN") == "1"
    if not dryrun:
        project = project_folder()
        if os.name != "nt" or not Path(project).is_dir():
            print("REFUSED: these scripts read the P1207 register database and run "
                  "only inside MONA.\n"
                  f"  The project folder {project} is not reachable from here.\n"
                  "  Set CANARIES_DRYRUN=1 to print the commands without running them.")
            return 2
        share = Path(os.environ.get("CANARIES_SHARE", project + r"\input"))
        absent = [f for f in INPUTS if not (share / f).is_file()]
        if absent:
            print(f"REFUSED: inputs missing from {share}: " + ", ".join(absent))
            return 2

    if not dryrun:
        LOGS.mkdir(exist_ok=True)
        log = open(LOGS / "master_log.txt", "a", encoding="utf-8")
    else:
        log = sys.stdout

    def say(text: str) -> None:
        print(text, file=log, flush=True)

    failed = []
    for n, j in enumerate(jobs, 1):
        env = {**os.environ, **j["env"]}
        cmd = [sys.executable, j["script"]]
        settings = " ".join(f"{k}={v}" for k, v in j["env"].items())
        say(f"=== {time.strftime('%Y-%m-%d %H:%M')} chapter {j['ch']}: "
            f"{settings} {j['script']}")
        if dryrun:
            continue
        job_log = LOGS / f"{j['ch']}_{Path(j['script']).stem}_{n}.txt"
        with open(job_log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=SCRIPTS, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        say(f"    exit code {rc}; output in {job_log.name}")
        if rc != 0:
            failed.append(j["script"])
            if not args.keep_going:
                break
    if dryrun:
        say("\nCANARIES_DRYRUN=1: commands printed, nothing was run.")
        return 0
    say("all jobs finished" if not failed else "failed: " + ", ".join(failed))
    log.close()
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())

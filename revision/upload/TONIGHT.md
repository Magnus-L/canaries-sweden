# Tonight: upload 14 files, submit 3

Source on the Mac: `~/Documents/Workspace/projects/canaries-sweden/revision/upload/`

Destination A (scripts): `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`
Destination B (data): `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\input\`

No new folders. Both exist.

## Upload to A — as `.txt`, then rename to `.py`

```
run_lane1.py
run_lane2.py
run_lane3.py
_lane.py
mona_common.py          <- CHANGED 19 Sep, must be re-uploaded
50_sim_moments.py
47L_age_baseline_exposure.py
51_vintage_ai_unboxed.py
48_gender_poisson.py
47k_settled_sample.py
47h_edu_horserace.py
47i_firmmix.py
47j_within_employer_triple.py
```

## Upload to B — directly, no rename

```
eloundou_ssyk4.dta
```

## Submit, all three at once

```
run_lane1.py     about 2 h 20
run_lane2.py     about 2 h 15
run_lane3.py     about 5 h 55
```

Nothing else. They share no cache and no output folder, so they cannot interfere.

## Export when they are done

```
output_50   output_47L   output_51   output_48
output_47k
output_47h   output_47i   output_47j
output_41   output_44   output_46      <- finished earlier, never fetched
```

## Then on the Mac, one command

```
python3 revision/assemble.py <the export folder>
```

---

## If something goes wrong

- **A lane stops early.** It will not: a failing stage is reported and the lane
  continues. Check `run_lane<N>_log.txt` for the summary and
  `run_lane<N>_stages.txt` for what the stage itself printed.
- **A lane is killed.** Resubmit the same file. Finished stages are skipped and
  47h resumes from its caches.
- **A file was not uploaded.** The lane names it as ABSENT and runs the rest.
- **47h finishes before lane 2 starts.** 47k notices and reuses its caches, saving
  about 75 minutes. If it does not, 47k pulls its own copy; that is by design.

## What each lane is for

- **Lane 1, the answer.** `47L` is the new design: score each employer's age group
  by the occupations that group actually held in 2019, then count workers by age
  alone. The occupation register is used only where it is good and never after.
  `50` carries the validation table the local analysis needs; `48` is the gender
  split; `51` is the AI Unboxed check (see below).
- **Lane 2, the estimand.** `47k` keeps the paper's exact question by restricting to
  young workers whose education record is already correct rather than stale.
- **Lane 3, the diagnosis.** `47h` establishes which classifiers survive register
  lag; `47i` and `47j` are supporting evidence and read its caches.

## 51 is for AI Unboxed, not for canaries

Six minutes, Individ aggregates only, no panel and no estimation. It answers the
question our own `data-notes/occupation-missingness.md` has carried unanswered since
10 August: what share of each year's occupation codes were actually assigned that
year, and is the staleness concentrated in clerical work.

It matters because AI Unboxed measures clerical employment within firms and ends in
2023. Stale codes at an endpoint attenuate a measured change toward zero, which for a
paper that finds an effect is the favourable direction: the true effect would be at
least as large. It changes how the measure is described, not whether the finding
survives. Delete the `51_vintage_ai_unboxed.py` line from `run_lane1.py`'s STAGES if
you would rather keep this round purely about canaries.

## Changed on 19 September, after reading the 16:45 export

- **`mona_common.py` must be re-uploaded.** Its `Tee` echoed to stdout before
  writing the log and had no cap, so when BatchClient's pipe filled the line
  never reached the log either. That is why 47h's log froze at 1,650 bytes
  while its results file kept growing, and why console 3's log ended
  mid-traceback on the 18th. It now writes the file first and caps the echo.
- **`47h_edu_horserace.py` has a third gate arm.** Its first run halted at the
  gate: the as-of arm gave -0.156 where 47b reported -0.370. The gate now also
  estimates 47b's EXACT cascade, so it measures whether the gap is our
  documented fix rather than assuming it. If the legacy arm reproduces 47b the
  run proceeds; if not it halts, as before.

## 50's export was over the cap: run 52 (19 Sep, 17:50)

Ten files came out above the 5 MB per-file limit: `m6b_inr_tertiary_2020..2023`
at about 5.2 MB and the six `m7_validation_*` at 6.6 to 7.3 MB.

**Fix, no SQL, seconds:**

1. Upload `revision/upload/52_slim_exports.py` to
   `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`
   (as `.txt`, rename to `.py`).
2. Run it in Spyder with F5, or submit it. It reads `output_50\` and writes
   `output_50_slim\`.
3. **Export `output_50_slim`, not `output_50`.**

It replaces the 4-digit occupation code with the two numbers the analysis
actually consumes from it -- the DAIOE-weighted sums -- so every design score
and every validation statistic is identical to the last digit; that is
asserted in `revision/local/test_52_slim.py` and checked end to end through
l13 on both shapes. It also raises the floor on those files from five to ten
and DROPS suppressed rows rather than blanking them, which is both smaller and
safer. Files already under the cap are copied through byte-identical, so the
slim folder is a complete replacement.

If its log still says OVER CAP for anything, tell me and I will split that file
by age band.

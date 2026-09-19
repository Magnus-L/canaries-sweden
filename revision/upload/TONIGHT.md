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

---

# ROUND 2 — 19 Sep, ~19:00. Two free slots.

Three lanes died: 47h on a cache-schema KeyError, 47L/47i/47j on
`OSError: [Errno 28] No space left on device` inside the R exchange writer.
Seven files fixed and re-tested (76/76 dry-run, 141 local passes).

## 1. Upload to A, as `.txt` then rename to `.py`

From `revision/upload/` to
`\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`

    mona_common.py
    47h_edu_horserace.py
    47i_firmmix.py
    47j_within_employer_triple.py
    47k_settled_sample.py
    47L_age_baseline_exposure.py
    _lane.py

Safe to overwrite while lane 2 runs: Python reads a module once, at import.

## 2. Start, in this order

    slot A:  run_lane3.py     47h -> 47i -> 47j      4.5-6 h
    slot B:  run_lane1.py     47L -> 48              1.5-2 h
    slot C:  (47k, already running — leave it)

Lane 3 first. 47h rebuilds the shared year cache that 47i and 47j read, and
they run after it inside the same lane, so there is no contention.

## 3. Runtime, lane 3

    year pulls        5 x ~19 min     the legacy columns are not in last
                                      night's cache, so these must re-pull
    collapses         ~10 min total   only the one missing arm, not all 33
    gate              3 fits, ~6 min
    Tier A            ~29 fits, ~1 h
    Tier B + C        ~1.5 h

## 4. What to watch

Lane 3's gate, about 2 h in. It prints three arms and decides:

    asof_legacy close to -0.3695  -> the pull is verified, Tier A proceeds
    asof_legacy far from it       -> halts, and we have a real problem

Lane 1's 47L: three fits at n=11,208,600 crashed R last night with
`*** recursive gc invocation`. They may crash again. If they do, the log now
names the R error properly and saves the whole stream to
`_rerr_<tag>.txt` in the script's exchange directory.

## 5. After lane 2 (47k) finishes

Delete `round1_EL67898\cache\*_k.parquet` in Explorer. They are a duplicate
copy of five 38-million-row year frames that nothing reads any more.

---

# LANE 4 — the triangulation lane (added ~21:00, 19 Sep)

## Upload to A, as `.txt` then rename to `.py`

From `revision/upload/` to
`\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`

    53_freshcode_panel.py          NEW
    47L_age_baseline_exposure.py   REPLACES tonight's copy (adds the gradient)
    run_lane4.py                   NEW

## Run

    run_lane4.py       53 then 47L      about 2 h

53 pulls five years (~3 min each) and runs 18 pooled fits plus 2 event
studies. 47L's caches are warm, so it is the fits only.

## What each answers, and how each FAILS

    53   the paper's own estimand -- young vs young, inside the employer,
         monthly -- on codes assigned in the observation year, 2019-2023.
         Uses NO education register. Window stops at 2023.
         FAILS IF: freshness is differentially selected across the
         exposure dimension over time.

    47L  exposure frozen in 2019; afterwards only a birth year and a
         payslip. Uses NO post-2019 occupation code and NO education.
         Runs to 2025. Now reports a coefficient per age band.
         FAILS IF: firms whose young did more exposed work in 2019 were
         already on different trends.

Neither failure mode is the other's, and neither is the register lag.

## Read rules, pre-committed

53: the fresh arm's 50+ coefficient must be within 0.03 of zero. If it is
not, the restriction is doing something of its own and the 22-25 number
must not be quoted alone.

The Q4-vs-rest freshness gap in `selection_did.csv` is DESCRIPTIVE and is
not a gate. It moves mechanically whenever misplacement is present, which
the synthetic test demonstrated before this ran on real data.

47L: the age profile is the object. A flat profile says the design finds
nothing anywhere; a profile steep at 22-25 and flat at 50+ is the paper's
claim surviving on register-immune measurement.

---

# LANE 5 — the fast margin (added ~21:30, 19 Sep)

Run this AFTER one of the others finishes. Batch, like the rest.

## Upload to A, as `.txt` then rename to `.py`

    54_hiring_flows.py     NEW
    run_lane5.py           NEW

## Run

    run_lane5.py       54 alone      about 90 min, least certain estimate
                                     in this round

## What it is

Hires and separations per employer x age x month, on 47L's exposure
frozen in 2019. A hire is an (employer, person) spell present this month
and absent last month. Birth year gives the age band. No occupation code
after 2019, no education register, nothing with a register lag in the
outcome.

47L bounded the effect on the employment STOCK at about one log point.
Headcount is the slowest margin there is, with notice periods and
collective agreements in the way, and the entry-level claim is about
HIRING. This is the margin where an effect should appear first.

## Watch the first year

Nothing of this shape has run on P1207: each month joins two
five-million-row AGI tables to each other. The pull runs one month at a
time and caches per year, so a failure names the month and a restart
keeps what was already fetched. If year 2019 alone takes more than 20
minutes, stop it and say so; the query needs an index hint, not patience.

## How to read it

Hires and separations side by side. A fall in hiring with separations
flat is an inflow adjustment, which is the entry-level claim. Both moving
is a scale effect and means something else. The age gradient is reported
for both outcomes.

The exposure is a 2019 proxy for who is exposed in 2025, so it is stale
by construction and attenuates toward zero. A null here bounds the effect
of BASELINE exposure, not of current exposure. That limit is in the
script's own summary and must travel with any number quoted from it.

---

# LAST STEP — the export pack

Run this ONCE, after every lane has finished. It reads every output_*
directory and writes ONE small folder.

## Upload to A

    55_export_pack.py      NEW

## Run it last (Spyder F5 or batch, it takes seconds and uses no SQL)

    55_export_pack.py

## Then export `export_pack/` and nothing else

On the real output we already hold, 23.3 MB of raw directories packed to
0.39 MB. Every summary, every log and every coefficient table is in it.
Large support files are NOT in it and are NAMED in `MANIFEST.txt` with
their row and column counts, so a missing result can never be mistaken
for a null. If one of them turns out to be needed, add its filename to
PRIORITY at the top of the script and re-run.

Every packed csv is re-floored on the way out and suppressed rows are
dropped rather than blanked. MANIFEST.txt carries a SHA-256 per file so
the copy that arrives can be verified against the copy that left.

You are at 31 MB of the 50 MB rolling budget. This should cost about 1 MB.

---

# LANE 6 — time and attenuation (added ~21:45, 19 Sep)

Run AFTER lanes 4 and 5. Batch.

## Upload to A

    57_baseline_vintage.py     NEW
    56_dynamics_by_age.py      NEW
    run_lane6.py               NEW

## Run

    run_lane6.py       57 then 56      about 80 min

## Why

Two gaps, both of which matter if the effect is late.

TIME. Every register-immune design so far reports one number pooled over
the whole post period. A pooled number cannot separate a shock that
arrived with ChatGPT from a trend already running in 2019. 56 estimates
Poisson event studies on the stock, on hires and on separations,
referenced to 2022H1, with the 22-25 differential estimated inside the
same fit. The pre-period is the test and it is printed first.

ATTENUATION. Exposure frozen in 2019 is a six-year-old proxy by 2025, so
the design is weakest exactly where the effect is most likely. 57 turns
that from a caveat into a number: it rebuilds the exposure measure in
2021, 2022 and 2023 and regresses each on the 2019 version, giving
lambda(y), the share of the 2019 signal still present. It then re-runs
the whole design on a 2022 baseline, which is still pre-ChatGPT and three
years closer to the outcome years.

## How to read lane 6

If the 2022 baseline finds MORE than the 2019 one, the 2019 estimate is
an attenuated version of a real effect and must be reported as a LOWER
BOUND on the magnitude, never as a null. In the synthetic test, a world
with a drifting occupation mix gave +0.004 on the 2019 baseline and
-0.017 on the 2022 baseline: same data, same design, and the stale
baseline saw nothing.

lambda is measured only for years with their own occupation register.
For 2024 and 2025 it is extrapolated, and the summary says so. Dividing
an estimate by lambda corrects the point estimate and inflates its
standard error by the same factor: do both or neither.

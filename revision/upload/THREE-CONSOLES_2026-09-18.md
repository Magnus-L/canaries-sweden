# Running three consoles: what to upload, what to launch

18 Sep 2026. Replaces the one-console assumption in `UPLOAD.md` section 4.

## 1. Re-upload first: MONA is running four-day-old code

Nothing has run on the share since 5 September, so the share still holds the **4 September**
scripts. Two sets of changes never arrived:

- **6 Sep, the restart package.** `_r_workdir()` moved the R exchange files (the multi-million-row
  `_rin_*.csv`) from the share to local disk. On the old code every Poisson fit writes about a
  gigabyte over SMB and reads it back; that is most of why 43 took 667 minutes. Also
  `read_cache()` learned to treat a truncated parquet as absent, which is exactly the state a
  killed job leaves behind.
- **18 Sep, the three-console package.** Below.

**Upload these seven to** `\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\`,
overwriting:

    mona_common.py
    run_all_mona.py
    44_decile_gradient.py
    46_wfh_horserace.py
    45_asof_backtest.py
    47_edu_exposure.py
    48_gender_poisson.py      (new today)
    MANIFEST.txt

All seven upload as-is (`.py` and `.txt` are both allowed formats). Pre-flight hashes the scripts
against `MANIFEST.txt`, so upload the manifest in the same trip or every row reports BAD HASH.

**If 45 is running right now:** it is running the old code, which works but writes its exchange
files over SMB. If it started less than an hour ago, kill it, upload, and relaunch. Its SQL pulls
are cached per truncation and survive the restart, so you lose only the fits. If it is already
deep into the fits, let it finish and upload afterwards.

## 2. What today's patch changes

1. `--console 1|2|3` gives each console **its own master log** (`run_all_mona_log_1.txt`), so three
   processes no longer interleave lines into one file over SMB.
2. Each console writes a **heartbeat**, `_ALIVE_1.txt`, once a minute: timestamp, current stage,
   free memory. A console whose file stops moving is dead. That is the thing we could not see on
   5 September, and it is visible in Explorer without opening anything.
3. **A memory floor.** Before starting a stage the runner reads free physical memory and refuses to
   start below 15 GB (`--mem-floor` to override). The node ceiling is 100 GB and over-runs are
   killed without warning.
4. **R exchange files are now per script**, in `…\Temp\canaries_rwork\<script>\`. Two stages can no
   longer touch each other's temp files.
5. 44 and 46 **free the 140-million-row vintage panel** the moment the collapse has consumed it.
   Same rows, same numbers, roughly half the peak memory each.

Verified locally: the 52-check dry-run suite passes, R wrappers included.

## 3. Stamp 42 before anything else

42 finished but its parent died before writing the marker, so a plain restart re-runs it. In a
Python console:

    open(r"\\micro.intra\Projekt\P1207$\P1207_Gem\Magnus_P1207\canaries-sweden\round1_EL67898\output_42\_DONE", "w").write("manual 2026-09-18")

## 4. The three lanes: submit three files to BatchClient

BatchClient cannot pass command-line arguments, so there is nothing to type. Each lane is a file
you submit, exactly the way you submit any script:

| Submit this file | It runs | Why in this order |
|---|---|---|
| `run_console1.py` | 45, then 46 | 45 is the coverage defence and the heaviest job; 46 is Tier 2 and the first thing to cut |
| `run_console2.py` | 40, then 41 | Both are named in the editor's letter |
| `run_console3.py` | 44, then 48, then 47 | 44 carries the 50+ panel; 48 pulls its own gender panel; 47 is standalone |

Submit all three. Each prints to its own log (`run_all_mona_log_1.txt` and so on) and writes its
own heartbeat (`_ALIVE_1.txt`). If MONA queues them instead of running them side by side, that is
the scheduler's choice and they will simply run one after another; tell me if that happens.

**If a job is killed, submit the same file again.** Finished stages are skipped, so it resumes.

## 5. While they run

Look at the three `_ALIVE_*.txt` files every couple of hours. If one has stopped, read the tail of
that console's log, stamp `_DONE` for any stage whose outputs are complete, and relaunch that lane.
Do not retire the caches: `panel_vintage.parquet` and `panel_frozen.parquet` are what make every
restart cheap.

Export, when a lane finishes, is the file list in the 18 Sep session note, folder by folder.

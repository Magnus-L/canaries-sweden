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

## 4. The three lanes

Paired heavy-with-light, so the node is never carrying three big panels at once. Each console runs
one stage at a time, in this order:

| Console | Stages, in order | Weight |
|---|---|---|
| **1** | `--only 45`, then `--only 46` | Heavy. 45 is the coverage defence; 46 is Tier 2 and the first thing to cut |
| **2** | `--only 40`, then `--only 41` | SQL-heavy, memory-moderate. Both are named in the editor's letter |
| **3** | `--only 44`, then `--only 48`, then `47_edu_exposure.py` | 44 carries 50+, the biggest single panel; 48 pulls its own gender panel; 47 is standalone and light |

**Why 48 is new.** Poisson is now the paper's primary estimator, but the gender split in the
abstract is still an ln(n+1) estimate. 48 re-runs it in Poisson, after gating on whether the
gender pull, summed over gender, reproduces the headline coefficient. If that gate fails, stop and
tell me: it means the gender pull is not the pull the rest of the paper rests on.

    python run_all_mona.py --console 1 --only 45
    python run_all_mona.py --console 2 --only 40
    python run_all_mona.py --console 3 --only 44

`--only` skips the gate, which would otherwise cost 46 minutes per console for a number we already
have. 47 does not go through the runner at all: `python 47_edu_exposure.py`.

If a console refuses to start a stage on the memory floor, that is the patch working. Wait for the
neighbouring stage to finish rather than lowering the floor.

## 5. While they run

Look at the three `_ALIVE_*.txt` files every couple of hours. If one has stopped, read the tail of
that console's log, stamp `_DONE` for any stage whose outputs are complete, and relaunch that lane.
Do not retire the caches: `panel_vintage.parquet` and `panel_frozen.parquet` are what make every
restart cheap.

Export, when a lane finishes, is the file list in the 18 Sep session note, folder by folder.

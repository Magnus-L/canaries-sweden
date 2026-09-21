#!/usr/bin/env python3
"""
stage_upload.py -- keep upload/ identical to mona/, and say so.

WHY THIS EXISTS. upload/ holds physical COPIES of the scripts in mona/,
not links. Nothing kept them in step, so a script could be fixed in
mona/ and uploaded stale from upload/. That is the same defect that hit
us on the morning of 21 September with the .R/.txt twins, where the .R
carried a crash fix, the .txt did not, and the stale one went up. It
recurred the same afternoon: the industry-source and thread fixes went
into mona/ and upload/ still held the broken copies.

Run it after editing anything in mona/, and before any MONA trip:

    python3 stage_upload.py            # sync, and report what moved
    python3 stage_upload.py --check    # report only; exit 1 on drift

Files that live only in upload/ (data, run_lane wrappers, UPLOAD.md) are
left alone; only names present in BOTH folders are mirrored.
"""

import filecmp
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MONA, UP = HERE / "mona", HERE / "upload"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    check = "--check" in sys.argv
    drift = []
    clobber = []
    for u in sorted(list(UP.glob("*.py")) + list(UP.glob("*.R"))):
        m = MONA / u.name
        if m.exists() and not filecmp.cmp(m, u, shallow=False):
            # The sync runs one way, mona/ -> upload/. An upload/ copy
            # that is NEWER than its source is an edit made in the wrong
            # folder, and copying over it destroys work: that is exactly
            # how the _lane.py failure guard was lost on 21 September,
            # minutes after it was written and tested.
            if u.stat().st_mtime > m.stat().st_mtime:
                clobber.append((m, u))
            else:
                drift.append((m, u))

    if clobber:
        print("REFUSING TO SYNC. These upload/ files are NEWER than "
              "mona/, so they hold edits made in the wrong folder:")
        for m, u in clobber:
            print(f"  {u.name}: upload/ {sha(u)} is newer than mona/ {sha(m)}")
        print("\nmona/ is the source. Move the change there, then re-run.")
        return 1

    if not drift:
        print("upload/ is identical to mona/ for every shared file.")
        return 0

    for m, u in drift:
        if check:
            print(f"  DRIFT {u.name}: mona {sha(m)} != upload {sha(u)}")
        else:
            shutil.copy2(m, u)
            print(f"  staged {u.name}  ({sha(u)})")

    if check:
        print(f"\n{len(drift)} stale file(s) in upload/. "
              f"Run without --check to sync.")
        return 1
    # copy2 preserves mtime, which _lane.py reads for staleness, so the
    # lane still sees a changed dependency and will not SKIP the stage.
    bad = [u.name for m, u in drift if sha(m) != sha(u)]
    if bad:
        print(f"\nSYNC FAILED for: {bad}")
        return 1
    print(f"\n{len(drift)} file(s) staged; upload/ now matches mona/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

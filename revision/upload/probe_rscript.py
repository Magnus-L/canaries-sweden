#!/usr/bin/env python3
"""
probe_rscript.py -- find R on THIS batch node and write the pin file.

Submit this to BatchClient when pre-flight reports "Rscript not found". It
searches widely, prints everything it finds, and if it finds a working
Rscript it writes rscript_path.txt beside these scripts, which every later
run reads first. Nothing else in the round depends on it.

The 4 September node had R at E:\\Programs\\R-4.5.0\\bin\\x64\\Rscript.exe,
reached through the registry. The 18 September node has neither that
registry key nor R under C:\\Program Files, and the nodes are not
interchangeable, so this probe exists to stop the guessing.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = open(HERE / "probe_rscript_log.txt", "w", encoding="utf-8",
           errors="replace")


def say(s=""):
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


say("PROBE: where is Rscript on this node?")
say("=" * 60)
found = []

from shutil import which
w = which("Rscript")
say(f"PATH           : {w or 'not on PATH'}")
if w:
    found.append(w)

say(f"R_HOME env     : {os.environ.get('R_HOME', 'unset')}")
if os.environ.get("R_HOME"):
    for sub in (r"bin\x64\Rscript.exe", r"bin\Rscript.exe"):
        c = Path(os.environ["R_HOME"]) / sub
        say(f"  {c}  {'EXISTS' if c.exists() else 'no'}")
        if c.exists():
            found.append(str(c))

say("\nRegistry:")
try:
    import winreg
    for hive, hname in ((winreg.HKEY_LOCAL_MACHINE, "HKLM"),
                        (winreg.HKEY_CURRENT_USER, "HKCU")):
        for subkey in (r"SOFTWARE\R-core\R", r"SOFTWARE\WOW6432Node\R-core\R",
                       r"SOFTWARE\R-core\R64",
                       r"SOFTWARE\WOW6432Node\R-core\R64"):
            try:
                with winreg.OpenKey(hive, subkey) as k:
                    base = winreg.QueryValueEx(k, "InstallPath")[0]
                say(f"  {hname}\\{subkey} -> {base}")
                for sub in (r"bin\x64\Rscript.exe", r"bin\Rscript.exe"):
                    c = Path(base) / sub
                    if c.exists():
                        found.append(str(c))
            except OSError as ex:
                say(f"  {hname}\\{subkey} -> {type(ex).__name__}")
except ImportError:
    say("  winreg unavailable")

say("\nDirectory search (this is the slow part):")
roots = [r"{d}:\Program Files\R", r"{d}:\Program Files (x86)\R",
         r"{d}:\Programs", r"{d}:\Program", r"{d}:\R", r"{d}:\Apps",
         r"{d}:\Tools", r"{d}:\Software"]
for drive in "CDEFGH":
    if not Path(f"{drive}:\\").exists():
        continue
    say(f"  drive {drive}: present")
    for root in roots:
        base = root.format(d=drive)
        if not Path(base).exists():
            continue
        say(f"    {base} exists; listing R-like entries:")
        try:
            for entry in sorted(os.listdir(base)):
                if entry.lower().startswith("r"):
                    say(f"      {entry}")
        except OSError as ex:
            say(f"      unreadable: {ex}")
        for pat in (r"\R-*\bin\x64\Rscript.exe", r"\R-*\bin\Rscript.exe",
                    r"\*\bin\x64\Rscript.exe", r"\*\bin\Rscript.exe"):
            for hit in sorted(glob.glob(base + pat)):
                say(f"      HIT {hit}")
                found.append(hit)

say("\n" + "=" * 60)
found = sorted(set(found))
if not found:
    say("NOTHING FOUND. Send this log back; R may be under a path this probe")
    say("does not reach, or it may not be installed on this node at all.")
    sys.exit(1)

for c in found:
    try:
        r = subprocess.run([c, "--version"], capture_output=True, text=True,
                           timeout=60)
        ver = (r.stdout + r.stderr).strip().splitlines()[0][:60]
        say(f"WORKS  {c}   {ver}")
    except Exception as ex:
        say(f"fails  {c}   {type(ex).__name__}")
        continue
    (HERE / "rscript_path.txt").write_text(c)
    say(f"\nWROTE rscript_path.txt -> {c}")
    say("Every later run reads that file first. Resubmit the console jobs.")
    break

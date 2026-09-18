#!/usr/bin/env python3
"""
l12_check_jobtech_quarters.py -- is there posting data we are not using?

The posting regressions run to the last closed quarter JobTech publishes.
This asks the server what it now has, compares that with the newest file in
the local cache, and says whether a rerun would add anything. One command,
so the question "are we current?" never again needs a manual look.

    python l12_check_jobtech_quarters.py

It fetches one HTML index from data.jobtechdev.se (allowlisted, CC0) and
downloads nothing.
"""

import re
import sys
import urllib.request
from pathlib import Path

INDEX = "https://data.jobtechdev.se/annonser/historiska/index.html"
CACHE = Path.home() / ".cache" / "aiel-jobads"
USED_THROUGH = "2026-Q2"          # what the current results are built on


def server_files():
    req = urllib.request.Request(
        INDEX, headers={"User-Agent": "AI-Econ Lab research (mlodefalk@gmail.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r"(20\d\d(?:-Q[1-4])?)\.jsonl\.zip", html)))


def main():
    files = server_files()
    local = sorted(p.name.replace(".jsonl.zip", "")
                   for p in CACHE.glob("*.jsonl.zip")) if CACHE.exists() else []
    newest_server = files[-1] if files else "none"
    print(f"server newest : {newest_server}")
    print(f"local newest  : {local[-1] if local else 'none'}")
    print(f"results use   : {USED_THROUGH}")

    missing = [f for f in files if f not in local]
    if missing:
        print(f"\nNOT IN THE LOCAL CACHE: {', '.join(missing)}")
    if newest_server > USED_THROUGH:
        print(f"\nACTION: {newest_server} is published and the results stop at "
              f"{USED_THROUGH}. Download it into {CACHE}, then rerun "
              f"l01, l03, l05 and l08, and l07 for the figure.")
        return 1
    print("\nThe results are built on the newest published quarter. Nothing to do.")
    print("JobStream is NOT an alternative for the months in between: it "
          "returns currently published advertisements, not everything ever "
          "published, which is what produced the 85 per cent apparent "
          "collapse in January 2026.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

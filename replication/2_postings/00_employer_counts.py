#!/usr/bin/env python3
"""
00_employer_counts.py: the employer-by-month-by-occupation advertisement counts
that the within-employer design of Online Appendix Part V reads (scripts 14
and 15), rebuilt from the public Platsbanken archives.

WHY IT EXISTS
The counts have one row per employer, month, occupation and municipality, so
the package ships the code that builds them rather than the file. The paper's
file was built by the authors on 21 August 2026 from the same archives under
the rules below; this script applies those rules and writes the columns Part V
reads.

THE RULES (as in the paper's build)
  - archives 2021 to 2026-Q2; within each archive, an advertisement is counted
    once, de-duplicated on an 8-byte BLAKE2b digest of the headline, the
    employer's name and the first 400 characters of the description;
  - publication month January 2021 to June 2026;
  - the employer is the organisation number printed in the advertisement,
    reduced to ten digits (a twelve-digit form loses its century prefix);
    advertisements without a usable number are dropped (about one per cent);
  - an organisation number whose third digit is 0 or 1 belongs to a sole
    trader and is that person's identity number. It is never written: it is
    replaced by a keyed BLAKE2b hash, "EF:" plus 16 hexadecimal characters;
  - the occupation is the first `legacy_ams_taxonomy_id` of `occupation_group`
    (four-digit SSYK 2012; "0000" when absent), the municipality the first four
    characters of the workplace municipality code;
  - an advertisement is entry-level when the regular expression ENTRY below
    matches its lower-cased headline, description, requirements and
    conditions.

THE HASH KEY
The key is read from CANARIES_EF_KEY. If it is not set, a random key is drawn
for the run and not stored, so the pseudonyms cannot be reversed by anyone,
including the authors. The key of the paper's build is not published, for the
same reason. The pseudonyms differ between keys, but every count and every
estimate does not: a sole trader is the same employer throughout one run, and
the design uses the employer only as a fixed effect and as a unit to count.

INPUTS   the Platsbanken archives in config.JOBADS_DIR (1_data_public/01)
OUTPUTS  config.FIRM_CUBE (default data/raw/firm_month_v2.csv.gz):
         orgnr, month, ssyk4, kommun, ads, entry
SERVES   Online Appendix Part V (Tables A37 to A39), through 14 and 15
RUNTIME  about 10 minutes on the seven archives
"""

import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ARCHIVES = ["2021", "2022", "2023", "2024", "2025", "2026-Q1", "2026-Q2"]
WINDOW = ("2021-01", "2026-06")
ENTRY = re.compile(r"\b(nyexaminerad|nyutexaminerad|junior|trainee(?:program)?|"
                   r"ingen erfarenhet|utan (?:tidigare )?erfarenhet)\b")
NON_DIGIT = re.compile(r"\D")

_key = os.environ.get("CANARIES_EF_KEY")
EF_KEY = _key.encode("utf-8") if _key else os.urandom(32)


def dedup_key(ad: dict) -> bytes:
    """Digest of headline, employer name and the first 400 characters of the text."""
    e = ad.get("employer") or {}
    name = e.get("name") if isinstance(e, dict) else str(e)
    body = ((ad.get("description") or {}).get("text") or "")[:400]
    h = hashlib.blake2b(digest_size=8)
    h.update(((ad.get("headline") or "") + "\x00" + (name or "") + "\x00"
              + body).encode("utf-8"))
    return h.digest()


def ad_text(ad: dict) -> str:
    """Headline, description, requirements and conditions, as the entry flag reads them."""
    d = ad.get("description") or {}
    parts = [ad.get("headline") or ""]
    for key in ("text", "requirements", "conditions"):
        if d.get(key):
            parts.append(d[key])
    return "\n".join(parts)


def occupation(node) -> str:
    """The first legacy AMS taxonomy id (four-digit SSYK 2012) of occupation_group."""
    if isinstance(node, dict):
        return node.get("legacy_ams_taxonomy_id") or ""
    if isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and item.get("legacy_ams_taxonomy_id"):
                return item["legacy_ams_taxonomy_id"]
    return ""


def employer_id(raw) -> str:
    """Ten-digit organisation number; a keyed pseudonym for a sole trader; '' if unusable."""
    if not raw:
        return ""
    d = NON_DIGIT.sub("", str(raw))
    if len(d) == 12 and d[:2] in ("16", "18", "19", "20"):
        d = d[2:]
    if len(d) != 10:
        return ""
    if int(d[2]) < 2:          # a personal identity number: never written as it is
        return "EF:" + hashlib.blake2b(d.encode(), digest_size=8, key=EF_KEY).hexdigest()
    return d


def main():
    print("Employer-by-month-by-occupation advertisement counts, 2021 to June 2026")
    counts = defaultdict(lambda: [0, 0])          # (orgnr, month, ssyk4, kommun) -> ads, entry
    for stem in ARCHIVES:
        zp = config.platsbanken_zip(stem)
        if not zp.exists():
            sys.exit(f"missing archive {zp}; run 1_data_public/01 first")
        t0, n, kept = time.time(), 0, 0
        seen = set()                               # de-duplication within an archive
        with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    ad = json.loads(line)
                except ValueError:
                    continue
                n += 1
                k = dedup_key(ad)
                if k in seen:
                    continue
                seen.add(k)
                month = (ad.get("publication_date") or "")[:7]
                if not (WINDOW[0] <= month <= WINDOW[1]):
                    continue
                e = ad.get("employer") or {}
                org = employer_id(e.get("organization_number") if isinstance(e, dict) else "")
                if not org:
                    continue
                ssyk = occupation(ad.get("occupation_group")) or "0000"
                wa = ad.get("workplace_address") or {}
                kommun = ((wa.get("municipality_code") or wa.get("municipality_concept_id")
                           or "")[:4] if isinstance(wa, dict) else "")
                c = counts[(org, month, ssyk, kommun)]
                c[0] += 1
                c[1] += 1 if ENTRY.search(ad_text(ad).lower()) else 0
                kept += 1
        print(f"  {stem}: {n:,} advertisements read, {kept:,} counted, "
              f"{time.time() - t0:.0f} s")
        del seen

    out = config.FIRM_CUBE
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orgnr", "month", "ssyk4", "kommun", "ads", "entry"])
        for key in sorted(counts, key=lambda k: (k[1], k[0], k[2], k[3])):
            w.writerow(list(key) + counts[key])
    print(f"  wrote {out}: {len(counts):,} cells, "
          f"{sum(c[0] for c in counts.values()):,} advertisements")


if __name__ == "__main__":
    main()

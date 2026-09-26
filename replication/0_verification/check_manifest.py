#!/usr/bin/env python3
r"""
check_manifest.py: compare every number the paper and its online appendix print
with the file it is computed from, and report PASS or FAIL for each.

MANIFEST.csv (at the package root) holds one row per printed number or per
generated table. Each row is checked twice:

  1. In print. The printed value must appear in the named manuscript file
     (main_v3.tex or appendix_v3.tex, or a table file the manuscript inputs).
     TeX typography is normalised first: $-0.0578$, $-$0.0578, -0.0578 and the
     Unicode minus all match; 104{,}217 and 104,217 match; a number never
     matches inside a longer number, and a positive value never matches its
     own negative.

  2. At source. The value is read from the export or result file the row names
     and must round to the printed value at the printed precision. How the
     value is read is set by the row's `locator`:

        (empty)                    any number in the file that rounds to the
                                   printed value
        csv:<query>|<column>       pandas query on a CSV, one row, one column,
                                   e.g.  csv:young_band=='22-25' and term=='post'|coef
        regex:<pattern>            first capture group of a regular expression
                                   applied to a text file

     and an optional `transform`, a Python expression in x (and the functions
     exp, log, abs, round), applied before rounding, e.g. 100*(exp(x)-1) to
     turn a Poisson coefficient into the per-cent change the text quotes.
     A `derived` row combines several values: its locator is a
     semicolon-separated list of name=locator pairs and its transform an
     expression in those names. A pair reads the row's source unless its
     locator starts with @<path>@, which names another file in the package.
     A csv locator whose column is written sum:<col>, max:<col>, min:<col> or
     mean:<col> aggregates over every row the query selects.

  A number the paper writes as a word ("one per cent", "twelvefold" is not
  covered) is entered as the word; it is looked up in print as the word and
  compared at source as its value.

  Rows of kind `table` compare a generated table in output/tables/ with the
  file of the same name that the manuscript inputs, line by line after
  collapsing runs of spaces (as LaTeX does), stripping trailing whitespace,
  dropping blank lines and whole-line comments,
  and accepting the co-author markup of the revision (\add{x} is read as x,
  \del{x} and \rem{x} are removed), so that a printed file still carrying
  markup compares equal when its accepted text is the built text; every
  number in the table is then checked at once.

Status per row: PASS, FAIL (with the reason), or MISSING when the source file
is not in the package (a MONA export that was never brought out). A row may
carry status PENDING in MANIFEST.csv, in which case it is reported and not
checked. The exit code is 1 if any row FAILs.

Usage:
    python 0_verification/check_manifest.py                 # manuscript at CANARIES_PAPER_DIR
    python 0_verification/check_manifest.py --paper PATH    # or any other folder
    python 0_verification/check_manifest.py --only-source   # skip the print check
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE))
import config  # noqa: E402

MANIFEST = PACKAGE / "MANIFEST.csv"

# Numbers the paper writes as words; the print check looks for the word, the
# source check compares the value.
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
         "fifteen": 15, "twenty": 20, "twenty-five": 25, "fifty": 50,
         "twice": 2}
NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+")


# -- the print check -------------------------------------------------------------
def normalise_tex(text: str) -> str:
    """Flatten TeX typography so that one pattern matches every printed form."""
    text = text.replace("{,}", ",").replace("\\,", ",")
    text = text.replace("−", "-").replace("$-$", "-").replace("$+$", "+")
    text = text.replace("$", "").replace("\\%", "%")
    text = text.replace("~", " ")
    return re.sub(r"\s+", " ", text)


def printed_pattern(value: str) -> re.Pattern:
    """Pattern for a printed number: exact digits, bounded on both sides."""
    v = value.strip()
    neg = v.startswith("-")
    body = v.lstrip("+-")
    variants = {re.escape(body), re.escape(body.replace(",", ""))}
    ip = body.split(".")[0].replace(",", "")
    rest = body[len(body.split(".")[0]):]
    if ip.isdigit() and len(ip) > 3:
        variants.add(re.escape(f"{int(ip):,}" + rest))
    alt = "(?:" + "|".join(sorted(variants, key=len, reverse=True)) + ")"
    if neg:
        return re.compile(r"(?<![\d.])-" + alt + r"(?![\d])(?!\.\d)")
    return re.compile(r"(?<![\d.])(?<!-)\+?" + alt + r"(?![\d])(?!\.\d)")


def is_number(value: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", value.strip()))


def in_print(row: dict, paper: Path, cache: dict) -> tuple[bool, str]:
    doc = row["document"].strip()
    path = paper / doc
    if not path.is_file():
        return False, f"manuscript file not found: {doc}"
    if path not in cache:
        cache[path] = normalise_tex(path.read_text(encoding="utf-8", errors="replace"))
    text = cache[path]
    value = row["printed"].strip()
    if is_number(value):
        ok = printed_pattern(value).search(text) is not None
    else:
        ok = normalise_tex(value) in text
    return ok, "" if ok else f"'{value}' not found in {doc}"


# -- the source check ----------------------------------------------------------------
def decimals_of(value: str) -> int:
    body = value.strip().lstrip("+-").replace(",", "")
    return len(body.split(".")[1]) if "." in body else 0


def rounds_to(x: float, value: str) -> bool:
    """True when x, rounded half-up at the printed precision, equals the print."""
    target = Decimal(value.strip().replace(",", "").lstrip("+"))
    q = Decimal(1).scaleb(-decimals_of(value))
    got = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if got == target:
        return True
    # Half-even and float representation can differ in the last digit; accept a
    # value within half a unit of the last printed digit, inclusive.
    return abs(Decimal(repr(float(x))) - target) <= q / 2 + Decimal("1e-12")


def read_locator(src: Path, locator: str):
    """Return one number (or None for 'any number in the file')."""
    if not locator:
        return None
    kind, _, spec = locator.partition(":")
    if kind == "csv":
        import pandas as pd
        query, _, column = spec.rpartition("|")
        df = pd.read_csv(src)
        sel = df.query(query) if query else df
        agg, _, col = column.rpartition(":")
        if agg:  # sum:, max:, min:, mean: over the selected rows
            return float(getattr(sel[col], agg)())
        if len(sel) != 1:
            raise ValueError(f"query returned {len(sel)} rows: {query}")
        return float(sel[column].iloc[0])
    if kind == "regex":
        m = re.search(spec, src.read_text(encoding="utf-8", errors="replace"), re.S)
        if not m:
            raise ValueError(f"pattern not found: {spec}")
        return float(m.group(1).replace(",", ""))
    raise ValueError(f"unknown locator kind: {kind}")


SAFE = {"exp": math.exp, "log": math.log, "abs": abs, "round": round, "sqrt": math.sqrt}


def at_source(row: dict) -> tuple[str, str]:
    src = PACKAGE / row["source"].strip()
    if not src.is_file():
        return "MISSING", f"source not in the package: {row['source']}"
    value = row["printed"].strip()
    value = str(WORDS.get(value.lower(), value))
    locator = row.get("locator", "").strip()
    transform = row.get("transform", "").strip()
    try:
        if row["kind"].strip() == "derived":
            names = {}
            for part in locator.split(";"):
                name, _, loc = part.partition("=")
                loc, part_src = loc.strip(), src
                if loc.startswith("@"):  # a value read from a second file
                    other, _, loc = loc[1:].partition("@")
                    part_src = PACKAGE / other
                names[name.strip()] = read_locator(part_src, loc)
            x = eval(transform, {"__builtins__": {}}, {**SAFE, **names})
            ok = rounds_to(x, value)
            return ("PASS", "") if ok else ("FAIL", f"source gives {x:.6g}")
        x = read_locator(src, locator)
        if x is None:
            numbers = [float(n.replace(",", "")) for n in NUM.findall(src.read_text(
                encoding="utf-8", errors="replace"))]
            if transform:
                numbers = [eval(transform, {"__builtins__": {}}, {**SAFE, "x": n})
                           for n in numbers]
            ok = any(rounds_to(n, value) for n in numbers)
            return ("PASS", "") if ok else ("FAIL", "no number in the source rounds to it")
        if transform:
            x = eval(transform, {"__builtins__": {}}, {**SAFE, "x": x})
        ok = rounds_to(x, value)
        return ("PASS", "") if ok else ("FAIL", f"source gives {x:.6g}")
    except Exception as ex:  # a broken locator is a failure of the row, not of the run
        return "FAIL", f"{type(ex).__name__}: {ex}"


def strip_command(text: str, cmd: str, keep: bool) -> str:
    r"""Remove every \cmd{...} from TeX source, brace-aware, keeping the
    argument's text when `keep` is true (an insertion) and dropping it
    otherwise (a deletion)."""
    out, i, tag = [], 0, "\\" + cmd + "{"
    while True:
        j = text.find(tag, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k, depth = j + len(tag), 1
        while depth and k < len(text):
            c = text[k]
            if c == "\\":
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            k += 1
        inner = text[j + len(tag):k - 1]
        out.append(strip_command(inner, cmd, keep) if keep else "")
        i = k
    return "".join(out)


def table_lines(text: str) -> list[str]:
    r"""The lines of a table file as they are compared: the co-author
    markup of the revision is accepted (\add{x} becomes x, \del{x} and
    \rem{x} are removed), whole-line comments and blank lines are dropped,
    and trailing whitespace is stripped. A builder writes no comments and
    no markup, so a printed file that carries either still compares equal
    when its accepted text is the built text."""
    for cmd, keep in (("del", False), ("add", True), ("rem", False)):
        text = strip_command(text, cmd, keep)
    return [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("%")]


def table_identity(row: dict, paper: Path) -> tuple[str, str]:
    built = PACKAGE / row["source"].strip()
    printed = paper / row["document"].strip()
    if not built.is_file():
        return "MISSING", f"not built: {row['source']} (run the pack that writes it)"
    if not printed.is_file():
        return "FAIL", f"manuscript table not found: {row['document']}"
    a = table_lines(built.read_text(encoding="utf-8"))
    b = table_lines(printed.read_text(encoding="utf-8"))
    if a == b:
        return "PASS", ""
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return "FAIL", f"line {i} differs: built '{x[:60]}' / printed '{y[:60]}'"
    return "FAIL", f"length differs: built {len(a)} lines, printed {len(b)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper", default=str(config.PAPER_DIR))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--only-source", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only rows that do not pass")
    args = ap.parse_args()
    paper = Path(args.paper).resolve()
    have_paper = paper.is_dir()
    if not have_paper and not args.only_source:
        print(f"manuscript folder not found ({paper}); checking sources only")

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    counts = {"PASS": 0, "FAIL": 0, "MISSING": 0, "PENDING": 0}
    cache: dict = {}
    for row in rows:
        kind = row["kind"].strip()
        if row.get("status", "").strip().upper() == "PENDING":
            verdict, why = "PENDING", row.get("note", "")
        elif kind == "table":
            verdict, why = (table_identity(row, paper) if have_paper
                            else ("MISSING", "no manuscript folder"))
        else:
            verdict, why = at_source(row)
            if verdict == "PASS" and have_paper and not args.only_source:
                ok, why_p = in_print(row, paper, cache)
                if not ok:
                    verdict, why = "FAIL", why_p
        counts[verdict] += 1
        if not args.quiet or verdict != "PASS":
            line = f"{verdict:7} {row['id']:>5}  {row['where'][:28]:28}  {row['printed']:>12}  {row['statistic'][:60]}"
            print(line + (f"   [{why}]" if why else ""))
    print("-" * 78)
    print(f"{len(rows)} rows: PASS {counts['PASS']}  FAIL {counts['FAIL']}  "
          f"MISSING {counts['MISSING']}  PENDING {counts['PENDING']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

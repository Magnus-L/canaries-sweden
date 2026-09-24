#!/usr/bin/env python3
"""
check_mona_scripts.py: prove that the register scripts shipped in
3_register_mona/scripts/ contain exactly the code that ran inside MONA.

The shipped copies differ from the copies that ran in their comments and
docstrings only: internal working notes were removed and the documentation
was brought up to date with the manuscript. This script checks that claim
for every file, in two ways.

  Python files. Both versions are parsed with the standard `ast` module,
  every docstring (module, class and function) is removed, and the
  remaining syntax tree is serialised canonically. Two files whose
  serialisations agree execute identically: comments never reach the tree,
  and every string literal, number, name, call and control-flow statement
  does.

  R files. Comments are removed (a '#' outside a string starts a comment)
  and the remaining text is compared as a stream of whitespace-separated
  tokens.

What each shipped file is compared against, in order of preference:

  1. the as-run copy in the research repository, revision/mona/<file>,
     when the package sits inside that repository (the default);
  2. otherwise the code fingerprint recorded for the as-run copy in
     3_register_mona/SCRIPTS.csv (column as_run_code_fingerprint), so
     that a standalone copy of the package can still be checked.

Output: one line per file, IDENTICAL or DIFFERENT, and a count. The exit
code is 1 if any file differs or cannot be checked.

Usage:
    python 0_verification/check_mona_scripts.py
    python 0_verification/check_mona_scripts.py --as-run PATH   # another as-run folder
    python 0_verification/check_mona_scripts.py --fingerprints-only
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
SHIPPED = PACKAGE / "3_register_mona" / "scripts"
SCRIPTS_CSV = PACKAGE / "3_register_mona" / "SCRIPTS.csv"
AS_RUN_DEFAULT = PACKAGE.parent / "revision" / "mona"


# -- Python ------------------------------------------------------------------------
def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _canon(node) -> object:
    """A serialisation of a syntax tree that does not depend on how a given
    Python version prints it: every non-empty field, by name, recursively."""
    if isinstance(node, ast.AST):
        fields = []
        for name in node._fields:
            value = getattr(node, name, None)
            if value is None or value == []:
                continue
            fields.append((name, _canon(value)))
        return (type(node).__name__, tuple(fields))
    if isinstance(node, list):
        return tuple(_canon(v) for v in node)
    return (type(node).__name__, repr(node))


def python_fingerprint(source: str) -> str:
    tree = _strip_docstrings(ast.parse(source))
    return hashlib.sha256(repr(_canon(tree)).encode("utf-8")).hexdigest()


# -- R -------------------------------------------------------------------------------
def r_tokens(source: str) -> str:
    out = []
    for line in source.splitlines():
        kept, quote, escaped = [], None, False
        for ch in line:
            if quote:
                kept.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
                kept.append(ch)
            elif ch == "#":
                break
            else:
                kept.append(ch)
        out.extend("".join(kept).split())
    return " ".join(out)


def r_fingerprint(source: str) -> str:
    return hashlib.sha256(r_tokens(source).encode("utf-8")).hexdigest()


def fingerprint(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return python_fingerprint(text) if path.suffix == ".py" else r_fingerprint(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-run", default=str(AS_RUN_DEFAULT),
                    help="folder holding the as-run copies (default: %(default)s)")
    ap.add_argument("--fingerprints-only", action="store_true",
                    help="compare with SCRIPTS.csv only, even if the as-run folder exists")
    args = ap.parse_args()
    as_run = Path(args.as_run)
    use_files = as_run.is_dir() and not args.fingerprints_only

    recorded = {}
    if SCRIPTS_CSV.is_file():
        with SCRIPTS_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                recorded[row["script"]] = row.get("as_run_code_fingerprint", "")

    shipped = sorted(p for p in SHIPPED.iterdir() if p.suffix in (".py", ".R"))
    n_same = n_diff = 0
    print(f"comparing {len(shipped)} shipped files with "
          + (f"the as-run copies in {as_run}" if use_files
             else "the fingerprints in SCRIPTS.csv"))
    for p in shipped:
        mine = fingerprint(p)
        if use_files and (as_run / p.name).is_file():
            theirs, source = fingerprint(as_run / p.name), "as-run file"
        elif recorded.get(p.name):
            theirs, source = recorded[p.name], "SCRIPTS.csv"
        else:
            print(f"  CANNOT CHECK  {p.name}  (no as-run copy and no recorded fingerprint)")
            n_diff += 1
            continue
        if mine == theirs:
            n_same += 1
            print(f"  IDENTICAL     {p.name}  [{source}]")
        else:
            n_diff += 1
            print(f"  DIFFERENT     {p.name}  [{source}]")
    print("-" * 72)
    print(f"{n_same} of {len(shipped)} files IDENTICAL in code; {n_diff} not")
    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

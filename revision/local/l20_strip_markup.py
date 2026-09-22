#!/usr/bin/env python3
r"""
l20_strip_markup.py -- regenerate the clean manuscript from the marked one.

`main_v2.tex` in the paper repo IS the marked source: it defines
\add{x} (red) and \del{x} (red, struck) and currently carries one tracked
change, the rewritten passage in the introduction. So the compiled
`main_v2.pdf` shows red struck-through text, which is right for a
co-author read and wrong for anything that leaves the house.

This writes `main_v2_clean.tex` beside it: \del{} spans dropped, \add{}
unwrapped, the markup preamble neutralised so ulem is not loaded. It
never modifies the marked file. Nesting and escaped braces are handled
the same way as the daioe original, which this is adapted from
(`projects/daioe/scripts/strip_markup.py`, hardcoded to that paper).

    python3 revision/local/l20_strip_markup.py [--count]

--count also prints the clean word count by section, which is the number
that matters against the journal limit: the marked file counts deleted
text that will not appear.
"""
import re
import sys
from pathlib import Path

PAPER = Path("/Users/mslk/Documents/Workspace/projects/canaries-sweden-paper")
SRC = PAPER / "main_v2.tex"
DST = PAPER / "main_v2_clean.tex"
TAIL = ("Declaration of competing interest", "Funding", "Data availability",
        "Acknowledgements", "generative AI")


def strip(t: str, cmd: str, keep: bool) -> str:
    """Remove \\cmd{...}, keeping the content only if keep. Brace-aware."""
    out, i, tag = [], 0, "\\" + cmd + "{"
    while True:
        j = t.find(tag, i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k, depth = j + len(tag), 1
        while depth:
            c = t[k]
            if c == "\\":
                k += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            k += 1
        inner = t[j + len(tag):k - 1]
        out.append(strip(inner, cmd, keep) if keep else "")
        i = k
    return "".join(out)


def words(t: str) -> int:
    t = re.sub(r"(?m)%.*$", "", t)
    t = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", "", t, flags=re.S)
    t = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", t, flags=re.S)
    t = re.sub(r"\\(bibliography|input|includegraphics|label|ref|eqref|cite\w*)"
               r"\s*\{[^}]*\}", " X ", t)
    t = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " X ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", t)
    t = re.sub(r"[{}\\&~^_#]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


def mask_comments(t: str) -> tuple:
    r"""
    Hide whole-line LaTeX comments from the stripper.

    The preamble documents the markup by writing \add{new text} and
    \del{old text} in comments. Without this the stripper ate its own
    instructions, which is why the note in main_v2.tex once read "drops
    spans and unwraps ." with the examples deleted out of it.
    """
    kept, out = [], []
    for line in t.split("\n"):
        if line.lstrip().startswith("%"):
            out.append(f"@@CMT{len(kept)}@@")
            kept.append(line)
        else:
            out.append(line)
    return "\n".join(out), kept


def unmask_comments(t: str, kept: list) -> str:
    for i, line in enumerate(kept):
        t = t.replace(f"@@CMT{i}@@", line)
    return t


def main() -> int:
    raw = SRC.read_text(encoding="utf-8")
    t, kept = mask_comments(raw)
    n_del, n_add, n_rem = (t.count(r"\del{"), t.count(r"\add{"),
                           t.count(r"\rem{"))
    t = strip(t, "del", False)
    t = strip(t, "add", True)
    t = strip(t, "rem", False)      # co-author comments never leave the house
    t = unmask_comments(t, kept)
    # the macros are now unused; leave them defined but inert so the file
    # still compiles if a stray \add survives a future edit
    t = t.replace(r"\usepackage[normalem]{ulem}",
                  "% ulem not needed in the clean file")
    t = t.replace(r"\newcommand{\del}[1]{\textcolor{red}{\sout{#1}}}",
                  r"\newcommand{\del}[1]{}")
    t = t.replace(r"\newcommand{\rem}[1]{\textcolor{red}{[#1]}}",
                  r"\newcommand{\rem}[1]{}")
    DST.write_text(t, encoding="utf-8")
    print(f"  wrote {DST.name}: dropped {n_del} \\del span(s), "
          f"unwrapped {n_add} \\add span(s), "
          f"dropped {n_rem} \\rem comment(s)")

    if "--count" in sys.argv:
        body = t.split(r"\begin{document}")[-1].split(r"\end{document}")[0]
        tot = 0
        print("\n  clean word count, four sections (journal limit 2,000):")
        for b in re.split(r"\\section\*?\{", body)[1:]:
            name = b.split("}")[0]
            if any(x in name for x in TAIL):
                continue
            c = words(b)
            tot += c
            print(f"    {c:5d}  {name}")
        print(f"    -----\n    {tot:5d}  total  ({tot - 2000:+d})")
        marked = SRC.read_text(encoding="utf-8")
        mb = marked.split(r"\begin{document}")[-1].split(r"\end{document}")[0]
        mt = sum(words(b) for b in re.split(r"\\section\*?\{", mb)[1:]
                 if not any(x in b.split("}")[0] for x in TAIL))
        print(f"    (the marked file counts {mt}, which over-states by "
              f"{mt - tot} words of struck text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

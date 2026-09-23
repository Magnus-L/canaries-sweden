#!/usr/bin/env python3
r"""
l20_strip_markup.py -- accept all tracked changes, in the file itself.

THIS REPLACES THE CLEAN-TWIN WORKFLOW, DROPPED 23 SEPTEMBER 2026.
Until today the script wrote `main_vN_clean.tex` beside the marked
`main_vN.tex`, and the twin was the file that left the house. It went
stale silently: on 23 September `main_v3_clean.tex` sat two hours behind
`main_v3.tex` through a run of edits, and had it been sent it would have
carried a withdrawn estimate, a superseded bound and a retired claim. A
duplicate that disagrees with its source is worse than no duplicate.

So markup is now TRANSIENT, the way track changes are in a word
processor. Mark the live file while a co-author reads it; run this when
the round closes and the marks are gone from that same file. There is
never a second manuscript, and `main_vN.tex` is always the truth.

    python3 revision/local/l20_strip_markup.py [v1|v2|v3] [--count] [--force]

\del{} spans are dropped, \add{} spans unwrapped, \rem{} comments
dropped, because co-author comments never leave the house.

THE PREAMBLE SURVIVES. The old script neutralised \usepackage{ulem} and
the macro definitions, which was right for a dead-end artefact and is
wrong here: the file goes on living and will be marked again next round.
Only the USES are removed.

SAFETY. The script refuses to rewrite a file with uncommitted changes,
so `git checkout` can always undo it. Pass --force to override.

--count prints the word count by section after stripping, which is the
number that matters against the journal limit, and says how much struck
text the marked file was over-counting by.
"""
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path("/Users/mslk/Documents/Workspace/projects/canaries-sweden-paper")
# v3 is the live manuscript, the occupation route. v1 and v2 are kept for
# comparison and are frozen; passing them is allowed but unusual.
VERSION = next((a for a in sys.argv[1:] if not a.startswith("-")), "v3")
SRC = PAPER / f"main_{VERSION}.tex"
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


def dirty(path: Path) -> bool:
    """True if the file has uncommitted changes, so a rewrite could not
    be undone with git checkout."""
    try:
        r = subprocess.run(["git", "-C", str(path.parent), "status",
                            "--porcelain", "--", path.name],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return True          # cannot tell, so assume the worst
    if r.returncode != 0:
        return True          # not a repo, or git failed: assume the worst
    return bool(r.stdout.strip())


def section_counts(body: str) -> tuple:
    tot, rows = 0, []
    for b in re.split(r"\\section\*?\{", body)[1:]:
        name = b.split("}")[0]
        if any(x in name for x in TAIL):
            continue
        c = words(b)
        tot += c
        rows.append((c, name))
    return tot, rows


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"  {SRC.name} does not exist")
    raw = SRC.read_text(encoding="utf-8")
    t, kept = mask_comments(raw)
    n_del, n_add, n_rem = (t.count(r"\del{"), t.count(r"\add{"),
                           t.count(r"\rem{"))
    if not (n_del or n_add or n_rem):
        print(f"  {SRC.name} carries no markup; nothing to accept.")
        if "--count" in sys.argv:
            tot, rows = section_counts(
                raw.split(r"\begin{document}")[-1].split(r"\end{document}")[0])
            print("\n  word count by section (journal limit 2,000):")
            for c, name in rows:
                print(f"    {c:5d}  {name}")
            print(f"    -----\n    {tot:5d}  total  ({tot - 2000:+d})")
        return 0
    if dirty(SRC) and "--force" not in sys.argv:
        raise SystemExit(
            f"  {SRC.name} has uncommitted changes. This rewrites the file "
            f"in place, so commit first and the rewrite can be undone with "
            f"git checkout. Pass --force to override.")

    before = raw.split(r"\begin{document}")[-1].split(r"\end{document}")[0]
    t = strip(t, "del", False)
    t = strip(t, "add", True)
    t = strip(t, "rem", False)     # co-author comments never leave the house
    t = unmask_comments(t, kept)
    # The ulem preamble and the macros STAY: this file goes on living and
    # will be marked again next round. Only the uses are gone.
    SRC.write_text(t, encoding="utf-8")
    print(f"  {SRC.name}: accepted {n_add} \\add span(s), dropped {n_del} "
          f"\\del span(s) and {n_rem} \\rem comment(s), in place")

    if "--count" in sys.argv:
        after = t.split(r"\begin{document}")[-1].split(r"\end{document}")[0]
        tot, rows = section_counts(after)
        print("\n  word count by section (journal limit 2,000):")
        for c, name in rows:
            print(f"    {c:5d}  {name}")
        print(f"    -----\n    {tot:5d}  total  ({tot - 2000:+d})")
        was, _ = section_counts(before)
        if was != tot:
            print(f"    (before accepting it counted {was}, over-stating by "
                  f"{was - tot} words of struck text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

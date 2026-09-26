#!/usr/bin/env python3
"""
l64_build_readability_prompt.py: build the cross-vendor readability prompt from
the CURRENT manuscript files, so the reviewer reads what the editor will read.

Why a script: the 10:10 prompts of 26 September were built by hand and went
stale within two hours (lanes 38b and 38c written in, four exhibits moved to
the offline appendix, red markup added to the paper). Rebuilding from the live
files removes that hazard, and the size split keeps every paste under the
~150 KB at which chatgpt.com still turns a paste into an attachment chip
(memory: feedback_chatgpt_crossvendor_chrome).

What it does
  * paper (main_v3.tex): accepts the red markup (\\add kept, \\del dropped) with
    the functions of l20_strip_markup, inlines every \\input{tables/...} (also
    with markup accepted), keeps the body from \\begin{document} to \\end{document};
  * letter (response_v5.tex): body only; whole-line comments removed;
  * online appendix (appendix_v3.tex): body with tables inlined, split into two
    pastes at Part IV (parts I-III, parts IV-VI).
Writes three files in notes/: part1 (instructions + letter + paper), part2a and
part2b (the appendix). Prints the byte size of each.

    python revision/local/l64_build_readability_prompt.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # projects/canaries-sweden
PAPER = ROOT.parent / "canaries-sweden-paper"
NOTES = ROOT / "notes"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l20_strip_markup as l20  # noqa: E402

STAMP = "2026-09-26"
OUT1 = NOTES / f"prompt-chatgpt-readability_{STAMP}_v2_part1.txt"
OUT2A = NOTES / f"prompt-chatgpt-readability_{STAMP}_v2_part2a.txt"
OUT2B = NOTES / f"prompt-chatgpt-readability_{STAMP}_v2_part2b.txt"

INSTRUCTIONS = """CROSS-VENDOR REVIEW, READABILITY ROUND: RESPONSE LETTER, REVISED PAPER AND ONLINE APPENDIX (Economics Letters, EL67898)

This is part 1 of 3. Part 1 holds these instructions, the response letter (Document A) and the revised paper (Document B). Parts 2a and 2b, pasted separately, hold the online appendix (Document C, Parts I to III and Parts IV to VI) with its tables inlined. Please wait for all three parts before answering; if a part is missing, say so and answer from what you have.

CONTEXT
You reviewed the findings of the final register runs on 26 September and the letter on 25 September, and the paper's notation and claims later on 26 September; every precision point you asked for was applied. The corresponding author then read all three documents and judged the paper good but the online appendix "very long, a difficult read, a bit like a lab report and quite technical", not ordered from the broadest result to the finest split, and written as if narrating a revision rather than as a fresh paper's companion; and the response letter "too technical ... not a nice read", harder than the editor's and referees' own comments. Both documents have since been rewritten for register and order with every number frozen (a diff of numeric tokens before and after was checked), the last two register runs (post-fit support of the backtest arms; a month-of-year seasonality check) have been written in, and four exhibits the paper never cited were moved to a non-public replication appendix. Your task is to judge whether the rewrite achieved readability without losing precision or correctness.

YOUR ROLE
Read as the Editor of Economics Letters re-reading a difficult revise-and-resubmit, and as Referee 1. You must never lose patience: if any passage would make you skim, say where and why.

THE MATERIAL
- LaTeX source with tracked changes accepted (the paper's red markup is accepted here; the letter and appendix carry none). In the letter every comment is quoted verbatim between \\bbox and \\ebox; the authors' answer follows. \\pt{X} cross-references point X. \\ref*{OA-xxx} points to \\label{xxx} in Document C.
- Figures are images and not included; judge them from captions.
- Nothing is pending: every register run is in.

TASKS
1. Readability and register. For each document: where does the prose still read as a lab report or an audit trail (narration about sections, hedges repeated, defensive sentences, construction detail in notes, data-science vocabulary)? Quote the passage and give the sentence you would write instead, in the register of a top field-journal paper by an applied economist.
2. Order. Does each document move from the broadest result to the finest split, and does every reader know the big answer before its pieces? Name any place where a finer split precedes the coarser result, or where the letter answers before it acknowledges.
3. Fresh-paper test for the online appendix. Apart from one deliberate sentence in VI.1 ("An earlier version of this appendix gave an overall rate rising to 15 and 20 per cent") and a two-sentence note at the end of IV.3 on how four estimates relate, does anything in Document C read as a comparison with a previous version, a revision log, or a reply to a referee? Quote it.
4. Precision preserved. Check that the rewrite kept, once each and at the right place: the young non-match rise 8.4 to 12.8 to 17.8 per cent with the denominator defined; the pooled rebuild's exact equality and the nine two-sex persons; "about 94 to 96 per cent" of the backtest difference from re-coding with sample inclusion non-zero and the harmonised A/B/C levels stated; the restricted-reference estimates beside their same-sample baseline; "no detectable additional decline" for credit; "no detectable linear drift" for the female pre-period; exposure "concentrated among the most exposed employers" without a threshold claim; the cohort exercise as not isolating the payroll reduction; the incumbent split as selected stock trajectories; the tipping point as a counterfactual on counts and not a bound; the month-of-year check as a movement of at most a fifth of a standard error, with no standard error claimed for the difference; the DAIOE-Eloundou agreement as a statement about occupation rankings, not about the employer classification. Flag any that is missing, doubled, or overstated.
5. Consistency across the three documents. Any number, label, section pointer (the paper cites appendix sections by number, e.g. "Online Appendix III.2") or quoted sentence that disagrees between letter, paper and appendix.
6. Length. The letter is about 6,000 of the authors' own words; the appendix 59 pages; the paper's body is at the 2,995-word cap, so any addition to it must name a cut of equal length. Name the cuts you would make first, with the paragraph and what to keep, and say whether any further exhibit could go to the non-public replication appendix without weakening the editor's test.

OUTPUT
A numbered list, most important first: [document, location] one-sentence finding; short quote; concrete fix. Then two sentences as Editor: would you now read the letter and appendix without losing patience, and does anything in the rewrite weaken the case that the findings are not artefacts of the data's construction and coverage?
"""

RULE = "=" * 66


def body(tex: str) -> str:
    """From \\begin{document} to \\end{document}, inclusive."""
    i = tex.index("\\begin{document}")
    j = tex.index("\\end{document}") + len("\\end{document}")
    return tex[i:j]


def accept(tex: str) -> str:
    """Accept the red markup: \\add{x} -> x, \\del{x} -> nothing; comments kept."""
    masked, kept = l20.mask_comments(tex)
    s = l20.strip(masked, "add", True)
    s = l20.strip(s, "del", False)
    return l20.unmask_comments(s, kept)


def drop_comment_lines(tex: str) -> str:
    return "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))


def inline_tables(tex: str, base: Path) -> str:
    def repl(m):
        rel = m.group(1)
        p = base / (rel if rel.endswith(".tex") else rel + ".tex")
        if not p.exists():
            return m.group(0)
        t = accept(p.read_text(encoding="utf-8"))
        return f"\n% --- inlined {rel} ---\n{drop_comment_lines(t)}\n"
    return re.sub(r"\\input\{([^}]+)\}", repl, tex)


def main() -> int:
    paper = accept((PAPER / "main_v3.tex").read_text(encoding="utf-8"))
    paper = drop_comment_lines(inline_tables(body(paper), PAPER))
    letter = drop_comment_lines(body((PAPER / "response_v5.tex").read_text(encoding="utf-8")))
    oa = (PAPER / "appendix_v3.tex").read_text(encoding="utf-8")
    oa = drop_comment_lines(inline_tables(body(oa), PAPER))
    cut = oa.index("\\apppart{Coverage of the occupation register}")
    oa_a, oa_b = oa[:cut], oa[cut:]

    part1 = (INSTRUCTIONS + "\n\n" + RULE + "\nDOCUMENT A: RESPONSE LETTER (response_v5.tex)\n" + RULE
             + "\n" + letter + "\n\n" + RULE + "\nDOCUMENT B: REVISED PAPER (main_v3.tex, tracked changes accepted, tables inlined)\n"
             + RULE + "\n" + paper + "\n")
    part2a = ("PART 2a of 3 for the readability review requested in part 1.\n\n" + RULE
              + "\nDOCUMENT C: ONLINE APPENDIX (appendix_v3.tex, tables inlined), PARTS I TO III\n" + RULE
              + "\n" + oa_a + "\n\n[continues in part 2b: Parts IV to VI]\n")
    part2b = ("PART 2b of 3 for the readability review requested in part 1: the rest of Document C.\n\n" + RULE
              + "\nDOCUMENT C, continued: ONLINE APPENDIX, PARTS IV TO VI\n" + RULE + "\n" + oa_b
              + "\n\nThis completes the material. Please answer the six tasks of part 1 now.\n")
    for p, s in ((OUT1, part1), (OUT2A, part2a), (OUT2B, part2b)):
        p.write_text(s, encoding="utf-8")
        print(f"  {p.name}: {len(s.encode('utf-8')):,} bytes, {len(s.split()):,} words")
    leftover = [ln for ln in (part1 + part2a + part2b).splitlines() if "\\add{" in ln or "\\del{" in ln]
    print(f"  markup lines left: {len(leftover)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

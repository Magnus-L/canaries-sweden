#!/usr/bin/env python3
"""
l66_build_editor_prompt.py: the editor's read of the revision. The reviewer
is asked to act as the handling editor of Economics Letters who wrote the
decision letter, and to decide on the revised submission: paper, online
appendix and response letter, built from the live files.

Reuses l64's readers (body, accept, inline_tables, drop_comment_lines).
Three pastes under 130 KB each, same split as l65.

    python revision/local/l66_build_editor_prompt.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l64_build_readability_prompt as l64  # noqa: E402

STAMP = "2026-09-26"
OUT1 = l64.NOTES / f"prompt-editor-read_{STAMP}_part1.txt"
OUT2A = l64.NOTES / f"prompt-editor-read_{STAMP}_part2a.txt"
OUT2B = l64.NOTES / f"prompt-editor-read_{STAMP}_part2b.txt"

INSTRUCTIONS = """THE EDITOR'S READ OF A REVISED SUBMISSION (Economics Letters, manuscript EL67898)

This is part 1 of 3. Part 1 holds these instructions, the authors' response letter (Document A) and the revised paper (Document B). Parts 2a and 2b, pasted separately, hold the online appendix (Document C: Parts I to III, then Parts IV to VI) with its tables inlined. Please wait for all three parts before answering.

YOUR ROLE
You are Eric Chyn, the handling editor at Economics Letters for this manuscript. On 2 August 2026 you returned the first submission for revision. Your decision letter raised your own comments and forwarded one referee report; every one of those comments is quoted verbatim in the response letter between \\bbox and \\ebox, with the authors' answer following each. The revised paper and its online appendix are now in front of you. Your task is to decide on the revised submission the way you would in the editorial system: read everything, judge whether each point was resolved as you meant it, judge whether the paper now meets the journal's standard, and write your decision.

Be the editor, not a coach and not an advocate. Do not praise. Treat every claim as something to be checked against the numbers and tables in front of you. Where the authors withdrew a result, judge whether the replacement is sound and whether the paper still has a contribution worth the journal's pages. Where the authors say a concern is answered, check the answer against the evidence they cite (section, table, panel), not against their summary of it. Remember that Economics Letters publishes short papers: a body of about 3,000 words here, five exhibits, and a long online appendix; judge whether the paper stands on its own and whether the appendix is being used to carry what the paper should say.

THE MATERIAL
LaTeX source, tracked changes accepted. In the letter \\pt{X} cross-references point X of the same letter, and \\ref*{OA-xxx} points to \\label{xxx} in Document C. Tables are inlined where they are referenced; figures are images and not included, so judge them from their captions and from the numbers the text gives. The paper's exhibits are Table 1, Figures 1 to 3 and an online appendix of six parts.

WHAT TO DO
1. Comment by comment. For each of your own comments and each referee comment quoted in the letter: is it resolved, partly resolved or not resolved, judged against what you meant when you wrote it and against the evidence in the paper and appendix, not against the letter's wording? Where partly or not, state exactly what is missing and the smallest change that would resolve it.
2. The coverage concern. Your substantive concern was that the register the first submission relied on could not carry the design (the occupation register lags and is incomplete for recent years). The revision withdraws that design, documents what the lag did to it (a backtest), and replaces it with a design on payroll counts that do not use occupation fields, with a coverage audit and an independent reconstruction of the counts. Does this convince you that the findings are not artefacts of how the data are constructed or of their coverage? Where exactly does it fall short, if anywhere, and what would you need to see?
3. The paper as a publishable Economics Letters paper. Contribution relative to the studies it cites; whether the abstract and introduction let a reader know the findings before the pieces; whether claims match their numbers and standard errors; whether the sensitivity of the pooled 22 to 25 estimate to the exposure index (Eloundou et al. 2024 versus the authors' own index) is stated with the right weight, neither buried nor allowed to swallow the paper; whether the female differential and the age profile are supported at the strength claimed; whether the postings evidence and the employment evidence are held apart where they should be; whether the interpretation (AI versus the rate cycle, remote work, the payroll-tax change, pension ages) is handled with the care you would require.
4. Consistency across the three documents: any number, label, exhibit pointer or quoted sentence that disagrees between letter, paper and appendix, and any place where the letter promises something the paper or appendix does not deliver.
5. What you would still require before acceptance (must-fix) and what you would only suggest (optional). Be concrete: document, location, quote, minimal fix.

OUTPUT
(i) Your decision, one of: accept; accept with minor changes; minor revision; major revision; reject, followed by the decision letter you would send the authors (two to four paragraphs, in your voice as editor).
(ii) A numbered list of remaining points, must-fix first, then optional: [document, location] one-sentence issue; short quote; minimal fix.
(iii) A table with one row per original comment (your own and the referee's, using the letter's labels): resolved / partly / not, and five words on why.
(iv) Two sentences on whether the online appendix is carrying anything that belongs in the paper, and two on anything in the paper that you would move out.
"""


def main() -> int:
    paper = l64.accept((l64.PAPER / "main_v3.tex").read_text(encoding="utf-8"))
    paper = l64.drop_comment_lines(l64.inline_tables(l64.body(paper), l64.PAPER))
    letter = l64.drop_comment_lines(l64.body((l64.PAPER / "response_v5.tex").read_text(encoding="utf-8")))
    oa = (l64.PAPER / "appendix_v3.tex").read_text(encoding="utf-8")
    oa = l64.drop_comment_lines(l64.inline_tables(l64.body(oa), l64.PAPER))
    cut = oa.index("\\apppart{Coverage of the occupation register}")
    oa_a, oa_b = oa[:cut], oa[cut:]
    R = l64.RULE
    part1 = (INSTRUCTIONS + "\n\n" + R + "\nDOCUMENT A: RESPONSE LETTER (response_v5.tex)\n" + R + "\n" + letter
             + "\n\n" + R + "\nDOCUMENT B: REVISED PAPER (main_v3.tex, tables inlined)\n" + R + "\n" + paper + "\n")
    part2a = ("PART 2a of 3 for the editor's read requested in part 1.\n\n" + R
              + "\nDOCUMENT C: ONLINE APPENDIX (appendix_v3.tex, tables inlined), PARTS I TO III\n" + R + "\n" + oa_a
              + "\n\n[continues in part 2b: Parts IV to VI]\n")
    part2b = ("PART 2b of 3 for the editor's read requested in part 1: the rest of Document C.\n\n" + R
              + "\nDOCUMENT C, continued: ONLINE APPENDIX, PARTS IV TO VI\n" + R + "\n" + oa_b
              + "\n\nThis completes the material. Please give your decision and the four output blocks of part 1 now.\n")
    for p, s in ((OUT1, part1), (OUT2A, part2a), (OUT2B, part2b)):
        p.write_text(s, encoding="utf-8")
        print(f"  {p.name}: {len(s.encode('utf-8')):,} bytes, {len(s.split()):,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

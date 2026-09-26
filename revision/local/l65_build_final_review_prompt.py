#!/usr/bin/env python3
"""
l65_build_final_review_prompt.py: the final cross-vendor read of paper, online
appendix and response letter before submission, built from the live files.

Reuses l64's readers (body, accept, inline_tables); only the instructions and
the file names differ. Three pastes under 130 KB each.

    python revision/local/l65_build_final_review_prompt.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l64_build_readability_prompt as l64  # noqa: E402

STAMP = "2026-09-26"
OUT1 = l64.NOTES / f"prompt-chatgpt-finalread_{STAMP}_part1.txt"
OUT2A = l64.NOTES / f"prompt-chatgpt-finalread_{STAMP}_part2a.txt"
OUT2B = l64.NOTES / f"prompt-chatgpt-finalread_{STAMP}_part2b.txt"

INSTRUCTIONS = """FINAL READ BEFORE SUBMISSION: PAPER, ONLINE APPENDIX AND RESPONSE LETTER (Economics Letters, EL67898)

This is part 1 of 3. Part 1 holds these instructions, the response letter (Document A) and the revised paper (Document B). Parts 2a and 2b, pasted separately, hold the online appendix (Document C, Parts I to III and Parts IV to VI) with its tables inlined. Please wait for all three parts before answering.

CONTEXT
These are the submission candidates. Every register run is in, all tracked changes are accepted, every number has been checked against the estimation output by a machine-readable manifest. Since your readability round earlier today, the appendix was reordered (Part II opens with its conclusion; III.2 runs from the headline to the finest split; III.5 was shortened), the letter's A2 was cut, and one new result was written in: the headline design re-run with employers classified by the GPT-exposure rating of Eloundou et al. (2024) instead of DAIOE (paper, last results paragraph; OA III.2 last paragraph and Table A25 Panel H; letter C5).

YOUR ROLE
Read as the Editor of Economics Letters and as Referee 1, and also as a leading applied labour economist judging voice. The standard is a top field-journal paper: upright, not defensive, understated, every claim carrying its reference or its number, the reader knowing the answer before the pieces.

THE AUTHOR'S VOICE (measured on 195,000 words of his published prose; the target is his best, not his average)
Sentences average about 24 words with real variation; British English; "we find" for results; "However," is the primary turn, "Therefore," "Thus," "Moreover," secondary; never "But" to open a sentence; never "Crucially", never "It is important to note that"; colons slightly more than semicolons, both used; em-dashes near zero; parentheses ordinary; concreteness before abstraction; the reader's question answered in the order they ask it; a willingness to name what is not known. No bullet lists in prose. Nothing that reads as written by an AI agent: no metaphor, no self-commentary about the analysis, no "screen" or "treatment" vocabulary for diagnostics.

THE MATERIAL
- LaTeX source, tracked changes accepted. In the letter every comment is quoted verbatim between \\bbox and \\ebox; the authors' answer follows. \\pt{X} cross-references point X. \\ref*{OA-xxx} points to \\label{xxx} in Document C. Figures are images and not included; judge them from captions.

TASKS
1. Voice and register. Where does any document slip from the voice above (agentic feel, defensiveness, hedges repeated, narration about sections, data-science vocabulary, a claim stated as stronger or weaker than its number)? Quote the passage and give the sentence you would write, in that voice.
2. Contribution and readability. Does each document make the reader know the answer before its pieces, and does the paper state its contribution at full strength without overstatement? Where does it under-sell, where over-sell?
3. THE ELOUNDOU CROSS-CHECK, a specific question. Facts: employers were re-classified by the Eloundou rating through the same chain as the DAIOE score; 92.5 per cent of DAIOE's top-quartile employers are in the Eloundou top quartile (91.8 per cent by incumbent employment), employer scores correlate at 0.91 (Spearman); on the 103,064 headline employers both indices score, the female differential is -0.071 (SE 0.012) against -0.072 (0.011) on DAIOE, tau at 26-30 is -0.039 (0.007) against -0.039 (0.007), and tau at 22-25 is -0.024 (SE 0.019 by employer, 0.023 by industry) against -0.039 (SE 0.010) on DAIOE: the same sign, about three fifths of the size, the DAIOE estimate within one Eloundou standard error, the Eloundou estimate not distinguishable from zero on its own; the two fits share their sample, so no standard error of the difference exists from separate covariances. The paper's last results paragraph says "Classification by the rating of Eloundou et al. (2024) leaves the female differential and the 26 to 30 estimate unchanged; at 22 to 25 it gives -0.024 (SE 0.019), within one standard error of tau-hat" and then "The 3.9 per cent decline at 22 to 25 is our central estimate." The corresponding author reads this as possibly contradictory: the central estimate loses significance under the other index. Please answer: (a) are our statements statistically correct, and is "within one standard error of tau-hat" the right descriptor, or does it read as special pleading; (b) how should the paragraph in the paper, the last paragraph of OA III.2, and letter C5 be worded so that the overall message (the age gradient inside exposed employers, falling on young women) and the nuance (the pooled 22-25 estimate is the one result whose size and precision depend on the index; its Eloundou classification is a noisier instrument, SE nearly double) are both right, tight and not contradictory; give exact wording for each of the three places, the paper's within the same word count; (c) would a stacked joint fit of the two classifications, which yields a standard error for the difference between the two estimates, materially strengthen the paper before submission, or does the existing evidence (invariant female differential and 26-30 result, top-quartile overlap, DAIOE within one SE) suffice, and is any other framing better (for example reporting the Eloundou 95 per cent interval, which spans about -0.062 to +0.013 and so contains both zero and the DAIOE estimate); (d) does the abstract or the introduction need to change in consequence, and if so how, within their word limits (abstract 100 words, body 2,995).
4. Consistency across the three documents: any number, label, section pointer or quoted sentence that disagrees between letter, paper and appendix.
5. As Editor: does anything in these documents weaken the case that the findings are not artefacts of the data's construction and coverage, and is there anything you would still ask for before accepting.

OUTPUT
A numbered list, most important first: [document, location] one-sentence finding; short quote; concrete fix. Then the Eloundou answer as a separate block, (a) to (d), with the exact wording. Then two sentences as Editor.
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
    part2a = ("PART 2a of 3 for the final read requested in part 1.\n\n" + R
              + "\nDOCUMENT C: ONLINE APPENDIX (appendix_v3.tex, tables inlined), PARTS I TO III\n" + R + "\n" + oa_a
              + "\n\n[continues in part 2b: Parts IV to VI]\n")
    part2b = ("PART 2b of 3 for the final read requested in part 1: the rest of Document C.\n\n" + R
              + "\nDOCUMENT C, continued: ONLINE APPENDIX, PARTS IV TO VI\n" + R + "\n" + oa_b
              + "\n\nThis completes the material. Please answer the five tasks of part 1 now, the Eloundou question as its own block.\n")
    for p, s in ((OUT1, part1), (OUT2A, part2a), (OUT2B, part2b)):
        p.write_text(s, encoding="utf-8")
        print(f"  {p.name}: {len(s.encode('utf-8')):,} bytes, {len(s.split()):,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

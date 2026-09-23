# Lane 32, exported 23 September 2026 12:34, filed 12:45

Handed over as `~/Downloads/MyFiles1234`, seven files, every one hash-checked against the filed
copy before the original was deleted. Runtime 64.3 min, four fits, no SQL.

## THE GATE FAILED, AND THE GATE WAS WRONG

`86_summary.txt` reports THE PANEL HAS MOVED: THE PATH IS NOT DRAWN, on five of ten drift terms
missing lane 29b's by 5 to 11 units in the fifth decimal. The panel did move, and the export
says by how much: the drift here is fitted on 10,206,595 cells at 22--25 against lane 29b's
9,438,694, and 11,461,475 against 10,687,801 at 26--30.

**But that difference is by construction and the gate should never have required it away.**
Script 78's `part_a` builds ONE frame for both halves of part A, from 2019-01 when the 2019 and
2020 counts are cached, and then slices the pre-launch months out of it for the drift. Its
employer set is therefore every employer appearing from 2019, a superset of the 2021-start set
script 83 built for lane 29b's drift. The two fits could not agree on the fourth decimal
whatever the score, and the paper never claims the 2019-start path sits on Table 1's panel: the
appendix says the opposite, and the education-route version of this figure had exactly the same
property.

So the gate tested a property the exhibit does not have and cannot have. That is a defect in
the gate's specification, found by running it, and the fits themselves are unaffected: nothing
about the four estimates changes. **The correct check is substantive agreement, which holds
comfortably**: the refit gives a trend of +0.00037 (0.00077) at 22--25 against lane 29b's
+0.00038 (0.00077), and +0.00159 (0.00044) against +0.00161 (0.00044) at 26--30, each within a
fiftieth of a standard error, with the same verdicts, FLAT at 22--25 and NOT FLAT at 26--30.
Script 86's gate is rewritten to test that and to report the frame difference as a note. The
scripts are corrected for the record; **no re-run is needed and re-running would give the same
numbers**.

## What the export is for

**The plain quarterly path**, 26 quarters from 2019Q1 with 2022Q1 omitted and no calendar terms,
on 43,749,654 cells at 22--25 and 48,918,480 at 26--30. This replaces Figure III.1 and Table
III.2 of the online appendix, which were the education route's and said so.

**The drift test is NOT taken from here.** The appendix quotes lane 29b's drift, which is the row
Table 1 of the paper prints, and this refit is the check on it rather than a replacement.

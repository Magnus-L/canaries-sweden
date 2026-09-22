# Lanes 21 and 22, exported 22 Sep 2026 07:12, filed 07:25

Both lanes were submitted together at about 04:40 (21 read-only, 22 pulling SQL once); 75 ran
31 min, 76 76 min, 77 30 min. Twenty-four files, byte-checked on filing, Downloads copy deleted.

Lane 21 (75): all four fits RECONCILED (window post = 68's rb + post to four decimals).
Level after adoption relative to Jan 2021-Mar 2022, cycle removed: 22-25 stock -0.0194 (0.0184);
26-30 stock -0.0172 (0.0121); 22-25 hires +0.0491 (0.0341); 22-25 separations +0.0868 (0.0243).
Tightening window: +0.0214 (0.0078) / +0.0222 (0.0043) / +0.0523 (0.0178) / +0.0081 (0.0174).

Lane 22 (76): gate PASS (-0.0659 (0.0131) reproduced). Split: within tracks -0.0508 (0.0118),
composition -0.0150, ratio 0.772, verdict HIT HARDER on the pre-committed rule. By track
(women minus men): ict +0.099 (0.067); engineering -0.058 (0.025); business/law/social -0.033
(0.020); health/education/care -0.105 (0.034); other -0.060 (0.020).

Lane 22 (77): gate PASS (-0.0153 (0.0126) reproduced). 22-25 vs 41-49 by track: engineering
-0.027 (0.022); business/law/social -0.069 (0.022)*; health/education/care -0.010 (0.020);
other +0.033 (0.018). ICT FAILED (rc 3221225477 at two threads, 940,788 rows); re-run as lane 23
after the retry ladder gained a one-thread attempt.

## Lane 23, exported 08:07, filed 08:40 (subfolder `lane23-0807/`, seven files, byte-checked)

Script 77 re-run alone after the retry ladder gained the one-thread attempt; the ICT contrast
fitted. Gate PASS again (-0.0153 (0.0126)). **ICT: 22-25 vs 41-49 -0.1167 (0.0501)*, 26-30 vs
41-49 -0.1323 (0.0345)*, 6,921 firms.** All other tracks identical to the 07:12 export. The paper
quotes THIS folder's `contrast_by_track.csv`; the 07:12 one lacks ICT and is superseded.

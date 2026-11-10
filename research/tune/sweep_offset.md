# Rule threshold sweep

Current defaults: yaw ramp 0.35-0.85, head-offset measure off, on threshold 0.6.

| Yaw ramp | Head-offset ramp | On threshold | Model | clip_4 recall | clip_4 precision | clip_4 false alarms, attentive / writing | clip_4 latency (s) | Labels: false alarms on focused | Labels: distracted caught | Lecture % | Exam % | Study hall % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.35-0.85 | off | 0.6 | n | 11% | 100% | 0% / 0% | 3.0 | 8/215 (4%) | 0/12 | 4.7% | 3.6% | 42.6% |
| 0.35-0.85 | off | 0.6 | m | 11% | 100% | 0% / 0% | 2.0 | 14/245 (6%) | 0/13 | 0.0% | 11.7% | 33.7% |
| 0.35-0.85 | off | 0.5 | n | 16% | 100% | 0% / 0% | 11.0 | 10/215 (5%) | 0/12 | 4.8% | 5.2% | 44.3% |
| 0.35-0.85 | off | 0.5 | m | 11% | 100% | 0% / 0% | 2.0 | 15/245 (6%) | 0/13 | 0.0% | 14.3% | 37.9% |
| 0.35-0.85 | 0.1-0.3 | 0.6 | n | 56% | 100% | 0% / 0% | 3.0 | 16/215 (7%) | 2/12 | 4.8% | 17.1% | 52.1% |
| 0.35-0.85 | 0.1-0.3 | 0.6 | m | 56% | 100% | 0% / 0% | 2.0 | 17/245 (7%) | 3/13 | 0.9% | 21.3% | 43.4% |
| 0.35-0.85 | 0.1-0.3 | 0.5 | n | 56% | 100% | 0% / 0% | 3.0 | 19/215 (9%) | 3/12 | 5.2% | 19.8% | 55.8% |
| 0.35-0.85 | 0.1-0.3 | 0.5 | m | 56% | 100% | 0% / 0% | 1.0 | 21/245 (9%) | 6/13 | 1.2% | 26.7% | 49.7% |
| 0.35-0.85 | 0.15-0.35 | 0.6 | n | 45% | 100% | 0% / 0% | 2.0 | 13/215 (6%) | 2/12 | 4.8% | 12.8% | 49.9% |
| 0.35-0.85 | 0.15-0.35 | 0.6 | m | 56% | 100% | 0% / 0% | 3.0 | 16/245 (7%) | 2/13 | 0.0% | 18.2% | 38.3% |
| 0.35-0.85 | 0.15-0.35 | 0.5 | n | 55% | 100% | 0% / 0% | 3.0 | 15/215 (7%) | 2/12 | 4.8% | 14.9% | 52.4% |
| 0.35-0.85 | 0.15-0.35 | 0.5 | m | 56% | 100% | 0% / 0% | 2.0 | 17/245 (7%) | 2/13 | 0.3% | 21.1% | 44.6% |
| 0.35-0.85 | 0.2-0.4 | 0.6 | n | 45% | 100% | 0% / 0% | 2.0 | 11/215 (5%) | 1/12 | 4.7% | 8.4% | 47.5% |
| 0.35-0.85 | 0.2-0.4 | 0.6 | m | 56% | 100% | 0% / 0% | 3.0 | 16/245 (7%) | 2/13 | 0.0% | 14.6% | 37.4% |
| 0.35-0.85 | 0.2-0.4 | 0.5 | n | 45% | 100% | 0% / 0% | 2.0 | 14/215 (7%) | 2/12 | 4.8% | 11.9% | 49.9% |
| 0.35-0.85 | 0.2-0.4 | 0.5 | m | 56% | 100% | 0% / 0% | 3.0 | 16/245 (7%) | 2/13 | 0.0% | 18.1% | 42.3% |
| 0.2-0.5 | off | 0.6 | n | 25% | 100% | 0% / 0% | 3.0 | 12/215 (6%) | 1/12 | 5.1% | 8.5% | 51.1% |
| 0.2-0.5 | off | 0.6 | m | 25% | 100% | 0% / 0% | 3.5 | 16/245 (7%) | 1/13 | 0.7% | 15.8% | 43.8% |
| 0.2-0.5 | off | 0.5 | n | 25% | 100% | 0% / 0% | 2.5 | 16/215 (7%) | 1/12 | 5.2% | 14.4% | 55.7% |
| 0.2-0.5 | off | 0.5 | m | 28% | 100% | 0% / 0% | 5.0 | 18/245 (7%) | 2/13 | 0.7% | 20.4% | 48.9% |
| 0.2-0.5 | 0.1-0.3 | 0.6 | n | 56% | 100% | 0% / 0% | 3.0 | 20/215 (9%) | 2/12 | 5.2% | 20.6% | 59.8% |
| 0.2-0.5 | 0.1-0.3 | 0.6 | m | 56% | 100% | 0% / 0% | 2.0 | 17/245 (7%) | 3/13 | 1.5% | 22.2% | 52.5% |
| 0.2-0.5 | 0.1-0.3 | 0.5 | n | 56% | 100% | 0% / 0% | 2.0 | 27/215 (13%) | 3/12 | 5.7% | 26.4% | 65.4% |
| 0.2-0.5 | 0.1-0.3 | 0.5 | m | 56% | 100% | 0% / 0% | 1.0 | 24/245 (10%) | 6/13 | 1.8% | 28.3% | 57.1% |
| 0.2-0.5 | 0.15-0.35 | 0.6 | n | 45% | 100% | 0% / 0% | 2.0 | 17/215 (8%) | 2/12 | 5.1% | 16.8% | 57.3% |
| 0.2-0.5 | 0.15-0.35 | 0.6 | m | 56% | 100% | 0% / 0% | 3.0 | 16/245 (7%) | 3/13 | 0.7% | 19.7% | 49.2% |
| 0.2-0.5 | 0.15-0.35 | 0.5 | n | 55% | 100% | 0% / 0% | 2.0 | 21/215 (10%) | 2/12 | 5.2% | 21.8% | 62.5% |
| 0.2-0.5 | 0.15-0.35 | 0.5 | m | 56% | 100% | 0% / 0% | 2.0 | 19/245 (8%) | 3/13 | 0.9% | 24.2% | 54.0% |
| 0.2-0.5 | 0.2-0.4 | 0.6 | n | 45% | 100% | 0% / 0% | 2.0 | 14/215 (7%) | 1/12 | 5.1% | 13.8% | 55.4% |
| 0.2-0.5 | 0.2-0.4 | 0.6 | m | 56% | 100% | 0% / 0% | 3.0 | 16/245 (7%) | 2/13 | 0.7% | 18.2% | 46.8% |
| 0.2-0.5 | 0.2-0.4 | 0.5 | n | 45% | 100% | 0% / 0% | 2.0 | 20/215 (9%) | 2/12 | 5.2% | 19.6% | 60.1% |
| 0.2-0.5 | 0.2-0.4 | 0.5 | m | 56% | 100% | 0% / 0% | 3.0 | 19/245 (8%) | 3/13 | 0.7% | 22.2% | 51.7% |

## Reading

The exam criterion is "no more than about 3 points above the model's current-default row". The defaults are 3.6% for n and 11.7% for m, so the caps are about 6.6% (n) and about 14.7% (m). clip_4 recall is optimistic for every row with the offset on, because clip_4 was used to choose the offset ramps. The "distracted caught" column (12-13 label moments) is too small to rank on.

**What the offset measure does.** It is the only thing in the grid that moves clip_4 recall a lot. m goes from 11% to 56% with any offset ramp, and n goes to 45-56%. clip_4 precision stays at 100%, with no attentive or writing false alarms in any row. The cost shows up on the real classes, mainly in the exam class %: every offset row raises it, and the looser the offset ramp, the bigger the rise. Lecture class % barely moves (n 4.7-5.7%, m 0.0-1.8%). Label false alarms on focused moments stay at 5-13%. For m, recall is flat at 56% across all three offset ramps, so the strictest ramp gives up nothing on clip_4.

**n: no setting meets all five.** Exam is the blocker. The yaw-only row 0.35-0.85 / on 0.5 stays within the cap (5.2%), but its recall of 16% is not clearly above 11%. Every row that clearly raises recall puts exam at 8.4% or higher. The closest is:
- 0.35-0.85 / offset 0.2-0.4 / on 0.6: recall 45%, 0% / 0%, labels 11/215 (5%), lecture 4.7%, exam 8.4%. That is +4.8 points, about 2 over the cap.

The best yaw-only row from `sweep_yaw.md`, 0.2-0.5 / on 0.6, is no better on exam (8.5%) and has lower recall (25%).

**m: one setting meets all five.**
- 0.35-0.85 / offset 0.2-0.4 / on 0.6: recall 56%, 0% / 0%, labels 16/245 (7%), lecture 0.0%, exam 14.6%. That is +2.9 points, right at the cap.

Every other m row that raises recall puts exam at 15.8% or higher.

**Study hall** (desk-height camera, not used to decide): at the pick it rises from 42.6% to 47.5% (n) and from 33.7% to 37.4% (m). That is a similar rise to the exam clip's, and it goes in the expected direction for a camera where heads often look turned.

**Pick: yaw ramp 0.35-0.85 (unchanged), head-offset cue on with ramp 0.2-0.4, on threshold 0.6 (unchanged), for both models.** It changes one thing, switching the offset cue on at its strictest tested ramp, and it is the most conservative row on exam for both models. Trade-off: on clip_4, recall rises from 11% to 45-56% (optimistic, since clip_4 tuned the ramp) with no new staged false alarms. The price is about 3 points (m) to 5 points (n) more class distracted % on the exam clip, where the truth is "low". That puts n slightly outside the tolerance.

Before making it the default:
- Confirm it on the held-out 9-person session.
- Because recall did not drop at the strictest ramp, a stricter one (for example 0.25-0.45) may recover some of n's exam margin, and is worth one more run.

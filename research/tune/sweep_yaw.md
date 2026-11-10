# Rule threshold sweep

Current defaults: head turn ramp 0.35-0.85, on threshold 0.6.

| Head turn ramp | On threshold | Model | clip_4 recall | clip_4 precision | clip_4 false alarms, attentive / writing | clip_4 latency (s) | Labels: false alarms on focused | Labels: distracted caught | Lecture % | Exam % | Study hall % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.35-0.85 | 0.6 | n | 11% | 100% | 0% / 0% | 3.0 | 8/215 (4%) | 0/12 | 4.7% | 3.6% | 42.6% |
| 0.35-0.85 | 0.6 | m | 11% | 100% | 0% / 0% | 2.0 | 14/245 (6%) | 0/13 | 0.0% | 11.7% | 33.7% |
| 0.35-0.85 | 0.5 | n | 16% | 100% | 0% / 0% | 11.0 | 10/215 (5%) | 0/12 | 4.8% | 5.2% | 44.3% |
| 0.35-0.85 | 0.5 | m | 11% | 100% | 0% / 0% | 2.0 | 15/245 (6%) | 0/13 | 0.0% | 14.3% | 37.9% |
| 0.35-0.85 | 0.4 | n | 17% | 90% | 0% / 4% | 10.0 | 13/215 (6%) | 1/12 | 4.8% | 9.8% | 48.9% |
| 0.35-0.85 | 0.4 | m | 18% | 100% | 0% / 0% | 1.0 | 22/245 (9%) | 1/13 | 0.0% | 21.3% | 46.5% |
| 0.25-0.65 | 0.6 | n | 20% | 100% | 0% / 0% | 8.0 | 10/215 (5%) | 1/12 | 4.7% | 5.6% | 45.8% |
| 0.25-0.65 | 0.6 | m | 11% | 100% | 0% / 0% | 2.0 | 15/245 (6%) | 1/13 | 0.7% | 13.0% | 37.6% |
| 0.25-0.65 | 0.5 | n | 25% | 100% | 0% / 0% | 3.5 | 12/215 (6%) | 1/12 | 4.8% | 8.1% | 48.2% |
| 0.25-0.65 | 0.5 | m | 24% | 100% | 0% / 0% | 4.0 | 17/245 (7%) | 1/13 | 0.7% | 16.7% | 43.8% |
| 0.25-0.65 | 0.4 | n | 25% | 93% | 0% / 4% | 2.5 | 16/215 (7%) | 1/12 | 4.8% | 15.4% | 53.4% |
| 0.25-0.65 | 0.4 | m | 32% | 100% | 0% / 0% | 2.0 | 25/245 (10%) | 3/13 | 0.7% | 28.2% | 50.2% |
| 0.2-0.5 | 0.6 | n | 25% | 100% | 0% / 0% | 3.0 | 12/215 (6%) | 1/12 | 5.1% | 8.5% | 51.1% |
| 0.2-0.5 | 0.6 | m | 25% | 100% | 0% / 0% | 3.5 | 16/245 (7%) | 1/13 | 0.7% | 15.8% | 43.8% |
| 0.2-0.5 | 0.5 | n | 25% | 100% | 0% / 0% | 2.5 | 16/215 (7%) | 1/12 | 5.2% | 14.4% | 55.7% |
| 0.2-0.5 | 0.5 | m | 28% | 100% | 0% / 0% | 5.0 | 18/245 (7%) | 2/13 | 0.7% | 20.4% | 48.9% |
| 0.2-0.5 | 0.4 | n | 25% | 87% | 0% / 8% | 2.5 | 23/215 (11%) | 3/12 | 6.1% | 22.9% | 60.3% |
| 0.2-0.5 | 0.4 | m | 32% | 100% | 0% / 0% | 2.0 | 26/245 (11%) | 4/13 | 2.1% | 31.8% | 55.5% |
| 0.15-0.4 | 0.6 | n | 25% | 100% | 0% / 0% | 3.0 | 18/215 (8%) | 4/12 | 5.9% | 13.4% | 58.0% |
| 0.15-0.4 | 0.6 | m | 28% | 100% | 0% / 0% | 4.0 | 19/245 (8%) | 3/13 | 2.8% | 18.7% | 49.8% |
| 0.15-0.4 | 0.5 | n | 25% | 87% | 0% / 8% | 2.0 | 25/215 (12%) | 4/12 | 7.4% | 20.9% | 63.3% |
| 0.15-0.4 | 0.5 | m | 32% | 100% | 0% / 0% | 4.0 | 27/245 (11%) | 5/13 | 4.0% | 27.5% | 55.9% |
| 0.15-0.4 | 0.4 | n | 25% | 87% | 0% / 8% | 2.0 | 38/215 (18%) | 5/12 | 10.3% | 30.8% | 67.8% |
| 0.15-0.4 | 0.4 | m | 32% | 100% | 0% / 0% | 2.0 | 40/245 (16%) | 6/13 | 5.4% | 40.6% | 62.3% |
| 0.1-0.3 | 0.6 | n | 25% | 77% | 0% / 15% | 1.0 | 33/215 (15%) | 4/12 | 10.7% | 26.0% | 67.0% |
| 0.1-0.3 | 0.6 | m | 33% | 84% | 0% / 13% | 3.5 | 29/245 (12%) | 4/13 | 8.1% | 28.4% | 58.8% |
| 0.1-0.3 | 0.5 | n | 36% | 72% | 12% / 18% | 2.0 | 48/215 (22%) | 6/12 | 15.1% | 35.1% | 70.8% |
| 0.1-0.3 | 0.5 | m | 38% | 84% | 0% / 15% | 2.5 | 36/245 (15%) | 6/13 | 10.5% | 41.8% | 64.1% |
| 0.1-0.3 | 0.4 | n | 49% | 67% | 15% / 36% | 2.0 | 62/215 (29%) | 10/12 | 17.2% | 48.7% | 74.9% |
| 0.1-0.3 | 0.4 | m | 41% | 84% | 0% / 17% | 2.0 | 49/245 (20%) | 6/13 | 13.3% | 56.1% | 69.4% |

## Reading

clip_4 recall is per seat-second inside the 7 scripted events (8 seat-events, two seats), so it moves in steps tied to a few events. It plateaus at 25% for n across most of the grid. That suggests the extra seconds come from the head-turn events (look to the side, turn around); the phone and talking events still go mostly undetected at any threshold. The label "distracted caught" column (12-13 moments) is too small to rank on. Study hall (desk-height camera) gets worse with every loosening, from 34-43% at the defaults to 44-51% at the candidates below. It is a known weak case and is not used to decide.

**n: three settings meet all five criteria.**
- 0.25-0.65 / on 0.6: recall 20%, false alarms 0% / 0%, labels 5%, lecture 4.7%, exam 5.6%. The smallest gain, and the latency is 8 s.
- 0.25-0.65 / on 0.5: recall 25%, 0% / 0%, labels 6%, lecture 4.8%, exam 8.1%, latency 3.5 s.
- 0.2-0.5 / on 0.6: recall 25%, 0% / 0%, labels 6%, lecture 5.1%, exam 8.5%, latency 3.0 s.

Settings looser than these push exam above 10% (13-49%). The writing false alarms first appear at on 0.4, or anywhere with the 0.1-0.3 ramp.

**m: no setting meets all five.** The exam clip is the blocker, and the m defaults already sit at 11.7%. Every setting that raises clip_4 recall above 11% puts exam at 15.8% or higher. The closest are:
- 0.2-0.5 / on 0.6: recall 25%, 0% / 0%, labels 7%, lecture 0.7%, exam 15.8%.
- 0.25-0.65 / on 0.5: recall 24%, 0% / 0%, labels 7%, lecture 0.7%, exam 16.7%.

Both pass every other criterion with margin. The exam excess is about 4-5 points above the current default.

**Pick: head turn ramp 0.2-0.5, on threshold 0.6 (off 0.4), for both models.** It ties with 0.25-0.65 / 0.5 within what this data can resolve: the same 25% recall, one label false alarm apart, and 0.4 points apart on exam. I prefer it for two reasons:
- It changes one knob only. The on/off hysteresis stays as it is, which is easier to defend and attribute.
- It is the better of the two on m's binding criterion (exam 15.8% vs 16.7%) and on latency for both models.

For n it meets every criterion, with lecture at 5.1%, just on the line. For m it more than doubles clip_4 recall with no clip_4 or label false-alarm cost, but exam rises from 11.7% to 15.8%. That trade-off should be stated in the thesis rather than hidden. The 9-person staged session remains the held-out test, and the pick should be confirmed there before it becomes the default.

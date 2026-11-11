# clip_5: 10-student class on the Pi 5 + Hailo-8, with human labels

A held-out check: the head-turn thresholds and the learned classifiers were tuned on other clips, and nothing was tuned on clip_5. Repo at `bf7b7a7`, INT8 HEFs `upper_body_n.hef` and `upper_body_m.hef`, run with `research/clips/run_clips.sh OUT clip_5_720p.mp4:center`.

The clip shows minors. The video, frames, skeleton files, analysis outputs and the labelling page stay off the repo; only this report and the labels (`research/labels/student_labels_clip5.json`: clip, time, box, seat, stratum, label; no images) are committed.

## Clip

- **Content:** 14 min, 10 secondary-school students in rows of 3/4/3 (3 at the back, 4 in the middle, 3 at the front). Loosely scripted, not directed.
- **Camera:** iPhone 16 Pro, high up at the front, centred.
- **Source (`ffprobe`):**
  - HEVC Main 10, 1920×1080 landscape, no rotation or display matrix
  - 59.94 fps nominal and average, 50,401 frames, 840.8 s
  - HDR: HLG (`arib-std-b67`), BT.2020, `yuv420p10le`, with Dolby Vision configuration side data
  - 1.48 GB
- **Conversion:** tone-mapped to SDR, 1280×720, 30 fps, x264 CRF 20, no audio. The result is 25,225 frames, 840.8 s, 73 MB, and a check frame looked upright with natural colours.
  ```
  ffmpeg -i clip_5.mov -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1280:720,fps=30" -c:v libx264 -preset veryfast -crf 20 -an clip_5_720p.mp4
  ```

## Runs

| Model | Tiles | FPS | Students / frame | Duplicates / frame |
|---|---|---|---|---|
| n | off | 131.4 | 10.4 | 0.02 |
| n | on | 32.9 | 10.3 | 0.02 |
| m | off | 31.6 | 10.3 | 0.02 |
| m | on | 7.9 | 10.2 | 0.01 |

Class indicator after 2:30:
- **Class share:** the mean class distracted %, with the mean 95% range in brackets.
- **Coverage:** students with a focused or distracted decision per frame, divided by 10. It can go over 100%, because some frames have 11–13 people (see below).

| Model | Tiles | Class size (real: 10) | Coverage | Rules | Trees | Logistic | MLP |
|---|---|---|---|---|---|---|---|
| n | off | 21 | 62% | 34.5% [15–62] | 30.6% [13–58] | 64.9% [37–85] | 44.4% [22–70] |
| n | on | 17 | 97% | 10.9% [4–35] | 18.5% [7–43] | 42.8% [22–66] | 27.0% [11–52] |
| m | off | 14 | 87% | 26.8% [11–55] | 12.7% [4–40] | 28.2% [11–56] | 23.8% [9–51] |
| m | on | 13 | 102% | 15.2% [5–43] | 9.3% [3–36] | 28.5% [11–56] | 13.1% [4–40] |

Share of time after 2:30 that the class indicator was above 30%:

| Model | Tiles | Rules | Trees | Logistic | MLP |
|---|---|---|---|---|---|
| n | off | 55.4% | 49.9% | 100% | 78.5% |
| n | on | 7.6% | 13.4% | 84.8% | 37.6% |
| m | off | 38.3% | 4.3% | 53.2% | 27.2% |
| m | on | 12.0% | 1.0% | 45.0% | 5.2% |

## Odd things

- **Class size is 13–21 against a real 10.** There are only about 10 people per frame, so the extra seats come from people being lost and given new seats. For example, n without tiles ends up with 23 seats that each last more than 10 s.
- **The front-row middle student's box switches between two shapes** (seat 15 for m with tiles, seat 18 for m without).
  - The two shapes are an upper-body box about 118 px tall and a tall box about 327 px tall that reaches the legs and bare feet under the desk.
  - Nose and shoulders stay in the same place, so it is one person, not two.
  - The box switches 366 times in the m-with-tiles run.
  - With the tall box the rules call the student distracted 68% of the time, mostly `body_turned`. With the upper-body box it is 32%.
  - A second seat (8) often appears on the same student. That explains most frames with more than 10 people, and it adds to both the class size and the class share.

## Human labels

150 moments were labelled blind on a page built from m with tiles, sampled by stratum (rules vs trees). There are 121 focused, 28 distracted and 1 can't tell. Distracted moments make up 18.8% of the labels as sampled, or 6.8% re-weighted to the whole clip.

Scored with the same matching as `pooled_summary.py`: nearest frame within 0.25 s and box IoU ≥ 0.3. A moment with no match, or matched to "unknown", counts as not flagged. The boxes come from m with tiles, so the other runs leave moments unmatched. With 92 of 149 unmatched, the n-without-tiles rows say little.

| Model | Tiles | Decision | Not matched (of 149) | False alarms on focused | Specificity (95% CI) | Distracted caught | Flagged share, re-weighted |
|---|---|---|---|---|---|---|---|
| n | off | Rules | 92 | 7 / 121 | 94% (89–97%) | 3 / 28 | 5.1% |
| n | off | Trees | 92 | 9 / 121 | 93% (86–96%) | 1 / 28 | 13.7% |
| n | off | Logistic | 92 | 15 / 121 | 88% (81–92%) | 7 / 28 | 24.0% |
| n | off | MLP | 92 | 15 / 121 | 88% (81–92%) | 1 / 28 | 18.2% |
| n | on | Rules | 18 | 11 / 121 | 91% (84–95%) | 8 / 28 | 4.7% |
| n | on | Trees | 18 | 35 / 121 | 71% (62–78%) | 3 / 28 | 17.7% |
| n | on | Logistic | 18 | 44 / 121 | 64% (55–72%) | 11 / 28 | 33.9% |
| n | on | MLP | 18 | 42 / 121 | 65% (56–73%) | 9 / 28 | 28.4% |
| m | off | Rules | 34 | 43 / 121 | 64% (56–72%) | 11 / 28 | 15.3% |
| m | off | Trees | 34 | 21 / 121 | 83% (75–88%) | 2 / 28 | 12.6% |
| m | off | Logistic | 34 | 38 / 121 | 69% (60–76%) | 7 / 28 | 13.9% |
| m | off | MLP | 34 | 30 / 121 | 75% (67–82%) | 6 / 28 | 25.7% |
| m | on | Rules | 0 | 43 / 121 | 64% (56–72%) | 21 / 28 | 10.1% |
| m | on | Trees | 0 | 51 / 121 | 58% (49–66%) | 12 / 28 | 11.4% |
| m | on | Logistic | 0 | 55 / 121 | 55% (46–63%) | 10 / 28 | 35.4% |
| m | on | MLP | 0 | 37 / 121 | 69% (61–77%) | 9 / 28 | 20.0% |

The m-with-tiles rules' 43 false alarms come from `body_turned` 12, `face_hidden` 11, `hand_at_face` 11, `head_down` 6 and `head_turned` 3.

**The front-row middle student:** 15 of the 149 moments (10%) are on this student, matched to seat 15 or seat 8. Ten have the tall box and five the upper-body box. The labels are 9 focused and 6 distracted.

The m-with-tiles rules decided as follows on those 15 moments:
- **Distracted moments:** 6 of 6 flagged: 5 `hand_at_face`, 1 `head_turned`.
- **Focused moments:** 6 of 9 flagged, all with the tall box. Five were `body_turned` and one was `hand_at_face`.

Without this student the rules' false alarms are 37 / 112 and they catch 15 / 22.

## Reading

- **The rules on n with tiles held up on this new clip.** Specificity is 91% (84–95%), against 90% on `student_labels_v1` (FP32 n with tiles). They catch 8 of 28 distracted moments and flag 4.7% of the clip re-weighted, close to the labelled 6.8%.
- **The learned models false-alarm less than on v1, but still far more than the rules.** On n with tiles their specificity is 64–71%, against 24–56% on v1. On m with tiles it is 55–69%. None beats the rules at the same setting, except trees on m without tiles (83% vs 64%), and that comes from flagging less (2 of 28 caught).
- **On m the rules do much worse than on n: 64% specificity, with or without tiles.** The m with tiles rules do catch the most (21 / 28). The front-middle student's tall box accounts for 6 of the 43 false alarms. The remaining false alarms are spread over `body_turned`, `face_hidden` and `hand_at_face`, so an m-specific calibration issue is likely too, not only the box switching.
- **The class indicator reads high.** After 2:30 it averages 11–15% for the rules with tiles, against a labelled 6.8% for the whole clip. The inflated class size (13–21 seats for 10 students) and the doubled front-middle student both push it up.
- **Without tiles, n misses students.** Coverage is 62%, and 92 of 149 labelled moments go unmatched, so its high specificity there is mostly moments it never scored.
- **Caveat:** 28 distracted moments is too few for precise recall, and the moments were sampled from m with tiles, which favours that run in matching.

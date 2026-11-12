# 5 Conclusion and future work

## 5.1 Answers to the research questions

**RQ1, feasibility at the edge: yes.** On a Raspberry Pi 5 with a Hailo-8, the deployment setting (n model with tiles) ran 15 minutes in real time:

- 30 FPS with 45 ms latency
- 4.3 W measured on the board, plus about 2.5 W for the accelerator
- peak SoC temperature 60 °C, no throttling

The pipeline is limited by the accelerator, not the CPU. The m model with tiles reaches only 7.9 FPS, enough for analysis but not for a live camera without dropping frames.

**RQ2, coverage: substantially improvable without retraining.** Tiling the back of the room raised the share of SCB students found from 49% to 76%, and in the far third from 17% to 74%. It doubled the students found in a dense study hall. Removing same-skeleton duplicates and anchoring seats at the head made the inferred class size match the real one (52 for about 50 students, 31 for about 30).

In a small room filmed at close range (clip\_5), seats still multiplied, to 13-17 for 10 students, so the class-size estimate holds for the camera distances it was developed on, not yet for very close cameras.

**RQ3, the per-student decision: calibration beats benchmark accuracy.** Classifiers trained on SCB rank SCB actions well (within-image AUC 0.84) but transfer badly:

- **Lecture clip:** they call 48-89% of an attentive lecture distracted.
- **Human labels:** they wrongly flag 114-199 of 262 focused moments.
- **Cause:** SCB has no "listening" label, so they learn that a raised head means distracted.

The calibrated rules have very few false alarms (specificity 90% on human labels; 0-5% on the lecture). They first missed most real distraction (11% on the staged session), because the ear-based head yaw collapses in profile. A second, body-relative head-turn measure raised that to 45-56%, with no new false alarms there and a small cost on real classes. On a 10-student class labelled after all thresholds were fixed (clip\_5), the rules on the deployment setting kept 91% specificity. Phone use remains invisible to pose.

**RQ4, the class indicator: yes, with honest width.** Seat memory, zone weighting and a Wilson interval around the shown value give a share that always lies inside its range. Without the real head count the range is wide (25-37 points for 20-30 students), which reflects how little a sample that small can say. Giving `--class-size` narrows it.

## 5.2 Future work

1. **A held-out test at classroom scale.** Run the 9-person scripted session (research/staged/script\_v2.csv). clip\_5 confirmed the rules' specificity on a class nothing was tuned on, but its recall still rests on one two-person session.
2. **Classroom-specific training labels.** The labelling and staged-session tools can grow a dataset that includes attentive, head-up students, the case SCB lacks. A learned decision trained on it, with per-seat baselines as features, could combine the classifiers' sensitivity with the rules' specificity.
3. **Phone use.** Detecting phones needs an object detector. On the Hailo-8 it could run on the same frames, at a cost in frame rate. Only a per-seat "phone visible" flag would leave the device.
4. **Longer field deployments.** Weeks of use would show drift in calibration, the effect of lighting and seating changes, and whether teachers find the indicator useful. That needs a study with consent, designed with the school.
5. **Live m with tiles.** A leaky queue or a lower camera rate would let the most accurate setting run live without growing latency.
6. **Keypoint-based geometry.** Anchor seats on the shoulders and measure torso rotation against the visible upper body instead of the box. That removes the two failures seen at close range in clip\_5: seats multiplying and a full-body box reading as a turned torso.
7. **Teacher-facing evaluation.** The real test of a class-level indicator is whether it helps teaching, which needs a user study rather than more accuracy metrics.

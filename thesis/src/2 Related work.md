# 2 Related work

## 2.1 Approaches to classroom engagement detection

Automatic engagement detection in classrooms has used four kinds of signal: faces, body pose, body orientation and wearable sensors. They trade accuracy against intrusiveness.

**Faces.** Facial expressions and gaze carry rich information, but they need a frontal, well-resolved face for each student. That fails in large classes, where back rows are small and often occluded \[1\]. Face analysis also identifies the student, which is the root of the privacy problems in section 2.3.

**Body pose.** Skeleton keypoints are less identifying and survive low resolution better.

- Tarigan et al. \[2\] detect students with YOLOv8m, extract seven keypoints per person with MediaPipe, and classify them with a small fully connected network into engaged, bored, raising a hand and on the phone.
  - **Data:** about 10,000 frames recorded with a laptop webcam in a small lab setting.
  - **Result:** 0.70 accuracy, against 0.41 for a YOLOv4 and OpenPose baseline.
  - **Weaknesses they report:** phone use was mostly confused with bored or engaged, and overlapping boxes caused double detections.
  - **Gaps:** privacy and deployment hardware are not discussed.
- Public datasets such as SCB \[3\] label classroom actions (reading, writing, raising a hand, turning the head, discussing) rather than attention, and are the usual training source.

**Body orientation.** Chen et al. \[1\] estimate each student's body orientation jointly with detection, using a one-stage YOLOv11-based model (JointBDOE). They read engagement from how often orientation changes and whether it follows the teacher.

- **Orientation accuracy:** 10.6° mean error on MEBOW.
- **Real classrooms:** ten lessons from kindergarten to university, with faces blurred.
- **Gaps:** no per-student engagement labels and no quantitative engagement accuracy. The models were trained on six RTX 4090 GPUs, and inference hardware is not stated.

**Wearables.** EEG headbands such as BrainCo's FocusEDU measure brain activity directly, but they require every student to wear a device and drew strong objections (section 2.3).

**Review.** The same group's systematic review \[4\] followed PRISMA 2020. It searched Scopus for 2019–2023 and narrowed 547 papers to 12 studies. Its main findings:

- **Methods:** YOLO detectors were the most used (5 of 12 studies), then VGG classifiers (4). Hybrids that split detection from pose did best among skeleton methods: YOLOv4 with OpenPose reached 86.1% accuracy on four classes.
- **Pose alone:** the pose-only study scored lowest. It found phone use with precision 0.22 and recall 0.11, the behaviour that the 13 keypoints here also cannot separate from writing (section 4).
- **Challenges:** costly manual annotation, class imbalance, small and occluded students in crowded rooms, and compute cost. The review points to lightweight YOLOv8 models and notes that keypoints help preserve privacy.
- **Comparability:** the studies use different classes and metrics, so their results cannot be ranked directly.

This thesis takes up the review's two open directions: a lightweight YOLOv8-pose model on edge hardware, and keypoints as a privacy measure.

Two gaps run through this work:

- Accuracy is reported on lab or action datasets rather than on attention in real lessons.
- Privacy and the computing platform are rarely part of the design.

## 2.2 Pose estimation on edge hardware

Neural accelerators have made multi-person pose estimation feasible on small boards. The Hailo-8 delivers 26 TOPS at a typical 2.5 W \[5\]. As the Raspberry Pi AI HAT+, it attaches to a Raspberry Pi 5 over PCIe. Hailo's example applications run a YOLOv8-pose model in a GStreamer pipeline and hand each frame's detections to a Python callback.

Core Electronics' guide \[6\] builds on that pipeline to read 17 COCO keypoints and drive single-person demos, such as gesture control and a game. It reports no frame rates. It also notes practical limits:

- some capture resolutions caused crashes
- a start-up delay
- extended use may need cooling

These examples target one stock COCO model and one or a few people near the camera. A classroom differs in four ways:

- a custom model (13 upper-body keypoints)
- 20-50 people at very different scales
- a camera far from the back rows
- the need to keep the full pipeline, including analysis, on the device

Section 3 describes the runtime built for that case.

## 2.3 Deployed systems and the controversies they raised

The best-documented deployments of attention monitoring in schools are from China in 2017-2019. All three below were criticised for exactly the properties this thesis avoids: identifying students, scoring them individually, and doing so without consent.

**Hangzhou No. 11 High School, 2018. \[7\]–\[10\]**

- **Vendor:** Hikvision's "smart classroom behavioural management system".
- **How it worked:** cameras above the blackboard scanned the class about every 30 seconds. They recognised each student and classified six behaviours (reading, writing, listening, standing, raising a hand, lying on the desk) and seven facial expressions.
- **Who saw the scores:** they appeared on classroom screens, with attention rankings in the hallway.
- **Data:** the school said data stayed on a local server and no images were kept.
- **Response:** educators objected to the lack of consent and questioned whether expressions say anything about learning. After the media coverage in May 2018, the school stopped scoring facial expressions.

**Hanwang Education's Class Care System, 2017-2019. \[10\]**

- **How it worked:** a dome camera photographed the class once per second. The footage went to a server, where deep learning identified each face and sorted behaviour into listening, answering, writing, interacting and sleeping.
- **Who saw the scores:** each student received a weekly score in an app for teachers and parents.
- **Rollout:** piloted in seven schools without Ministry of Education approval.
- **Students' response:** at one school students were not told about it. The company's own manager said students "hate it", and students at another school unplugged the cameras before exams.

**BrainCo FocusEDU headbands, Jinhua, 2019. \[11\]**

- **How it worked:** a primary school had pupils wear brainwave-sensing headbands during training sessions. According to the vendor, teachers saw only the class average.
- **Response:** after a related Weibo topic passed 220 million views, Zhejiang education officials ordered the school to suspend their use.

Two lessons carry into this work:

- Identity and individual scores are what made these systems objectionable, more than the camera itself.
- Centralising footage on servers widens who can see it and what it can be reused for.

At that time China had no specific law on facial or biometric data. Vietnam now does, through its Personal Data Protection Law (2025) \[12\], discussed in section 4.

## 2.4 The gap this thesis addresses

No prior system combines the four properties this work targets:

- sensing without identity
- processing entirely on a device in the room
- an aggregate rather than per-student output
- an evaluation of attention in real classrooms, including on the deployed hardware

| Work | Sensing | Where it runs | Output for the teacher | Identifies students | Evaluated on real lessons | Measured on edge hardware |
| --- | --- | --- | --- | --- | --- | --- |
| Hangzhou No. 11 / Hikvision (2018) | faces, behaviour | local server | per-student scores, rankings | yes | not published | no |
| Hanwang Class Care System | faces, behaviour | server | weekly per-student score | yes | not published | no |
| BrainCo FocusEDU (2019) | EEG headband | device + app | class average (vendor) | per device | not published | – |
| Tarigan et al. \[2\] | 7 keypoints | not stated | per-person class | no | no (lab webcam) | no |
| Chen et al. \[1\] | body orientation | GPU (not stated) | orientation patterns | no | qualitative only | no |
| Core Electronics guide \[6\] | 17 keypoints | Pi 5 + Hailo-8 | demo only | no | no | partly |
| **This thesis** | 13 upper-body keypoints | Pi 5 + Hailo-8, offline | class share + 95% range | no (anonymous seats) | yes: clips, human labels, staged sessions | yes: FPS, latency, temperature, power |

# 1 Introduction

## 1.1 Motivation

A teacher facing 30-50 students cannot see every student's attention at every moment, and a simple class-level indicator of how many students seem distracted could help them pace a lesson. Camera-based systems that do this already exist, but the best known of them, in Chinese schools in 2018-2019, recognised every student's face, sent footage to servers, and scored individual students for teachers and parents. They drew public backlash and one was suspended by the authorities (chapter 2).

The question this thesis asks is whether the useful part, an aggregate attention indicator, can be built without the surveillance. Three design choices follow from that:

- **Pose instead of faces.** Only body keypoints (13 upper-body points) are extracted; no face is recognised or stored, and students are only anonymous seat numbers within one session.
- **Edge instead of cloud.** Everything runs on a Raspberry Pi 5 with a Hailo-8 accelerator in the room. No frame leaves the device, and by default none is stored.
- **Class level instead of individual scores.** The output meant for the teacher is the share of the class that appears distracted, with an honest uncertainty range, not a per-student report.

The remaining problems are technical: lightweight pose models miss small students in the back rows, and "looking away" is hard to judge from a skeleton across different classrooms and camera positions.

## 1.2 Research questions

1. **RQ1, feasibility at the edge.** Can a pose-based engagement pipeline run in real time on a Raspberry Pi 5 with a Hailo-8 for a full classroom, and at what power and thermal cost?
2. **RQ2, coverage.** How many students does a lightweight pose model find in a real classroom, and can pre- and post-processing raise that without retraining the model?
3. **RQ3, the per-student decision.** How well do pose-based decisions, a calibrated rule set and classifiers trained on a public classroom dataset, separate focused from distracted students in real classrooms?
4. **RQ4, the class indicator.** Can a class-level share with a trustworthy uncertainty range be built when many students are missed?

## 1.3 Contributions

- **hcpose**, an open-source runtime for custom YOLOv8-pose models on the Hailo-8, with a GStreamer engine and a Python engine that give the same results, installable and runnable with one command.
- **Tiling for far rows:** extra inferences on zoomed tiles of the back of the room, merged without duplicates in C++ inside the GStreamer pipeline. On SCB images it raised the share of students found from 49% to 76%, and from 17% to 74% in the far third, without changing the model.
- **Duplicate-skeleton removal:** a post-processing step that removes the second box YOLO-pose often gives one student.
- **An engagement analyzer** with per-seat calibration and two head-turn measures. The second measure, the nose's offset from the shoulder centre, keeps working in profile, where the usual ear-based estimate fails.
- **A class estimate** that remembers seats, weights near, middle and far zones, and reports a Wilson interval, with seat counting that resists spurious detections.
- **An evaluation on four kinds of evidence:**
  - the SCB dataset
  - four real classroom clips (three public, one recorded for this work)
  - 451 blind human labels from three clips
  - a scripted staged session

  Together they show that classifiers trained on SCB do not transfer to real lectures, while per-student calibration does.
- **Edge measurements** on the device: throughput, latency, CPU, temperature and board power.
- **A privacy analysis** against Vietnam's Personal Data Protection Law (2025) and Decree 356/2025.

## 1.4 Scope

The system estimates visible attention from posture and head direction; it does not measure learning, emotion or cognitive engagement, and it cannot see a phone held like a pen. The pose models were trained before this work and kept fixed: every improvement is in the runtime, the post-processing and the analysis. The evaluation uses short recordings and staged sessions rather than a long field deployment.

## 1.5 Structure

Chapter 2 reviews engagement detection, pose estimation at the edge and the deployed systems that raised privacy concerns. Chapter 3 describes the system. Chapter 4 evaluates it and analyses its privacy properties and legal fit. Chapter 5 answers the research questions and lists future work.

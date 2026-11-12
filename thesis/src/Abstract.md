# Abstract

Camera-based attention monitoring in schools has drawn strong objections where it recognised students' faces, sent footage to servers and scored individual students. This thesis asks whether the useful part, a class-level indicator of how many students appear distracted, can be built without that surveillance.

The system, hcpose, runs entirely on a Raspberry Pi 5 with a Hailo-8 accelerator. A YOLOv8-pose model reduced to 13 upper-body keypoints finds the students. No face is recognised, frames stay in memory, and students exist only as anonymous seats. Per-seat calibration, two head-turn measures and smoothed rules decide whether each seat looks focused. The teacher sees only the share of the class that appears distracted, with a 95% Wilson interval.

On the device, the deployment setting (the n model, with tiling of the far rows) runs in real time at 30 FPS with 45 ms latency. It draws about 4.3 W at the board plus about 2.5 W for the accelerator, without thermal throttling. Tiling raises the share of students found on the SCB dataset from 49% to 76%, and from 17% to 74% in the far third, without retraining. On COCO val2017, the deployed INT8 models lose 5-7 keypoint AP points against their PyTorch versions.

Classifiers trained on SCB do not transfer to real lessons: they flag 48-89% of an attentive lecture as distracted. The calibrated rules keep false alarms low, with 90-91% specificity on 451 human labels from three clips, including a 10-student class recorded after every threshold was fixed. With a nose-to-shoulder head-turn measure that keeps working in profile, they find 45-56% of scripted distraction. Phone use remains invisible to pose. The results show that an aggregate, on-device indicator is feasible, while reliable per-student judgement needs classroom-specific training labels.

**Keywords:** student engagement, pose estimation, edge computing, privacy by design, Hailo-8, Raspberry Pi

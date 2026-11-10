# COCO keypoint results: stock vs trained vs HEF

Run 2026-11-11. COCO val2017 (5,000 images, 11,004 person annotations), pycocotools, square 640×640 letterbox for every model. Method: [README.md](README.md).

- PyTorch rows: FP32 on an RTX 3060 (Ultralytics 8.4.208, torch 2.15.0+cu132).
- Stock models: Ultralytics `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, run with `--keep 17` and then cut to the 13 upper-body points.
- Trained .pt: `upper_body_n.pt` (3.21 M parameters), `upper_body_s.pt` (11.50 M), `upper_body_m.pt` (26.42 M), all with kpt_shape [13, 3].
- HEF rows: `upper_body_{n,s,m}.hef` on the Hailo-8 (Raspberry Pi 5). Each HEF run printed `class scores: sigmoid inside the HEF` (n, s and m).

## 13 keypoints, COCO sigmas (main result)

| Model | AP | AP50 | AP75 | AP medium | AP large | AR | AR50 | AR75 | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stock_n | 0.523 | 0.795 | 0.564 | 0.445 | 0.647 | 0.605 | 0.860 | 0.653 | 0.513 | 0.734 |
| stock_s | 0.613 | 0.858 | 0.686 | 0.550 | 0.718 | 0.688 | 0.907 | 0.756 | 0.613 | 0.793 |
| stock_m | 0.665 | 0.886 | 0.737 | 0.616 | 0.752 | 0.736 | 0.929 | 0.802 | 0.673 | 0.826 |
| pt_n | 0.470 | 0.781 | 0.495 | 0.405 | 0.578 | 0.550 | 0.836 | 0.589 | 0.465 | 0.668 |
| pt_s | 0.568 | 0.843 | 0.629 | 0.515 | 0.666 | 0.643 | 0.890 | 0.705 | 0.572 | 0.743 |
| pt_m | 0.625 | 0.875 | 0.698 | 0.586 | 0.709 | 0.697 | 0.913 | 0.763 | 0.638 | 0.782 |
| hef_n | 0.417 | 0.772 | 0.407 | 0.346 | 0.525 | 0.494 | 0.821 | 0.511 | 0.411 | 0.608 |
| hef_s | 0.260 | 0.683 | 0.127 | 0.242 | 0.304 | 0.358 | 0.757 | 0.293 | 0.318 | 0.414 |
| hef_m | 0.557 | 0.870 | 0.628 | 0.524 | 0.631 | 0.631 | 0.905 | 0.704 | 0.578 | 0.704 |

## 13 keypoints, uniform sigmas (1/13, as in the Ultralytics validator without `kpt_oks_sigmas`)

| Model | AP | AP50 | AP75 | AP medium | AP large | AR | AR50 | AR75 | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stock_n | 0.594 | 0.830 | 0.658 | 0.533 | 0.703 | 0.680 | 0.895 | 0.740 | 0.605 | 0.785 |
| stock_s | 0.673 | 0.878 | 0.747 | 0.624 | 0.764 | 0.748 | 0.928 | 0.812 | 0.689 | 0.833 |
| stock_m | 0.717 | 0.903 | 0.794 | 0.679 | 0.791 | 0.787 | 0.942 | 0.851 | 0.737 | 0.857 |
| pt_n | 0.579 | 0.833 | 0.644 | 0.523 | 0.678 | 0.655 | 0.882 | 0.717 | 0.584 | 0.754 |
| pt_s | 0.660 | 0.879 | 0.731 | 0.617 | 0.748 | 0.730 | 0.920 | 0.792 | 0.674 | 0.811 |
| pt_m | 0.706 | 0.897 | 0.786 | 0.675 | 0.780 | 0.772 | 0.932 | 0.837 | 0.726 | 0.839 |
| hef_n | 0.551 | 0.828 | 0.608 | 0.487 | 0.655 | 0.623 | 0.871 | 0.684 | 0.550 | 0.726 |
| hef_s | 0.438 | 0.788 | 0.450 | 0.394 | 0.508 | 0.520 | 0.839 | 0.563 | 0.472 | 0.588 |
| hef_m | 0.679 | 0.893 | 0.764 | 0.644 | 0.754 | 0.741 | 0.927 | 0.818 | 0.693 | 0.812 |

## Stock models, all 17 keypoints, COCO sigmas (sanity check)

| Model | AP | AP50 | AP75 | AP medium | AP large | AR | AR50 | AR75 | AR medium | AR large |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stock_n | 0.499 | 0.786 | 0.535 | 0.425 | 0.619 | 0.579 | 0.854 | 0.623 | 0.489 | 0.705 |
| stock_s | 0.594 | 0.853 | 0.656 | 0.532 | 0.699 | 0.667 | 0.901 | 0.730 | 0.592 | 0.773 |
| stock_m | 0.645 | 0.880 | 0.716 | 0.599 | 0.731 | 0.717 | 0.924 | 0.784 | 0.653 | 0.807 |

## Reading

- **Sanity check passes.** Stock 17-keypoint AP is 49.9 / 59.4 / 64.5 (n / s / m), against Ultralytics' published 50.4 / 60.0 / 65.0. The gaps are 0.5-0.6 points, the size expected from the square letterbox, so the evaluation path is sound.
- **Training on 13 keypoints costs some accuracy on those 13 points.** Trained .pt vs stock (17 → 13), COCO sigmas: n 0.470 vs 0.523 (−5.3 points), s 0.568 vs 0.613 (−4.5), m 0.625 vs 0.665 (−4.0). With uniform sigmas the gap shrinks to −1.5 / −1.3 / −1.1. That suggests most of the loss is in fine placement of the tight-sigma face points (nose, eyes, ears) rather than in finding people or the body points.
- **INT8 costs n and m about 5-7 points; s is an outlier.** HEF vs trained .pt, COCO sigmas: n 0.417 vs 0.470 (−5.3), m 0.557 vs 0.625 (−6.8), **s 0.260 vs 0.568 (−30.8)**. The s HEF still detects people (AP50 0.683), but its keypoints are imprecise (AP75 0.127). On detections matched to the .pt (box IoU ≥ 0.7), its median keypoint error is 4-9% of box size, against 2-3% for the n and m HEFs. The errors are worst at elbows and wrists, and every point is shifted the same way (about −1.4% of box size in x, +1.2% in y). Box scores match the .pt. The three HEFs have the same output layout (keypoints as UINT16), so this is not a decoding difference; it points to the s model's quantization or compilation. The s HEF needs re-checking before any s result is reported, and the same skew would affect any on-device run that uses it.
- **The training log's keypoint mAP is low because its validation labels include people with no upper-body keypoints.** Trained .pt with uniform sigmas: 0.579 / 0.660 / 0.706, against the log's 0.329 / 0.376 / 0.401.
  - The training set's YOLO val2017 labels have 10,777 person lines (2,693 label files). In 4,503 of them (41.8%), all 13 keypoint visibilities are 0. pycocotools ignores such people; Ultralytics' validator counts them as targets that can never be matched (OKS is always 0).
  - Ultralytics 8.4.208 `yolo pose val` (imgsz 640, batch 16), Pose P / R / mAP50 / mAP50-95. On the unfiltered labels it reproduces the log: n 0.726 / 0.451 / 0.461 / 0.324, s 0.721 / 0.474 / 0.485 / 0.368, m 0.734 / 0.473 / 0.490 / 0.390.
  - On a copy with the 4,503 lines removed (6,274 left): n 0.764 / 0.744 / 0.792 / 0.555, s 0.784 / 0.762 / 0.833 / 0.632, m 0.784 / 0.770 / 0.842 / 0.669. Box mAP50-95 also rises, from 0.523 / 0.579 / 0.612 to 0.653 / 0.691 / 0.713, because the same people are box targets too.
  - Conclusion: the unmatched no-keypoint people explain about 0.23-0.28 of the 0.25-0.31 gap. The remaining 0.02-0.04 is the expected difference between Ultralytics' and pycocotools' matching. So the training-log numbers understate keypoint accuracy, and the pycocotools columns are the ones to report.
  - The dataset and training runs are in `~/Workspaces/tmp/hailo-custom-yolo-pose` and `~/Workspaces/tmp/custom-yolo-data` on the GPU machine. The .pt files used here are byte-identical to each run's `weights/best.pt`.

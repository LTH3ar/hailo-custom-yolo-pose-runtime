"""Keypoint name / skeleton presets, picked automatically from the keypoint count.

A model sidecar YAML can always override these.
"""

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 0-indexed pairs (same pairs the stock Hailo yolov8pose post-process draws)
COCO17_SKELETON = [
    [0, 1], [1, 3], [0, 2], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [12, 14], [13, 15], [14, 16],
]

# Upper body = COCO keypoints 0..12 (knees and ankles removed), the layout
# produced by hailo-custom-yolo-pose/config/upper-body-pose.yaml
UPPER_BODY13_NAMES = COCO17_NAMES[:13]
UPPER_BODY13_SKELETON = [p for p in COCO17_SKELETON if max(p) < 13]

PRESETS = {
    17: {"name": "coco17", "keypoint_names": COCO17_NAMES, "skeleton": COCO17_SKELETON},
    13: {"name": "upper_body13", "keypoint_names": UPPER_BODY13_NAMES, "skeleton": UPPER_BODY13_SKELETON},
}


def preset_for(num_keypoints: int) -> dict:
    if num_keypoints in PRESETS:
        return PRESETS[num_keypoints]
    return {
        "name": f"generic{num_keypoints}",
        "keypoint_names": [f"kp{i}" for i in range(num_keypoints)],
        "skeleton": [],
    }

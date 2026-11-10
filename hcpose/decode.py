"""NumPy YOLOv8-pose decoder (same math as the C++ hailofilter post-process).

Used by the pure-Python engine and by evaluation scripts, so both paths
produce identical detections for the same HEF outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .layout import Layout


@dataclass
class Pose:
    """Coordinates are normalized to the network input (0..1, letterboxed space)."""

    box: np.ndarray  # (4,) x1, y1, x2, y2
    score: float
    class_id: int
    keypoints: np.ndarray  # (K, 3) x, y, visibility


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def needs_sigmoid(arrays) -> bool:
    """Logits contain strongly negative background values; probabilities stay in [0, 1]."""
    mn = min(float(a.min()) for a in arrays)
    mx = max(float(a.max()) for a in arrays)
    return mn < -0.01 or mx > 1.01


class PoseDecoder:
    def __init__(
        self,
        layout: Layout,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        max_detections: int = 100,
        class_agnostic_nms: bool = True,
        score_activation: str = "auto",
        keypoint_activation: str = "auto",
        keypoint_threshold: float = 0.5,
        pose_dedup: float = 0.1,
    ):
        self.layout = layout
        self.keypoint_threshold = keypoint_threshold
        self.pose_dedup = pose_dedup
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.class_agnostic_nms = class_agnostic_nms
        self.score_sigmoid: Optional[bool] = self._parse(score_activation)
        self.kpt_sigmoid: Optional[bool] = self._parse(keypoint_activation)
        self._grids = []
        for s in layout.scales:
            rows, cols = np.mgrid[0 : s.h, 0 : s.w]
            self._grids.append((cols.reshape(-1).astype(np.float32), rows.reshape(-1).astype(np.float32)))
        self._bins = np.arange(layout.reg_max, dtype=np.float32)

    @staticmethod
    def _parse(a: str) -> Optional[bool]:
        a = (a or "auto").lower()
        if a in ("sigmoid", "logits"):
            return True
        if a in ("none", "identity", "probabilities"):
            return False
        return None

    def __call__(self, outputs: Dict[str, np.ndarray]) -> List[Pose]:
        """outputs: name -> dequantized float array of shape (H, W, F) or (1, H, W, F)."""
        L = self.layout
        get = lambda n: np.asarray(outputs[n], dtype=np.float32).reshape(-1)  # noqa: E731

        scores = [get(s.score).reshape(-1, L.num_classes) for s in L.scales]
        kpts = [get(s.kpt).reshape(-1, L.num_keypoints, 3) for s in L.scales]
        if self.score_sigmoid is None:
            self.score_sigmoid = needs_sigmoid(scores)
        if self.kpt_sigmoid is None:
            self.kpt_sigmoid = needs_sigmoid([k[..., 2] for k in kpts])

        all_conf, all_cls, all_box, all_kpt = [], [], [], []
        for si, s in enumerate(L.scales):
            sc = scores[si]
            cls = sc.argmax(axis=1)
            conf = sc[np.arange(sc.shape[0]), cls]
            if self.score_sigmoid:
                conf = _sigmoid(conf)
            keep = np.nonzero(conf >= self.score_threshold)[0]
            if keep.size == 0:
                continue
            cols, rows = self._grids[si][0][keep], self._grids[si][1][keep]

            dfl = get(s.box).reshape(-1, 4, L.reg_max)[keep]
            dfl = np.exp(dfl - dfl.max(axis=2, keepdims=True))
            dist = (dfl * self._bins).sum(axis=2) / dfl.sum(axis=2)  # (n, 4) ltrb in grid units
            cx, cy = (cols + 0.5) * s.stride_x, (rows + 0.5) * s.stride_y
            box = np.stack(
                [
                    (cx - dist[:, 0] * s.stride_x) / L.input_width,
                    (cy - dist[:, 1] * s.stride_y) / L.input_height,
                    (cx + dist[:, 2] * s.stride_x) / L.input_width,
                    (cy + dist[:, 3] * s.stride_y) / L.input_height,
                ],
                axis=1,
            )
            kp = kpts[si][keep].copy()
            kp[..., 0] = (kp[..., 0] * 2.0 + cols[:, None]) * s.stride_x / L.input_width
            kp[..., 1] = (kp[..., 1] * 2.0 + rows[:, None]) * s.stride_y / L.input_height
            if self.kpt_sigmoid:
                kp[..., 2] = _sigmoid(kp[..., 2])
            all_conf.append(conf[keep])
            all_cls.append(cls[keep])
            all_box.append(box)
            all_kpt.append(kp)

        if not all_conf:
            return []
        conf = np.concatenate(all_conf)
        cls = np.concatenate(all_cls)
        box = np.concatenate(all_box)
        kp = np.concatenate(all_kpt)
        keep = self._nms(box, conf, cls)
        if self.pose_dedup > 0:
            keep = self._dedup(keep, box, kp)
        return [Pose(box=box[i], score=float(conf[i]), class_id=int(cls[i]), keypoints=kp[i]) for i in keep]

    def _dedup(self, keep: List[int], box, kp) -> List[int]:
        """Drop a detection whose skeleton matches a higher-scored one: its box lies inside
        the other (>= 70% of the smaller box), >= 3 keypoints are visible in both, and their
        mean distance is below pose_dedup * sqrt(smaller box area). YOLO-pose often emits a
        tight and a loose box for one person, and box-IoU NMS keeps both. Same rule as the
        C++ post-process (dedup_skeletons)."""
        W, H = float(self.layout.input_width), float(self.layout.input_height)
        out: List[int] = []
        for i in keep:
            ai = (box[i, 2] - box[i, 0]) * W * (box[i, 3] - box[i, 1]) * H
            dup = False
            for j in out:
                aj = (box[j, 2] - box[j, 0]) * W * (box[j, 3] - box[j, 1]) * H
                iw = max(0.0, min(box[i, 2], box[j, 2]) - max(box[i, 0], box[j, 0])) * W
                ih = max(0.0, min(box[i, 3], box[j, 3]) - max(box[i, 1], box[j, 1])) * H
                small = max(min(ai, aj), 1e-9)
                if iw * ih / small < 0.7:
                    continue
                m = (kp[i, :, 2] >= self.keypoint_threshold) & (kp[j, :, 2] >= self.keypoint_threshold)
                if m.sum() < 3:
                    continue
                d = np.hypot((kp[i, m, 0] - kp[j, m, 0]) * W, (kp[i, m, 1] - kp[j, m, 1]) * H).mean()
                if d < self.pose_dedup * np.sqrt(small):
                    dup = True
                    break
            if not dup:
                out.append(i)
        return out

    def _nms(self, box, conf, cls) -> List[int]:
        order = np.argsort(-conf, kind="stable")
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        removed = np.zeros(len(order), dtype=bool)
        keep: List[int] = []
        for oi, i in enumerate(order):
            if removed[oi]:
                continue
            keep.append(int(i))
            if len(keep) >= self.max_detections:
                break
            rest = order[oi + 1 :]
            xx1 = np.maximum(box[i, 0], box[rest, 0])
            yy1 = np.maximum(box[i, 1], box[rest, 1])
            xx2 = np.minimum(box[i, 2], box[rest, 2])
            yy2 = np.minimum(box[i, 3], box[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            union = area[i] + area[rest] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
            sup = iou > self.iou_threshold
            if not self.class_agnostic_nms:
                sup &= cls[rest] == cls[i]
            removed[oi + 1 :] |= sup
        return keep

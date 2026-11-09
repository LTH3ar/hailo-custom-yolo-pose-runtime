"""Stand-in for the TAPPAS `hailo` python bindings, for pipeline tests without Hailo.

Each GStreamer buffer gets a synthetic ROI (keyed by PTS) with a small seated
class; student 0 turns their head away after 5 s. Implements only what
hcpose's gst engine uses.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAILO_DETECTION, HAILO_LANDMARKS, HAILO_UNIQUE_ID, HAILO_CLASSIFICATION = "det", "lm", "uid", "cls"
W, H = 1280, 720
_rois = {}
stats = {"removed": 0, "added": 0, "labels": []}


class HailoBBox:
    def __init__(self, xmin, ymin, width, height):
        self._b = (xmin, ymin, width, height)

    def xmin(self):
        return self._b[0]

    def ymin(self):
        return self._b[1]

    def width(self):
        return self._b[2]

    def height(self):
        return self._b[3]


class _Obj:
    def __init__(self, kind):
        self.kind = kind
        self.objects = []

    def get_objects_typed(self, kind):
        return [o for o in self.objects if o.kind == kind]

    def get_objects(self):
        return list(self.objects)

    def add_object(self, o):
        self.objects.append(o)
        if o.kind == HAILO_DETECTION and isinstance(self, _ROI):
            stats["added"] += 1
            stats["labels"].append((o.get_class_id(), o.get_label()))

    def remove_object(self, o):
        self.objects.remove(o)
        stats["removed"] += 1


class _ROI(_Obj):
    def __init__(self):
        super().__init__("roi")


class _Point:
    def __init__(self, x, y, c):
        self._p = (x, y, c)

    def x(self):
        return self._p[0]

    def y(self):
        return self._p[1]

    def confidence(self):
        return self._p[2]


class _Landmarks(_Obj):
    def __init__(self, pts):
        super().__init__(HAILO_LANDMARKS)
        self.pts = pts

    def get_points(self):
        return self.pts


class HailoDetection(_Obj):
    def __init__(self, bbox, class_id, label, confidence):
        super().__init__(HAILO_DETECTION)
        self.bbox, self.cid, self.label, self.conf = bbox, class_id, label, confidence

    def get_bbox(self):
        return self.bbox

    def get_confidence(self):
        return self.conf

    def get_class_id(self):
        return self.cid

    def get_label(self):
        return self.label


def _make_roi(t):
    from test_engagement import seats_grid, synth_person

    roi = _ROI()
    for i, s in enumerate(seats_grid(6)):
        p = synth_person(s["cx"], s["cy"], s["sw"], yaw=1.0 if (i == 0 and t > 5.0) else 0.0)
        x1, y1, x2, y2 = p.box
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        det = HailoDetection(HailoBBox(x1 / W, y1 / H, bw, bh), 0, "person", 0.9)
        pts = [_Point((k[0] / W - x1 / W) / bw, (k[1] / H - y1 / H) / bh, float(k[2])) for k in p.keypoints]
        det.add_object(_Landmarks(pts))
        roi.objects.append(det)
    return roi


def get_roi_from_buffer(buf):
    key = buf.pts
    if key not in _rois:
        _rois[key] = _make_roi(key / 1e9)
    return _rois[key]

"""Work out what each HEF output tensor is, from shapes alone.

Mirrors resolve_layout() in cpp/yolov8_pose_postprocess.cpp so the Python
engine, `hcpose inspect` and the C++ hailofilter agree.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


class LayoutError(RuntimeError):
    pass


@dataclass
class Scale:
    box: str
    score: str
    kpt: str
    h: int
    w: int
    stride_x: float = 0.0
    stride_y: float = 0.0


@dataclass
class Layout:
    scales: List[Scale]
    num_keypoints: int
    num_classes: int
    reg_max: int
    input_width: int
    input_height: int
    notes: List[str] = field(default_factory=list)


def _describe(outputs: Sequence[Tuple[str, Tuple[int, int, int]]]) -> str:
    return "".join(f"\n    {n}  {s[0]}x{s[1]}x{s[2]}" for n, s in outputs)


def resolve_layout(
    outputs: Sequence[Tuple[str, Tuple[int, int, int]]],
    input_hw: Optional[Tuple[int, int]] = None,
    num_keypoints: Optional[int] = None,
    num_classes: Optional[int] = None,
    reg_max: Optional[int] = None,
) -> Layout:
    """outputs: [(name, (h, w, features)), ...]"""
    outputs = list(outputs)
    if not outputs or len(outputs) % 3:
        raise LayoutError(
            f"expected 3 outputs per scale (boxes, scores, keypoints), got {len(outputs)}:"
            f"{_describe(outputs)}\n  The HEF must be compiled with the 9 raw head convs as end "
            "nodes (no on-chip NMS), as in hailo-custom-yolo-pose/scripts/04-pytorch-to-hailo.py."
        )
    want_kp = num_keypoints * 3 if num_keypoints else None
    want_nc = num_classes if num_classes else None
    want_box = reg_max * 4 if reg_max else 64

    groups: Dict[Tuple[int, int], list] = defaultdict(list)
    for name, shape in outputs:
        groups[(shape[0], shape[1])].append((name, shape[2]))

    scales: List[Scale] = []
    box_f = score_f = kp_f = None
    for (h, w), ts in groups.items():
        if len(ts) != 3:
            raise LayoutError(f"grid {h}x{w} has {len(ts)} tensors, expected 3:{_describe(outputs)}")
        box = None
        rest = []
        for name, f in ts:
            if box is None and f == want_box and f != want_kp and f != want_nc:
                box = (name, f)
            else:
                rest.append((name, f))
        if box is None or len(rest) != 2:
            raise LayoutError(
                f"no box (DFL) tensor with {want_box} channels in grid {h}x{w}; set reg_max in the "
                f"sidecar if your head is non-standard:{_describe(outputs)}"
            )
        (n0, f0), (n1, f1) = rest
        if want_kp:
            kp_i = 0 if f0 == want_kp else (1 if f1 == want_kp else None)
        elif want_nc:
            kp_i = 1 if f0 == want_nc else (0 if f1 == want_nc else None)
        elif (f0 % 3 == 0) != (f1 % 3 == 0):
            kp_i = 0 if f0 % 3 == 0 else 1
        else:
            kp_i = 0 if f0 >= f1 else 1
        if kp_i is None:
            raise LayoutError(
                f"channel counts in grid {h}x{w} don't match num_keypoints/num_classes:{_describe(outputs)}"
            )
        kpt = rest[kp_i]
        score = rest[1 - kp_i]
        if kpt[1] % 3:
            raise LayoutError(f"keypoint tensor {kpt[0]} has {kpt[1]} channels, not a multiple of 3")
        if (box_f, score_f, kp_f) != (None, None, None) and (box[1], score[1], kpt[1]) != (box_f, score_f, kp_f):
            raise LayoutError(f"channel counts differ between scales:{_describe(outputs)}")
        box_f, score_f, kp_f = box[1], score[1], kpt[1]
        scales.append(Scale(box=box[0], score=score[0], kpt=kpt[0], h=h, w=w))

    scales.sort(key=lambda s: -s.w)
    if input_hw:
        in_h, in_w = input_hw
    else:
        in_h, in_w = scales[0].h * 8, scales[0].w * 8
    for s in scales:
        s.stride_x = in_w / s.w
        s.stride_y = in_h / s.h
    return Layout(
        scales=scales,
        num_keypoints=kp_f // 3,
        num_classes=score_f,
        reg_max=box_f // 4,
        input_width=in_w,
        input_height=in_h,
    )

"""Decoder correctness tests.

Fixture: the 9 raw head outputs of the official yolov8n-pose on Ultralytics'
bus.jpg (letterboxed to 640x640), plus Ultralytics' own decoded + NMS'd
predictions. These are exactly the tensors a HEF compiled with the 9 head
convs as end nodes produces (before quantization).

  * NumPy decoder must match Ultralytics bit-for-bit (float32 noise).
  * C++ hailofilter post-process must match the NumPy decoder for float32,
    uint8/uint16-quantized, sigmoid-in-HEF, shuffled and 13-keypoint inputs.

The C++ part needs the harness binary: set HCPOSE_CPP_HARNESS, or build it with
  g++ -std=c++17 -O2 -I<tappas>/core/hailo/general -I<hailort>/include -I<rapidjson>/include \
      tests/cpp_harness.cpp cpp/yolov8_pose_postprocess.cpp -o build/cpp_harness
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hcpose.decode import PoseDecoder  # noqa: E402
from hcpose.layout import resolve_layout  # noqa: E402

FIXTURE = ROOT / "tests" / "data" / "yolov8n_pose_bus_head.npz"
HARNESS = os.environ.get("HCPOSE_CPP_HARNESS", str(ROOT / "build" / "cpp_harness"))
THR, IOU = 0.25, 0.7


def load():
    ref = np.load(FIXTURE)
    raw = {k.replace("__", "/"): ref[k] for k in ref.files if k != "det"}
    return raw, ref["det"]


def decode(outputs, **kw):
    layout = resolve_layout([(n, a.shape) for n, a in outputs.items()], input_hw=(640, 640))
    dec = PoseDecoder(layout, score_threshold=THR, iou_threshold=IOU, **kw)
    return layout, dec(outputs), dec


def variants(raw):
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    yield "float32 logits", raw, 17
    yield "sigmoid in HEF", {k: (sig(v) if "cls" in k else v) for k, v in raw.items()}, 17
    items = list(raw.items())
    perm = np.random.default_rng(1).permutation(len(items))
    yield "shuffled + renamed", {f"model/conv{i * 7 + 3}": items[j][1] for i, j in enumerate(perm)}, 17
    yield "13 keypoints", {k: (v[..., :39] if "kpt" in k else v) for k, v in raw.items()}, 13


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing")
def test_numpy_matches_ultralytics():
    raw, det = load()
    det = det[np.argsort(-det[:, 4])]
    for name, outs, k in variants(raw):
        layout, poses, _ = decode(outs)
        assert layout.num_keypoints == k and layout.num_classes == 1 and layout.reg_max == 16, name
        assert [s.stride_x for s in layout.scales] == [8, 16, 32]
        assert len(poses) == len(det), name
        for p, d in zip(poses, det):
            np.testing.assert_allclose(p.box * 640, d[:4], atol=1e-3, err_msg=name)
            assert abs(p.score - d[4]) < 1e-5, name
            dk = d[6:].reshape(17, 3)[:k]
            np.testing.assert_allclose(p.keypoints[:, :2] * 640, dk[:, :2], atol=1e-3, err_msg=name)
            np.testing.assert_allclose(p.keypoints[:, 2], dk[:, 2], atol=1e-5, err_msg=name)


def quantize(a, bits):
    lo, hi = float(a.min()), float(a.max())
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    zp = round(-lo / scale)
    q = np.clip(np.round(a / scale + zp), 0, levels).astype(np.uint8 if bits == 8 else np.uint16)
    return q, zp, scale


def run_cpp(outputs, tmp, quant=None, letterbox=None, extra_cfg=None):
    """quant: None (float32) or dict name->bits"""
    lines, deq = [], {}
    for i, (name, a) in enumerate(outputs.items()):
        path = Path(tmp) / f"t{i}.bin"
        if quant:
            q, zp, scale = quantize(a, quant(name))
            q.tofile(path)
            deq[name] = (q.astype(np.float32) - zp) * scale
            t = 1 if q.dtype == np.uint8 else 2
        else:
            a.astype(np.float32).tofile(path)
            deq[name] = a
            zp, scale, t = 0.0, 1.0, 3
        h, w, f = a.shape
        lines.append(f"{name} {h} {w} {f} {t} {zp} {scale} {path}")
    man = Path(tmp) / "manifest.txt"
    man.write_text("\n".join(lines) + "\n")
    cfg = {"score_threshold": THR, "iou_threshold": IOU, "input_width": 640, "input_height": 640,
           "skeleton": [[0, 1], [1, 2], [50, 51]], "verbose": False}
    cfg.update(extra_cfg or {})
    cfg_path = Path(tmp) / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    args = [HARNESS, str(man), str(cfg_path), "1" if letterbox else "0"] + [str(v) for v in (letterbox or [])]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1]), deq


@pytest.mark.skipif(not (FIXTURE.exists() and os.path.isfile(HARNESS)), reason="fixture or C++ harness missing")
def test_cpp_matches_numpy():
    raw, _ = load()
    cases = [("float32", None), ("u8 boxes/scores + u16 kpts (Hailo default)", lambda n: 16 if "kpt" in n else 8),
             ("all uint16", lambda n: 16)]
    with tempfile.TemporaryDirectory() as tmp:
        for vname, outs, k in variants(raw):
            for qname, quant in cases:
                cpp, deq = run_cpp(outs, tmp, quant)
                _, poses, _ = decode(deq)
                label = f"{vname} / {qname}"
                assert len(cpp) == len(poses), (label, len(cpp), len(poses))
                for c, p in zip(cpp, poses):
                    np.testing.assert_allclose(c["box"], np.clip(p.box, 0, 1), atol=2e-4, err_msg=label)
                    assert abs(c["conf"] - p.score) < 1e-4, label
                    assert len(c["kpts"]) == k
                    kp = np.array(c["kpts"])
                    np.testing.assert_allclose(kp[:, :2], np.clip(p.keypoints[:, :2], 0, 1), atol=2e-4, err_msg=label)
                    np.testing.assert_allclose(kp[:, 2], p.keypoints[:, 2], atol=1e-4, err_msg=label)
                    assert c["label"] == "person"
                    assert c["pairs"] == 2  # out-of-range pair dropped


@pytest.mark.skipif(not (FIXTURE.exists() and os.path.isfile(HARNESS)), reason="fixture or C++ harness missing")
def test_cpp_letterbox_mapping():
    """1280x720 frame letterboxed into 640x640: content spans y in [0.21875, 0.78125]."""
    raw, _ = load()
    top, h = 0.21875, 0.5625
    scaling = [0.0, -top / h, 1.0, 1.0 / h]
    with tempfile.TemporaryDirectory() as tmp:
        cpp, _ = run_cpp(raw, tmp, None, letterbox=scaling)
        _, poses, _ = decode(raw)
        assert len(cpp) == len(poses)
        for c, p in zip(cpp, poses):
            exp = p.box.copy()
            exp[[1, 3]] = (exp[[1, 3]] - top) / h
            np.testing.assert_allclose(c["box"], np.clip(exp, 0, 1), atol=2e-4)
            ky = np.clip((p.keypoints[:, 1] - top) / h, 0, 1)
            np.testing.assert_allclose(np.array(c["kpts"])[:, 1], ky, atol=2e-4)


@pytest.mark.skipif(not (FIXTURE.exists() and os.path.isfile(HARNESS)), reason="fixture or C++ harness missing")
def test_cpp_forced_activation_and_sidecar_counts():
    raw, _ = load()
    with tempfile.TemporaryDirectory() as tmp:
        a, _ = run_cpp(raw, tmp, None, extra_cfg={"score_activation": "sigmoid", "num_keypoints": 17, "num_classes": 1})
        b, _ = run_cpp(raw, tmp, None)
        assert a == b
        # wrong forced activation must change results (proves the flag is honoured)
        c, _ = run_cpp(raw, tmp, None, extra_cfg={"score_activation": "none"})
        assert len(c) != len(b)

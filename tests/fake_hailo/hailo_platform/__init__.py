"""Minimal stand-in for HailoRT's python API, for exercising hcpose without hardware.

Outputs come from tests/data/yolov8n_pose_bus_head.npz (real yolov8n-pose head
outputs for bus.jpg). Set FAKE_HAILO_SIGMOID=1 to emulate a HEF whose .alls
applied sigmoid to the class-score outputs, FAKE_HAILO_K=13 for a 13-kpt model.
"""
import enum
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__version__ = "fake-0.0"
_FIX = Path(__file__).resolve().parents[2] / "data" / "yolov8n_pose_bus_head.npz"


def _outputs():
    ref = np.load(_FIX)
    k = int(os.environ.get("FAKE_HAILO_K", "17"))
    out = {}
    for name in ref.files:
        if name == "det":
            continue
        a = ref[name]
        if "kpt" in name:
            a = a[..., : 3 * k]
        if "cls" in name and os.environ.get("FAKE_HAILO_SIGMOID") == "1":
            a = 1 / (1 + np.exp(-a))
        out[name.replace("__", "/")] = a.astype(np.float32)
    return out


class FormatType(enum.Enum):
    AUTO = 0
    UINT8 = 1
    UINT16 = 2
    FLOAT32 = 3


class HailoStreamInterface(enum.Enum):
    PCIe = 0


class HEF:
    def __init__(self, path):
        self.path = path

    def get_input_vstream_infos(self):
        return [SimpleNamespace(name="net/input_layer1", shape=(640, 640, 3))]

    def get_output_vstream_infos(self):
        return [
            SimpleNamespace(name=n, shape=a.shape, format=SimpleNamespace(type=FormatType.UINT8),
                            quant_info=SimpleNamespace(qp_scale=0.1, qp_zp=128.0))
            for n, a in _outputs().items()
        ]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _NG:
    def create_params(self):
        return None

    def activate(self, params=None):
        return _Ctx()


class VDevice(_Ctx):
    def configure(self, hef, params):
        return [_NG()]

    def release(self):
        pass


class ConfigureParams:
    @staticmethod
    def create_from_hef(hef, interface=None):
        return None


class InputVStreamParams:
    @staticmethod
    def make(ng, format_type=None):
        return None


OutputVStreamParams = InputVStreamParams


class InferVStreams(_Ctx):
    def __init__(self, ng, ivp, ovp):
        self.out = _outputs()

    def infer(self, inputs):
        (x,) = inputs.values()
        assert x.dtype == np.uint8 and x.shape[1:] == (640, 640, 3), (x.dtype, x.shape)
        return {k: np.repeat(v[None], x.shape[0], axis=0) for k, v in self.out.items()}

"""Read input/output metadata from a HEF with HailoRT's Python API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .layout import Layout, resolve_layout


@dataclass
class OutputInfo:
    name: str
    shape: Tuple[int, int, int]
    dtype: str
    qp_scale: float
    qp_zp: float


@dataclass
class HefInfo:
    path: str
    input_name: str
    input_shape: Tuple[int, int, int]  # h, w, c
    outputs: List[OutputInfo]

    def layout(self, num_keypoints=None, num_classes=None, reg_max=None) -> Layout:
        return resolve_layout(
            [(o.name, o.shape) for o in self.outputs],
            input_hw=self.input_shape[:2],
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            reg_max=reg_max,
        )


def import_hailo_platform():
    try:
        import hailo_platform  # noqa: F401

        return hailo_platform
    except ImportError as e:
        raise RuntimeError(
            "HailoRT Python bindings (hailo_platform) are not importable.\n"
            "  On Raspberry Pi OS: sudo apt install hailo-all, and make sure the venv was created with\n"
            "  --system-site-packages (./install.sh does this). Run `hcpose doctor` for details."
        ) from e


def read_hef(path: str) -> HefInfo:
    hp = import_hailo_platform()
    hef = hp.HEF(path)
    inputs = hef.get_input_vstream_infos()
    if len(inputs) != 1:
        raise RuntimeError(f"expected a single input, HEF has {len(inputs)}")
    outs = []
    for info in hef.get_output_vstream_infos():
        shape = tuple(int(x) for x in info.shape)
        if len(shape) != 3:
            raise RuntimeError(
                f"output {info.name} has shape {shape}; expected HxWxC raw head outputs (no on-chip NMS)"
            )
        q = info.quant_info
        outs.append(
            OutputInfo(
                name=info.name,
                shape=shape,
                dtype=str(info.format.type).split(".")[-1],
                qp_scale=float(q.qp_scale),
                qp_zp=float(q.qp_zp),
            )
        )
    return HefInfo(
        path=path,
        input_name=inputs[0].name,
        input_shape=tuple(int(x) for x in inputs[0].shape),
        outputs=outs,
    )

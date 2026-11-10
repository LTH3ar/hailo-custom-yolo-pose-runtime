"""Validate the GStreamer pipeline the gst engine builds, on a machine without Hailo.

Hailo elements are swapped for stand-ins with the same pad topology
(hailocropper -> tee, hailoaggregator -> funnel, others -> identity), then the
pipeline is run on a real video file through GstRunner.run(), exercising
caps negotiation, the tee/encoder/mux branch, the metadata probe and EOS
handling. Run with a python that has gi + GStreamer:

    python tests/check_gst_pipeline.py input.mp4 out.mp4 [--tiles auto]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hcpose import env, gst_engine  # noqa: E402
from hcpose.config import ModelConfig  # noqa: E402
from hcpose.layout import Layout  # noqa: E402

env.find_postprocess_so = lambda: Path("/tmp/fake.so")
env.find_whole_buffer_cropper = lambda: Path("/tmp/whole.so")

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
_real_find = Gst.ElementFactory.find
Gst.ElementFactory.find = staticmethod(lambda n: True if n.startswith("hailo") else _real_find(n))

STANDINS = {
    "hailocropper": "tee",
    "hailoaggregator": "input-selector",
    "hailonet": "identity",
    "hailofilter": "identity",
    "hailotracker": "identity",
    "hailooverlay": "identity",
    "autovideosink": "fakesink",
}


def strip_hailo(desc: str) -> str:
    out = []
    for seg in re.split(r"(\s!\s|\s(?=crop\.|agg\.|t\.))", desc):
        m = re.match(r"\s*(hailo\w+|autovideosink)\b(.*)", seg, re.S)
        if m:
            el, rest = m.group(1), m.group(2)
            name = re.search(r"name=(\S+)", rest)
            sync = re.search(r"sync=(\S+)", rest)
            new = STANDINS[el] + (f" name={name.group(1)}" if name else "")
            if el == "autovideosink" and sync:
                new += f" sync={sync.group(1)}"
            if el == "hailocropper":
                # cropper is declared together with the aggregator in one segment
                new += " input-selector name=agg sync-streams=false"
            out.append(new + " ")
        else:
            out.append(seg)
    s = "".join(out)
    s = s.replace("hailoaggregator name=agg", "")
    return s


def main():
    src, out = sys.argv[1], sys.argv[2]
    tiles = sys.argv[sys.argv.index("--tiles") + 1] if "--tiles" in sys.argv else None
    cfg = ModelConfig(hef="/tmp/model.hef", labels=["person"])
    layout = Layout(scales=[], num_keypoints=13, num_classes=1, reg_max=16, input_width=640, input_height=640)
    frames = []
    r = gst_engine.GstRunner(cfg, layout, source=src, display=True, output=out, callbacks=[], tiles=tiles)
    r.callbacks = []  # metadata extraction needs the real `hailo` module
    orig = r.pipeline_string
    r.pipeline_string = lambda: strip_hailo(orig())
    desc = orig()
    if tiles:
        assert "function-name=create_tile_crops" in desc and "function-name=merge_tiles" in desc, desc
        assert desc.index("agg. !") < desc.index("merge_tiles") < desc.index("hailotracker"), desc
    print(r.pipeline_string().replace(" ! ", " !\n  "))
    r._on_buffer_orig = r._on_buffer

    def counting(pad, info):
        frames.append(1)
        return r._on_buffer_orig(pad, info)

    r._on_buffer = counting
    code = r.run()
    print(f"exit={code} frames={len(frames)}")
    assert code == 0 and len(frames) > 0


if __name__ == "__main__":
    main()

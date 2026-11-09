"""End-to-end check of the gst engine with engagement analysis, without Hailo hardware.

Uses the stand-in pipeline from check_gst_pipeline.py plus a fake `hailo`
bindings module (tests/fake_hailo/hailo.py) that attaches a synthetic seated
class to every buffer; student 0 turns away after 5 s. Verifies that the
analyzer runs on real buffers with media timestamps, that detections are
re-labelled for hailooverlay, that the class text reaches textoverlay, and
that the JSONL carries the labels.

    python tests/check_gst_engagement.py input.mp4 out.mp4 out.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "fake_hailo"))
sys.path.insert(0, str(ROOT))

import check_gst_pipeline as base  # noqa: E402  (patches env + Gst.ElementFactory.find)
import hailo  # noqa: E402  (the fake)
from hcpose import gst_engine  # noqa: E402
from hcpose.config import ModelConfig  # noqa: E402
from hcpose.engagement import EngagementAnalyzer, load_settings  # noqa: E402
from hcpose.layout import Layout  # noqa: E402
from hcpose.presets import UPPER_BODY13_NAMES, UPPER_BODY13_SKELETON  # noqa: E402
from hcpose.results import JsonlWriter  # noqa: E402


def main():
    src, out, jsonl = sys.argv[1:4]
    cfg = ModelConfig(hef="/tmp/model.hef", labels=["person"], keypoint_names=UPPER_BODY13_NAMES,
                      skeleton=UPPER_BODY13_SKELETON)
    layout = Layout(scales=[], num_keypoints=13, num_classes=1, reg_max=16, input_width=640, input_height=640)
    an = EngagementAnalyzer(load_settings(overrides={"display": "both"}), UPPER_BODY13_NAMES)
    writer = JsonlWriter(jsonl, {"keypoint_names": UPPER_BODY13_NAMES})
    r = gst_engine.GstRunner(cfg, layout, source=src, display=True, output=out, callbacks=[writer], engagement=an)
    orig = r.pipeline_string
    r.pipeline_string = lambda: base.strip_hailo(orig())
    desc = r.pipeline_string()
    assert "textoverlay name=class_text" in desc, desc
    texts = []
    r._annotate_orig = r._annotate

    def annotate(buf, result):
        r._annotate_orig(buf, result)
        if r._text_overlay is not None:
            texts.append(r._text_overlay.get_property("text"))

    r._annotate = annotate
    code = r.run()
    writer.close()
    lines = [json.loads(x) for x in open(jsonl)][1:]
    print(f"exit={code} frames={len(lines)} relabelled={hailo.stats['added']} last class text: {texts[-1]!r}")
    assert code == 0 and len(lines) > 250
    assert hailo.stats["added"] == hailo.stats["removed"] == 6 * len(lines)
    last = lines[-1]
    assert last["t"] > 9.0, last["t"]  # media time from PTS (the 300-frame clip is 10 s long)
    st = [p["engagement"]["state"] for p in last["persons"]]
    assert st[0] == "distracted" and st[1:] == ["focused"] * 5, st
    assert last["class"]["counted"] == 6 and abs(last["class"]["pct_distracted"] - 16.7) < 0.1
    labels = hailo.stats["labels"][-6:]
    assert labels[0] == (0, "distracted") and all(lb == (1, "focused") for lb in labels[1:]), labels
    assert texts[-1].startswith("17% distracted") or texts[-1][0].isdigit(), texts[-1]
    print("gst engagement check OK")


if __name__ == "__main__":
    main()

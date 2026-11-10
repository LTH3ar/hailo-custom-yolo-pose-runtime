# Edge measurements

`bench.sh CLIP.mp4 SOAK_SOURCE.mp4 [OUT_DIR]` measures, on the Pi 5 + Hailo-8:
- **Idle baseline:** 60 s with nothing running.
- **Hailo-8 alone:** `hailortcli benchmark` for each HEF.
- **End to end on a clip, as fast as possible:**
  - FPS
  - CPU load (whole Pi, and hcpose's processes)
  - memory
  - SoC temperature
  - throttling flags
  - Pi board power from the Pi 5 PMIC rails (`vcgencmd pmic_read_adc`; it excludes the AI HAT and USB devices, so Hailo-8 power comes from the vendor's specification)
- **Pipeline latency:** the clip is paced at its frame rate like a camera (`HCPOSE_PACE_FILE=1` inserts `identity name=live_pace sync=true` after the decoder); GstShark's `interlatency` tracer logs each frame at `live_pace` and at the sink, and `report.py` takes the difference of the log timestamps. (GStreamer's core `latency` tracer deadlocks this pipeline on GStreamer 1.26.2.)
- `FROM=latency bench.sh ...` resumes at the latency runs.
- **Two 15-minute soaks:** n + tiles in real time (the deployment setting), and m + tiles as fast as possible (worst case).

`monitor.py` is the 1 Hz sampler. `report.py OUT_DIR` writes `edge_report.md`.

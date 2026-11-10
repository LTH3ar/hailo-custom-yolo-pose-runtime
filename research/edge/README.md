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
- **Pipeline latency:** source to sink, with the GStreamer latency tracer while the clip plays in real time.
- **Two 15-minute soaks:** n + tiles in real time (the deployment setting), and m + tiles as fast as possible (worst case).

`monitor.py` is the 1 Hz sampler. `report.py OUT_DIR` writes `edge_report.md`.

# Rule threshold sweep

`sweep_rules.py` replays the saved INT8 skeleton files in `research/results/` with a grid of rule thresholds. No model is run; it is NumPy on recorded keypoints. It scores each setting on:
- the staged clip_4: recall, precision, false alarms, latency
- the human labels: false alarms on focused moments, distracted moments caught
- the class distracted % on the lecture, exam and study-hall clips

Clip_4 and the labels choose the setting. The 9-person staged session is kept back as the held-out test.

Setup on any Linux machine (no Hailo needed): `python3 -m venv .venv && .venv/bin/pip install -e . opencv-python-headless`, then `.venv/bin/python research/tune/sweep_rules.py --jobs N`.

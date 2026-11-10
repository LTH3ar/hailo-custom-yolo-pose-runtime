# Classroom clips on the deployed models

`run_clips.sh OUT_DIR clip.mp4:auto clip_2.mp4:center clip_3.mp4:center` runs the INT8 HEFs on each clip (n and m, tiles off and on). It replays the skeletons with the rules and the three learned classifiers. `summarize.py` writes per-clip tables:
- students per frame
- same-skeleton duplicates per frame
- inferred class size
- class distracted % per classifier
- the width of the range
- FPS

The clips are third-party recordings of minors and stay out of the repo; only the skeleton files and the reports are committed.

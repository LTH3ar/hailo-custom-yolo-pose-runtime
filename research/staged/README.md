# Staged classroom session

A scripted 15-minute session with 8-10 adult volunteers. Every participant follows a card ("03:00-03:45 turn to S3 and talk"), so each student's true state is known second by second. It measures what the classroom clips can't: whether hcpose catches real distraction (recall, latency), and not only whether it avoids false alarms.

- `script_v1.csv`: the behaviour script (times relative to the "hands up" start signal).
- `run_all.sh VIDEO script_v1.csv --rows 3,4,3`: runs hcpose with n and m, tiles off and on, replays the skeletons with all four classifiers, and scores them (`results/summary.md`).
- `score.py`: the scorer.
  - Seats are found by position in the first 3 minutes: front row = lowest in the image, numbered left to right as the camera sees it.
  - Scripted changes have a 5 s margin on each side, which is excluded from scoring.
  - It reports coverage, recall, precision, false-alarm rates on attentive and on hard-focused seconds (writing, reading, hand raised, glances), and latency.
  - At class level it reports the error, the correlation and how often the true share falls inside the shown range.

The full protocol (room, camera, consent, roles, participant cards) is a separate document. `tests/test_staged.py` checks the scorer on a synthetic session with a known answer.

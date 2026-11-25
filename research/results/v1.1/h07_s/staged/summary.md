| File | Coverage | Recall | Precision | False alarms, attentive | False alarms, hard focused | Latency (s) | Events missed | Class error (pts) | Class r | Range hit |
|---|---|---|---|---|---|---|---|---|---|---|
| clip_4_720p_s_auto_logistic.jsonl | 100% | 35% | 64% | 44% | 0% | 2.0 | 4 of 7 | 29.0 | 0.45 | 54% |
| clip_4_720p_s_auto_mlp.jsonl | 100% | 49% | 100% | 0% | 0% | 3.5 | 3 of 7 | 24.0 | 0.67 | 77% |
| clip_4_720p_s_auto_none.jsonl | 100% | 43% | 100% | 0% | 0% | 2.0 | 3 of 7 | 26.4 | 0.85 | 63% |
| clip_4_720p_s_auto_trees.jsonl | 100% | 35% | 100% | 0% | 0% | 4.0 | 4 of 7 | 32.1 | 0.66 | 63% |
| clip_4_720p_s_none_logistic.jsonl | 100% | 35% | 64% | 44% | 0% | 2.0 | 4 of 7 | 29.0 | 0.45 | 54% |
| clip_4_720p_s_none_mlp.jsonl | 100% | 49% | 100% | 0% | 0% | 3.5 | 3 of 7 | 24.0 | 0.67 | 77% |
| clip_4_720p_s_none_none.jsonl | 100% | 43% | 100% | 0% | 0% | 2.0 | 3 of 7 | 26.4 | 0.85 | 63% |
| clip_4_720p_s_none_trees.jsonl | 100% | 35% | 100% | 0% | 0% | 4.0 | 4 of 7 | 32.1 | 0.66 | 63% |

Recall per scripted behaviour (share of the event's seconds labelled distracted):

| File | look to the side | phone on the table | phone under the table | talk to each other | turn around |
|---|---|---|---|---|---|
| clip_4_720p_s_auto_logistic.jsonl | 47% | 0% | 0% | 50% | 100% |
| clip_4_720p_s_auto_mlp.jsonl | 50% | 0% | 0% | 100% | 100% |
| clip_4_720p_s_auto_none.jsonl | 100% | 0% | 0% | 50% | 100% |
| clip_4_720p_s_auto_trees.jsonl | 47% | 0% | 0% | 50% | 100% |
| clip_4_720p_s_none_logistic.jsonl | 47% | 0% | 0% | 50% | 100% |
| clip_4_720p_s_none_mlp.jsonl | 50% | 0% | 0% | 100% | 100% |
| clip_4_720p_s_none_none.jsonl | 100% | 0% | 0% | 50% | 100% |
| clip_4_720p_s_none_trees.jsonl | 47% | 0% | 0% | 50% | 100% |

Start signal at 0.1 s into the video (first file). Seats found (x, y as a share of the frame): S1 (0.44, 0.38), S2 (0.61, 0.40)

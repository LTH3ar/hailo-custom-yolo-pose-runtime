## Handoff 04 report
- PR #6 merged (main 47e2592), classifiers tests: 8 passed.
- Scan (scan.txt): 0.95 143 groups (132 single) ... 0.85 86 (68) ... 0.40 42 (22). No plateau; the two large groups (155, 122) are stable from 0.85 down.
- Image sizes: only two in the whole set (2560x1440: 232, 3840x2160: 168), so the size check cannot separate views.
- No single --min-corr chosen; C3/C4 run at 0.85 and 0.40 (corr0.85/, corr0.40/).
- Groups after pose matching: 0.85 -> 85 ClaBehavior groups; 0.40 -> 41 (Handoff 02: 106).

C3 strict, both (AUC / within / bal acc), H02 (107 groups) | 0.85 | 0.40
- logistic pooled 0.772/0.730/70.9 | 0.770/0.733/70.9 | 0.772/0.730/70.9
  on SCB 0.779/0.751/71.6 | 0.777/0.755/71.6 | 0.779/0.752/71.6
  on Cla 0.703/0.685/65.1 | 0.707/0.686/65.5 | 0.706/0.683/65.3
- trees pooled 0.873/0.843/78.7 | 0.875/0.847/78.9 | 0.876/0.845/79.1
  on SCB 0.878/0.848/79.0 | 0.879/0.853/79.2 | 0.881/0.851/79.5
  on Cla 0.841/0.832/75.9 | 0.846/0.834/76.1 | 0.840/0.831/75.7
- MLP pooled 0.883/0.857/79.7 | 0.889/0.859/80.4 | 0.889/0.861/80.2
  on SCB 0.888/0.867/80.2 | 0.894/0.867/80.8 | 0.894/0.868/80.6
  on Cla 0.848/0.836/75.3 | 0.854/0.843/77.3 | 0.858/0.847/77.7

C4 lenient, both
- logistic pooled 0.762/0.726/70.5 | 0.756/0.725/69.9 | 0.759/0.723/70.2
  on SCB 0.762/0.729/70.4 | 0.757/0.731/70.0 | 0.760/0.730/70.3
  on Cla 0.742/0.718/68.7 | 0.731/0.710/67.5 | 0.725/0.703/67.4
- trees pooled 0.855/0.836/77.8 | 0.852/0.835/77.6 | 0.851/0.833/77.3
  on SCB 0.855/0.833/77.7 | 0.852/0.837/77.5 | 0.852/0.835/77.3
  on Cla 0.848/0.842/76.4 | 0.840/0.828/76.1 | 0.841/0.830/75.4
- MLP pooled 0.865/0.842/78.5 | 0.862/0.834/78.6 | 0.865/0.844/78.6
  on SCB 0.864/0.838/78.4 | 0.864/0.838/78.7 | 0.865/0.842/78.5
  on Cla 0.855/0.851/76.7 | 0.837/0.825/76.0 | 0.852/0.847/76.5

- Cross-dataset lines: identical to Handoff 02 at both thresholds (all 12 lines).
- Odd: the SCB-only rows move by up to 0.006 AUC though SCB's groups did not change; that is the fold-assignment noise floor. The ClaBehavior rows move by at most 0.018 (lenient MLP at 0.85) and not in step with coarser grouping (0.40 is not lower than 0.85).
- ~/datasets/clabehavior_yolo/groups.json is left at the 0.40 grouping; the Handoff 02 file is h04/groups_h02.json.

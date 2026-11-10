# Pi 5 + Hailo-8 results, 2026-11-10

On-device numbers for the thesis: the INT8 HEFs (`upper_body_{n,s,m}.hef`) on the three classroom clips, and the edge measurements (load, temperature, board power, latency, 15-minute soaks).

- **Hardware:** Raspberry Pi 5 Model B Rev 1.1, Hailo-8 AI HAT (26 TOPS), firmware 4.23.0.
- **Software:** Raspberry Pi OS trixie, kernel 6.18.60, hailo-all 5.1.1, GStreamer 1.26.2.
- **Commit:** branch `edge-eval`. The changes in "What failed" below are committed alongside these results.

## Cooling

There is a fan on the Pi 5's 4-pin fan header: `/sys/class/thermal/cooling_device0` is a `pwm-fan` with 4 speed steps. It sat at step 1 of 4 at idle. That rules out passive cooling. From software I can't tell the official Active Cooler (heatsink + fan) apart from the official case fan. **To confirm:** which one it is.

Throttling flags stayed `0x0` in every run, and the SoC peaked at 61.1 °C.

## Files

- `clips_report.md` and `<clip>_<model>_<tiles>.log`: the clip runs (step 1).
- `<clip>_<model>_<tiles>.skeleton.jsonl.xz`: skeletons, compressed with xz.
  - gzip -9 only got them from 497 MB to about 124 MB, over the 100 MB budget; xz -9 gives 79 MB.
  - Unpack with `xz -dk FILE.xz`.
  - Rebuild the analysis files with `hcpose analyze FILE.skeleton.jsonl --classifier {none,trees,logistic,mlp} --camera-position {auto,center}`, using `auto` for clip and `center` for clip_2 and clip_3.
- `edge_report.md`, `system.txt`, `*.csv` (1 Hz monitor samples), `throughput_*.log`, `latency_*.log`, `soak_*.log`, `hailo_benchmark_*.txt`: the edge measurements (step 2).
- `latency_*.trace.gz`: GstShark interlatency traces.
- Left out: the per-classifier analysis `.jsonl` files (they can be regenerated), `soak_input.mp4`, and all video and images.

## What failed or looked wrong

1. **The GStreamer core `latency` tracer deadlocks this pipeline.**
   - What happened: with `GST_TRACERS=latency`, the pipeline never passed a first buffer. The trace held only the tracer's registration lines, and SIGINT at 70 s didn't stop it. bench.sh would have hung there indefinitely, so I killed it.
   - Narrowing it down:
     - The same command without the tracer runs fine (25 FPS, real time).
     - It hangs with the tracer whether or not frames are synced, and whether or not debug output is on.
   - Fix: switched to GstShark's `interlatency`, which ships with Hailo's `hailotracers` and works.
2. **Real-time playback of a file isn't a valid latency test.**
   - With `--sync` the decoder runs ahead and every 3-buffer queue fills, so "source to sink" reads about 2.2 s. That is queue waiting; only the first frame (141 ms) saw empty queues. A camera doesn't behave like this.
   - Added an opt-in `HCPOSE_PACE_FILE=1` to `hcpose/sources.py`. For file inputs only, it inserts `identity name=live_pace sync=true` after the decoder, so frames are released at the clip's frame rate the way a camera releases them. The sink runs with `--no-sync`. Default behaviour is unchanged; tests: 42 passed.
   - Latency is measured from `live_pace` to the sink.
   - interlatency's own "time since source" value doesn't follow frames through the H.264 decoder: B-frame reordering gave negative differences. `report.py` therefore pairs the n-th `live_pace` record with the n-th sink record by their log (wall-clock) timestamps. Frames are in order, so there are no negatives.
3. **Resumed run.**
   - The system info, idle, Hailo-only and throughput stages are from the first bench.sh run (21:23–21:51); they finished before the hang.
   - The latency runs and both soaks are from a resumed run (`FROM=latency research/edge/bench.sh ...`, a new option, 21:59–22:37).
   - The soaks also got `timeout -k` so a hung run can't block the script.
4. **The report's Hailo-only section** kept hailortcli's first progress lines and dropped its Summary block. `report.py` now prints the Summary.
5. **`hailortcli benchmark` worked for n, s and m**; no fallback was needed. `vcgencmd pmic_read_adc` works on this firmware: `pmic_w` is filled in every CSV.
6. **The latency table's FPS column** comes from the log's "frames in X s", which includes about 4 s of start and stop. The paced runs actually ran at the clip's 25 FPS (1743 frames in 70 s); the column shows 23.4–25.0.
7. **m + tiles can't keep up with a 25 FPS camera** (7.9 FPS). In the paced run frames back up and latency settles at about 2.5 s; the HCPOSE_PACE_FILE identity holds frames back instead of dropping them. A live deployment of m + tiles would need a leaky queue or a lower input rate.
8. **Soak 1** (n + tiles, clip_2 looped 3×, 30 FPS) ended at 31% distracted with "105 seats" and still "calibrating" after 15 min. The loop's jumps back to the start probably create new seats, so read the soak only for load, temperature and power, not for class %.
9. **Soak 2 CPU clock** averaged 1701 MHz (minimum 1600) with throttling `0x0`. The whole Pi was at only 14% CPU, so this is the frequency governor scaling down, not heat.
10. **End-to-end n without tiles** runs at 119 FPS in the bench, against 128–131 in the clip runs and 131.5 on the Hailo alone. The bench runs with `--engagement`, which costs about 2.8 cores of CPU in hcpose.

## Reading the clip results against the earlier FP32 (best.pt, PC) numbers

A realistic class distracted % is low in all three clips.

- **clip_3** (attentive lecture, high front camera, about 35 students):
  - Rules: INT8 m 0.0% (tiles off and on), n 4–6%. FP32 was 0–1%, so m matches and n reads a few points high.
  - Learned classifiers (FP32 → INT8):
    - trees: 60% → 48–65%
    - logistic: 89% → 74–78%
    - MLP: 61–66% → 52–74%
  - The learned classifiers still over-call distracted on this clip by a lot. The INT8 models don't make this worse (for m, a bit better), so it is a classifier/domain-shift issue, not quantization.
  - Students per frame: 16–22, inferred class size 21–26. That's below the ~35 present; tiles add only 2–3 per frame here.
- **clip_2** (study hall, desk-height camera, about 50 students; known weak case for face-hidden):
  - With tiles: rules 35–38% (FP32 44%), trees 24–34% (FP32 37%). Still far above a realistic value, the same failure as FP32 and somewhat lower.
  - Without tiles only about 10 students per frame are found (class size 16–20). Tiles roughly double that to 21–24.
  - The n + tiles class size of 55 matches the ~50 present best; m + tiles gives 37.
- **clip** (exam hall, high camera, about 30 students):
  - Rules: n + tiles 5.2% is inside the FP32 5–8%. n untiled reads 10.7%, m 11–14%: a few points above FP32.
  - Learned classifiers do well with m (6–19%) but badly with untiled n (34–45%); tiles bring n down to 13–24%.
  - Class size 26–33 with tiles, against ~30 present.
- **Duplicates per frame** are ≤ 0.01 everywhere.
- **Overall:** INT8 m reproduces the FP32 rules behaviour closely. INT8 n adds a few points of false distraction without tiles, and tiles fix most of that. clip_2 remains the weak case.

## Edge summary

- **Hailo-8 alone (hailortcli):**

  | Model | FPS | HW latency |
  |---|---|---|
  | n | 131.5 | 5.7 ms |
  | s | 69.5 | 11.9 ms |
  | m | 31.5 | 26.7 ms |

  End-to-end FPS matches these, and tiles (4 inferences per frame) divide them by 4 exactly. The pipeline is bound by the Hailo-8, not the CPU.
- **Paced latency (frame enters → sink, 25 FPS):** median 20 ms for n, 45 ms for n + tiles, 40 ms for m; the 95th percentile is within 8 ms of the median in each case.
- **Pi board power** (PMIC rails; excludes the AI HAT, whose power comes from Hailo's spec): 2.3 W idle and 2.8–5.4 W under load. The deployment setting (n + tiles in real time) drew 4.26 W.
- **Temperatures:** no throttling anywhere. The maximum SoC temperature was 61.1 °C (n at full speed). The 15-minute soaks levelled off at 57–60 °C.

## Clip report (clips_report.md)


## clip_2 (class % after 2:30)

| Model | Tiles | FPS | Students / frame | Duplicates / frame | Class size | Rules | Trees | Logistic | MLP | Range width, rules (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| n | off | 131.1 | 10.3 | 0.00 | 20 | 38.0% | 21.8% | 29.1% | 20.9% | 46 |
| n | on | 32.9 | 23.5 | 0.00 | 55 | 37.9% | 34.0% | 32.7% | 38.9% | 29 |
| m | off | 31.6 | 10.6 | 0.00 | 16 | 28.0% | 7.5% | 12.3% | 8.3% | 43 |
| m | on | 7.9 | 21.0 | 0.00 | 37 | 34.8% | 23.9% | 21.7% | 18.1% | 33 |

## clip_3 (class % after 2:30)

| Model | Tiles | FPS | Students / frame | Duplicates / frame | Class size | Rules | Trees | Logistic | MLP | Range width, rules (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| n | off | 130.5 | 15.6 | 0.00 | 21 | 5.8% | 65.2% | 76.8% | 74.4% | 25 |
| n | on | 32.8 | 17.9 | 0.00 | 25 | 4.4% | 57.8% | 76.2% | 66.3% | 21 |
| m | off | 31.5 | 20.1 | 0.00 | 23 | 0.0% | 58.0% | 77.5% | 56.8% | 15 |
| m | on | 7.9 | 22.0 | 0.00 | 26 | 0.0% | 48.4% | 73.9% | 52.3% | 14 |

## clip (class % whole clip)

| Model | Tiles | FPS | Students / frame | Duplicates / frame | Class size | Rules | Trees | Logistic | MLP | Range width, rules (pts) |
|---|---|---|---|---|---|---|---|---|---|---|
| n | off | 128.0 | 12.6 | 0.01 | 22 | 10.7% | 34.0% | 38.5% | 44.7% | 26 |
| n | on | 32.8 | 18.6 | 0.01 | 33 | 5.2% | 13.3% | 23.9% | 17.3% | 18 |
| m | off | 31.4 | 14.2 | 0.00 | 20 | 14.4% | 12.5% | 16.9% | 12.6% | 32 |
| m | on | 7.9 | 22.1 | 0.00 | 26 | 11.2% | 9.6% | 18.9% | 6.1% | 25 |


## Edge report (edge_report.md)


```
Raspberry Pi 5 Model B Rev 1.1 
Linux raspberrypi 6.18.60+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.18.60-1+rpt1 (2026-11-03) aarch64 GNU/Linux
2026/05/26 16:01:25 
Copyright (c) 2012 Broadcom
version 086b83e3 (release) (embedded)
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8                         
Device Architecture: HAILO8
```

## Idle (60 s, nothing running)

CPU 1.5%, SoC 50.1 °C max, Pi board rails (PMIC) 2.33 W, throttling: none.

## Hailo-8 only (hailortcli)

**m**

```
FPS     (hw_only)                 = 31.5316
        (streaming)               = 31.5316
Latency (hw)                      = 26.6938 ms
```

**n**

```
FPS     (hw_only)                 = 131.546
        (streaming)               = 131.465
Latency (hw)                      = 5.71612 ms
```

**s**

```
FPS     (hw_only)                 = 69.4508
        (streaming)               = 69.4503
Latency (hw)                      = 11.8665 ms
```

## End to end (clip, as fast as possible)

| Model | Tiles | FPS | CPU, whole Pi | CPU, hcpose (% of a core) | Memory (MB) | Max SoC temp (°C) | Pi board rails (W) | Throttling |
|---|---|---|---|---|---|---|---|---|
| n | off | 119.1 | 79% | 279% | 332 | 61.1 | 5.42 | none |
| n | on | 32.8 | 46% | 165% | 286 | 60.6 | 4.84 | none |
| s | off | 69.3 | 47% | 168% | 329 | 60.1 | 4.77 | none |
| s | on | 17.4 | 26% | 96% | 294 | 59.5 | 3.87 | none |
| m | off | 31.5 | 23% | 80% | 339 | 58.4 | 3.19 | none |
| m | on | 7.9 | 14% | 49% | 307 | 57.3 | 2.82 | none |

## Pipeline latency (clip paced at its frame rate like a camera; GstShark interlatency, live_pace to sink)

| Model | Tiles | Frames | Median (ms) | 95th percentile (ms) | FPS |
|---|---|---|---|---|---|
| n | off | 1743 | 20 | 23 | 23.4 |
| n | on | 1743 | 45 | 53 | 25.0 |
| m | off | 1742 | 40 | 47 | 23.6 |
| m | on | 571 | 2504 | 2510 | 7.9 |

When a setting runs below the clip's frame rate (m with tiles), frames queue up and the latency includes that wait.

## 15-minute soak: n + tiles, real time (deployment setting)

- FPS: 30.0
- SoC temperature: 58.4 °C after 10 s, 56.8 after 5 min, 57.9 after 10 min, 56.8 after 15 min (max 60.1)
- CPU clock: mean 2388 MHz, minimum 1900 MHz
- Throttling flags seen: none
- CPU: 44% of the whole Pi; Pi board rails 4.26 W

## 15-minute soak: m + tiles, as fast as possible (worst case)

- FPS: 7.9
- SoC temperature: 50.1 °C after 10 s, 53.5 after 5 min, 55.6 after 10 min, 54.6 after 15 min (max 56.2)
- CPU clock: mean 1701 MHz, minimum 1600 MHz
- Throttling flags seen: none
- CPU: 14% of the whole Pi; Pi board rails 2.70 W

Pi board rails = sum of V x I over the Pi 5 PMIC rails (`vcgencmd pmic_read_adc`): the board's own supplies, without the AI HAT and USB devices. Hailo-8 power comes from the vendor's specification.

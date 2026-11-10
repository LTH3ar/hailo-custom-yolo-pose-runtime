"""Sample the Pi's load once a second into a CSV, until stopped (Ctrl+C / SIGTERM) or --duration.

columns: t (s since start), cpu_total (%, all cores = 100), cpu_hcpose (% of one core, summed
over processes whose command line contains --match), rss_mb (same processes), temp_c (SoC),
freq_mhz (cpu0), throttled (vcgencmd get_throttled, hex), pmic_w (sum of V x I over the Pi 5
PMIC rails: the board's own supplies, NOT the AI HAT or USB devices; empty if unavailable)

usage: python research/edge/monitor.py out.csv [--duration 60] [--match hcpose]
"""
import argparse
import csv
import re
import signal
import subprocess
import time

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until stopped)")
ap.add_argument("--match", default="hcpose")
ap.add_argument("--interval", type=float, default=1.0)
args = ap.parse_args()

stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def read(path, scale=1.0):
    try:
        return int(open(path).read().strip()) * scale
    except (OSError, ValueError):
        return ""


def vcgencmd(*a):
    try:
        return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def pmic_watts():
    out = vcgencmd("pmic_read_adc")
    amps, volts = {}, {}
    for name, kind, val in re.findall(r"(\S+)_([AV])\s+\w+\(\d+\)=([\d.]+)", out):
        (amps if kind == "A" else volts)[name] = float(val)
    common = set(amps) & set(volts)
    return round(sum(amps[k] * volts[k] for k in common), 3) if common else ""


procs = {}


def tracked():
    alive = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if args.match in cmd and "monitor.py" not in cmd:
            if p.pid not in procs:
                procs[p.pid] = p
                p.cpu_percent(None)  # prime
            alive.append(procs[p.pid])
    return alive


psutil.cpu_percent(None)
t0 = time.monotonic()
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "cpu_total", "cpu_hcpose", "rss_mb", "temp_c", "freq_mhz", "throttled", "pmic_w"])
    while not stop and (not args.duration or time.monotonic() - t0 < args.duration):
        time.sleep(args.interval)
        cpu_p = rss = 0.0
        for p in tracked():
            try:
                cpu_p += p.cpu_percent(None)
                rss += p.memory_info().rss / 2**20
            except psutil.Error:
                pass
        thr = vcgencmd("get_throttled").strip().split("=")[-1]
        w.writerow([round(time.monotonic() - t0, 1), psutil.cpu_percent(None), round(cpu_p, 1), round(rss),
                    read("/sys/class/thermal/thermal_zone0/temp", 1e-3), read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1e-3),
                    thr, pmic_watts()])
        f.flush()

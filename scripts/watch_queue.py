#!/usr/bin/env python3
"""
Live progress of the long-run queue (jobs/long_runs.sh).

Shows the step that is running with its own progress bar, rate and ETA, an
overall bar for the whole queue, and the latest machine health. Read-only:
it only reads files the queue writes, so it is safe to run any time.
Ctrl-C to stop watching (the queue keeps running).

Usage:
    python3 scripts/watch_queue.py
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

OUT = Path("outputs/long_runs")
# Queue steps with rough durations in hours (measured on the RTX 5070 Ti laptop),
# used to weight the overall bar.
STEPS = [
    ("phase2_ema", 1.0), ("phase2_sigreg", 1.7), ("probe_ema", 0.02),
    ("probe_sigreg", 0.02), ("select_phase2", 0.01), ("dreamer_resume", 3.5),
]
DREAMER_LOGDIR = Path("outputs/logs/cnn_bev_seed42")
DREAMER_TARGET = 150_000
BATCH = 64  # Phase 2 clips per step (configs/phase2_jepa_laptop.yaml)


def bar(fraction: float, width: int) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    full = int(fraction * width)
    return "█" * full + "░" * (width - full)


def fmt(seconds) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "--"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def queue_state() -> dict:
    """State per step (START / OK / FAIL / SKIP) since the latest queue start."""
    state = {}
    path = OUT / "status.txt"
    if path.is_file():
        lines = path.read_text().splitlines()
        starts = [i for i, line in enumerate(lines) if "long-run queue started" in line]
        for line in lines[starts[-1] if starts else 0:]:
            m = re.match(r"\S+ \S+\s+(START|OK|FAIL|SKIP)\s+(\S+)", line)
            if m:
                state[m.group(2)] = m.group(1)
    return state


def phase2_progress(step: str):
    """(done steps, total, steps/s, seconds since last log line) from the trainer log."""
    log = OUT / f"{step}.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="ignore")
    total = re.findall(r"Steps: (\d+)", text)
    if not total:
        return None
    done = [int(s) for s in re.findall(r"step (\d+)/\d+", text)]
    resumed = re.findall(r"at step (\d+)", text)
    done_step = max(done + [int(r) for r in resumed] + [0])
    rates = re.findall(r"([\d.]+) clips/s", text)
    rate = float(rates[-1]) / BATCH if rates else None
    return done_step, int(total[-1]), rate, time.time() - log.stat().st_mtime


def dreamer_progress(history: list):
    path = DREAMER_LOGDIR / "episodes.jsonl"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 16384))
        lines = f.read().decode(errors="ignore").splitlines()[1:]
    steps = [json.loads(line)["agent_step"] for line in lines if line.startswith("{")]
    if not steps:
        return None
    step, now = max(steps), time.time()
    history.append((now, step))
    while len(history) > 2 and now - history[0][0] > 600:
        history.pop(0)
    rate = None
    if len(history) >= 2 and history[-1][1] > history[0][1]:
        rate = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
    return step, DREAMER_TARGET, rate


def health() -> str:
    path = OUT / "health.log"
    if not path.is_file():
        return "health: --"
    last = path.read_text().splitlines()[-1]
    m = re.search(r"gpu\[temp,MiB,util,W\]=([\d.]+),(\d+),(\d+),([\d.]+) cpu=(\S+) "
                  r"ram_avail=(\S+) disk_free=(\S+)", last)
    if not m:
        return "health: " + last
    t, mib, util, watts, cpu, ram, disk = m.groups()
    return (f"GPU {t}°C {util}% {int(mib) / 1024:.1f} GB {float(watts):.0f} W | CPU {cpu} | "
            f"RAM free {ram} | disk free {disk}  (as of {last.split()[0]})")


def render(dreamer_history: list) -> list:
    width = max(10, min(40, shutil.get_terminal_size().columns - 70))
    state = queue_state()
    if not state:
        return ["Queue not started (no outputs/long_runs/status.txt)."]

    total_h = sum(h for _, h in STEPS)
    done_h, current, line, eta = 0.0, None, "", None
    for name, hours in STEPS:
        s = state.get(name)
        if s in ("OK", "FAIL", "SKIP"):
            done_h += hours
        elif s == "START" and current is None:
            current = name
            frac, rate_txt = 0.0, ""
            if name.startswith("phase2"):
                p = phase2_progress(name)
                if p:
                    done, total, rate, age = p
                    est = done + (rate * age if rate else 0)  # smooth between epoch lines
                    est = min(est, total)
                    frac = est / total
                    eta = (total - est) / rate if rate else None
                    rate_txt = (f"{int(est):>6,}/{total:,} steps  {rate or 0:4.1f} st/s  "
                                f"ETA {fmt(eta)}")
            elif name == "dreamer_resume":
                p = dreamer_progress(dreamer_history)
                if p:
                    step, total, rate = p
                    frac = step / total
                    eta = (total - step) / rate if rate else None
                    rate_txt = (f"{step:>7,}/{total:,} env steps  {rate or 0:4.1f} st/s  "
                                f"ETA {fmt(eta)}")
            else:
                rate_txt = "running (a few minutes)"
            done_h += hours * frac
            line = f"{name:15s} [{bar(frac, width)}] {frac:6.1%}  {rate_txt}"

    finished = all(state.get(n) in ("OK", "FAIL", "SKIP") for n, _ in STEPS)
    overall = done_h / total_h
    lines = []
    if finished:
        lines.append("Queue finished: " + ", ".join(f"{n} {state.get(n, '-')}" for n, _ in STEPS))
    else:
        lines.append(line or "between steps...")
    rest_h = sum(h for n, h in STEPS if state.get(n) not in ("OK", "FAIL", "SKIP", "START"))
    total_eta = (eta or 0) + rest_h * 3600 if not finished else 0
    lines.append(f"{'whole queue':15s} [{bar(overall, width)}] {overall:6.1%}  "
                 f"steps done {sum(state.get(n) in ('OK', 'FAIL', 'SKIP') for n, _ in STEPS)}"
                 f"/{len(STEPS)}  ETA ~{fmt(total_eta)}")
    failed = [n for n, _ in STEPS if state.get(n) == "FAIL"]
    if failed:
        lines.append("FAILED: " + ", ".join(failed) + f"  (see {OUT}/<step>.log)")
    if (OUT / "STOP").exists():
        lines.append("STOP requested: " + (OUT / "STOP").read_text().strip())
    lines.append(health())
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh seconds")
    args = parser.parse_args()
    history: list = []
    printed = 0
    try:
        while True:
            lines = render(history)
            if printed:
                sys.stdout.write(f"\033[{printed}F")  # back to the first line we printed
            for line in lines:
                sys.stdout.write("\033[K" + line + "\n")
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            printed = len(lines)
            if lines[0].startswith("Queue finished"):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

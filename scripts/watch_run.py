#!/usr/bin/env python3
"""
Live progress bar for a Dreamer run, read from its log directory.

Only reads episodes.jsonl / eval.jsonl / metrics.json, so it can run in a
second terminal without touching training. Ctrl-C to stop.

Usage:
    python scripts/watch_run.py                      # most recently active run
    python scripts/watch_run.py outputs/logs/cnn_bev_seed42 --target 100000
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def last_json_lines(path: Path, n: int = 1) -> list:
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64 * 1024))
        lines = f.read().decode(errors="ignore").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def render(logdir: Path, target: int, history: list) -> str:
    episodes = last_json_lines(logdir / "episodes.jsonl", 20)
    evals = last_json_lines(logdir / "eval.jsonl", 1)
    if (logdir / "metrics.json").is_file():
        m = json.loads((logdir / "metrics.json").read_text())
        final = m.get("final_eval") or {}
        return (f"DONE  {m.get('env_steps')} steps, {m.get('updates')} updates, "
                f"final eval success {final.get('success_rate', 0):.0%}, "
                f"return {final.get('mean_reward', 0):.1f}, "
                f"wall clock {fmt_duration(m.get('wall_clock_sec', 0))}")
    if not episodes:
        return "waiting for the first finished episode..."

    step = max(e["agent_step"] for e in episodes)
    now = time.time()
    history.append((now, step))
    while len(history) > 2 and now - history[0][0] > 300:  # 5-minute rate window
        history.pop(0)
    rate = 0.0
    if len(history) >= 2 and history[-1][1] > history[0][1]:
        rate = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
    eta = fmt_duration((target - step) / rate) if rate > 0 else "--"

    width = max(10, min(40, shutil.get_terminal_size().columns - 90))
    frac = min(step / target, 1.0)
    bar = "█" * int(frac * width) + "░" * (width - int(frac * width))
    mode = "eval " if episodes[-1]["mode"] == "eval" else "train"
    train = [e for e in episodes if e["mode"] == "train"][-10:]
    recent = (sum(e["return"] for e in train) / len(train)) if train else float("nan")
    line = (f"{mode} [{bar}] {frac:6.1%} {step:>7,}/{target:,}  "
            f"{rate:5.1f} st/s  ETA {eta:>6}  train ret(10) {recent:6.1f}")
    if evals:
        e = evals[-1]
        line += (f"  | eval@{e['agent_step'] // 1000}k ret {e['mean_reward']:.0f} "
                 f"route {e.get('mean_route_completion', 0):.0%} "
                 f"succ {e['success_rate']:.0%} coll {e['collision_rate']:.0%}")
    return line


def health_line() -> str:
    """Latest line of the most recent queue health log, if written in the last 2 min."""
    logs = sorted(Path("outputs").glob("*/health.log"), key=lambda p: p.stat().st_mtime)
    if not logs or time.time() - logs[-1].stat().st_mtime > 120:
        return ""
    last = logs[-1].read_text().splitlines()[-1]
    gpu = last.split("gpu[temp,MiB,util,W]=")[-1].split()[0].split(",")
    return f"  | GPU {gpu[0]}°C {gpu[2]}%" if len(gpu) >= 3 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logdir", type=Path, nargs="?", default=None,
                        help="Run log directory (default: the most recently active one)")
    parser.add_argument("--target", type=int, default=100_000, help="Env-step budget of the run")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh seconds")
    args = parser.parse_args()

    if args.logdir is None:
        runs = sorted(Path("outputs").glob("**/episodes.jsonl"), key=lambda p: p.stat().st_mtime)
        if not runs:
            parser.error("no run found under outputs/; pass the log directory")
        args.logdir = runs[-1].parent
    print(f"Watching {args.logdir} (target {args.target:,} steps)")

    history: list = []
    try:
        while True:
            line = render(args.logdir, args.target, history)
            if not line.startswith("DONE"):
                line += health_line()
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
            if line.startswith("DONE"):
                print()
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())

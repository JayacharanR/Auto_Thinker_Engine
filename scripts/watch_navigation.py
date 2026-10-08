#!/usr/bin/env python3
"""Read-only progress for bounded navigation runs; Ctrl-C only stops watching."""

import argparse
import json
import time
from pathlib import Path


def render(output):
    path = Path(output) / "progress.json"
    if not path.exists():
        return "Waiting for progress.json", False
    p = json.loads(path.read_text())
    percent = min(100, max(0, p.get("route_progress_percent", 0)))
    target = p.get("target_frames", 0)
    fraction = min(1, p.get("frames", 0) / target) if target else percent / 100
    filled = int(fraction * 20)
    age = max(0, time.time() - p.get("updated_at", 0))
    bar = "█" * filled + "░" * (20 - filled)
    text = (
        f"{p['status']:16} [{bar}] {'profile' if target else 'route'} {fraction * 100:5.1f}% | "
        f"route {percent:5.1f}% | "
        f"{p.get('frames', 0):,} frames | {p.get('frames_per_second', 0):.1f} fps | "
        f"{p.get('speed_kmh', 0):.1f} km/h | update {age:.0f}s ago"
    )
    if p.get("error"):
        text += f"\n{p['error']}"
    if p.get("summary"):
        s = p["summary"]
        text += (
            f"\nSafe completed: {s['safe_completed']}/{s['count']}; "
            f"all collision events: {s['all_collisions']}; "
            f"stop reason: {s.get('stop_reason')}"
        )
    return text, p["status"] not in ("starting", "running")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/navigation_m0")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            text, done = render(args.out)
            print(text if args.once else "\033[2J\033[H" + text, flush=True)
            if args.once or done:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

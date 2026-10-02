#!/usr/bin/env python3
"""
Inspect what an agent saw and did in recent episodes of a run.

For each episode: a strip of frames (evenly spaced, plus the last frames
before the episode ended) and an action summary. Discrete actions are decoded
with CarDreamer's table (acceleration = acc[a // n_steer], steering =
steer[a % n_steer]; positive steering turns LEFT, because CarDreamer negates
it for CARLA). Continuous actions are summarised per dimension.

Usage:
    python scripts/inspect_episode.py outputs/logs/cnn_bev_seed42            # last 3 train episodes
    python scripts/inspect_episode.py outputs/logs/cnn_bev_seed42 --eval -n 1
Writes <logdir>/inspect/<episode>.png and prints the action summary.
"""

import argparse
import collections
from pathlib import Path

import cv2
import numpy as np

# carla_right_turn_simple (CarDreamer configs: common discrete_acc, task discrete_steer)
DEFAULT_ACC = [-2.0, 0.0, 2.0]
DEFAULT_STEER = [-0.9, -0.3, 0.0, 0.3, 0.9]


def frame_strip(images: np.ndarray, size: int = 128) -> np.ndarray:
    n = len(images)
    idx = sorted(set([0, n // 4, n // 2, 3 * n // 4, max(0, n - 6), n - 1]))
    frames = [cv2.resize(images[i], (size, size), interpolation=cv2.INTER_NEAREST) for i in idx]
    for frame, i in zip(frames, idx):
        cv2.putText(frame, str(i), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.concatenate(frames, axis=1)


def summarize_actions(actions: np.ndarray, acc_values, steer_values) -> list[str]:
    if actions.ndim == 2 and actions.shape[1] == len(acc_values) * len(steer_values) \
            and np.allclose(actions.sum(1), 1.0):
        counts = collections.Counter(np.argmax(actions, 1).tolist())
        n_steer, total = len(steer_values), len(actions)
        lines = []
        for a, c in counts.most_common():
            acc, steer = acc_values[a // n_steer], steer_values[a % n_steer]
            turn = "straight" if steer == 0 else ("left" if steer > 0 else "right")
            lines.append(
                f"  #{a:2d} acc {acc:+.1f} steer {steer:+.1f} ({turn:8s}) {c / total:6.1%}"
            )
        steer_share = collections.Counter()
        for a, c in counts.items():
            s = steer_values[a % n_steer]
            steer_share["straight" if s == 0 else ("left" if s > 0 else "right")] += c
        lines.append("  steering: " + ", ".join(
            f"{k} {v / total:.0%}" for k, v in sorted(steer_share.items())))
        return lines
    dims = actions.shape[-1]
    names = ["acc", "steer"] if dims == 2 else [str(i) for i in range(dims)]
    return [f"  {n}: mean {actions[:, i].mean():+.2f} std {actions[:, i].std():.2f} "
            f"min {actions[:, i].min():+.2f} max {actions[:, i].max():+.2f}"
            for i, n in enumerate(names)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--eval", action="store_true", help="Eval episodes instead of train")
    parser.add_argument("-n", type=int, default=3, help="Number of most recent episodes")
    parser.add_argument("--acc", type=float, nargs="+", default=DEFAULT_ACC)
    parser.add_argument("--steer", type=float, nargs="+", default=DEFAULT_STEER)
    args = parser.parse_args()

    folder = args.logdir / ("eval_eps" if args.eval else "train_eps")
    files = sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime)[-args.n:]
    if not files:
        print(f"No episodes in {folder}")
        return 1
    out = args.logdir / "inspect"
    out.mkdir(exist_ok=True)
    for f in files:
        data = np.load(f)
        images, actions = data["image"], data["action"]
        rewards = data["reward"]
        print(f"{f.name}: length {len(images) - 1}, return {rewards.sum():.1f}, "
              f"final discount {float(data['discount'][-1]):.0f}")
        if "route" in data:
            r = data["route"][-1]
            print(f"  last route: next waypoint fwd {r[0] * 20:.1f} m left {r[1] * 20:.1f} m, "
                  f"speed {r[-3] * 10:.1f} m/s")
        for line in summarize_actions(actions[1:], args.acc, args.steer):  # [0] is the reset step
            print(line)
        png = out / f"{f.stem}.png"
        cv2.imwrite(str(png), cv2.cvtColor(frame_strip(images), cv2.COLOR_RGB2BGR))
        print(f"  frames -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

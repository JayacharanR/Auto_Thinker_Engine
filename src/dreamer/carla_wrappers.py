"""
Gym wrappers between CarDreamer tasks and dreamerv3-torch.

CarDreamer follows the legacy Gym API: ``reset() -> obs`` and
``step() -> (obs, reward, done, info)``. Its terminal conditions arrive in
``info`` as one-element bool arrays (``is_collision``, ``out_of_lane``,
``destination_reached``, ``time_exceeded``).

- ``DreamerObservation`` selects the observation contract seen by the agent
  and sets the episode-boundary fields and ``discount`` that dreamerv3-torch
  stores in replay.
- ``EpisodeMetricsRecorder`` writes one JSON line per finished episode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import gym
import numpy as np
import torch

from src.eval.metrics import MetricsTracker

# Observation contracts: the CarDreamer key that becomes Dreamer's ``image``,
# and whether a ``route`` vector is added.
OBS_CONTRACTS = {
    "bev": ("birdeye_wpt", False),  # bird's-eye view with the route drawn in
    "camera": ("camera", False),  # front RGB camera only (no route information)
    "camera_route": ("camera", True),  # front camera + route/speed vector
}
# Conditions that end the episode for real. ``time_exceeded`` is a truncation:
# the episode is cut off, so the value after it is not zero.
TERMINAL_CONDITIONS = ("is_collision", "out_of_lane", "destination_reached")
ROUTE_WAYPOINTS = 10
# CarDreamer's planner spaces waypoints 0.8 m apart; every 4th gives ~30 m lookahead.
ROUTE_STRIDE = 4
ROUTE_SCALE_M = 20.0  # waypoint offsets are divided by this
SPEED_SCALE_MPS = 10.0


def _flag(info: dict, key: str) -> bool:
    return bool(np.asarray(info.get(key, False)).reshape(-1)[0])


def route_features(
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
    speed_mps: float,
    waypoints,
    num_waypoints: int = ROUTE_WAYPOINTS,
    stride: int = ROUTE_STRIDE,
) -> np.ndarray:
    """
    The planned route in the ego frame, plus speed and heading error.

    Layout (float32, ``2 * num_waypoints + 3``): for every ``stride``-th upcoming waypoint,
    ``(forward, left)`` offsets in metres / ROUTE_SCALE_M (the last waypoint
    repeats when fewer remain); then speed / SPEED_SCALE_MPS; then the sine and
    cosine of the angle from the ego heading to the first waypoint's heading.
    ``waypoints`` are CarDreamer's world-frame ``(x, y, yaw_deg)``.
    """
    yaw = np.deg2rad(ego_yaw_deg)
    cos, sin = np.cos(yaw), np.sin(yaw)
    points = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)[::stride][:num_waypoints]
    if len(points) == 0:
        points = np.array([[ego_x, ego_y, ego_yaw_deg]])
    points = np.concatenate([points, np.repeat(points[-1:], num_waypoints - len(points), 0)])
    dx, dy = points[:, 0] - ego_x, points[:, 1] - ego_y
    # CARLA is left-handed (y points right of x), so "left" is -y in the ego frame.
    forward = cos * dx + sin * dy
    left = -(-sin * dx + cos * dy)
    heading_error = np.deg2rad(points[0, 2]) - yaw
    return np.concatenate([
        np.stack([forward, left], 1).reshape(-1) / ROUTE_SCALE_M,
        [speed_mps / SPEED_SCALE_MPS, np.sin(heading_error), np.cos(heading_error)],
    ]).astype(np.float32)


def carla_ego_state(env) -> tuple:
    """``(x, y, yaw_deg, speed_mps, waypoints)`` from a CarDreamer waypoint task."""
    task = env.unwrapped
    transform = task.get_ego_vehicle().get_transform()
    velocity = task.get_ego_vehicle().get_velocity()
    speed = float(np.hypot(velocity.x, velocity.y))
    return (transform.location.x, transform.location.y, transform.rotation.yaw, speed,
            getattr(task, "waypoints", []))


class DreamerObservation(gym.Wrapper):
    """Build the observation contract Dreamer sees from a CarDreamer task.

    Keeps ``image``, optionally ``route`` (see ``route_features``) and ``feat``
    (a frozen encoder's embedding of the raw camera frame, computed once per
    step by ``feature_extractor``), plus the episode-boundary fields. Replay
    stores nothing else. Metrics read events from ``info``.

    Args:
        obs: One of OBS_CONTRACTS.
        size: Size of the stored ``image``.
        feature_extractor: Optional object with ``dim``, ``reset()`` and
            ``__call__(frame_hwc_uint8) -> np.ndarray`` (frozen encoder arms).
        ego_state_fn: Returns ``(x, y, yaw_deg, speed, waypoints)``; defaults to
            reading the CarDreamer ego vehicle.
    """

    def __init__(self, env, obs: str = "bev", size: tuple = (64, 64),
                 feature_extractor=None, ego_state_fn=carla_ego_state):
        super().__init__(env)
        if obs not in OBS_CONTRACTS:
            raise ValueError(f"obs must be one of {sorted(OBS_CONTRACTS)}, got {obs!r}")
        self._source, self._route = OBS_CONTRACTS[obs]
        spaces = self.env.observation_space.spaces
        if self._source not in spaces:
            raise ValueError(
                f"obs={obs!r} needs the CarDreamer observation {self._source!r}; "
                f"task provides {sorted(spaces)}"
            )
        self._size = tuple(int(s) for s in size)
        self._features = feature_extractor
        self._ego_state = ego_state_fn
        channels = spaces[self._source].shape[-1]
        boundary = gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8)
        observation = {
            "image": gym.spaces.Box(0, 255, self._size + (channels,), dtype=np.uint8),
            "is_first": boundary,
            "is_last": boundary,
            "is_terminal": boundary,
        }
        if self._route:
            observation["route"] = gym.spaces.Box(
                -np.inf, np.inf, (2 * ROUTE_WAYPOINTS + 3,), dtype=np.float32
            )
        if feature_extractor is not None:
            observation["feat"] = gym.spaces.Box(
                -np.inf, np.inf, (feature_extractor.dim,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(observation)

    def _observation(self, raw: dict, is_first: bool, is_last: bool, is_terminal: bool) -> dict:
        self.last_raw = raw  # full CarDreamer observation (camera, bird's-eye view) for videos
        frame = np.asarray(raw[self._source], dtype=np.uint8)
        image = frame
        if image.shape[:2] != self._size:
            image = cv2.resize(image, self._size[::-1], interpolation=cv2.INTER_AREA)
        obs = {
            "image": image,
            "is_first": np.uint8(is_first),
            "is_last": np.uint8(is_last),
            "is_terminal": np.uint8(is_terminal),
        }
        if self._route:
            x, y, yaw, speed, waypoints = self._ego_state(self.env)
            obs["route"] = route_features(x, y, yaw, speed, waypoints)
        if self._features is not None:
            obs["feat"] = np.asarray(self._features(frame), dtype=np.float32)
        return obs

    def reset(self):
        raw = self.env.reset()
        if self._features is not None:
            self._features.reset()
        return self._observation(raw, True, False, False)

    def step(self, action):
        raw, reward, done, info = self.env.step(action)
        info = dict(info)
        is_terminal = bool(done) and any(_flag(info, k) for k in TERMINAL_CONDITIONS)
        info["discount"] = np.array(0.0 if is_terminal else 1.0, dtype=np.float32)
        return self._observation(raw, False, bool(done), is_terminal), reward, done, info


class EpisodeMetricsRecorder(gym.Wrapper):
    """Append one JSON record per finished episode to ``path``.

    Train and eval share one CARLA environment, so the runner sets ``mode``
    before each phase and points ``step_fn`` at the agent's step counter.
    Episodes cut short by a reset are not recorded.

    ``route_completion`` is the fraction of the planned route's waypoints
    passed (CarDreamer's per-step ``num_completed + num_obsolete`` over the
    route length at reset); 1.0 when the destination is reached. Tasks whose
    planner does not load the whole route at reset report 0.
    """

    def __init__(self, env, path: Path | str):
        super().__init__(env)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = "train"
        self.step_fn = lambda: 0
        self.trackers = {"train": MetricsTracker("train"), "eval": MetricsTracker("eval")}
        self._started = 0.0
        self._episode_index = 0
        self._route_total = 0
        self._route_passed = 0

    def _route_length(self) -> int:
        get_planner = getattr(self.env.unwrapped, "get_ego_planner", None)
        planner = get_planner() if get_planner else None
        return int(planner.get_waypoint_num()) if hasattr(planner, "get_waypoint_num") else 0

    def reset(self, **kwargs):
        self.trackers[self.mode].start_episode(self._episode_index)
        self._started = time.time()
        obs = self.env.reset(**kwargs)
        self._route_total = self._route_length()
        self._route_passed = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._route_passed += int(info.get("num_completed", 0)) + int(info.get("num_obsolete", 0))
        tracker = self.trackers[self.mode]
        tracker.step({
            "reward/total": float(reward),
            "collision": _flag(info, "is_collision"),
            "wpt_dis": float(info.get("wpt_dis", 0.0)),
            "speed_norm": float(info.get("speed_norm", 0.0)),
            "ttc": float(info.get("ttc", float("inf"))),
        })
        if done:
            self._finish(tracker, info)
        return obs, reward, done, info

    def _finish(self, tracker: MetricsTracker, info: dict) -> None:
        flags = {k: _flag(info, k) for k in (*TERMINAL_CONDITIONS, "time_exceeded")}
        reason = next((k for k, v in flags.items() if v), "time_limit")
        completion = 1.0 if flags["destination_reached"] else (
            self._route_passed / self._route_total if self._route_total else 0.0
        )
        episode = tracker.end_episode(
            success=flags["destination_reached"],
            route_completion=completion,
            termination_reason=reason,
            collision=flags["is_collision"],
            out_of_lane=flags["out_of_lane"],
            time_exceeded=flags["time_exceeded"] or reason == "time_limit",
        )
        record = {
            "mode": self.mode,
            "episode": self._episode_index,
            "agent_step": int(self.step_fn()),
            "return": round(episode.total_reward, 4),
            "length": episode.steps,
            "termination": reason,
            "route_completion": round(episode.route_completion, 4),
            "success": episode.success,
            "collision": episode.collision,
            "out_of_lane": episode.out_of_lane,
            "time_exceeded": episode.time_exceeded,
            "avg_speed_kmh": round(episode.avg_speed_kmh, 2),
            "wall_clock_sec": round(time.time() - self._started, 2),
            "time": time.time(),
        }
        if torch.cuda.is_available():
            record["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._episode_index += 1

    def reset_eval(self) -> MetricsTracker:
        """Start a fresh eval tracker and return the previous one."""
        previous = self.trackers["eval"]
        self.trackers["eval"] = MetricsTracker("eval")
        return previous

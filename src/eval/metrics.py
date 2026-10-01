"""
Driving metrics for evaluation and auditing (P0 Contract).

EPDMS-inspired metrics for Phase 1 (DreamerV3 baseline) and Phase 3
(three-way comparison). Tracks success rate, collision rate, route completion,
safety supervisor interventions, reward statistics, and exports auditable
metrics.json and comparison tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected for a single evaluation episode."""

    episode_id: int = 0
    success: bool = False
    collision: bool = False
    out_of_lane: bool = False
    time_exceeded: bool = False
    total_reward: float = 0.0
    steps: int = 0
    route_completion: float = 0.0
    num_traffic_violations: int = 0
    avg_lateral_deviation: float = 0.0
    max_lateral_deviation: float = 0.0
    min_ttc: float = float("inf")
    avg_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    safety_interventions: int = 0
    termination_reason: str = "running"

    def to_dict(self) -> dict:
        d = asdict(self)
        if np.isinf(d["min_ttc"]):
            d["min_ttc"] = None
        return d


class MetricsTracker:
    """
    Tracks and aggregates driving metrics across evaluation episodes.

    Used for Phase 1 baseline evaluation, Phase 3 encoder comparisons,
    and runtime auditing under the P0 schema.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.episodes: List[EpisodeMetrics] = []
        self._current_episode: Optional[EpisodeMetrics] = None
        self._step_deviations: List[float] = []
        self._step_speeds: List[float] = []
        self._system_metrics: Dict[str, Any] = {}

    def set_system_metrics(
        self,
        steps_per_sec: Optional[float] = None,
        peak_vram_mb: Optional[float] = None,
        total_duration_sec: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Record system performance telemetry."""
        if steps_per_sec is not None:
            self._system_metrics["steps_per_sec"] = round(steps_per_sec, 2)
        if peak_vram_mb is not None:
            self._system_metrics["peak_vram_mb"] = round(peak_vram_mb, 2)
        if total_duration_sec is not None:
            self._system_metrics["total_duration_sec"] = round(total_duration_sec, 2)
        self._system_metrics.update(kwargs)

    def start_episode(self, episode_id: Optional[int] = None) -> None:
        """Start tracking a new episode."""
        ep_id = episode_id if episode_id is not None else len(self.episodes) + 1
        self._current_episode = EpisodeMetrics(episode_id=ep_id)
        self._step_deviations = []
        self._step_speeds = []

    def step(self, reward_info: dict[str, Any]) -> None:
        """Record metrics from a single step."""
        if self._current_episode is None:
            return

        self._current_episode.steps += 1
        self._current_episode.total_reward += float(reward_info.get("reward/total", 0.0))

        if reward_info.get("reward/collision", 0.0) > 0 or reward_info.get("collision", False):
            self._current_episode.collision = True

        deviation = abs(float(reward_info.get("reward/lane_keeping", reward_info.get("wpt_dis", 0.0))))
        self._step_deviations.append(deviation)
        self._current_episode.max_lateral_deviation = max(
            self._current_episode.max_lateral_deviation, deviation
        )

        speed_norm = float(reward_info.get("speed_norm", reward_info.get("speed", 0.0)))
        speed_kmh = speed_norm * 3.6  # convert m/s to km/h
        self._step_speeds.append(speed_kmh)
        self._current_episode.max_speed_kmh = max(
            self._current_episode.max_speed_kmh, speed_kmh
        )

        ttc_info = reward_info.get("reward/time_to_collision", reward_info.get("ttc", float("inf")))
        if isinstance(ttc_info, (int, float)) and ttc_info > 0:
            self._current_episode.min_ttc = min(
                self._current_episode.min_ttc, float(ttc_info)
            )

        if reward_info.get("safety_intervention", False):
            self._current_episode.safety_interventions += 1

    def end_episode(
        self,
        success: bool = False,
        route_completion: float = 0.0,
        termination_reason: str = "done",
        collision: Optional[bool] = None,
        out_of_lane: bool = False,
        time_exceeded: bool = False,
    ) -> EpisodeMetrics:
        """End the current episode and record it."""
        if self._current_episode is None:
            return EpisodeMetrics()

        self._current_episode.success = bool(success)
        self._current_episode.route_completion = float(np.clip(route_completion, 0.0, 1.0))
        self._current_episode.termination_reason = str(termination_reason)
        self._current_episode.out_of_lane = bool(out_of_lane)
        self._current_episode.time_exceeded = bool(time_exceeded)

        if collision is not None:
            self._current_episode.collision = bool(collision)

        if self._step_deviations:
            self._current_episode.avg_lateral_deviation = float(
                np.mean(self._step_deviations)
            )
        if self._step_speeds:
            self._current_episode.avg_speed_kmh = float(
                np.mean(self._step_speeds)
            )

        self.episodes.append(self._current_episode)
        result = self._current_episode
        self._current_episode = None
        return result

    @property
    def success_rate(self) -> float:
        """Fraction of episodes that succeeded."""
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def collision_rate(self) -> float:
        """Fraction of episodes with at least one collision."""
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.collision) / len(self.episodes)

    @property
    def out_of_lane_rate(self) -> float:
        """Fraction of episodes with out-of-lane termination."""
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.out_of_lane) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        """Mean total reward across episodes."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.total_reward for e in self.episodes]))

    @property
    def mean_route_completion(self) -> float:
        """Mean route completion fraction."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.route_completion for e in self.episodes]))

    @property
    def mean_speed_kmh(self) -> float:
        """Mean vehicle speed across episodes in km/h."""
        if not self.episodes:
            return 0.0
        return float(np.mean([e.avg_speed_kmh for e in self.episodes]))

    @property
    def total_safety_interventions(self) -> int:
        """Total number of safety supervisor interventions across all episodes."""
        return sum(e.safety_interventions for e in self.episodes)

    def summary(self) -> dict[str, float]:
        """Get summary statistics as a flat dict (preserves legacy interface)."""
        return {
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "out_of_lane_rate": self.out_of_lane_rate,
            "mean_reward": self.mean_reward,
            "mean_route_completion": self.mean_route_completion,
            "mean_speed_kmh": self.mean_speed_kmh,
            "num_episodes": len(self.episodes),
            "mean_steps": (
                float(np.mean([e.steps for e in self.episodes]))
                if self.episodes
                else 0.0
            ),
            "mean_lateral_deviation": (
                float(np.mean([e.avg_lateral_deviation for e in self.episodes]))
                if self.episodes
                else 0.0
            ),
            "total_safety_interventions": float(self.total_safety_interventions),
        }

    def to_dict(self) -> dict:
        """Export comprehensive P0 JSON schema representation."""
        return {
            "name": self.name,
            "summary": {
                **self.summary(),
                "success_rate_pct": round(self.success_rate * 100, 2),
                "collision_rate_pct": round(self.collision_rate * 100, 2),
                "route_completion_pct": round(self.mean_route_completion * 100, 2),
            },
            "system_metrics": self._system_metrics,
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def save_json(self, path: Path | str) -> None:
        """Save metrics to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_summary_markdown(self) -> str:
        """Generate a formatted markdown summary table."""
        s = self.summary()
        lines = [
            f"### Evaluation Driving Metrics: {self.name}",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| **Evaluated Episodes** | {int(s['num_episodes'])} |",
            f"| **Route Completion** | {s['mean_route_completion'] * 100:.1f}% |",
            f"| **Success Rate** | {s['success_rate'] * 100:.1f}% |",
            f"| **Collision Rate** | {s['collision_rate'] * 100:.1f}% |",
            f"| **Out of Lane Rate** | {s['out_of_lane_rate'] * 100:.1f}% |",
            f"| **Mean Episode Return** | {s['mean_reward']:.2f} |",
            f"| **Mean Episode Steps** | {s['mean_steps']:.1f} |",
            f"| **Mean Speed** | {s['mean_speed_kmh']:.1f} km/h |",
            f"| **Mean Lateral Deviation** | {s['mean_lateral_deviation']:.2f} m |",
            f"| **Safety Supervisor Interventions** | {int(s['total_safety_interventions'])} |",
        ]
        if self._system_metrics:
            if "steps_per_sec" in self._system_metrics:
                lines.append(f"| **Simulation Throughput** | {self._system_metrics['steps_per_sec']:.1f} steps/s |")
            if "peak_vram_mb" in self._system_metrics:
                lines.append(f"| **Peak VRAM** | {self._system_metrics['peak_vram_mb']:.1f} MB |")
        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"MetricsTracker('{self.name}'): "
            f"success={s['success_rate']:.2%}, "
            f"collision={s['collision_rate']:.2%}, "
            f"completion={s['mean_route_completion']:.2%}, "
            f"reward={s['mean_reward']:.2f}, "
            f"episodes={s['num_episodes']}"
        )


class ComparisonTable:
    """
    Generates comparison tables for Phase 3's three-arm experiment.

    Produces a formatted table showing metrics across all arms
    with matched seeds for a controlled comparison.
    """

    def __init__(self):
        self.arm_results: dict[str, list[dict[str, float]]] = {}

    def add_result(self, arm: str, seed: int, metrics: dict[str, float]) -> None:
        """
        Add a result for one arm+seed combination.

        Args:
            arm: Arm name ('cnn', 'custom_jepa', 'vjepa2').
            seed: Random seed used.
            metrics: Dict of metric name → value.
        """
        if arm not in self.arm_results:
            self.arm_results[arm] = []
        metrics_copy = dict(metrics)
        metrics_copy["seed"] = seed
        self.arm_results[arm].append(metrics_copy)

    def generate_table(self) -> str:
        """
        Generate a formatted comparison table.

        Shows mean ± std across seeds for each arm.

        Returns:
            Formatted table string.
        """
        if not self.arm_results:
            return "No results to compare."

        all_metrics = set()
        for results in self.arm_results.values():
            for r in results:
                all_metrics.update(k for k in r.keys() if k != "seed")

        metric_names = sorted(all_metrics)

        lines = []
        header = "| Metric |"
        separator = "|--------|"
        for arm in self.arm_results:
            header += f" {arm} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        for metric in metric_names:
            row = f"| {metric} |"
            for arm in self.arm_results:
                values = [r.get(metric, float("nan")) for r in self.arm_results[arm]]
                mean = np.mean(values)
                if len(values) > 1:
                    std = np.std(values)
                    row += f" {mean:.4f} ± {std:.4f} |"
                else:
                    row += f" {mean:.4f} |"
            lines.append(row)

        return "\n".join(lines)

    def save_table(self, path: str) -> None:
        """Save comparison table to a file."""
        table = self.generate_table()
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Phase 3: Three-way Encoder Comparison\n\n")
            f.write(table)
            f.write("\n")

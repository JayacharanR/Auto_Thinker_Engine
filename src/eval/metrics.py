"""
Driving metrics for evaluation.

EPDMS-inspired metrics for Phase 1 (DreamerV3 baseline) and Phase 3
(three-way comparison). Tracks success rate, collision rate,
reward statistics, and produces comparison tables.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected for a single evaluation episode."""

    success: bool = False
    collision: bool = False
    total_reward: float = 0.0
    steps: int = 0
    route_completion: float = 0.0
    num_traffic_violations: int = 0
    avg_lateral_deviation: float = 0.0
    min_ttc: float = float("inf")


class MetricsTracker:
    """
    Tracks and aggregates driving metrics across evaluation episodes.

    Used for both Phase 1 (Gate 1 evaluation) and Phase 3
    (three-arm comparison).
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.episodes: list[EpisodeMetrics] = []
        self._current_episode: Optional[EpisodeMetrics] = None
        self._step_deviations: list[float] = []

    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self._current_episode = EpisodeMetrics()
        self._step_deviations = []

    def step(self, reward_info: dict[str, Any]) -> None:
        """Record metrics from a single step."""
        if self._current_episode is None:
            return

        self._current_episode.steps += 1
        self._current_episode.total_reward += reward_info.get("reward/total", 0.0)

        if reward_info.get("reward/collision", 0.0) > 0:
            self._current_episode.collision = True

        deviation = reward_info.get("reward/lane_keeping", 0.0)
        self._step_deviations.append(abs(deviation))

        ttc_info = reward_info.get("reward/time_to_collision", float("inf"))
        if isinstance(ttc_info, (int, float)):
            self._current_episode.min_ttc = min(
                self._current_episode.min_ttc, ttc_info
            )

    def end_episode(self, success: bool = False, route_completion: float = 0.0) -> EpisodeMetrics:
        """End the current episode and record it."""
        if self._current_episode is None:
            return EpisodeMetrics()

        self._current_episode.success = success
        self._current_episode.route_completion = route_completion
        if self._step_deviations:
            self._current_episode.avg_lateral_deviation = float(
                np.mean(self._step_deviations)
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

    def summary(self) -> dict[str, float]:
        """Get summary statistics as a flat dict."""
        return {
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "mean_reward": self.mean_reward,
            "mean_route_completion": self.mean_route_completion,
            "num_episodes": len(self.episodes),
            "mean_steps": float(np.mean([e.steps for e in self.episodes]))
            if self.episodes
            else 0.0,
            "mean_lateral_deviation": float(
                np.mean([e.avg_lateral_deviation for e in self.episodes])
            )
            if self.episodes
            else 0.0,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"MetricsTracker('{self.name}'): "
            f"success={s['success_rate']:.2%}, "
            f"collision={s['collision_rate']:.2%}, "
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

        # Collect all metric names
        all_metrics = set()
        for results in self.arm_results.values():
            for r in results:
                all_metrics.update(k for k in r.keys() if k != "seed")

        metric_names = sorted(all_metrics)

        # Header
        lines = []
        header = "| Metric |"
        separator = "|--------|"
        for arm in self.arm_results:
            header += f" {arm} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Rows
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
        with open(path, "w") as f:
            f.write("# Phase 3: Three-way Encoder Comparison\n\n")
            f.write(table)
            f.write("\n")

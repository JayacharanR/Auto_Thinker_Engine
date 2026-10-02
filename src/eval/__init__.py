"""
Evaluation tools: linear probes, driving metrics, comparison tables.
"""

from src.eval.linear_probe import extract_features, fit_linear_probe, run_probe_comparison
from src.eval.metrics import MetricsTracker, ComparisonTable, EpisodeMetrics

__all__ = [
    "extract_features",
    "fit_linear_probe",
    "run_probe_comparison",
    "MetricsTracker",
    "ComparisonTable",
    "EpisodeMetrics",
]

"""Frozen plan/source boundaries, without any native server or rendered outcomes."""

import json
from pathlib import Path

from src.navigation.dagger import digest

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_signal_diversity_is_train_only_and_does_not_resplit():
    root = ROOT / "outputs/navigation_signal_diversity_geometry_20261021"
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "geometry_summary.json").read_text())
    assert digest(root / "plan.json") == summary["plan_sha256"]
    allowed = json.loads(Path(plan["eligibility_request"]).read_text())["allowed_junctions"]
    assert plan["allowed_junctions"] == allowed
    assert len(plan["tasks"]) == 8 and plan["planned_cases"] == 200
    assert sum(len(t["cases"]) for t in plan["tasks"]) == 200
    assert len(plan["cases"]) == 100
    for case in plan["cases"]:
        assert case["role"] == "train" and "Town05" not in case["town"]
        assert case["junction_id"] in allowed[case["town"]]
        assert case["reserved_clearance_m"] >= 1
    assert len({(c["town"], c["junction_id"]) for c in plan["cases"]}) == 21
    for task in plan["tasks"]:
        assert all(c["town"] == task["town"] for c in task["cases"])
        assert task["weather"]["precipitation"] == 0


def test_collection_supervisor_cannot_import_carla_at_module_scope():
    import ast

    tree = ast.parse((ROOT / "scripts/collect_navigation_signal_diversity.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name != "carla" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "carla"


def test_weather_mismatch_retains_measured_values_before_failure(tmp_path):
    from types import SimpleNamespace

    import pytest

    from scripts.collect_navigation_signal_diversity import verify_weather

    path = tmp_path / "weather.json"
    with pytest.raises(ValueError, match="Native weather"):
        verify_weather(SimpleNamespace(cloudiness=12), {"cloudiness": 0}, path)
    receipt = json.loads(path.read_text())
    assert receipt["observed"] == {"cloudiness": 12}
    assert receipt["deltas"] == {"cloudiness": 12} and not receipt["matches"]
    verify_weather(SimpleNamespace(cloudiness=0), {"cloudiness": 0}, path)
    assert json.loads(path.read_text())["matches"]

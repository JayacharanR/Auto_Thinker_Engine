"""Reporting integrity for partial runs, retries and invalid scenario setups."""

import json

from scripts.assess_encounters import (
    cluster_interval,
    collision_audit,
    planned_headline,
    setup_audit,
)


def test_setup_audit_uses_final_attempt_and_keeps_pending_separate(tmp_path):
    entry = {
        "route_id": 1,
        "town": "Town03",
        "scenario": "NavigationSignalHold",
        "signal_ids_at_inventory": [7],
    }
    final = {"route_id": "RouteScenario_1_rep0", "status": "Completed"}
    attempts = {
        1: [
            {**final, "attempt": "chunk_a0", "status": "Failed - Simulation crashed"},
            {**final, "attempt": "chunk_a1"},
        ]
    }
    for attempt, started in (("chunk_a0", True), ("chunk_a1", False)):
        path = tmp_path / "Town03" / attempt / "scenario_setup" / "1.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "started": [{"type": "NavigationSignalHold", "ego_light_id": 7}]
                    if started
                    else []
                }
            )
        )
    catalog = {"encounters": [entry, {**entry, "route_id": 2}]}
    assert setup_audit(tmp_path, catalog, {1: final}, attempts) == {
        1: "setup_failed",
        2: "unsettled",
    }
    (tmp_path / "Town03/chunk_a1/scenario_setup/1.json").unlink()
    assert setup_audit(tmp_path, catalog, {1: final}, attempts)[1] == "setup_missing"


def test_all_collision_events_include_retried_and_invalid_setup_attempts(tmp_path):
    for attempt, collisions in (
        ("chunk_a0", ["before native failure"]),
        ("chunk_a1", ["invalid setup hit 1", "invalid setup hit 2"]),
    ):
        path = tmp_path / "Town03" / attempt / "leaderboard.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "_checkpoint": {
                        "records": [
                            {
                                "route_id": "RouteScenario_1_rep0",
                                "infractions": {"collisions_vehicle": collisions},
                            }
                        ]
                    }
                }
            )
        )
    events = collision_audit(tmp_path)
    assert len(events) == 3
    assert {e["attempt"] for e in events} == {"Town03/chunk_a0", "Town03/chunk_a1"}


def test_empty_and_all_success_bootstraps_do_not_claim_certainty():
    assert cluster_interval([], lambda x: x, lambda x: x) == (None, None, 0)
    rows = [(0, True), (0, True), (1, True)]
    assert cluster_interval(rows, lambda x: x[0], lambda x: x[1]) == (None, None, 2)
    rows.append((2, False))
    low, high, n = cluster_interval(rows, lambda x: x[0], lambda x: x[1], reps=100)
    assert n == 3 and 0 <= low < high <= 1


def test_headline_keeps_missing_setup_pending_trials_and_retry_collisions():
    catalog = dict(encounters=[dict(route_id=i, town="Town01", junction_id=i) for i in range(4)])
    success = dict(completed=True, safe_completed=True, rule_safe_completed=True)
    result = planned_headline(
        catalog,
        {0: success, 1: success, 2: success},
        {0: "setup_started", 1: "setup_missing", 2: "wrong_light", 3: "unsettled"},
        5,
    )
    assert result["expected"] == 4 and result["recorded"] == 3
    assert result["raw_completed"] == 3 and result["completed"] == 1
    assert result["rule_safe_fraction"] == 0.25 and result["strict_safe"] == 1
    assert result["setup_failures"] == [1, 2] and result["missing"] == [3]
    assert result["all_collision_events"] == 5


def test_live_trial_is_not_reported_as_a_crash(tmp_path, monkeypatch):
    from scripts import assess_encounters as assess

    catalog = {"towns": ["Town03"], "encounters": [{"route_id": 1, "town": "Town03"}]}
    (tmp_path / "request.json").write_text(json.dumps({"chunk": 10}))
    town = tmp_path / "Town03"
    town.mkdir()
    (town / "chunk_001_001_a0.routes.json").write_text("[1]")
    chunk = town / "chunk_001_001_a0"
    chunk.mkdir()
    (chunk / "manifest.json").write_text("{}")
    (chunk / "current_route.json").write_text('{"route_id": "1"}')
    monkeypatch.setattr(assess, "worker_is_alive", lambda p: True)
    finals, attempts = assess.settled(tmp_path, catalog)
    assert finals == {} and attempts[1][0]["status"] == "in progress"
    monkeypatch.setattr(assess, "worker_is_alive", lambda p: False)
    assert assess.settled(tmp_path, catalog)[1][1][0]["status"] == "no evaluator record"


def test_live_worker_detection_rejects_reused_pid(tmp_path):
    import os

    from scripts.assess_encounters import worker_is_alive

    (tmp_path / "worker.json").write_text(json.dumps({"pid": os.getpid()}))
    # This pytest process is alive but is not the evaluator owning that output.
    assert not worker_is_alive(tmp_path)


def test_light_identity_survives_id_permutation_but_rejects_wrong_physical_light():
    from scripts.assess_encounters import light_identity

    entry = {
        "town": "Town10HD_Opt",
        "movement_key": ["town", "189", "(20, 0, 5)"],
        "signal_ids_at_inventory": [101],
    }
    reference = {"lights": [{"id": 101, "xyz": [1, 2, 3], "stop_lanes": [{"lane": [20, 0, 5]}]}]}
    runtime = {
        "map": "Town10HD_Opt",
        "signs": [
            {"id": 99, "type": "traffic.traffic_light", "xyz": [1.0001, 2, 3]},
            {"id": 101, "type": "traffic.traffic_light", "xyz": [20, 30, 3]},
        ],
    }
    assert light_identity(entry, 99, runtime, reference)["status"] == "setup_started"
    # The same numerical ID is insufficient if it now belongs to a different lamp.
    assert light_identity(entry, 101, runtime, reference)["status"] == "wrong_light"
    reference["lights"][0]["stop_lanes"][0]["lane"] = [21, 0, 5]
    assert light_identity(entry, 99, runtime, reference)["status"] == "wrong_light"

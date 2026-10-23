import json

from src.navigation.final_audit import (
    conservative_final_counts,
    final_opportunity_audit,
    physical_light_identity,
    physical_opportunity_counts,
    raw_attempt_events,
)


def test_final_counts_every_collision_including_crashed_and_interrupted_attempts(tmp_path):
    for attempt, status, count in ((0, "Failed - Simulation crashed", 2), (1, "Completed", 1)):
        path = tmp_path / "seed_3031" / f"all_routes_a{attempt}" / "leaderboard.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                dict(
                    _checkpoint=dict(
                        records=[
                            dict(
                                route_id="RouteScenario_0_rep0",
                                status=status,
                                infractions=dict(
                                    collisions_vehicle=["contact"] * count,
                                    red_light=[],
                                    stop_infraction=[],
                                ),
                            )
                        ]
                    )
                )
            )
        )
    measured = raw_attempt_events(tmp_path)
    assert measured["collisions"] == 3 and measured["raw_records"] == 2
    assert len(measured["attempts"]) == 2


def test_physical_light_ids_are_session_local_and_ambiguous_poles_fail():
    entry = dict(town="Town05_Opt", held_signal=dict(actor_id=100, lane=[1, 0, -1]))
    inventory = dict(lights=[dict(id=100, xyz=[1, 2, 3], stop_lanes=[dict(lane=[1, 0, -1])])])
    runtime = dict(
        map="Town05_Opt", signs=[dict(id=44, type="traffic.traffic_light", xyz=[1, 2, 3])]
    )
    assert physical_light_identity(entry, 44, runtime, inventory)["valid"]
    runtime["signs"].append(dict(id=45, type="traffic.traffic_light", xyz=[1, 2, 3]))
    assert not physical_light_identity(entry, 44, runtime, inventory)["valid"]


def test_missing_setup_never_looks_like_an_easy_final_success():
    records = {f"3031:{i}": dict(safe_completed=True, rule_safe_completed=True) for i in range(3)}
    setups = {f"3031:{i}": dict(valid=i != 1) for i in range(3)}
    measured = conservative_final_counts(records, setups, 36)
    assert measured["native_safe_completed"] == 3 and measured["safe_completed"] == 2
    assert measured["planned_missions"] == 36
    assert measured["invalid_or_missing_setup"] == ["3031:1"]


def test_contact_evidence_preserves_native_counts_and_separates_incomplete_recorders():
    records = {str(i): dict(safe_completed=True, rule_safe_completed=True) for i in range(3)}
    setups = {str(i): dict(valid=True) for i in range(3)}
    contacts = {
        "0": dict(complete=True, contacts=2),
        "1": dict(complete=True, contacts=0),
        "2": dict(complete=False, contacts=0),
    }
    counts = conservative_final_counts(records, setups, 36, contacts)
    assert counts["native_rule_safe_completed"] == 3 and counts["rule_safe_completed"] == 1
    assert counts["invalid_or_missing_setup"] == [] and counts["incomplete_contact_evidence"] == [
        "2"
    ]


def test_native_stop_lane_change_does_not_double_the_opportunity_denominator():
    opportunity = dict(
        errors=[],
        criterion_returns_unchanged=True,
        events=[dict(kind="stop", actor_id=40, lane=lane) for lane in ([19, 0, -1], [783, 0, -1])],
    )
    runtime = dict(signs=[dict(id=40, type="traffic.stop", xyz=[1, 2, 3])])
    inventory = dict(stops=[dict(id=100, xyz=[1, 2, 3])], lights=[])
    counted = physical_opportunity_counts(opportunity, runtime, inventory)
    assert counted["valid"] and counted["stop_opportunities_lower_bound"] == 1
    assert counted["native_actor_lane_pairs"] == 2
    assert counted["distinct_stop_inventory_ids"] == [100]
    runtime["signs"][0]["xyz"] = [9, 9, 9]
    assert not physical_opportunity_counts(opportunity, runtime, inventory)["valid"]


def test_final_coverage_requires_the_five_declared_stops_not_any_five(tmp_path):
    seeds = [3031, 4031, 5031]
    suite = dict(
        traffic_seeds=seeds,
        routes=[dict(route_id=i, required_stop_actor=87 + i % 5) for i in range(12)],
    )
    inventory = dict(
        stops=[dict(id=i, xyz=[i, 0, 0]) for i in range(87, 93)],
        lights=[dict(id=146, xyz=[0, 0, 0])],
    )
    runtime = dict(
        signs=[dict(id=i + 400, type="traffic.stop", xyz=[i, 0, 0]) for i in range(87, 93)]
        + [dict(id=700, type="traffic.traffic_light", xyz=[0, 0, 0])]
    )
    records, attempts = {}, {}
    for seed in seeds:
        base = tmp_path / f"seed_{seed}" / "all_routes_a0"
        logs = base / "rule_opportunities"
        logs.mkdir(parents=True)
        (base / "runtime_sign_inventory.json").write_text(json.dumps(runtime))
        attempts[str(seed)] = {}
        for i in range(12):
            record = dict(
                route_id=f"RouteScenario_{i}_rep0", status="Completed", attempt="all_routes_a0"
            )
            records[f"{seed}:{i}"] = record
            attempts[str(seed)][i] = [record]
            (logs / f"{i}.json").write_text(
                json.dumps(
                    dict(
                        errors=[],
                        criterion_returns_unchanged=True,
                        events=[
                            dict(kind="stop", actor_id=488 + i % 5, lane=[1, 0, -1]),
                            dict(kind="red", actor_id=700, lane=[2, 0, -1]),
                        ],
                    )
                )
            )
    result = final_opportunity_audit(tmp_path, suite, records, attempts, inventory)
    assert len(result["final_distinct_stop_locations"]) == 5
    assert result["final_red_opportunities_lower_bound"] == 36
    assert result["final_stop_opportunities_lower_bound"] == 36
    assert result["unobserved_required_mapped_stops"] == [87]
    assert not result["declared_coverage_observed"]

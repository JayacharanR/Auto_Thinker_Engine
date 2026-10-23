"""Scoring-only conservative final setup and all-attempt event accounting."""

import json
import re
from pathlib import Path

import numpy as np

from src.navigation.dagger import digest
from src.navigation.recorder_collisions import recorder_contacts


def raw_attempt_events(root):
    totals = dict(collisions=0, red_violations=0, stop_violations=0, raw_records=0)
    attempts = []
    for path in sorted(Path(root).glob("seed_*/all_routes_a*/leaderboard.json")):
        records = json.loads(path.read_text())["_checkpoint"]["records"]
        counts = dict(collisions=0, red_violations=0, stop_violations=0, raw_records=len(records))
        for record in records:
            if not re.fullmatch(r"RouteScenario_\d+_rep\d+", record["route_id"]):
                raise ValueError("Unknown final native record identity")
            infractions = record["infractions"]
            counts["collisions"] += sum(
                len(v) for k, v in infractions.items() if k.startswith("collisions_")
            )
            counts["red_violations"] += len(infractions.get("red_light", []))
            counts["stop_violations"] += len(infractions.get("stop_infraction", []))
        for key in totals:
            totals[key] += counts[key]
        attempts.append(dict(path=str(path), sha256=digest(path), **counts))
    return dict(
        totals,
        attempts=attempts,
        note=(
            "Every native recorded event in every raw attempt, including retries; "
            "native scoring has a stationary fault filter, so use recorder contacts too"
        ),
    )


def physical_light_identity(entry, runtime_id, runtime, inventory, tolerance=0.01):
    expected = [
        light for light in inventory["lights"] if light["id"] == entry["held_signal"]["actor_id"]
    ]
    actual = [
        sign
        for sign in runtime["signs"]
        if sign["type"] == "traffic.traffic_light" and sign["id"] == runtime_id
    ]
    result = dict(valid=False, status="wrong_or_ambiguous_light", runtime_id=runtime_id)
    if runtime["map"] != entry["town"] or len(expected) != 1 or len(actual) != 1:
        return result
    if entry["held_signal"]["lane"] not in [row["lane"] for row in expected[0]["stop_lanes"]]:
        return result
    position = np.asarray(expected[0]["xyz"])
    distances = [
        float(np.linalg.norm(np.asarray(row["xyz"]) - position))
        for row in runtime["signs"]
        if row["type"] == "traffic.traffic_light"
    ]
    distance = float(np.linalg.norm(np.asarray(actual[0]["xyz"]) - position))
    result.update(position_difference_m=distance, tolerance_m=tolerance)
    if distance <= tolerance and sum(d <= tolerance for d in distances) == 1:
        result.update(valid=True, status="setup_started")
    return result


def final_setup_audit(root, suite, records, attempts, inventory):
    root = Path(root)
    entries = {entry["route_id"]: entry for entry in suite["routes"]}
    results = {}
    for seed in suite["traffic_seeds"]:
        history = attempts.get(str(seed), attempts.get(seed, {}))
        for rid, entry in entries.items():
            key = f"{seed}:{rid}"
            if key not in records:
                results[key] = dict(valid=False, status="no_final_record")
                continue
            route_history = history.get(rid, history.get(str(rid), []))
            final = records[key]
            selected = next(
                (
                    row
                    for row in reversed(route_history)
                    if row.get("route_id") == final["route_id"]
                    and "crash" not in row["status"].lower()
                ),
                None,
            )
            if selected is None:
                results[key] = dict(valid=False, status="final_attempt_not_identified")
                continue
            attempt = root / f"seed_{seed}" / selected["attempt"]
            path = attempt / "scenario_setup" / f"{rid}.json"
            if not path.exists():
                results[key] = dict(valid=False, status="setup_not_observed", attempt=str(attempt))
                continue
            started = [
                row
                for row in json.loads(path.read_text()).get("started", [])
                if row["type"] == "NavigationSignalHold"
            ]
            runtime = attempt / "runtime_sign_inventory.json"
            if len(started) != 1 or not runtime.exists():
                results[key] = dict(
                    valid=False, status="missing_or_duplicate_setup", attempt=str(attempt)
                )
                continue
            identity = physical_light_identity(
                entry, started[0]["ego_light_id"], json.loads(runtime.read_text()), inventory
            )
            results[key] = dict(
                identity,
                attempt=str(attempt),
                setup_sha256=digest(path),
                runtime_inventory_sha256=digest(runtime),
            )
    return results


def conservative_final_counts(records, setup_audit, denominator, contact_evidence=None):
    valid = {key for key, result in setup_audit.items() if result["valid"]}

    def contact_free(key):
        if contact_evidence is None:
            return True  # explicit native/setup-only diagnostic
        evidence = contact_evidence.get(key, {})
        return evidence.get("complete") is True and evidence.get("contacts") == 0

    return dict(
        planned_missions=denominator,
        native_safe_completed=sum(row["safe_completed"] for row in records.values()),
        native_rule_safe_completed=sum(row["rule_safe_completed"] for row in records.values()),
        safe_completed=sum(
            row["safe_completed"]
            for key, row in records.items()
            if key in valid and contact_free(key)
        ),
        rule_safe_completed=sum(
            row["rule_safe_completed"]
            for key, row in records.items()
            if key in valid and contact_free(key)
        ),
        invalid_or_missing_setup=[
            key for key, result in setup_audit.items() if not result["valid"]
        ],
        incomplete_contact_evidence=[]
        if contact_evidence is None
        else [
            key for key in setup_audit if contact_evidence.get(key, {}).get("complete") is not True
        ],
        note=("Invalid/missing setups never count as successes; keep the full planned denominator"),
    )


def recorder_attempt_contacts(root, records, attempts, seeds):
    """Count every preserved attempt and identify the contact count of final records."""
    audited, failures, by_path = [], [], {}
    for path in sorted(Path(root).glob("seed_*/all_routes_a*/recordings/route_*_rep*.log")):
        try:
            record = recorder_contacts(path)
            audited.append(record)
            by_path[str(path.resolve())] = record
        except Exception as error:
            failures.append(dict(path=str(path), error=str(error)))
    final_contacts = {}
    for seed in seeds:
        history = attempts.get(str(seed), attempts.get(seed, {}))
        for key, final in records.items():
            if not key.startswith(f"{seed}:"):
                continue
            rid = int(key.split(":")[1])
            route_history = history.get(rid, history.get(str(rid), []))
            selected = next(
                (
                    row
                    for row in reversed(route_history)
                    if row.get("route_id") == final["route_id"]
                    and "crash" not in row["status"].lower()
                ),
                None,
            )
            path = (
                (
                    Path(root)
                    / f"seed_{seed}"
                    / selected["attempt"]
                    / "recordings"
                    / f"route_{rid}_rep0.log"
                )
                if selected
                else None
            )
            contact = by_path.get(str(path.resolve())) if path else None
            final_contacts[key] = dict(
                complete=bool(contact and contact["complete"]),
                contacts=contact["ego_contact_starts"] if contact else None,
                path=str(path) if path else None,
            )
    return dict(
        all_attempt_ego_contact_starts=sum(row["ego_contact_starts"] for row in audited),
        all_attempt_raw_contact_samples=sum(row["ego_raw_contact_samples"] for row in audited),
        complete_recordings=sum(row["complete"] for row in audited),
        partial_recordings=[row["path"] for row in audited if not row["complete"]],
        parse_failures=failures,
        final_contacts=final_contacts,
        records=audited,
        note=(
            "No speed/fault filter. Partial recorder counts are lower bounds; "
            "never a certified zero."
        ),
    )


def physical_opportunity_counts(opportunity, runtime, inventory):
    """Conservative distinct physical sign/mission counts, not lane-pair counts.

    Native stop eligibility may span both an incoming and a junction lane for
    the SAME encounter. Merge those rather than doubling the denominator.
    Repeated visits to one physical sign are also merged: this is explicitly
    a lower bound on encounters, never an exact per-visit rate denominator.
    """
    errors = list(opportunity.get("errors", []))
    identified = []
    for event in opportunity.get("events", []):
        kind = event["kind"]
        if kind not in ("red", "stop"):
            errors.append(f"Unknown native opportunity:{kind}")
            continue
        type_id = "traffic.traffic_light" if kind == "red" else "traffic.stop"
        actual = [
            s for s in runtime["signs"] if s["id"] == event["actor_id"] and s["type"] == type_id
        ]
        if len(actual) != 1:
            errors.append(f"Native opportunity has no unique runtime sign:{event['actor_id']}")
            continue
        reference = [
            s
            for s in inventory["lights" if kind == "red" else "stops"]
            if np.linalg.norm(np.asarray(s["xyz"]) - actual[0]["xyz"]) <= 0.01
        ]
        if len(reference) != 1:
            errors.append(
                f"Native opportunity has no unique physical inventory match:{event['actor_id']}"
            )
            continue
        identified.append(dict(event, physical_inventory_id=reference[0]["id"]))
    unique = {(e["kind"], e["physical_inventory_id"]) for e in identified}
    return dict(
        valid=not errors and opportunity.get("criterion_returns_unchanged") is True,
        errors=errors,
        bound_events=identified,
        native_actor_lane_pairs=len(opportunity.get("events", [])),
        red_opportunities_lower_bound=sum(kind == "red" for kind, _ in unique),
        stop_opportunities_lower_bound=sum(kind == "stop" for kind, _ in unique),
        distinct_stop_inventory_ids=sorted(i for kind, i in unique if kind == "stop"),
        denominator_semantics=(
            "Distinct physical sign and rule kind per mission; "
            "merges multiple lanes and repeated visits. "
            "Lower bound on encounters, not an exact per-visit violation-rate denominator."
        ),
    )


def final_opportunity_audit(root, suite, records, attempts, inventory):
    final, all_attempts, failures = {}, [], []
    for path in sorted(Path(root).glob("seed_*/all_routes_a*/rule_opportunities/*.json")):
        runtime_path = path.parents[1] / "runtime_sign_inventory.json"
        try:
            measured = physical_opportunity_counts(
                json.loads(path.read_text()), json.loads(runtime_path.read_text()), inventory
            )
            all_attempts.append(dict(path=str(path), sha256=digest(path), **measured))
        except Exception as error:
            failures.append(dict(path=str(path), error=str(error)))
    lookup = {str(Path(r["path"]).resolve()): r for r in all_attempts}
    for seed in suite["traffic_seeds"]:
        history = attempts.get(str(seed), attempts.get(seed, {}))
        for entry in suite["routes"]:
            rid = entry["route_id"]
            key = f"{seed}:{rid}"
            record = records.get(key)
            selected = next(
                (
                    r
                    for r in reversed(history.get(rid, history.get(str(rid), [])))
                    if record
                    and r.get("route_id") == record["route_id"]
                    and "crash" not in r["status"].lower()
                ),
                None,
            )
            path = (
                Path(root)
                / f"seed_{seed}"
                / selected["attempt"]
                / "rule_opportunities"
                / f"{rid}.json"
                if selected
                else None
            )
            measured = lookup.get(str(path.resolve())) if path else None
            final[key] = dict(
                measured or dict(valid=False, errors=["No final physical opportunity log"]),
                required_stop_observed=bool(
                    measured
                    and entry["required_stop_actor"] in measured["distinct_stop_inventory_ids"]
                ),
            )
    valid = [r for r in final.values() if r["valid"]]
    red = sum(r["red_opportunities_lower_bound"] for r in valid)
    stop = sum(r["stop_opportunities_lower_bound"] for r in valid)
    locations = sorted({i for r in valid for i in r["distinct_stop_inventory_ids"]})
    required = {entry["required_stop_actor"] for entry in suite["routes"]}
    missing_required = sorted(required - set(locations))
    return dict(
        final_missions=final,
        all_raw_attempts=all_attempts,
        parse_failures=failures,
        final_red_opportunities_lower_bound=red,
        final_stop_opportunities_lower_bound=stop,
        final_distinct_stop_locations=locations,
        required_mapped_stop_locations=sorted(required),
        unobserved_required_mapped_stops=missing_required,
        missions_missing_required_stop=[
            key for key, r in final.items() if not r["required_stop_observed"]
        ],
        missing_or_invalid_final_logs=[key for key, r in final.items() if not r["valid"]],
        declared_coverage_observed=bool(
            not failures
            and len(valid) == 36
            and red >= 30
            and stop >= 30
            and len(required) == 5
            and not missing_required
            and all(r["required_stop_observed"] for r in final.values())
        ),
        note=(
            "Missing/blocked missions stay in36 denominator. "
            "Coverage is observed, never imputed from route plans."
        ),
    )

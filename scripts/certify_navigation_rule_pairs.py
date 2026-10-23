#!/usr/bin/env python3
"""Derive paired physical certificates before any controlled student trial."""

import argparse
import ast
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.assess_navigation_rule_prototypes import assess  # noqa:E402
from src.navigation.dagger import digest  # noqa:E402
from src.navigation.rule_suite import SETUP_RUNTIME_SOURCES, validate_pair  # noqa:E402
from src.navigation.runtime import atomic_json  # noqa:E402
from src.navigation.vendor import activate  # noqa:E402


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def resource_only_config_change(configuration):
    """Allow only the explicitly approved25→10GB storage floor, with full disclosure."""
    current_path = ROOT / "configs/navigation.yaml"
    if configuration["config_sha256"] == digest(current_path):
        return None
    current = yaml.safe_load(current_path.read_text())
    reference = copy.deepcopy(configuration["configuration"])
    old_floor = reference["runtime"].pop("minimum_free_gb")
    new_floor = current["runtime"].pop("minimum_free_gb")
    plan_path = ROOT / "reports/navigation_local_storage_plan_20261020.json"
    approval = read(plan_path)
    guard_request = (
        ROOT / "outputs/navigation_compact_execution_20261020/storage_guard/request.json"
    )
    if (
        reference != current or old_floor != 25 or new_floor != 10
        or approval.get("user_approved") is not True
        or approval["minimum_free_bytes_proposed"] != 10_000_000_000
        or read(guard_request)["plan_sha256"] != digest(plan_path)
    ):
        raise ValueError("Native physical configuration changed beyond approved storage-only floor")
    return dict(
        original_config_sha256=configuration["config_sha256"],
        current_config_sha256=digest(current_path),
        only_changed_key="runtime.minimum_free_gb", original_value=25, current_value=10,
        approved_plan_sha256=digest(plan_path), all_sensor_and_physics_values_identical=True,
    )


def prefix_valid(spec, audit):
    prefix, handoff = spec["treatment"]["prefix"], audit["actual_handoff"]
    if not handoff or not audit["prefix_never_reactivated"]:
        return False
    if prefix["mode"] == "none":
        return True
    if handoff["timestamp"] > prefix["maximum_seconds"] + 0.1:
        return False
    if abs(handoff["distance_to_registered_line_m"] - prefix["handoff_distance_m"]) > 0.5:
        return False
    return (
        handoff["speed_mps"] < 0.1
        if prefix["mode"] == "stop_before_line"
        else 0.1 < handoff["speed_mps"] <= prefix["speed_mps"] + 0.5
    )


def physical_audit(case, reference, actuation, maps, guard, carla_root):
    import carla

    path = Path(case["path"])
    if digest(path) != case["sha256"]:
        raise ValueError("Prospective physical geometry changed")
    spec = read(path)
    native = Path(reference[case["sha256"]]["audit"]["output"])
    audited = assess(native)
    registered = reference[case["sha256"]]["audit"]
    if (audited["manifest_sha256"] != registered["manifest_sha256"]
            or audited["recorder"]["sha256"] != registered["recorder"]["sha256"]):
        raise ValueError("Original registered native reference artifacts changed")
    # Known changes since the references are limited to the student certificate
    # gate/checkpoint metadata, certificate validation and the unused manual rig.
    # All actual native setup/control functions must still match archived code.
    exceptions = {
        "scripts/run_navigation.py",
        "src/navigation/rule_suite.py",
        "src/navigation/sensors.py",
    }
    archived = read(native / "source_hashes.json")
    configuration = read(native / "manifest.json")
    resource_change = resource_only_config_change(configuration)
    for source in set(SETUP_RUNTIME_SOURCES) - exceptions:
        expected = archived.get(source)
        if (expected is not None and digest(ROOT / source) != expected
                and not (source == "configs/navigation.yaml" and resource_change)):
            raise ValueError(f"Native physical execution source changed:{source}")
        if expected is None:
            if source == "configs/navigation.yaml" and configuration["config_sha256"] != digest(
                ROOT / source
            ) and not resource_change:
                raise ValueError("Native camera/runtime configuration changed")
    for source, functions in (
        ("src/navigation/sensors.py", ("calibrations", "camera_specs")),
        ("src/navigation/rule_suite.py", ("arms", "differences", "validate_pair")),
    ):
        trees = [
            ast.parse(path.read_text()) for path in (ROOT / source, native / "source" / source)
        ]
        for name in functions:
            bodies = [
                ast.dump(
                    next(
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name
                    )
                )
                for tree in trees
            ]
            if bodies[0] != bodies[1]:
                raise ValueError(f"Native setup function changed:{source}:{name}")
    if not all(
        audited[k]
        for k in (
            "source_hashes_verified",
            "synchronized_frames",
            "declared_actor_setup",
            "evaluated_controls_unchanged",
        )
    ):
        raise ValueError("Independent physical input/control audit failed")
    if audited["binding"]["case_sha256"] != case["sha256"]:
        raise ValueError("Physical binding belongs to a different case")
    action = actuation[str(native.resolve())]
    if (
        action["differences"]
        or action["missing_final_ack_frames"]
        or action["recorder_sha256"] != audited["recorder"]["sha256"]
        or not audited["recorder"]["complete"]
    ):
        raise ValueError("Complete exact executed-actuation evidence required")
    episode = next((native / "episodes").iterdir())
    controls = episode / "controlled_rule/requested_executed.jsonl"
    if action["control_log_sha256"] != digest(controls):
        raise ValueError("Actuation audit/control artifact changed")
    if spec["town"] not in maps:
        xodr = Path(carla_root) / f"CarlaUE4/Content/Carla/Maps/OpenDrive/{spec['town']}.xodr"
        maps[spec["town"]] = carla.Map(spec["town"], xodr.read_text())
    allowed = set(guard["allowed_junctions"][spec["town"]])
    for frame in lines(episode / "world_trace.jsonl"):
        ego = next(a for a in frame["actors"] if a["role"] == "hero")
        point = maps[spec["town"]].get_waypoint(carla.Location(*ego["pose"][:3]))
        if point.is_junction and point.junction_id not in allowed:
            raise ValueError("Actual physical path enters newly reserved geography")
        bound = audited["binding"]["ego_light_id"]
        if spec.get("signal_reference"):
            signal = next(a for a in frame["actors"] if a["id"] == bound)
            if (
                np.linalg.norm(np.array(signal["pose"][:3]) - spec["signal_reference"]["xyz"])
                > 0.02
            ):
                raise ValueError("Bound light differs from declared physical position")
        if any(
            a.get("signal") != "Red"
            for a in frame["actors"]
            if a["type"] == "traffic.traffic_light" and a["id"] != bound
        ):
            raise ValueError("Other lights differ from common-red setup")
    if not prefix_valid(spec, audited):
        raise ValueError("Measured handoff outside declared prefix bounds")
    scored = [
        r for r in lines(episode / "controlled_rule/setup_trace.jsonl") if not r["scripted_setup"]
    ]
    if spec["treatment"]["family"] == "out_of_view_memory":
        if not all(
            boxes and all(not b["in_image"] for b in boxes)
            for boxes in scored[0]["housing_frusta"].values()
        ):
            raise ValueError("Memory light still in view at handoff")
        if spec["treatment"]["arm"] == 0 and (
            not audited["all_applicable_housings_outside_every_view"]
            or audited["actual_scored_seconds"] < 8 - 1e-5
        ):
            raise ValueError("Red-memory phenomenon absent for full8s")
    paths = [
        native / "manifest.json",
        native / "source_hashes.json",
        controls,
        episode / "world_trace.jsonl",
        episode / "controlled_rule/setup_trace.jsonl",
        episode / "controlled_rule/binding.json",
        episode / "controlled_rule/dynamic_setup_acknowledgment.json",
        *sorted((native / "recordings").glob("*.log")),
        *sorted((episode / "controlled_rule/raw_setup_rgb").glob("*.png")),
    ]
    if resource_change:
        paths += [ROOT / "reports/navigation_local_storage_plan_20261020.json",
                  ROOT / "outputs/navigation_compact_execution_20261020/storage_guard/request.json"]
    return dict(
        case=case,
        spec=spec,
        audit=audited,
        native=str(native),
        evidence_sha256={str(p.resolve()): digest(p) for p in paths},
        camera_config=read(native / "manifest.json")["configuration"]["camera"],
        native_setup_functions_equivalent=True,
        source_equivalence_exceptions=sorted(exceptions),
        approved_resource_only_configuration_change=resource_change,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--actuation", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify all evidence/pairs without releasing setup certificates")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Never replace a frozen physical rule suite")
    args.out = args.out.resolve()
    original = read(ROOT / "outputs/navigation_rule_cases_v3_20261019/cases.json")["cases"]
    repaired = read(ROOT / "outputs/navigation_memory_cases_20261019/cases.json")["cases"]
    cases = [c for c in original if c["family"] != "out_of_view_memory"] + repaired
    if len(cases) != 24:
        raise ValueError("Exactly24 setup arms and72 prospective trials required")
    reference = {
        r["case_sha256"]: r for root in args.references for r in read(root / "records.json")
    }
    actuation = {
        str(Path(r["output"]).resolve()): r
        for path in args.actuation
        for r in read(path)["records"]
    }
    guard_path = (
        ROOT / "outputs/navigation_spatial_data_repaired_20261019/actual_training_guard.json"
    )
    guard = read(guard_path)
    vendor = activate()
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "certifier_source.py")
    atomic_json(
        args.out / "request.json",
        dict(
            cases=cases,
            source_sha256=digest(__file__),
            actual_guard_sha256=digest(guard_path),
            references_sha256={
                str(r / "records.json"): digest(r / "records.json") for r in args.references
            },
            actuation_sha256={str(p): digest(p) for p in args.actuation},
            no_student_outcomes=True,
            preflight_only=args.preflight_only,
            expert_success_never_a_selection_condition=True,
            yellow_kinematics="Requested6m/s; actual speeds reported and matched within.2m/s",
        ),
    )
    maps, audits, errors = {}, {}, []
    for case in cases:
        try:
            audits[str(case["path"])] = physical_audit(
                case, reference, actuation, maps, guard, vendor["carla_root"]
            )
        except Exception as error:
            errors.append(dict(case=case["path"], error=str(error)))
    atomic_json(args.out / "physical_audits.json", dict(records=audits, errors=errors))
    if errors:
        atomic_json(
            args.out / "assessment.json",
            dict(
                status="invalid_setup", errors=errors, certified_cases=0, no_student_outcomes=True
            ),
        )
        raise ValueError("Physical setup errors retained; no certificate released")
    groups = {}
    for row in audits.values():
        case = row["case"]
        groups.setdefault((case["family"], case["location"]), []).append(row)
    certificates = []
    runtime_hashes = {p: digest(ROOT / p) for p in SETUP_RUNTIME_SOURCES}
    for (family, location), pair in sorted(groups.items()):
        pair.sort(key=lambda r: r["case"]["arm"])
        if len(pair) != 2:
            raise ValueError("Both intervention arms required")
        validate_pair(*(r["spec"]["treatment"] for r in pair))
        if pair[0]["camera_config"] != pair[1]["camera_config"] or {
            k: v for k, v in pair[0]["spec"].items() if k != "treatment"
        } != {k: v for k, v in pair[1]["spec"].items() if k != "treatment"}:
            raise ValueError("Physical pair differs beyond registered treatment")
        if (
            family == "yellow_distance"
            and abs(
                pair[0]["audit"]["actual_handoff"]["speed_mps"]
                - pair[1]["audit"]["actual_handoff"]["speed_mps"]
            )
            > 0.2
        ):
            raise ValueError("Yellow distance pair has unmatched actual speeds")
        if args.preflight_only:
            continue
        destinations = []
        for row in pair:
            folder = args.out / f"{location}_{family}_a{row['case']['arm']}"
            folder.mkdir()
            for name in ("case.json", "case.sha256", "route.xml", "route.json"):
                shutil.copyfile(Path(row["case"]["path"]).parent / name, folder / name)
            destinations.append(folder / "case.json")
        paired = {str(p): digest(p) for p in destinations}
        evidence = {p: h for r in pair for p, h in r["evidence_sha256"].items()}
        evidence[str(args.out / "physical_audits.json")] = digest(args.out / "physical_audits.json")
        evidence.update({str(p.resolve()): digest(p) for p in args.actuation})
        for row, path in zip(pair, destinations):
            certificate = dict(
                schema="navigation-physical-setup-certificate-v2",
                case_sha256=digest(path),
                synchronized_frames=True,
                declared_actor_setup=True,
                physical_signal_binding=True,
                one_factor_pair_verified=True,
                prefix_bounds_verified=True,
                source_hashes_verified=True,
                paired_case_sha256=paired,
                evidence_sha256=evidence,
                runtime_source_sha256=runtime_hashes,
                native_reference=row["native"],
                measured_handoff=row["audit"]["actual_handoff"],
                expert_success_used_to_select_case=False,
                student_success_claim=False,
            )
            atomic_json(path.parent / "setup_certificate.json", certificate)
            certificates.append(
                dict(
                    path=str(path),
                    sha256=digest(path),
                    certificate_sha256=digest(path.parent / "setup_certificate.json"),
                    family=family,
                    location=location,
                    arm=row["case"]["arm"],
                )
            )
    if args.preflight_only:
        atomic_json(args.out / "assessment.json", dict(
            status="preflight_complete", verified_arms=len(audits), verified_pairs=len(groups),
            certified_cases=0, no_student_results=True, no_certificates_released=True,
        ))
        print(json.dumps(dict(verified_arms=len(audits), certified_cases=0)), flush=True)
        return
    atomic_json(
        args.out / "cases.json",
        dict(
            cases=certificates,
            seeds=[61001, 61002, 61003],
            student_trials=72,
            no_student_results=True,
            setup_certified=True,
        ),
    )
    (args.out / "cases.sha256").write_text(digest(args.out / "cases.json") + "\n")
    atomic_json(
        args.out / "assessment.json",
        dict(
            status="complete",
            certified_cases=len(certificates),
            no_student_results=True,
            invalid_original_memory_retained=True,
        ),
    )
    print(json.dumps(dict(certified_cases=len(certificates), student_results=0)), flush=True)


if __name__ == "__main__":
    main()

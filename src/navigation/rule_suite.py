"""Controlled rule-test contracts; independent from canonical encounter scoring.

Pairs change one registered treatment. Scripted prefix states may consequently
differ; such tests are component interventions, not autonomous route successes.
"""

from copy import deepcopy

from src.navigation.rule_prefix import PrefixSpec

SEEDS = (61001, 61002, 61003)
FAMILIES = (
    "empty_light",
    "green_stopped_lead",
    "out_of_view_memory",
    "yellow_distance",
    "stop_and_release",
    "green_clearing_cross",
)
SETUP_RUNTIME_SOURCES = (
    "scripts/run_navigation.py",
    "src/navigation/harness.py",
    "src/navigation/rule_case_runtime.py",
    "src/navigation/rule_case_contract.py",
    "src/navigation/rule_prefix.py",
    "src/navigation/rule_geometry.py",
    "src/navigation/rule_suite.py",
    "src/navigation/controller.py",
    "src/navigation/sensors.py",
    "src/navigation/native_opportunities.py",
    "configs/navigation.yaml",
    "configs/navigation_vendor.json",
    "patches/navigation_garage.patch",
)


def arms(family):
    if family not in FAMILIES:
        raise ValueError("Unknown registered rule-test family")
    common = dict(
        family=family,
        background="controlled_empty",
        other_lights="Red",
        rule_overrides=False,
        emergency_overrides=False,
        prefix=dict(mode="none"),
        ego_signal="Green",
        lead=False,
        crossing=False,
        evaluated_seconds=60.0,
    )
    if family == "empty_light":
        field, values = "ego_signal", ("Red", "Green")
        common["signal_hold_seconds"] = 20.0
    elif family == "green_stopped_lead":
        field, values = "lead", (False, True)
        common.update(lead_distance_m=15.0, lead_release_seconds=12.0)
    elif family == "out_of_view_memory":
        field, values = "ego_signal", ("Red", "Green")
        common.update(
            prefix=dict(
                mode="stop_before_line",
                speed_mps=6.0,
                handoff_distance_m=3.0,
                stationary_seconds=1.0,
                maximum_seconds=30.0,
            ),
            evaluated_seconds=8.0,
            all_view_housings_outside_after_handoff_required=True,
            signal_hold_seconds=120.0,
        )
    elif family == "yellow_distance":
        field, values = (
            "prefix",
            (
                dict(
                    mode="cruise_to_handoff",
                    speed_mps=6.0,
                    handoff_distance_m=6.0,
                    stationary_seconds=0.0,
                    maximum_seconds=30.0,
                ),
                dict(
                    mode="cruise_to_handoff",
                    speed_mps=6.0,
                    handoff_distance_m=35.0,
                    stationary_seconds=0.0,
                    maximum_seconds=30.0,
                ),
            ),
        )
        common.update(
            ego_signal="Yellow",
            yellow_seconds=3.0,
            evaluated_seconds=12.0,
            treatment_leaf="prefix.handoff_distance_m",
            prefix_signal="Green",
        )
    elif family == "stop_and_release":
        field, values = (
            "prefix",
            (
                dict(
                    mode="stop_before_line",
                    speed_mps=3.0,
                    handoff_distance_m=3.0,
                    stationary_seconds=1.0,
                    maximum_seconds=30.0,
                ),
                dict(
                    mode="cruise_to_handoff",
                    speed_mps=3.0,
                    handoff_distance_m=3.0,
                    stationary_seconds=1.0,
                    maximum_seconds=30.0,
                ),
            ),
        )
        common.update(
            evaluated_seconds=12.0,
            treatment_leaf="prefix.mode",
            ego_signal="None",
            native_stop_criterion_includes_prefix=True,
            prefix_treatment="Already stopped versus still rolling at the same line offset",
        )
    else:
        field, values = "crossing", (False, True)
        common.update(
            prefix=dict(
                mode="cruise_to_handoff",
                speed_mps=4.0,
                handoff_distance_m=8.0,
                stationary_seconds=0.0,
                maximum_seconds=30.0,
            ),
            prefix_signal="Red",
            evaluated_seconds=20.0,
            crossing_speed_mps=4.0,
            crossing_controller="Path tracking with ego-responsive braking; no recorder playback",
        )
    result = []
    for index, value in enumerate(values):
        arm = deepcopy(common)
        arm[field] = deepcopy(value)
        arm["arm"] = index
        arm["treatment"] = field
        result.append(arm)
    validate_pair(*result)
    return result


def differences(first, second, prefix=""):
    changed = []
    for key in set(first) | set(second):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(first.get(key), dict) and isinstance(second.get(key), dict):
            changed.extend(differences(first[key], second[key], path))
        elif first.get(key) != second.get(key):
            changed.append(path)
    return sorted(changed)


def validate_pair(first, second):
    for arm in (first, second):
        PrefixSpec(**arm["prefix"]).validate()
        if (
            arm["family"] not in FAMILIES
            or arm["background"] != "controlled_empty"
            or arm["rule_overrides"]
            or arm["emergency_overrides"]
        ):
            raise ValueError("Controlled-rule contract drift")
    changed = differences(first, second)
    expected = first.get("treatment_leaf", first["treatment"])
    if changed != sorted(["arm", expected]):
        raise ValueError(f"Pair differs beyond its treatment:{changed}")
    return True


def setup_certificate_valid(certificate, case_path=None):
    """Setup validity cannot depend on whether the reference driver succeeded."""
    required = (
        "synchronized_frames",
        "declared_actor_setup",
        "physical_signal_binding",
        "one_factor_pair_verified",
        "prefix_bounds_verified",
        "source_hashes_verified",
    )
    if not all(certificate.get(key) is True for key in required):
        return False
    if case_path is None:
        return True
    # The native student entry point requires evidence-bound certificates;
    # booleans alone are only suitable for inspecting individual setup checks.
    from pathlib import Path

    from src.navigation.dagger import digest

    root = Path(__file__).resolve().parents[2]
    case_path = Path(case_path).resolve()
    try:
        evidence = certificate["evidence_sha256"]
        sources = certificate["runtime_source_sha256"]
        paired = certificate["paired_case_sha256"]
        if (
            certificate["schema"] != "navigation-physical-setup-certificate-v2"
            or str(case_path) not in paired
            or paired[str(case_path)] != digest(case_path)
            or certificate["case_sha256"] != digest(case_path)
            or len(paired) != 2
            or not evidence
            or set(sources) != set(SETUP_RUNTIME_SOURCES)
        ):
            return False
        return (
            all(digest(Path(path)) == expected for path, expected in evidence.items())
            and all(digest(Path(path)) == expected for path, expected in paired.items())
            and all(digest(root / path) == expected for path, expected in sources.items())
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False

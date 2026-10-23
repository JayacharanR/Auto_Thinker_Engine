from copy import deepcopy

import pytest

from src.navigation.rule_suite import FAMILIES, arms, setup_certificate_valid, validate_pair


def test_all_six_rule_pairs_have_exactly_one_registered_treatment():
    for family in FAMILIES:
        first, second = arms(family)
        assert first["family"] == second["family"] == family
        assert validate_pair(first, second)
        assert not first["rule_overrides"] and not second["emergency_overrides"]


def test_incidental_light_changes_and_safety_overrides_invalidate_pair():
    first, second = arms("empty_light")
    bad = deepcopy(second)
    bad["other_lights"] = "Green"
    with pytest.raises(ValueError, match="beyond"):
        validate_pair(first, bad)
    bad = dict(second, rule_overrides=True)
    with pytest.raises(ValueError, match="drift"):
        validate_pair(first, bad)


def test_expert_failure_is_never_a_reason_to_filter_valid_rule_test():
    certificate = dict.fromkeys(
        (
            "synchronized_frames",
            "declared_actor_setup",
            "physical_signal_binding",
            "one_factor_pair_verified",
            "prefix_bounds_verified",
            "source_hashes_verified",
        ),
        True,
    )
    assert setup_certificate_valid(dict(certificate, expert_passed=False))
    assert not setup_certificate_valid(
        dict(certificate, declared_actor_setup=False, expert_passed=True)
    )


def test_native_certificate_rejects_flags_only_and_changed_paired_evidence(tmp_path):
    from pathlib import Path

    from src.navigation.dagger import digest
    from src.navigation.rule_suite import SETUP_RUNTIME_SOURCES

    case = tmp_path / "case.json"
    partner = tmp_path / "partner.json"
    evidence = tmp_path / "physical_trace.json"
    for path in (case, partner, evidence):
        path.write_text("{}")
    certificate = dict.fromkeys(
        (
            "synchronized_frames",
            "declared_actor_setup",
            "physical_signal_binding",
            "one_factor_pair_verified",
            "prefix_bounds_verified",
            "source_hashes_verified",
        ),
        True,
    )
    assert not setup_certificate_valid(certificate, case)
    root = Path(__file__).resolve().parents[1]
    certificate.update(
        schema="navigation-physical-setup-certificate-v2",
        case_sha256=digest(case),
        paired_case_sha256={str(p): digest(p) for p in (case, partner)},
        evidence_sha256={str(evidence): digest(evidence)},
        runtime_source_sha256={p: digest(root / p) for p in SETUP_RUNTIME_SOURCES},
    )
    assert setup_certificate_valid(certificate, case)
    partner.write_text('{"treatment":"changed"}')
    assert not setup_certificate_valid(certificate, case)

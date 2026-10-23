"""Frozen final route/candidate boundary and no silent final-data collection."""

import json

import pytest

from src.navigation.final_evaluation import (
    SUITE,
    candidate_request,
    digest,
    register_candidate,
    validate_final_request,
)


def test_final_registry_cannot_move_candidate_to_a_new_result_root(tmp_path):
    request = candidate_request(None, None, tmp_path / "results", expert=True)
    assert len(request["route_ids"]) == 12
    registry = tmp_path / "expert.json"
    register_candidate(registry, request)
    assert register_candidate(registry, request) == request
    with pytest.raises(ValueError, match="cannot be changed"):
        register_candidate(registry, dict(request, evaluation_root=str(tmp_path / "different")))


def test_unfinished_research_gates_do_not_authorize_a_final_test(tmp_path):
    checkpoint, evidence = tmp_path / "policy.pt", tmp_path / "qualification.json"
    checkpoint.write_bytes(b"synthetic")
    evidence.write_text(
        json.dumps(
            dict(candidate_sha256=digest(checkpoint), town05_used_for_training_or_selection=False)
        )
    )
    with pytest.raises(ValueError, match="gated by M2"):
        candidate_request(checkpoint, evidence, tmp_path / "results")
    with pytest.raises(ValueError, match="qualification"):
        candidate_request(checkpoint, None, tmp_path / "results")


def test_native_final_request_rejects_seed_drift_substitution_and_collection(tmp_path):
    registry = tmp_path / "expert.json"
    register_candidate(registry, candidate_request(None, None, tmp_path / "results", expert=True))
    kwargs = dict(
        registration=registry,
        route=SUITE / "Town05_Opt.xml",
        output=tmp_path / "results" / "a0",
        seed=3031,
        subset="0,1,2",
        student=None,
        collection=None,
    )
    verified = validate_final_request(**kwargs)
    assert verified["candidate_once"]
    for changed in (
        dict(seed=42),
        dict(seed_per_route=True),
        dict(collection={"mode": "expert"}),
        dict(max_steps=100),
        dict(route=tmp_path / "substitute.xml"),
        dict(output=tmp_path / "other_results"),
        dict(subset="0,12"),
    ):
        with pytest.raises(ValueError):
            validate_final_request(**dict(kwargs, **changed))


def test_qualification_sources_and_candidate_identity_are_bound(tmp_path):
    candidate, report, evidence = (
        tmp_path / "candidate.pt",
        tmp_path / "report.json",
        tmp_path / "qualification.json",
    )
    import torch

    torch.save({"policy_input_schema": "navigation-sensor-packet-v1"}, candidate)
    report.write_text("{}")
    qualified = dict(
        candidate_sha256=digest(candidate),
        town05_used_for_training_or_selection=False,
        milestones={
            stage: dict(
                evidence_complete=True,
                gate_passed=True,
                reports=[dict(path=str(report), sha256=digest(report))],
            )
            for stage in ("M2", "M3", "M4", "M5")
        },
    )
    evidence.write_text(json.dumps(qualified))
    request = candidate_request(candidate, evidence, tmp_path / "results")
    registry = tmp_path / "registered.json"
    register_candidate(registry, request)
    student = dict(sha256=digest(candidate), rule_overrides=False)
    validate_final_request(
        registry, SUITE / "Town05_Opt.xml", tmp_path / "results" / "a0", 3031, "0", student, None
    )
    with pytest.raises(ValueError, match="override"):
        validate_final_request(
            registry,
            SUITE / "Town05_Opt.xml",
            tmp_path / "results" / "a1",
            3031,
            "0",
            dict(student, rule_overrides=True),
            None,
        )
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence file changed"):
        candidate_request(candidate, evidence, tmp_path / "results")

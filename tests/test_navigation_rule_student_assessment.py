import json
import struct

import pytest

from scripts.assess_navigation_rule_student import grid, summarize
from src.navigation.rule_suite import FAMILIES, SEEDS


def request():
    trials = []
    for family in FAMILIES:
        for location in ("loc0", "loc1"):
            for arm in (0, 1):
                for seed in SEEDS:
                    trials.append(dict(
                        name=f"{family}_{location}_a{arm}_s{seed}", family=family,
                        location=location, arm=arm, seed=seed,
                    ))
    return dict(trials=trials, rule_overrides=False, checkpoint_sha256="frozen",
                architecture="spatial-visual-facts-v3", inference_source_sha256={})


def setup(tmp_path, value):
    root = tmp_path / "run"
    root.mkdir()
    (root / "request.json").write_text(json.dumps(value))
    actions = tmp_path / "actuation.json"
    actions.write_text(json.dumps(dict(records=[], errors=[])))
    return root, actions


def test_all_missing_are_failures_in_full_denominator_and_contacts_unknown(tmp_path):
    root, actions = setup(tmp_path, request())
    result = summarize(root, actions)
    assert result["planned_trials"] == result["failed_or_unverified_trials"] == 72
    assert result["evidence_complete_trials"] == result["passed_trials"] == 0
    assert result["contacts_lower_bound"] and not result["gate_passed"]
    assert all(r["planned"] == 12 for r in result["family_counts"].values())
    assert "? / ? / ?" in grid(result)


def test_duplicate_physical_identity_cannot_replace_a_missing_trial(tmp_path):
    value = request()
    value["trials"][1] = dict(value["trials"][0], name="different_name_same_physical_trial")
    root, actions = setup(tmp_path, value)
    with pytest.raises(ValueError, match="Duplicate physical"):
        summarize(root, actions)


def test_expert_terminal_summary_never_counts_as_camera_rule_success(tmp_path):
    value = request()
    root, actions = setup(tmp_path, value)
    case = root / value["trials"][0]["name"]
    case.mkdir()
    (case / "summary.json").write_text(json.dumps(dict(episodes=[dict(completed=True)])))
    (case / "manifest.json").write_text(json.dumps(dict(student=None)))
    result = summarize(root, actions)
    assert result["passed_trials"] == 0
    assert "override-disabled camera" in result["rows"][0]["evidence_error"]


def test_orphan_attempt_contact_still_counts_in_full_headline(tmp_path):
    root, actions = setup(tmp_path, request())
    folder = root / "unexpected_interrupted_attempt" / "recordings"
    folder.mkdir(parents=True)

    def packet(kind, payload):
        return struct.pack("<BI", kind, len(payload)) + payload

    def text(value):
        raw = value.encode()
        return struct.pack("<H", len(raw)) + raw

    raw = struct.pack("<H", 1) + text("CARLA_RECORDER") + struct.pack("<Q", 0) + text("Town01_Opt")
    raw += packet(0, struct.pack("<Qdd", 1, .05, 0))
    raw += packet(5, struct.pack("<HIIIBB", 1, 1, 10, 20, 1, 0))
    raw += packet(1, b"")
    (folder / "route_0.log").write_bytes(raw)
    result = summarize(root, actions)
    assert result["unfiltered_all_attempt_contact_starts"] == 1
    assert len(result["unexpected_attempt_recorders"]) == 1
    assert result["contacts_lower_bound"] and not result["gate_passed"]

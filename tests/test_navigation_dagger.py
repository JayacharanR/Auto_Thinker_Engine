"""Teacher-go corrections, no privileged camera input, and frozen validation isolation."""

import json

import numpy as np
import pytest

from src.navigation.collection import finalize_collection
from src.navigation.contracts import TrainingTargets
from src.navigation.dagger import CorrectionMixer, correction_records, internal_location_guard
from src.navigation.dataset import EpisodeWriter, training_records
from src.navigation.policy_worker import PrivilegeGuard
from tests.test_navigation_collection import packet


def test_collection_recovers_a_stationary_student_only_when_expert_would_go():
    mixer = CorrectionMixer()
    student, go = (0, 0, 1), (0.1, 0.4, 0)
    for tick in range(80):
        assert mixer.select(tick / 20, student, go, 0, 4, False) == (student, None)
    command, reason = mixer.select(4, student, go, 0, 4, False)
    assert command == go and reason == "privileged_teacher_recovery"
    assert mixer.counts["recovery_events"] == 1
    # A red/hazard arriving during recovery has priority over any move command.
    stop = (0.1, 0, 1)
    assert mixer.select(4.05, student, stop, 0.1, 0, True) == (stop, "privileged_teacher_hazard")
    assert mixer.recover_until is None
    for tick in range(82, 200):
        assert mixer.select(tick / 20, go, stop, 0, 0, True)[0] == stop
    assert mixer.counts["recovery_events"] == 1


def test_comfort_braking_is_not_overridden_by_a_positive_teacher_target():
    mixer = CorrectionMixer()
    teacher = (0.2, 0, 1)
    assert mixer.select(0, (0, 0.4, 0), teacher, 7, 4, False) == (
        teacher,
        "privileged_teacher_braking",
    )
    with pytest.raises(ValueError, match="contiguous"):
        mixer.select(0.2, (0, 0.4, 0), teacher, 7, 4, False)


def test_internal_validation_locations_remain_reserved_outcome_blindly():
    assignments = {"TownA": {"train": [1, 2, 3], "heldout_dev": [4]}}
    manifest = {
        "episodes": [
            dict(town="TownA", partition="train", role="train", junction_ids=[1]),
            dict(town="TownA", partition="train", role="validation", junction_ids=[2]),
        ]
    }
    guard = internal_location_guard(manifest, assignments)
    assert guard["allowed_junctions"] == {"TownA": [1, 3]}
    assert guard == internal_location_guard(
        {"episodes": [{**e, "success": False} for e in manifest["episodes"]]}, assignments
    )
    manifest["episodes"][0]["junction_ids"] = [2]
    with pytest.raises(ValueError, match="overlap"):
        internal_location_guard(manifest, assignments)


@pytest.mark.parametrize("collision", [False, True])
def test_assisted_rollout_is_never_clean_imitation_and_corrections_use_expert(collision, tmp_path):
    episode = tmp_path / "episodes" / "episode0"
    data = episode / "data"
    writer = EpisodeWriter(data, dict(mode="dagger", route_id=1, scenario=None), minimum_free_gb=0)
    writer.append(
        packet(),
        TrainingTargets(
            10,
            (0.1, 0.4, 0),
            (0, 0, 1),
            None,
            {"expert_label_valid": True, "teacher_target_speed_mps": 4},
        ),
    )
    writer.close()
    (episode / "collection_profile.json").write_text(
        json.dumps(
            dict(
                samples=1,
                control_differences=0,
                safety_stop_ticks=0,
                privileged_intervention_ticks=1,
            )
        )
    )
    record = dict(
        route_id="RouteScenario_1_rep0",
        completed=True,
        rule_safe_completed=not collision,
        collisions=int(collision),
    )
    (tmp_path / "summary.json").write_text(json.dumps({"episodes": [record]}))
    finalize_collection(tmp_path)
    outcome = json.loads((data / "outcome.json").read_text())
    assert not outcome["imitation_eligible"] and outcome["episode_censored"]
    assert not outcome["counterfactual_safe_label"]
    assert list(training_records(data, "imitation")) == []
    corrections = list(correction_records(data))
    assert bool(corrections) != collision
    if corrections:
        label = corrections[0][2]
        assert label["expert_control"] == [0.1, 0.4, 0]
        assert label["executed_control"] == [0, 0, 1]


@pytest.mark.parametrize(
    "name",
    ["src.navigation.dagger", "src.navigation.dagger_agent", "src.navigation.correction_learning"],
)
def test_camera_worker_cannot_import_correction_oracle(name):
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec(name)


def test_correction_cache_refuses_failed_exposure_gate_before_reading_data(tmp_path):
    from src.navigation.correction_learning import prepare_corrections

    report = tmp_path / "assessment.json"
    report.write_text(json.dumps(dict(audit_passed=True, correction_data_gate_passed=False)))
    with pytest.raises(ValueError, match="exposure gate"):
        prepare_corrections(report, tmp_path / "cache", tmp_path, tmp_path / "STOP", 0, None)
    assert not (tmp_path / "cache").exists()


def test_correction_sampler_stays_inside_episodes_and_never_supervises_mixture_future(tmp_path):
    from src.navigation.correction_learning import CorrectionPool

    episodes = [dict(start=0, end=48), dict(start=48, end=96)]
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": episodes}))
    for key, shape, dtype in [
        ("rgb", (96, 3, 3, 8, 16), np.uint8),
        ("front", (96, 3, 16, 32), np.uint8),
        ("navigation", (96, 16, 2), np.float32),
        ("ego", (96, 3), np.float32),
        ("speed_mask", (96, 8), np.float32),
    ]:
        np.save(tmp_path / f"{key}.npy", np.zeros(shape, dtype))
    speed, priority = np.ones(96, np.float32) * 4, np.zeros(96, np.float32)
    for offset in (0, 48):
        speed[offset + 16 : offset + 36] = 0
        priority[offset + 36 : offset + 41] = 1
    np.save(tmp_path / "current_speed.npy", speed)
    np.save(tmp_path / "priority_go.npy", priority)
    pool = CorrectionPool(tmp_path, 32)
    starts = pool.sample(np.random.default_rng(7), 100)
    assert all(any(e["start"] <= i and i + 32 <= e["end"] for e in episodes) for i in starts)
    endpoints = np.asarray(starts) + 31
    assert (speed[endpoints] == 0).any() and priority[endpoints].any()
    inputs, targets = pool.batch(starts[:2], "cpu")
    assert "priority_go" not in inputs and "priority_go" not in targets
    assert not targets["speed_mask"].any()
    mask = np.zeros((96, 8), np.float32)
    mask[0, 0] = 1
    np.save(tmp_path / "speed_mask.npy", mask)
    with pytest.raises(ValueError, match="future labels"):
        CorrectionPool(tmp_path, 32)

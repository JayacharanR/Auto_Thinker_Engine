"""Training-only gap boundaries and the actual inherited IDM parameter path."""

import copy
from types import SimpleNamespace

import pytest

from src.navigation.collection import training_light_gap
from src.navigation.policy_worker import PrivilegeGuard


def scope(gap=12):
    return dict(
        collection=dict(
            background="empty",
            mode="expert",
            training_light_gap_m=gap,
            learning_guard=dict(kind="continuous-approach-train-v1", guard_sha256="fixture"),
        )
    )


def test_gap_requires_guarded_empty_expert_and_never_enters_camera_policy():
    assert training_light_gap({}) is None and training_light_gap(scope()) == 12
    for value in (0, 5.9, 16.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            training_light_gap(scope(value))
    for change in (dict(background="canonical"), dict(mode="noisy"), dict(learning_guard=None)):
        spec = scope()
        spec["collection"].update(change)
        with pytest.raises(ValueError):
            training_light_gap(spec)
    with pytest.raises(ValueError):
        training_light_gap(dict(scope(), student=dict(checkpoint="camera.pt")))
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.training_gap_agent")


def test_training_teacher_passes_gap_to_inherited_idm_without_changing_reference(monkeypatch):
    from src.navigation.vendor import activate

    activate()
    import carla

    from src.navigation.collection_agent import CollectionExpert
    from src.navigation.training_gap_agent import TrainingGapExpert

    baseline = SimpleNamespace(
        idm_red_light_minimum_distance=6.0, idm_red_light_desired_time_headway=0.1
    )

    def setup(self, *args):
        self.navigation_spec = scope()
        self.config = copy.deepcopy(baseline)
        self.recorded_metadata = self.collection_metadata({})

    monkeypatch.setattr(CollectionExpert, "setup", setup)
    teacher = TrainingGapExpert.__new__(TrainingGapExpert)
    teacher.setup("fixture.json")
    assert (
        teacher.config.idm_red_light_minimum_distance == 12
        and baseline.idm_red_light_minimum_distance == 6
    )
    assert teacher.recorded_metadata["standard_light_gap_m"] == 6
    assert teacher.recorded_metadata["driver"] == "privileged_pdm_lite_training_gap"
    assert teacher.extra_facts()["training_light_gap_m"] == 12
    teacher.close_traffic_lights, teacher.list_traffic_lights = [], []
    seen = {}

    def idm(**kwargs):
        seen.update(kwargs)
        return 0.0

    teacher._compute_target_speed_idm = idm
    red = SimpleNamespace(state=carla.TrafficLightState.Red)
    assert teacher.ego_agent_affected_by_red_light(None, 4.0, 18.0, red, None, 8.0) == 0
    assert seen["s0"] == 12 and seen["distance_to_leading_actor"] == 18
    green = SimpleNamespace(state=carla.TrafficLightState.Green)
    assert teacher.ego_agent_affected_by_red_light(None, 4.0, 18.0, green, None, 8.0) == 8

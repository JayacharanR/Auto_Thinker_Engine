"""Camera projection checks independent of labels; static geometry stays privileged."""

import json
import math

import numpy as np
import pytest

from src.navigation.contracts import CameraCalibration
from src.navigation.policy_worker import PrivilegeGuard
from src.navigation.rule_geometry import (
    camera_matrix,
    ego_pose_from_facts,
    match_light,
    project_box,
    project_points,
    project_rule_sample,
    stopped_runs,
    transform_matrix,
)


def calibration(yaw=0):
    return CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, yaw, 0))


def test_known_pixel_axes_and_behind_camera_exclusion():
    c = calibration()
    pixels, depth = project_points([[11.3, 0, 2.3], [11.3, 1, 3.3], [-1, 0, 0]], [0] * 6, c)
    assert np.allclose(pixels[0], [256, 144])
    assert np.allclose(pixels[1], [256 + c.intrinsic[0, 0] / 10, 144 - c.intrinsic[1, 1] / 10])
    assert np.isnan(pixels[2]).all() and depth[2] < 0


@pytest.mark.parametrize("yaw", [-60, 60])
def test_side_camera_uses_rotated_extrinsics_and_translation(yaw):
    angle = math.radians(yaw)
    point = [1.3 + 10 * math.cos(angle), 10 * math.sin(angle), 2.3]
    pixels, depth = project_points([point], [0] * 6, calibration(yaw))
    assert np.allclose(pixels, [[256, 144]]) and depth[0] == pytest.approx(10)


def test_full_pitch_roll_composition_matches_installed_carla():
    from src.navigation.vendor import activate

    activate()
    carla = pytest.importorskip("carla")
    rng = np.random.default_rng(9)
    for _ in range(32):
        pose = [*rng.uniform(-200, 200, 3), *rng.uniform(-180, 180, 3)]
        reference = carla.Transform(
            carla.Location(*pose[:3]), carla.Rotation(pitch=pose[3], yaw=pose[4], roll=pose[5])
        )
        assert np.allclose(transform_matrix(pose), reference.get_matrix(), atol=2e-5)
        assert np.allclose(
            np.linalg.inv(transform_matrix(pose)), reference.get_inverse_matrix(), atol=8e-5
        )
    # Distinct pitched/rolled ego and sensor must compose in the declared order.
    c = CameraCalibration(512, 288, 100, (1.3, 0.2, 2.3), (5, -60, 3))
    assert np.allclose(
        camera_matrix(pose, c),
        np.asarray(reference.get_matrix()) @ transform_matrix([*c.xyz, *c.pitch_yaw_roll]),
        atol=2e-5,
    )


def box_at(x, y=0, z=2.3, size=0.2):
    return [
        [x + a * size, y + b * size, z + c * size]
        for a in (-1, 1)
        for b in (-1, 1)
        for c in (-1, 1)
    ]


def test_frustum_overlap_is_distinct_from_near_plane_indeterminacy():
    c = calibration()
    assert project_box(box_at(11.3), [0] * 6, c)["in_image"] is True
    assert project_box(box_at(11.3, y=100), [0] * 6, c)["in_image"] is False
    assert project_box(box_at(-10), [0] * 6, c)["status"] == "behind_near_plane"
    assert project_box(box_at(1.3), [0] * 6, c)["in_image"] is None
    with pytest.raises(ValueError, match="Eight"):
        project_box([[1, 2, 3]], [0] * 6, c)


def test_cross_server_signal_identity_requires_full_pose_and_unique_match():
    actor = {
        "id": 7,
        "type": "traffic.traffic_light",
        "pose": [1, 2, 3, 0, 180, 0],
        "signal": "Red",
    }
    row = {"capture_actor_id": 99, "pose": [1, 2, 3, 0, -180, 0]}
    assert match_light(actor, {"lights": [row]}) is row
    with pytest.raises(ValueError, match="unique"):
        match_light(actor, {"lights": [row, row]})
    with pytest.raises(ValueError, match="unique"):
        match_light(actor, {"lights": [{**row, "pose": [1, 2, 3, 0, 170, 0]}]})


def test_projection_preserves_ego_pitch_and_roll_and_checks_same_frame_pose():
    pose = [1, 2, 3, 5, 90, -2]
    facts = {
        "world": {"actors": [{"type": "vehicle.test", "role": "hero", "pose": pose}]},
        "ego_pose_world": [1, 2, 3, math.pi / 2],
    }
    assert ego_pose_from_facts(facts) == pose
    facts["ego_pose_world"][0] += 1
    with pytest.raises(ValueError, match="mismatch"):
        ego_pose_from_facts(facts)


def test_long_stop_and_release_remain_one_event_and_end_censoring_is_explicit():
    rows = [
        {
            "frame": i * 2,
            "timestamp": i / 10,
            "teacher_target_speed_mps": 0 if i < 100 or i >= 102 else 4,
            "teacher_hazards": {"light_hazard": True},
        }
        for i in range(110)
    ]
    runs = stopped_runs(rows)
    assert len(runs) == 2
    assert runs[0]["duration_s"] == pytest.approx(10)
    assert runs[0]["longer_than_3_2s"] and runs[0]["release_frame"] == 200
    assert runs[1]["release_observed"] is False


def test_camera_worker_cannot_import_geometry_oracle():
    with pytest.raises(ImportError, match="Forbidden"):
        PrivilegeGuard().find_spec("src.navigation.rule_geometry")


def test_projected_housing_is_not_certified_as_visible_or_released_as_state_supervision():
    facts = {
        "world": {
            "frame": 10,
            "actors": [
                {"id": 4, "role": "hero", "type": "vehicle.test", "pose": [0] * 6},
                {
                    "id": 7,
                    "role": "",
                    "type": "traffic.traffic_light",
                    "pose": [11.3, 0, 2.3, 0, 0, 0],
                    "signal": "Red",
                },
            ],
        },
        "ego_pose_world": [0] * 4,
        "applicable_light_id": 7,
        "next_light_state": "Red",
    }
    geometry = {
        "lights": [
            {
                "capture_actor_id": 99,
                "opendrive_id": "42",
                "pose": [11.3, 0, 2.3, 0, 0, 0],
                "boxes_world": [{"vertices": box_at(11.3)}],
                "evaluator_stop_lines": [],
            }
        ]
    }
    result = project_rule_sample(facts, {"front": calibration()}, geometry)
    assert result["views"]["front"]["housing_in_frustum"] is True
    assert result["visibility_valid"] is None
    assert result["state_supervision_approved"] is False
    geometry["lights"][0]["boxes_world"] = []
    assert (
        project_rule_sample(facts, {"front": calibration()}, geometry)["views"]["front"][
            "housing_in_frustum"
        ]
        is None
    )
    facts["next_light_state"] = "Green"
    with pytest.raises(ValueError, match="identity/state"):
        project_rule_sample(facts, {"front": calibration()}, geometry)


def test_paired_color_contrast_reports_repeat_control_and_clips_image_bounds():
    from scripts.probe_navigation_signal_pixels import contrast

    red = np.zeros((8, 8, 3), np.uint8)
    green = red.copy()
    red[4, 4], green[4, 4] = [220, 0, 0], [0, 220, 0]
    result = contrast(red, green, red.copy(), [4, 4, 6, 6])
    assert result["red_green_pixels_change_gt10"] == 1
    assert result["red_repeat_pixels_change_gt10"] == 0
    assert result["red_green_mae"] == pytest.approx(440 / 12)
    assert contrast(red, green, red, [-3, -2, 1, 1])["crop_size"] == [1, 1]
    assert contrast(red, green, red, [12, 12, 20, 20]) is None


def test_static_pair_selection_rejects_changed_audit_before_reading_any_poses(tmp_path):
    from scripts.probe_navigation_signal_pixels import select_cases

    (tmp_path / "samples.jsonl").write_text("")
    (tmp_path / "assessment.json").write_text(
        json.dumps({"audit_complete": True, "provenance": {"samples_sha256": "wrong"}})
    )
    with pytest.raises(ValueError, match="unchanged"):
        select_cases(tmp_path, {})


def test_resource_guard_excludes_self_but_refuses_another_probe(monkeypatch, tmp_path):
    import os
    from types import SimpleNamespace

    from src.navigation import runtime

    def proc(pid):
        return SimpleNamespace(
            pid=pid,
            info={"cmdline": ["python", "-u", "scripts/probe_example.py"], "status": "running"},
        )

    monkeypatch.setattr(runtime.psutil, "process_iter", lambda fields: [proc(os.getpid())])
    assert runtime.check_resources(tmp_path, 0) > 0
    monkeypatch.setattr(
        runtime.psutil,
        "process_iter",
        lambda fields: [proc(os.getpid()), proc(os.getpid() + 10000)],
    )
    with pytest.raises(RuntimeError, match="Existing simulation/training process"):
        runtime.check_resources(tmp_path, 0)


def test_static_probe_restores_world_ticking_after_a_setup_failure(monkeypatch, tmp_path):
    import hashlib
    import sys
    from types import SimpleNamespace

    import scripts.probe_navigation_signal_pixels as probe

    (tmp_path / "geometry.json").write_text(
        json.dumps({"opendrive_sha256": hashlib.sha256(b"map").hexdigest()})
    )
    applied = []
    world = SimpleNamespace(
        get_map=lambda: SimpleNamespace(name="Town03_Opt", to_opendrive=lambda: "map"),
        get_settings=lambda: SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None),
        apply_settings=lambda settings: applied.append(
            (settings.synchronous_mode, settings.fixed_delta_seconds)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(
            Client=lambda host, port: SimpleNamespace(
                set_timeout=lambda timeout: None, get_world=lambda: world
            )
        ),
    )

    def fail(*args):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(probe, "render_cases", fail)
    with pytest.raises(RuntimeError, match="setup failed"):
        probe.worker(SimpleNamespace(geometry=tmp_path), {"runtime": {"port": 2100}})
    assert applied == [(True, 0.05), (False, None)]


def test_static_probe_drains_the_requested_frame_with_delayed_gpu_output(tmp_path):
    from types import SimpleNamespace

    from scripts.probe_navigation_signal_pixels import capture_static_frame
    from src.navigation.sensors import FrameBarrier

    barrier = FrameBarrier()

    class DelayedWorld:
        frame = 0

        def tick(self, timeout):
            self.frame += 1
            if self.frame > 2:
                captured = self.frame - 2
                for view in barrier.names:
                    barrier.put(view, captured, captured / 20, np.array([captured]))
            return self.frame

    world = DelayedWorld()
    frame, packet = capture_static_frame(
        world, SimpleNamespace(barrier=barrier), tmp_path / "STOP", 10
    )
    assert world.frame == 12 and frame == 10
    assert all(value[0] == frame and value[2][0] == frame for value in packet.values())
    (tmp_path / "STOP").touch()
    with pytest.raises(InterruptedError, match="STOP"):
        capture_static_frame(world, SimpleNamespace(barrier=barrier), tmp_path / "STOP", 4)


def test_static_response_inputs_equal_production_camera_preprocessing():
    import torch

    from scripts.assess_navigation_signal_response import approved_inputs
    from src.navigation.contracts import VIEWS, SensorPacket
    from src.navigation.policy_runtime import CameraPolicy

    images = {
        view: np.random.default_rng(i).integers(0, 256, (288, 512, 3), dtype=np.uint8)
        for i, view in enumerate(VIEWS)
    }
    nav = np.column_stack([np.arange(2, 34, 2), np.zeros(16)]).astype(np.float32)
    inputs = approved_inputs(images, nav, 0.2, "cpu", steps=1)
    seen = {}

    class Network:
        def __call__(self, **kwargs):
            seen.update({key: value for key, value in kwargs.items() if key != "hidden"})
            prediction = {
                key: torch.zeros(1, 1, count)
                for key, count in [
                    ("correction", 8),
                    ("speeds", 8),
                    ("hazards", 4),
                    ("light", 5),
                    ("distance", 2),
                    ("presence", 2),
                ]
            }
            prediction.update(current_speed=torch.ones(1, 1), brake=torch.ones(1, 1))
            return prediction, None

    policy = CameraPolicy.__new__(CameraPolicy)
    policy.device, policy.model, policy.rule_overrides = "cpu", Network(), False
    policy.fact_arm = None  # Legacy-camera fixture bypasses __init__ intentionally.
    policy.hidden, policy.previous, policy.ticks = None, None, 0
    packet = SensorPacket(
        10, 0.5, images, {v: calibration() for v in VIEWS}, nav, np.array([0, 0.2, 0], np.float32)
    )
    policy.step(packet)
    assert set(inputs) == set(seen) == {"rgb", "front", "navigation", "ego"}
    assert all(torch.equal(value, seen[key]) for key, value in inputs.items())


def test_signal_response_rejects_an_incomplete_pixel_probe_before_loading_model(tmp_path):
    from scripts.assess_navigation_signal_response import assess

    (tmp_path / "assessment.json").write_text(json.dumps({"probe_complete": False}))
    (tmp_path / "request.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed paired"):
        assess(tmp_path, tmp_path / "missing_policy.pt", tmp_path / "out", device="cpu")
    assert not (tmp_path / "out").exists()

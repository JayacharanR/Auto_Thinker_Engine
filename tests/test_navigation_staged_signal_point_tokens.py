import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.navigation.contracts import CameraCalibration, SensorPacket
from src.navigation.rule_geometry import project_points

PATH = Path(__file__).resolve().parents[1] / "patches/navigation_signal_point_tokens_candidate.py"
SPEC = importlib.util.spec_from_file_location("staged_point_tokens", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def packet():
    return SensorPacket(
        10,
        0.5,
        {v: np.zeros((288, 512, 3), np.uint8) for v in MODULE.VIEWS},
        {
            v: CameraCalibration(512, 288, 100, (1.3, 0.0, 2.3), (0.0, yaw, 0.0))
            for v, yaw in zip(MODULE.VIEWS, (0.0, -60.0, 60.0))
        },
        np.zeros((16, 2)),
        np.zeros(3),
    )


def observation(**updates):
    return dict(
        dict(
            view="front",
            x=256,
            y=144,
            probabilities=[0.1, 0.8, 0.05, 0.05],
            frame=10,
            timestamp=0.5,
        ),
        **updates,
    )


def test_rays_reproject_without_inventing_depth_for_all_camera_rotations():
    for angles in ((0, 0, 0), (0, -60, 0), (0, 60, 0), (13, 29, -7)):
        c = CameraCalibration(512, 288, 100, (1.3, 0.1, 2.3), angles)
        for x, y in ((256, 144), (0, 0), (511, 287), (123, 78)):
            direction, origin = MODULE.calibrated_ray(x, y, c)
            assert np.isclose(np.linalg.norm(direction), 1)
            for distance in (5.0, 20.0, 60.0):
                pixels, depth = project_points(
                    np.array([origin + distance * direction]), np.zeros(6), c
                )
                assert depth[0] > 0
                np.testing.assert_allclose(pixels[0], [x, y], atol=3e-5)


def test_no_300_object_truncation_padding_and_null_behavior():
    data = packet()
    empty = MODULE.pack_current_points(data, [])
    many = MODULE.pack_current_points(data, [observation() for _ in range(501)])
    batch, valid = MODULE.pad_point_batch([empty, many])
    assert batch.shape == (2, 502, 16)
    assert valid.sum(axis=1).tolist() == [1, 502]
    np.testing.assert_array_equal(batch[1, 1, 6:10], np.array([0.1, 0.8, 0.05, 0.05], np.float32))
    assert not batch[0, 1:].any()


def test_stale_future_teacher_and_invalid_observations_rejected():
    data = packet()
    for change in (
        dict(frame=9),
        dict(frame=11),
        dict(timestamp=0.4),
        dict(timestamp=float("nan")),
        dict(actor_id=1),
        dict(x=512),
        dict(x=1.5),
        dict(view="rear"),
        dict(probabilities=[0.6, 0.2, 0.1, 0.1]),
        dict(probabilities=[0, 1, float("nan"), 0]),
    ):
        with pytest.raises(ValueError):
            MODULE.pack_current_points(data, [observation(**change)])
    assert MODULE.pack_current_points(data, [observation()])[0].shape == (2, 16)

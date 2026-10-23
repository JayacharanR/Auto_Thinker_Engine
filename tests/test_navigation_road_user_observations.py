"""Unlocated RGB detections cannot silently disappear or masquerade as free space."""

from dataclasses import replace

import numpy as np
import pytest

from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
from src.navigation.road_user_observations import observation_pack


def packet():
    return SensorPacket(
        7,
        0.35,
        {v: np.zeros((288, 512, 3), np.uint8) for v in VIEWS},
        {
            v: CameraCalibration(512, 288, 100.0, (1.3, 0.0, 2.3), (0.0, yaw, 0.0))
            for v, yaw in zip(VIEWS, (0.0, -60.0, 60.0))
        },
        np.array([[3.0, 0.0], [6.0, 0.0]], np.float32),
        np.zeros(3, np.float32),
    )


def test_unlocated_objects_remain_present_separate_from_padding_and_empty_scene():
    p = packet()
    rows = [
        dict(view="front", label=3, score=0.9, box=[240, 120, 270, 144]),
        dict(view="front", label=3, score=0.8, box=[240, 200, 270, 288]),
        dict(view="left", label=3, score=0.7, box=[240, 145, 270, 146]),
    ]
    result = observation_pack(p, rows)
    assert result["presence"].sum() == 3 and result["unknown_ground_contacts"] == 3
    assert result["presence"][0, 0] and result["values"][0, 0, 14] == 0
    assert result["values"][0, 0, 0] > 0 and result["values"][0, 0, 18] == 1
    assert result["values"][0, 1, 17] == 1 and result["values"][1, 0, 19] == 1
    assert result["observations"][1]["ground_status"] == "image_bottom_unknown"
    assert observation_pack(p, [])["presence"].sum() == 0
    # Own navigation and speed cannot inject object/world facts into this pack.
    changed = replace(
        p,
        navigation_waypoints=p.navigation_waypoints + 20,
        ego_motion=np.array([12.0, 1.0, 0.2], np.float32),
    )
    assert np.array_equal(observation_pack(changed, rows)["values"], result["values"])


def test_no_privileged_fields_or_hidden_truncation_and_rays_keep_camera_identity():
    p = packet()
    row = dict(view="front", label=3, score=0.9, box=[245, 150, 267, 210])
    with pytest.raises(ValueError, match="Only RGB detector"):
        observation_pack(p, [row | dict(actor_id=32)])
    with pytest.raises(ValueError, match="Never truncate"):
        observation_pack(p, [row] * 101)
    result = observation_pack(p, [row, row | dict(view="left"), row | dict(view="right")])
    ray = result["values"][:, 0, 11:14]
    assert np.allclose(np.linalg.norm(ray, axis=1), 1)
    assert ray[1, 1] < -0.5 and ray[2, 1] > 0.5
    assert result["values"][:, 0, 14].all()

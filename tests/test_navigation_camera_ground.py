import numpy as np

from src.navigation.camera_ground import ground_contact, mount_rotation
from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket
from src.navigation.vehicle_perception import CameraVehicleTracker


def test_ground_contact_rays_declared_camera_yaw_and_unknown_horizon():
    camera = CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, 0, 0))
    point = ground_contact([240, 150, 272, 200], camera)
    assert point is not None and point[0] > 1.3 and abs(point[1]) < 1e-5
    assert ground_contact([240, 100, 272, 140], camera) is None
    assert ground_contact([240, 150, 272, 288], camera) is None
    left = CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, -60, 0))
    assert ground_contact([240, 150, 272, 200], left)[1] < 0
    assert np.allclose(mount_rotation(left).T @ mount_rotation(left), np.eye(3), atol=1e-6)


def packet(t, speed=2):
    c = CameraCalibration(512, 288, 100, (1.3, 0, 2.3), (0, 0, 0))
    return SensorPacket(
        int(t * 20),
        t,
        {v: np.zeros((288, 512, 3), np.uint8) for v in VIEWS},
        dict.fromkeys(VIEWS, c),
        np.zeros((16, 2), np.float32),
        np.array([speed, 0, 0], np.float32),
    )


def test_track_velocity_removes_ego_motion_without_other_actor_oracle():
    tracker = CameraVehicleTracker()
    first = dict(position_xy=np.array([10.0, 0.0]), sigma_m=0.5, label=3, score=0.9)
    a = tracker.step(packet(0), [first])
    second = dict(first, position_xy=np.array([9.6, 0.0]))
    b = tracker.step(packet(0.2), [second])
    assert a[0].id == b[0].id and np.linalg.norm(b[0].velocity_xy) < 1e-5
    assert b[0].position_sigma_m >= 0.5 and b[0].velocity_sigma_mps >= 1.5
    assert tracker.step(packet(0.4), [])[0].age_seconds > 0.1

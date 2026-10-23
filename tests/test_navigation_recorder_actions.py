import struct

import numpy as np
import pytest

from src.navigation.recorder_actions import (
    align_world_frames,
    ordered_executed_controls,
    vehicle_frames,
)


def packet(kind, payload):
    return struct.pack("<BI", kind, len(payload)) + payload


def test_official_packed_controls_pose_units_and_next_tick_acknowledgment(tmp_path):
    magic, town = b"CARLA_RECORDER", b"Town01_Opt"
    data = struct.pack("<HH", 1, len(magic)) + magic + struct.pack("<QH", 1, len(town)) + town
    for frame, control in enumerate(([0, 0, 1], [0.2, 0.3, 0], [-0.1, 0, 1]), 1):
        data += packet(0, struct.pack("<Qdd", frame, 0.05, (frame - 1) * 0.05))
        data += packet(6, struct.pack("<HIffffff", 1, 44, frame * 100, 200, 300, 4, 5, 6))
        data += packet(8, struct.pack("<HIfffBi", 1, 44, *control, 0, 1))
        data += packet(1, b"")
    path = tmp_path / "native.log"
    path.write_bytes(data)
    frames = vehicle_frames(path, 44)
    assert frames[0]["pose"] == [1, 2, 3, 5, 6, 4]
    world = [
        dict(frame=20 + i, actors=[dict(id=44, role="hero", pose=r["pose"])])
        for i, r in enumerate(frames)
    ]
    aligned, receipt = align_world_frames(frames, world)
    assert receipt["offset"] == 0
    np.testing.assert_allclose(
        ordered_executed_controls(aligned, 20, 2), [[0.2, 0.3, 0], [-0.1, 0, 1]]
    )
    with pytest.raises(ValueError, match="Incomplete"):
        ordered_executed_controls(aligned, 21, 2)
    path.write_bytes(data[:-1])
    with pytest.raises(EOFError):
        vehicle_frames(path, 44)


def test_stationary_ambiguous_alignment_and_frame_gaps_are_not_guessed():
    pose = [0] * 6
    native = [dict(frame=i, pose=pose, control=[0, 0, 1]) for i in range(3)]
    world = [dict(frame=10, actors=[dict(id=2, role="hero", pose=pose)])]
    with pytest.raises(ValueError, match="ambiguous"):
        align_world_frames(native, world)
    world.append(dict(frame=12, actors=[dict(id=2, role="hero", pose=pose)]))
    with pytest.raises(ValueError, match="frame gap"):
        align_world_frames(native, world)


def test_final_ack_tail_remains_distinct_from_pose_matched_camera_trace():
    native = [dict(frame=i + 1, pose=[i, 0, 0, 0, 0, 0], control=[0, 0.2 * i, 0]) for i in range(4)]
    world = [
        dict(frame=i + 50, actors=[dict(id=2, role="hero", pose=row["pose"])])
        for i, row in enumerate(native[:3])
    ]
    aligned, receipt = align_world_frames(native, world)
    assert receipt["aligned_frames"] == 3
    assert receipt["native_tail_frames_after_world_trace"] == 1
    np.testing.assert_allclose(ordered_executed_controls(aligned, 52, 1), [[0, 0.6, 0]])

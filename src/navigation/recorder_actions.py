"""Training/audit-only CARLA0.9.15 executed controls and aligned world poses.

Official recorder packet6 uses packed uint32 plus6 floats; positions are in cm
and rotation stores roll,pitch,yaw. Packet8 is uint32,3 floats,bool,int32 gear.
Controls in the next physics frame acknowledge the command at an RGB frame.
No camera inference may import this privileged factual-data decoder.
"""

import struct
from pathlib import Path

import numpy as np

from src.navigation.recorder_collisions import exact, string


def vehicle_frames(path, actor_id):
    frames, current = [], None
    with Path(path).open("rb") as stream:
        version = struct.unpack("<H", exact(stream, 2))[0]
        magic = string(stream)
        if version != 1 or magic != "CARLA_RECORDER":
            raise ValueError("Unknown native recorder header")
        exact(stream, 8)
        string(stream)
        size = Path(path).stat().st_size
        while stream.tell() < size:
            kind, length = struct.unpack("<BI", exact(stream, 5))
            payload = exact(stream, length)
            if kind == 0:
                if current is not None or length != 24:
                    raise ValueError("Malformed native frame start")
                frame, duration, elapsed = struct.unpack("<Qdd", payload)
                current = dict(frame=frame, duration=duration, elapsed=elapsed)
            elif kind in (6, 8):
                if current is None or length < 2:
                    raise ValueError("Vehicle data outside native frame")
                count = struct.unpack_from("<H", payload)[0]
                stride = 28 if kind == 6 else 21
                if length != 2 + count * stride:
                    raise ValueError("Native vehicle record size differs")
                for i in range(count):
                    offset = 2 + i * stride
                    if struct.unpack_from("<I", payload, offset)[0] != actor_id:
                        continue
                    if kind == 6:
                        _, x, y, z, roll, pitch, yaw = struct.unpack_from(
                            "<Iffffff", payload, offset
                        )
                        current["pose"] = [x / 100, y / 100, z / 100, pitch, yaw, roll]
                    else:
                        _, steer, throttle, brake, handbrake, gear = struct.unpack_from(
                            "<IfffBi", payload, offset
                        )
                        if (
                            handbrake not in (0, 1)
                            or not np.isfinite([steer, throttle, brake]).all()
                        ):
                            raise ValueError("Invalid native executed vehicle control")
                        current["control"] = [steer, throttle, brake]
                        current["handbrake"], current["gear"] = bool(handbrake), gear
            elif kind == 1:
                if length or current is None:
                    raise ValueError("Malformed native frame end")
                if current["frame"] != len(frames) + 1:
                    raise ValueError("Native recorder frame gap")
                frames.append(current)
                current = None
        if current is not None:
            raise ValueError("Partial final native frame; cannot certify action sequence")
    return frames


def align_world_frames(recorded, world, position_tolerance_m=0.001, angle_tolerance_deg=0.01):
    if not recorded or not world:
        raise ValueError("Nonempty native recorder and same-frame world trace required")
    poses, ids = [], set()
    for row in world:
        heroes = [a for a in row["actors"] if a["role"] == "hero"]
        if len(heroes) != 1:
            raise ValueError("Exactly one world-trace hero required")
        ids.add(heroes[0]["id"])
        poses.append(heroes[0]["pose"])
    if len(ids) != 1 or any(b["frame"] != a["frame"] + 1 for a, b in zip(world, world[1:])):
        raise ValueError("Hero identity changed or world trace has a frame gap")
    # Bounded startup offset is determined by every pose in the full trace.
    candidates = []
    reference = np.asarray(poses)
    for offset in range(min(64, len(recorded) - len(world) + 1)):
        subset = recorded[offset : offset + len(world)]
        if any("pose" not in r or "control" not in r for r in subset):
            continue
        native = np.array([r["pose"] for r in subset])
        positions = np.linalg.norm(native[:, :3] - reference[:, :3], axis=1)
        angles = np.abs((native[:, 3:] - reference[:, 3:] + 180) % 360 - 180)
        if positions.max() <= position_tolerance_m and angles.max() <= angle_tolerance_deg:
            candidates.append((offset, float(positions.max()), float(angles.max())))
    if len(candidates) != 1:
        raise ValueError(f"Native/world pose alignment is missing or ambiguous:{len(candidates)}")
    offset, position_error, angle_error = candidates[0]
    # The evaluator can acknowledge the final command on a physics tick after
    # the last camera trace row. Keep contiguous recorder frames, distinguishing
    # this tail from states whose pose was independently matched above.
    aligned = {
        world[0]["frame"] + i: recorded[offset + i]
        for i in range(len(recorded) - offset)
    }
    return aligned, dict(
        offset=offset,
        aligned_frames=len(world),
        maximum_position_error_m=position_error,
        maximum_angle_error_deg=angle_error,
        hero_id=next(iter(ids)),
        alignment_unique=True,
        native_tail_frames_after_world_trace=len(recorded) - offset - len(world),
        limitation="World pose agreement and control acknowledgments, not hidden-state determinism",
    )


def ordered_executed_controls(aligned, input_world_frame, physics_steps=80):
    # At frame f RGB observes the state before action a_f is applied. Recorder
    # control at f+1 confirms a_f; never use the stale control recorded at f.
    wanted = range(input_world_frame + 1, input_world_frame + physics_steps + 1)
    if any(frame not in aligned for frame in wanted):
        raise ValueError("Incomplete executed control horizon")
    result = np.array([aligned[frame]["control"] for frame in wanted], dtype=np.float32)
    if result.shape != (physics_steps, 3) or not np.isfinite(result).all():
        raise ValueError("Invalid ordered native controls")
    return result

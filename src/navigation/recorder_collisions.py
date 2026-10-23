"""Read official CARLA0.9.15 recorder contacts without native fault filtering.

Format: CarlaRecorderInfo/Frames/Collision and QueryCollisions in the official
0.9.15 C++ source. Linux little-endian uint16 strings/time_t64 are explicit.
Continuous frames and reciprocal reports form one physical actor-pair contact
episode; retain every raw hero-involved sample and official ordered starts too.
"""

import struct
from pathlib import Path

from src.navigation.dagger import digest

BASE = "https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Recorder/"


def exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise EOFError("Truncated CARLA recorder")
    return value


def string(stream):
    length = struct.unpack("<H", exact(stream, 2))[0]
    return exact(stream, length).decode("utf-8")


def recorder_contacts(path):
    path = Path(path)
    size = path.stat().st_size
    samples, starts = [], []
    ordered_starts = []
    previous, current, previous_ordered, current_ordered = set(), set(), set(), set()
    frames, elapsed, world_samples, packets = 0, 0.0, 0, 0
    frame_open = False
    errors = []
    with path.open("rb") as stream:
        version = struct.unpack("<H", exact(stream, 2))[0]
        magic = string(stream)
        if version != 1 or magic != "CARLA_RECORDER":
            raise ValueError("Unsupported recorder version/magic; do not guess a zero count")
        date = struct.unpack("<Q", exact(stream, 8))[0]
        town = string(stream)
        while stream.tell() < size:
            offset = stream.tell()
            if size - offset < 5:
                errors.append(dict(error="partial_packet_header", offset=offset))
                break
            kind, length = struct.unpack("<BI", exact(stream, 5))
            if length > size - stream.tell():
                errors.append(dict(error="partial_packet", offset=offset, kind=kind))
                break
            packets += 1
            if kind == 0:
                if length != 24:
                    raise ValueError("Native frame header size differs")
                number, duration, elapsed = struct.unpack("<Qdd", exact(stream, length))
                if number != frames + 1 or frame_open:
                    errors.append(dict(error="frame_sequence_gap", frame=number))
                frames, frame_open = number, True
                previous, current = current, set()
                previous_ordered, current_ordered = current_ordered, set()
            elif kind == 1:
                if length != 0 or not frame_open:
                    raise ValueError("Invalid native frame end")
                frame_open = False
            elif kind == 5:
                payload = exact(stream, length)
                if not frame_open or len(payload) < 2:
                    raise ValueError("Collision packet outside native frame")
                count = struct.unpack_from("<H", payload)[0]
                if length != 2 + 14 * count:
                    raise ValueError("Native collision payload size differs")
                world_samples += count
                for index in range(count):
                    collision, first, second, hero1, hero2 = struct.unpack_from(
                        "<IIIBB", payload, 2 + 14 * index
                    )
                    if hero1 not in (0, 1) or hero2 not in (0, 1):
                        raise ValueError("Invalid native hero flags")
                    if not (hero1 or hero2):
                        continue
                    row = dict(
                        frame=frames,
                        time_s=elapsed,
                        collision_id=collision,
                        actor1=first,
                        actor2=second,
                        hero1=bool(hero1),
                        hero2=bool(hero2),
                    )
                    samples.append(row)
                    ordered = first, second
                    if ordered not in previous_ordered and ordered not in current_ordered:
                        ordered_starts.append(row)
                    current_ordered.add(ordered)
                    pair = tuple(sorted((first, second)))
                    if pair not in previous and pair not in current:
                        starts.append(dict(row, unordered_pair=list(pair)))
                    current.add(pair)
            else:
                stream.seek(length, 1)
        if frame_open:
            errors.append(dict(error="missing_final_frame_end", frame=frames))
        if not frames:
            errors.append(dict(error="no_recorded_frames"))
    return dict(
        schema="carla015-recorder-ego-contacts-v1",
        path=str(path.resolve()),
        sha256=digest(path),
        version=version,
        town=town,
        recorder_date=date,
        frames=frames,
        duration_s=elapsed,
        packets=packets,
        complete=not errors,
        errors=errors,
        ego_contact_starts=len(starts),
        ego_raw_contact_samples=len(samples),
        official_ordered_ego_contact_starts=len(ordered_starts),
        official_h_a_contact_starts=sum(r["hero1"] for r in ordered_starts),
        official_a_h_contact_starts=sum(r["hero2"] for r in ordered_starts),
        all_actor_raw_contact_samples=world_samples,
        starts=starts,
        ordered_starts=ordered_starts,
        raw_samples=samples,
        fault_speed_or_impact_filter=False,
        count_unit=(
            "Unordered ego-involved actor-pair contact starts; adjacent-frame continuation merges"
        ),
        source_urls=[
            BASE + name
            for name in (
                "CarlaRecorderInfo.h",
                "CarlaRecorderFrames.cpp",
                "CarlaRecorderCollision.cpp",
                "CarlaRecorderQuery.cpp",
            )
        ],
        official_rpc_crosscheck_complete=False,
    )

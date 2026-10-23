import struct

from src.navigation.recorder_collisions import recorder_contacts


def text(value):
    data = value.encode()
    return struct.pack("<H", len(data)) + data


def packet(kind, payload):
    return struct.pack("<BI", kind, len(payload)) + payload


def recording(frames):
    data = struct.pack("<H", 1) + text("CARLA_RECORDER") + struct.pack("<Q", 0) + text("Town03_Opt")
    for index, contacts in enumerate(frames, 1):
        data += packet(0, struct.pack("<Qdd", index, 0.05, (index - 1) * 0.05))
        data += packet(
            5,
            struct.pack("<H", len(contacts))
            + b"".join(struct.pack("<IIIBB", *c) for c in contacts),
        )
        data += packet(1, b"")
    return data


def test_stationary_independent_contacts_reciprocal_and_continuous_frames(tmp_path):
    path = tmp_path / "test.log"
    path.write_bytes(
        recording(
            [
                [(1, 10, 20, 1, 0), (2, 20, 10, 0, 1), (3, 30, 40, 0, 0)],
                [(4, 10, 20, 1, 0)],
                [],
                [(5, 10, 20, 1, 0)],
            ]
        )
    )
    measured = recorder_contacts(path)
    assert measured["complete"] and measured["frames"] == 4
    assert measured["ego_contact_starts"] == 2 and measured["ego_raw_contact_samples"] == 4
    assert measured["official_ordered_ego_contact_starts"] == 3
    assert (
        measured["official_h_a_contact_starts"] == 2
        and measured["official_a_h_contact_starts"] == 1
    )
    assert not measured["fault_speed_or_impact_filter"]


def test_partial_crashed_recording_preserves_prior_contact_without_claiming_complete(tmp_path):
    path = tmp_path / "test.log"
    path.write_bytes(recording([[(1, 10, 20, 1, 0)]]) + b"\x00\x18")
    measured = recorder_contacts(path)
    assert not measured["complete"] and measured["ego_contact_starts"] == 1
    assert measured["errors"][0]["error"] == "partial_packet_header"


def test_official_query_parser_retains_duplicate_rows_and_unknown_static_names():
    from scripts.verify_navigation_recorder_rpc import query_rows

    text = (
        "Version: 1\nTime Types Id Actor1 Id Actor2\n"
        "    12   v v      92 vehicle.audi.tt                         93 vehicle.tesla.model3\n"
        "    12   v v      92 vehicle.audi.tt                         93 vehicle.tesla.model3\n"
        "    13   v o      92 vehicle.audi.tt                 4294967295\n"
        "Frames: 300\nDuration: 15 seconds\n"
    )
    assert query_rows(text) == [(12, 92, 93), (12, 92, 93), (13, 92, 4294967295)]

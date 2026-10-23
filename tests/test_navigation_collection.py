"""Crash recovery, raw-pixel preservation, label isolation and control alignment."""

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.navigation.collection import CollectionRecorder, RouteProjector, verify_collection_catalog
from src.navigation.contracts import VIEWS, CameraCalibration, SensorPacket, TrainingTargets
from src.navigation.dataset import (
    EpisodeWriter,
    load_policy_sample,
    shard_records,
    training_records,
)


def packet(frame=10):
    calibration = {v: CameraCalibration(16, 8, 100, (1, 0, 2), (0, 0, 0)) for v in VIEWS}
    return SensorPacket(
        frame,
        frame / 20,
        {
            v: np.random.default_rng(i).integers(0, 256, (8, 16, 3), dtype=np.uint8)
            for i, v in enumerate(VIEWS)
        },
        calibration,
        np.array([[2.0, 0], [4.0, 0]]),
        np.array([1.0, 0.0, 0.0]),
    )


def targets(frame):
    return TrainingTargets(frame, (0, 0.2, 0), (0, 0.2, 0), None, {"secret_world": [1, 2, 3]})


def test_committed_shards_survive_a_truncated_current_archive(tmp_path):
    root = tmp_path / "episode"
    writer = EpisodeWriter(root, {}, shard_samples=2, minimum_free_gb=0)
    for frame in (10, 12, 14):
        writer.append(packet(frame), targets(frame))
    # Simulate abrupt exit before close: a committed zip and a truncated .part remain.
    writer.archive.close()
    writer.archive = None
    writer.part.write_bytes(b"truncated archive")
    committed = list(training_records(root, "prediction"))
    assert len(committed) == 2 and committed[0][1]["frame"] == 10
    assert list(training_records(root, "imitation")) == []
    assert len(list(root.glob("*.part"))) == 1
    with pytest.raises(FileExistsError):
        EpisodeWriter(root, {}, minimum_free_gb=0)


def test_rgb_pixels_roundtrip_exactly_without_privileged_label_reads(tmp_path, monkeypatch):
    writer = EpisodeWriter(tmp_path / "episode", {}, shard_samples=1, minimum_free_gb=0)
    original = packet()
    writer.append(original, targets(original.frame))
    writer.close()
    path = next(writer.root.glob("*.zip"))
    row = next(shard_records(path))
    read = zipfile.ZipFile.read

    def reject_targets(archive, name, *args, **kwargs):
        if "targets.json" in str(name):
            raise AssertionError("Policy reader accessed privileged labels")
        return read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", reject_targets)
    restored = load_policy_sample(path, row)
    for view in VIEWS:
        assert np.array_equal(original.rgb[view], restored.rgb[view])
    assert not hasattr(restored, "facts")


def test_label_frame_and_sampling_gaps_are_rejected(tmp_path):
    writer = EpisodeWriter(tmp_path / "episode", {}, minimum_free_gb=0)
    with pytest.raises(ValueError, match="frame differ"):
        writer.append(packet(), targets(12))
    writer.append(packet(), targets(10))
    with pytest.raises(ValueError, match="contiguous"):
        writer.append(packet(14), targets(14))
    writer.close()


def test_actuator_feedback_is_recorded_one_physics_tick_after_observation(tmp_path):
    writer = EpisodeWriter(tmp_path / "episode", {}, minimum_free_gb=0)
    recorder = CollectionRecorder(writer)
    recorder.capture(packet(), (0, 0.2, 0), (0.1, 0.2, 0), {})
    recorder.acknowledge(10, (0.1, 0.2, 0))
    assert writer.samples == 0
    with pytest.raises(ValueError, match="next physics frame"):
        recorder.acknowledge(12, (0.1, 0.2, 0))
    recorder.acknowledge(11, (0.1, 0.2, 0))
    assert recorder.close()["samples"] == 1
    path = next(writer.root.glob("*.zip"))
    with zipfile.ZipFile(path) as z:
        labels = json.loads(z.read("00000000/targets.json"))
    assert labels["expert_control"] == [0, 0.2, 0]
    assert labels["executed_control"] == [0.1, 0.2, 0]
    assert labels["facts"]["actuator_feedback_frame"] == 11


def test_unacknowledged_final_sample_is_never_published(tmp_path):
    writer = EpisodeWriter(tmp_path / "episode", {}, minimum_free_gb=0)
    recorder = CollectionRecorder(writer)
    recorder.capture(packet(), (0, 0.2, 0), (0, 0.2, 0), {})
    assert recorder.close()["unacknowledged_samples"] == 1
    assert list(writer.root.glob("*.zip")) == []


def test_route_projection_uses_fixed_geometry_and_ego_heading():
    projector = RouteProjector([[0, 0], [0, 0], [0, 10], [0, 20]], distances=[2, 4])
    assert np.allclose(projector.project(np.array([0, 1]), np.pi / 2), [[2, 0], [4, 0]])


def test_collection_refuses_development_catalog():
    with pytest.raises(ValueError, match="training partition"):
        verify_collection_catalog(
            "configs/encounters/dev_v1", "configs/encounters/dev_v1/Town03_Opt.xml"
        )


def test_lossless_video_preserves_frame_order_and_detects_corruption(tmp_path):
    writer = EpisodeWriter(tmp_path / "episode", {}, shard_samples=3, minimum_free_gb=0)
    originals = []
    for frame in (10, 12, 14):
        p = packet(frame)
        p = replace(p, rgb={v: (a + frame).astype(np.uint8) for v, a in p.rgb.items()})
        originals.append(p)
        writer.append(p, targets(frame))
    writer.close()
    shard = next(writer.root.glob("*.zip"))
    rows = list(shard_records(shard))
    for expected, row in zip(originals, rows):
        actual = load_policy_sample(shard, row)
        assert all(np.array_equal(expected.rgb[v], actual.rgb[v]) for v in VIEWS)
    with zipfile.ZipFile(shard) as z:
        members = {name: z.read(name) for name in z.namelist()}
    members["front.mkv"] = b"damaged encoded data"
    with zipfile.ZipFile(shard, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    with pytest.raises(ValueError, match="media changed"):
        list(shard_records(shard))


def test_location_partition_is_disjoint_and_outcome_blind():
    from src.navigation.splits import partition_locations

    movements = []
    for junction in range(10):
        base = {
            "town": "TownA",
            "junction_id": junction,
            "valid": True,
            "approach_m": 50,
            "has_opposite_entry": True,
            "planner_options": ["LEFT"],
            "signal_ids": [junction],
            "stop_ids": [junction],
            "entry_lane": [junction, 0, -1],
            "exit_lane": [junction + 100, 0, -1],
            "crossed_junction_ids": [junction],
        }
        movements += [{**base, "control": ["signal"]}, {**base, "control": ["stop"]}]
    split, train, dev, _ = partition_locations(movements)
    assert not set(split["TownA"]["train"]) & set(split["TownA"]["heldout_dev"])
    assert not {m["junction_id"] for m in train} & {m["junction_id"] for m in dev}
    with_outcomes = [{**m, "expert_result": "collision"} for m in movements]
    assert partition_locations(with_outcomes)[0] == split


def test_noise_is_seeded_bounded_and_stops_on_new_hazards():
    from src.navigation.perturbations import SteeringPerturbation

    first, repeat = SteeringPerturbation(42), SteeringPerturbation(42)
    teacher = (0.2, 0.4, 0.0)
    command, reason = first.control(teacher, 40, 3.0, False, False)
    assert (command, reason) == repeat.control(teacher, 40, 3.0, False, False)
    assert 0.039 <= abs(command[0] - teacher[0]) <= 0.081 and reason == "steering_noise"
    command, reason = first.control(teacher, 41, 3.0, False, True)
    assert command[1:] == (0.0, 1.0) and reason == "privileged_safety_stop"
    assert first.safety_stop_ticks == 1
    assert first.control(teacher, 42, 3.0, False, False) == (teacher, None)
    assert first.control(teacher, 80, 3.0, True, False) == (teacher, None)


def test_collection_schedule_does_not_select_by_outcome():
    from itertools import islice

    from scripts.collect_navigation import episode_schedule

    entries = [
        {"route_id": i, "family": f, "town": t}
        for i, (f, t) in enumerate((("signal", "TownA"), ("signal", "TownB"), ("stop", "TownA")))
    ]
    original = list(islice(episode_schedule({"encounters": entries}, 42), 9))
    altered = list(
        islice(
            episode_schedule({"encounters": [{**e, "outcome": "collision"} for e in entries]}, 42),
            9,
        )
    )
    assert [e["route_id"] for e in original] == [e["route_id"] for e in altered]
    assert len({e["town"] for e in original[:3]}) == 2


def test_unsafe_unknown_and_safety_stopped_episodes_are_not_clean_demonstrations(tmp_path):
    from src.navigation.collection import finalize_collection
    from src.navigation.metrics import summarize_records

    records = [
        {
            "route_id": "RouteScenario_0_rep0",
            "status": "Perfect",
            "scores": {"score_route": 100},
            "infractions": {},
        },
        {
            "route_id": "RouteScenario_1_rep0",
            "status": "Completed",
            "scores": {"score_route": 100},
            "infractions": {"collisions_vehicle": ["collision"]},
        },
        {
            "route_id": "RouteScenario_2_rep0",
            "status": "Perfect",
            "scores": {"score_route": 100},
            "infractions": {},
        },
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summarize_records(records)))
    for i in range(4):
        ep = tmp_path / "episodes" / str(i)
        (ep / "data").mkdir(parents=True)
        (ep / "data/episode.json").write_text(json.dumps({"route_id": i}))
        if i != 3:
            (ep / "collection_profile.json").write_text(
                json.dumps(
                    {
                        "samples": 10,
                        "control_differences": 0,
                        "safety_stop_ticks": int(i == 2),
                    }
                )
            )
    finalize_collection(tmp_path)
    outcomes = [
        json.loads((tmp_path / f"episodes/{i}/data/outcome.json").read_text()) for i in range(4)
    ]
    assert [o["imitation_eligible"] for o in outcomes] == [True, False, False, False]
    assert outcomes[2]["episode_censored"] and outcomes[3]["episode_censored"]
    assert all(not o["counterfactual_safe_label"] for o in outcomes)


def test_gpu_readiness_failure_does_not_allow_a_new_simulator(monkeypatch):
    from src.navigation import runtime

    monkeypatch.setattr(runtime, "free_vram_gb", lambda: None)
    assert "GPU memory query failed" in runtime.resource_shortfall({"minimum_free_vram_gb": 3})


def test_spatial_clearance_detects_coincident_endpoints_despite_distinct_road_ids():
    from src.navigation.splits import nearest_junction

    route = [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
    # A nominally nonjunction road endpoint can coincide with a held-out junction.
    heldout = [[0.000015, 0, 0], [50, 0, 0]]
    nearest = nearest_junction(route, heldout, [103, 200])
    assert nearest["junction_id"] == 103 and nearest["clearance_m"] < 0.00002
    # A bridge with identical XY coordinates is spatially distinct.
    assert nearest_junction(route, [[0, 0, 10]], [103])["clearance_m"] == 10
    with pytest.raises(ValueError):
        nearest_junction(route, [], [])


def test_collector_detects_agent_crash_even_when_harness_exits_zero(tmp_path):
    from scripts.collect_navigation import collection_error

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"episodes": [{"status": "Failed - Agent crashed"}]}))
    assert collection_error(tmp_path, 0) == "agent_crashed"
    summary.write_text(json.dumps({"episodes": [{"status": "Completed"}]}))
    assert collection_error(tmp_path, 0) is None  # Collisions are retained driving outcomes.
    assert collection_error(tmp_path, 1) == "worker_exit"
    (tmp_path / "collection_partition_violation.json").write_text("{}")
    assert collection_error(tmp_path, 0) == "partition_violation"


def test_collection_queue_resumes_without_repeating_committed_episodes(tmp_path, monkeypatch):
    import hashlib
    import sys

    from scripts import collect_navigation as collect

    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    payload = json.dumps(
        {
            "role": "train",
            "encounters": [
                {"town": "TownA", "route_id": 4, "family": "route_junction"},
            ],
        }
    ).encode()
    (catalog_dir / "catalog.json").write_bytes(payload)
    (catalog_dir / "catalog.sha256").write_text(hashlib.sha256(payload).hexdigest())
    output = tmp_path / "collection"
    argv = [
        "collect_navigation.py",
        "--catalog",
        str(catalog_dir),
        "--out",
        str(output),
        "--target-samples",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(collect, "resource_shortfall", lambda r: None)
    monkeypatch.setattr(
        collect, "collection_geometry_eligibility", lambda p: {"kept_route_ids": [4]}
    )
    calls = []

    class Child:
        returncode = 0

        def __init__(self, command, **kwargs):
            calls.append(command)
            count = int(command[command.index("--collection-max-samples") + 1])
            root = Path(command[command.index("--out") + 1]) / "episodes/test/data"
            root.mkdir(parents=True)
            with zipfile.ZipFile(root / "shard_000000.zip", "w") as z:
                z.writestr("manifest.json", json.dumps({"samples": [{}] * count}))

        def poll(self):
            return 0

    monkeypatch.setattr(collect.subprocess, "Popen", Child)
    monkeypatch.setattr(
        collect, "audit", lambda p: {"totals": {"samples": collect.committed_samples(p)}}
    )
    collect.main()
    assert len(calls) == 1
    assert json.loads((output / "progress.json").read_text())["frames"] == 3
    collect.main()
    assert len(calls) == 1  # every committed episode is preserved on resume
    monkeypatch.setattr(
        collect, "collection_geometry_eligibility", lambda p: {"kept_route_ids": [4, 5]}
    )
    with pytest.raises(ValueError, match="Resume request"):
        collect.main()
    assert json.loads((output / "geometry_eligibility.json").read_text()) == {"kept_route_ids": [4]}
